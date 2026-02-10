#!/usr/bin/env python3
"""ktop - Terminal system resource monitor for hybrid LLM workloads."""

import argparse
import os
import select
import signal
import sys
import termios
import time
import tty
from collections import deque

import psutil
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

try:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        from pynvml import (
            nvmlDeviceGetCount,
            nvmlDeviceGetHandleByIndex,
            nvmlDeviceGetMemoryInfo,
            nvmlDeviceGetName,
            nvmlDeviceGetUtilizationRates,
            nvmlInit,
            nvmlShutdown,
        )

    _PYNVML = True
except ImportError:
    _PYNVML = False

# ── constants ────────────────────────────────────────────────────────────────
SPARK = " ▁▂▃▄▅▆▇█"
HISTORY_LEN = 60


# ── helpers ──────────────────────────────────────────────────────────────────
def _color_for(pct: float) -> str:
    if pct < 50:
        return "green"
    if pct < 80:
        return "yellow"
    return "red"


def _bar(pct: float, width: int = 25) -> str:
    """Render a coloured progress bar as Rich markup."""
    filled = int(pct / 100 * width)
    empty = width - filled
    c = _color_for(pct)
    return f"[{c}]{'█' * filled}[/{c}][dim]{'░' * empty}[/dim]"


def _sparkline(values, width: int | None = None) -> str:
    """Return a sparkline string from values in 0-100 range."""
    if not values:
        return ""
    vals = list(values)
    if width and len(vals) > width:
        vals = vals[-width:]
    out = []
    for v in vals:
        v = max(0.0, min(100.0, v))
        idx = int(v / 100 * (len(SPARK) - 1))
        out.append(SPARK[idx])
    return "".join(out)


def _fmt_bytes(b: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(b) < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} PB"


# ── main monitor ─────────────────────────────────────────────────────────────
class KTop:
    def __init__(self, refresh: float = 1.0):
        self.refresh = refresh
        self.console = Console()

        # rolling histories
        self.cpu_hist: deque[float] = deque(maxlen=HISTORY_LEN)

        # GPU init
        self.gpu_ok = False
        self.gpu_count = 0
        self.gpu_util_hist: dict[int, deque] = {}
        self.gpu_mem_hist: dict[int, deque] = {}

        if _PYNVML:
            try:
                nvmlInit()
                self.gpu_count = nvmlDeviceGetCount()
                self.gpu_ok = True
                for i in range(self.gpu_count):
                    self.gpu_util_hist[i] = deque(maxlen=HISTORY_LEN)
                    self.gpu_mem_hist[i] = deque(maxlen=HISTORY_LEN)
            except Exception:
                pass

        # prime one CPU sample so the first frame has data
        psutil.cpu_percent(interval=None)

    # ── data collectors ──────────────────────────────────────────────────
    def _sample_cpu(self) -> float:
        pct = psutil.cpu_percent(interval=None)
        self.cpu_hist.append(pct)
        return pct

    def _gpu_info(self) -> list[dict]:
        gpus = []
        if not self.gpu_ok:
            return gpus
        for i in range(self.gpu_count):
            try:
                h = nvmlDeviceGetHandleByIndex(i)
                name = nvmlDeviceGetName(h)
                if isinstance(name, bytes):
                    name = name.decode()
                util = nvmlDeviceGetUtilizationRates(h)
                mem = nvmlDeviceGetMemoryInfo(h)
                mem_pct = mem.used / mem.total * 100 if mem.total else 0
                self.gpu_util_hist[i].append(util.gpu)
                self.gpu_mem_hist[i].append(mem_pct)
                gpus.append(
                    {
                        "id": i,
                        "name": name,
                        "util": util.gpu,
                        "mem_used_gb": mem.used / 1024**3,
                        "mem_total_gb": mem.total / 1024**3,
                        "mem_pct": mem_pct,
                    }
                )
            except Exception:
                pass
        return gpus

    def _top_procs(self, key: str, n: int = 10) -> list[dict]:
        """Return top-n processes sorted by *key* ('memory_percent' or 'cpu_percent')."""
        procs = []
        for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent", "memory_info"]):
            try:
                info = p.info
                if info["pid"] == 0:
                    continue
                procs.append(info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        procs.sort(key=lambda x: x.get(key, 0) or 0, reverse=True)
        return procs[:n]

    # ── panel builders ───────────────────────────────────────────────────
    def _gpu_panels(self) -> Layout:
        """Return a horizontal Layout with one panel per GPU."""
        gpus = self._gpu_info()
        if not gpus:
            return Panel(
                Text("No GPUs detected (install pynvml for GPU monitoring)", style="dim italic"),
                title="[bold magenta] GPU [/bold magenta]",
                border_style="magenta",
            )

        gpu_layout = Layout()
        children = []
        for g in gpus:
            uc = _color_for(g["util"])
            mc = _color_for(g["mem_pct"])
            spark_u = _sparkline(self.gpu_util_hist[g["id"]], width=20)
            spark_m = _sparkline(self.gpu_mem_hist[g["id"]], width=20)
            body = (
                f"[bold]Util[/bold] {_bar(g['util'], 15)} [{uc}]{g['util']:5.1f}%[/{uc}]\n"
                f"      {spark_u}\n"
                f"[bold]Mem [/bold] {_bar(g['mem_pct'], 15)} [{mc}]{g['mem_pct']:5.1f}%[/{mc}]\n"
                f"      {g['mem_used_gb']:.1f}/{g['mem_total_gb']:.1f} GB\n"
                f"      {spark_m}"
            )
            name_short = g["name"].replace("NVIDIA ", "").replace(" Generation", "")
            panel = Panel(
                Text.from_markup(body),
                title=f"[bold magenta] GPU {g['id']} [/bold magenta]",
                subtitle=f"[dim]{name_short}[/dim]",
                border_style="magenta",
            )
            child = Layout(name=f"gpu{g['id']}", ratio=1)
            child.update(panel)
            children.append(child)

        gpu_layout.split_row(*children)
        return gpu_layout

    def _cpu_panel(self) -> Panel:
        pct = self._sample_cpu()
        c = _color_for(pct)
        cores = psutil.cpu_count(logical=True)
        freq = psutil.cpu_freq()
        freq_str = f"{freq.current:.0f} MHz" if freq else "N/A"

        spark = _sparkline(self.cpu_hist)
        body = (
            f"[bold]Overall[/bold]  {_bar(pct, 30)} [{c}]{pct:5.1f}%[/{c}]\n"
            f"[dim]Cores: {cores}  Freq: {freq_str}[/dim]\n\n"
            f"[bold]History[/bold]\n{spark}"
        )
        return Panel(
            Text.from_markup(body),
            title="[bold cyan] CPU [/bold cyan]",
            border_style="cyan",
        )

    def _mem_panel(self) -> Panel:
        vm = psutil.virtual_memory()
        sw = psutil.swap_memory()

        used_pct = vm.percent
        c = _color_for(used_pct)

        body = (
            f"[bold]RAM[/bold]  {_bar(used_pct, 30)} [{c}]{used_pct:5.1f}%[/{c}]\n"
            f"  Used: {_fmt_bytes(vm.used)}  Free: {_fmt_bytes(vm.available)}  Total: {_fmt_bytes(vm.total)}\n\n"
            f"[bold]Swap[/bold] {_bar(sw.percent, 30)} [dim]{sw.percent:5.1f}%[/dim]\n"
            f"  Used: {_fmt_bytes(sw.used)}  Total: {_fmt_bytes(sw.total)}"
        )
        return Panel(
            Text.from_markup(body),
            title="[bold green] Memory [/bold green]",
            border_style="green",
        )

    def _proc_table(self, by: str) -> Panel:
        is_mem = by == "memory_percent"
        procs = self._top_procs(by)
        title = "Top Processes by Memory" if is_mem else "Top Processes by CPU"
        colour = "green" if is_mem else "cyan"

        table = Table(expand=True, box=None, pad_edge=False)
        table.add_column("PID", style="dim", width=8, justify="right")
        table.add_column("Name", ratio=2)
        if is_mem:
            table.add_column("RSS", justify="right", width=10)
            table.add_column("Mem %", justify="right", width=8)
        else:
            table.add_column("CPU %", justify="right", width=8)
            table.add_column("Mem %", justify="right", width=8)

        for p in procs:
            pid = str(p.get("pid", ""))
            name = (p.get("name") or "?")[:28]
            mem_pct = p.get("memory_percent") or 0
            cpu_pct = p.get("cpu_percent") or 0
            if is_mem:
                mi = p.get("memory_info")
                rss = _fmt_bytes(mi.rss) if mi else "N/A"
                table.add_row(pid, name, rss, f"{mem_pct:.1f}%")
            else:
                table.add_row(pid, name, f"{cpu_pct:.1f}%", f"{mem_pct:.1f}%")

        return Panel(table, title=f"[bold {colour}] {title} [/bold {colour}]", border_style=colour)

    # ── layout ───────────────────────────────────────────────────────────
    def _build(self) -> Layout:
        layout = Layout()

        layout.split_column(
            Layout(name="gpu", ratio=2),
            Layout(name="mid", ratio=2),
            Layout(name="bot", ratio=3),
        )
        layout["mid"].split_row(
            Layout(name="cpu", ratio=1),
            Layout(name="mem", ratio=1),
        )
        layout["bot"].split_row(
            Layout(name="mem_procs", ratio=1),
            Layout(name="cpu_procs", ratio=1),
        )

        layout["gpu"].update(self._gpu_panels())
        layout["cpu"].update(self._cpu_panel())
        layout["mem"].update(self._mem_panel())
        layout["mem_procs"].update(self._proc_table("memory_percent"))
        layout["cpu_procs"].update(self._proc_table("cpu_percent"))

        return layout

    # ── keyboard helpers ────────────────────────────────────────────────
    @staticmethod
    def _key_pressed() -> str | None:
        """Return a key character if one is waiting on stdin, else None."""
        if select.select([sys.stdin], [], [], 0)[0]:
            ch = sys.stdin.read(1)
            # ESC could be standalone or start of an escape sequence
            if ch == "\x1b":
                # Read any remaining escape sequence bytes and discard
                while select.select([sys.stdin], [], [], 0)[0]:
                    sys.stdin.read(1)
                return "ESC"
            return ch
        return None

    # ── run loop ─────────────────────────────────────────────────────────
    def run(self) -> None:
        def _cleanup():
            if self.gpu_ok:
                try:
                    nvmlShutdown()
                except Exception:
                    pass

        def _quit(*_):
            _cleanup()
            sys.exit(0)

        signal.signal(signal.SIGINT, _quit)
        signal.signal(signal.SIGTERM, _quit)

        # Put terminal in raw mode so we can read single key presses
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            with Live(
                self._build(),
                console=self.console,
                screen=True,
                refresh_per_second=int(1 / self.refresh),
            ) as live:
                while True:
                    time.sleep(self.refresh)
                    key = self._key_pressed()
                    if key in ("q", "Q", "ESC"):
                        break
                    live.update(self._build())
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            _cleanup()


# ── CLI ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="ktop — system monitor for hybrid LLM workloads")
    parser.add_argument(
        "-r",
        "--refresh",
        type=float,
        default=1.0,
        help="Refresh interval in seconds (default: 1.0)",
    )
    args = parser.parse_args()
    KTop(refresh=args.refresh).run()


if __name__ == "__main__":
    main()
