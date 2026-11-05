#!/usr/bin/env python3
"""
ROCm SMI Live Dashboard
Visualizes AMD GPU metrics in real-time using amd-smi (JSON output).
Usage: python rocm_visualizer.py [--interval SECONDS]
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
import subprocess
from typing import Optional
from collections import deque
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box


HISTORY_LEN = 60  # data points kept for sparklines

# ---------------------------------------------------------------------------
# Journal log watcher
# ---------------------------------------------------------------------------

# Keywords that make a kernel log line relevant to the GPU
_LOG_INCLUDE = re.compile(
    r"amdgpu|kfd|rocm|gpu|drm\[",
    re.IGNORECASE,
)
# Severity classification based on message content
_LOG_ERROR   = re.compile(r"\b(error|fail|fault|hang|reset|timeout|died|crash|oops|bug|panic)\b", re.IGNORECASE)
_LOG_WARN    = re.compile(r"\b(warn|deprecated|throttl|limit|exceed|retry)\b", re.IGNORECASE)
_LOG_VERBOSE = re.compile(r"Freeing queue|queue evicted|alloc|mapping|unmap", re.IGNORECASE)

# Short output format: "Mär 28 21:03:11 hostname kernel: amdgpu: ..."
_LOG_LINE_RE = re.compile(r"^(\S+\s+\d+\s+\S+)\s+\S+\s+\S+:\s+(.+)$")


class LogEntry:
    __slots__ = ("timestamp", "message", "style")

    def __init__(self, timestamp: str, message: str, style: str):
        self.timestamp = timestamp
        self.message   = message
        self.style     = style


class JournalWatcher:
    """Reads GPU-related kernel log lines from journalctl in a background thread."""

    MAX_ENTRIES = 200

    def __init__(self):
        self._entries: deque[LogEntry] = deque(maxlen=self.MAX_ENTRIES)
        self._lock    = threading.Lock()
        self._thread: threading.Thread | None = None
        self.available = platform.system() == "Linux"

    def start(self):
        if not self.available:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            proc = subprocess.Popen(
                ["journalctl", "-k", "-f", "-n", "50", "--no-pager", "-o", "short"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            for raw_line in proc.stdout:
                line = raw_line.rstrip()
                if not _LOG_INCLUDE.search(line):
                    continue
                m = _LOG_LINE_RE.match(line)
                if m:
                    ts  = m.group(1)
                    msg = m.group(2)
                else:
                    ts  = ""
                    msg = line

                if _LOG_ERROR.search(msg):
                    style = "bold red"
                elif _LOG_WARN.search(msg):
                    style = "yellow"
                elif _LOG_VERBOSE.search(msg):
                    style = "dim"
                else:
                    style = "white"

                with self._lock:
                    self._entries.append(LogEntry(ts, msg, style))
        except (FileNotFoundError, OSError):
            self.available = False

    def recent(self, n: int = 30) -> list[LogEntry]:
        with self._lock:
            return list(self._entries)[-n:]


# ---------------------------------------------------------------------------
# amd-smi helpers (JSON output)
# ---------------------------------------------------------------------------

def run_amd_smi(*args: str):
    """Run `amd-smi <args> --json` and return the parsed JSON (or None)."""
    try:
        out = subprocess.run(
            ["amd-smi", *args, "--json"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        return json.loads(out)
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return None


def detect_amd_smi_version() -> tuple[int, int]:
    """Return (major, minor) of the amd-smi tool, (0, 0) if unavailable."""
    data = run_amd_smi("version")
    try:
        entry = data[0] if isinstance(data, list) else data
        m = re.match(r"(\d+)\.(\d+)", entry["version"])
        if m:
            return int(m.group(1)), int(m.group(2))
    except (TypeError, KeyError, IndexError):
        pass
    return (0, 0)


def _val(node, default=None):
    """Extract a number from amd-smi values: {"value": 3, "unit": "W"}, 3 or "N/A"."""
    if isinstance(node, dict):
        node = node.get("value")
    if isinstance(node, (int, float)):
        return float(node)
    if isinstance(node, str):
        try:
            return float(node)
        except ValueError:
            pass
    return default


def _gpu_entries(data) -> list[dict]:
    if isinstance(data, dict):
        data = data.get("gpu_data", [])
    return [e for e in (data or []) if isinstance(e, dict) and "gpu" in e]


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def fetch_driver_version() -> str:
    for e in _gpu_entries(run_amd_smi("static", "--driver")):
        v = (e.get("driver") or {}).get("version")
        if v:
            return str(v)
    data = run_amd_smi("version")
    try:
        return str((data[0] if isinstance(data, list) else data)["amdgpu_version"])
    except (TypeError, KeyError, IndexError):
        return "N/A"


def fetch_static_info() -> dict[int, dict]:
    """Per-GPU static data: device name and power cap (fetched once)."""
    info: dict[int, dict] = {}
    for e in _gpu_entries(run_amd_smi("static", "--asic", "--limit")):
        ppt0 = (e.get("limit") or {}).get("ppt0") or {}
        cap = _val(ppt0.get("socket_power_limit")) or _val(ppt0.get("max_power_limit"))
        info[int(e["gpu"])] = {
            "name":      (e.get("asic") or {}).get("market_name") or f"GPU {e['gpu']}",
            "power_cap": cap,
        }
    return info


def fetch_power_caps() -> dict:
    return {i: d["power_cap"] for i, d in fetch_static_info().items() if d["power_cap"]}


def fetch_loaded_models() -> Optional[str]:
    """
    Führt `ollama ps` aus, extrahiert die Spalten
    NAME und SIZE und gibt die Modelle als einen
    Oneliner im Format:
        NAME (SIZE), NAME (SIZE), …
    zurück.  Bei keinem geladenen Modell oder Fehler
    wird None zurückgegeben.
    """
    try:
        result = subprocess.run(
            ["ollama", "ps"],
            capture_output=True,
            text=True,
            check=True,
        )

        # Ausgabe in Zeilen zerlegen, Header ignorieren
        lines = result.stdout.strip().splitlines()
        if len(lines) < 2:          # keine Datenzeilen vorhanden
            return None

        formatted_parts = []
        for line in lines[1:]:      # Header (erste Zeile) überspringen
            parts = line.split()
            if len(parts) >= 3:    # Mindestens NAME, ID, SIZE vorhanden
                name, size = parts[0], parts[2]
                formatted_parts.append(f"{name} ({size} Gb)")

        return ", ".join(formatted_parts) if formatted_parts else None

    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _fmt_mem(node) -> str:
    """Format a memory value from amd-smi (usually {"value": N, "unit": "MB"})."""
    unit = node.get("unit", "B") if isinstance(node, dict) else "B"
    v = _val(node)
    if v is None:
        return str(node)
    mult = {"B": 1, "KB": 1024, "MB": 1024 ** 2, "GB": 1024 ** 3}.get(unit.upper(), 1)
    mb = v * mult / 1024 ** 2
    return f"{mb / 1024:.1f} GB" if mb >= 1024 else f"{mb:.0f} MB"


def fetch_processes() -> dict[int, list[dict]]:
    """GPU processes per GPU index from `amd-smi process --json`."""
    result: dict[int, list[dict]] = {}
    for e in _gpu_entries(run_amd_smi("process")):
        procs = []
        for item in e.get("process_list", []) or []:
            p = item.get("process_info") if isinstance(item, dict) else None
            if not isinstance(p, dict):
                continue  # e.g. "No running processes detected"
            mem = p.get("mem_usage") or p.get("memory_usage") or {}
            if isinstance(mem, dict) and "vram_mem" in mem:
                mem = mem["vram_mem"]
            procs.append({
                "pid":  str(p.get("pid", "?")),
                "name": str(p.get("name", "?")),
                "gpus": "1",
                "vram": _fmt_mem(mem) if mem else "N/A",
            })
        result[int(e["gpu"])] = procs
    return result


def decode_throttle(status) -> tuple[str, str]:
    """Return (label, rich_style) for a throttle status (0/False = OK)."""
    if not status:
        return "OK", "bold green"
    return "THROTTLED", "bold red"


def collect_gpu_data(static_info: dict[int, dict] | None = None,
                     power_caps: dict | None = None) -> list[dict]:
    metric_data = run_amd_smi("metric")
    entries = _gpu_entries(metric_data)
    if not entries:
        return []

    static_info = static_info or {}
    processes   = fetch_processes()

    profiles = {
        int(e["gpu"]): (e.get("profile") or {}).get("current", "N/A")
        for e in _gpu_entries(run_amd_smi("static", "--profile"))
    }

    gpus = []
    for e in entries:
        idx   = int(e["gpu"])
        usage = e.get("usage") or {}
        power = e.get("power") or {}
        clock = e.get("clock") or {}
        temp  = e.get("temperature") or {}
        pcie  = e.get("pcie") or {}
        fan   = e.get("fan") or {}
        mem   = e.get("mem_usage") or {}

        def clk(key: str) -> float:
            return _val((clock.get(key) or {}).get("clk"), 0.0)

        vcn = usage.get("vcn_activity")
        vcn_val = _val(vcn[0], 0.0) if isinstance(vcn, list) and vcn else _val(vcn, 0.0)

        vram_used  = _val(mem.get("used_vram"), 0.0)
        vram_total = _val(mem.get("total_vram"), 1.0) or 1.0

        throttle_raw = str(power.get("throttle_status", "")).upper()
        throttle_val = 1 if throttle_raw and throttle_raw not in ("UNTHROTTLED", "N/A") else 0
        throttle_label, throttle_style = decode_throttle(throttle_val)

        perf_level = str(e.get("perf_level", "N/A")).replace("AMDSMI_DEV_PERF_LEVEL_", "").lower()

        fan_pct = _val(fan.get("usage"))
        if fan_pct is None:
            speed, fmax = _val(fan.get("speed")), _val(fan.get("max"))
            fan_pct = round(speed / fmax * 100, 1) if speed is not None and fmax else None

        sinfo = static_info.get(idx, {})
        cap = (power_caps or {}).get(idx) or sinfo.get("power_cap") or 327.0

        gpus.append({
            "idx":             idx,
            "name":            sinfo.get("name", f"GPU {idx}"),
            # basic
            "temp_edge":       _val(temp.get("edge"), 0.0),
            "temp_junction":   _val(temp.get("hotspot"), 0.0),
            "temp_mem":        _val(temp.get("mem"), 0.0),
            "gpu_use":         _val(usage.get("gfx_activity"), 0.0),
            "vram_pct":        round(vram_used / vram_total * 100, 1),
            "vram_used_mb":    vram_used,
            "vram_total_mb":   vram_total,
            "mem_activity":    _val(usage.get("umc_activity"), 0.0),
            "power_avg":       _val(power.get("socket_power"), 0.0),
            "power_cap":       cap,
            "sclk":            clk("gfx_0"),
            "mclk":            clk("mem_0"),
            "fclk":            clk("fclk_0"),
            "socclk":          clk("socclk_0"),
            "voltage":         _val(power.get("gfx_voltage"), 0.0),
            "fan_pct":         fan_pct,
            "fan_rpm":         _val(fan.get("rpm")),
            # metrics (amd-smi does not expose VR temperatures → 0)
            "temp_vrgfx":      0.0,
            "temp_vrsoc":      0.0,
            "temp_vrmem":      0.0,
            "throttle_val":    throttle_val,
            "throttle_label":  throttle_label,
            "throttle_style":  throttle_style,
            "pcie_width":      _val(pcie.get("width")),
            "pcie_speed":      _val(pcie.get("speed")),
            "voltage_gfx":     _val(power.get("gfx_voltage"), 0.0),
            "voltage_soc":     _val(power.get("soc_voltage"), 0.0),
            "voltage_mem":     _val(power.get("mem_voltage"), 0.0),
            "vcn_activity":    vcn_val,
            "avg_gfxclk":      0.0,
            # system
            "perf_level":      perf_level,
            "power_profile":   profiles.get(idx, "N/A"),
            "processes":       processes.get(idx, []),
        })

    return gpus


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def sparkline(history: deque, width: int = 20) -> str:
    blocks = " ▁▂▃▄▅▆▇█"
    if not history:
        return " " * width
    vals = list(history)[-width:]
    lo, hi = min(vals), max(vals)
    span = hi - lo or 1
    chars = [blocks[int((v - lo) / span * (len(blocks) - 1))] for v in vals]
    return "".join(chars).rjust(width)


def color_temp(t: float) -> str:
    if t >= 95: return "bold red"
    if t >= 85: return "red"
    if t >= 70: return "yellow"
    return "green"


def color_use(u: float) -> str:
    if u >= 95: return "bold magenta"
    if u >= 80: return "magenta"
    if u >= 50: return "cyan"
    return "green"


def make_bar(value: float, total: float, width: int = 20, color: str = "cyan") -> Text:
    pct = min(value / total, 1.0) if total else 0
    filled = int(pct * width)
    bar = "█" * filled + "░" * (width - filled)
    t = Text()
    t.append(bar, style=color)
    t.append(f" {value:.0f}/{total:.0f}")
    return t


# ---------------------------------------------------------------------------
# Dashboard layout
# ---------------------------------------------------------------------------

def build_dashboard(
    all_gpus: list[dict],
    loadedModels: str,
    histories: dict,
    interval: float,
    rocm_version: tuple[int, int],
    log_entries: list[LogEntry] | None = None,
    driver_version: str = "N/A",
) -> Layout:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=4),
        Layout(name="body"),
        Layout(name="logs", size=10),
        Layout(name="footer", size=1),
    )

    # ── Header ──────────────────────────────────────────────────────────────
    ver_str   = f"amd-smi {rocm_version[0]}.{rocm_version[1]}"
    gpu_names = "  |  ".join(
        f"[bold cyan]GPU {g['idx']}: {g['name']}[/bold cyan]" for g in all_gpus
    ) if all_gpus else "[red]no GPU detected[/red]"

    header_table = Table(box=None, show_header=False, padding=0, expand=True)
    header_table.add_column(justify="left")
    header_table.add_row(
        Text.from_markup(
            f"[bold white on dark_blue] ROCm SMI Dashboard [/bold white on dark_blue]"
            f"   [dim]{now}[/dim]   [bold cyan]{ver_str}[/bold cyan]"
            f"   [dim]interval: {interval}s[/dim]",
            justify="left",
        )
    )
    header_table.add_row(
        Text.from_markup(
            f"  {gpu_names}   [dim]AMD Driver: [/dim][bold yellow]{driver_version}[/bold yellow]"
        )
    )
    layout["header"].update(Panel(header_table, box=box.SIMPLE))

    # ── Footer ───────────────────────────────────────────────────────────────
    layout["footer"].update(Text("  Press Ctrl+C to quit", style="dim", justify="left"))

    if not all_gpus:
        layout["body"].update(
            Panel("[red]No ROCm GPU detected or amd-smi not available.[/red]")
        )
        return layout

    gpu_layouts = [Layout(name=f"gpu{g['idx']}") for g in all_gpus]
    layout["body"].split_row(*gpu_layouts)

    for g in all_gpus:
        idx  = g["idx"]
        hist = histories[idx]

        # ── Utilization & Power ──────────────────────────────────────────────
        util_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        util_table.add_column("metric", style="bold", width=14)
        util_table.add_column("bar",   width=28)
        util_table.add_column("spark", width=22)

        util_table.add_row(
            "GPU Use",
            make_bar(g["gpu_use"], 100, color=color_use(g["gpu_use"])),
            Text(sparkline(hist["gpu_use"]) + " %", style="dim cyan"),
        )
        vram_used = g["vram_used_mb"]
        vram_total = g["vram_total_mb"]
        vram_label = f"{vram_used / 1024:.1f}/{vram_total / 1024:.1f} GB" \
                     if vram_total >= 1024 else f"{vram_used:.0f}/{vram_total:.0f} MB"
        util_table.add_row(
            "VRAM",
            make_bar(vram_used, vram_total, color="blue"),
            Text(sparkline(hist["vram_pct"]) + f"  {vram_label}", style="dim blue"),
        )
        util_table.add_row(
            "Mem Activity",
            make_bar(g["mem_activity"], 100, color="dark_orange"),
            Text(sparkline(hist["mem_activity"]) + " %", style="dim"),
        )
        if g["vcn_activity"] is not None:
            util_table.add_row(
                "VCN (Video)",
                make_bar(g["vcn_activity"], 100, color="purple"),
                Text(sparkline(hist["vcn_activity"]) + " %", style="dim purple"),
            )
        power_color = "red" if g["power_avg"] > g["power_cap"] * 0.9 else "yellow"
        util_table.add_row(
            "Power",
            make_bar(g["power_avg"], g["power_cap"], color=power_color),
            Text(sparkline(hist["power_avg"]) + " W", style="dim yellow"),
        )
        if g["fan_pct"] is not None:
            rpm_str = f"{g['fan_rpm']:.0f} RPM" if g["fan_rpm"] is not None else "N/A"
            util_table.add_row(
                "Fan",
                make_bar(g["fan_pct"], 100, color="green"),
                Text(rpm_str, style="dim green"),
            )
        else:
            fan_rpm = g["fan_rpm"]
            rpm_txt = f"{fan_rpm:.0f} RPM" if fan_rpm is not None else "N/A"
            util_table.add_row("Fan", Text(rpm_txt, style="dim"), Text("", style="dim"))

        throttle_text = Text(g["throttle_label"], style=g["throttle_style"])
        util_table.add_row("Throttle", throttle_text, Text(""))

        # ── Temperatures ────────────────────────────────────────────────────
        temp_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
        temp_table.add_column("Sensor",       style="bold", width=16)
        temp_table.add_column("°C",  justify="right",       width=6)
        temp_table.add_column("History",                    width=22)

        for label, key in [
            ("Edge",       "temp_edge"),
            ("Junction",   "temp_junction"),
            ("Memory",     "temp_mem"),
            ("VR GFX",     "temp_vrgfx"),
            ("VR SoC",     "temp_vrsoc"),
            ("VR Memory",  "temp_vrmem"),
        ]:
            val = g[key]
            if key.startswith("temp_vr") and not val:
                continue  # amd-smi does not report VR temperatures
            temp_table.add_row(
                label,
                Text(f"{val:.1f}", style=color_temp(val)),
                Text(sparkline(hist[key]), style=f"dim {color_temp(val)}"),
            )

        # ── Clock Frequencies ────────────────────────────────────────────────
        clock_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
        clock_table.add_column("Clock",         style="bold", width=22)
        clock_table.add_column("MHz", justify="right",        width=6)
        clock_table.add_column("Avg",           justify="right", width=6)
        clock_table.add_column("History",                     width=18)

        avg_gfx = g["avg_gfxclk"]
        for label, key, avg in [
            ("SCLK  (GPU Core)",    "sclk",   f"{avg_gfx:.0f}" if avg_gfx else "—"),
            ("MCLK  (Memory)",      "mclk",   "—"),
            ("FCLK  (Inf.Fabric)",  "fclk",   "—"),
            ("SOCCLK (SoC)",        "socclk", "—"),
        ]:
            val = g[key]
            clock_table.add_row(
                label,
                Text(f"{val:.0f}",  style="bold white"),
                Text(avg,           style="dim white"),
                Text(sparkline(hist[key]), style="dim white"),
            )

        # ── Voltages & System ────────────────────────────────────────────────
        sys_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        sys_table.add_column("key",   style="bold",           width=16)
        sys_table.add_column("value", style="cyan",           width=16)
        sys_table.add_column("spark",                         width=18)

        sys_table.add_row(
            "Volt GFX", f"{g['voltage_gfx']:.0f} mV",
            Text(sparkline(hist["voltage_gfx"]), style="dim cyan"),
        )
        sys_table.add_row(
            "Volt SoC", f"{g['voltage_soc']:.0f} mV",
            Text(sparkline(hist["voltage_soc"]), style="dim cyan"),
        )
        sys_table.add_row(
            "Volt Mem", f"{g['voltage_mem']:.0f} mV",
            Text(sparkline(hist["voltage_mem"]), style="dim cyan"),
        )
        pcie_str = (
            f"x{g['pcie_width']:.0f}  {g['pcie_speed']:.1f} GT/s"
            if g["pcie_width"] and g["pcie_speed"] else "N/A"
        )
        sys_table.add_row("PCIe",        pcie_str,          Text(""))
        sys_table.add_row("Perf Level",  g["perf_level"],   Text(""))
        sys_table.add_row("Power Profile", g["power_profile"], Text(""))
        sys_table.add_row("PwrCap",      f"{g['power_cap']:.0f} W", Text(""))

        # ── Active Processes ─────────────────────────────────────────────────
        proc_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
        proc_table.add_column("PID",   style="bold", width=8)
        proc_table.add_column("Process",             width=16)
        proc_table.add_column("VRAM",  justify="right", width=10)
        proc_table.add_column("GPU(s)",              width=6)
        proc_table.add_column("Loaded Models",              width=100)

        if g["processes"]:
            for p in g["processes"]:
                proc_table.add_row(p["pid"], p["name"], p["vram"], p["gpus"],str(loadedModels))
        else:
            proc_table.add_row("[dim]—[/dim]", "[dim]idle[/dim]", "", "")

        # ── Inner layout ─────────────────────────────────────────────────────
        inner = Layout()
        inner.split_column(
            Layout(Panel(util_table,   title="Utilization & Power",  box=box.ROUNDED, title_align="left"), size=12),
            Layout(Panel(temp_table,   title="Temperatures",         box=box.ROUNDED, title_align="left"), size=10),
            Layout(Panel(clock_table,  title="Clock Frequencies",    box=box.ROUNDED, title_align="left"), size=8),
            Layout(Panel(sys_table,    title="Voltages & System",    box=box.ROUNDED, title_align="left"), size=11),
            Layout(Panel(proc_table,   title="Active Processes",     box=box.ROUNDED, title_align="left"), size=6),
        )

        layout[f"gpu{idx}"].update(
            Panel(
                inner,
                title=f"[bold cyan] GPU {idx}: {g['name']} [/bold cyan]",
                box=box.HEAVY,
                title_align="left",
            )
        )

    # ── Journal Log Panel ────────────────────────────────────────────────────
    if log_entries is not None:
        log_text = Text()
        visible = log_entries[-8:] if log_entries else []
        if visible:
            for e in visible:
                log_text.append(f"{e.timestamp}  ", style="dim")
                log_text.append(e.message, style=e.style)
                log_text.append("\n")
        else:
            log_text = Text("  Waiting for GPU log entries…", style="dim")
        layout["logs"].update(
            Panel(log_text, title="GPU Kernel Log  (journalctl -k)", title_align="left", box=box.ROUNDED)
        )
    else:
        layout["logs"].update(
            Panel(
                Text("  journalctl not available on this system.", style="dim"),
                title="GPU Kernel Log",
                title_align="left",
                box=box.ROUNDED,
            )
        )

    return layout


# ---------------------------------------------------------------------------
# Shared data store (dashboard loop → API)
# ---------------------------------------------------------------------------

class DataStore:
    """Thread-safe container for the latest GPU snapshot used by the API."""

    def __init__(self):
        self._lock    = threading.Lock()
        self._gpus:   list[dict] = []
        self._logs:   list[dict] = []
        self._updated: str | None = None

    def update(self, gpus: list[dict], logs: list[LogEntry]) -> None:
        serialisable_gpus = []
        for g in gpus:
            entry = {k: v for k, v in g.items() if k not in ("throttle_style",)}
            serialisable_gpus.append(entry)

        serialisable_logs = [
            {"timestamp": e.timestamp, "message": e.message, "severity": _severity(e.style)}
            for e in logs
        ]
        with self._lock:
            self._gpus    = serialisable_gpus
            self._logs    = serialisable_logs
            self._updated = datetime.now().isoformat()

    def gpu_list(self) -> list[dict]:
        with self._lock:
            return list(self._gpus)

    def gpu_by_idx(self, idx: int) -> dict | None:
        with self._lock:
            for g in self._gpus:
                if g["idx"] == idx:
                    return g
        return None

    def log_list(self, n: int = 100) -> list[dict]:
        with self._lock:
            return list(self._logs)[-n:]

    def meta(self) -> dict:
        with self._lock:
            return {"updated": self._updated, "gpu_count": len(self._gpus)}


def _severity(style: str) -> str:
    if "red"    in style: return "error"
    if "yellow" in style: return "warning"
    if "dim"    in style: return "verbose"
    return "info"


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

def load_api_token(env_path: Path) -> str | None:
    """Load API_TOKEN from a .env file (simple key=value parser, no shell expansion)."""
    if not env_path.exists():
        return None
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        if key.strip() == "API_TOKEN":
            return val.strip().strip('"').strip("'") or None
    return None


def load_api_config(env_path: Path) -> dict:
    """Load API_HOST, API_PORT and API_URL from a .env file."""
    config = {"host": "0.0.0.0", "port": 8080, "api_url": None}
    if not env_path.exists():
        return config
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key == "API_HOST" and val:
            config["host"] = val
        elif key == "API_PORT" and val.isdigit():
            config["port"] = int(val)
        elif key == "API_URL" and val:
            config["api_url"] = val.rstrip("/")
    return config


def create_api_app(store: DataStore, token: str, info: dict | None = None):
    """Build and return the FastAPI application."""
    try:
        from fastapi import Depends, FastAPI, HTTPException, Security, status
        from fastapi.security import APIKeyHeader
    except ImportError:
        return None

    app = FastAPI(
        title="ROCm SMI REST API",
        description="Live AMD GPU metrics from amd-smi",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
    )

    api_key_header = APIKeyHeader(name="X-API-Token", auto_error=True)

    async def require_token(key: str = Security(api_key_header)) -> str:
        if key != token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API token.",
            )
        return key

    @app.get("/api/v1/health", tags=["system"])
    async def health():
        """Public health check — no authentication required."""
        return {"status": "ok", **store.meta()}

    @app.get("/api/v1/info", tags=["system"])
    async def get_info():
        """System info: driver version, amd-smi version. No authentication required."""
        return info or {}

    @app.get("/api/v1/gpus", tags=["gpu"], dependencies=[Depends(require_token)])
    async def list_gpus():
        """Return the latest snapshot of all GPUs."""
        return {"gpus": store.gpu_list(), **store.meta()}

    @app.get("/api/v1/gpus/{idx}", tags=["gpu"], dependencies=[Depends(require_token)])
    async def get_gpu(idx: int):
        """Return data for a single GPU by device index."""
        g = store.gpu_by_idx(idx)
        if g is None:
            raise HTTPException(status_code=404, detail=f"GPU {idx} not found.")
        return g

    @app.get("/api/v1/gpus/{idx}/temperatures", tags=["gpu"], dependencies=[Depends(require_token)])
    async def get_temperatures(idx: int):
        g = store.gpu_by_idx(idx)
        if g is None:
            raise HTTPException(status_code=404, detail=f"GPU {idx} not found.")
        return {
            "gpu_idx": idx,
            "edge_c":     g.get("temp_edge"),
            "junction_c": g.get("temp_junction"),
            "memory_c":   g.get("temp_mem"),
            "vr_gfx_c":   g.get("temp_vrgfx"),
            "vr_soc_c":   g.get("temp_vrsoc"),
            "vr_mem_c":   g.get("temp_vrmem"),
        }

    @app.get("/api/v1/gpus/{idx}/clocks", tags=["gpu"], dependencies=[Depends(require_token)])
    async def get_clocks(idx: int):
        g = store.gpu_by_idx(idx)
        if g is None:
            raise HTTPException(status_code=404, detail=f"GPU {idx} not found.")
        return {
            "gpu_idx": idx,
            "sclk_mhz":    g.get("sclk"),
            "mclk_mhz":    g.get("mclk"),
            "fclk_mhz":    g.get("fclk"),
            "socclk_mhz":  g.get("socclk"),
            "avg_gfxclk_mhz": g.get("avg_gfxclk"),
        }

    @app.get("/api/v1/gpus/{idx}/power", tags=["gpu"], dependencies=[Depends(require_token)])
    async def get_power(idx: int):
        g = store.gpu_by_idx(idx)
        if g is None:
            raise HTTPException(status_code=404, detail=f"GPU {idx} not found.")
        return {
            "gpu_idx":      idx,
            "avg_w":        g.get("power_avg"),
            "cap_w":        g.get("power_cap"),
            "throttle":     g.get("throttle_label"),
            "throttle_raw": g.get("throttle_val"),
            "perf_level":   g.get("perf_level"),
            "power_profile":g.get("power_profile"),
        }

    @app.get("/api/v1/gpus/{idx}/memory", tags=["gpu"], dependencies=[Depends(require_token)])
    async def get_memory(idx: int):
        g = store.gpu_by_idx(idx)
        if g is None:
            raise HTTPException(status_code=404, detail=f"GPU {idx} not found.")
        return {
            "gpu_idx":        idx,
            "vram_used_mb":   g.get("vram_used_mb"),
            "vram_total_mb":  g.get("vram_total_mb"),
            "vram_pct":       g.get("vram_pct"),
            "mem_activity_pct": g.get("mem_activity"),
        }

    @app.get("/api/v1/gpus/{idx}/processes", tags=["gpu"], dependencies=[Depends(require_token)])
    async def get_processes(idx: int):
        g = store.gpu_by_idx(idx)
        if g is None:
            raise HTTPException(status_code=404, detail=f"GPU {idx} not found.")
        return {"gpu_idx": idx, "processes": g.get("processes", [])}

    @app.get("/api/v1/logs", tags=["system"], dependencies=[Depends(require_token)])
    async def get_logs(limit: int = 100):
        """Return recent GPU kernel log entries (from journalctl)."""
        limit = max(1, min(limit, 500))
        return {"logs": store.log_list(limit)}

    return app


def start_api_server(app, host: str, port: int) -> None:
    """Launch uvicorn in a daemon thread."""
    try:
        import uvicorn
        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        server = uvicorn.Server(config)
        server.run()
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# API client (client mode: UI fetches from remote API)
# ---------------------------------------------------------------------------

class ApiClient:
    """Fetches GPU data and logs from a remote rocm-visualizer API server."""

    def __init__(self, base_url: str, token: str):
        self.base_url   = base_url.rstrip("/")
        self._headers   = {"X-API-Token": token, "Accept": "application/json"}
        self.last_error: str | None = None

    def _get(self, path: str) -> dict | None:
        import json
        import urllib.error
        import urllib.request
        url = self.base_url + path
        req = urllib.request.Request(url, headers=self._headers)
        try:
            with urllib.request.urlopen(req, timeout=5) as r:
                self.last_error = None
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            self.last_error = f"HTTP {e.code} {e.reason}"
        except urllib.error.URLError as e:
            self.last_error = f"Connection error: {e.reason}"
        except Exception as e:
            self.last_error = str(e)
        return None

    def fetch_gpus(self) -> list[dict]:
        loadedModels = self._get("/api/ps")
        data = self._get("/api/v1/gpus")
        if data is None:
            return []
        gpus = data.get("gpus", [])
        # throttle_style is UI-internal and not serialised by the server — rebuild it
        for g in gpus:
            _, style = decode_throttle(g.get("throttle_val", 0))
            g["throttle_style"] = style
        return gpus

    def fetch_logs(self, limit: int = 100) -> list[LogEntry]:
        data = self._get(f"/api/v1/logs?limit={limit}")
        if data is None:
            return []
        _style_map = {"error": "bold red", "warning": "yellow", "verbose": "dim", "info": "white"}
        return [
            LogEntry(
                e.get("timestamp", ""),
                e.get("message", ""),
                _style_map.get(e.get("severity", "info"), "white"),
            )
            for e in data.get("logs", [])
        ]

    def fetch_loaded_models(self) -> str:
        return self._get("/api/ps") or "yes"
    def fetch_info(self) -> dict:
        return self._get("/api/v1/info") or {}

    def check_health(self) -> bool:
        import urllib.request, urllib.error, json
        try:
            with urllib.request.urlopen(self.base_url + "/api/v1/health", timeout=3) as r:
                return json.loads(r.read()).get("status") == "ok"
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ROCm SMI live dashboard",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--interval", "-i",
        type=float, default=2.0, metavar="SECONDS",
        help="Refresh interval in seconds",
    )
    parser.add_argument(
        "--api", action="store_true",
        help="Enable the REST API server — server mode (requires .env with API_TOKEN)",
    )
    parser.add_argument(
        "--env", default=".env", metavar="FILE",
        help="Path to the .env file",
    )
    args = parser.parse_args()

    if args.interval < 0.5:
        print("Minimum interval is 0.5 seconds.", file=sys.stderr)
        sys.exit(1)

    console = Console()
    env_path = Path(args.env)
    api_cfg  = load_api_config(env_path)
    api_token = load_api_token(env_path)

    history_keys = [
        "gpu_use", "vram_pct", "mem_activity", "vcn_activity",
        "power_avg", "fan_pct",
        "temp_edge", "temp_junction", "temp_mem",
        "temp_vrgfx", "temp_vrsoc", "temp_vrmem",
        "sclk", "mclk", "fclk", "socclk",
        "voltage_gfx", "voltage_soc", "voltage_mem",
    ]

    # ── CLIENT MODE ───────────────────────────────────────────────────────────
    if api_cfg["api_url"]:
        if not api_token:
            console.print(
                f"[red]API_URL is set but no API_TOKEN found in {env_path}.[/red]"
            )
            sys.exit(1)

        client = ApiClient(api_cfg["api_url"], api_token)
        console.print(f"[cyan]Client mode — connecting to {api_cfg['api_url']} …[/cyan]")

        if not client.check_health():
            console.print(f"[red]Cannot reach {api_cfg['api_url']}/api/v1/health — is the server running?[/red]")
            sys.exit(1)

        info          = client.fetch_info()
        driver_version = info.get("driver_version", "N/A")
        rv             = info.get("amd_smi_version", info.get("rocm_smi_version", [0, 0]))
        rocm_version   = (rv[0], rv[1]) if isinstance(rv, list) else (0, 0)
        console.print(f"[green]Connected. Driver: {driver_version}  amd-smi: {rocm_version[0]}.{rocm_version[1]}[/green]")

        gpus      = client.fetch_gpus()
        loadedModels      = client.fetch_loaded_models()
        histories = {
            g["idx"]: {k: deque(maxlen=HISTORY_LEN) for k in history_keys}
            for g in gpus
        }

        with Live(console=console, refresh_per_second=4, screen=True) as live:
            while True:
                gpus = client.fetch_gpus()
                logs = client.fetch_logs(200)

                if client.last_error:
                    # Show error overlay instead of crashing
                    err_text = Text(f"  Connection error: {client.last_error}", style="bold red")
                    live.update(Panel(err_text, title="ROCm SMI Dashboard — DISCONNECTED", title_align="left"))
                    time.sleep(args.interval)
                    continue

                for g in gpus:
                    idx = g["idx"]
                    if idx not in histories:
                        histories[idx] = {k: deque(maxlen=HISTORY_LEN) for k in history_keys}
                    for k in history_keys:
                        v = g.get(k)
                        if v is not None:
                            histories[idx][k].append(v)

                live.update(build_dashboard(
                    gpus,loadedModels, histories, args.interval,
                    rocm_version,
                    log_entries=logs,
                    driver_version=driver_version,
                ))
                time.sleep(args.interval)
        return

    # ── SERVER / LOCAL MODE ───────────────────────────────────────────────────
    console.print("[cyan]Local mode — detecting amd-smi version…[/cyan]")
    rocm_version = detect_amd_smi_version()
    if rocm_version == (0, 0):
        console.print("[red]amd-smi not found or not working. Is it installed and in $PATH?[/red]")
        sys.exit(1)
    console.print(f"[green]amd-smi {rocm_version[0]}.{rocm_version[1]}[/green]")

    log_watcher = JournalWatcher()
    log_watcher.start()
    if log_watcher.available:
        console.print("[green]Journal watcher started.[/green]")
    else:
        console.print("[dim]journalctl not available — log panel disabled.[/dim]")

    driver_version = fetch_driver_version()
    console.print(f"[green]AMD Driver: {driver_version}[/green]")

    # ── Optional API server ───────────────────────────────────────────────────
    store = DataStore()
    if args.api:
        if not api_token:
            console.print(
                f"[red]--api requires API_TOKEN in {env_path}. "
                f"Copy .env.example to .env and set a token.[/red]"
            )
            sys.exit(1)
        sys_info = {
            "driver_version":   driver_version,
            "amd_smi_version": list(rocm_version),
        }
        api_app = create_api_app(store, api_token, info=sys_info)
        if api_app is None:
            console.print("[red]fastapi/uvicorn not installed. Run: pip install fastapi uvicorn[/red]")
            sys.exit(1)
        threading.Thread(
            target=start_api_server,
            args=(api_app, api_cfg["host"], api_cfg["port"]),
            daemon=True,
        ).start()
        console.print(
            f"[green]REST API listening on "
            f"http://{api_cfg['host']}:{api_cfg['port']}/api/docs[/green]"
        )

    static_info = fetch_static_info()
    power_caps  = {i: d["power_cap"] for i, d in static_info.items() if d["power_cap"]}

    loadedModels      = fetch_loaded_models()
    gpus = collect_gpu_data(static_info, power_caps)
    if not gpus:
        console.print("[red]No ROCm GPUs found. Is amd-smi installed?[/red]")
        sys.exit(1)

    histories = {
        g["idx"]: {k: deque(maxlen=HISTORY_LEN) for k in history_keys}
        for g in gpus
    }

    with Live(console=console, refresh_per_second=4, screen=True) as live:
        while True:
            gpus = collect_gpu_data(static_info, power_caps)
            logs = log_watcher.recent(200) if log_watcher.available else None

            for g in gpus:
                for k in history_keys:
                    v = g.get(k)
                    if v is not None:
                        histories[g["idx"]][k].append(v)

            if args.api:
                store.update(gpus, logs or [])

            live.update(build_dashboard(
                gpus,loadedModels, histories, args.interval,
                rocm_version,
                log_entries=logs,
                driver_version=driver_version,
            ))
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
