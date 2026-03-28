#!/usr/bin/env python3
"""
ROCm SMI Live Dashboard
Visualizes AMD GPU metrics in real-time using rocm-smi.
Usage: python rocm_visualizer.py [--interval SECONDS]

Adding support for a new rocm-smi version:
  1. Copy the closest existing entry in PATTERNS below.
  2. Change the key to the new (major, minor) version tuple.
  3. Adjust cli_flags_* and/or regex strings to match the new output format.
  The runtime picks the highest version <= detected version as fallback.
"""

import argparse
import os
import platform
import re
import subprocess
import sys
import threading
import time
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

# Throttle status bitmask → human-readable reason (RDNA2/3, metrics v1.x)
THROTTLE_BITS: dict[int, str] = {
    0:  "PPT0",
    1:  "PPT1",
    2:  "SPL",
    3:  "FPPT",
    4:  "APT",
    5:  "TDC GFX",
    6:  "TDC SOC",
    7:  "TDC MEM",
    8:  "TDC VDD",
    13: "PROCHOT CPU",
    14: "PROCHOT GPU",
    15: "PROCHOT MEM",
}

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
# Version-specific pattern registry
# ---------------------------------------------------------------------------
# Key   : (major, minor) of "ROCM-SMI version" from `rocm-smi --version`
#
# cli_flags_*  : which flags are passed to rocm-smi for each data group
# All other keys : regex patterns.
#   - Per-GPU patterns are prepended with  GPU\[N\]\s*:\s*  at runtime.
#   - Full-line patterns (fan_pct, fan_rpm, power_cap) embed GPU[N] themselves
#     and use two capture groups: group 1 = GPU index, last group = value.
#
# To add a new version: copy the (4, 0) block, bump the key, tweak patterns.
# ---------------------------------------------------------------------------
PATTERNS: dict[tuple[int, int], dict] = {
    (4, 0): {
        # ---- CLI flags per data group ----
        "cli_flags":          ["-t", "-u", "--showmemuse", "-P", "-c", "--showvoltage", "--showid"],
        "cli_flags_powercap": ["--showmaxpower"],
        "cli_flags_fan":      ["--showfan"],
        "cli_flags_metrics":  ["--showmetrics"],
        "cli_flags_procs":    ["--showpids"],
        "cli_flags_perf":     ["--showperflevel"],
        "cli_flags_profile":  ["--showprofile"],
        "cli_flags_vram":     ["--showmeminfo", "vram"],
        "cli_flags_driver":   ["--showdriverversion"],

        # ---- Driver version (global, fetched once) ----
        "driver_version":     r"Driver version:\s*(.+)",

        # ---- Detection ----
        "gpu_index":    r"GPU\[(\d+)\]",

        # ---- Per-GPU: basic ----
        "device_name":  r"Device Name:\s*(.+)",
        "temp_edge":    r"Temperature \(Sensor edge\)[^:]*:\s*([\d.]+)",
        "temp_junction":r"Temperature \(Sensor junction\)[^:]*:\s*([\d.]+)",
        "temp_mem":     r"Temperature \(Sensor memory\)[^:]*:\s*([\d.]+)",
        "gpu_use":      r"GPU use \(%\):\s*(\d+)",
        "vram_pct":      r"GPU Memory Allocated \(VRAM%\):\s*(\d+)",
        "vram_used_b":   r"VRAM Total Used Memory \(B\):\s*(\d+)",
        "vram_total_b":  r"VRAM Total Memory \(B\):\s*(\d+)",
        "mem_activity": r"GPU Memory Read/Write Activity \(%\):\s*(\d+)",
        "power_avg":    r"Average Graphics Package Power \(W\):\s*([\d.]+)",
        "voltage":      r"Voltage \(mV\):\s*(\d+)",
        # clocks: "sclk clock level: 1: (2712Mhz)"
        "sclk":         r"sclk clock level[^(]*\(?(\d+)Mhz",
        "mclk":         r"mclk clock level[^(]*\(?(\d+)Mhz",
        "fclk":         r"fclk clock level[^(]*\(?(\d+)Mhz",
        "socclk":       r"socclk clock level[^(]*\(?(\d+)Mhz",

        # ---- Per-GPU: --showmetrics ----
        "temp_vrgfx":   r"temperature_vrgfx \(C\):\s*([\d.]+)",
        "temp_vrsoc":   r"temperature_vrsoc \(C\):\s*([\d.]+)",
        "temp_vrmem":   r"temperature_vrmem \(C\):\s*([\d.]+)",
        "throttle":     r"throttle_status:\s*(\d+)",
        "pcie_width":   r"pcie_link_width \(Lanes\):\s*(\d+)",
        "pcie_speed":   r"pcie_link_speed \(0\.1 GT/s\):\s*(\d+)",   # ÷10 → GT/s
        "voltage_gfx":  r"voltage_gfx \(mV\):\s*(\d+)",
        "voltage_soc":  r"voltage_soc \(mV\):\s*(\d+)",
        "voltage_mem":  r"voltage_mem \(mV\):\s*(\d+)",
        "vcn_activity": r"vcn_activity \(%\):\s*\[(\d+)",            # first engine
        "avg_gfxclk":   r"average_gfxclk_frequency \(MHz\):\s*(\d+)",
        "fan_rpm_metrics": r"current_fan_speed \(rpm\):\s*(\d+)",

        # ---- Per-GPU: --showperflevel ----
        "perf_level":   r"Performance Level:\s*(\w+)",

        # ---- Full-line patterns (GPU index embedded) ----
        "power_cap":    r"GPU\[(\d+)\][^\n]*Max Graphics Package Power \(W\):\s*([\d.]+)",
        "fan_pct":      r"GPU\[(\d+)\][^\n]*Fan Level[^\(]*\((\d+)%\)",
        "fan_rpm":      r"GPU\[(\d+)\][^\n]*Fan RPM:\s*(\d+)",

        # ---- Full-output: --showprofile (active entry marked with *) ----
        "power_profile": r"Available power profile[^:]*:\s*([^\n*]+)\*",
    },

    # Template for next version — copy, bump key, adjust as needed:
    # (5, 0): { ... },
}


# ---------------------------------------------------------------------------
# Version detection & pattern lookup
# ---------------------------------------------------------------------------

def detect_rocm_smi_version() -> tuple[int, int]:
    try:
        out = subprocess.run(
            ["rocm-smi", "--version"],
            capture_output=True, text=True, timeout=5,
        ).stdout
        m = re.search(r"ROCM-SMI version:\s*(\d+)\.(\d+)", out)
        if m:
            return int(m.group(1)), int(m.group(2))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return (4, 0)


def get_patterns(version: tuple[int, int] | None = None) -> tuple[dict, tuple[int, int]]:
    if not PATTERNS:
        raise RuntimeError("PATTERNS registry is empty.")
    if version is None:
        version = detect_rocm_smi_version()
    candidates = [v for v in PATTERNS if v <= version]
    key = max(candidates) if candidates else min(PATTERNS)
    return PATTERNS[key], key


# ---------------------------------------------------------------------------
# rocm-smi subprocess helper
# ---------------------------------------------------------------------------

def run_rocm_smi(*flags: str) -> str:
    try:
        return subprocess.run(
            ["rocm-smi", *flags],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def _float(text: str, pattern: str, group: int = 1, default=None):
    m = re.search(pattern, text)
    if m:
        try:
            return float(m.group(group))
        except ValueError:
            return m.group(group)
    return default


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def fetch_driver_version(pat: dict) -> str:
    raw = run_rocm_smi(*pat["cli_flags_driver"])
    m = re.search(pat["driver_version"], raw)
    return m.group(1).strip() if m else "N/A"


def fetch_power_caps(pat: dict) -> dict:
    raw = run_rocm_smi(*pat["cli_flags_powercap"])
    caps = {}
    for m in re.finditer(pat["power_cap"], raw):
        caps[int(m.group(1))] = float(m.group(2))
    return caps


def parse_kfd_processes(raw: str) -> list[dict]:
    """Parse the KFD process table from --showpids output."""
    procs = []
    in_table = False
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("PID") and "PROCESS NAME" in line:
            in_table = True
            continue
        if not in_table or not line:
            continue
        parts = line.split("\t")
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) >= 4:
            try:
                vram_bytes = int(parts[3])
                vram_mb = vram_bytes / (1024 ** 2)
                vram_str = f"{vram_mb / 1024:.1f} GB" if vram_mb >= 1024 else f"{vram_mb:.0f} MB"
            except ValueError:
                vram_str = parts[3]
            procs.append({
                "pid":   parts[0],
                "name":  parts[1],
                "gpus":  parts[2],
                "vram":  vram_str,
            })
    return procs


def decode_throttle(status: int) -> tuple[str, str]:
    """Return (label, rich_style) for a throttle_status bitmask value."""
    if status == 0:
        return "OK", "bold green"
    reasons = [name for bit, name in THROTTLE_BITS.items() if status & (1 << bit)]
    label = "THROTTLED" + (f" ({', '.join(reasons)})" if reasons else f" (0x{status:x})")
    return label, "bold red"


def collect_gpu_data(pat: dict, power_caps: dict | None = None) -> list[dict]:
    raw       = run_rocm_smi(*pat["cli_flags"])
    metrics   = run_rocm_smi(*pat["cli_flags_metrics"])
    fan_raw   = run_rocm_smi(*pat["cli_flags_fan"])
    perf_raw  = run_rocm_smi(*pat["cli_flags_perf"])
    prof_raw  = run_rocm_smi(*pat["cli_flags_profile"])
    procs_raw = run_rocm_smi(*pat["cli_flags_procs"])
    vram_raw  = run_rocm_smi(*pat["cli_flags_vram"])

    gpu_indices = sorted(set(re.findall(pat["gpu_index"], raw)))
    if not gpu_indices:
        return []

    processes = parse_kfd_processes(procs_raw)

    # Active power profile (global, not per-GPU)
    prof_m = re.search(pat["power_profile"], prof_raw)
    active_profile = prof_m.group(1).strip() if prof_m else "N/A"

    gpus = []
    for idx in gpu_indices:
        pfx = rf"GPU\[{idx}\]\s*:\s*"

        def pv(key: str, default=None, src: str = raw) -> float | None:
            return _float(src, pfx + pat[key], 1, default)

        def pv_m(key: str, default=None) -> float | None:
            return _float(metrics, pfx + pat[key], 1, default)

        def pv_v(key: str, default=None) -> float | None:
            return _float(vram_raw, pfx + pat[key], 1, default)

        # Fan: full-line patterns embed GPU index
        def fan_val(key: str) -> float | None:
            pattern = pat[key].replace(r"(\d+)", idx, 1)
            m = re.search(pattern, fan_raw)
            if m:
                try:
                    return float(m.group(m.lastindex))
                except (ValueError, TypeError):
                    pass
            return None

        name_m = re.search(pfx + pat["device_name"], raw)
        device_name = name_m.group(1).strip() if name_m else f"GPU {idx}"

        perf_m = re.search(pfx + pat["perf_level"], perf_raw)
        perf_level = perf_m.group(1).strip() if perf_m else "N/A"

        throttle_raw = pv_m("throttle", 0)
        throttle_val = int(throttle_raw) if throttle_raw is not None else 0
        throttle_label, throttle_style = decode_throttle(throttle_val)

        pcie_speed_raw = pv_m("pcie_speed")
        pcie_speed = round(pcie_speed_raw / 10, 1) if pcie_speed_raw else None

        # Fan RPM: prefer --showfan, fall back to --showmetrics
        fan_rpm = fan_val("fan_rpm")
        if fan_rpm is None:
            fan_rpm = pv_m("fan_rpm_metrics")
        fan_pct = fan_val("fan_pct")

        # Note: --showpids "GPU(s)" column is a count, not an index,
        # so we show all KFD processes here (correct for single-GPU setups).
        gpu_procs = processes

        gpus.append({
            "idx":             int(idx),
            "name":            device_name,
            # basic
            "temp_edge":       pv("temp_edge",    0.0),
            "temp_junction":   pv("temp_junction",0.0),
            "temp_mem":        pv("temp_mem",      0.0),
            "gpu_use":         pv("gpu_use",       0.0),
            "vram_pct":        pv("vram_pct",       0.0),
            "vram_used_mb":    (pv_v("vram_used_b",  0.0) or 0.0) / 1024 ** 2,
            "vram_total_mb":   (pv_v("vram_total_b", 1.0) or 1.0) / 1024 ** 2,
            "mem_activity":    pv("mem_activity",  0.0),
            "power_avg":       pv("power_avg",     0.0),
            "power_cap":       (power_caps or {}).get(int(idx), 327.0),
            "sclk":            pv("sclk",          0.0),
            "mclk":            pv("mclk",          0.0),
            "fclk":            pv("fclk",          0.0),
            "socclk":          pv("socclk",        0.0),
            "voltage":         pv("voltage",       0.0),
            "fan_pct":         fan_pct,
            "fan_rpm":         fan_rpm,
            # metrics
            "temp_vrgfx":      pv_m("temp_vrgfx",  0.0),
            "temp_vrsoc":      pv_m("temp_vrsoc",  0.0),
            "temp_vrmem":      pv_m("temp_vrmem",  0.0),
            "throttle_val":    throttle_val,
            "throttle_label":  throttle_label,
            "throttle_style":  throttle_style,
            "pcie_width":      pv_m("pcie_width"),
            "pcie_speed":      pcie_speed,
            "voltage_gfx":     pv_m("voltage_gfx", 0.0),
            "voltage_soc":     pv_m("voltage_soc", 0.0),
            "voltage_mem":     pv_m("voltage_mem", 0.0),
            "vcn_activity":    pv_m("vcn_activity", 0.0),
            "avg_gfxclk":      pv_m("avg_gfxclk",  0.0),
            # system
            "perf_level":      perf_level,
            "power_profile":   active_profile,
            "processes":       gpu_procs,
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
    histories: dict,
    interval: float,
    rocm_version: tuple[int, int],
    pattern_key: tuple[int, int],
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
    ver_str   = f"rocm-smi {rocm_version[0]}.{rocm_version[1]}"
    pat_note  = f"  [dim](patterns {pattern_key[0]}.{pattern_key[1]})[/dim]" \
                if pattern_key != rocm_version else ""
    gpu_names = "  |  ".join(
        f"[bold cyan]GPU {g['idx']}: {g['name']}[/bold cyan]" for g in all_gpus
    ) if all_gpus else "[red]no GPU detected[/red]"

    header_table = Table(box=None, show_header=False, padding=0, expand=True)
    header_table.add_column(justify="left")
    header_table.add_row(
        Text.from_markup(
            f"[bold white on dark_blue] ROCm SMI Dashboard [/bold white on dark_blue]"
            f"   [dim]{now}[/dim]   [bold cyan]{ver_str}[/bold cyan]{pat_note}"
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
            Panel("[red]No ROCm GPU detected or rocm-smi not available.[/red]")
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

        if g["processes"]:
            for p in g["processes"]:
                proc_table.add_row(p["pid"], p["name"], p["vram"], p["gpus"])
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
        description="Live AMD GPU metrics from rocm-smi",
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
        """System info: driver version, rocm-smi version. No authentication required."""
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
        rv             = info.get("rocm_smi_version", [4, 0])
        rocm_version   = (rv[0], rv[1]) if isinstance(rv, list) else (4, 0)
        pattern_key    = rocm_version
        console.print(f"[green]Connected. Driver: {driver_version}  rocm-smi: {rocm_version[0]}.{rocm_version[1]}[/green]")

        gpus      = client.fetch_gpus()
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
                    gpus, histories, args.interval,
                    rocm_version, pattern_key,
                    log_entries=logs,
                    driver_version=driver_version,
                ))
                time.sleep(args.interval)
        return

    # ── SERVER / LOCAL MODE ───────────────────────────────────────────────────
    console.print("[cyan]Local mode — detecting ROCm SMI version…[/cyan]")
    rocm_version = detect_rocm_smi_version()
    pat, pattern_key = get_patterns(rocm_version)

    if pattern_key != rocm_version:
        console.print(
            f"[yellow]rocm-smi {rocm_version[0]}.{rocm_version[1]} not in registry — "
            f"using patterns {pattern_key[0]}.{pattern_key[1]} as fallback.[/yellow]"
        )
    else:
        console.print(f"[green]rocm-smi {rocm_version[0]}.{rocm_version[1]} — patterns matched.[/green]")

    log_watcher = JournalWatcher()
    log_watcher.start()
    if log_watcher.available:
        console.print("[green]Journal watcher started.[/green]")
    else:
        console.print("[dim]journalctl not available — log panel disabled.[/dim]")

    driver_version = fetch_driver_version(pat)
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
            "rocm_smi_version": list(rocm_version),
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

    power_caps = fetch_power_caps(pat)
    gpus = collect_gpu_data(pat, power_caps)
    if not gpus:
        console.print("[red]No ROCm GPUs found. Is rocm-smi installed?[/red]")
        sys.exit(1)

    histories = {
        g["idx"]: {k: deque(maxlen=HISTORY_LEN) for k in history_keys}
        for g in gpus
    }

    with Live(console=console, refresh_per_second=4, screen=True) as live:
        while True:
            gpus = collect_gpu_data(pat, power_caps)
            logs = log_watcher.recent(200) if log_watcher.available else None

            for g in gpus:
                for k in history_keys:
                    v = g.get(k)
                    if v is not None:
                        histories[g["idx"]][k].append(v)

            if args.api:
                store.update(gpus, logs or [])

            live.update(build_dashboard(
                gpus, histories, args.interval,
                rocm_version, pattern_key,
                log_entries=logs,
                driver_version=driver_version,
            ))
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
