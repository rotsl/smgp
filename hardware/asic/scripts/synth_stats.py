#!/usr/bin/env python3
"""
OpenLane Synthesis Report Parser and Statistics Summary

Parses the output of an OpenLane ASIC synthesis run and extracts:
  - Area reports (cell area, utilization)
  - Timing reports (setup slack, WNS, TNS)
  - Power reports (dynamic, leakage, total)
  - Cell count by type

Usage:
    python synth_stats.py <openlane_run_directory>

Example:
    python synth_stats.py runs/RUN_2024-01-15_14-30-00/
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional


# ===========================================================================
# Report Parsers
# ===========================================================================

def parse_area_report(filepath: str) -> Dict:
    """Parse OpenLane area report (stats/area.rpt or logs/synthesis/5-report_area.txt)."""
    result = {
        "total_area_um2": 0.0,
        "cell_area_um2": 0.0,
        "net_area_um2": 0.0,
        "utilization_pct": 0.0,
        "cell_count": 0,
        "cell_counts_by_type": {},
    }

    try:
        with open(filepath, "r") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"  [WARN] Area report not found: {filepath}")
        return result

    # OpenLane / OpenROAD area format
    # "Core Area" or "Design area" line
    for pattern, key in [
        (r"Design area:\s*([\d.]+)\s*um\^2", "cell_area_um2"),
        (r"Core area:\s*([\d.]+)\s*um\^2", "total_area_um2"),
        (r"Total area:\s*([\d.]+)\s*um\^2", "total_area_um2"),
    ]:
        m = re.search(pattern, content, re.IGNORECASE)
        if m:
            result[key] = float(m.group(1))

    # Utilization
    m = re.search(r"Utilization[:\s]+([\d.]+)\s*%", content)
    if m:
        result["utilization_pct"] = float(m.group(1))

    # Instance count
    m = re.search(r"Number of instances[:\s]+(\d+)", content)
    if m:
        result["cell_count"] = int(m.group(1))

    # Per-cell-type counts
    for m in re.finditer(r"^\s*(\w+)\s+(\d+)\s+[\d.]+\s*$", content, re.MULTILINE):
        cell_type, count = m.group(1), int(m.group(2))
        result["cell_counts_by_type"][cell_type] = count

    return result


def parse_timing_report(filepath: str) -> Dict:
    """Parse OpenLane timing report (logs/synthesis/4-cts.rpt or similar)."""
    result = {
        "clock_period_ns": 0.0,
        "wns_ns": 0.0,
        "tns_ns": 0.0,
        "slack_met": False,
        "critical_path_ns": 0.0,
        "max_freq_mhz": 0.0,
    }

    try:
        with open(filepath, "r") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"  [WARN] Timing report not found: {filepath}")
        return result

    # WNS (Worst Negative Slack)
    m = re.search(r"WNS[:\s]+([-\d.]+)\s*ns", content)
    if m:
        result["wns_ns"] = float(m.group(1))
        result["slack_met"] = result["wns_ns"] >= 0.0

    # TNS (Total Negative Slack)
    m = re.search(r"TNS[:\s]+([-\d.]+)\s*ns", content)
    if m:
        result["tns_ns"] = float(m.group(1))

    # Clock period
    m = re.search(r"Clock period[:\s]+([\d.]+)\s*ns", content)
    if m:
        result["clock_period_ns"] = float(m.group(1))

    # Critical path delay
    m = re.search(r"Critical path delay[:\s]+([\d.]+)\s*ns", content)
    if m:
        result["critical_path_ns"] = float(m.group(1))

    # Derive max frequency
    if result["critical_path_ns"] > 0:
        result["max_freq_mhz"] = 1000.0 / result["critical_path_ns"]
    elif result["clock_period_ns"] > 0:
        result["max_freq_mhz"] = 1000.0 / result["clock_period_ns"]

    return result


def parse_power_report(filepath: str) -> Dict:
    """Parse OpenLane power report."""
    result = {
        "dynamic_mw": 0.0,
        "leakage_mw": 0.0,
        "total_mw": 0.0,
    }

    try:
        with open(filepath, "r") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"  [WARN] Power report not found: {filepath}")
        return result

    for pattern, key in [
        (r"Total Dynamic Power[:\s]+([\d.]+)\s*mW", "dynamic_mw"),
        (r"Cell Leakage Power[:\s]+([\d.]+)\s*mW", "leakage_mw"),
        (r"Total Power[:\s]+([\d.]+)\s*mW", "total_mw"),
    ]:
        m = re.search(pattern, content, re.IGNORECASE)
        if m:
            result[key] = float(m.group(1))

    return result


# ===========================================================================
# Report Discovery
# ===========================================================================

def find_reports(run_dir: str) -> Dict[str, Optional[str]]:
    """Search for standard OpenLane report files in a run directory."""
    run_path = Path(run_dir)
    reports = {
        "area": None,
        "timing": None,
        "power": None,
    }

    # Common locations for OpenLane reports
    search_patterns = {
        "area": [
            "results/synthesis/smgp_core.area.stat",
            "logs/synthesis/5-report_area.txt",
            "reports/synthesis/area.rpt",
        ],
        "timing": [
            "results/synthesis/smgp_core.timing.stat",
            "logs/synthesis/4-cts.rpt",
            "reports/synthesis/timing.rpt",
        ],
        "power": [
            "results/synthesis/smgp_core.power.stat",
            "logs/synthesis/6-report_power.txt",
            "reports/synthesis/power.rpt",
        ],
    }

    for category, patterns in search_patterns.items():
        for pattern in patterns:
            candidate = run_path / pattern
            if candidate.exists():
                reports[category] = str(candidate)
                break

        # Fallback: recursive search
        if reports[category] is None:
            for p in run_path.rglob("*"):
                if p.is_file():
                    name_lower = p.name.lower()
                    if category == "area" and "area" in name_lower and p.suffix in (".rpt", ".txt", ".stat"):
                        reports[category] = str(p)
                        break
                    elif category == "timing" and ("timing" in name_lower or "sta" in name_lower) and p.suffix in (".rpt", ".txt", ".stat"):
                        reports[category] = str(p)
                        break
                    elif category == "power" and "power" in name_lower and p.suffix in (".rpt", ".txt", ".stat"):
                        reports[category] = str(p)
                        break

    return reports


# ===========================================================================
# Summary Printer
# ===========================================================================

def print_summary(area: Dict, timing: Dict, power: Dict) -> None:
    """Print a formatted summary of synthesis results."""
    print()
    print("=" * 64)
    print("  SMGPU ASIC Synthesis Results Summary")
    print("=" * 64)

    # Area
    print()
    print("  AREA")
    print("  " + "-" * 40)
    print(f"    Design area:        {area['cell_area_um2']:>12.2f} um^2")
    print(f"    Core area:          {area['total_area_um2']:>12.2f} um^2")
    print(f"    Utilization:        {area['utilization_pct']:>12.1f} %")
    print(f"    Cell count:         {area['cell_count']:>12d}")

    # Top 5 cell types
    if area["cell_counts_by_type"]:
        sorted_cells = sorted(
            area["cell_counts_by_type"].items(), key=lambda x: x[1], reverse=True
        )[:5]
        print(f"    Top cell types:")
        for cell_type, count in sorted_cells:
            print(f"      {cell_type:<30s} {count:>8d}")

    # Timing
    print()
    print("  TIMING")
    print("  " + "-" * 40)
    print(f"    Clock period:       {timing['clock_period_ns']:>12.2f} ns")
    print(f"    Critical path:      {timing['critical_path_ns']:>12.2f} ns")
    print(f"    WNS:                {timing['wns_ns']:>12.3f} ns")
    print(f"    TNS:                {timing['tns_ns']:>12.3f} ns")
    print(f"    Max frequency:      {timing['max_freq_mhz']:>12.1f} MHz")
    slack_str = "MET" if timing["slack_met"] else "VIOLATED"
    print(f"    Slack status:       {slack_str:>12s}")

    # Power
    print()
    print("  POWER")
    print("  " + "-" * 40)
    print(f"    Dynamic:            {power['dynamic_mw']:>12.4f} mW")
    print(f"    Leakage:            {power['leakage_mw']:>12.4f} mW")
    print(f"    Total:              {power['total_mw']:>12.4f} mW")

    print()
    print("=" * 64)


# ===========================================================================
# Main
# ===========================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse OpenLane synthesis reports for SMGPU"
    )
    parser.add_argument(
        "run_dir",
        help="Path to the OpenLane run directory (containing results/ or logs/)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format",
    )
    parser.add_argument(
        "--area-report",
        default=None,
        help="Explicit path to area report (overrides auto-discovery)",
    )
    parser.add_argument(
        "--timing-report",
        default=None,
        help="Explicit path to timing report (overrides auto-discovery)",
    )
    parser.add_argument(
        "--power-report",
        default=None,
        help="Explicit path to power report (overrides auto-discovery)",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.run_dir):
        print(f"ERROR: Run directory not found: {args.run_dir}")
        return 1

    print(f"SMGPU Synthesis Stats — scanning: {args.run_dir}")

    # Discover or use explicit report paths
    discovered = find_reports(args.run_dir)

    area_report = args.area_report or discovered["area"]
    timing_report = args.timing_report or discovered["timing"]
    power_report = args.power_report or discovered["power"]

    # Parse
    area = parse_area_report(area_report) if area_report else {}
    timing = parse_timing_report(timing_report) if timing_report else {}
    power = parse_power_report(power_report) if power_report else {}

    if args.json:
        import json
        summary = {"area": area, "timing": timing, "power": power}
        print(json.dumps(summary, indent=2))
    else:
        print_summary(area, timing, power)

    return 0


if __name__ == "__main__":
    sys.exit(main())
