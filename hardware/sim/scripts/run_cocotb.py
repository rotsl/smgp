#!/usr/bin/env python3
"""Cocotb-based co-simulation runner for SMGP hardware.

Runs Python-driven testbenches against the RTL using Cocotb and Verilator.
"""
import subprocess
import sys
from pathlib import Path

PROJ_ROOT = Path(__file__).parent.parent.parent
SIM_DIR = PROJ_ROOT / "hardware" / "sim"


def run_cocotb_test(tb_name: str) -> bool:
    """Run a single Cocotb testbench."""
    makefile = SIM_DIR / "Makefile_cocotb"
    if not makefile.exists():
        print(f"[SKIP] Cocotb Makefile not found for {tb_name}")
        return True

    env = {"MODULE": tb_name, "TOPLEVEL_LANG": "verilog"}
    result = subprocess.run(
        ["make", "-C", str(SIM_DIR), "-f", str(makefile)],
        capture_output=True,
        text=True,
        env={**dict(__import__('os').environ), **env},
    )
    if result.returncode != 0:
        print(f"[FAIL] {tb_name}:\n{result.stdout}\n{result.stderr}")
        return False
    print(f"[PASS] {tb_name}")
    return True


def main():
    tests = [
        "tb_spectral_engine",
        "tb_hd_engine",
        "tb_topology_engine",
        "tb_graph_rewrite",
        "tb_associative_memory",
        "tb_system_end_to_end",
    ]

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode == "all":
        failed = sum(1 for t in tests if not run_cocotb_test(t))
        print(f"\nResults: {len(tests) - failed}/{len(tests)} passed")
        sys.exit(1 if failed > 0 else 0)
    elif mode in tests:
        sys.exit(0 if run_cocotb_test(mode) else 1)
    else:
        print(f"Unknown test: {mode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
