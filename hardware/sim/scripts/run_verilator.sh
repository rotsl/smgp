#!/bin/bash
# =============================================================================
# SMGP Hardware Simulation Script (Verilator)
# =============================================================================
set -e

PROJ_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"

# GNU Make cannot handle paths with spaces (or Unicode em-dash).
# 1. Create a temporary symlink so all *source* paths are space-free.
# 2. Build into /tmp/smgp_build (guaranteed no spaces in CURDIR).
SYM_ROOT="/tmp/smgp_sim"
BUILD_DIR="/tmp/smgp_build"
if [ -L "$SYM_ROOT" ]; then rm -f "$SYM_ROOT"; fi
ln -s "$PROJ_ROOT" "$SYM_ROOT"
trap 'rm -f "$SYM_ROOT"' EXIT

RTL_DIR="$SYM_ROOT/hardware/rtl"
TB_DIR="$SYM_ROOT/hardware/sim/testbench"
WAVES_DIR="$SYM_ROOT/hardware/sim/waves"

mkdir -p "$BUILD_DIR" "$WAVES_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo " SMGP Hardware Simulation (Verilator)"
echo "=========================================="

check_verilator() {
    if ! command -v verilator &> /dev/null; then
        echo -e "${RED}[ERROR] Verilator not found. Install with:${NC}"
        echo "  apt-get install verilator"
        exit 1
    fi
}

run_tb() {
    local name=$1
    local tb_file="$TB_DIR/${name}.sv"
    local mdir="$BUILD_DIR/$name"

    if [ ! -f "$tb_file" ]; then
        echo -e "${YELLOW}[SKIP] $name (file not found)${NC}"
        return 0
    fi

    echo -e "${YELLOW}[BUILD] $name${NC}"
    mkdir -p "$mdir"

    # Step 1: generate C++ (no --build; we invoke make manually to avoid
    # the 'spaces in CURDIR' limitation of Verilator's bundled make call)
    verilator \
        --cc \
        --exe \
        --main \
        --timing \
        --top-module "$name" \
        -I"$RTL_DIR" \
        -I"$RTL_DIR/isa" \
        -I"$RTL_DIR/lib" \
        -I"$RTL_DIR/compute" \
        -I"$RTL_DIR/memory" \
        -I"$RTL_DIR/interconnect" \
        -I"$RTL_DIR/top" \
        -I"$TB_DIR" \
        -Wall \
        -Wno-fatal \
        -Wno-UNUSED \
        -Wno-UNDRIVEN \
        --Mdir "$mdir" \
        "$RTL_DIR/isa/smgp_isa_pkg.sv" \
        "$RTL_DIR/lib/fixed_point_pkg.sv" \
        "$RTL_DIR/lib/hd_ops_pkg.sv" \
        "$RTL_DIR/compute/spectral_engine.sv" \
        "$RTL_DIR/compute/hd_engine.sv" \
        "$RTL_DIR/compute/topology_engine.sv" \
        "$RTL_DIR/compute/graph_rewrite_engine.sv" \
        "$RTL_DIR/memory/associative_cache.sv" \
        "$RTL_DIR/memory/memristor_crossbar_sim.sv" \
        "$RTL_DIR/memory/hbm_controller.sv" \
        "$RTL_DIR/interconnect/noc_router.sv" \
        "$RTL_DIR/interconnect/graph_dma.sv" \
        "$RTL_DIR/top/smgp_core.sv" \
        "$RTL_DIR/top/smgp_system.sv" \
        "$tb_file" \
        2>&1

    if [ $? -ne 0 ]; then
        echo -e "${RED}[FAIL] $name code-gen failed${NC}"
        return 1
    fi

    # Step 2: compile (mdir is /tmp/smgp_build/... - guaranteed no spaces)
    make -C "$mdir" -f "V${name}.mk" -j "$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)" 2>&1

    if [ $? -ne 0 ]; then
        echo -e "${RED}[FAIL] $name build failed${NC}"
        return 1
    fi

    echo -e "${GREEN}[PASS] $name built successfully${NC}"

    echo -e "${YELLOW}[SIM] Running $name...${NC}"
    "$mdir/V${name}" +define+WAVES_DIR="$WAVES_DIR"

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[PASS] $name simulation passed${NC}"
    else
        echo -e "${RED}[FAIL] $name simulation failed${NC}"
        return 1
    fi
}

lint_only() {
    echo -e "${YELLOW}[LINT] Checking RTL with Verilator...${NC}"
    verilator \
        --lint-only \
        -I"$RTL_DIR" \
        -I"$RTL_DIR/isa" \
        -I"$RTL_DIR/lib" \
        -I"$RTL_DIR/compute" \
        -I"$RTL_DIR/memory" \
        -I"$RTL_DIR/interconnect" \
        -I"$RTL_DIR/top" \
        -Wall \
        -Wno-fatal \
        -Wno-UNUSED \
        -Wno-UNDRIVEN \
        "$RTL_DIR/isa/smgp_isa_pkg.sv" \
        "$RTL_DIR/lib/fixed_point_pkg.sv" \
        "$RTL_DIR/lib/hd_ops_pkg.sv" \
        "$RTL_DIR/compute/spectral_engine.sv" \
        "$RTL_DIR/compute/hd_engine.sv" \
        "$RTL_DIR/compute/topology_engine.sv" \
        "$RTL_DIR/compute/graph_rewrite_engine.sv" \
        "$RTL_DIR/memory/associative_cache.sv" \
        "$RTL_DIR/memory/memristor_crossbar_sim.sv" \
        "$RTL_DIR/memory/hbm_controller.sv" \
        "$RTL_DIR/interconnect/noc_router.sv" \
        "$RTL_DIR/interconnect/graph_dma.sv" \
        "$RTL_DIR/top/smgp_core.sv" \
        "$RTL_DIR/top/smgp_system.sv" \
        2>&1

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[PASS] Lint passed${NC}"
    else
        echo -e "${RED}[FAIL] Lint failed${NC}"
        exit 1
    fi
}

# Main
MODE="${1:-all}"

case "$MODE" in
    lint)
        check_verilator
        lint_only
        ;;
    tb_spectral_engine|tb_hd_engine|tb_topology_engine|tb_graph_rewrite|tb_associative_memory|tb_system_end_to_end)
        check_verilator
        run_tb "$MODE"
        ;;
    all)
        check_verilator
        lint_only
        FAILED=0
        for tb in tb_spectral_engine tb_hd_engine tb_topology_engine tb_graph_rewrite tb_associative_memory tb_system_end_to_end; do
            run_tb "$tb" || FAILED=$((FAILED + 1))
        done
        echo "=========================================="
        echo -e " Results: $((6 - FAILED))/6 passed"
        if [ $FAILED -gt 0 ]; then
            echo -e "${RED} $FAILED test(s) FAILED${NC}"
            exit 1
        else
            echo -e "${GREEN} All tests PASSED${NC}"
        fi
        ;;
    *)
        echo "Usage: $0 {lint|tb_name|all}"
        echo "  lint                        - Lint only (no simulation)"
        echo "  all                         - Lint + all testbenches"
        echo "  tb_spectral_engine          - Run spectral engine TB"
        echo "  tb_hd_engine                - Run HD engine TB"
        echo "  tb_topology_engine          - Run topology engine TB"
        echo "  tb_graph_rewrite            - Run graph rewrite TB"
        echo "  tb_associative_memory       - Run associative memory TB"
        echo "  tb_system_end_to_end        - Run end-to-end system TB"
        exit 1
        ;;
esac
