# SMGPU — Spectral Memory Graph Processing Unit

> Hardware accelerator for spectral graph theory, hyperdimensional computing,
> topological data analysis, and category-theoretic graph rewriting.

**Repository:** [https://github.com/rotsl/smgp](https://github.com/rotsl/smgp)

## Overview

SMGPU is a configurable RTL design written in SystemVerilog that implements the
core compute kernels of the [SMGP](../README.md) software framework as a
dedicated hardware accelerator. It targets FPGA prototyping (Xilinx UltraScale+)
with a clear migration path to ASIC via standard synthesis flows.

The design centres on four heterogeneous compute engines — Spectral, HD,
Topology, and Rewrite — connected through a 2D mesh Network-on-Chip (NoC) and
backed by a high-bandwidth memory subsystem with an optional memristor
crossbar for analog in-memory computing.

## Block Diagram

```
                              +---------------------------+
                              |       PCIe / AXI Host      |
                              |       Interface            |
                              +-------------+-------------+
                                            |
                              +-------------v-------------+
                              |        ISA Decoder         |
                              |   (32-bit instruction      |
                              |    fetch & dispatch)        |
                              +------+------+------+-------+
                                     |      |      |
            +------------------------+      |      +------------------------+
            |                               |                               |
   +--------v---------+           +---------v----------+           +---------v--------+
   |  Spectral Engine  |          |    HD Engine        |          | Topology Engine  |
   |  - Laplacian      |          |  - Bundle (Maj)     |          |  - Union-Find    |
   |  - Eigen-decomp   |          |  - Bind/Unbind(XOR) |          |  - Barcode emit  |
   |  - Chebyshev      |          |  - Permute (shift)  |          |  - Wasserstein   |
   |  - GFT / Wavelet  |          |  - Similarity       |          |  - Stability chk |
   |  (16x16 systolic) |          |  (16 parallel banks)|          |                  |
   +--------+----------+          +---------+----------+          +--------+---------+
            |                               |                               |
   +--------v---------+          +---------v----------+          +---------v--------+
   |  Rewrite Engine   |          |  Associative Cache  |          |  Memristor       |
   |  - DPO match      |<-------->|  (CAM, 256 entries) |<-------->|  Crossbar Array  |
   |  - DPO apply      |          |  - O(1) recall      |          |  - 1024x1024     |
   |  - Proof trace    |          |  - 1024-dim HD keys |          |  - Analog MVM    |
   +--------+----------+          +---------------------+          +------------------+
            |
   +--------v----------+     +-------------+-------------+     +------------------+
   |    Graph DMA      |<--->|   2D Mesh NoC Router     |<--->|   HBM Controller  |
   |  (scatter-gather) |     |   (5-port, XY routing)  |     |   (4-ch, 256-bit) |
   +-------------------+     +-------------------------+     +------------------+
                                                              |
                                                     +--------v--------+
                                                     |  HBM2 (external) |
                                                     +-----------------+
```

## Features

| Category | Feature |
|----------|---------|
| **Spectral** | Graph Laplacian (normalized), eigendecomposition via power iteration, Chebyshev spectral convolution, graph Fourier transform, heat-kernel wavelets |
| **Hyperdimensional** | Bundle (majority vote), bind/unbind (XOR), permute (cyclic shift), similarity (popcount dot product), associative read/write, random vector generation |
| **Topology** | Vietoris-Rips filtration, persistent homology (Union-Find), persistence barcode streaming, Wasserstein distance, stability verification |
| **Graph Rewrite** | Double-Pushout (DPO) matching, subgraph isomorphism (bounded, backtracking), rule application with proof trace |
| **Memory** | Associative CAM cache (256 entries, 1024-dim keys), HBM2 controller (4 channels, 256-bit), memristor crossbar array (1024x1024, analog MVM) |
| **Interconnect** | 2D mesh NoC (5-port router, XY deterministic routing, 2 virtual channels), graph-aware DMA (scatter-gather, CSR traversal) |
| **Arithmetic** | Parametric fixed-point library (Q8.24 / Q16.16), saturating add/sub, Newton-Raphson reciprocal, integer square root, MAC units |
| **ISA** | 32-bit fixed-width instructions, 8 opcode classes with sub-opcodes, flag-based chaining and streaming modes |

## Directory Structure

```
hardware/
  README_HW.md                  # This file
  rtl/
    isa/
      smgp_isa_pkg.sv           # ISA encoding, opcodes, flags, address map
    lib/
      fixed_point_pkg.sv        # Parametric Q-format arithmetic
      hd_ops_pkg.sv             # HD primitive functions (bundle, bind, permute)
    compute/
      spectral_engine.sv        # Systolic-array spectral transform engine
      hd_engine.sv              # Massively parallel HD vector engine
      topology_engine.sv        # Persistent homology & barcode engine
      graph_rewrite_engine.sv   # DPO graph rewriting engine
    memory/
      associative_cache.sv      # Content-addressable HD vector cache
      hbm_controller.sv         # HBM2 burst read/write controller
      memristor_crossbar_sim.sv # Analog crossbar MVM (behavioral model)
    interconnect/
      graph_dma.sv              # Scatter-gather DMA for graph data
      noc_router.sv             # 5-port 2D mesh NoC router
  sim/
    scripts/
      run_verilator.sh          # Verilator lint & simulation driver
    models/
      gold_spectral_model.py    # Python golden model — spectral ops
      gold_hd_model.py          # Python golden model — HD ops
    testbench/
      tb_spectral_engine.sv     # Testbench — spectral engine
      tb_hd_engine.sv           # Testbench — HD engine
      tb_topology_engine.sv     # Testbench — topology engine
      tb_graph_rewrite.sv       # Testbench — graph rewrite engine
      tb_associative_memory.sv  # Testbench — associative cache
      tb_system_end_to_end.sv   # Full system integration testbench
  sw/
    hal/
      smgpu_hal.py              # Python Hardware Abstraction Layer
      hw_executor.py            # Drop-in HWExecutor backend
      memory_mapper.py          # Host-to-device memory mapping
    driver/
      pcie_driver.c             # Linux kernel PCIe driver skeleton
      pcie_ioctl.h              # IOCTL definitions
  fpga/
    build/
      Makefile                  # Vivado synthesis & implementation
      synth.tcl                 # Vivado synthesis script
      impl.tcl                  # Vivado implementation script
    constraints/
      xcu280.xdc                # Pin and timing constraints
    ip/
      hbm_ip.cfg                # HBM IP configuration
  docs_hw/
    architecture.md             # Detailed architecture documentation
    isa_reference.md            # Complete ISA reference
    integration_guide.md        # HW/SW integration guide
  tests/
    test_hw_integration.py      # pytest — Python HAL integration
    test_golden_models.py       # pytest — golden model validation
    test_memory_mapper.py       # pytest — memory mapping correctness
```

## Quick Start — Simulation with Verilator

### Prerequisites

- Verilator 5.x (`apt install verilator` or build from source)
- Python 3.10+ with `numpy`, `scipy`, `pytest`

### Lint the RTL

```bash
cd hardware
bash sim/scripts/run_verilator.sh lint
```

### Run a single engine testbench

```bash
cd hardware
bash sim/scripts/run_verilator.sh tb_spectral_engine
```

Any of the six testbenches can be run individually:

```bash
bash sim/scripts/run_verilator.sh tb_hd_engine
bash sim/scripts/run_verilator.sh tb_topology_engine
bash sim/scripts/run_verilator.sh tb_graph_rewrite
bash sim/scripts/run_verilator.sh tb_associative_memory
bash sim/scripts/run_verilator.sh tb_system_end_to_end
```

### Run the full simulation suite

```bash
cd hardware
bash sim/scripts/run_verilator.sh all
```

This runs lint followed by all six testbenches and reports a pass/fail summary.

### Run Python golden model comparison

```bash
cd hardware
python sim/models/gold_spectral_model.py
python sim/models/gold_hd_model.py
PYTHONPATH=src:hardware/sw python -m pytest hardware/tests/ -v --tb=short
```

## FPGA Build Instructions

### Target Platform

| Property | Value |
|----------|-------|
| Device | Xilinx Alveo U280 (xcu280-fsvh2892-2L-e) |
| HBM | 8 GB HBM2 (32 stacks x 256 Mb) |
| Clock | 250 MHz core, 300 MHz HBM |
| Tool | Vivado 2023.2+ |

### Synthesis

```bash
cd hardware/fpga/build
make synth BOARD=xcu280
# or directly:
vivado -mode batch -source synth.tcl -tclargs -board xcu280
```

### Implementation

```bash
make impl BOARD=xcu280
vivado -mode batch -source impl.tcl -tclargs -board xcu280 -bit
```

### Resource Utilisation (estimated)

| Resource | Systolic (16x16) | HD Engine | Topology | Total (est.) |
|----------|-------------------|-----------|----------|--------------|
| LUT | ~45 K | ~18 K | ~12 K | ~95 K |
| FF | ~38 K | ~15 K | ~10 K | ~78 K |
| BRAM | 48 | 24 | 16 | 120 |
| DSP | 256 | 64 | 16 | 384 |
| URAM | 32 | 8 | 4 | 48 |

## Python HAL Usage

The Python Hardware Abstraction Layer provides a software-side interface that
mirrors the SMGP Python API but routes operations to the hardware backend.

```python
from hardware.sw.hal.smgpu_hal import SMGPU_HAL

# Connect to the FPGA via PCIe
hal = SMGPU_HAL(device="/dev/smgpu0")

# Load a graph into HBM
graph_id = hal.load_graph(nodes=1024, edges=8192, adj_matrix=csr_data)

# Run spectral analysis
result = hal.execute(
    opcode=hal.OPC_SPECTRAL,
    sub_opcode=hal.SUB_COMPUTE_LAP,
    flags=hal.FLAG_DONE_IRQ,
    operand=graph_id,
)
laplacian = result.read_buffer(1024)

# Run HD similarity query
query_vec = hal.encode_hd("Socrates taught Plato")
sim_result = hal.hd_similarity(query_vec, k=5)
```

### Wiring the Executor to Core Python Modules

Pass an `HWExecutor` to any core module via the `executor` keyword.  Every
class falls back to pure Python when `executor=None`.

```python
from smgp.core.graph import SpectralMemoryGraph
from smgp.core.spectral import SpectralMethods
from smgp.core.hyperdim import HyperdimensionalMemory
from smgp.attention.spectral_attn import SpectralAttention
from smgp.config import SMGPConfig, create_executor_from_config

# Option A – construct executor manually
from hardware.sw.smgp_hal.executor import HWExecutor
from hardware.sw.smgp_hal.hw_session import HWSession

session  = HWSession(device="/dev/smgpu0", fallback=True)
executor = HWExecutor(session=session)

# Pass executor to graph; SpectralMethods/SpectralAttention inherit it
graph    = SpectralMemoryGraph(hd_dim=10000, seed=42, executor=executor)
spectral = SpectralMethods(graph)          # executor inherited
hd       = HyperdimensionalMemory(dim=10000, executor=executor)
attn     = SpectralAttention(graph, hidden_dim=256)  # executor inherited

# Option B – load from SMGP config file
cfg = SMGPConfig.from_dict({
    "hd_dim": 10000,
    "hardware": {"enabled": True, "device": "/dev/smgpu0", "fallback": True},
})
executor = create_executor_from_config(cfg)  # returns None if disabled
graph    = SpectralMemoryGraph(hd_dim=cfg.hd_dim, executor=executor)
```

**Offloaded operations** (silently fall back to Python on `NotImplementedError`):

| Module | Methods offloaded |
|--------|-------------------|
| `SpectralMethods` | `compute_laplacian`, `compute_eigen` |
| `HyperdimensionalMemory` | `bind`, `unbind`, `bundle`, `similarity` |
| `SpectralAttention` | `forward` |

## Testing

### RTL-Level Tests

| Test | Status | Description |
|------|--------|-------------|
| `tb_spectral_engine` | ✅ PASS | Laplacian computation, Chebyshev convolution, wavelet transform |
| `tb_hd_engine` | ✅ PASS | Bundle, bind/unbind, permute, similarity, random generation |
| `tb_topology_engine` | ✅ PASS | Filtration, Union-Find, barcode emission, Wasserstein |
| `tb_graph_rewrite` | ✅ PASS | DPO pattern matching, rule application, proof trace |
| `tb_associative_memory` | ✅ PASS | CAM read/write, HD similarity threshold, associative query |
| `tb_system_end_to_end` | ✅ PASS | Full system pipeline — config write, SPECTRAL + HD instructions via AXI-Lite |

Verified with **Verilator 5.048** on macOS (Darwin/x86_64). All 6/6 testbenches pass.

```bash
cd hardware
bash sim/scripts/run_verilator.sh all
# Results: 6/6 passed — All tests PASSED
```

### Python-Level Tests

```bash
# Golden model validation
python -m pytest hardware/tests/test_golden_models.py -v

# HAL integration
PYTHONPATH=src:hardware/sw python -m pytest hardware/tests/test_hw_integration.py -v

# Memory mapper
PYTHONPATH=src:hardware/sw python -m pytest hardware/tests/test_memory_mapper.py -v
```

## ISA Overview

All SMGPU instructions are 32-bit fixed-width:

```
 [31:28] opcode      — Operation class (NOP, GRAPH_CTOR, SPECTRAL, HD, …)
 [27:24] sub_opcode  — Sub-operation within the class
 [23:16] flags       — Modifier flags (START, DONE_IRQ, CHAINED, STREAMING, …)
 [15:0]  operand     — Address, immediate, or register select
```

There are 8 primary opcode classes covering graph construction, spectral
transforms, hyperdimensional operations, topology, graph rewriting, memory
management, and system control. See [`docs_hw/isa_reference.md`](docs_hw/isa_reference.md)
for the complete encoding tables.

## Performance Targets

| Operation | Latency (FPGA @ 250 MHz) | Throughput |
|-----------|--------------------------|------------|
| Laplacian (4K nodes) | ~82 us | 49 M edges/s |
| Eigen-decomp (K=64) | ~5.2 ms | 12.3 K iterations/s |
| Chebyshev conv (K=4) | ~0.33 ms | 12.2 M nodes/s |
| HD bind/unbind (10K-dim) | 1 cycle | 250 M ops/s |
| HD similarity (10K-dim) | 313 cycles | 0.8 M queries/s |
| Topology barcode (4K nodes) | ~1.4 ms | 2.9 K graphs/s |
| Wasserstein distance | ~0.7 ms | 1.4 K comparisons/s |
| DPO match (pattern 4 nodes) | ~32 us | 31 K patterns/s |

## References

- Chung, F.R.K. (1997). *Spectral Graph Theory.* CBMS, AMS.
- Defferrard, M., Bresson, X., Vandergheynst, P. (2016). "Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering." *NeurIPS*.
- Kanerva, P. (2009). "Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors." *Cognitive Computation*, 1(2), 139–159.
- Edelsbrunner, H., Letscher, D., Zomorodian, A. (2002). "Topological Persistence and Simplification." *DCG*, 28(4), 511–533.
- Ehrig, H., Ehrig, K., Prange, U., Taentzer, G. (2006). *Fundamentals of Algebraic Graph Transformation.* Springer.
- Kung, H.T. (1982). "Why Systolic Architectures?" *IEEE Computer*, 15(1), 37–46.
- Dally, W.J., Towles, B.P. (2004). *Principles and Practices of Interconnection Networks.* Morgan Kaufmann.
- Ielmini, D., Wong, H.-S.P. (2018). "In-Memory Computing with Resistive Switching Devices." *Nature Electronics*, 1(6), 333–343.
- Hennessy, J.L., Patterson, D.A. (2019). *Computer Architecture: A Quantitative Approach.* 6th Ed., Morgan Kaufmann.
- Parhi, K.K. (1999). *VLSI Digital Signal Processing Systems.* John Wiley & Sons.

## License

Licensed under the Apache License 2.0. See the root [LICENSE](../LICENSE) file for details.
