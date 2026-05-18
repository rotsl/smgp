# SMGP — Spectral Memory Graph Processor

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![SystemVerilog](https://img.shields.io/badge/RTL-SystemVerilog_2017-orange.svg)](https://standards.ieee.org/ieee/1800/6603/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![CI Tests](https://img.shields.io/github/actions/workflow/status/rotsl/smgp/tests.yml?branch=main&label=CI&logo=github&logoColor=white)](https://github.com/rotsl/smgp/actions/workflows/tests.yml)
[![Tests](https://img.shields.io/badge/tests-230%20passed-brightgreen.svg)](https://github.com/rotsl/smgp/actions)
[![RTL Sim](https://img.shields.io/badge/RTL_sim-6%2F6_pass-brightgreen.svg)](https://github.com/rotsl/smgp/tree/main/hardware)
[![PyPI](https://img.shields.io/badge/PyPI-v1.0.0-blue)](https://pypi.org/project/smgp/)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/smgp?style=flat-square&logo=appveyor&labelColor=black)](https://pypi.org/project/smgp/)
[![Code style: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff)

**Persistent, hallucination-free AI reasoning through spectral graph theory, hyperdimensional computing, and dedicated hardware acceleration.**

SMGP combines spectral graph theory, topological data analysis (TDA), hyperdimensional computing (HDC), and category-theoretic graph rewriting to achieve persistent memory, sublinear context processing, and verifiable reasoning. The framework ships with a complete Python software stack *and* an open-source hardware accelerator design (SMGPU) in SystemVerilog, enabling transparent offloading of compute-intensive kernels to an FPGA.

**[GitHub](https://github.com/rotsl/smgp) · [Documentation](https://github.com/rotsl/smgp#readme) · [Issues](https://github.com/rotsl/smgp/issues)**

---

## Installation

```bash
pip install smgp
```

With optional extras:

```bash
# Topological analysis (requires gudhi)
pip install "smgp[topology]"

# REST API + JWT auth
pip install "smgp[integration,auth]"

# HuggingFace / ONNX
pip install "smgp[huggingface,onnx]"

# Vector database sync (Qdrant)
pip install "smgp[vectordb]"

# Everything
pip install "smgp[all]"
```

---

## Quick Start

```python
from smgp.core.graph import SpectralMemoryGraph

# Build a knowledge graph with 10,000-dimensional HD addressing
graph = SpectralMemoryGraph(hd_dim=10000, seed=42)
graph.add_node("Socrates", label="person", properties={"era": "ancient Greece"})
graph.add_node("Plato",    label="person", properties={"era": "ancient Greece"})
graph.add_edge("Socrates", "Plato", "taught")

# Verify claims against the graph
from smgp.reasoning.verifier import ClaimVerifier
verifier = ClaimVerifier(graph)
result = verifier.verify("Socrates taught Plato")
print(result["verified"], result["reasoning"])
```

### Spectral Analysis

```python
from smgp.core.spectral import SpectralMethods

spectral = SpectralMethods(graph, num_eigenvalues=8)
L        = spectral.compute_laplacian()
evals, evecs = spectral.compute_eigen()
signal   = spectral.graph_fourier_transform(node_signal, evecs)
```

### Hyperdimensional Memory

```python
from smgp.core.hyperdim import HyperdimensionalMemory

hd = HyperdimensionalMemory(dim=10000, seed=42)
v1, v2 = hd.random_vectors(2)
bound  = hd.bind(v1, v2)
print(hd.similarity(hd.unbind(bound, v2), v1))  # ≈ 1.0
```

### Hardware-Accelerated (FPGA)

```python
from hardware.sw.smgp_hal.executor import HWExecutor
from hardware.sw.smgp_hal.hw_session import HWSession

session  = HWSession(device="/dev/smgpu0", fallback=True)
executor = HWExecutor(session=session)

# Drop executor into any core class — pure Python fallback when executor=None
graph    = SpectralMemoryGraph(hd_dim=10000, seed=42, executor=executor)
spectral = SpectralMethods(graph)          # inherits executor from graph
hd       = HyperdimensionalMemory(dim=10000, executor=executor)
```

Or via config:

```python
from smgp.config import SMGPConfig, create_executor_from_config

cfg      = SMGPConfig.from_dict({"hd_dim": 10000,
           "hardware": {"enabled": True, "device": "/dev/smgpu0"}})
executor = create_executor_from_config(cfg)
graph    = SpectralMemoryGraph(hd_dim=cfg.hd_dim, executor=executor)
```

---

## Features

| Category | Component | Description |
|----------|-----------|-------------|
| **Core** | `SpectralMemoryGraph` | Typed property graph with HD vector addressing |
| **Core** | `SpectralMethods` | Laplacian, GFT, Chebyshev convolution, wavelets |
| **Core** | `HyperdimensionalMemory` | 10,000-dim bipolar vectors, bind/bundle/similarity |
| **Core** | `TopologicalAnalyzer` | Persistent homology, Betti numbers, barcode |
| **Core** | `GraphRewriter` | Category-theoretic DPO graph rewriting |
| **Memory** | `MemoryStore` | Key-value persistence backend |
| **Memory** | `AssociativeMemory` | O(1) content-addressable HD recall |
| **Memory** | `MemoryLifecycle` | Persistence-based forgetting & eviction |
| **Attention** | `SpectralAttention` | O(N log N) multiscale spectral attention |
| **Attention** | `StreamingSpectralAttention` | Incremental online attention |
| **Reasoning** | `ClaimVerifier` | Path-based fact verification with proof subgraphs |
| **Reasoning** | `NeuroSymbolicPlanner` | Goal decomposition + symbolic constraint solving |
| **Integration** | HuggingFace | `SMGPForCausalLM` drop-in adapter |
| **Integration** | LangChain | `SMGPMemory`, `SMGPVerifierTool` |
| **Integration** | REST API | FastAPI server with JWT auth + rate limiting |
| **Integration** | Vector DB | Qdrant / Pinecone / Weaviate bidirectional sync |
| **Integration** | ONNX/vLLM | Export-ready attention wrapper |
| **Hardware** | SMGPU RTL | Open SystemVerilog 2017 FPGA accelerator |
| **Hardware** | Python HAL | `HWExecutor` — transparent FPGA offload |
| **Hardware** | Bitstream | `make xclbin` target for Alveo U280 |
| **CLI** | `smgp` | `version`, `graph-stats`, `verify`, `server` |

---

## Configuration

```python
from smgp.config import SMGPConfig

cfg = SMGPConfig(
    hd_dim=10000,
    hidden_dim=256,
    num_heads=8,
    num_scales=3,
    max_eigenvalues=64,
    memory_capacity=100000,
    pruning_threshold=0.1,
    seed=42,
)

# Load from file
cfg = SMGPConfig.from_yaml("config.yaml")
cfg = SMGPConfig.from_dict({"hd_dim": 5000, "hardware": {"enabled": False}})
```

---

## Testing

```bash
pip install "smgp[dev,topology]"
python -m pytest tests/ tests_enhanced/ -v
```

| Suite | Tests | Notes |
|-------|-------|-------|
| `tests/` | 112 | Core, spectral, HD, reasoning, integration, hardware wiring |
| `tests_enhanced/` | 123 | CLI, distributed, event log, ONNX, streaming, vector DB, Hypothesis |
| **Total** | **235** | **230 passed / 5 skipped** (torch-dependent, no PyTorch required) |

---

## Hardware Acceleration

SMGPU is a domain-specific accelerator targeting Xilinx Alveo U280 (250 MHz, 8 GB HBM2).

All 6 RTL testbenches verified with **Verilator 5.048**:

| Testbench | Status |
|-----------|--------|
| `tb_spectral_engine` | ✅ PASS |
| `tb_hd_engine` | ✅ PASS |
| `tb_topology_engine` | ✅ PASS |
| `tb_graph_rewrite` | ✅ PASS |
| `tb_associative_memory` | ✅ PASS |
| `tb_system_end_to_end` | ✅ PASS |

Build targets:

```bash
cd hardware/fpga/build
make synth        # Vivado synthesis
make impl         # Place & route
make xclbin       # Generate Alveo U280 .xclbin
make bit_script   # Generate .bit bitstream
```

---

## Links

- **Source**: [github.com/rotsl/smgp](https://github.com/rotsl/smgp)
- **Documentation**: [github.com/rotsl/smgp#readme](https://github.com/rotsl/smgp#readme)
- **Issues**: [github.com/rotsl/smgp/issues](https://github.com/rotsl/smgp/issues)
- **License**: Apache 2.0
