# SMGP Roadmap

**Repository:** [https://github.com/rotsl/smgp](https://github.com/rotsl/smgp)

## Current Status — v0.2.0

SMGP (Spectral Memory Graph Processor) is in alpha. The core architecture is stable: hyperdimensional vector addressing, spectral graph methods (Laplacian decomposition, Graph Fourier Transform, Chebyshev convolution, wavelets), topological persistence analysis, DPO-based graph rewriting, and a complete Python HAL for the SMGPU FPGA accelerator (SystemVerilog RTL, Verilator-verified testbenches, Vivado FPGA build scripts including XCLBIN generation for Alveo U280).

v0.2.0 adds **hardware executor wiring** — every core module (`SpectralMemoryGraph`, `SpectralMethods`, `HyperdimensionalMemory`, `SpectralAttention`) now accepts an optional `executor` parameter that transparently offloads computation to the FPGA or cycle-accurate simulator, with silent fallback to pure Python.

The project ships with **230 passing tests** across 27 modules (235 collected, 5 skipped torch-dependent), a CI/CD pipeline covering Python 3.10–3.12 on Linux and macOS, and integrations with HuggingFace, LangChain, FastAPI, Qdrant, SQLAlchemy, and ONNX/vLLM.

---

## ✅ Completed — v0.1.0 / v0.2.0

| Feature | Version | Description |
|---------|---------|-------------|
| **Core graph engine** | v0.1.0 | `SpectralMemoryGraph` with HD addressing, spectral methods, topological analysis, DPO rewriting |
| **Hyperdimensional memory** | v0.1.0 | 10,000-dim bipolar vectors, bind/unbind/bundle/similarity |
| **SpectralAttention** | v0.1.0 | O(N log N) multiscale spectral attention + streaming variant |
| **ClaimVerifier / Planner** | v0.1.0 | Path-based fact verification; DPO goal decomposition |
| **Explainability** | v0.1.0 | Attribution tracing, `ExplainableClaimVerifier`, `ExplainableNeuroSymbolicPlanner` |
| **Memory lifecycle** | v0.1.0 | LRU/time/access-count pruning, event sourcing (JSONL + SQLite) |
| **Integration suite** | v0.1.0 | HuggingFace, LangChain, FastAPI + auth, Qdrant/Pinecone/Weaviate, ONNX/vLLM, federation |
| **CLI** | v0.1.0 | `smgp version/graph-stats/verify/server` via `click` |
| **Property testing** | v0.1.0 | Hypothesis fuzz tests for HD and graph operations |
| **SMGPU RTL** | v0.1.0 | SystemVerilog 2017 accelerator — spectral, HD, topology, graph-rewrite engines |
| **Verilator testbenches** | v0.1.0 | 6/6 TBs verified (spectral, HD, topology, graph-rewrite, associative memory, system e2e) |
| **Python HAL** | v0.1.0 | `HWExecutor`, `HWSession`, `MemoryMapper`, cycle simulator |
| **FPGA build** | v0.1.0 | Vivado Makefile: `synth`, `impl` targets |
| **Hardware executor wiring** | v0.2.0 | `executor=` param on all core modules; `HardwareBridge` adapter; graceful fallback |
| **XCLBIN / bitstream targets** | v0.2.0 | `make xclbin` (Alveo U280) and `make bit_script` targets + TCL scripts |
| **HardwareConfig** | v0.2.0 | `HardwareConfig` dataclass; `create_executor_from_config()` helper |
| **PACKAGE.md** | v0.2.0 | PyPI-specific README (pip-install focused, no clone instructions) |
| **MANIFEST.in** | v0.2.0 | Controlled sdist — excludes RESEARCH.md, ROADMAP.md, PAPER.md |

---

## Short-term Goals — v0.3.0

| Feature | Description | Status |
|---------|-------------|--------|
| **Persistent Graph Storage** | Disk-backed graph snapshots with incremental delta compression and WAL for crash recovery. | 🔄 Planned |
| **Real-time Streaming API** | SSE and WebSocket endpoints for streaming graph updates to clients. | 🔄 Planned |
| **LLM Prompt Cache** | Use the HD memory graph as a semantic KV cache for LLM inference — deduplicate and reuse activations across similar prompts. | 🔄 Planned |
| **Async Graph Workers** | Background worker pool for deferred spectral recomputation, pruning, and federation sync. | 🔄 Planned |
| **OpenTelemetry Tracing** | Distributed tracing spans for every graph operation, exportable to Jaeger / Prometheus. | 🔄 Planned |
| **Multimodal Extensions** | Image and audio node embeddings enabling cross-modal reasoning. | 🔄 Planned |
| **Windows CI** | Native Windows runner in GitHub Actions. | 🔄 Blocked on qdrant-client wheels |

---

## Medium-term Goals — v0.4.0

| Feature | Description | Status |
|---------|-------------|--------|
| **Distributed Graph Engine** | Partitioned graph storage with sharding beyond single-machine memory; consensus-based conflict resolution. | 🔄 Planned |
| **Formal Verification** | Mathematically proven correctness guarantees for core spectral and topological operations. | 🔬 Exploratory |
| **PYNQ Overlay** | Pre-built Jupyter-friendly PYNQ overlay for Kria KV260 rapid prototyping. | 🔄 Planned |
| **Alveo Vitis Shell** | Package spectral and HD engines as Vitis kernels for data-centre deployment. | 🔄 Planned |

---

## Long-term Goals — v1.0.0

- **Production Hardening**: Comprehensive observability (OpenTelemetry), graceful degradation, multi-tenant isolation, hardened API authentication.
- **Extended Hardware Support**: Port SMGPU to Xilinx Alveo U200/U250 and Intel Agilex 7; initiate ASIC design flow for volume deployment.
- **Ecosystem Integrations**: Native plugins for AutoGen, CrewAI, Neo4j, ArangoDB, and cloud providers (AWS, GCP, Azure).
- **Performance Targets**: Sub-millisecond spectral attention on 100K-node graphs; 10M+ edge graphs on a single FPGA; linear scaling across federated nodes.

---

## Hardware Roadmap

| Platform | Target | Status |
|----------|--------|--------|
| **Xilinx Zynq-7000** (ZedBoard) | Initial development & validation | ✅ Supported |
| **Xilinx Alveo U280** | Data-centre acceleration — XCLBIN target live | ✅ Build target shipped (v0.2.0) |
| **Xilinx Kria KV260** | Edge inference with PYNQ overlay | 🔄 Planned |
| **Xilinx Alveo U200/U250** | Broader data-centre support | 🔄 Planned |
| **Intel Agilex 7** | Cross-vendor FPGA | 🔄 Planned |
| **ASIC Flow (TSMC 7nm)** | Volume production path | 🔬 Exploratory |

### Hardware Milestones

1. ✅ **Verilator RTL verification** — all 6 testbenches pass (v0.1.0)
2. ✅ **Python HAL + executor wiring** — `HWExecutor` integrates with all core modules (v0.2.0)
3. ✅ **XCLBIN build target** — `make xclbin` for Alveo U280 (v0.2.0)
4. 🔄 **PYNQ Overlay** — Jupyter-friendly overlay for Kria KV260
5. 🔄 **Alveo Vitis Shell** — Vitis kernel packaging for data-centre deployment
6. 🔬 **ASIC Feasibility Study** — area, power, and frequency trade-offs for a dedicated inference chip

---

## Contributing

Contributions welcome at every level — bug fixes, new spectral methods, hardware designs, documentation improvements, and more. Please see [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) for guidelines.
