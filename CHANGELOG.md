# Changelog

All notable changes to the Spectral Memory Graph Processor (SMGP) will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-05-19

### Fixed

- **BUG** GFT → IGFT roundtrip error (≈0.28 max error) — `compute_eigen()` capped `k`
  at `n - 1`, preventing a complete orthonormal basis.  When `num_eigenvalues ≥ n`,
  `np.linalg.eigh` is now used directly (full dense decomposition, complete basis,
  `U @ U^T = I`).  For partial bases via `eigsh`, eigenvectors are re-orthonormalized
  with `np.linalg.qr`.  Laplacian symmetry is now enforced with `(L + L^T) / 2`
  to eliminate floating-point drift.  Roundtrip error is now < 1e-10 for full basis.
- **BUG** `smgp.enhanced.speed.__init__` was empty — `fast_eigen_decompose`,
  `fast_graph_traversal`, `fast_hd_bind`, and `fast_hd_similarity` were defined
  in `accelerated.py` but not re-exported.  Added explicit imports and `__all__`
  so `from smgp.enhanced.speed import fast_eigen_decompose` works.
- **DEP** `pyyaml` was not declared as a dependency despite `SMGPConfig.from_yaml()`
  requiring it.  Added `pyyaml>=6.0` to base `dependencies` in `pyproject.toml`
  and `setup.cfg`.

### Bumped

- Version `0.3.0` → `1.0.0` reflecting stable public API.

## [0.3.0] - 2026-05-18

### Fixed

- **CRITICAL** `smgp server` crash (`FastAPIError`) — all four API request classes
  (`AddKnowledgeRequest`, `QueryRequest`, `ReasonRequest`, `PruneRequest`) were
  plain Python classes and not Pydantic `BaseModel` subclasses, breaking FastAPI
  0.136+ / Pydantic v2.  Replaced with proper Pydantic models built lazily inside
  `_make_models()` so the module remains importable without `pydantic`.
- **HIGH** `smgp.__version__` returned `"0.1.0"`; bumped to match PyPI metadata.
  Updated hardcoded assertion in `tests_enhanced/test_cli.py` accordingly.
- **HIGH** `setup.cfg` version was `0.1.0` and `long_description` still pointed to
  `README.md`; synced to current version and switched to `PACKAGE.md`.  Synced all
  extras with `pyproject.toml` (added `auth`, `vectordb`, `onnx`, `hw`, full `dev` set).
- **MEDIUM** `click` was an undeclared runtime dependency: `smgp` entry-point always
  requires it, but bare `pip install smgp` did not pull it in.  Added `click>=8.0`
  to base `dependencies` in both `pyproject.toml` and `setup.cfg`.
- **MEDIUM** `huggingface` and `onnx` extras pinned `numpy>=1.24.0,<2.0`, causing
  an irresolvable conflict with numpy 2.x environments.  Removed the `<2.0` upper
  bound; modern `transformers` and `onnx` both support numpy 2.x.
- **LOW** `smgp --version` raised an error; added `@click.version_option()` to the
  root CLI group so `smgp --version` works alongside the existing `smgp version`
  sub-command.
- **LOW** `smgp.__all__` exported only `SMGPConfig`; added `SpectralMemoryGraph`
  and `HyperdimensionalMemory` so `from smgp import *` works for common usage.
- **LOW** Documentation URL in `pyproject.toml` and `PACKAGE.md` pointed to
  `smgp.readthedocs.io` (empty page); replaced with
  `https://github.com/rotsl/smgp#readme`.
- **LOW** Ruff UP045: `Optional[X]` annotations in `api.py` replaced with `X | None`;
  unused `Optional` import removed.

## [0.2.0] - 2026-05-17

### Added — Vivado Tcl script to convert a routed `.dcp` checkpoint into an Alveo U280-compatible `.xclbin` file
- **Hardware**: `hardware/fpga/build/scripts/write_bitstream.tcl` — Vivado Tcl script to write a `.bit` bitstream from a routed checkpoint
- **Hardware**: `hardware/fpga/build/Makefile` — new `xclbin` and `bit_script` targets for local bitstream generation
- **Core**: `src/smgp/hw_bridge.py` — `HardwareBridge` duck-typed adapter; transparently routes operations to any hardware executor and falls back to pure Python on `NotImplementedError`
- **Config**: `HardwareConfig` dataclass in `src/smgp/config.py`; `hardware` field on `SMGPConfig`; `from_dict()` sub-dict handling; `create_executor_from_config()` helper
- **Core**: `SpectralMemoryGraph` now accepts optional `executor` parameter; stores `self.executor` and `self.hw_bridge`
- **Core**: `SpectralMethods` now accepts optional `executor` parameter (inherited from graph); offloads `compute_laplacian` and `compute_eigen` to hardware when available
- **Core**: `HyperdimensionalMemory` now accepts optional `executor` parameter; offloads `bind`, `unbind`, `bundle`, and `similarity` to hardware when available
- **Attention**: `SpectralAttention` now accepts optional `executor` parameter (inherited from graph); offloads `forward` to hardware when available
- **Tests**: `tests/test_hardware_integration_wired.py` — 20 mock-based integration tests covering all wired modules and config; pass without the hardware HAL installed
- **Package**: `MANIFEST.in` — controls sdist contents; excludes `RESEARCH.md` and `ROADMAP.md`
- **Package**: `pyproject.toml` — bumped to `0.2.0`; added `hw` extra; added `topology` to CI install extras; added `pytest-anyio`/`anyio` to `dev`; added `py.typed` marker; `testpaths` now includes `tests_enhanced`; author updated to `rotsl`
- **CI**: `.github/workflows/tests.yml` — added `topology` extra to CI install; upgraded `codecov-action` to v4 with token support; added `publish` job for PyPI releases on version tags
- **Docs**: README badges updated — removed stale `hw_ci.yml` badge; added `Tests 230 passed`, `RTL sim 6/6 pass`, and `Verilator 5.048` static badges

### Changed

- All executor-accepting classes use a `try/except ImportError` guard around `hw_bridge` import, keeping the package fully importable without the `hardware/` directory on the Python path
- `from_dict` on `SMGPConfig` now handles a nested `hardware` dict, converting it to `HardwareConfig` automatically

### Fixed

- `src/smgp/config.py` — removed stale `Optional` / `TYPE_CHECKING` imports; replaced quoted forward-ref in `from_dict` with bare class name; `create_executor_from_config` return type now uses `X | None` union syntax
- `src/smgp/core/graph.py` — removed unused `TYPE_CHECKING` import (ruff `F401`)
- `tests/test_hardware_integration_wired.py` — fixed ruff `I001` import ordering (stdlib `unittest.mock` sorted before third-party)



### Added

- Initial release of SMGP
- **Core**: `SpectralMemoryGraph` with HD vector addressing
- **Core**: `SpectralMethods` — Laplacian eigendecomposition, Graph Fourier Transform (GFT), Chebyshev convolution, spectral wavelets
- **Core**: `HyperdimensionalMemory` — 10,000-dimensional bipolar vector representations
- **Core**: `TopologicalAnalyzer` — persistent homology via Betti numbers and barcode computation
- **Core**: `GraphRewriter` — Direct Preference Optimization (DPO) based graph structure rewriting
- **Core**: `CategoryTheoryOps` — categorical constructs for graph composition
- **Memory**: `MemoryStore` — persistent key-value memory backend
- **Memory**: `AssociativeMemory` — content-addressable retrieval using HD similarity
- **Memory**: `MemoryLifecycle` — birth, decay, and eviction management for memory entries
- **Attention**: `SpectralAttention` — O(N log N) multiscale attention via spectral decomposition
- **Attention**: `StreamingSpectralAttention` — incremental streaming variant
- **Reasoning**: `ClaimVerifier` — fact-checking against the memory graph with explainable proof subgraphs
- **Reasoning**: `NeuroSymbolicPlanner` — goal decomposition with symbolic constraint solving
- **Reasoning**: `ExplainableClaimVerifier` / `ExplainableNeuroSymbolicPlanner` — attribution-traced reasoning
- **Integration**: HuggingFace transformers adapter
- **Integration**: LangChain memory and retrieval integration
- **Integration**: FastAPI REST API with JWT auth, API-key auth, and global rate limiting middleware
- **Integration**: `VectorDBSyncer` — bidirectional sync with Qdrant, Pinecone, Weaviate
- **Integration**: `GraphFederator` — multi-node federation via SQLAlchemy-backed SQLite/Postgres
- **Integration**: `SMGPAttentionWrapper` — ONNX-exportable attention wrapper for vLLM integration
- **Integration**: `DistributedGraphBackend` — sharded in-memory distributed graph
- **Memory**: `EventLog` + `EventSourcedGraph` — JSONL and SQLite event sourcing
- **Memory**: `EnhancedMemoryLifecycle` + pruning policies (LRU, time-based, access-count)
- **Utils**: `HyperparameterTuner` — grid-search and random-search with save/restore
- **Utils**: `MultimodalEmbedder` — image and text node embeddings
- **CLI**: `smgp` entry-point with `version`, `graph-stats`, `verify`, and `server` sub-commands
- **Hardware**: SMGPU FPGA accelerator — complete SystemVerilog RTL
- **Hardware**: Python HAL (`smgp_hal`) for host-side FPGA interaction
- **Hardware**: Simulation testbenches (Verilator and Cocotb)
- **Hardware**: FPGA build scripts (Vivado Makefile and TCL project generator)
- **Hardware**: C driver library for PCIe communication
- **Tests**: 214 passing tests across 26 modules (210 without PyTorch)
- **Tests**: Property-based fuzz testing via Hypothesis for HD operations
- **CI/CD**: GitHub Actions workflow for Python tests and PyPI publishing
- **Documentation**: Quickstart guide, API reference, and hardware integration guide
- **Optional extras**: `auth`, `vectordb`, `onnx` extras added to `pyproject.toml`

### Fixed

- `tests_enhanced/test_hd_hypothesis.py` — `HealthCheck` module-level reference guarded behind `HAS_HYPOTHESIS` flag
- `src/smgp/integration/api_auth.py` — `/protected` endpoint now uses FastAPI `Depends()` in route signature instead of manually calling the dependency (which bypassed DI and raised `AttributeError`)
- `src/smgp/integration/api_auth.py` — Rate limiter replaced with a global per-IP sliding-window ASGI middleware; previous slowapi-based approach applied limits per-route rather than globally
- `src/smgp/integration/vectordb.py` — `client.search()` updated to `client.query_points().points` for qdrant-client ≥ 1.7 compatibility
- `src/smgp/integration/onnx_vllm.py` — `export_onnx` now traces with a real dummy mask tensor so `torch.onnx.export` sees both inputs and `input_names` length matches
