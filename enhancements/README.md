# SMGP Enhancements

This directory contains additive enhancement patches for the SMGP project. These patches extend the base v0.1.0 release with new features, dependencies, and tooling **without modifying any existing source files**.

## Applying the Patches

To apply the pyproject.toml patch (Enhancement 25):

```bash
# From the project root
bash enhancements/apply_patch.sh
```

Or manually:

```bash
patch -p1 < enhancements/pyproject.patch
```

After applying, reinstall the package:

```bash
pip install -e .
```

## Enhancement Overview

### Enhancement 23: Documentation & Community Files

Adds project governance and community files:

| File | Description |
|---|---|
| `ROADMAP.md` | Project roadmap with versioned goals (v0.2.0 → v1.0.0) and hardware plans |
| `CHANGELOG.md` | Structured changelog following [Keep a Changelog](https://keepachangelog.com/) format |
| `CODE_OF_CONDUCT.md` | Contributor Covenant Code of Conduct v2.1 |
| `.github/ISSUE_TEMPLATE/bug_report.md` | Bug report issue template |
| `.github/ISSUE_TEMPLATE/feature_request.md` | Feature request issue template |
| `.github/PULL_REQUEST_TEMPLATE.md` | Pull request checklist template |

### Enhancement 24: Research Paper Companion

| File | Description |
|---|---|
| `PAPER.md` | Companion paper titled *"SMGP: Spectral Memory Graph Processing for Persistent, Verifiable AI Reasoning"* with abstract, methodology, experimental results, and references |

### Enhancement 25: PyPI Publishing Patch

| File | Description |
|---|---|
| `pyproject.patch` | Unified diff patch for `pyproject.toml` adding optional dependency groups, CLI entry point, and enhanced test paths |
| `enhancements/apply_patch.sh` | Automated patch application script |

## Installing Optional Extras

The patch adds the following installable extras:

```bash
# Install everything (includes all enhanced extras)
pip install smgp[all]

# Install all enhanced features
pip install smgp[enhanced]

# Individual extras
pip install smgp[speed]          # Cython-accelerated operations
pip install smgp[pruning]        # Memory pruning policies (pure Python)
pip install smgp[streaming]      # Streaming attention (pure Python)
pip install smgp[federation]     # Multi-node federation (SQLAlchemy)
pip install smgp[multimodal]     # Image/audio support (Pillow, torch, transformers)
pip install smgp[tuner]          # Hyperparameter tuning (pure Python)
pip install smgp[explainable]    # Attention attribution (pure Python)
pip install smgp[onnx]           # ONNX/vLLM export (torch, onnx, vllm)
pip install smgp[vectordb]       # Vector DB sync (Qdrant, Pinecone, Weaviate)
pip install smgp[api-auth]       # API authentication (python-jose, passlib, slowapi)
pip install smgp[distributed]    # Distributed graph (pure Python)
pip install smgp[eventlog]       # Event logging (pure Python)
pip install smgp[cli]            # CLI interface (click)
pip install smgp[hypothesis]     # Property-based testing (hypothesis)
pip install smgp[hardware-cycle] # Hardware cycle-accurate sim (C++ build required)
pip install smgp[pynq]           # PYNQ FPGA support (pynq)
```

## Running Tests

After applying the patch, the test runner will discover both the original and enhanced test suites:

```bash
# Run all tests
pytest

# Run only original tests
pytest tests/

# Run only enhanced tests
pytest tests_enhanced/

# Run with coverage
pytest --cov=smgp
```

## Reverting

To revert the pyproject.toml patch:

```bash
patch -R -p1 < enhancements/pyproject.patch
```
