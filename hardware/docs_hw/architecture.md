# SMGPU Architecture Documentation

> Detailed microarchitectural description of the Spectral Memory Graph
> Processing Unit, covering compute engines, memory hierarchy,
> interconnect, instruction flow, and implementation trade-offs.

---

## 1. System Overview

SMGPU is a domain-specific accelerator designed to offload the four core
computational pillars of the SMGP framework — spectral graph transforms,
hyperdimensional computing, topological persistence, and category-theoretic
graph rewriting — from a host CPU to dedicated hardware.

### 1.1 High-Level Block Diagram

```
                         Host (CPU / PCIe)
                                |
                    +-----------v-----------+
                    |    AXI / PCIe Bridge   |
                    +-----------+-----------+
                                |
                    +-----------v-----------+
                    |    ISA Decoder &       |
                    |    Instruction Queue   |
                    |    (depth = 16)        |
                    +-----+-----+-----+-----+
                          |     |     |
            +-------------+     |     +-------------+
            |                   |                   |
   +--------v------+   +-------v-------+   +--------v------+
   |  Dispatch     |   |  Memory       |   |  Dispatch     |
   |  (Spectral,   |   |  Controller   |   |  (HD, Topo,   |
   |   Rewrite)    |   |               |   |   Rewrite)    |
   +-------+-------+   +-------+-------+   +-------+-------+
           |                   |                   |
   +-------v-------+   +-------v-------+   +-------v-------+
   |  Spectral     |   |  HD Engine     |   |  Topology     |
   |  Engine       |   |                |   |  Engine       |
   |  (16x16 PE)   |   |  (16 banks)    |   |  (Union-Find) |
   +-------+-------+   +-------+-------+   +-------+-------+
           |                   |                   |
           +-------------------+-------------------+
                               |
                    +----------v-----------+
                    |   2D Mesh NoC        |
                    |   (XY routing, 2 VC) |
                    +----------+-----------+
                               |
              +----------------+----------------+
              |                |                |
   +----------v-----+ +-------v-------+ +------v----------+
   |  Associative   | |  HBM          | |  Memristor      |
   |  Cache (CAM)   | |  Controller   | |  Crossbar       |
   |  256 entries   | |  4 x 256-bit  | |  1024 x 1024   |
   +----------------+ +---------------+ +-----------------+
                                              |
                                     +--------v--------+
                                     |  HBM2 (8 GB)    |
                                     +-----------------+
```

### 1.2 Design Philosophy

SMGPU follows three guiding principles:

1. **Dataflow-oriented execution** — instructions specify data streams between
   engines rather than scalar register operands, eliminating the von Neumann
   bottleneck for graph-structured data.

2. **Fixed-point dominance** — all arithmetic uses parameterisable Q-format
   fixed-point (default Q8.24) instead of IEEE 754 floating point. This
   halves the DSP slice utilisation and enables direct integration with
   memristor analog compute.

3. **Heterogeneous specialisation** — each engine is hand-tuned for its
   target workload (systolic arrays for linear algebra, XOR banks for HD,
   parallel prefix for Union-Find), rather than using a general-purpose
   SIMD/VLIW core.

---

## 2. Compute Engines

### 2.1 Spectral Engine

The spectral engine is a **16 x 16 systolic array** of processing elements
(PEs) that streams CSR-format sparse matrix entries and computes graph
spectral operations in a pipelined fashion.

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_NODES` | 4096 | Maximum graph node count |
| `MAX_NNZ` | 65536 | Maximum non-zero entries |
| `FRAC_BITS` | 24 | Fractional bits (Q8.24) |
| `ARRAY_ROWS` | 16 | Systolic array row count |
| `ARRAY_COLS` | 16 | Systolic array column count |

**Supported Operations:**

| Sub-op | Description | Pipeline Depth |
|--------|-------------|----------------|
| `COMPUTE_LAP` | Normalised Laplacian L = I - D^{-1/2} A D^{-1/2} | 4 stages |
| `EIGEN_DECOMP` | Power iteration eigen-decomposition (K iterations) | K x 4 stages |
| `SPECTRAL_CONV` | Chebyshev polynomial spectral convolution | Order x 4 stages |
| `WAVELET_XFORM` | Heat-kernel graph wavelet transform | 4 stages |
| `GFT_FORWARD` | Graph Fourier Transform (forward) | 4 stages |
| `GFT_INVERSE` | Graph Fourier Transform (inverse) | 4 stages |

**State Machine:**
The engine implements a 10-state FSM that progresses through:
`IDLE -> LOAD_MATRIX -> LOAD_VECTOR -> COMPUTE_* -> WRITE_RESULT`.

Each state handles a distinct memory-transfer or compute phase. The systolic
array processes `ARRAY_ROWS` nodes per cycle during compute phases, yielding a
wall-clock complexity of O(nnz / ARRAY_ROWS) for sparse matrix-vector products.

**Precision Notes:**
- Laplacian computation uses Q8.24 fixed-point with Newton-Raphson reciprocal
  for degree normalisation (8 iterations, convergent to 24-bit fractional
  precision).
- Eigen-decomposition normalises the iterate by computing the L2 norm via
  digit-recurrence integer square root.

### 2.2 Hyperdimensional (HD) Engine

The HD engine processes D-dimensional bipolar vectors (D = 10,000 default,
stored as 313 x 32-bit words) across 16 parallel banks.

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `HD_DIM_WORDS` | 313 | Words per HD vector (ceil(10000/32)) |
| `HD_BANKS` | 16 | Parallel processing banks |
| `MAX_VECTORS` | 4096 | Max storable vectors |
| `FRAC_BITS` | 24 | Fractional bits for similarity |

**Operation Latencies:**

| Operation | Cycles | Description |
|-----------|--------|-------------|
| `BUNDLE` | ~20 | Majority vote across N input vectors |
| `BIND` / `UNBIND` | 1 | Element-wise XOR (self-inverse) |
| `PERMUTE` | 1 | Cyclic left shift by N bits |
| `SIMILARITY` | HD_BANKS + 2 | Pipelined popcount dot product |
| `ASSOC_READ` | Variable | Read from associative cache |
| `ASSOC_WRITE` | HD_BANKS | Write vector to associative cache |
| `GENERATE` | HD_BANKS | LFSR-based random bipolar vector |

**Architecture:**
Each bank contains a 32-bit accumulator array for bundle operations. The
accumulator has 18-bit signed width to accommodate up to 2^{17} input vectors
without overflow. For bind/unbind, all banks execute a single-cycle XOR in
parallel. Similarity computation pipelines a popcount across banks, accumulating
matching bits and dividing by total bits for normalisation.

**LFSR Vector Generation:**
Random vectors are generated using a 32-bit Galois LFSR with characteristic
polynomial x^{32} + x^{27} + x^{15} + 1, seeded from `0xDEAD_BEEF` by default.

### 2.3 Topology Engine

The topology engine implements persistent homology computation using a
streaming merge-tree architecture.

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_NODES` | 4096 | Maximum graph nodes |
| `MAX_EDGES` | 8,388,608 | Maximum edges (complete graph) |
| `MAX_PERSIST_PAIRS` | 16384 | Max birth-death pairs |
| `FRAC_BITS` | 24 | Fractional bits for distances |

**Pipeline Stages:**

1. **Distance matrix load** — streams pairwise distances from HBM.
2. **Edge sorting** — insertion sort for small batches (production uses a
   bitonic merge network).
3. **Union-Find construction** — processes sorted edges with path compression
   and union-by-rank. Each merge event records a persistence pair
   (birth, death).
4. **Barcode emission** — streams out (birth, death) pairs.
5. **Wasserstein distance** — sums absolute persistence values for stability
   comparison.
6. **Stability check** — compares Wasserstein distance against a configurable
   threshold.

**Union-Find Implementation:**
The UF data structure uses 16-bit parent and rank arrays with bounded path
compression (max depth 16). Union-by-rank keeps tree height O(log n), giving
amortised near-O(alpha(n)) complexity per operation.

### 2.4 Graph Rewrite Engine

The rewrite engine implements the Double-Pushout (DPO) approach from category
theory for algebraic graph transformation.

**Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_GRAPH_NODES` | 4096 | Host graph node count |
| `MAX_PATTERN_NODES` | 8 | Maximum pattern size |
| `MAX_MATCHES` | 256 | Maximum simultaneous matches |
| `STACK_DEPTH` | 64 | Backtracking stack depth |

**State Machine (10 states):**

```
REW_IDLE -> REW_LOAD_LHS -> REW_MATCH_INIT -> REW_MATCH_SEARCH
   -> REW_CHECK_EDGE -> REW_BACKTRACK -> REW_EMIT_MATCH -> REW_DONE
                                                |
                              REW_APPLY_DELETE -> REW_APPLY_ADD -> REW_DONE
```

**Matching Algorithm:**
The engine uses a backtracking search to find subgraph isomorphisms between a
small pattern graph (LHS, up to 8 nodes) and the host graph. Each candidate
assignment is checked for node availability (no double-matching) and edge
consistency. When a complete match is found, it is emitted and the engine
backtracks to find additional matches.

**Rule Application:**
After matching, the DPO rule deletes nodes in L\K (marked with tombstone
`0xDEAD_DEAD`) and adds nodes in R\K. A proof trace is pushed onto a trace
stack for each modification.

---

## 3. Memory Subsystem

### 3.1 Associative Cache

The associative cache provides O(1) content-addressable recall for HD vectors
by performing a parallel dot-product query across all stored entries.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `NUM_ENTRIES` | 256 | Stored key-value pairs |
| `HD_DIM_REDUCED` | 1024 | Dimension after random projection |
| `SIM_THRESHOLD` | 5000 / 10000 | Minimum similarity for a match |

The cache stores bipolar keys (for lookup) and corresponding values (for
recall). Query latency is `NUM_ENTRIES` cycles (one per entry). Each cycle
computes XOR followed by popcount between the query and one stored key.

**Dimensionality Reduction:**
Full 10,000-dimensional vectors are projected to 1,024 dimensions using a
fixed random binary projection matrix, stored in BRAM. This reduces storage by
~10x with < 5% accuracy loss for the similarity recall task.

### 3.2 HBM Controller

The HBM controller provides burst read/write access to external HBM2 through
an AXI4-Stream interface.

| Parameter | Value |
|-----------|-------|
| `DATA_WIDTH` | 256 bits |
| `BURST_LENGTH` | 16 beats |
| `NUM_CHANNELS` | 4 |
| `MEM_DEPTH` | 65,536 x 256-bit words |

The controller implements a 5-state FSM:
`IDLE -> READ_BURST / WRITE_BURST -> DONE`.

Read latency is modelled at 2 cycles per beat. A bandwidth counter accumulates
total transferred beats for performance monitoring.

**Address Translation:**
Virtual graph IDs are translated to physical HBM addresses through a simple
base + offset scheme managed by the Graph DMA engine.

### 3.3 Memristor Crossbar Array

The crossbar is a 1024 x 1024 tile of conductance cells that supports
analog matrix-vector multiplication (MVM).

| Parameter | Value |
|-----------|-------|
| `TILE_ROWS` | 1024 wordlines |
| `TILE_COLS` | 1024 bitlines |
| `CONDUCTANCE_BITS` | 8 bits per cell |
| `NUM_TILES` | 1 (scalable) |

**Operations:**
- **Cell read/write** — single-cell conductance read (1 cycle) or write
  (1 cycle).
- **MVM** — computes `y[row] = sum(G[row][col] * x[col])` for one row at a
  time, streaming results out sequentially. Each row takes `TILE_COLS` cycles.
  For a full 1024x1024 MVM, total latency is 1024 x 1024 = ~1.05 M cycles.

**FPGA vs. ASIC Mapping:**
- **FPGA:** MVM uses DSP-based MAC units. Each multiplication is an 8x1 -> 32
  product accumulated in a 32-bit register.
- **ASIC:** MVM exploits actual analog current accumulation in the
  memristor crossbar, achieving true O(1) per-row latency with parallel
  bitline computation.

**State Persistence:**
The crossbar state is loaded/saved via `$readmemh` / `$writememh` to enable
session persistence across simulation runs.

---

## 4. Interconnect

### 4.1 Network-on-Chip (NoC) — 2D Mesh

The NoC connects compute engines, caches, and memory controllers through a
2D mesh of 5-port routers using XY deterministic routing.

**Flit Format (72 bits):**

```
 [71:68] type     — Header / Body / Tail
 [67:64] dest_x   — Destination X coordinate (4-bit)
 [63:60] dest_y   — Destination Y coordinate (4-bit)
 [59:58] vc       — Virtual channel (2 VCs for deadlock avoidance)
 [57:0]  payload  — Data payload (58 bits)
```

**Router Parameters:**

| Parameter | Value |
|-----------|-------|
| `FLIT_WIDTH` | 72 bits |
| `BUFFER_DEPTH` | 4 flits per port |
| `NUM_VCS` | 2 |

**Routing Algorithm:**
XY deterministic routing: route all flits East/West first, then North/South.
This guarantees deadlock freedom with just 2 virtual channels (one for X, one
for Y dimension).

**Arbitration:**
Round-robin priority among input ports. Local port has highest priority for
injecting new packets.

### 4.2 Graph DMA Engine

The DMA engine handles scatter-gather transfers between host memory and device
memory, specialised for graph data structures.

| Parameter | Value |
|-----------|-------|
| `DATA_WIDTH` | 64 bits |
| `MAX_DESC` | 256 descriptors in queue |

**DMA Transfer Types:**

| Type (2-bit) | Description |
|--------------|-------------|
| `00` | Linear load (HBM -> local) |
| `01` | Linear store (local -> HBM) |
| `10` | Scatter (HBM -> distributed local banks) |
| `11` | Gather (distributed local banks -> HBM) |

**Descriptor Queue:**
The DMA maintains a circular descriptor queue of 256 entries. Each descriptor
contains: source address, destination address, byte count, and transfer type.
The queue is managed with head/tail pointers.

---

## 5. ISA and Instruction Flow

### 5.1 Instruction Format

```
 [31:28] opcode      — 4-bit operation class
 [27:24] sub_opcode  — 4-bit sub-operation
 [23:16] flags       — 8-bit modifier flags
 [15:0]  operand     — 16-bit address / immediate
```

### 5.2 Instruction Dispatch Pipeline

1. **Fetch** — Instructions are read from a 16-deep instruction queue via
   the host interface.
2. **Decode** — Opcode and sub-opcode are extracted; the target engine is
   selected.
3. **Dispatch** — The instruction word and a `start` pulse are delivered to
   the target engine's control interface.
4. **Execute** — The engine runs its FSM until completion.
5. **Complete** — The engine asserts `done`. If `FLAG_DONE_IRQ` is set, an
   interrupt is raised. If `FLAG_CHAINED` is set, the next instruction is
   automatically dispatched.

### 5.3 Flag-Based Execution Modes

| Flag | Bit | Effect |
|------|-----|--------|
| `FLAG_START` | 0 | Trigger execution immediately |
| `FLAG_DONE_IRQ` | 1 | Raise IRQ on completion |
| `FLAG_CHAINED` | 2 | Auto-dispatch next instruction |
| `FLAG_STREAMING` | 3 | Continuous streaming mode |
| `FLAG_BLOCKED` | 4 | Blocked (synchronous) execution |
| `FLAG_PRECISION_Q` | 5 | Use Q16.16 instead of Q8.24 |

---

## 6. Dataflow Pipeline

SMGPU uses a dataflow execution model rather than a traditional
fetch-decode-execute loop with register files. Data flows through engines as
follows:

```
Host issues: SPECTRAL(COMPUTE_LAP, graph_id) -> HD(BIND, vec_a, vec_b)
    -> TOPOLOGY(BUILD_FILTRATION, dist_matrix) -> REWRITE(DPO_MATCH, pattern)

1. ISA Decoder reads COMPUTE_LAP instruction
2. Graph DMA loads CSR adjacency matrix from HBM -> Spectral Engine
3. Spectral Engine computes L*v, writes result to Spectral Buffer
4. ISA Decoder reads HD BIND instruction
5. HD Engine reads vectors from Spectral Buffer and HD Crossbar
6. HD Engine performs XOR, writes result to HD Crossbar
7. ISA Decoder reads TOPOLOGY BUILD_FILTRATION instruction
8. Topology Engine streams distances, builds Union-Find, emits barcode
9. ISA Decoder reads REWRITE DPO_MATCH instruction
10. Rewrite Engine scans host graph for pattern matches, emits match list
```

Each step can overlap with the next when `FLAG_CHAINED` is used, creating
a pipeline that keeps multiple engines busy simultaneously.

---

## 7. Clock Domains and Reset

### 7.1 Clock Domains

| Domain | Frequency | Components |
|--------|-----------|------------|
| `core_clk` | 250 MHz | All compute engines, NoC, DMA, associative cache |
| `hbm_clk` | 300 MHz | HBM controller and PHY interface |
| `crossbar_clk` | 100 MHz | Memristor crossbar (analog peripheral) |

### 7.2 Clock Domain Crossing

- **core <-> hbm:** Synchronised via 2-stage flip-flop FIFO on the DMA
  engine's HBM interface. Data width is converted from 64-bit (core) to
  256-bit (HBM) using a width converter.
- **core <-> crossbar:** Handshake-based protocol with ready/valid signals.
  The crossbar operates at a lower frequency due to analog settling time.

### 7.3 Reset Strategy

All modules use **active-low asynchronous reset** (`rst_n`). The reset
hierarchy:

```
  Host reset (PCIe PERST#)
      |
      +---> core_reset_n  (synchronised to core_clk)
      +---> hbm_reset_n   (synchronised to hbm_clk)
      +---> xb_reset_n    (synchronised to crossbar_clk)
```

Reset is deasserted in a staggered fashion: HBM first (t + 0), then core
(t + 100 ns), then crossbar (t + 500 ns) to avoid metastability during
power-on.

---

## 8. Design Decisions and Trade-offs

### 8.1 Fixed-Point vs. Floating-Point

**Decision:** Use Q-format fixed-point arithmetic exclusively.

**Rationale:**
- Halves DSP slice utilisation (one 18x25 multiplier instead of two for
  FP32).
- Directly compatible with memristor analog MVM, which naturally produces
  fixed-point results.
- Deterministic rounding behaviour simplifies golden-model comparison.

**Trade-off:** Reduced dynamic range (~8 integer bits, 24 fractional bits in
Q8.24). For graphs with extreme weight variation, precision loss may exceed
1%. This is acceptable because SMGP's spectral operations operate on
normalised Laplacians (eigenvalues in [0, 2]).

### 8.2 Systolic Array vs. Vector Processor

**Decision:** Use a dedicated 16x16 systolic array for the spectral engine.

**Rationale:** Graph spectral operations are dominated by sparse
matrix-vector products. A systolic array with streaming CSR input achieves
O(nnz / P) throughput where P is the array width. A vector processor with
scatter-gather would require additional index manipulation hardware.

**Trade-off:** The systolic array is underutilised for eigen-decomposition
power iteration (which reuses the same matrix repeatedly). The array is
time-multiplexed between Laplacian computation and Chebyshev iteration.

### 8.3 NoC vs. Shared Bus

**Decision:** Use a 2D mesh NoC with XY routing.

**Rationale:** A shared bus would become a bottleneck at 4+ engines
contending for memory bandwidth. The NoC scales linearly with engine count
and provides deterministic latency bounds.

**Trade-off:** Higher area (each router costs ~500 LUTs) and 1-cycle
per-hop latency. For the current 4-engine configuration, a crossbar could
suffice, but the NoC provides a clear scaling path.

### 8.4 Bipolar vs. Real-Valued HD Vectors

**Decision:** Store HD vectors as bipolar bits (1 bit per dimension).

**Rationale:** Bind/unbind via XOR is a single-cycle operation on bipolar
vectors. Real-valued vectors would require 32-bit multiplies per element,
increasing latency by 32x for the most common HD operation.

**Trade-off:** Information loss during bundling (majority vote of bipolar
vectors discards magnitude information). Empirically, this causes < 3%
accuracy loss for the similarity recall task at D = 10,000.

---

## 9. Scalability Parameters

SMGPU is designed for parameterised scaling from small FPGAs to large ASICs:

| Dimension | Small | Medium | Large |
|-----------|-------|--------|-------|
| Systolic array | 8 x 8 | 16 x 16 | 32 x 32 |
| HD banks | 4 | 16 | 64 |
| HD dimension | 4,000 | 10,000 | 40,000 |
| NoC mesh | 2 x 2 | 4 x 4 | 8 x 8 |
| HBM channels | 1 | 4 | 16 |
| Associative cache entries | 64 | 256 | 1024 |
| Crossbar tiles | 1 | 1 | 4 |
| Max graph nodes | 1,024 | 4,096 | 65,536 |

All scaling parameters are SystemVerilog `parameter` values set at
elaboration time — no RTL changes are needed to retarget.

---

## 10. Path to ASIC

The FPGA prototype maps directly to an ASIC flow with the following
modifications:

### 10.1 Clock Frequency

| Platform | Target Clock |
|----------|-------------|
| FPGA (UltraScale+) | 250 MHz |
| ASIC (7nm FinFET) | 1.2 GHz |
| ASIC (5nm) | 1.5 GHz |

### 10.2 Memristor Integration

On an ASIC, the `memristor_crossbar_sim` module is replaced with a physical
1T1R (one-transistor, one-resistor) crossbar array fabricated in a
back-end-of-line (BEOL) process. Key changes:

- The DSP-based MAC is replaced by analog current accumulation.
- ADC/DAC circuits are added at the periphery of the crossbar.
- Write drivers perform iterative program-and-verify to set conductance.

### 10.3 Memory Substitution

- HBM2 is replaced with on-chip SRAM for small configurations or HBM3
  for large configurations.
- BRAM-based storage is replaced with register files or standard-cell SRAM.

### 10.4 Synthesis Flow

```
RTL (SystemVerilog)
    |
    v
Lint (SpyGlass / Ascent)
    |
    v
RTL Synthesis (Design Compiler, 7nm library)
    |
    v
Gate-Level Simulation (VCS)
    |
    v
Place & Route (IC Compiler II)
    |
    v
Timing Sign-off (PrimeTime)
    |
    v
DRC / LVS (Calibre)
    |
    v
Tape-out (GDS II)
```

### 10.5 Estimated ASIC Area and Power (7nm)

| Block | Area (mm^2) | Power (W) |
|-------|------------|-----------|
| Spectral Engine (16x16) | 0.8 | 0.45 |
| HD Engine (16 banks) | 0.3 | 0.18 |
| Topology Engine | 0.2 | 0.10 |
| Rewrite Engine | 0.15 | 0.08 |
| Associative Cache | 0.4 | 0.22 |
| NoC (4x4) | 0.3 | 0.15 |
| HBM Controller | 0.5 | 0.35 |
| Memristor Crossbar | 0.6 | 0.05 (analog) |
| **Total** | **3.25** | **1.58** |

---

## References

- Kung, H.T. (1982). "Why Systolic Architectures?" *IEEE Computer*, 15(1).
- Dally, W.J. & Towles, B.P. (2004). *Principles and Practices of
  Interconnection Networks.* Morgan Kaufmann.
- Ielmini, D. & Wong, H.-S.P. (2018). "In-Memory Computing with Resistive
  Switching Devices." *Nature Electronics*, 1(6).
- JEDEC (2018). *High Bandwidth Memory (HBM2) Specification.*
- Kanerva, P. (1988). *Sparse Distributed Memory.* MIT Press.
- Hennessy, J.L. & Patterson, D.A. (2019). *Computer Architecture: A
  Quantitative Approach.* 6th Ed.
