# SMGPU Instruction Set Architecture (ISA) Reference

> Complete specification of the 32-bit instruction encoding, all opcodes,
> sub-opcodes, flag definitions, address space layout, and usage examples.

---

## 1. Instruction Format

Every SMGPU instruction is a fixed-width 32-bit word with the following layout:

```
 31      28 27      24 23      16 15                    0
 +----------+----------+----------+---------------------+
 |  opcode  |sub_opcode|  flags   |      operand        |
 |  [31:28] | [27:24]  | [23:16]  |       [15:0]        |
 +----------+----------+----------+---------------------+
      4 bits    4 bits    8 bits        16 bits
```

| Field | Bits | Description |
|-------|------|-------------|
| `opcode` | [31:28] | Primary operation class |
| `sub_opcode` | [27:24] | Sub-operation within the class |
| `flags` | [23:16] | Execution modifier flags |
| `operand` | [15:0] | Address, immediate value, or register selector |

The instruction word is defined in RTL as:

```systemverilog
typedef struct packed {
    logic [3:0]  opcode;
    logic [3:0]  sub_opcode;
    logic [7:0]  flags;
    logic [15:0] operand;
} instruction_t;
```

---

## 2. Opcode Definitions

### 2.1 Primary Opcodes (4 bits)

| Hex | Enum | Description |
|-----|------|-------------|
| `0x0` | `OPC_NOP` | No operation |
| `0x1` | `OPC_GRAPH_CTOR` | Graph construction and mutation |
| `0x2` | `OPC_SPECTRAL` | Spectral transform operations |
| `0x3` | `OPC_HD` | Hyperdimensional vector operations |
| `0x4` | `OPC_TOPOLOGY` | Topological persistence analysis |
| `0x5` | `OPC_REWRITE` | Graph rewriting (DPO) |
| `0x6` | `OPC_MEMORY` | Memory management and transfer |
| `0x7` | `OPC_SYSTEM` | System control and configuration |
| `0x8`–`0xF` | Reserved | Reserved for future extension |

### 2.2 Sub-Opcodes by Opcode Class

#### 2.2.1 Graph Construction (`OPC_GRAPH_CTOR = 0x1`)

| Hex | Enum | Description | Operand |
|-----|------|-------------|---------|
| `0x0` | `SUB_ADD_NODE` | Add a new node to the graph | Node ID (16-bit) |
| `0x1` | `SUB_ADD_EDGE` | Add an edge between two nodes | {src[15:8], dst[7:0]} |
| `0x2` | `SUB_DEL_NODE` | Delete a node and its edges | Node ID (16-bit) |
| `0x3` | `SUB_DEL_EDGE` | Delete an edge | {src[15:8], dst[7:0]} |
| `0x4` | `SUB_SET_WEIGHT` | Set edge weight | {edge_idx[15:8], weight[7:0]} |
| `0x5` | `SUB_QUERY_NODE` | Query node properties | Node ID (16-bit) |
| `0x6` | `SUB_SUBGRAPH_EXT` | Extract subgraph by node list | Buffer address (16-bit) |

#### 2.2.2 Spectral (`OPC_SPECTRAL = 0x2`)

| Hex | Enum | Description | Operand |
|-----|------|-------------|---------|
| `0x0` | `SUB_COMPUTE_LAP` | Compute normalised Laplacian | Graph buffer address |
| `0x1` | `SUB_EIGEN_DECOMP` | Eigendecomposition (power iter.) | K = operand[15:0] |
| `0x2` | `SUB_SPECTRAL_CONV` | Chebyshev spectral convolution | Polynomial order K |
| `0x3` | `SUB_WAVELET_XFORM` | Heat-kernel graph wavelet | Scale factor (16-bit) |
| `0x4` | `SUB_GFT_FORWARD` | Graph Fourier Transform (fwd) | Buffer address |
| `0x5` | `SUB_GFT_INVERSE` | Graph Fourier Transform (inv) | Buffer address |

#### 2.2.3 Hyperdimensional (`OPC_HD = 0x3`)

| Hex | Enum | Description | Operand |
|-----|------|-------------|---------|
| `0x0` | `SUB_HD_BUNDLE` | Majority-vote bundle of N vectors | N = operand[15:0] |
| `0x1` | `SUB_HD_BIND` | Bind: a XOR b (associative) | {addr_a[15:8], addr_b[7:0]} |
| `0x2` | `SUB_HD_UNBIND` | Unbind: bound XOR b (retrieval) | {addr_bound[15:8], addr_b[7:0]} |
| `0x3` | `SUB_HD_PERMUTE` | Cyclic left-shift permutation | Shift amount (16-bit) |
| `0x4` | `SUB_HD_SIMILARITY` | Cosine similarity via popcount | {addr_a[15:8], addr_b[7:0]} |
| `0x5` | `SUB_HD_ASSOC_READ` | Associative recall from cache | Query vector address |
| `0x6` | `SUB_HD_ASSOC_WRITE` | Associative store to cache | {addr_key[15:8], addr_val[7:0]} |
| `0x7` | `SUB_HD_GENERATE` | Generate random bipolar vector | Seed (16-bit) |

#### 2.2.4 Topology (`OPC_TOPOLOGY = 0x4`)

| Hex | Enum | Description | Operand |
|-----|------|-------------|---------|
| `0x0` | `SUB_BUILD_FILTRATION` | Build Vietoris-Rips filtration | Distance buffer addr |
| `0x1` | `SUB_PERSIST_HOMOLOGY` | Compute persistence pairs | Max pairs (16-bit) |
| `0x2` | `SUB_CHECK_STABILITY` | Verify topological stability | Threshold (16-bit) |
| `0x3` | `SUB_WASSERSTEIN_DIST` | Compute Wasserstein distance | Buffer address |
| `0x4` | `SUB_PRUNE_BY_PERSIST` | Prune graph by persistence | Cutoff threshold |

#### 2.2.5 Graph Rewrite (`OPC_REWRITE = 0x5`)

| Hex | Enum | Description | Operand |
|-----|------|-------------|---------|
| `0x0` | `SUB_DPO_MATCH` | Match pattern LHS in host graph | Pattern buffer addr |
| `0x1` | `SUB_DPO_APPLY` | Apply rewrite rule (del + add) | Match index (16-bit) |
| `0x2` | `SUB_DPO_VERIFY` | Verify rule application | Match index (16-bit) |
| `0x3` | `SUB_PATTERN_LOAD` | Load a rewrite pattern | Pattern buffer addr |

#### 2.2.6 Memory Management (`OPC_MEMORY = 0x6`)

| Hex | Enum | Description | Operand |
|-----|------|-------------|---------|
| `0x0` | `SUB_LOAD_GRAPH` | Load graph from HBM to local | {hbm_addr, local_addr} |
| `0x1` | `SUB_STORE_GRAPH` | Store graph from local to HBM | {local_addr, hbm_addr} |
| `0x2` | `SUB_PRUNE_SUBGRAPH` | Free subgraph memory | Buffer address |
| `0x3` | `SUB_MEM_READ` | Raw memory read | Address (16-bit) |
| `0x4` | `SUB_MEM_WRITE` | Raw memory write | Address (16-bit) |
| `0x5` | `SUB_MEM_FLUSH` | Flush write buffers to HBM | Channel mask (16-bit) |

#### 2.2.7 System Control (`OPC_SYSTEM = 0x7`)

| Hex | Enum | Description | Operand |
|-----|------|-------------|---------|
| `0x0` | `SUB_SYS_CONFIG` | Configure engine parameters | Config register addr |
| `0x1` | `SUB_SYS_START` | Start execution pipeline | Start mode (16-bit) |
| `0x2` | `SUB_SYS_HALT` | Halt execution pipeline | Reason code (16-bit) |
| `0x3` | `SUB_SYS_STATUS` | Read system status | Status register addr |
| `0x4` | `SUB_SYS_IRQ_ACK` | Acknowledge interrupt | IRQ number (16-bit) |
| `0x5` | `SUB_SYS_RESET` | Soft reset engine(s) | Engine mask (16-bit) |

---

## 3. Flag Definitions

Flags are in bits [23:16] of the instruction word. Multiple flags may be set
simultaneously by ORing their values.

| Hex | Bit | Name | Description |
|-----|-----|------|-------------|
| `0x01` | 0 | `FLAG_START` | Assert the `start` signal to the target engine immediately upon decode |
| `0x02` | 1 | `FLAG_DONE_IRQ` | Raise a host interrupt when the engine asserts `done` |
| `0x04` | 2 | `FLAG_CHAINED` | Automatically dispatch the next instruction from the queue when `done` asserts |
| `0x08` | 3 | `FLAG_STREAMING` | Enable continuous streaming mode (engine does not return to IDLE between iterations) |
| `0x10` | 4 | `FLAG_BLOCKED` | Block the host until the instruction completes (synchronous execution) |
| `0x20` | 5 | `FLAG_PRECISION_Q` | Use Q16.16 fixed-point format instead of the default Q8.24 |
| `0x40` | 6 | Reserved | Reserved for future use |
| `0x80` | 7 | Reserved | Reserved for future use |

**Common Flag Combinations:**

| Value | Name | Use Case |
|-------|------|----------|
| `0x01` | `START` | Fire-and-forget execution |
| `0x03` | `START \| DONE_IRQ` | Asynchronous execution with completion notification |
| `0x07` | `START \| DONE_IRQ \| CHAINED` | Pipelined multi-instruction sequence |
| `0x19` | `START \| BLOCKED \| PRECISION_Q` | Synchronous execution with higher precision |
| `0x0B` | `START \| DONE_IRQ \| STREAMING` | Continuous streaming with interrupt notification |

---

## 4. Status Codes

Engines report their status in a 4-bit field:

| Hex | Enum | Description |
|-----|------|-------------|
| `0x0` | `STATUS_IDLE` | Engine is idle, ready to accept work |
| `0x1` | `STATUS_RUNNING` | Engine is actively processing an instruction |
| `0x2` | `STATUS_DONE` | Engine completed the last instruction successfully |
| `0x3` | `STATUS_ERROR` | Engine encountered an error (illegal opcode, address, etc.) |
| `0x4` | `STATUS_STALL` | Engine is stalled waiting for memory or NoC |
| `0x5` | `STATUS_FAULT` | Engine detected a hardware fault (parity, ECC, etc.) |

---

## 5. Address Space Map

The 16-bit operand field addresses a unified memory space. The top nibble of
the address selects the memory region:

| Base Address | End Address | Region | Description |
|-------------|-------------|--------|-------------|
| `0x0000` | `0x1FFF` | Graph Memory | CSR adjacency matrix, node properties |
| `0x2000` | `0x3FFF` | HD Crossbar | HD vector storage (memristor crossbar) |
| `0x4000` | `0x5FFF` | Spectral Buffer | Eigenvector and Laplacian workspace |
| `0x6000` | `0x7FFF` | Topology Buffer | Distance matrix, persistence pairs |
| `0x8000` | `0x9FFF` | Rewrite Buffer | Pattern storage, match results |
| `0xA000` | `0xBFFF` | DMA Descriptors | Scatter-gather descriptor queue |
| `0xC000` | `0xDFFF` | Associative Cache | CAM key-value pairs |
| `0xE000` | `0xEFFF` | Instruction Queue | Host-to-device instruction buffer |
| `0xF000` | `0xFFFF` | Configuration Registers | Engine configuration and status |

### 5.1 Configuration Register Map (offsets from `0xF000`)

| Offset | Register | Description |
|--------|----------|-------------|
| `0xF000` | `SYS_VERSION` | Hardware version (read-only) |
| `0xF001` | `SYS_CLK_FREQ` | Core clock frequency in MHz |
| `0xF002` | `SYS_NUM_NODES` | Current graph node count |
| `0xF003` | `SYS_NUM_EDGES` | Current graph edge count |
| `0xF004` | `SYS_HD_DIM` | Hyperdimensional dimension |
| `0xF005` | `SYS_FRAC_BITS` | Current fractional bit width |
| `0xF006` | `SYS_ENGINE_MASK` | Bitmask of active engines |
| `0xF007` | `SYS_IRQ_PENDING` | Pending interrupt bitmask |
| `0xF008` | `SYS_IRQ_ENABLE` | Interrupt enable mask |
| `0xF009` | `SYS_PERF_COUNTERS` | Performance counter base |

---

## 6. Instruction Encoding Examples

### 6.1 Compute Normalised Laplacian

```python
# Opcode: 0x2 (SPECTRAL), Sub: 0x0 (COMPUTE_LAP)
# Flags: 0x03 (START | DONE_IRQ)
# Operand: 0x0001 (graph buffer at address 1)

opcode     = 0b0010   # OPC_SPECTRAL
sub_opcode = 0b0000   # SUB_COMPUTE_LAP
flags      = 0b00000011  # FLAG_START | FLAG_DONE_IRQ
operand    = 0x0001   # graph buffer address

instruction = (opcode << 28) | (sub_opcode << 24) | (flags << 16) | operand
# => 0x20030001
```

### 6.2 HD Bind Two Vectors

```python
# Opcode: 0x3 (HD), Sub: 0x1 (BIND)
# Flags: 0x01 (START)
# Operand: {addr_a=0x20, addr_b=0x21}

opcode     = 0b0011   # OPC_HD
sub_opcode = 0b0001   # SUB_HD_BIND
flags      = 0b00000001  # FLAG_START
operand    = 0x2021   # addr_a in upper byte, addr_b in lower byte

instruction = (opcode << 28) | (sub_opcode << 24) | (flags << 16) | operand
# => 0x31012021
```

### 6.3 Chained Pipeline: Laplacian -> Eigen-decomp

```python
# Instruction 1: Compute Laplacian
instr_1 = 0x20050001  # SPECTRAL, COMPUTE_LAP, START|CHAINED|DONE_IRQ, addr=1

# Instruction 2: Eigen-decomposition (K=32, auto-started after instr_1)
instr_2 = 0x21050001  # SPECTRAL, EIGEN_DECOMP, START|CHAINED|DONE_IRQ, K=1 (extended)

# The ISA decoder dispatches instr_1, waits for done, then auto-dispatches instr_2
```

### 6.4 Build Filtration and Compute Persistence

```python
# Instruction 1: Build filtration
instr_1 = 0x40030000  # TOPOLOGY, BUILD_FILTRATION, START|DONE_IRQ, addr=0x0000

# Instruction 2: Compute persistence homology (max 4096 pairs)
instr_2 = 0x41030000  # TOPOLOGY, PERSIST_HOMOLOGY, START|DONE_IRQ, max=0x0000
```

### 6.5 DPO Pattern Match

```python
# Instruction: Match pattern loaded at buffer 0x8000
instr = 0x50030080  # REWRITE, DPO_MATCH, START|DONE_IRQ, addr=0x0080
```

### 6.6 System Soft Reset

```python
# Reset engines 0b0101 (Spectral + Topology)
instr = 0x70000005  # SYSTEM, RESET, no flags, engine_mask=0x0005
```

---

## 7. Encoding Tables (Binary Reference)

### 7.1 Full Opcode + Sub-Opcode Quick Reference

| Instruction | Opcode | Sub | Hex Prefix |
|-------------|--------|-----|------------|
| NOP | 0000 | 0000 | `0x00` |
| ADD_NODE | 0001 | 0000 | `0x10` |
| ADD_EDGE | 0001 | 0001 | `0x11` |
| DEL_NODE | 0001 | 0010 | `0x12` |
| DEL_EDGE | 0001 | 0011 | `0x13` |
| SET_WEIGHT | 0001 | 0100 | `0x14` |
| QUERY_NODE | 0001 | 0101 | `0x15` |
| SUBGRAPH_EXT | 0001 | 0110 | `0x16` |
| COMPUTE_LAP | 0010 | 0000 | `0x20` |
| EIGEN_DECOMP | 0010 | 0001 | `0x21` |
| SPECTRAL_CONV | 0010 | 0010 | `0x22` |
| WAVELET_XFORM | 0010 | 0011 | `0x23` |
| GFT_FORWARD | 0010 | 0100 | `0x24` |
| GFT_INVERSE | 0010 | 0101 | `0x25` |
| HD_BUNDLE | 0011 | 0000 | `0x30` |
| HD_BIND | 0011 | 0001 | `0x31` |
| HD_UNBIND | 0011 | 0010 | `0x32` |
| HD_PERMUTE | 0011 | 0011 | `0x33` |
| HD_SIMILARITY | 0011 | 0100 | `0x34` |
| HD_ASSOC_READ | 0011 | 0101 | `0x35` |
| HD_ASSOC_WRITE | 0011 | 0110 | `0x36` |
| HD_GENERATE | 0011 | 0111 | `0x37` |
| BUILD_FILTRATION | 0100 | 0000 | `0x40` |
| PERSIST_HOMOLOGY | 0100 | 0001 | `0x41` |
| CHECK_STABILITY | 0100 | 0010 | `0x42` |
| WASSERSTEIN_DIST | 0100 | 0011 | `0x43` |
| PRUNE_BY_PERSIST | 0100 | 0100 | `0x44` |
| DPO_MATCH | 0101 | 0000 | `0x50` |
| DPO_APPLY | 0101 | 0001 | `0x51` |
| DPO_VERIFY | 0101 | 0010 | `0x52` |
| PATTERN_LOAD | 0101 | 0011 | `0x53` |
| LOAD_GRAPH | 0110 | 0000 | `0x60` |
| STORE_GRAPH | 0110 | 0001 | `0x61` |
| PRUNE_SUBGRAPH | 0110 | 0010 | `0x62` |
| MEM_READ | 0110 | 0011 | `0x63` |
| MEM_WRITE | 0110 | 0100 | `0x64` |
| MEM_FLUSH | 0110 | 0101 | `0x65` |
| SYS_CONFIG | 0111 | 0000 | `0x70` |
| SYS_START | 0111 | 0001 | `0x71` |
| SYS_HALT | 0111 | 0010 | `0x72` |
| SYS_STATUS | 0111 | 0011 | `0x73` |
| SYS_IRQ_ACK | 0111 | 0100 | `0x74` |
| SYS_RESET | 0111 | 0101 | `0x75` |

---

## 8. RTL Helper Functions

The ISA package provides utility functions for constructing and parsing
instructions:

```systemverilog
// Construct an instruction
instruction_t instr = smgp_isa_pkg::make_instr(
    .opcode(OPC_SPECTRAL),
    .sub_opcode(SUB_COMPUTE_LAP),
    .flags(FLAG_START | FLAG_DONE_IRQ),
    .operand(16'h0001)
);

// Convert to 32-bit wire
logic [31:0] bits = smgp_isa_pkg::instr_to_bits(instr);

// Parse from 32-bit wire
instruction_t parsed = smgp_isa_pkg::bits_to_instr(bits);
```

---

## References

- Hennessy, J.L. & Patterson, D.A. (2019). *Computer Architecture: A Quantitative Approach.* 6th Ed.
- Asanovic, K., et al. (2009). "The Landscape of Parallel Computing Research: A View from Berkeley." UCB/EECS-2006-183.
