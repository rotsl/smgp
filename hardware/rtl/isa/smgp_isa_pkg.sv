// =============================================================================
// SMGPU Instruction Set Architecture Package
// =============================================================================
// Defines the complete instruction encoding for the Graph Processing Unit.
// All instructions are 32-bit fixed-width with the following format:
//
//   [31:28] opcode      - Operation class
//   [27:24] sub_opcode  - Sub-operation within class
//   [23:16] flags       - Modifier flags
//   [15:0]  operand     - Address / immediate / register select
//
// The ISA follows a dataflow-oriented microarchitecture where compute engines
// operate on graph data streams rather than scalar registers.
//
// References:
//   - Hennessy, J.L. & Patterson, D.A. (2019). "Computer Architecture:
//     A Quantitative Approach." 6th Ed., Morgan Kaufmann.
//   - Asanovic, K., et al. (2009). "The Landscape of Parallel Computing
//     Research: A View from Berkeley." Technical Report UCB/EECS-2006-183.
// =============================================================================

package smgp_isa_pkg;

    // ---------------------------------------------------------------------------
    // Opcode definitions (4 bits, top nibble)
    // ---------------------------------------------------------------------------
    typedef enum logic [3:0] {
        OPC_NOP            = 4'h0,
        OPC_GRAPH_CTOR     = 4'h1,  // Graph construction ops
        OPC_SPECTRAL       = 4'h2,  // Spectral transform ops
        OPC_HD             = 4'h3,  // Hyperdimensional ops
        OPC_TOPOLOGY       = 4'h4,  // Topological persistence ops
        OPC_REWRITE        = 4'h5,  // Graph rewriting ops
        OPC_MEMORY         = 4'h6,  // Memory management ops
        OPC_SYSTEM         = 4'h7,  // System control ops
        OPC_RESERVED_8     = 4'h8,
        OPC_RESERVED_9     = 4'h9,
        OPC_RESERVED_A     = 4'hA,
        OPC_RESERVED_B     = 4'hB,
        OPC_RESERVED_C     = 4'hC,
        OPC_RESERVED_D     = 4'hD,
        OPC_RESERVED_E     = 4'hE,
        OPC_RESERVED_F     = 4'hF
    } opcode_e;

    // ---------------------------------------------------------------------------
    // Graph Construction Sub-Opcodes
    // ---------------------------------------------------------------------------
    typedef enum logic [3:0] {
        SUB_ADD_NODE       = 4'h0,
        SUB_ADD_EDGE       = 4'h1,
        SUB_DEL_NODE       = 4'h2,
        SUB_DEL_EDGE       = 4'h3,
        SUB_SET_WEIGHT     = 4'h4,
        SUB_QUERY_NODE     = 4'h5,
        SUB_SUBGRAPH_EXT   = 4'h6
    } graph_ctor_sub_e;

    // ---------------------------------------------------------------------------
    // Spectral Operation Sub-Opcodes
    // ---------------------------------------------------------------------------
    typedef enum logic [3:0] {
        SUB_COMPUTE_LAP    = 4'h0,
        SUB_EIGEN_DECOMP   = 4'h1,
        SUB_SPECTRAL_CONV  = 4'h2,
        SUB_WAVELET_XFORM  = 4'h3,
        SUB_GFT_FORWARD    = 4'h4,
        SUB_GFT_INVERSE    = 4'h5
    } spectral_sub_e;

    // ---------------------------------------------------------------------------
    // Hyperdimensional Operation Sub-Opcodes
    // ---------------------------------------------------------------------------
    typedef enum logic [3:0] {
        SUB_HD_BUNDLE      = 4'h0,
        SUB_HD_BIND        = 4'h1,
        SUB_HD_UNBIND      = 4'h2,
        SUB_HD_PERMUTE     = 4'h3,
        SUB_HD_SIMILARITY  = 4'h4,
        SUB_HD_ASSOC_READ  = 4'h5,
        SUB_HD_ASSOC_WRITE = 4'h6,
        SUB_HD_GENERATE    = 4'h7
    } hd_sub_e;

    // ---------------------------------------------------------------------------
    // Topology Sub-Opcodes
    // ---------------------------------------------------------------------------
    typedef enum logic [3:0] {
        SUB_BUILD_FILTRATION  = 4'h0,
        SUB_PERSIST_HOMOLOGY  = 4'h1,
        SUB_CHECK_STABILITY   = 4'h2,
        SUB_WASSERSTEIN_DIST  = 4'h3,
        SUB_PRUNE_BY_PERSIST  = 4'h4
    } topology_sub_e;

    // ---------------------------------------------------------------------------
    // Graph Rewrite Sub-Opcodes
    // ---------------------------------------------------------------------------
    typedef enum logic [3:0] {
        SUB_DPO_MATCH    = 4'h0,
        SUB_DPO_APPLY    = 4'h1,
        SUB_DPO_VERIFY   = 4'h2,
        SUB_PATTERN_LOAD = 4'h3
    } rewrite_sub_e;

    // ---------------------------------------------------------------------------
    // Memory Management Sub-Opcodes
    // ---------------------------------------------------------------------------
    typedef enum logic [3:0] {
        SUB_LOAD_GRAPH    = 4'h0,
        SUB_STORE_GRAPH   = 4'h1,
        SUB_PRUNE_SUBGRAPH = 4'h2,
        SUB_MEM_READ      = 4'h3,
        SUB_MEM_WRITE     = 4'h4,
        SUB_MEM_FLUSH     = 4'h5
    } memory_sub_e;

    // ---------------------------------------------------------------------------
    // System Sub-Opcodes
    // ---------------------------------------------------------------------------
    typedef enum logic [3:0] {
        SUB_SYS_CONFIG    = 4'h0,
        SUB_SYS_START     = 4'h1,
        SUB_SYS_HALT      = 4'h2,
        SUB_SYS_STATUS    = 4'h3,
        SUB_SYS_IRQ_ACK   = 4'h4,
        SUB_SYS_RESET     = 4'h5
    } system_sub_e;

    // ---------------------------------------------------------------------------
    // Instruction word structure
    // ---------------------------------------------------------------------------
    typedef struct packed {
        logic [3:0]  opcode;
        logic [3:0]  sub_opcode;
        logic [7:0]  flags;
        logic [15:0] operand;
    } instruction_t;

    // ---------------------------------------------------------------------------
    // Flag bit definitions
    // ---------------------------------------------------------------------------
    localparam logic [7:0] FLAG_NONE        = 8'h00;
    localparam logic [7:0] FLAG_START       = 8'h01;  // Start execution
    localparam logic [7:0] FLAG_DONE_IRQ    = 8'h02;  // Assert IRQ on completion
    localparam logic [7:0] FLAG_CHAINED     = 8'h04;  // Chain to next instruction
    localparam logic [7:0] FLAG_STREAMING   = 8'h08;  // Streaming mode
    localparam logic [7:0] FLAG_BLOCKED     = 8'h10;  // Blocked execution mode
    localparam logic [7:0] FLAG_PRECISION_Q = 8'h20;  // Use Q8.24 vs Q16.16

    // ---------------------------------------------------------------------------
    // Status codes
    // ---------------------------------------------------------------------------
    typedef enum logic [3:0] {
        STATUS_IDLE     = 4'h0,
        STATUS_RUNNING  = 4'h1,
        STATUS_DONE     = 4'h2,
        STATUS_ERROR    = 4'h3,
        STATUS_STALL    = 4'h4,
        STATUS_FAULT    = 4'h5
    } status_e;

    // ---------------------------------------------------------------------------
    // Address space map
    // ---------------------------------------------------------------------------
    localparam logic [15:0] ADDR_GRAPH_MEM_BASE  = 16'h0000;
    localparam logic [15:0] ADDR_HD_CROSSBAR_BASE = 16'h2000;
    localparam logic [15:0] ADDR_SPECTRAL_BUF     = 16'h4000;
    localparam logic [15:0] ADDR_TOPO_BUF         = 16'h6000;
    localparam logic [15:0] ADDR_REWRITE_BUF      = 16'h8000;
    localparam logic [15:0] ADDR_CONFIG_REGS      = 16'hF000;

    // ---------------------------------------------------------------------------
    // Helper functions
    // ---------------------------------------------------------------------------
    function automatic instruction_t make_instr(
        input opcode_e    opcode,
        input logic [3:0] sub_opcode,
        input logic [7:0] flags,
        input logic [15:0] operand
    );
        make_instr.opcode     = opcode;
        make_instr.sub_opcode = sub_opcode;
        make_instr.flags      = flags;
        make_instr.operand    = operand;
    endfunction

    function automatic logic [31:0] instr_to_bits(input instruction_t instr);
        instr_to_bits = {instr.opcode, instr.sub_opcode, instr.flags, instr.operand};
    endfunction

    function automatic instruction_t bits_to_instr(input logic [31:0] bits);
        bits_to_instr.opcode     = bits[31:28];
        bits_to_instr.sub_opcode = bits[27:24];
        bits_to_instr.flags      = bits[23:16];
        bits_to_instr.operand    = bits[15:0];
    endfunction

endpackage
