// =============================================================================
// SMGP Core — Top-Level Integration
// =============================================================================
// Wraps all compute engines, memory subsystem, and interconnect into a
// single coherent processing core. The core exposes a unified instruction
// interface and memory bus for host interaction.
//
// Architecture:
//   Host -> ISA Decoder -> Engine Arbitration -> {Spectral, HD, Topo, Rewrite}
//                                  |
//                            NoC Interconnect
//                                  |
//                         {HBM, Crossbar, Cache}
//
// References:
//   - Hennessy & Patterson (2019). "Computer Architecture." 6th Ed.
//   - Asanovic et al. (2009). "The Landscape of Parallel Computing." UCB.
// =============================================================================

`timescale 1ns / 1ps

module smgp_core #(
    parameter int HD_DIM_WORDS   = 313,
    parameter int MAX_NODES      = 4096,
    parameter int MAX_EDGES      = 65536,
    parameter int FRAC_BITS      = 24,
    parameter int DATA_WIDTH     = 32,
    parameter int ADDR_WIDTH     = 16,
    parameter int NUM_ENGINES    = 4
)(
    input  logic                        clk,
    input  logic                        rst_n,

    // Host instruction interface
    input  logic [31:0]                 host_instruction,
    input  logic                        host_valid,
    output logic                        host_ready,
    output logic [3:0]                  core_status,

    // Host memory interface (AXI4-Lite-like)
    output logic [ADDR_WIDTH-1:0]       mem_addr,
    output logic                        mem_rd_en,
    output logic                        mem_wr_en,
    input  logic [DATA_WIDTH-1:0]       mem_rd_data,
    output logic [DATA_WIDTH-1:0]       mem_wr_data,
    input  logic                        mem_ready,

    // Interrupt
    output logic                        irq
);

    import smgp_isa_pkg::*;

    // =========================================================================
    // ISA Decoder
    // =========================================================================
    instruction_t current_instr;
    logic [3:0] opcode, sub_opcode;
    logic [7:0] flags;
    logic [15:0] operand;

    assign current_instr = bits_to_instr(host_instruction);
    assign opcode     = current_instr.opcode;
    assign sub_opcode = current_instr.sub_opcode;
    assign flags      = current_instr.flags;
    assign operand    = current_instr.operand;

    // =========================================================================
    // Engine busy/done signals
    // =========================================================================
    logic spectral_start, spectral_done;
    logic [3:0] spectral_status;
    logic hd_start, hd_done;
    logic [3:0] hd_status;
    logic topo_start, topo_done;
    logic [3:0] topo_status;
    logic rewrite_start, rewrite_done;
    logic [3:0] rewrite_status;

    // =========================================================================
    // Engine memory mux (shared bus)
    // =========================================================================
    logic [ADDR_WIDTH-1:0] eng_mem_addr [0:NUM_ENGINES-1];
    logic                  eng_mem_rd  [0:NUM_ENGINES-1];
    logic                  eng_mem_wr  [0:NUM_ENGINES-1];
    logic [DATA_WIDTH-1:0] eng_mem_wdata [0:NUM_ENGINES-1];
    logic                  eng_mem_ready;

    // Arbitration: round-robin engine selection
    logic [1:0] arb_grant;
    logic [NUM_ENGINES-1:0] engine_busy;
    logic [NUM_ENGINES-1:0] engine_done;

    assign engine_busy = {rewrite_start | !rewrite_done,
                          topo_start | !topo_done,
                          hd_start | !hd_done,
                          spectral_start | !spectral_done};
    assign engine_done = {rewrite_done, topo_done, hd_done, spectral_done};

    // Simple round-robin arbiter
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) arb_grant <= 2'b00;
        else if (host_valid && host_ready) begin
            case (opcode)
                OPC_SPECTRAL: arb_grant <= 2'b00;
                OPC_HD:       arb_grant <= 2'b01;
                OPC_TOPOLOGY: arb_grant <= 2'b10;
                OPC_REWRITE:  arb_grant <= 2'b11;
                default:      arb_grant <= arb_grant;
            endcase
        end else if (!engine_busy[arb_grant]) begin
            arb_grant <= arb_grant + 1;
        end
    end

    // Memory mux
    always_comb begin
        mem_addr    = eng_mem_addr[arb_grant];
        mem_rd_en   = eng_mem_rd[arb_grant];
        mem_wr_en   = eng_mem_wr[arb_grant];
        mem_wr_data = eng_mem_wdata[arb_grant];
    end
    assign eng_mem_ready = mem_ready;

    // =========================================================================
    // Engine Start Logic
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            spectral_start <= 1'b0;
            hd_start       <= 1'b0;
            topo_start     <= 1'b0;
            rewrite_start  <= 1'b0;
            host_ready     <= 1'b1;
            irq            <= 1'b0;
        end else begin
            spectral_start <= 1'b0;
            hd_start       <= 1'b0;
            topo_start     <= 1'b0;
            rewrite_start  <= 1'b0;

            if (host_valid && host_ready) begin
                host_ready <= 1'b0;
                case (opcode)
                    OPC_SPECTRAL: spectral_start <= 1'b1;
                    OPC_HD:       hd_start       <= 1'b1;
                    OPC_TOPOLOGY: topo_start     <= 1'b1;
                    OPC_REWRITE:  rewrite_start  <= 1'b1;
                    OPC_SYSTEM: begin
                        host_ready <= 1'b1;
                    end
                    default: host_ready <= 1'b1;
                endcase
            end

            // Deassert start on done
            if (spectral_done) spectral_start <= 1'b0;
            if (hd_done)       hd_start       <= 1'b0;
            if (topo_done)     topo_start     <= 1'b0;
            if (rewrite_done)  rewrite_start  <= 1'b0;

            // Return ready when engine finishes
            if (spectral_done || hd_done || topo_done || rewrite_done) begin
                host_ready <= 1'b1;
            end

            // IRQ on done with flag
            irq <= (spectral_done || hd_done || topo_done || rewrite_done) && (flags[1]);
        end
    end

    // =========================================================================
    // Core status
    // =========================================================================
    always_comb begin
        case (arb_grant)
            2'b00: core_status = spectral_status;
            2'b01: core_status = hd_status;
            2'b10: core_status = topo_status;
            2'b11: core_status = rewrite_status;
        endcase
        if (!engine_busy[arb_grant])
            core_status = STATUS_IDLE;
    end

    // =========================================================================
    // Engine Instantiations
    // =========================================================================
    // Spectral Engine
    spectral_engine #(
        .MAX_NODES(MAX_NODES),
        .MAX_NNZ(MAX_EDGES),
        .FRAC_BITS(FRAC_BITS),
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) u_spectral (
        .clk(clk), .rst_n(rst_n),
        .opcode(host_instruction),
        .start(spectral_start),
        .done(spectral_done),
        .status(spectral_status),
        .mem_addr(eng_mem_addr[0]),
        .mem_rd_en(eng_mem_rd[0]),
        .mem_wr_en(eng_mem_wr[0]),
        .mem_rd_data(mem_rd_data),
        .mem_wr_data(eng_mem_wdata[0]),
        .mem_ready(eng_mem_ready),
        .result_out(),
        .result_valid()
    );

    // HD Engine
    logic [DATA_WIDTH-1:0] hd_vec_a [0:15], hd_vec_b [0:15];
    logic [DATA_WIDTH-1:0] hd_result [0:15];

    assign hd_vec_a = '{default: 32'b0};
    assign hd_vec_b = '{default: 32'b0};

    hd_engine #(
        .HD_DIM_WORDS(HD_DIM_WORDS),
        .HD_BANKS(16),
        .FRAC_BITS(FRAC_BITS),
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) u_hd (
        .clk(clk), .rst_n(rst_n),
        .opcode(host_instruction),
        .start(hd_start),
        .done(hd_done),
        .status(hd_status),
        .mem_addr(eng_mem_addr[1]),
        .mem_rd_en(eng_mem_rd[1]),
        .mem_wr_en(eng_mem_wr[1]),
        .mem_rd_data(mem_rd_data),
        .mem_wr_data(eng_mem_wdata[1]),
        .mem_ready(eng_mem_ready),
        .vector_a(hd_vec_a),
        .vector_b(hd_vec_b),
        .result_out(hd_result),
        .result_valid(),
        .similarity_out()
    );

    // Topology Engine
    topology_engine #(
        .MAX_NODES(MAX_NODES),
        .MAX_EDGES(MAX_EDGES),
        .FRAC_BITS(FRAC_BITS),
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH + 4)
    ) u_topology (
        .clk(clk), .rst_n(rst_n),
        .opcode(host_instruction),
        .start(topo_start),
        .done(topo_done),
        .status(topo_status),
        .mem_addr(eng_mem_addr[2]),
        .mem_rd_en(eng_mem_rd[2]),
        .mem_wr_en(eng_mem_wr[2]),
        .mem_rd_data(mem_rd_data),
        .mem_wr_data(eng_mem_wdata[2]),
        .mem_ready(eng_mem_ready),
        .barcode_birth(),
        .barcode_death(),
        .barcode_valid(),
        .wasserstein_distance(),
        .stability_ok()
    );

    // Graph Rewrite Engine
    logic [15:0] pattern_nodes [0:7];
    logic [7:0]  pattern_node_count;

    graph_rewrite_engine #(
        .MAX_GRAPH_NODES(MAX_NODES),
        .MAX_PATTERN_NODES(8),
        .MAX_EDGES(MAX_EDGES),
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) u_rewrite (
        .clk(clk), .rst_n(rst_n),
        .opcode(host_instruction),
        .start(rewrite_start),
        .done(rewrite_done),
        .status(rewrite_status),
        .mem_addr(eng_mem_addr[3]),
        .mem_rd_en(eng_mem_rd[3]),
        .mem_wr_en(eng_mem_wr[3]),
        .mem_rd_data(mem_rd_data),
        .mem_wr_data(eng_mem_wdata[3]),
        .mem_ready(eng_mem_ready),
        .pattern_nodes(pattern_nodes),
        .pattern_node_count(pattern_node_count),
        .pattern_edge_count(8'b0),
        .pattern_edges('{default: 32'b0}),
        .match_out(),
        .match_valid(),
        .match_count(),
        .proof_trace_push(),
        .proof_trace_valid()
    );

endmodule
