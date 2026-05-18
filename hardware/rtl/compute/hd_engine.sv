// =============================================================================
// Hyperdimensional Vector Processing Engine
// =============================================================================
// Massively parallel compute unit for hyperdimensional vector operations.
// Processes D-dimensional bipolar vectors (D=10000 default) organized as
// 313 x 32-bit words. Supports:
//   - Bundle (majority vote): accumulates N vectors and outputs majority
//   - Bind/Unbind (XOR): element-wise XOR, self-inverse for bipolar
//   - Permute: configurable cyclic shift for order encoding
//   - Similarity: parallel dot product with popcount-based acceleration
//   - Associative read/write: content-addressable lookup
//
// The engine processes all 313 words in parallel using replicated logic,
// achieving single-cycle latency for bind/unbind/permute and multi-cycle
// for bundle (pipelined accumulation).
//
// References:
//   - Kanerva, P. (2009). "Hyperdimensional Computing." Cognitive Computation.
//   - Rahimi, A., et al. (2017). "Hyperdimensional Computing for Blind
//     Classification." ISVLSI.
//   - Ielmini, D. & Wong, H.-S.P. (2018). "In-Memory Computing with
//     Resistive Switching Devices." Nature Electronics.
// =============================================================================

`timescale 1ns / 1ps

module hd_engine #(
    parameter int HD_DIM_WORDS    = 313,      // 10000 bits / 32 = 312.5 -> 313
    parameter int HD_DIM_BITS     = HD_DIM_WORDS * 32,
    parameter int HD_BANKS        = 16,       // Number of parallel processing banks
    parameter int MAX_VECTORS     = 4096,     // Max stored vectors
    parameter int FRAC_BITS       = 24,
    parameter int DATA_WIDTH      = 32,
    parameter int ADDR_WIDTH      = 16
)(
    input  logic                        clk,
    input  logic                        rst_n,

    // Control
    input  logic [31:0]                 opcode,
    input  logic                        start,
    output logic                        done,
    output logic [3:0]                  status,

    // Memory interface (to associative cache / crossbar)
    output logic [ADDR_WIDTH-1:0]       mem_addr,
    output logic                        mem_rd_en,
    output logic                        mem_wr_en,
    input  logic [DATA_WIDTH-1:0]       mem_rd_data,
    output logic [DATA_WIDTH-1:0]       mem_wr_data,
    input  logic                        mem_ready,

    // Vector I/O ports
    input  logic [DATA_WIDTH-1:0]       vector_a    [0:HD_BANKS-1],
    input  logic [DATA_WIDTH-1:0]       vector_b    [0:HD_BANKS-1],
    output logic [DATA_WIDTH-1:0]       result_out  [0:HD_BANKS-1],
    output logic                        result_valid,
    output logic [31:0]                 similarity_out
);

    import smgp_isa_pkg::*;
    import hd_ops_pkg::*;

    // =========================================================================
    // State machine
    // =========================================================================
    typedef enum logic [3:0] {
        HD_IDLE         = 4'd0,
        HD_BUNDLE       = 4'd1,
        HD_BIND         = 4'd2,
        HD_PERMUTE      = 4'd3,
        HD_SIMILARITY   = 4'd4,
        HD_ASSOC_READ   = 4'd5,
        HD_ASSOC_WRITE  = 4'd6,
        HD_GENERATE     = 4'd7,
        HD_DONE         = 4'd8
    } hd_state_e;

    hd_state_e state;

    // =========================================================================
    // Internal storage
    // =========================================================================
    // Accumulator for bundle operation
    logic signed [17:0] accumulator [0:HD_BANKS-1][0:31];
    logic [DATA_WIDTH-1:0] bundle_result [0:HD_BANKS-1];

    // Working registers
    logic [DATA_WIDTH-1:0] work_a    [0:HD_BANKS-1];
    logic [DATA_WIDTH-1:0] work_b    [0:HD_BANKS-1];
    logic [DATA_WIDTH-1:0] work_out  [0:HD_BANKS-1];

    // Similarity computation
    logic signed [63:0] dot_product;
    logic [31:0] match_count;
    logic [31:0] total_count;
    logic [31:0] word_idx;

    // Permute shift register
    logic [31:0] shift_amount;

    // LFSR for vector generation
    logic [31:0] lfsr_state;

    // =========================================================================
    // HD Bank Processing (parallel across banks)
    // =========================================================================
    genvar g_bank;
    generate
        for (g_bank = 0; g_bank < HD_BANKS; g_bank++) begin : gen_bank
            // Bundle: accumulate bits for majority vote
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (int b = 0; b < 32; b++)
                        accumulator[g_bank][b] <= 18'sd0;
                    bundle_result[g_bank] <= 32'b0;
                    work_out[g_bank] <= 32'b0;
                end else begin
                    case (state)
                        HD_BUNDLE: begin
                            if (start) begin
                                for (int b = 0; b < 32; b++) begin
                                    // +1 for bit=1, -1 for bit=0 (bipolar)
                                    accumulator[g_bank][b] <= accumulator[g_bank][b] +
                                        (vector_a[g_bank][b] ? 18'sd1 : -18'sd1);
                                end
                            end else begin
                                // Finalize: majority vote
                                for (int b = 0; b < 32; b++) begin
                                    bundle_result[g_bank][b] <= (accumulator[g_bank][b] >= 0);
                                end
                            end
                        end

                        HD_BIND: begin
                            // XOR for bipolar bind (self-inverse)
                            work_out[g_bank] <= vector_a[g_bank] ^ vector_b[g_bank];
                        end

                        HD_PERMUTE: begin
                            // Cyclic left shift
                            work_out[g_bank] <= (vector_a[g_bank] << shift_amount[4:0]) |
                                                (vector_a[g_bank] >> (6'd32 - shift_amount[4:0]));
                        end

                        HD_SIMILARITY: begin
                            // Compute per-bank contribution to dot product
                            // Similarity = (matches - mismatches) / total_bits
                            logic [31:0] xor_bits;
                            logic [31:0] matched;
                            xor_bits = vector_a[g_bank] ^ vector_b[g_bank];
                            matched  = ~xor_bits;
                        end

                        default: begin
                            work_out[g_bank] <= work_out[g_bank];
                        end
                    endcase
                end
            end
        end
    endgenerate

    // =========================================================================
    // Dot product accumulation (across all banks, pipelined)
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dot_product  <= 64'sd0;
            match_count  <= 32'b0;
            total_count  <= 32'b0;
            word_idx     <= 32'b0;
            done         <= 1'b0;
            status       <= STATUS_IDLE;
            result_valid <= 1'b0;
            similarity_out <= 32'b0;
            lfsr_state   <= 32'hDEAD_BEEF;
        end else begin
            result_valid <= 1'b0;
            done         <= 1'b0;

            case (state)
                HD_IDLE: begin
                    if (start) begin
                        case (opcode[27:24])
                            4'h0: state <= HD_BUNDLE;       // HD_BUNDLE
                            4'h1: state <= HD_BIND;         // HD_BIND
                            4'h2: state <= HD_BIND;         // HD_UNBIND (same as bind)
                            4'h3: state <= HD_PERMUTE;      // HD_PERMUTE
                            4'h4: state <= HD_SIMILARITY;   // HD_SIMILARITY
                            4'h7: state <= HD_GENERATE;     // HD_GENERATE
                            default: state <= HD_IDLE;
                        endcase
                        status <= STATUS_RUNNING;
                        shift_amount <= opcode;
                        word_idx <= 32'b0;
                        dot_product <= 64'sd0;
                    end
                end

                HD_BUNDLE: begin
                    // Multi-cycle accumulation
                    if (word_idx >= 32'd16) begin
                        state <= HD_DONE;
                    end else begin
                        word_idx <= word_idx + 1;
                    end
                end

                HD_BIND: begin
                    state <= HD_DONE;
                end

                HD_PERMUTE: begin
                    state <= HD_DONE;
                end

                HD_SIMILARITY: begin
                    // Accumulate similarity across banks (pipelined)
                    if (word_idx < HD_BANKS) begin
                        logic [31:0] xor_bits, matched;
                        xor_bits = work_out[word_idx] ^ 32'b0; // use current bank output
                        matched  = ~xor_bits;
                        match_count <= match_count + $countones(matched);
                        total_count <= total_count + 32'd32;
                        word_idx   <= word_idx + 1;
                    end else begin
                        // Final similarity in Q8.24
                        similarity_out <= (match_count << 8) / total_count;
                        state <= HD_DONE;
                    end
                end

                HD_GENERATE: begin
                    // LFSR-based random bipolar vector generation
                    lfsr_state <= lfsr_next(lfsr_state);
                    if (word_idx < HD_BANKS) begin
                        work_out[word_idx] <= lfsr_state;
                        word_idx <= word_idx + 1;
                    end else begin
                        state <= HD_DONE;
                    end
                end

                HD_ASSOC_READ: begin
                    // Read from associative cache
                    mem_addr  <= ADDR_HD_CROSSBAR_BASE;
                    mem_rd_en <= 1'b1;
                    if (mem_ready) begin
                        mem_rd_en <= 1'b0;
                        state <= HD_DONE;
                    end
                end

                HD_ASSOC_WRITE: begin
                    // Write to associative cache
                    mem_addr  <= ADDR_HD_CROSSBAR_BASE;
                    mem_wr_en <= 1'b1;
                    mem_wr_data <= work_out[word_idx[4:0]];
                    if (mem_ready) begin
                        mem_wr_en <= 1'b0;
                        if (word_idx >= HD_BANKS - 1) begin
                            state <= HD_DONE;
                        end else begin
                            word_idx <= word_idx + 1;
                        end
                    end
                end

                HD_DONE: begin
                    done         <= 1'b1;
                    result_valid <= 1'b1;
                    status       <= STATUS_DONE;
                    // Clear accumulator
                    for (int b = 0; b < HD_BANKS; b++) begin
                        for (int i = 0; i < 32; i++)
                            accumulator[b][i] <= 18'sd0;
                    end
                    state <= HD_IDLE;
                end
            endcase
        end
    end

    // =========================================================================
    // Output assignment
    // =========================================================================
    genvar g_out;
    generate
        for (g_out = 0; g_out < HD_BANKS; g_out++) begin : gen_out
            assign result_out[g_out] = (state == HD_DONE) ? work_out[g_out] : bundle_result[g_out];
        end
    endgenerate

endmodule
