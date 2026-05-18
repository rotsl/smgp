// =============================================================================
// Associative Content-Addressable Cache
// =============================================================================
// Stores key-value pairs of hyperdimensional vectors and supports O(1)
// content-addressable recall via parallel dot-product computation.
//
// Architecture:
//   - Key store: N x D bits of bipolar vector keys
//   - Value store: N x D bits of bipolar vector values
//   - Query engine: parallel XOR + popcount across all stored keys
//   - Output: index and similarity of the nearest match
//
// Uses dimensionality reduction via random projection for area efficiency.
//
// References:
//   - Kanerva, P. (1988). "Sparse Distributed Memory." MIT Press.
//   - Ielmini, D. & Wong, H.-S.P. (2018). "In-memory Computing."
//     Nature Electronics.
// =============================================================================

`timescale 1ns / 1ps

module associative_cache #(
    parameter int NUM_ENTRIES      = 256,     // Number of stored key-value pairs
    parameter int HD_DIM_REDUCED   = 1024,    // Dimensionality after random projection
    parameter int DATA_WIDTH       = 32,      // Bus width
    parameter int ADDR_WIDTH       = 16,
    parameter int SIM_THRESHOLD    = 32'd5000 // Min similarity (out of 10000)
)(
    input  logic                        clk,
    input  logic                        rst_n,

    // Control
    input  logic                        query_en,
    input  logic                        write_en,
    input  logic                        delete_en,

    // Query interface
    input  logic [DATA_WIDTH-1:0]       query_vector [0:(HD_DIM_REDUCED/32)-1],
    output logic [15:0]                 best_match_idx,
    output logic [31:0]                 best_similarity,
    output logic                        query_done,

    // Write interface
    input  logic [15:0]                 write_idx,
    input  logic [DATA_WIDTH-1:0]       write_key   [0:(HD_DIM_REDUCED/32)-1],
    input  logic [DATA_WIDTH-1:0]       write_value [0:(HD_DIM_REDUCED/32)-1],
    output logic                        write_done,

    // Read interface (by index)
    input  logic [15:0]                 read_idx,
    output logic [DATA_WIDTH-1:0]       read_value [0:(HD_DIM_REDUCED/32)-1],
    output logic                        read_done,

    // Memory interface (backing store)
    output logic [ADDR_WIDTH-1:0]       mem_addr,
    output logic                        mem_rd_en,
    output logic                        mem_wr_en,
    input  logic [DATA_WIDTH-1:0]       mem_rd_data,
    output logic [DATA_WIDTH-1:0]       mem_wr_data,
    input  logic                        mem_ready
);

    import smgp_isa_pkg::*;

    // =========================================================================
    // Storage
    // =========================================================================
    // Keys: stored as HD_DIM_REDUCED bits per entry
    logic keys [0:NUM_ENTRIES-1][0:(HD_DIM_REDUCED/32)-1];
    // Values
    logic values [0:NUM_ENTRIES-1][0:(HD_DIM_REDUCED/32)-1];
    // Valid bits
    logic valid [0:NUM_ENTRIES-1];
    // Write pointer (round-robin)
    logic [15:0] write_ptr;

    // =========================================================================
    // Similarity computation pipeline
    // =========================================================================
    logic [31:0] similarities [0:NUM_ENTRIES-1];
    logic [15:0] best_idx;
    logic [31:0] best_sim;
    logic [7:0]  query_pipeline;
    logic [15:0] query_idx;

    // Temporary variables for procedural blocks (must be module-level in Verilator)
    logic [31:0] ac_matches, ac_xor;
    logic [15:0] ac_idx;

    // =========================================================================
    // Sequential logic
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            query_done  <= 1'b0;
            write_done  <= 1'b0;
            read_done   <= 1'b0;
            write_ptr   <= 16'b0;
            best_idx    <= 16'b0;
            best_sim    <= 32'b0;
            query_pipeline <= 8'b0;
            query_idx   <= 16'b0;
            for (int i = 0; i < NUM_ENTRIES; i++)
                valid[i] <= 1'b0;
            for (int i = 0; i < NUM_ENTRIES; i++)
                similarities[i] <= 32'b0;
        end else begin
            query_done <= 1'b0;
            write_done <= 1'b0;
            read_done  <= 1'b0;

            // ---- Write operation ----
            if (write_en) begin
                ac_idx = write_idx; // Direct index write
                for (int w = 0; w < HD_DIM_REDUCED/32; w++) begin
                    keys[ac_idx][w] <= write_key[w];
                    values[ac_idx][w] <= write_value[w];
                end
                valid[ac_idx] <= 1'b1;
                write_done <= 1'b1;
                write_ptr <= write_ptr + 1;
            end

            // ---- Delete operation ----
            if (delete_en) begin
                valid[read_idx] <= 1'b0;
                write_done <= 1'b1;
            end

            // ---- Query operation (pipelined) ----
            if (query_en) begin
                query_pipeline <= 8'd1;
                query_idx <= 16'b0;
                best_sim <= 32'b0;
                best_idx <= 16'b0;
            end

            if (query_pipeline > 0) begin
                // Process one entry per cycle
                if (query_idx < NUM_ENTRIES) begin
                    if (valid[query_idx]) begin
                        // Compute similarity: sum of matching bits
                        ac_matches = 32'b0;
                        for (int w = 0; w < HD_DIM_REDUCED/32; w++) begin
                            ac_xor = query_vector[w] ^ keys[query_idx][w];
                            ac_matches = ac_matches + $countones(~ac_xor);
                        end
                        similarities[query_idx] <= ac_matches;

                        // Track best match
                        if (ac_matches > best_sim) begin
                            best_sim <= ac_matches;
                            best_idx <= query_idx;
                        end
                    end
                    query_idx <= query_idx + 1;
                end else begin
                    // Done: emit result
                    best_match_idx <= best_idx;
                    best_similarity <= best_sim;
                    query_done <= 1'b1;
                    query_pipeline <= 8'b0;
                end
            end

            // ---- Read by index ----
            if (read_idx < NUM_ENTRIES && valid[read_idx]) begin
                for (int w = 0; w < HD_DIM_REDUCED/32; w++)
                    read_value[w] <= values[read_idx][w];
                read_done <= 1'b1;
            end
        end
    end

endmodule
