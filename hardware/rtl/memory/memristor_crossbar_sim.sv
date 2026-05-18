// =============================================================================
// Memristor Crossbar Array — Behavioral Model
// =============================================================================
// Models a non-volatile resistive crossbar memory array organized as tiles.
// Each tile is TILE_ROWS x TILE_COLS cells, where each cell stores a
// multi-bit conductance value. The crossbar supports:
//   - Cell-level read/write of conductance
//   - Analog matrix-vector multiply (MVM): y = G * x
//   - State persistence via $saveh/$readmemh
//
// This behavioral model can be replaced with a synthesized analog peripheral
// in an actual ASIC flow. For FPGA prototyping, the MVM is implemented
// using DSP slices.
//
// References:
//   - Ielmini, D. & Wong, H.-S.P. (2018). "In-memory Computing with Resistive
//     Switching Devices." Nature Electronics, 1(6), 333–343.
//   - Xia, Q. & Yang, J.J. (2019). "Memristive Crossbar Arrays for Brain-
//     Inspired Computing." Nature Materials, 18(4), 309–323.
//   - Hu, M., et al. (2014). "Memristor-Based Analog Computation." Neural
//     Networks, 56, 1–6.
// =============================================================================

`timescale 1ns / 1ps

module memristor_crossbar_sim #(
    parameter int TILE_ROWS      = 1024,    // Number of wordlines
    parameter int TILE_COLS      = 1024,    // Number of bitlines
    parameter int CONDUCTANCE_BITS = 8,     // Bits per conductance value
    parameter int NUM_TILES       = 1,      // Number of tiles (for scalability)
    parameter int ADDR_WIDTH      = 20,
    parameter int DATA_WIDTH      = 32,
    parameter string STATE_FILE   = "crossbar_state.hex" // Persistence file
)(
    input  logic                        clk,
    input  logic                        rst_n,

    // Control interface
    input  logic                        read_en,
    input  logic                        write_en,
    input  logic                        mvm_en,        // Matrix-vector multiply enable
    input  logic [ADDR_WIDTH-1:0]       row_addr,      // Wordline address
    input  logic [ADDR_WIDTH-1:0]       col_addr,      // Bitline address
    input  logic [CONDUCTANCE_BITS-1:0] conductance_in, // Value to write
    output logic [CONDUCTANCE_BITS-1:0] conductance_out,// Value read
    output logic                        data_valid,

    // Vector input for MVM (applied to bitlines)
    input  logic [DATA_WIDTH-1:0]       mvm_vector [0:TILE_COLS/DATA_WIDTH*32-1],
    // MVM result output (one per wordline, streamed out)
    output logic [DATA_WIDTH-1:0]       mvm_result,
    output logic                        mvm_result_valid,

    // Status
    output logic                        busy
);

    // =========================================================================
    // Crossbar conductance storage
    // Each cell stores an integer conductance (0 to 2^G-1).
    // For FPGA, this maps to BRAM. For ASIC, to actual memristor cells.
    // =========================================================================
    logic [CONDUCTANCE_BITS-1:0] conductance [0:TILE_ROWS-1][0:TILE_COLS-1];

    // =========================================================================
    // MVM pipeline registers
    // =========================================================================
    logic [31:0] accumulator;
    logic [15:0] mvm_row_idx;
    logic [15:0] mvm_col_idx;
    logic        mvm_active;

    // =========================================================================
    // Persistence: save state on reset deassertion
    // =========================================================================
    initial begin
        $readmemh(STATE_FILE, conductance);
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // Save state for persistence
            $writememh(STATE_FILE, conductance);
        end
    end

    // =========================================================================
    // Single-cell read
    // =========================================================================
    always_ff @(posedge clk) begin
        data_valid <= 1'b0;
        if (read_en && !mvm_en && !write_en) begin
            if (row_addr < TILE_ROWS && col_addr < TILE_COLS)
                conductance_out <= conductance[row_addr][col_addr];
            data_valid <= 1'b1;
        end
    end

    // =========================================================================
    // Single-cell write
    // =========================================================================
    always_ff @(posedge clk) begin
        if (write_en && !read_en && !mvm_en) begin
            if (row_addr < TILE_ROWS && col_addr < TILE_COLS)
                conductance[row_addr][col_addr] <= conductance_in;
            data_valid <= 1'b1;
        end
    end

    // =========================================================================
    // Analog Matrix-Vector Multiply: y[row] = sum(G[row][col] * x[col])
    // For FPGA: uses DSP-based MAC units
    // For ASIC: actual analog current accumulation
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accumulator      <= 32'b0;
            mvm_row_idx      <= 16'b0;
            mvm_col_idx      <= 16'b0;
            mvm_active       <= 1'b0;
            mvm_result       <= 32'b0;
            mvm_result_valid <= 1'b0;
            busy             <= 1'b0;
        end else begin
            mvm_result_valid <= 1'b0;

            if (mvm_en) begin
                busy <= 1'b1;
                mvm_active <= 1'b1;
            end

            if (mvm_active) begin
                // Process one column per cycle (pipelined)
                if (mvm_col_idx < TILE_COLS) begin
                    logic [CONDUCTANCE_BITS:0] g_val;
                    logic [31:0] x_val;
                    logic [31:0] product;
                    g_val = {1'b0, conductance[mvm_row_idx][mvm_col_idx]};
                    // Extract input vector element (32-bit slice)
                    x_val = mvm_vector[mvm_col_idx / 32];
                    x_val = x_val >> (mvm_col_idx % 32);
                    x_val = x_val & 32'b1;
                    // MAC
                    product = g_val * x_val;
                    accumulator <= accumulator + product;
                    mvm_col_idx <= mvm_col_idx + 1;
                end else begin
                    // Row complete: emit result
                    mvm_result       <= accumulator;
                    mvm_result_valid <= 1'b1;
                    accumulator      <= 32'b0;
                    mvm_col_idx      <= 16'b0;
                    mvm_row_idx      <= mvm_row_idx + 1;
                    if (mvm_row_idx >= TILE_ROWS - 1) begin
                        mvm_active <= 1'b0;
                        busy       <= 1'b0;
                    end
                end
            end
        end
    end

endmodule
