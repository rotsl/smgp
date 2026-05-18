// =============================================================================
// Graph DMA Engine
// =============================================================================
// DMA engine specialized for traversing graph data structures in memory.
// Handles scatter-gather operations for:
//   - Loading CSR-format sparse matrices
//   - Streaming adjacency lists for spectral operations
//   - Coalescing HD vector reads/writes
//
// The engine issues AXI4-Stream transactions to the HBM controller and
// manages address translation between virtual graph IDs and physical
// memory addresses.
//
// References:
//   - Cummings, C.E. (2008). "Clock Domain Crossing (CDC) Design & Verification
//     Using SystemVerilog." SNUG.
// =============================================================================

`timescale 1ns / 1ps

module graph_dma #(
    parameter int ADDR_WIDTH  = 32,
    parameter int DATA_WIDTH  = 64,
    parameter int ID_WIDTH    = 4,
    parameter int MAX_DESC    = 256    // Max DMA descriptors in queue
)(
    input  logic                        clk,
    input  logic                        rst_n,

    // Host control interface
    input  logic                        dma_start,
    input  logic [1:0]                  dma_type,  // 0:load, 1:store, 2:scatter, 3:gather
    input  logic [ADDR_WIDTH-1:0]       src_addr,
    input  logic [ADDR_WIDTH-1:0]       dst_addr,
    input  logic [31:0]                 byte_count,
    output logic                        dma_done,
    output logic                        dma_error,

    // HBM interface (AXI4-Stream)
    output logic                        hbm_rd_req,
    output logic                        hbm_wr_req,
    output logic [ADDR_WIDTH-1:0]       hbm_addr,
    output logic [7:0]                  hbm_burst_len,
    output logic [ID_WIDTH-1:0]         hbm_id,
    input  logic                        hbm_ready,
    input  logic                        hbm_done,

    // Local data interface (to compute engines)
    output logic [DATA_WIDTH-1:0]       local_data_out,
    output logic                        local_valid_out,
    input  logic [DATA_WIDTH-1:0]       local_data_in,
    input  logic                        local_valid_in
);

    // =========================================================================
    // DMA descriptor queue
    // =========================================================================
    typedef struct packed {
        logic [ADDR_WIDTH-1:0] src;
        logic [ADDR_WIDTH-1:0] dst;
        logic [31:0]          bytes;
        logic [1:0]           xfer_type;
    } dma_desc_t;

    dma_desc_t desc_queue [0:MAX_DESC-1];
    logic [$clog2(MAX_DESC)-1:0] desc_head, desc_tail;
    logic [31:0] transferred;
    logic [31:0] remaining;
    logic [31:0] dma_burst_bytes; // Temp for DMA_TRANSFER (module-level for Verilator)

    // =========================================================================
    // State machine
    // =========================================================================
    typedef enum logic [2:0] {
        DMA_IDLE    = 3'd0,
        DMA_ISSUE   = 3'd1,
        DMA_TRANSFER = 3'd2,
        DMA_COMPLETE = 3'd3,
        DMA_ERROR   = 3'd4
    } dma_state_e;

    dma_state_e state;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= DMA_IDLE;
            dma_done      <= 1'b0;
            dma_error     <= 1'b0;
            hbm_rd_req    <= 1'b0;
            hbm_wr_req    <= 1'b0;
            hbm_addr      <= '0;
            hbm_burst_len <= 8'd16;
            hbm_id        <= '0;
            local_valid_out <= 1'b0;
            local_data_out  <= '0;
            transferred    <= 32'b0;
            remaining      <= 32'b0;
            desc_head      <= '0;
            desc_tail      <= '0;
        end else begin
            dma_done <= 1'b0;
            local_valid_out <= 1'b0;

            case (state)
                DMA_IDLE: begin
                    if (dma_start) begin
                        remaining   <= byte_count;
                        transferred <= 32'b0;
                        hbm_addr    <= src_addr;
                        hbm_id      <= desc_head[ID_WIDTH-1:0];
                        desc_tail   <= desc_tail + 1;
                        state       <= DMA_ISSUE;
                    end
                end

                DMA_ISSUE: begin
                    // Issue read or write request to HBM
                    if (dma_type == 2'b00 || dma_type == 2'b10) begin
                        // Load or scatter: read from HBM
                        hbm_rd_req    <= 1'b1;
                        hbm_wr_req    <= 1'b0;
                    end else begin
                        // Store or gather: write to HBM
                        hbm_rd_req    <= 1'b0;
                        hbm_wr_req    <= 1'b1;
                    end
                    hbm_burst_len <= (remaining > 1024) ? 8'd16 : 8'd1;
                    if (hbm_ready) begin
                        hbm_rd_req <= 1'b0;
                        hbm_wr_req <= 1'b0;
                        state      <= DMA_TRANSFER;
                    end
                end

                DMA_TRANSFER: begin
                    // Transfer data
                    if (hbm_done) begin
                        dma_burst_bytes = (remaining > 1024) ? 32'd1024 : remaining;
                        transferred <= transferred + dma_burst_bytes;
                        remaining   <= remaining - dma_burst_bytes;
                        hbm_addr    <= hbm_addr + dma_burst_bytes;

                        if (remaining <= dma_burst_bytes) begin
                            state <= DMA_COMPLETE;
                        end else begin
                            state <= DMA_ISSUE;
                        end
                    end
                end

                DMA_COMPLETE: begin
                    dma_done  <= 1'b1;
                    desc_head <= desc_head + 1;
                    state     <= DMA_IDLE;
                end

                DMA_ERROR: begin
                    dma_error <= 1'b1;
                    dma_done  <= 1'b1;
                    state     <= DMA_IDLE;
                end
            endcase
        end
    end

endmodule
