// =============================================================================
// SMGP System — Top-Level System Integration
// =============================================================================
// Wraps the SMGP core with clock domain crossing, reset logic, and
// configuration registers. Provides a clean interface for FPGA
// integration (AXI4-Lite for control, AXI4-Stream for data).
//
// References:
//   - Xilinx (2022). "UG583: Zynq UltraScale+ MPSoC Technical Reference Manual."
//   - Cummings, C.E. (2008). "Clock Domain Crossing Design & Verification."
//     SNUG.
// =============================================================================

`timescale 1ns / 1ps

module smgp_system #(
    parameter int HD_DIM_WORDS   = 313,
    parameter int MAX_NODES      = 4096,
    parameter int FRAC_BITS      = 24,
    parameter int DATA_WIDTH     = 32,
    parameter int ADDR_WIDTH     = 16
)(
    // Clock and reset
    input  logic                        clk_sys,       // System clock (e.g., 250 MHz)
    input  logic                        clk_hbm,       // HBM clock (if applicable)
    input  logic                        rst_n_sys,     // System reset (active low)

    // AXI4-Lite Slave (Host Control)
    input  logic [ADDR_WIDTH-1:0]       s_axil_awaddr,
    input  logic                        s_axil_awvalid,
    output logic                        s_axil_awready,
    input  logic [31:0]                 s_axil_wdata,
    input  logic [3:0]                  s_axil_wstrb,
    input  logic                        s_axil_wvalid,
    output logic                        s_axil_wready,
    output logic [1:0]                  s_axil_bresp,
    output logic                        s_axil_bvalid,
    input  logic                        s_axil_bready,
    input  logic [ADDR_WIDTH-1:0]       s_axil_araddr,
    input  logic                        s_axil_arvalid,
    output logic                        s_axil_arready,
    output logic [31:0]                 s_axil_rdata,
    output logic [1:0]                  s_axil_rresp,
    output logic                        s_axil_rvalid,
    input  logic                        s_axil_rready,

    // External memory interface
    output logic [ADDR_WIDTH-1:0]       ext_mem_addr,
    output logic                        ext_mem_rd_en,
    output logic                        ext_mem_wr_en,
    input  logic [DATA_WIDTH-1:0]       ext_mem_rd_data,
    output logic [DATA_WIDTH-1:0]       ext_mem_wr_data,
    input  logic                        ext_mem_ready,

    // Interrupt output
    output logic                        irq_to_host,

    // Status LEDs (for FPGA debug)
    output logic [3:0]                  status_led
);

    import smgp_isa_pkg::*;

    // =========================================================================
    // Internal signals
    // =========================================================================
    logic core_rst_n;
    logic [31:0] instruction_reg;
    logic instruction_valid;
    logic instruction_ready;
    logic [3:0]  core_status;

    // =========================================================================
    // Reset synchronization (2-stage synchronizer)
    // =========================================================================
    logic rst_sync_0, rst_sync_1;
    always_ff @(posedge clk_sys or negedge rst_n_sys) begin
        if (!rst_n_sys) begin
            rst_sync_0 <= 1'b0;
            rst_sync_1 <= 1'b0;
        end else begin
            rst_sync_0 <= 1'b1;
            rst_sync_1 <= rst_sync_0;
        end
    end
    assign core_rst_n = rst_sync_1;

    // =========================================================================
    // Configuration Register File (8 registers)
    // =========================================================================
    logic [31:0] config_regs [0:7];
    logic config_written;

    // Register addresses
    localparam logic [ADDR_WIDTH-1:0] REG_CTRL     = 16'h0000;
    localparam logic [ADDR_WIDTH-1:0] REG_STATUS   = 16'h0004;
    localparam logic [ADDR_WIDTH-1:0] REG_INSTR    = 16'h0008;
    localparam logic [ADDR_WIDTH-1:0] REG_RESULT   = 16'h000C;
    localparam logic [ADDR_WIDTH-1:0] REG_HD_DIM   = 16'h0010;
    localparam logic [ADDR_WIDTH-1:0] REG_MAX_NODES = 16'h0014;
    localparam logic [ADDR_WIDTH-1:0] REG_FRAC     = 16'h0018;
    localparam logic [ADDR_WIDTH-1:0] REG_IRQ_EN   = 16'h001C;

    // AXI4-Lite Write
    always_ff @(posedge clk_sys or negedge core_rst_n) begin
        if (!core_rst_n) begin
            for (int i = 0; i < 8; i++) config_regs[i] <= 32'b0;
            config_written <= 1'b0;
            instruction_valid <= 1'b0;
        end else begin
            config_written <= 1'b0;
            instruction_valid <= 1'b0;
            if (s_axil_awvalid && s_axil_awready && s_axil_wvalid && s_axil_wready) begin
                case (s_axil_awaddr[3:0])
                    4'h0: config_regs[0] <= s_axil_wdata; // CTRL
                    4'h2: begin
                        config_regs[2] <= s_axil_wdata; // INSTR
                        instruction_valid <= 1'b1;
                    end
                    4'h4: config_regs[4] <= s_axil_wdata; // HD_DIM
                    4'h5: config_regs[5] <= s_axil_wdata; // MAX_NODES
                    4'h6: config_regs[6] <= s_axil_wdata; // FRAC
                    4'h7: config_regs[7] <= s_axil_wdata; // IRQ_EN
                    default: ;
                endcase
                config_written <= 1'b1;
            end
        end
    end

    // AXI4-Lite read
    logic [31:0] read_data;
    always_comb begin
        case (s_axil_araddr[3:0])
            4'h0: read_data = config_regs[0];         // CTRL
            4'h1: read_data = {28'b0, core_status};   // STATUS
            4'h2: read_data = config_regs[2];         // INSTR
            4'h3: read_data = 32'b0;                  // RESULT (placeholder)
            4'h4: read_data = config_regs[4];         // HD_DIM
            4'h5: read_data = config_regs[5];         // MAX_NODES
            4'h6: read_data = config_regs[6];         // FRAC
            4'h7: read_data = config_regs[7];         // IRQ_EN
            default: read_data = 32'hDEAD_DEAD;
        endcase
    end

    // =========================================================================
    // AXI4-Lite handshake logic
    // =========================================================================
    // Write address channel
    assign s_axil_awready = 1'b1; // Always ready (single-cycle)
    // Write data channel
    assign s_axil_wready = 1'b1;
    // Write response
    always_ff @(posedge clk_sys or negedge core_rst_n) begin
        if (!core_rst_n) s_axil_bvalid <= 1'b0;
        else s_axil_bvalid <= config_written;
    end
    assign s_axil_bresp = 2'b00; // OKAY
    // Read address channel
    assign s_axil_arready = 1'b1;
    // Read data channel
    always_ff @(posedge clk_sys or negedge core_rst_n) begin
        if (!core_rst_n) begin
            s_axil_rvalid <= 1'b0;
            s_axil_rdata <= 32'b0;
        end else begin
            s_axil_rvalid <= s_axil_arvalid;
            s_axil_rdata  <= read_data;
            s_axil_rresp  <= 2'b00;
        end
    end

    // =========================================================================
    // Core instantiation
    // =========================================================================
    assign instruction_reg = config_regs[2];

    smgp_core #(
        .HD_DIM_WORDS(HD_DIM_WORDS),
        .MAX_NODES(MAX_NODES),
        .FRAC_BITS(FRAC_BITS),
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) u_core (
        .clk(clk_sys),
        .rst_n(core_rst_n),
        .host_instruction(instruction_reg),
        .host_valid(instruction_valid),
        .host_ready(instruction_ready),
        .core_status(core_status),
        .mem_addr(ext_mem_addr),
        .mem_rd_en(ext_mem_rd_en),
        .mem_wr_en(ext_mem_wr_en),
        .mem_rd_data(ext_mem_rd_data),
        .mem_wr_data(ext_mem_wr_data),
        .mem_ready(ext_mem_ready),
        .irq(irq_to_host)
    );

    // =========================================================================
    // Status LEDs
    // =========================================================================
    always_ff @(posedge clk_sys or negedge core_rst_n) begin
        if (!core_rst_n)
            status_led <= 4'b0000;
        else
            status_led <= core_status;
    end

endmodule
