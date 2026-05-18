`timescale 1ns / 1ps
module tb_system_end_to_end;

    logic clk_sys, clk_hbm, rst_n_sys;
    // AXI4-Lite signals
    logic [15:0] s_axil_awaddr, s_axil_araddr;
    logic s_axil_awvalid, s_axil_wvalid, s_axil_bready;
    logic [31:0] s_axil_wdata;
    logic [3:0] s_axil_wstrb;
    logic s_axil_arvalid, s_axil_rready;
    logic s_axil_awready, s_axil_wready, s_axil_bvalid, s_axil_arready;
    logic [1:0] s_axil_bresp, s_axil_rresp;
    logic [31:0] s_axil_rdata;
    logic s_axil_rvalid;
    // Memory
    logic [15:0] ext_mem_addr;
    logic ext_mem_rd_en, ext_mem_wr_en;
    logic [31:0] ext_mem_rd_data, ext_mem_wr_data;
    logic ext_mem_ready;
    logic irq_to_host;
    logic [3:0] status_led;

    initial clk_sys = 0;
    always #4 clk_sys = ~clk_sys; // 125 MHz
    initial clk_hbm = 0;
    always #2 clk_hbm = ~clk_hbm; // 250 MHz HBM clock

    smgp_system #(
        .HD_DIM_WORDS(16),
        .MAX_NODES(64),
        .FRAC_BITS(24),
        .DATA_WIDTH(32),
        .ADDR_WIDTH(16)
    ) dut (.*);

    assign ext_mem_rd_data = 32'b0;
    assign ext_mem_ready = 1'b1;

    // AXI write task
    task axi_write(input [15:0] addr, input [31:0] data);
        begin
            @(posedge clk_sys);
            s_axil_awaddr = addr;
            s_axil_awvalid = 1;
            s_axil_wdata = data;
            s_axil_wvalid = 1;
            s_axil_wstrb = 4'hF;
            @(posedge clk_sys);
            while (!s_axil_awready) @(posedge clk_sys);
            while (!s_axil_wready) @(posedge clk_sys);
            s_axil_awvalid = 0;
            s_axil_wvalid = 0;
            @(posedge clk_sys);
            while (!s_axil_bvalid) @(posedge clk_sys);
            s_axil_bready = 1;
            @(posedge clk_sys);
            s_axil_bready = 0;
        end
    endtask

    // AXI read task
    task axi_read(input [15:0] addr, output [31:0] data);
        begin
            @(posedge clk_sys);
            s_axil_araddr = addr;
            s_axil_arvalid = 1;
            @(posedge clk_sys);
            while (!s_axil_arready) @(posedge clk_sys);
            s_axil_arvalid = 0;
            while (!s_axil_rvalid) @(posedge clk_sys);
            data = s_axil_rdata;
            s_axil_rready = 1;
            @(posedge clk_sys);
            s_axil_rready = 0;
        end
    endtask

    logic [31:0] read_data;

    initial begin
        $dumpfile("waves/tb_system_e2e.vcd");
        $dumpvars(0, tb_system_end_to_end);

        rst_n_sys = 0;
        s_axil_awvalid = 0; s_axil_wvalid = 0; s_axil_arvalid = 0;
        s_axil_bready = 0; s_axil_rready = 0;
        #40; rst_n_sys = 1; #20;

        $display("=== SMGP System End-to-End Test ===");

        // Configure
        $display("[E2E] Writing config...");
        axi_write(16'h0010, 32'd10000); // HD_DIM
        axi_write(16'h0014, 32'd4096);  // MAX_NODES
        axi_write(16'h001C, 32'd1);     // IRQ enable

        // Issue a spectral instruction
        $display("[E2E] Issuing SPECTRAL_COMPUTE_LAP instruction...");
        // Opcode=2, sub=0, flags=1(start), operand=4(nodes)
        axi_write(16'h0008, 32'h21010004);

        // Wait and check status
        repeat(50) @(posedge clk_sys);
        axi_read(16'h0004, read_data);
        $display("[E2E] Status: %b", read_data[3:0]);

        // Issue HD bind
        $display("[E2E] Issuing HD_BIND instruction...");
        axi_write(16'h0008, 32'h31010000);
        repeat(50) @(posedge clk_sys);
        axi_read(16'h0004, read_data);
        $display("[E2E] Status: %b", read_data[3:0]);

        $display("=== End-to-End Test Complete ===");
        $finish;
    end

    initial begin #50000; $display("[ERROR] E2E Timeout"); $finish; end
endmodule
