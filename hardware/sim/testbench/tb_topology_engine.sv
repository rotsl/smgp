`timescale 1ns / 1ps
module tb_topology_engine;

    import smgp_isa_pkg::*;

    logic clk, rst_n;
    logic [31:0] opcode;
    logic start, done;
    logic [3:0] status;
    logic [19:0] mem_addr;
    logic mem_rd_en, mem_wr_en;
    logic [31:0] mem_rd_data, mem_wr_data;
    logic mem_ready;
    logic [31:0] barcode_birth, barcode_death;
    logic barcode_valid;
    logic [31:0] wasserstein_distance;
    logic stability_ok;

    initial clk = 0;
    always #5 clk = ~clk;

    topology_engine #(
        .MAX_NODES(64),
        .MAX_EDGES(1024),
        .FRAC_BITS(24),
        .DATA_WIDTH(32),
        .ADDR_WIDTH(20)
    ) dut (.*);

    assign mem_rd_data = 32'h0000_0001; // dummy distances
    assign mem_ready = 1'b1;

    initial begin
        $dumpfile("waves/tb_topology.vcd");
        $dumpvars(0, tb_topology_engine);

        rst_n = 0; start = 0; opcode = 0;
        #20; rst_n = 1; #10;

        $display("[TEST] Build filtration...");
        opcode = instr_to_bits(make_instr(OPC_TOPOLOGY, SUB_BUILD_FILTRATION, FLAG_START, 16'd8));
        start = 1; #10; start = 0;
        wait(done); #10;
        $display("[TEST] Filtration done. Status: %b", status);

        $display("[TEST] All topology tests complete.");
        $finish;
    end

    initial begin #10000; $display("[ERROR] Timeout"); $finish; end
endmodule
