`timescale 1ns / 1ps
module tb_graph_rewrite;

    import smgp_isa_pkg::*;

    logic clk, rst_n;
    logic [31:0] opcode;
    logic start, done;
    logic [3:0] status;
    logic [15:0] mem_addr;
    logic mem_rd_en, mem_wr_en;
    logic [31:0] mem_rd_data, mem_wr_data;
    logic mem_ready;
    logic [15:0] pattern_nodes [0:7];
    logic [7:0] pattern_node_count;
    logic [31:0] pattern_edges [0:63]; // 8x8 pattern node adjacency
    logic [7:0]  pattern_edge_count;
    logic [15:0] match_out [0:7];
    logic match_valid;
    logic [7:0] match_count;
    logic [31:0] proof_trace_push;
    logic proof_trace_valid;

    initial clk = 0;
    always #5 clk = ~clk;

    graph_rewrite_engine #(
        .MAX_GRAPH_NODES(256),
        .MAX_PATTERN_NODES(8),
        .MAX_EDGES(1024),
        .DATA_WIDTH(32),
        .ADDR_WIDTH(16)
    ) dut (.*);

    assign mem_rd_data = 32'b0;
    assign mem_ready = 1'b1;

    initial begin
        $dumpfile("waves/tb_graph_rewrite.vcd");
        $dumpvars(0, tb_graph_rewrite);

        rst_n = 0; start = 0; opcode = 0;
        #20; rst_n = 1; #10;

        // Pattern: 2 nodes with a "parent_of" edge
        pattern_nodes[0] = 16'd0;
        pattern_nodes[1] = 16'd1;
        pattern_node_count = 8'd3;

        $display("[TEST] DPO Pattern Match...");
        opcode = instr_to_bits(make_instr(OPC_REWRITE, SUB_DPO_MATCH, FLAG_START, 16'd0));
        start = 1; #10; start = 0;
        wait(done); #10;
        $display("[TEST] Match done. Status: %b, Matches: %d", status, match_count);

        $display("[TEST] All rewrite tests complete.");
        $finish;
    end

    initial begin #10000; $display("[ERROR] Timeout"); $finish; end
endmodule
