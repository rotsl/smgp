// =============================================================================
// Testbench: HD Engine
// =============================================================================
`timescale 1ns / 1ps

module tb_hd_engine;

    import smgp_isa_pkg::*;

    localparam int DATA_WIDTH = 32;
    localparam int ADDR_WIDTH = 16;
    localparam int HD_BANKS   = 16;

    logic clk, rst_n;
    logic [31:0] opcode;
    logic start, done;
    logic [3:0] status;
    logic [ADDR_WIDTH-1:0] mem_addr;
    logic mem_rd_en, mem_wr_en;
    logic [DATA_WIDTH-1:0] mem_rd_data, mem_wr_data;
    logic mem_ready;
    logic [DATA_WIDTH-1:0] vector_a [0:HD_BANKS-1], vector_b [0:HD_BANKS-1];
    logic [DATA_WIDTH-1:0] result_out [0:HD_BANKS-1];
    logic result_valid;
    logic [31:0] similarity_out;

    initial clk = 0;
    always #5 clk = ~clk;

    // Generate test vectors
    genvar g;
    generate
        for (g = 0; g < HD_BANKS; g++) begin : gen_vecs
            initial vector_a[g] = $random;
            initial vector_b[g] = $random;
        end
    endgenerate

    hd_engine #(
        .HD_BANKS(HD_BANKS),
        .FRAC_BITS(24),
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) dut (.*);

    assign mem_rd_data = 32'b0;
    assign mem_ready = 1'b1;

    initial begin
        $dumpfile("waves/tb_hd.vcd");
        $dumpvars(0, tb_hd_engine);

        rst_n = 0; start = 0; opcode = 0;
        #20; rst_n = 1; #10;

        // Test 1: Bind operation
        $display("[TEST] HD Bind operation...");
        opcode = instr_to_bits(make_instr(OPC_HD, SUB_HD_BIND, FLAG_START, 16'd0));
        start = 1; #10; start = 0;
        wait(done); #10;
        $display("[TEST] Bind done. Status: %b", status);

        // Test 2: Generate random vectors
        $display("[TEST] HD Generate...");
        opcode = instr_to_bits(make_instr(OPC_HD, SUB_HD_GENERATE, FLAG_START, 16'd0));
        start = 1; #10; start = 0;
        wait(done); #10;
        $display("[TEST] Generate done.");

        // Test 3: Similarity
        $display("[TEST] HD Similarity...");
        opcode = instr_to_bits(make_instr(OPC_HD, SUB_HD_SIMILARITY, FLAG_START, 16'd0));
        start = 1; #10; start = 0;
        wait(done); #10;
        $display("[TEST] Similarity = %d", similarity_out);

        $display("[TEST] All HD tests complete.");
        $finish;
    end

    initial begin #10000; $display("[ERROR] Timeout"); $finish; end

endmodule
