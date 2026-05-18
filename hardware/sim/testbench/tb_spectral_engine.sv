// =============================================================================
// Testbench: Spectral Engine
// =============================================================================
`timescale 1ns / 1ps

module tb_spectral_engine;

    import smgp_isa_pkg::*;

    // Parameters
    localparam int DATA_WIDTH = 32;
    localparam int ADDR_WIDTH = 16;
    localparam int FRAC_BITS  = 24;
    localparam int MAX_NODES  = 16;

    // Signals
    logic clk, rst_n;
    logic [31:0] opcode;
    logic start, done;
    logic [3:0] status;
    logic [ADDR_WIDTH-1:0] mem_addr;
    logic mem_rd_en, mem_wr_en;
    logic [DATA_WIDTH-1:0] mem_rd_data, mem_wr_data;
    logic mem_ready;
    logic [DATA_WIDTH-1:0] result_out;
    logic result_valid;

    // Simple memory model
    logic [DATA_WIDTH-1:0] mem [0:65535];
    logic mem_rd_delay;

    // Clock generation
    initial clk = 0;
    always #5 clk = ~clk;

    // DUT
    spectral_engine #(
        .MAX_NODES(MAX_NODES),
        .MAX_NNZ(256),
        .FRAC_BITS(FRAC_BITS),
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) dut (.*);

    // Memory model
    always_ff @(posedge clk) begin
        mem_rd_delay <= mem_rd_en;
        if (mem_wr_en && mem_ready)
            mem[mem_addr] <= mem_wr_data;
    end
    assign mem_rd_data = mem_rd_delay ? mem[mem_addr] : 32'b0;
    assign mem_ready = 1'b1;

    // Test sequence
    initial begin
        $dumpfile("waves/tb_spectral.vcd");
        $dumpvars(0, tb_spectral_engine);

        // Reset
        rst_n = 0;
        start = 0;
        opcode = 0;
        #20;
        rst_n = 1;
        #10;

        // Test 1: Compute Laplacian on a 4-node graph
        $display("[TEST] Computing Laplacian for 4-node graph...");
        opcode = instr_to_bits(make_instr(OPC_SPECTRAL, SUB_COMPUTE_LAP, FLAG_START, 16'd4));
        start = 1;
        #10;
        start = 0;

        // Wait for done
        wait(done);
        #10;
        $display("[TEST] Spectral engine done. Status: %b", status);

        if (status == STATUS_DONE)
            $display("[PASS] Laplacian computation completed.");
        else
            $display("[FAIL] Laplacian computation failed.");

        #50;

        // Test 2: Chebyshev convolution
        $display("[TEST] Chebyshev convolution (order 3)...");
        opcode = instr_to_bits(make_instr(OPC_SPECTRAL, SUB_SPECTRAL_CONV, FLAG_START, 16'd3));
        start = 1;
        #10;
        start = 0;
        wait(done);
        #10;
        $display("[TEST] Chebyshev done. Status: %b", status);

        #50;
        $display("[TEST] All spectral tests complete.");
        $finish;
    end

    // Timeout
    initial begin
        #10000;
        $display("[ERROR] Simulation timeout!");
        $finish;
    end

endmodule
