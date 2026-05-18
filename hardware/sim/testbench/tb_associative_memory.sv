`timescale 1ns / 1ps
module tb_associative_memory;

    logic clk, rst_n;
    logic query_en, write_en, delete_en;
    logic [31:0] query_vector [0:31], write_key [0:31], write_value [0:31];
    logic [15:0] best_match_idx, write_idx, read_idx;
    logic [31:0] best_similarity;
    logic query_done, write_done, read_done;
    logic [31:0] read_value [0:31];
    logic [15:0] mem_addr;
    logic mem_rd_en, mem_wr_en;
    logic [31:0] mem_rd_data, mem_wr_data;
    logic mem_ready;

    initial clk = 0;
    always #5 clk = ~clk;

    associative_cache #(
        .NUM_ENTRIES(64),
        .HD_DIM_REDUCED(1024),
        .DATA_WIDTH(32),
        .ADDR_WIDTH(16)
    ) dut (.*);

    assign mem_rd_data = 32'b0;
    assign mem_ready = 1'b1;

    // Init vectors
    initial begin
        for (int i = 0; i < 32; i++) begin
            query_vector[i] = $random;
            write_key[i] = $random;
            write_value[i] = $random;
        end
    end

    initial begin
        $dumpfile("waves/tb_assoc_mem.vcd");
        $dumpvars(0, tb_associative_memory);

        rst_n = 0; query_en = 0; write_en = 0; delete_en = 0;
        #20; rst_n = 1; #10;

        // Write a key-value pair
        $display("[TEST] Associative write...");
        write_en = 1;
        write_idx = 16'd0;
        #10;
        write_en = 0;
        wait(write_done);
        $display("[TEST] Write done.");

        // Query
        $display("[TEST] Associative query...");
        query_en = 1;
        #10;
        query_en = 0;
        wait(query_done);
        $display("[TEST] Query done. Best match: %d, similarity: %d", best_match_idx, best_similarity);

        $display("[TEST] All associative memory tests complete.");
        $finish;
    end

    initial begin #10000; $display("[ERROR] Timeout"); $finish; end
endmodule
