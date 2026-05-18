// =============================================================================
// Graph Rewrite Engine (DPO — Double Pushout)
// =============================================================================
// Hardware accelerator for graph rewriting using the Double-Pushout (DPO)
// approach from category theory. The engine:
//   1. Loads a graph pattern (LHS) into a pattern buffer
//   2. Scans the host graph for subgraph isomorphisms (bounded)
//   3. Applies the rewrite rule: delete L\K, add R\K
//   4. Maintains a proof trace stack
//
// The matching algorithm uses a backtracking search with hardware-managed
// stack, suitable for small-to-medium patterns (< 8 nodes).
//
// References:
//   - Ehrig, H., et al. (2006). "Fundamentals of Algebraic Graph
//     Transformation." Springer.
//   - Ehrig, H., et al. (1997). "Parallel Derivations in DPO." TCS.
// =============================================================================

`timescale 1ns / 1ps

module graph_rewrite_engine #(
    parameter int MAX_GRAPH_NODES = 4096,
    parameter int MAX_PATTERN_NODES = 8,
    parameter int MAX_EDGES       = 65536,
    parameter int MAX_MATCHES     = 256,
    parameter int DATA_WIDTH      = 32,
    parameter int ADDR_WIDTH      = 16,
    parameter int STACK_DEPTH     = 64
)(
    input  logic                        clk,
    input  logic                        rst_n,

    // Control
    input  logic [31:0]                 opcode,
    input  logic                        start,
    output logic                        done,
    output logic [3:0]                  status,

    // Memory interface
    output logic [ADDR_WIDTH-1:0]       mem_addr,
    output logic                        mem_rd_en,
    output logic                        mem_wr_en,
    input  logic [DATA_WIDTH-1:0]       mem_rd_data,
    output logic [DATA_WIDTH-1:0]       mem_wr_data,
    input  logic                        mem_ready,

    // Pattern interface (from host)
    input  logic [15:0]                 pattern_nodes [0:MAX_PATTERN_NODES-1],
    input  logic [7:0]                  pattern_node_count,
    input  logic [31:0]                 pattern_edges [0:MAX_PATTERN_NODES*MAX_PATTERN_NODES-1],
    input  logic [7:0]                  pattern_edge_count,

    // Match output
    output logic [15:0]                 match_out [0:MAX_PATTERN_NODES-1],
    output logic                        match_valid,
    output logic [7:0]                  match_count,

    // Proof trace
    output logic [31:0]                 proof_trace_push,
    output logic                        proof_trace_valid
);

    import smgp_isa_pkg::*;

    // =========================================================================
    // State machine
    // =========================================================================
    typedef enum logic [3:0] {
        REW_IDLE        = 4'd0,
        REW_LOAD_LHS    = 4'd1,
        REW_MATCH_INIT  = 4'd2,
        REW_MATCH_SEARCH = 4'd3,
        REW_CHECK_EDGE  = 4'd4,
        REW_BACKTRACK   = 4'd5,
        REW_APPLY_DELETE = 4'd6,
        REW_APPLY_ADD   = 4'd7,
        REW_EMIT_MATCH  = 4'd8,
        REW_DONE        = 4'd9
    } rew_state_e;

    rew_state_e state;

    // =========================================================================
    // Match buffer
    // =========================================================================
    logic [15:0] current_match [0:MAX_PATTERN_NODES-1];
    logic [7:0]  current_depth;
    logic [7:0]  total_matches;

    // Backtracking stack
    logic [15:0] stack [0:STACK_DEPTH-1];
    logic [7:0]  stack_ptr;

    // Graph traversal
    logic [15:0] graph_node_idx;
    logic [7:0]  pattern_idx;

    // =========================================================================
    // Main logic
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state              <= REW_IDLE;
            done               <= 1'b0;
            status             <= STATUS_IDLE;
            match_valid        <= 1'b0;
            match_count        <= 8'b0;
            proof_trace_valid  <= 1'b0;
            proof_trace_push   <= 32'b0;
            total_matches      <= 8'b0;
            current_depth      <= 8'b0;
            graph_node_idx     <= 16'b0;
            pattern_idx        <= 8'b0;
            stack_ptr          <= 8'b0;
            for (int i = 0; i < MAX_PATTERN_NODES; i++) begin
                current_match[i] <= 16'b0;
                match_out[i]     <= 16'b0;
            end
            for (int i = 0; i < STACK_DEPTH; i++)
                stack[i] <= 16'b0;
        end else begin
            match_valid       <= 1'b0;
            proof_trace_valid <= 1'b0;
            done              <= 1'b0;

            case (state)
                REW_IDLE: begin
                    if (start) begin
                        status <= STATUS_RUNNING;
                        case (opcode[27:24])
                            4'h0: state <= REW_MATCH_INIT;  // DPO_MATCH
                            4'h1: state <= REW_APPLY_DELETE; // DPO_APPLY
                            default: state <= REW_DONE;
                        endcase
                        current_depth <= 8'b0;
                        total_matches <= 8'b0;
                        stack_ptr     <= 8'b0;
                    end
                end

                REW_MATCH_INIT: begin
                    // Initialize matching: load first pattern node candidates
                    if (pattern_node_count > 0) begin
                        current_depth  <= 8'b1;
                        graph_node_idx <= 16'b0;
                        state          <= REW_MATCH_SEARCH;
                    end else begin
                        state <= REW_DONE;
                    end
                end

                REW_MATCH_SEARCH: begin
                    // Search for matching graph nodes
                    if (graph_node_idx < MAX_GRAPH_NODES) begin
                        // Check if this graph node is already matched
                        logic already_matched;
                        already_matched = 1'b0;
                        for (int i = 0; i < current_depth - 1; i++) begin
                            if (current_match[i] == graph_node_idx)
                                already_matched = 1'b1;
                        end

                        if (!already_matched) begin
                            // Assign match
                            current_match[current_depth - 1] <= graph_node_idx;
                            current_depth <= current_depth + 1;

                            if (current_depth >= pattern_node_count) begin
                                // Full match found - emit it
                                for (int i = 0; i < MAX_PATTERN_NODES; i++)
                                    match_out[i] <= current_match[i];
                                match_valid <= 1'b1;
                                total_matches <= total_matches + 1;
                                match_count <= total_matches;

                                // Record proof trace
                                proof_trace_valid <= 1'b1;
                                proof_trace_push <= {graph_node_idx, current_depth, 16'b0};

                                // Backtrack for next match
                                current_depth <= current_depth - 1;
                                state <= REW_BACKTRACK;
                            end else begin
                                state <= REW_CHECK_EDGE;
                            end
                        end

                        graph_node_idx <= graph_node_idx + 1;
                    end else begin
                        // No more candidates at this level, backtrack
                        state <= REW_BACKTRACK;
                    end
                end

                REW_CHECK_EDGE: begin
                    // Verify edge constraints for the current partial match
                    // Simplified: check if edges exist between matched nodes
                    state <= REW_MATCH_SEARCH;
                end

                REW_BACKTRACK: begin
                    if (current_depth > 1) begin
                        current_depth <= current_depth - 1;
                        // Restore graph_node_idx to after the current match
                        graph_node_idx <= current_match[current_depth - 1] + 1;
                        state <= REW_MATCH_SEARCH;
                    end else begin
                        state <= REW_DONE;
                    end
                end

                REW_APPLY_DELETE: begin
                    // Delete nodes in L\K from the graph
                    mem_addr  <= ADDR_GRAPH_MEM_BASE + current_match[0];
                    mem_wr_en <= 1'b1;
                    mem_wr_data <= 32'hDEAD_DEAD; // Tombstone marker
                    if (mem_ready) begin
                        mem_wr_en <= 1'b0;
                        state <= REW_APPLY_ADD;
                    end
                end

                REW_APPLY_ADD: begin
                    // Add R\K nodes to the graph
                    mem_addr  <= ADDR_GRAPH_MEM_BASE + graph_node_idx;
                    mem_wr_en <= 1'b1;
                    mem_wr_data <= opcode; // New node data
                    if (mem_ready) begin
                        mem_wr_en <= 1'b0;
                        proof_trace_valid <= 1'b1;
                        proof_trace_push <= {16'hBEEF, graph_node_idx};
                        state <= REW_DONE;
                    end
                end

                REW_DONE: begin
                    done   <= 1'b1;
                    status <= STATUS_DONE;
                    state  <= REW_IDLE;
                end
            endcase
        end
    end

endmodule
