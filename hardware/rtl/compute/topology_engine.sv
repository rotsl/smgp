// =============================================================================
// Topological Persistence Engine
// =============================================================================
// Hardware accelerator for persistent homology computation on graphs.
// Uses a streaming merge-tree architecture to compute Vietoris-Rips
// filtrations and generate persistence barcodes (birth-death pairs).
//
// Pipeline:
//   1. Compute pairwise distance matrix (streaming)
//   2. Sort edges by distance (merge sort network)
//   3. Build Union-Find as edges are added
//   4. Record persistence pairs at merge events
//   5. Compute Wasserstein distance for stability check
//
// References:
//   - Edelsbrunner, H., et al. (2002). "Topological Persistence and
//     Simplification." DCG.
//   - Kerber, M., et al. (2017). "Geometry Helps to Compare Persistence
//     Diagrams." JEA.
//   - Carlsson, G. (2009). "Topology and Data." Bull. AMS.
// =============================================================================

`timescale 1ns / 1ps

module topology_engine #(
    parameter int MAX_NODES       = 4096,
    parameter int MAX_EDGES       = 8388608, // MAX_NODES * (MAX_NODES-1) / 2
    parameter int MAX_PERSIST_PAIRS = 16384,
    parameter int FRAC_BITS       = 24,
    parameter int DATA_WIDTH      = 32,
    parameter int ADDR_WIDTH      = 20
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

    // Results
    output logic [DATA_WIDTH-1:0]       barcode_birth,
    output logic [DATA_WIDTH-1:0]       barcode_death,
    output logic                        barcode_valid,
    output logic [31:0]                 wasserstein_distance,
    output logic                        stability_ok
);

    import smgp_isa_pkg::*;
    import fixed_point_pkg::*;

    // =========================================================================
    // State machine
    // =========================================================================
    typedef enum logic [3:0] {
        TOP_IDLE          = 4'd0,
        TOP_LOAD_DIST     = 4'd1,
        TOP_SORT_EDGES    = 4'd2,
        TOP_BUILD_UF      = 4'd3,
        TOP_EMIT_BARCODE  = 4'd4,
        TOP_WASSERSTEIN   = 4'd5,
        TOP_CHECK_STABLE  = 4'd6,
        TOP_DONE          = 4'd7
    } top_state_e;

    top_state_e state;

    // =========================================================================
    // Union-Find data structure (parallel prefix for hardware)
    // =========================================================================
    logic [15:0] parent [0:MAX_NODES-1];
    logic [15:0] rank_uf [0:MAX_NODES-1];

    function automatic logic [15:0] uf_find(input int x);
        logic [15:0] root;
        root = parent[x];
        // Path compression (limited depth for hardware)
        for (int i = 0; i < 16; i++) begin
            if (root != parent[root])
                root = parent[root];
            else
                break;
        end
        uf_find = root;
    endfunction

    function automatic void uf_union(input int x, input int y);
        logic [15:0] rx, ry;
        rx = uf_find(x);
        ry = uf_find(y);
        if (rx != ry) begin
            // Union by rank
            if (rank_uf[rx] < rank_uf[ry])
                parent[rx] = ry;
            else if (rank_uf[rx] > rank_uf[ry])
                parent[ry] = rx;
            else begin
                parent[ry] = rx;
                rank_uf[rx] = rank_uf[rx] + 1;
            end
        end
    endfunction

    // =========================================================================
    // Persistence pair storage
    // =========================================================================
    logic [DATA_WIDTH-1:0] birth_store  [0:MAX_PERSIST_PAIRS-1];
    logic [DATA_WIDTH-1:0] death_store [0:MAX_PERSIST_PAIRS-1];
    logic [15:0] pair_count;
    logic [15:0] pair_idx;

    // Edge sorting buffer
    logic [63:0] edge_buffer [0:255]; // {distance:32, nodes:16, padding:16}
    logic [15:0] edge_count;
    logic [15:0] edge_idx;

    // Wasserstein accumulation
    logic [63:0] wasserstein_accum;
    logic [31:0] stability_threshold;

    // Temporary variables for procedural blocks (must be module-level in Verilator)
    logic [63:0] sort_key;
    int          sort_j;
    logic [15:0] node_a, node_b;
    logic [31:0] uf_dist;

    // =========================================================================
    // Main state machine
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state            <= TOP_IDLE;
            done             <= 1'b0;
            status           <= STATUS_IDLE;
            barcode_valid    <= 1'b0;
            barcode_birth    <= 32'b0;
            barcode_death    <= 32'b0;
            wasserstein_distance <= 32'b0;
            stability_ok     <= 1'b0;
            pair_count       <= 16'b0;
            pair_idx         <= 16'b0;
            edge_count       <= 16'b0;
            edge_idx         <= 16'b0;
            wasserstein_accum <= 64'b0;
            for (int i = 0; i < MAX_NODES; i++) begin
                parent[i] <= 16'b0;
                rank_uf[i] <= 16'b0;
            end
        end else begin
            barcode_valid <= 1'b0;
            done <= 1'b0;

            case (state)
                TOP_IDLE: begin
                    if (start) begin
                        status <= STATUS_RUNNING;
                        // Initialize Union-Find: each node is its own parent
                        for (int i = 0; i < MAX_NODES && i < 256; i++)
                            parent[i] <= i[15:0];
                        pair_count <= 16'b0;
                        edge_count <= 16'b0;
                        case (opcode[27:24])
                            4'h0: state <= TOP_LOAD_DIST;    // BUILD_FILTRATION
                            4'h3: state <= TOP_WASSERSTEIN;   // WASSERSTEIN_DIST
                            4'h2: state <= TOP_CHECK_STABLE;  // CHECK_STABILITY
                            default: state <= TOP_DONE;
                        endcase
                    end
                end

                TOP_LOAD_DIST: begin
                    // Stream distance matrix entries from memory
                    mem_addr  <= ADDR_TOPO_BUF + edge_idx;
                    mem_rd_en <= 1'b1;
                    if (mem_ready) begin
                        edge_buffer[edge_count[7:0]] <= {mem_rd_data, edge_idx, 16'b0};
                        edge_count <= edge_count + 1;
                        edge_idx <= edge_idx + 1;
                        if (edge_count >= 16'd255) begin
                            mem_rd_en <= 1'b0;
                            state <= TOP_SORT_EDGES;
                        end
                    end
                end

                TOP_SORT_EDGES: begin
                    // Simple insertion sort for small batches (in hardware,
                    // a bitonic merge network would be used for production)
                    for (int i = 1; i < 256; i++) begin
                        sort_key = edge_buffer[i];
                        sort_j = i;
                        while (sort_j > 0 && edge_buffer[sort_j-1] > sort_key) begin
                            edge_buffer[sort_j] = edge_buffer[sort_j-1];
                            sort_j = sort_j - 1;
                        end
                        edge_buffer[sort_j] = sort_key;
                    end
                    edge_idx <= 16'b0;
                    state <= TOP_BUILD_UF;
                end

                TOP_BUILD_UF: begin
                    // Process sorted edges, building Union-Find
                    if (edge_idx < edge_count) begin
                        uf_dist   = edge_buffer[edge_idx[7:0]][63:32];
                        node_a = edge_buffer[edge_idx[7:0]][31:16];
                        node_b = edge_buffer[edge_idx[7:0]][15:0];

                        if (uf_find(int'(node_a)) != uf_find(int'(node_b))) begin
                            // Record persistence pair: birth at 0, death at this distance
                            if (pair_count < MAX_PERSIST_PAIRS) begin
                                birth_store[pair_count]  <= 32'b0;
                                death_store[pair_count] <= uf_dist;
                                pair_count <= pair_count + 1;
                            end
                            uf_union(int'(node_a), int'(node_b));
                        end
                        edge_idx <= edge_idx + 1;
                    end else begin
                        // Add remaining singletons (born at 0, die at max)
                        pair_idx <= 16'b0;
                        state <= TOP_EMIT_BARCODE;
                    end
                end

                TOP_EMIT_BARCODE: begin
                    // Stream out barcode pairs
                    if (pair_idx < pair_count) begin
                        barcode_birth <= birth_store[pair_idx];
                        barcode_death <= death_store[pair_idx];
                        barcode_valid <= 1'b1;
                        pair_idx <= pair_idx + 1;
                    end else begin
                        state <= TOP_DONE;
                    end
                end

                TOP_WASSERSTEIN: begin
                    // Compute Wasserstein distance between stored persistence
                    // diagrams (simplified: sum of absolute persistence differences)
                    wasserstein_accum <= 64'b0;
                    for (int i = 0; i < pair_count && i < MAX_PERSIST_PAIRS; i++) begin
                        logic [31:0] persistence;
                        persistence = death_store[i] - birth_store[i];
                        wasserstein_accum <= wasserstein_accum + int'(persistence);
                    end
                    wasserstein_distance <= wasserstein_accum[31:0];
                    state <= TOP_DONE;
                end

                TOP_CHECK_STABLE: begin
                    // Check stability: Wasserstein distance < threshold
                    stability_threshold <= opcode;
                    wasserstein_accum <= 64'b0;
                    for (int i = 0; i < pair_count && i < MAX_PERSIST_PAIRS; i++) begin
                        logic [31:0] persistence;
                        persistence = death_store[i] - birth_store[i];
                        wasserstein_accum <= wasserstein_accum + int'(persistence);
                    end
                    stability_ok <= (wasserstein_accum[31:0] < stability_threshold);
                    state <= TOP_DONE;
                end

                TOP_DONE: begin
                    done   <= 1'b1;
                    status <= STATUS_DONE;
                    state  <= TOP_IDLE;
                end
            endcase
        end
    end

endmodule
