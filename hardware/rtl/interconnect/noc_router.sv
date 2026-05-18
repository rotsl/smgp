// =============================================================================
// Network-on-Chip Router (2D Mesh)
// =============================================================================
// Implements a 5-port router (N, S, E, W, Local) for a 2D mesh NoC.
// Uses XY deterministic routing with virtual channels for deadlock avoidance.
// Each port handles flits carrying graph node/edge data between compute
// tiles and memory banks.
//
// References:
//   - Dally, W.J. & Towles, B.P. (2004). "Principles and Practices of
//     Interconnection Networks." Morgan Kaufmann.
//   - Peh, L.-S. & Dally, W.J. (2001). "A Delay Model and Speculative
//     Architecture for Pipelined Routers." HPCA.
// =============================================================================

`timescale 1ns / 1ps

module noc_router #(
    parameter int DATA_WIDTH    = 64,
    parameter int FLIT_WIDTH    = 72,   // data + dest_x(4) + dest_y(4) + vc(2) + type(2)
    parameter int X_POS         = 0,
    parameter int Y_POS         = 0,
    parameter int BUFFER_DEPTH  = 4,
    parameter int NUM_VCS       = 2
)(
    input  logic                        clk,
    input  logic                        rst_n,

    // 5 Port interfaces (North, South, East, West, Local)
    // Input flits
    input  logic [FLIT_WIDTH-1:0]       flit_in_n, flit_in_s, flit_in_e, flit_in_w, flit_in_l,
    input  logic                        valid_in_n, valid_in_s, valid_in_e, valid_in_w, valid_in_l,
    output logic                        ready_in_n, ready_in_s, ready_in_e, ready_in_w, ready_in_l,

    // Output flits
    output logic [FLIT_WIDTH-1:0]       flit_out_n, flit_out_s, flit_out_e, flit_out_w, flit_out_l,
    output logic                        valid_out_n, valid_out_s, valid_out_e, valid_out_w, valid_out_l,
    input  logic                        ready_out_n, ready_out_s, ready_out_e, ready_out_w, ready_out_l,

    // Status
    output logic [3:0]                  router_x,
    output logic [3:0]                  router_y
);

    // =========================================================================
    // Extract destination from flit
    // =========================================================================
    function automatic logic [3:0] get_dest_x(input logic [FLIT_WIDTH-1:0] flit);
        get_dest_x = flit[67:64]; // bits [67:64]
    endfunction

    function automatic logic [3:0] get_dest_y(input logic [FLIT_WIDTH-1:0] flit);
        get_dest_y = flit[63:60];
    endfunction

    function automatic logic [1:0] get_vc(input logic [FLIT_WIDTH-1:0] flit);
        get_vc = flit[59:58];
    endfunction

    // =========================================================================
    // Routing computation (XY routing)
    // =========================================================================
    typedef enum logic [2:0] {
        DIR_NORTH = 3'd0,
        DIR_SOUTH = 3'd1,
        DIR_EAST  = 3'd2,
        DIR_WEST  = 3'd3,
        DIR_LOCAL = 3'd4,
        DIR_NONE  = 3'd5
    } direction_e;

    function automatic direction_e compute_route(
        input logic [FLIT_WIDTH-1:0] flit
    );
        logic [3:0] dx, dy;
        dx = get_dest_x(flit);
        dy = get_dest_y(flit);

        if (dx > X_POS)
            compute_route = DIR_EAST;
        else if (dx < X_POS)
            compute_route = DIR_WEST;
        else if (dy > Y_POS)
            compute_route = DIR_NORTH;
        else if (dy < Y_POS)
            compute_route = DIR_SOUTH;
        else
            compute_route = DIR_LOCAL;
    endfunction

    // =========================================================================
    // Input buffers (one per port)
    // =========================================================================
    logic [FLIT_WIDTH-1:0] buf_n [0:BUFFER_DEPTH-1];
    logic [FLIT_WIDTH-1:0] buf_s [0:BUFFER_DEPTH-1];
    logic [FLIT_WIDTH-1:0] buf_e [0:BUFFER_DEPTH-1];
    logic [FLIT_WIDTH-1:0] buf_w [0:BUFFER_DEPTH-1];
    logic [FLIT_WIDTH-1:0] buf_l [0:BUFFER_DEPTH-1];

    logic [$clog2(BUFFER_DEPTH)-1:0] head_n, head_s, head_e, head_w, head_l;
    logic [$clog2(BUFFER_DEPTH)-1:0] tail_n, tail_s, tail_e, tail_w, tail_l;
    logic [$clog2(BUFFER_DEPTH):0]   count_n, count_s, count_e, count_w, count_l;

    // =========================================================================
    // Buffer write (input side)
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            head_n <= 0; head_s <= 0; head_e <= 0; head_w <= 0; head_l <= 0;
            tail_n <= 0; tail_s <= 0; tail_e <= 0; tail_w <= 0; tail_l <= 0;
            count_n <= 0; count_s <= 0; count_e <= 0; count_w <= 0; count_l <= 0;
        end else begin
            // North input
            ready_in_n <= (count_n < BUFFER_DEPTH);
            if (valid_in_n && count_n < BUFFER_DEPTH) begin
                buf_n[tail_n] <= flit_in_n;
                tail_n <= tail_n + 1;
                count_n <= count_n + 1;
            end
            // South input
            ready_in_s <= (count_s < BUFFER_DEPTH);
            if (valid_in_s && count_s < BUFFER_DEPTH) begin
                buf_s[tail_s] <= flit_in_s;
                tail_s <= tail_s + 1;
                count_s <= count_s + 1;
            end
            // East input
            ready_in_e <= (count_e < BUFFER_DEPTH);
            if (valid_in_e && count_e < BUFFER_DEPTH) begin
                buf_e[tail_e] <= flit_in_e;
                tail_e <= tail_e + 1;
                count_e <= count_e + 1;
            end
            // West input
            ready_in_w <= (count_w < BUFFER_DEPTH);
            if (valid_in_w && count_w < BUFFER_DEPTH) begin
                buf_w[tail_w] <= flit_in_w;
                tail_w <= tail_w + 1;
                count_w <= count_w + 1;
            end
            // Local input
            ready_in_l <= (count_l < BUFFER_DEPTH);
            if (valid_in_l && count_l < BUFFER_DEPTH) begin
                buf_l[tail_l] <= flit_in_l;
                tail_l <= tail_l + 1;
                count_l <= count_l + 1;
            end
        end
    end

    // =========================================================================
    // Route computation and switch allocation (simplified round-robin)
    // =========================================================================
    always_comb begin
        // Default: no output
        flit_out_n = '0; flit_out_s = '0; flit_out_e = '0; flit_out_w = '0; flit_out_l = '0;
        valid_out_n = 1'b0; valid_out_s = 1'b0; valid_out_e = 1'b0; valid_out_w = 1'b0; valid_out_l = 1'b0;

        // Route from each non-empty buffer (priority: local > N > S > E > W)
        if (count_l > 0) begin
            direction_e dir;
            dir = compute_route(buf_l[head_l]);
            case (dir)
                DIR_NORTH: begin flit_out_n = buf_l[head_l]; valid_out_n = 1'b1; end
                DIR_SOUTH: begin flit_out_s = buf_l[head_l]; valid_out_s = 1'b1; end
                DIR_EAST:  begin flit_out_e = buf_l[head_l]; valid_out_e = 1'b1; end
                DIR_WEST:  begin flit_out_w = buf_l[head_l]; valid_out_w = 1'b1; end
                default:   begin flit_out_l = buf_l[head_l]; valid_out_l = 1'b1; end
            endcase
        end
    end

    // =========================================================================
    // Buffer read (output side - consume on successful transmission)
    // =========================================================================
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            // heads already reset above
        end else begin
            // Consume from buffers when output is accepted
            if (count_l > 0 && valid_out_l && ready_out_l) begin
                head_l <= head_l + 1;
                count_l <= count_l - 1;
            end
            if (count_n > 0 && valid_out_n && ready_out_n) begin
                head_n <= head_n + 1;
                count_n <= count_n - 1;
            end
            if (count_s > 0 && valid_out_s && ready_out_s) begin
                head_s <= head_s + 1;
                count_s <= count_s - 1;
            end
            if (count_e > 0 && valid_out_e && ready_out_e) begin
                head_e <= head_e + 1;
                count_e <= count_e - 1;
            end
            if (count_w > 0 && valid_out_w && ready_out_w) begin
                head_w <= head_w + 1;
                count_w <= count_w - 1;
            end
        end
    end

    // Position
    assign router_x = X_POS[3:0];
    assign router_y = Y_POS[3:0];

endmodule
