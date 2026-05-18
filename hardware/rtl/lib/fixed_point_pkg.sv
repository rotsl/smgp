// =============================================================================
// Parametric Fixed-Point Arithmetic Library
// =============================================================================
// Provides synthesisable fixed-point mathematical operations for all SMGPU
// compute engines. Supports configurable Q-format (integer + fractional bits).
//
// References:
//   - Kastner, R. et al. (2010). "Floating-Point to Fixed-Point Conversion."
//     In: "Embedded Processor Design Challenges." Springer.
//   - Parhi, K.K. (1999). "VLSI Digital Signal Processing Systems."
//     John Wiley & Sons.
// =============================================================================

package fixed_point_pkg;

    // ---------------------------------------------------------------------------
    // Parameters: Default Q8.24 format (32-bit)
    // ---------------------------------------------------------------------------
    localparam int INT_BITS_DEFAULT  = 8;
    localparam int FRAC_BITS_DEFAULT = 24;
    localparam int TOTAL_BITS_DEFAULT = INT_BITS_DEFAULT + FRAC_BITS_DEFAULT;

    // ---------------------------------------------------------------------------
    // Rounding modes
    // ---------------------------------------------------------------------------
    typedef enum logic [1:0] {
        ROUND_TRUNCATE = 2'b00,
        ROUND_NEAREST  = 2'b01,
        ROUND_CEIL     = 2'b10,
        ROUND_FLOOR    = 2'b11
    } rounding_mode_e;

    // ---------------------------------------------------------------------------
    // Q-format multiply: (Q_a.b * Q_c.d) => Q_(a+c).(b+d), then truncate
    // ---------------------------------------------------------------------------
    function automatic logic [31:0] fp_mul(
        input logic [31:0] a,
        input logic [31:0] b,
        input int frac_bits
    );
        logic [63:0] product;
        product = $signed(a) * $signed(b);
        // Adjust: shift right by frac_bits to re-align the binary point
        fp_mul = product[63:32];
    endfunction

    // ---------------------------------------------------------------------------
    // Q-format add with saturation
    // ---------------------------------------------------------------------------
    function automatic logic [31:0] fp_add(
        input logic [31:0] a,
        input logic [31:0] b
    );
        logic [32:0] sum;
        sum = $signed({1'b0, a}) + $signed({1'b0, b});
        // Saturation
        if (sum[32] != sum[31])
            fp_add = sum[32] ? 32'h8000_0000 : 32'h7FFF_FFFF;
        else
            fp_add = sum[31:0];
    endfunction

    // ---------------------------------------------------------------------------
    // Q-format subtract with saturation
    // ---------------------------------------------------------------------------
    function automatic logic [31:0] fp_sub(
        input logic [31:0] a,
        input logic [31:0] b
    );
        fp_sub = fp_add(a, ~b + 1'b1);
    endfunction

    // ---------------------------------------------------------------------------
    // Absolute value
    // ---------------------------------------------------------------------------
    function automatic logic [31:0] fp_abs(input logic [31:0] a);
        fp_abs = (a[31]) ? (~a + 1'b1) : a;
    endfunction

    // ---------------------------------------------------------------------------
    // Newton-Raphson reciprocal: 1/x
    // Converges in ~8 iterations for 24-bit fractional precision.
    // ---------------------------------------------------------------------------
    function automatic logic [31:0] fp_reciprocal(
        input logic [31:0] x,
        input int frac_bits
    );
        logic [31:0] y;
        logic [31:0] two;
        logic [31:0] xy;
        logic [31:0] two_minus_xy;
        logic [63:0] product;
        two = (32'b1 << frac_bits) + (32'b1 << frac_bits); // 2.0 in Q-format
        // Initial guess: approximate 1/x
        y = (32'hFFFF_FFFF ^ x) + 1'b1; // Bit-invert as crude estimate
        y = y >>> 1;
        // 8 Newton-Raphson iterations: y = y * (2 - x*y)
        for (int i = 0; i < 8; i++) begin
            product = $signed(x) * $signed(y);
            xy = product[63:32];
            two_minus_xy = fp_sub(two, xy);
            product = $signed(y) * $signed(two_minus_xy);
            y = product[63:32];
        end
        fp_reciprocal = y;
    endfunction

    // ---------------------------------------------------------------------------
    // Integer square root (for Euclidean distance computation)
    // Uses digit-recurrence algorithm.
    // ---------------------------------------------------------------------------
    function automatic logic [31:0] fp_sqrt(input logic [31:0] x, input int frac_bits);
        logic [63:0] val, rem, root, trial;
        val  = {x, 32'b0};
        rem  = 64'b0;
        root = 64'b0;
        for (int i = 31; i >= 0; i--) begin
            rem   = {rem, val[63:62]};
            val   = val << 2;
            trial = (root << 2) | 64'b01;
            if (rem >= trial) begin
                rem  = rem - trial;
                root = (root << 1) | 64'b01;
            end else begin
                root = root << 1;
            end
        end
        // Adjust for fractional bits: divide result by sqrt(2^frac_bits)
        // = shift right by frac_bits/2
        fp_sqrt = root[31:0] >>> (frac_bits / 2);
    endfunction

    // ---------------------------------------------------------------------------
    // Int-to-fixed-point conversion
    // ---------------------------------------------------------------------------
    function automatic logic [31:0] int_to_fp(input int val, input int frac_bits);
        int_to_fp = val << frac_bits;
    endfunction

    // ---------------------------------------------------------------------------
    // Fixed-point to integer conversion (truncate)
    // ---------------------------------------------------------------------------
    function automatic int fp_to_int(input logic [31:0] fp_val, input int frac_bits);
        logic signed [31:0] shifted;
        shifted = $signed(fp_val) >>> frac_bits;
        fp_to_int = int'(shifted);
    endfunction

    // ---------------------------------------------------------------------------
    // Real-to-fixed-point conversion (for testbench use)
    // ---------------------------------------------------------------------------
    function automatic logic [31:0] real_to_fp(input real val, input int frac_bits);
        real scaled;
        scaled = val * real'(1 <<< frac_bits);
        real_to_fp = 32'(int'(scaled));
        // Use intermediate variable to satisfy function return rules
    endfunction

    // ---------------------------------------------------------------------------
    // MAC (Multiply-Accumulate): acc += a * b
    // ---------------------------------------------------------------------------
    function automatic logic [31:0] fp_mac(
        input logic [31:0] acc,
        input logic [31:0] a,
        input logic [31:0] b,
        input int frac_bits
    );
        logic [31:0] product;
        product = fp_mul(a, b, frac_bits);
        fp_mac = fp_add(acc, product);
    endfunction

endpackage
