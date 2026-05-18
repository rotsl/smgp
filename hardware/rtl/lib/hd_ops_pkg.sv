// =============================================================================
// Hyperdimensional Computing Operations Package
// =============================================================================
// Provides synthesizable primitives for HD vector operations:
//   - Bundle (majority vote on bipolar vectors)
//   - Bind/Unbind (element-wise XOR for bipolar = element-wise multiply)
//   - Permute (cyclic shift)
//   - Similarity (dot product with normalization)
//
// All operations are parameterized for vector dimension and can be
// fully unrolled for single-cycle latency or pipelined for area efficiency.
//
// References:
//   - Kanerva, P. (2009). "Hyperdimensional Computing." Cognitive Computation.
//   - Rahimi, A., et al. (2017). "Hyperdimensional Computing for Blind and
//     One-Shot Classification." ISVLSI.
// =============================================================================

package hd_ops_pkg;

    // ---------------------------------------------------------------------------
    // Bundle: element-wise majority vote over multiple bipolar vectors
    // For bipolar {-1,+1} vectors, this equals sign(sum(vectors))
    // ---------------------------------------------------------------------------
    function automatic logic [0:0] hd_bundle_bit(
        input logic [0:0] vecs []  // Array of single bits across vectors
    );
        int count_pos;
        count_pos = 0;
        for (int i = 0; i < vecs.size(); i++) begin
            count_pos = count_pos + int'(vecs[i]);
        end
        hd_bundle_bit = (count_pos >= vecs.size()/2) ? 1'b1 : 1'b0;
    endfunction

    // ---------------------------------------------------------------------------
    // Bind: element-wise XOR for bipolar vectors stored as single bits
    // bind(a, b) = a XOR b (self-inverse for binary/bipolar)
    // ---------------------------------------------------------------------------
    function automatic logic [0:0] hd_bind_bit(
        input logic a,
        input logic b
    );
        hd_bind_bit = a ^ b;
    endfunction

    // ---------------------------------------------------------------------------
    // Permute: cyclic left shift by N positions
    // ---------------------------------------------------------------------------
    function automatic logic [31:0] hd_permute_word(
        input logic [31:0] word,
        input int shift,
        input int total_words
    );
        int effective_shift;
        effective_shift = shift % total_words;
        hd_permute_word = (word << effective_shift) | (word >> (32 - effective_shift));
    endfunction

    // ---------------------------------------------------------------------------
    // Similarity: popcount-based dot product for bipolar vectors
    // sim(a, b) = (matching_bits - mismatching_bits) / D
    //           = (D - 2*hamming_distance) / D
    // ---------------------------------------------------------------------------

    // Popcount for 32-bit word
    function automatic int popcount32(input logic [31:0] v);
        popcount32 = $countones(v);
    endfunction

    // ---------------------------------------------------------------------------
    // Generate random bipolar vector (for testbench initialization)
    // Uses LFSR-based PRNG seeded from input.
    // ---------------------------------------------------------------------------
    function automatic logic [31:0] lfsr_next(input logic [31:0] state);
        logic [31:0] next;
        next = (state << 1) ^ ((state[31]) ? 32'h8020_0003 : 32'h0000_0000);
        lfsr_next = next;
    endfunction

    // ---------------------------------------------------------------------------
    // Convert between bipolar {0,1} and {-1,+1} representations
    // Storage: 0 = +1, 1 = -1 ( saves inversion for accumulation )
    // ---------------------------------------------------------------------------
    function automatic logic signed to_signed_bipolar(input logic bit_val);
        to_signed_bipolar = bit_val ? -1'sd1 : 1'sd1;
    endfunction

endpackage
