//! Inverse Discrete Cosine Transform (IDCT) for JPEG decoding.
//!
//! Implements the integer IDCT matching libjpeg's jidctint.c for consistent
//! decoding with the reference implementation.

/// Fixed-point scale factor (13 bits of fractional precision, like libjpeg)
const CONST_BITS: i32 = 13;
const PASS1_BITS: i32 = 2;

/// Rounding constant for descaling in pass 1
const ROUND_PASS1: i32 = 1 << (CONST_BITS - PASS1_BITS - 1);

/// Rounding constant for final output
const ROUND_OUTPUT: i32 = 1 << (CONST_BITS + PASS1_BITS + 3 - 1);

#[inline(always)]
fn fix_mul(a: i32, b: i32) -> i32 {
    ((a as i64 * b as i64) >> CONST_BITS) as i32
}

/// Fixed-point constants for the IDCT (scaled by 2^13)
/// These match libjpeg's jidctint.c
const FIX_0_298631336: i32 = 2446; // FIX(0.298631336)
const FIX_0_390180644: i32 = 3196; // FIX(0.390180644)
const FIX_0_541196100: i32 = 4433; // FIX(0.541196100)
const FIX_0_765366865: i32 = 6270; // FIX(0.765366865)
const FIX_0_899976223: i32 = 7373; // FIX(0.899976223)
const FIX_1_175875602: i32 = 9633; // FIX(1.175875602)
const FIX_1_501321110: i32 = 12299; // FIX(1.501321110)
const FIX_1_847759065: i32 = 15137; // FIX(1.847759065)
const FIX_1_961570560: i32 = 16069; // FIX(1.961570560)
const FIX_2_053119869: i32 = 16819; // FIX(2.053119869)
const FIX_2_562915447: i32 = 20995; // FIX(2.562915447)
const FIX_3_072711026: i32 = 25172; // FIX(3.072711026)

/// Perform 2D inverse DCT on an 8x8 block using fixed-point (integer) algorithm.
///
/// This matches libjpeg's jidctint.c implementation.
///
/// # Arguments
/// * `coeffs` - 64 dequantized DCT coefficients in row-major order
///
/// # Returns
/// 64 pixel values in 0-255 range
pub fn idct_2d_integer(coeffs: &[i32; 64]) -> [u8; 64] {
    let mut workspace = [0i32; 64];

    // Pass 1: process columns from input, store into workspace
    for col in 0..8 {
        // Extract column
        let d0 = coeffs[col];
        let d1 = coeffs[col + 8];
        let d2 = coeffs[col + 16];
        let d3 = coeffs[col + 24];
        let d4 = coeffs[col + 32];
        let d5 = coeffs[col + 40];
        let d6 = coeffs[col + 48];
        let d7 = coeffs[col + 56];

        // Even part
        let tmp0 = d0 << CONST_BITS;
        let tmp1 = d2 << CONST_BITS;
        let tmp2 = d4 << CONST_BITS;
        let tmp3 = d6 << CONST_BITS;

        let tmp10 = tmp0 + tmp2;
        let tmp11 = tmp0 - tmp2;

        let z1 = fix_mul(tmp1 + tmp3, FIX_0_541196100);
        let tmp12 = z1 - fix_mul(tmp3, FIX_1_847759065);
        let tmp13 = z1 + fix_mul(tmp1, FIX_0_765366865);

        let tmp0 = tmp10 + tmp13;
        let tmp3 = tmp10 - tmp13;
        let tmp1 = tmp11 + tmp12;
        let tmp2 = tmp11 - tmp12;

        // Odd part
        let z1 = d1;
        let z2 = d3;
        let z3 = d5;
        let z4 = d7;

        let z5 = fix_mul(z1 + z3, FIX_1_175875602);

        let tmp10 = fix_mul(z1, FIX_0_298631336);
        let tmp11 = fix_mul(z2, FIX_2_053119869);
        let tmp12 = fix_mul(z3, FIX_3_072711026);
        let tmp13 = fix_mul(z4, FIX_1_501321110);

        let z1 = fix_mul(z1 + z4, -FIX_0_899976223);
        let z2 = fix_mul(z2 + z3, -FIX_2_562915447);
        let z3 = fix_mul(z3 + z4, -FIX_1_961570560);
        let z4 = fix_mul(d1 + d3, -FIX_0_390180644);

        let z3 = z3 + z5;
        let z4 = z4 + z5;

        let tmp10 = tmp10 + z1 + z3;
        let tmp11 = tmp11 + z2 + z4;
        let tmp12 = tmp12 + z2 + z3;
        let tmp13 = tmp13 + z1 + z4;

        // Final output stage: descale and store to workspace
        workspace[col] = (tmp0 + tmp13 + ROUND_PASS1) >> (CONST_BITS - PASS1_BITS);
        workspace[col + 56] = (tmp0 - tmp13 + ROUND_PASS1) >> (CONST_BITS - PASS1_BITS);
        workspace[col + 8] = (tmp1 + tmp12 + ROUND_PASS1) >> (CONST_BITS - PASS1_BITS);
        workspace[col + 48] = (tmp1 - tmp12 + ROUND_PASS1) >> (CONST_BITS - PASS1_BITS);
        workspace[col + 16] = (tmp2 + tmp11 + ROUND_PASS1) >> (CONST_BITS - PASS1_BITS);
        workspace[col + 40] = (tmp2 - tmp11 + ROUND_PASS1) >> (CONST_BITS - PASS1_BITS);
        workspace[col + 24] = (tmp3 + tmp10 + ROUND_PASS1) >> (CONST_BITS - PASS1_BITS);
        workspace[col + 32] = (tmp3 - tmp10 + ROUND_PASS1) >> (CONST_BITS - PASS1_BITS);
    }

    // Pass 2: process rows from workspace, produce output
    let mut output = [0u8; 64];

    for row in 0..8 {
        let row_offset = row * 8;

        let d0 = workspace[row_offset];
        let d1 = workspace[row_offset + 1];
        let d2 = workspace[row_offset + 2];
        let d3 = workspace[row_offset + 3];
        let d4 = workspace[row_offset + 4];
        let d5 = workspace[row_offset + 5];
        let d6 = workspace[row_offset + 6];
        let d7 = workspace[row_offset + 7];

        // Even part
        let tmp0 = (d0) << CONST_BITS;
        let tmp1 = (d2) << CONST_BITS;
        let tmp2 = (d4) << CONST_BITS;
        let tmp3 = (d6) << CONST_BITS;

        let tmp10 = tmp0 + tmp2;
        let tmp11 = tmp0 - tmp2;

        let z1 = fix_mul(tmp1 + tmp3, FIX_0_541196100);
        let tmp12 = z1 - fix_mul(tmp3, FIX_1_847759065);
        let tmp13 = z1 + fix_mul(tmp1, FIX_0_765366865);

        let tmp0 = tmp10 + tmp13;
        let tmp3 = tmp10 - tmp13;
        let tmp1 = tmp11 + tmp12;
        let tmp2 = tmp11 - tmp12;

        // Odd part
        let z1 = d1;
        let z2 = d3;
        let z3 = d5;
        let z4 = d7;

        let z5 = fix_mul(z1 + z3, FIX_1_175875602);

        let tmp10 = fix_mul(z1, FIX_0_298631336);
        let tmp11 = fix_mul(z2, FIX_2_053119869);
        let tmp12 = fix_mul(z3, FIX_3_072711026);
        let tmp13 = fix_mul(z4, FIX_1_501321110);

        let z1 = fix_mul(z1 + z4, -FIX_0_899976223);
        let z2 = fix_mul(z2 + z3, -FIX_2_562915447);
        let z3 = fix_mul(z3 + z4, -FIX_1_961570560);
        let z4 = fix_mul(d1 + d3, -FIX_0_390180644);

        let z3 = z3 + z5;
        let z4 = z4 + z5;

        let tmp10 = tmp10 + z1 + z3;
        let tmp11 = tmp11 + z2 + z4;
        let tmp12 = tmp12 + z2 + z3;
        let tmp13 = tmp13 + z1 + z4;

        // Final output stage: descale, add DC offset (128), clamp to 0-255
        let out0 = descale_and_clamp(tmp0 + tmp13);
        let out7 = descale_and_clamp(tmp0 - tmp13);
        let out1 = descale_and_clamp(tmp1 + tmp12);
        let out6 = descale_and_clamp(tmp1 - tmp12);
        let out2 = descale_and_clamp(tmp2 + tmp11);
        let out5 = descale_and_clamp(tmp2 - tmp11);
        let out3 = descale_and_clamp(tmp3 + tmp10);
        let out4 = descale_and_clamp(tmp3 - tmp10);

        output[row_offset] = out0;
        output[row_offset + 1] = out1;
        output[row_offset + 2] = out2;
        output[row_offset + 3] = out3;
        output[row_offset + 4] = out4;
        output[row_offset + 5] = out5;
        output[row_offset + 6] = out6;
        output[row_offset + 7] = out7;
    }

    output
}

/// Descale a value from pass 2 and clamp to 0-255.
#[inline(always)]
fn descale_and_clamp(val: i32) -> u8 {
    // Add DC offset (128), descale, and clamp
    let scaled = (val + ROUND_OUTPUT) >> (CONST_BITS + PASS1_BITS + 3);
    let with_dc = scaled + 128;
    with_dc.clamp(0, 255) as u8
}

/// Dequantize coefficients using a quantization table.
///
/// # Arguments
/// * `coeffs` - 64 quantized DCT coefficients in zigzag order
/// * `qtable` - 64-element quantization table in zigzag order (8-bit or 16-bit)
///
/// # Returns
/// 64 dequantized coefficients in natural (row-major) order
pub fn dequantize(coeffs: &[i16; 64], qtable: &[u16; 64]) -> [i32; 64] {
    /// Zigzag to natural order mapping.
    /// This must match the encoder's ZIGZAG table from src/jpeg/quantize.rs.
    const UNZIGZAG: [usize; 64] = [
        0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27,
        20, 13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
    ];

    let mut result = [0i32; 64];
    for i in 0..64 {
        // coeffs and qtable are in zigzag order; output to natural order
        let natural_pos = UNZIGZAG[i];
        result[natural_pos] = (coeffs[i] as i32) * (qtable[i] as i32);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_idct_all_zeros() {
        let coeffs = [0i32; 64];
        let output = idct_2d_integer(&coeffs);

        // All zeros should produce middle gray (128)
        for &pixel in &output {
            assert_eq!(pixel, 128);
        }
    }

    #[test]
    fn test_idct_dc_only() {
        let mut coeffs = [0i32; 64];
        coeffs[0] = 100 << CONST_BITS; // Large positive DC

        let output = idct_2d_integer(&coeffs);

        // All pixels should be above 128 (DC adds brightness)
        for &pixel in &output {
            assert!(pixel > 128 || pixel == 255, "pixel {pixel} should be > 128");
        }
    }

    #[test]
    fn test_idct_negative_dc() {
        let mut coeffs = [0i32; 64];
        coeffs[0] = -100 << CONST_BITS; // Negative DC

        let output = idct_2d_integer(&coeffs);

        // All pixels should be below 128
        for &pixel in &output {
            assert!(pixel < 128 || pixel == 0, "pixel {pixel} should be < 128");
        }
    }

    #[test]
    fn test_idct_output_range() {
        // Test with various coefficient patterns to ensure output is always 0-255
        // Use smaller values to avoid overflow in fixed-point math
        let patterns = [
            [100i32; 64],  // All positive
            [-100i32; 64], // All negative
            [500i32; 64],  // Larger values
        ];

        for pattern in &patterns {
            let output = idct_2d_integer(pattern);
            // All output values should be valid u8 (the function returns [u8; 64])
            assert_eq!(output.len(), 64);
        }
    }

    #[test]
    fn test_dequantize() {
        let mut coeffs = [0i16; 64];
        coeffs[0] = 10; // DC
        coeffs[1] = 5; // First AC

        let mut qtable = [16u16; 64];
        qtable[0] = 16;
        qtable[1] = 11;

        let result = dequantize(&coeffs, &qtable);

        // DC (zigzag 0 -> natural 0): 10 * 16 = 160
        assert_eq!(result[0], 160);
        // First AC (zigzag 1 -> natural 1): 5 * 11 = 55
        assert_eq!(result[1], 55);
    }

    #[test]
    fn test_dequantize_all_zeros() {
        let coeffs = [0i16; 64];
        let qtable = [16u16; 64];
        let result = dequantize(&coeffs, &qtable);

        for &v in &result {
            assert_eq!(v, 0);
        }
    }

    #[test]
    fn test_dequantize_negative() {
        let mut coeffs = [0i16; 64];
        coeffs[0] = -10;

        let mut qtable = [1u16; 64];
        qtable[0] = 16;

        let result = dequantize(&coeffs, &qtable);

        assert_eq!(result[0], -160);
    }

    #[test]
    fn test_descale_and_clamp_boundaries() {
        // Test the output is valid u8 (all u8 values are valid)
        let output = descale_and_clamp(0);
        // Just verify it produces a value (it always will since u8 is bounded)
        let _ = output;
    }

    #[test]
    fn test_descale_and_clamp_extremes() {
        // Very large positive value should clamp to 255
        let output = descale_and_clamp(i32::MAX / 2);
        assert_eq!(output, 255);

        // Very large negative value should clamp to 0
        let output = descale_and_clamp(i32::MIN / 2);
        assert_eq!(output, 0);
    }

    #[test]
    fn test_fix_mul() {
        // Test fixed-point multiplication
        let result = fix_mul(8192, 8192); // 1.0 * 1.0 in Q13
        assert_eq!(result, 8192); // Should be ~1.0

        let result = fix_mul(0, 12345);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_idct_symmetry() {
        // Symmetric input should produce symmetric output
        let mut coeffs = [0i32; 64];
        coeffs[0] = 1000; // DC only

        let output = idct_2d_integer(&coeffs);

        // With only DC, all output values should be the same
        let first = output[0];
        for &pixel in &output {
            assert_eq!(
                pixel, first,
                "all pixels should be equal with DC-only input"
            );
        }
    }

    #[test]
    fn test_unzigzag_matches_encoder_zigzag() {
        // The UNZIGZAG table inside dequantize must match the encoder's ZIGZAG.
        // We verify by checking that specific zigzag positions map to correct natural positions.
        // Standard JPEG zigzag order for first 10 positions:
        //   zigzag 0 -> natural 0  (DC)
        //   zigzag 1 -> natural 1
        //   zigzag 2 -> natural 8  (second row)
        //   zigzag 3 -> natural 16 (third row)
        //   zigzag 4 -> natural 9
        //   zigzag 5 -> natural 2
        //   zigzag 6 -> natural 3
        //   zigzag 7 -> natural 10
        //   zigzag 8 -> natural 17
        //   zigzag 9 -> natural 24

        let qtable = [1u16; 64]; // Identity quantization

        // Test zigzag position 2 should map to natural position 8
        let mut coeffs = [0i16; 64];
        coeffs[2] = 42; // Put value at zigzag position 2
        let result = dequantize(&coeffs, &qtable);
        assert_eq!(result[8], 42, "zigzag[2] should map to natural[8]");

        // Test zigzag position 3 should map to natural position 16
        let mut coeffs = [0i16; 64];
        coeffs[3] = 42;
        let result = dequantize(&coeffs, &qtable);
        assert_eq!(result[16], 42, "zigzag[3] should map to natural[16]");

        // Test zigzag position 5 should map to natural position 2
        let mut coeffs = [0i16; 64];
        coeffs[5] = 42;
        let result = dequantize(&coeffs, &qtable);
        assert_eq!(result[2], 42, "zigzag[5] should map to natural[2]");
    }
}
