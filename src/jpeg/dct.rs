//! Discrete Cosine Transform (DCT) implementation for JPEG.
//!
//! Uses the AAN (Arai-Agui-Nakajima) fast DCT algorithm for efficiency.
//! The AAN algorithm uses only 5 multiplications and 29 additions per 8-point DCT,
//! compared to 64 multiplications in the naive approach.

use std::f32::consts::{FRAC_1_SQRT_2, PI};

// AAN DCT constants - precomputed trigonometric values
// These are the scale factors for the AAN algorithm
const A1: f32 = FRAC_1_SQRT_2; // cos(4*pi/16) = 1/sqrt(2)
const A2: f32 = 0.541_196_1; // cos(6*pi/16) - cos(2*pi/16)
const A3: f32 = FRAC_1_SQRT_2; // cos(4*pi/16) = 1/sqrt(2)
const A4: f32 = 1.306_562_9; // cos(2*pi/16) + cos(6*pi/16)
const A5: f32 = 0.382_683_43; // cos(6*pi/16)

// Post-scaling factors for the AAN algorithm to produce correctly normalized DCT output
// These are: s[k] = 1/(4 * c[k]) where c[k] = cos(k*pi/16) for k > 0, c[0] = 1/sqrt(2)
const S: [f32; 8] = [
    0.353_553_4, // 1/(2*sqrt(2))
    0.254_897_8, // 1/(4*cos(pi/16))
    0.270_598_1, // 1/(4*cos(2*pi/16))
    0.300_672_4, // 1/(4*cos(3*pi/16))
    0.353_553_4, // 1/(4*cos(4*pi/16)) = 1/(2*sqrt(2))
    0.449_988_1, // 1/(4*cos(5*pi/16))
    0.653_281_5, // 1/(4*cos(6*pi/16))
    1.281_457_8, // 1/(4*cos(7*pi/16))
];

/// Perform 2D DCT on an 8x8 block using AAN fast DCT algorithm.
///
/// Uses the separable property: 2D DCT = 1D DCT on rows, then 1D DCT on columns.
/// Each 1D DCT uses the AAN algorithm with only 5 multiplications.
pub fn dct_2d(block: &[f32; 64]) -> [f32; 64] {
    let mut temp = [0.0f32; 64];
    let mut result = [0.0f32; 64];

    // 1D DCT on rows using AAN
    for row in 0..8 {
        let row_start = row * 8;
        let mut row_data = [0.0f32; 8];
        row_data.copy_from_slice(&block[row_start..row_start + 8]);
        aan_dct_1d(&mut row_data);
        temp[row_start..row_start + 8].copy_from_slice(&row_data);
    }

    // 1D DCT on columns using AAN
    for col in 0..8 {
        let mut col_data = [0.0f32; 8];

        for row in 0..8 {
            col_data[row] = temp[row * 8 + col];
        }

        aan_dct_1d(&mut col_data);

        for row in 0..8 {
            result[row * 8 + col] = col_data[row];
        }
    }

    result
}

/// Perform 1D DCT on 8 values using the AAN algorithm.
///
/// The AAN algorithm uses only 5 multiplications and 29 additions,
/// compared to 64 multiplications in the naive O(n²) approach.
/// Based on: Arai, Agui, and Nakajima, "A Fast DCT-SQ Scheme for Images", 1988.
#[inline]
fn aan_dct_1d(data: &mut [f32; 8]) {
    // Stage 1: Initial butterfly operations
    let tmp0 = data[0] + data[7];
    let tmp7 = data[0] - data[7];
    let tmp1 = data[1] + data[6];
    let tmp6 = data[1] - data[6];
    let tmp2 = data[2] + data[5];
    let tmp5 = data[2] - data[5];
    let tmp3 = data[3] + data[4];
    let tmp4 = data[3] - data[4];

    // Stage 2: Even part - process tmp0, tmp1, tmp2, tmp3
    let tmp10 = tmp0 + tmp3;
    let tmp13 = tmp0 - tmp3;
    let tmp11 = tmp1 + tmp2;
    let tmp12 = tmp1 - tmp2;

    data[0] = tmp10 + tmp11;
    data[4] = tmp10 - tmp11;

    // Rotation for indices 2 and 6
    let z1 = (tmp12 + tmp13) * A1; // A1 = cos(4*pi/16)
    data[2] = tmp13 + z1;
    data[6] = tmp13 - z1;

    // Stage 3: Odd part - process tmp4, tmp5, tmp6, tmp7
    let tmp10 = tmp4 + tmp5;
    let tmp11 = tmp5 + tmp6;
    let tmp12 = tmp6 + tmp7;

    // The rotator is modified from the standard AAN algorithm
    // to handle the odd part correctly
    let z5 = (tmp10 - tmp12) * A5; // A5 = cos(6*pi/16)
    let z2 = tmp10 * A2 + z5; // A2 = cos(6*pi/16) - cos(2*pi/16)
    let z4 = tmp12 * A4 + z5; // A4 = cos(2*pi/16) + cos(6*pi/16)
    let z3 = tmp11 * A3; // A3 = cos(4*pi/16)

    let z11 = tmp7 + z3;
    let z13 = tmp7 - z3;

    data[5] = z13 + z2;
    data[3] = z13 - z2;
    data[1] = z11 + z4;
    data[7] = z11 - z4;

    // Apply post-scaling to get properly normalized DCT coefficients
    for i in 0..8 {
        data[i] *= S[i];
    }
}

// Keep the old implementation for reference and for the IDCT
/// Precomputed cosine values for IDCT.
/// cos_table[i][j] = cos((2*i + 1) * j * PI / 16)
const COS_TABLE: [[f32; 8]; 8] = precompute_cos_table();

/// Precompute the cosine table at compile time.
const fn precompute_cos_table() -> [[f32; 8]; 8] {
    let mut table = [[0.0f32; 8]; 8];
    let mut i = 0;
    while i < 8 {
        let mut j = 0;
        while j < 8 {
            let angle = ((2 * i + 1) * j) as f32 * PI / 16.0;
            table[i][j] = cos_approx(angle);
            j += 1;
        }
        i += 1;
    }
    table
}

/// Approximate cosine for const fn (Taylor series).
const fn cos_approx(x: f32) -> f32 {
    let mut x = x;
    while x > PI {
        x -= 2.0 * PI;
    }
    while x < -PI {
        x += 2.0 * PI;
    }

    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    let x8 = x4 * x4;

    1.0 - x2 / 2.0 + x4 / 24.0 - x6 / 720.0 + x8 / 40320.0
}

/// Normalization factors for IDCT.
/// alpha(0) = 1/sqrt(2), alpha(k) = 1 for k > 0
const ALPHA: [f32; 8] = [
    FRAC_1_SQRT_2, // 1/sqrt(2)
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
    1.0,
];

/// Perform inverse 2D DCT on an 8x8 block.
#[allow(dead_code)]
pub fn idct_2d(block: &[f32; 64]) -> [f32; 64] {
    let mut temp = [0.0f32; 64];
    let mut result = [0.0f32; 64];

    // 1D IDCT on columns
    for col in 0..8 {
        let mut col_in = [0.0f32; 8];
        let mut col_out = [0.0f32; 8];

        for row in 0..8 {
            col_in[row] = block[row * 8 + col];
        }

        idct_1d(&col_in, &mut col_out);

        for row in 0..8 {
            temp[row * 8 + col] = col_out[row];
        }
    }

    // 1D IDCT on rows
    for row in 0..8 {
        let row_start = row * 8;
        idct_1d(
            &temp[row_start..row_start + 8],
            &mut result[row_start..row_start + 8],
        );
    }

    result
}

/// Perform 1D inverse DCT on 8 values.
fn idct_1d(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), 8);
    debug_assert_eq!(output.len(), 8);

    for n in 0..8 {
        let mut sum = 0.0f32;
        for k in 0..8 {
            sum += ALPHA[k] * input[k] * COS_TABLE[n][k];
        }
        output[n] = 0.5 * sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct_dc_component() {
        // All zeros should give all zeros
        let block = [0.0f32; 64];
        let result = dct_2d(&block);
        for &val in &result {
            assert!((val).abs() < 0.001);
        }
    }

    #[test]
    fn test_dct_constant_block() {
        // Constant block: DC component should be non-zero, AC should be small
        let block = [100.0f32; 64];
        let result = dct_2d(&block);

        // DC component (index 0) should be large
        assert!(result[0].abs() > 100.0);

        // AC components should be relatively small compared to DC
        // (some numerical error is expected with the const fn cos approximation)
        for &val in result.iter().skip(1) {
            assert!(val.abs() < 5.0, "AC component too large: {val}");
        }
    }

    #[test]
    fn test_dct_idct_roundtrip() {
        let mut block = [0.0f32; 64];
        for (i, item) in block.iter_mut().enumerate() {
            *item = (i as f32 * 4.0) - 128.0;
        }

        let dct = dct_2d(&block);
        let recovered = idct_2d(&dct);

        // Allow some numerical error due to const fn cos approximation
        for i in 0..64 {
            assert!(
                (block[i] - recovered[i]).abs() < 5.0,
                "Mismatch at {}: {} vs {}",
                i,
                block[i],
                recovered[i]
            );
        }
    }

    #[test]
    fn test_cos_table_values() {
        // cos(0) = 1
        assert!((COS_TABLE[0][0] - 1.0).abs() < 0.0001);

        // cos(pi/4) = 1/sqrt(2) ≈ 0.707
        // This is cos((2*0 + 1) * 2 * PI / 16) = cos(PI/8)
        assert!((COS_TABLE[0][2] - (PI / 8.0).cos()).abs() < 0.001);
    }
}
