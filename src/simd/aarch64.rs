//! ARM64 NEON SIMD implementations.
//!
//! This module provides NEON-accelerated implementations of:
//! - Adler-32 checksum
//! - LZ77 match length comparison
//! - PNG filter operations (Sub, Up, Average, Paeth)
//! - Filter scoring

use crate::simd::fallback::fallback_paeth_predictor;
use std::arch::aarch64::*;

/// Compute Adler-32 checksum using NEON instructions.
///
/// Processes 16 bytes at a time for improved throughput.
///
/// # Safety
/// Caller must ensure NEON is available (always true on aarch64).
#[target_feature(enable = "neon")]
pub unsafe fn adler32_neon(data: &[u8]) -> u32 {
    const MOD_ADLER: u32 = 65_521;
    // NMAX for 16-byte processing: largest n where 255*n*(n+1)/2 + (n+1)*65520 < 2^32
    const BLOCK_SIZE: usize = 5552 / 16 * 16;

    let mut s1: u32 = 1;
    let mut s2: u32 = 0;

    let mut remaining = data;

    while remaining.len() >= BLOCK_SIZE {
        let (block, rest) = remaining.split_at(BLOCK_SIZE);
        let (new_s1, new_s2) = adler32_block_neon(block, s1, s2);
        s1 = new_s1 % MOD_ADLER;
        s2 = new_s2 % MOD_ADLER;
        remaining = rest;
    }

    // Process remaining complete 16-byte chunks
    if remaining.len() >= 16 {
        let chunk_count = remaining.len() / 16 * 16;
        let (block, rest) = remaining.split_at(chunk_count);
        let (new_s1, new_s2) = adler32_block_neon(block, s1, s2);
        s1 = new_s1 % MOD_ADLER;
        s2 = new_s2 % MOD_ADLER;
        remaining = rest;
    }

    // Process remaining bytes with scalar
    for &b in remaining {
        s1 += b as u32;
        s2 += s1;
    }
    s1 %= MOD_ADLER;
    s2 %= MOD_ADLER;

    (s2 << 16) | s1
}

/// Process a block of data for Adler-32 using NEON.
///
/// The Adler-32 algorithm computes:
/// - s1 = 1 + sum of all bytes
/// - s2 = sum of all s1 values after each byte
///
/// For a chunk of 16 bytes at position p:
/// - s1 contribution = sum(b[i]) for i in 0..16
/// - s2 contribution = 16*s1_before + 16*b[0] + 15*b[1] + ... + 1*b[15]
#[target_feature(enable = "neon")]
unsafe fn adler32_block_neon(data: &[u8], mut s1: u32, mut s2: u32) -> (u32, u32) {
    // Weights for s2 accumulation within a 16-byte chunk
    // Position 0 contributes 16 times, position 15 contributes 1 time
    let weights: [u8; 16] = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
    let weights_vec = vld1q_u8(weights.as_ptr());

    for chunk in data.chunks_exact(16) {
        let v = vld1q_u8(chunk.as_ptr());

        // s2 += 16 * s1 (contribution from previous s1 value over 16 positions)
        s2 = s2.wrapping_add(s1.wrapping_mul(16));

        // Compute sum of bytes for s1 update
        // Use pairwise addition: u8x16 -> u16x8 -> u32x4 -> u64x2
        let sum_16 = vpaddlq_u8(v); // 8 x u16
        let sum_32 = vpaddlq_u16(sum_16); // 4 x u32
        let sum_64 = vpaddlq_u32(sum_32); // 2 x u64
        let chunk_sum = (vgetq_lane_u64(sum_64, 0) + vgetq_lane_u64(sum_64, 1)) as u32;

        // Compute weighted sum for s2: 16*b[0] + 15*b[1] + ... + 1*b[15]
        let v_lo = vget_low_u8(v);
        let v_hi = vget_high_u8(v);
        let w_lo = vget_low_u8(weights_vec);
        let w_hi = vget_high_u8(weights_vec);

        // Multiply u8 -> u16, then sum
        let prod_lo = vmull_u8(v_lo, w_lo); // 8 x u16
        let prod_hi = vmull_u8(v_hi, w_hi); // 8 x u16

        let prod_sum = vaddq_u16(prod_lo, prod_hi);
        let prod_32 = vpaddlq_u16(prod_sum); // 4 x u32
        let prod_64 = vpaddlq_u32(prod_32); // 2 x u64
        let weighted_sum = (vgetq_lane_u64(prod_64, 0) + vgetq_lane_u64(prod_64, 1)) as u32;

        s2 = s2.wrapping_add(weighted_sum);
        s1 = s1.wrapping_add(chunk_sum);
    }

    (s1, s2)
}

/// Compute match length using NEON 16-byte comparison.
///
/// # Safety
/// Caller must ensure NEON is available.
#[target_feature(enable = "neon")]
pub unsafe fn match_length_neon(data: &[u8], pos1: usize, pos2: usize, max_len: usize) -> usize {
    let mut length = 0;

    // Compare 16 bytes at a time
    while length + 16 <= max_len {
        let a = vld1q_u8(data[pos1 + length..].as_ptr());
        let b = vld1q_u8(data[pos2 + length..].as_ptr());

        // Compare bytes: result is 0xFF where equal, 0x00 where different
        let cmp = vceqq_u8(a, b);

        // Find first differing byte using min across lanes
        // If all equal, min will be 0xFF
        let min_val = vminvq_u8(cmp);

        if min_val != 0xFF {
            // Found a mismatch - find which byte
            // Store comparison result and scan for first 0x00
            let mut cmp_bytes = [0u8; 16];
            vst1q_u8(cmp_bytes.as_mut_ptr(), cmp);

            for (i, &b) in cmp_bytes.iter().enumerate() {
                if b == 0 {
                    return length + i;
                }
            }
        }
        length += 16;
    }

    // Handle remaining bytes with u64 comparison
    while length + 8 <= max_len {
        let a = u64::from_ne_bytes(data[pos1 + length..pos1 + length + 8].try_into().unwrap());
        let b = u64::from_ne_bytes(data[pos2 + length..pos2 + length + 8].try_into().unwrap());
        if a != b {
            let xor = a ^ b;
            // On little-endian (which ARM64 typically is), trailing zeros give us the position
            return length + (xor.trailing_zeros() / 8) as usize;
        }
        length += 8;
    }

    // Handle remaining bytes
    while length < max_len && data[pos1 + length] == data[pos2 + length] {
        length += 1;
    }

    length
}

/// Score a filtered row using sum of absolute values (NEON implementation).
///
/// # Safety
/// Caller must ensure NEON is available.
#[target_feature(enable = "neon")]
pub unsafe fn score_filter_neon(filtered: &[u8]) -> u64 {
    let mut sum: u64 = 0;
    let mut remaining = filtered;

    // Process 16 bytes at a time
    while remaining.len() >= 16 {
        let v = vld1q_u8(remaining.as_ptr());

        // For signed absolute values: treat bytes as signed, compute abs
        // bytes 0-127 stay as is, 128-255 (signed -128 to -1) become their abs
        // abs(x as i8) = if x >= 128 { 256 - x } else { x }

        // Convert to signed interpretation and compute absolute value
        let v_signed = vreinterpretq_s8_u8(v);
        let v_abs = vabsq_s8(v_signed);
        let v_unsigned = vreinterpretq_u8_s8(v_abs);

        // Sum using horizontal add: u8 -> u16 -> u32 -> u64
        let sum_16 = vpaddlq_u8(v_unsigned); // 8 x u16
        let sum_32 = vpaddlq_u16(sum_16); // 4 x u32
        let sum_64 = vpaddlq_u32(sum_32); // 2 x u64

        sum += vgetq_lane_u64(sum_64, 0) + vgetq_lane_u64(sum_64, 1);

        remaining = &remaining[16..];
    }

    // Process remaining bytes with scalar
    for &b in remaining {
        sum += (b as i8).unsigned_abs() as u64;
    }

    sum
}

/// Apply Sub filter using NEON.
///
/// # Safety
/// Caller must ensure NEON is available.
#[target_feature(enable = "neon")]
pub unsafe fn filter_sub_neon(row: &[u8], bpp: usize, output: &mut Vec<u8>) {
    let len = row.len();
    output.reserve(len);

    // First bpp bytes have no left neighbor
    for &byte in &row[..bpp.min(len)] {
        output.push(byte);
    }

    if len <= bpp {
        return;
    }

    let remaining = &row[bpp..];
    let left = &row[..len - bpp];

    let mut i = 0;
    let rem_len = remaining.len();

    // Process 16 bytes at a time
    while i + 16 <= rem_len {
        let curr = vld1q_u8(remaining[i..].as_ptr());
        let prev = vld1q_u8(left[i..].as_ptr());
        let diff = vsubq_u8(curr, prev);

        let mut buf = [0u8; 16];
        vst1q_u8(buf.as_mut_ptr(), diff);
        output.extend_from_slice(&buf);
        i += 16;
    }

    // Handle remaining bytes
    while i < rem_len {
        output.push(remaining[i].wrapping_sub(left[i]));
        i += 1;
    }
}

/// Apply Up filter using NEON.
///
/// # Safety
/// Caller must ensure NEON is available.
#[target_feature(enable = "neon")]
pub unsafe fn filter_up_neon(row: &[u8], prev_row: &[u8], output: &mut Vec<u8>) {
    let len = row.len();
    output.reserve(len);

    let mut i = 0;

    // Process 16 bytes at a time
    while i + 16 <= len {
        let curr = vld1q_u8(row[i..].as_ptr());
        let prev = vld1q_u8(prev_row[i..].as_ptr());
        let diff = vsubq_u8(curr, prev);

        let mut buf = [0u8; 16];
        vst1q_u8(buf.as_mut_ptr(), diff);
        output.extend_from_slice(&buf);
        i += 16;
    }

    // Handle remaining bytes
    while i < len {
        output.push(row[i].wrapping_sub(prev_row[i]));
        i += 1;
    }
}

/// Apply Average filter using NEON.
///
/// # Safety
/// Caller must ensure NEON is available.
#[target_feature(enable = "neon")]
pub unsafe fn filter_average_neon(row: &[u8], prev_row: &[u8], bpp: usize, output: &mut Vec<u8>) {
    let len = row.len();
    output.reserve(len);

    // First bpp bytes: average with just the above pixel
    for i in 0..bpp.min(len) {
        let above = prev_row[i];
        let avg = (above as u16 / 2) as u8;
        output.push(row[i].wrapping_sub(avg));
    }

    if len <= bpp {
        return;
    }

    let mut i = bpp;

    // Process 16 bytes at a time
    while i + 16 <= len {
        let curr = vld1q_u8(row[i..].as_ptr());
        let above = vld1q_u8(prev_row[i..].as_ptr());
        let left = vld1q_u8(row[i - bpp..].as_ptr());

        // Compute average: (left + above) / 2
        // Use vhaddq_u8 which computes (a + b) / 2 with rounding down
        let avg = vhaddq_u8(left, above);

        let diff = vsubq_u8(curr, avg);

        let mut buf = [0u8; 16];
        vst1q_u8(buf.as_mut_ptr(), diff);
        output.extend_from_slice(&buf);
        i += 16;
    }

    // Handle remaining bytes
    while i < len {
        let left = row[i - bpp] as u16;
        let above = prev_row[i] as u16;
        let avg = ((left + above) / 2) as u8;
        output.push(row[i].wrapping_sub(avg));
        i += 1;
    }
}

/// Apply Paeth filter using NEON.
///
/// The Paeth predictor is: choose the value among a (left), b (above), c (upper-left)
/// that is closest to p = a + b - c.
///
/// # Safety
/// Caller must ensure NEON is available.
#[target_feature(enable = "neon")]
pub unsafe fn filter_paeth_neon(row: &[u8], prev_row: &[u8], bpp: usize, output: &mut Vec<u8>) {
    let len = row.len();
    output.reserve(len);

    // First bpp bytes: left=0, upper_left=0
    for i in 0..bpp.min(len) {
        let above = prev_row[i];
        // Paeth with a=0, c=0: p = b, pa = |b|, pb = 0, pc = |b|
        // pb <= pa and pb <= pc, so predict = b
        output.push(row[i].wrapping_sub(above));
    }

    if len <= bpp {
        return;
    }

    let mut i = bpp;

    // NEON Paeth is complex due to the conditional logic
    // Process 16 bytes at a time using vectorized Paeth
    while i + 16 <= len {
        let curr = vld1q_u8(row[i..].as_ptr());
        let left = vld1q_u8(row[i - bpp..].as_ptr());
        let above = vld1q_u8(prev_row[i..].as_ptr());
        let upper_left = vld1q_u8(prev_row[i - bpp..].as_ptr());

        // Compute Paeth predictor for each byte
        let predicted = paeth_predict_neon(left, above, upper_left);
        let diff = vsubq_u8(curr, predicted);

        let mut buf = [0u8; 16];
        vst1q_u8(buf.as_mut_ptr(), diff);
        output.extend_from_slice(&buf);
        i += 16;
    }

    // Handle remaining bytes with scalar
    while i < len {
        let left = row[i - bpp];
        let above = prev_row[i];
        let upper_left = prev_row[i - bpp];
        let predicted = fallback_paeth_predictor(left, above, upper_left);
        output.push(row[i].wrapping_sub(predicted));
        i += 1;
    }
}

/// Compute Paeth predictor for 16 bytes using NEON.
#[inline]
#[target_feature(enable = "neon")]
unsafe fn paeth_predict_neon(
    left: uint8x16_t,
    above: uint8x16_t,
    upper_left: uint8x16_t,
) -> uint8x16_t {
    // Widen to i16 to avoid overflow in p = a + b - c
    let a_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(left)));
    let a_hi = vreinterpretq_s16_u16(vmovl_high_u8(left));
    let b_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(above)));
    let b_hi = vreinterpretq_s16_u16(vmovl_high_u8(above));
    let c_lo = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(upper_left)));
    let c_hi = vreinterpretq_s16_u16(vmovl_high_u8(upper_left));

    // p = a + b - c
    let p_lo = vsubq_s16(vaddq_s16(a_lo, b_lo), c_lo);
    let p_hi = vsubq_s16(vaddq_s16(a_hi, b_hi), c_hi);

    // pa = |p - a|, pb = |p - b|, pc = |p - c|
    let pa_lo = vabsq_s16(vsubq_s16(p_lo, a_lo));
    let pa_hi = vabsq_s16(vsubq_s16(p_hi, a_hi));
    let pb_lo = vabsq_s16(vsubq_s16(p_lo, b_lo));
    let pb_hi = vabsq_s16(vsubq_s16(p_hi, b_hi));
    let pc_lo = vabsq_s16(vsubq_s16(p_lo, c_lo));
    let pc_hi = vabsq_s16(vsubq_s16(p_hi, c_hi));

    // Select: if pa <= pb && pa <= pc then a, else if pb <= pc then b, else c
    // mask_a = (pa <= pb) & (pa <= pc)
    let mask_a_lo = vandq_u16(vcleq_s16(pa_lo, pb_lo), vcleq_s16(pa_lo, pc_lo));
    let mask_a_hi = vandq_u16(vcleq_s16(pa_hi, pb_hi), vcleq_s16(pa_hi, pc_hi));

    // mask_b = !mask_a & (pb <= pc)
    let mask_b_lo = vandq_u16(vmvnq_u16(mask_a_lo), vcleq_s16(pb_lo, pc_lo));
    let mask_b_hi = vandq_u16(vmvnq_u16(mask_a_hi), vcleq_s16(pb_hi, pc_hi));

    // result = (a & mask_a) | (b & mask_b) | (c & mask_c)
    // where mask_c = !mask_a & !mask_b
    let a_lo_u16 = vreinterpretq_u16_s16(a_lo);
    let a_hi_u16 = vreinterpretq_u16_s16(a_hi);
    let b_lo_u16 = vreinterpretq_u16_s16(b_lo);
    let b_hi_u16 = vreinterpretq_u16_s16(b_hi);
    let c_lo_u16 = vreinterpretq_u16_s16(c_lo);
    let c_hi_u16 = vreinterpretq_u16_s16(c_hi);

    let result_lo = vorrq_u16(
        vorrq_u16(
            vandq_u16(a_lo_u16, mask_a_lo),
            vandq_u16(b_lo_u16, mask_b_lo),
        ),
        vandq_u16(c_lo_u16, vmvnq_u16(vorrq_u16(mask_a_lo, mask_b_lo))),
    );
    let result_hi = vorrq_u16(
        vorrq_u16(
            vandq_u16(a_hi_u16, mask_a_hi),
            vandq_u16(b_hi_u16, mask_b_hi),
        ),
        vandq_u16(c_hi_u16, vmvnq_u16(vorrq_u16(mask_a_hi, mask_b_hi))),
    );

    // Narrow back to u8
    let result_lo_u8 = vmovn_u16(result_lo);
    let result_hi_u8 = vmovn_u16(result_hi);

    vcombine_u8(result_lo_u8, result_hi_u8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simd::fallback;

    #[test]
    fn test_adler32_neon_matches_scalar() {
        let data: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();

        let scalar_result = fallback::adler32(&data);
        let neon_result = unsafe { adler32_neon(&data) };

        assert_eq!(
            scalar_result, neon_result,
            "NEON Adler32 should match scalar"
        );
    }

    #[test]
    fn test_adler32_neon_empty() {
        let data: Vec<u8> = vec![];
        let result = unsafe { adler32_neon(&data) };
        assert_eq!(result, 1, "Adler32 of empty data should be 1");
    }

    #[test]
    fn test_match_length_neon() {
        let data: Vec<u8> = vec![1, 2, 3, 4, 5, 1, 2, 3, 4, 6, 7, 8];
        let len = unsafe { match_length_neon(&data, 0, 5, 5) };
        assert_eq!(len, 4, "Should match first 4 bytes");
    }

    #[test]
    fn test_filter_sub_neon_matches_scalar() {
        let row: Vec<u8> = (0..100).map(|i| (i * 7 % 256) as u8).collect();
        let bpp = 3;

        let mut scalar_out = Vec::new();
        fallback::filter_sub(&row, bpp, &mut scalar_out);

        let mut neon_out = Vec::new();
        unsafe { filter_sub_neon(&row, bpp, &mut neon_out) };

        assert_eq!(scalar_out, neon_out, "NEON filter_sub should match scalar");
    }

    #[test]
    fn test_filter_up_neon_matches_scalar() {
        let row: Vec<u8> = (0..100).map(|i| (i * 7 % 256) as u8).collect();
        let prev: Vec<u8> = (0..100).map(|i| (i * 11 % 256) as u8).collect();

        let mut scalar_out = Vec::new();
        fallback::filter_up(&row, &prev, &mut scalar_out);

        let mut neon_out = Vec::new();
        unsafe { filter_up_neon(&row, &prev, &mut neon_out) };

        assert_eq!(scalar_out, neon_out, "NEON filter_up should match scalar");
    }

    #[test]
    fn test_score_filter_neon_matches_scalar() {
        let data: Vec<u8> = (0..200).map(|i| i as u8).collect();

        let scalar = fallback::score_filter(&data);
        let neon = unsafe { score_filter_neon(&data) };

        assert_eq!(scalar, neon, "NEON score_filter should match scalar");
    }

    #[test]
    fn test_filter_average_neon_matches_scalar() {
        let row: Vec<u8> = (0..100).map(|i| (i * 7 % 256) as u8).collect();
        let prev: Vec<u8> = (0..100).map(|i| (i * 11 % 256) as u8).collect();
        let bpp = 3;

        let mut scalar_out = Vec::new();
        fallback::filter_average(&row, &prev, bpp, &mut scalar_out);

        let mut neon_out = Vec::new();
        unsafe { filter_average_neon(&row, &prev, bpp, &mut neon_out) };

        assert_eq!(
            scalar_out, neon_out,
            "NEON filter_average should match scalar"
        );
    }

    #[test]
    fn test_filter_paeth_neon_matches_scalar() {
        let row: Vec<u8> = (0..100).map(|i| (i * 7 % 256) as u8).collect();
        let prev: Vec<u8> = (0..100).map(|i| (i * 11 % 256) as u8).collect();
        let bpp = 3;

        let mut scalar_out = Vec::new();
        fallback::filter_paeth(&row, &prev, bpp, &mut scalar_out);

        let mut neon_out = Vec::new();
        unsafe { filter_paeth_neon(&row, &prev, bpp, &mut neon_out) };

        assert_eq!(
            scalar_out, neon_out,
            "NEON filter_paeth should match scalar"
        );
    }

    #[test]
    fn test_filter_paeth_neon_tie_breaking() {
        // Test case where pa == pb == pc (all equal values trigger tie-breaking)
        // With a = b = c = 128, we have p = 128 + 128 - 128 = 128
        // pa = |128 - 128| = 0, pb = 0, pc = 0
        // Since pa <= pb && pa <= pc, we should choose a (128)
        let row = vec![128u8; 32];
        let prev = vec![128u8; 32];
        let bpp = 3;

        let mut scalar_out = Vec::new();
        fallback::filter_paeth(&row, &prev, bpp, &mut scalar_out);

        let mut neon_out = Vec::new();
        unsafe { filter_paeth_neon(&row, &prev, bpp, &mut neon_out) };

        assert_eq!(
            scalar_out, neon_out,
            "NEON filter_paeth should handle tie-breaking correctly"
        );
    }
}
