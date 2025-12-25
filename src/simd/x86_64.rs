//! x86_64 SIMD implementations using SSE2, SSSE3, SSE4.2, and PCLMULQDQ.

use crate::simd::fallback::fallback_paeth_predictor;
use std::arch::x86_64::*;

// ============================================================================
// CRC32 using PCLMULQDQ (carry-less multiplication)
// ============================================================================

/// Pre-computed constants for CRC32 using the ISO-HDLC polynomial (0x04C11DB7).
/// These are the "folding" constants used for PCLMULQDQ-based CRC computation.
mod crc32_constants {
    /// Fold by 4 constants (for 64-byte chunks)
    pub const K1K2: (u64, u64) = (0x154442bd4, 0x1c6e41596);
    /// Fold by 1 constants (for 16-byte chunks)
    pub const K3K4: (u64, u64) = (0x1751997d0, 0x0ccaa009e);
    /// Final reduction constants
    pub const K5K6: (u64, u64) = (0x163cd6124, 0x1db710640);
    /// Barrett reduction constant and polynomial
    pub const POLY_MU: (u64, u64) = (0x1f7011641, 0x1db710641);
}

/// Compute CRC32 using PCLMULQDQ instruction for the ISO-HDLC polynomial.
///
/// This implementation uses carry-less multiplication to compute CRC32
/// with the correct polynomial (0x04C11DB7) required by PNG/zlib.
///
/// # Safety
/// Caller must ensure PCLMULQDQ and SSE4.1 are available on the current CPU.
#[target_feature(enable = "pclmulqdq", enable = "sse4.1")]
pub unsafe fn crc32_pclmulqdq(data: &[u8]) -> u32 {
    // For small inputs, use the scalar fallback
    if data.len() < 64 {
        return crate::simd::fallback::crc32(data);
    }

    let mut crc = !0u32;
    let mut remaining = data;

    // Align to 16-byte boundary if needed (process bytes one at a time)
    let align_offset = remaining.as_ptr().align_offset(16);
    if align_offset > 0 && align_offset <= remaining.len() {
        for &byte in &remaining[..align_offset] {
            crc = crc32_table_byte(crc, byte);
        }
        remaining = &remaining[align_offset..];
    }

    // Need at least 64 bytes for the folding loop
    if remaining.len() >= 64 {
        // Initialize four 128-bit accumulators with the first 64 bytes XORed with CRC
        let mut x0 = _mm_loadu_si128(remaining.as_ptr() as *const __m128i);
        let mut x1 = _mm_loadu_si128(remaining.as_ptr().add(16) as *const __m128i);
        let mut x2 = _mm_loadu_si128(remaining.as_ptr().add(32) as *const __m128i);
        let mut x3 = _mm_loadu_si128(remaining.as_ptr().add(48) as *const __m128i);

        // XOR the CRC into the first accumulator
        let crc_xmm = _mm_cvtsi32_si128(crc as i32);
        x0 = _mm_xor_si128(x0, crc_xmm);
        remaining = &remaining[64..];

        // Load fold-by-4 constants
        let k1k2 = _mm_set_epi64x(
            crc32_constants::K1K2.1 as i64,
            crc32_constants::K1K2.0 as i64,
        );

        // Fold 64 bytes at a time
        while remaining.len() >= 64 {
            x0 = fold_16(
                x0,
                _mm_loadu_si128(remaining.as_ptr() as *const __m128i),
                k1k2,
            );
            x1 = fold_16(
                x1,
                _mm_loadu_si128(remaining.as_ptr().add(16) as *const __m128i),
                k1k2,
            );
            x2 = fold_16(
                x2,
                _mm_loadu_si128(remaining.as_ptr().add(32) as *const __m128i),
                k1k2,
            );
            x3 = fold_16(
                x3,
                _mm_loadu_si128(remaining.as_ptr().add(48) as *const __m128i),
                k1k2,
            );
            remaining = &remaining[64..];
        }

        // Fold down to a single 128-bit value
        let k3k4 = _mm_set_epi64x(
            crc32_constants::K3K4.1 as i64,
            crc32_constants::K3K4.0 as i64,
        );

        x0 = fold_16(x0, x1, k3k4);
        x0 = fold_16(x0, x2, k3k4);
        x0 = fold_16(x0, x3, k3k4);

        // Fold remaining 16-byte chunks
        while remaining.len() >= 16 {
            let next = _mm_loadu_si128(remaining.as_ptr() as *const __m128i);
            x0 = fold_16(x0, next, k3k4);
            remaining = &remaining[16..];
        }

        // Final reduction from 128 bits to 32 bits
        crc = reduce_128_to_32(x0);
    }

    // Process remaining bytes with scalar code
    for &byte in remaining {
        crc = crc32_table_byte(crc, byte);
    }

    !crc
}

/// Fold 16 bytes into the accumulator using PCLMULQDQ.
///
/// Computes: (acc.low * K1) XOR (acc.high * K2) XOR data
/// where K1 = k[63:0] and K2 = k[127:64]
#[inline]
#[target_feature(enable = "pclmulqdq")]
unsafe fn fold_16(acc: __m128i, data: __m128i, k: __m128i) -> __m128i {
    // Multiply low 64 bits of acc by low 64 bits of k (K1)
    let lo = _mm_clmulepi64_si128(acc, k, 0x00);
    // Multiply high 64 bits of acc by high 64 bits of k (K2)
    let hi = _mm_clmulepi64_si128(acc, k, 0x11);
    // XOR together with new data
    _mm_xor_si128(_mm_xor_si128(lo, hi), data)
}

/// Reduce 128-bit value to 32-bit CRC using Barrett reduction.
///
/// This follows the algorithm from Intel's "Fast CRC Computation Using PCLMULQDQ":
/// 1. Fold 128 -> 64 bits using x.high * K5
/// 2. Fold 64 -> 32 bits using result.low * K6
/// 3. Barrett reduction to get final 32-bit CRC
#[inline]
#[target_feature(enable = "pclmulqdq", enable = "sse4.1")]
unsafe fn reduce_128_to_32(x: __m128i) -> u32 {
    let k5k6 = _mm_set_epi64x(
        crc32_constants::K5K6.1 as i64, // K6 in high 64 bits
        crc32_constants::K5K6.0 as i64, // K5 in low 64 bits
    );
    let poly_mu = _mm_set_epi64x(
        crc32_constants::POLY_MU.1 as i64, // poly in high 64 bits
        crc32_constants::POLY_MU.0 as i64, // mu in low 64 bits
    );
    let mask32 = _mm_set_epi32(0, 0, 0, -1);

    // Step 1: Fold 128 -> 64 bits
    // Multiply x.high by K5, XOR with x
    let t0 = _mm_clmulepi64_si128(x, k5k6, 0x01); // x[127:64] * k5k6[63:0] = x.high * K5
    let crc = _mm_xor_si128(t0, x);
    // Now crc[63:0] contains the 64-bit intermediate

    // Step 2: Fold 64 -> 32 bits
    // Multiply crc.low by K6, XOR low 32 bits of result with high 32 bits of crc
    let t1 = _mm_clmulepi64_si128(_mm_and_si128(crc, mask32), k5k6, 0x10); // crc[31:0] * K6
    let crc = _mm_xor_si128(_mm_srli_si128(crc, 4), t1); // crc[63:32] XOR t1
                                                         // Now the 32-bit value to reduce is at crc[31:0], with extra bits in [63:32]

    // Step 3: Barrett reduction
    // T1 = floor(crc[31:0] / x^32) * mu = crc[31:0] * mu, take high part
    let t2 = _mm_clmulepi64_si128(_mm_and_si128(crc, mask32), poly_mu, 0x00); // crc[31:0] * mu
                                                                              // T2 = floor(T1 / x^32) * P = T1[63:32] * P
    let t2_high = _mm_srli_si128(t2, 4);
    let t3 = _mm_clmulepi64_si128(_mm_and_si128(t2_high, mask32), poly_mu, 0x10); // t2[63:32] * poly
                                                                                  // CRC = (crc XOR T2)[31:0]
    let result = _mm_xor_si128(crc, t3);

    _mm_extract_epi32(result, 0) as u32
}

/// CRC32 table lookup for a single byte.
#[inline]
fn crc32_table_byte(crc: u32, byte: u8) -> u32 {
    const CRC_TABLE: [u32; 256] = {
        let mut table = [0u32; 256];
        let mut i = 0;
        while i < 256 {
            let mut c = i as u32;
            let mut j = 0;
            while j < 8 {
                if c & 1 != 0 {
                    c = (c >> 1) ^ 0xEDB88320;
                } else {
                    c >>= 1;
                }
                j += 1;
            }
            table[i] = c;
            i += 1;
        }
        table
    };

    let index = ((crc ^ byte as u32) & 0xFF) as usize;
    (crc >> 8) ^ CRC_TABLE[index]
}

// ============================================================================
// Adler-32 Implementations
// ============================================================================

/// Compute Adler-32 checksum using SSSE3 instructions.
///
/// Processes 16 bytes at a time for improved throughput.
///
/// # Safety
/// Caller must ensure SSSE3 is available on the current CPU.
#[target_feature(enable = "ssse3")]
pub unsafe fn adler32_ssse3(data: &[u8]) -> u32 {
    const MOD_ADLER: u32 = 65_521;
    // NMAX for 16-byte chunks: largest n where 255*n*(n+1)/2 + (n+1)*65520 < 2^32
    // For 16-byte processing, we use a smaller block size to be safe
    const BLOCK_SIZE: usize = 5552 / 16 * 16;

    let mut s1: u32 = 1;
    let mut s2: u32 = 0;

    let mut remaining = data;

    while remaining.len() >= BLOCK_SIZE {
        let (block, rest) = remaining.split_at(BLOCK_SIZE);
        let (new_s1, new_s2) = adler32_block_ssse3(block, s1, s2);
        s1 = new_s1 % MOD_ADLER;
        s2 = new_s2 % MOD_ADLER;
        remaining = rest;
    }

    // Process remaining complete 16-byte chunks
    if remaining.len() >= 16 {
        let chunk_count = remaining.len() / 16 * 16;
        let (block, rest) = remaining.split_at(chunk_count);
        let (new_s1, new_s2) = adler32_block_ssse3(block, s1, s2);
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

/// Process a block of data for Adler-32 using SSSE3.
#[target_feature(enable = "ssse3")]
unsafe fn adler32_block_ssse3(data: &[u8], mut s1: u32, mut s2: u32) -> (u32, u32) {
    // Weights for s2 accumulation within a 16-byte chunk
    // s2 += 16*b[0] + 15*b[1] + ... + 1*b[15]
    let weights = _mm_setr_epi8(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
    let zeros = _mm_setzero_si128();

    let mut vs1 = _mm_setzero_si128(); // accumulator for s1 (sum of bytes)
    let mut vs2 = _mm_setzero_si128(); // accumulator for s2 (weighted sum)
    let mut vs1_total = _mm_setzero_si128(); // running s1 total for s2 calculation

    for chunk in data.chunks_exact(16) {
        let v = _mm_loadu_si128(chunk.as_ptr() as *const __m128i);

        // Add this chunk's contribution to s2 based on previous s1 total
        // s2 += s1 * 16 (for each byte position)
        vs2 = _mm_add_epi32(vs2, _mm_slli_epi32(vs1_total, 4));

        // Compute sum of bytes for s1 using SAD against zero
        let sad = _mm_sad_epu8(v, zeros);
        vs1 = _mm_add_epi32(vs1, sad);
        vs1_total = _mm_add_epi32(vs1_total, sad);

        // Compute weighted sum for s2
        // Multiply bytes by weights and accumulate
        let v_lo = _mm_unpacklo_epi8(v, zeros);
        let v_hi = _mm_unpackhi_epi8(v, zeros);
        let w_lo = _mm_unpacklo_epi8(weights, zeros);
        let w_hi = _mm_unpackhi_epi8(weights, zeros);

        let prod_lo = _mm_madd_epi16(v_lo, w_lo);
        let prod_hi = _mm_madd_epi16(v_hi, w_hi);
        let weighted_sum = _mm_add_epi32(prod_lo, prod_hi);
        vs2 = _mm_add_epi32(vs2, weighted_sum);
    }

    // Horizontal sum of vs1
    let vs1_hi = _mm_shuffle_epi32(vs1, 0b00_00_11_10);
    let vs1_sum = _mm_add_epi32(vs1, vs1_hi);
    s1 += _mm_cvtsi128_si32(vs1_sum) as u32;

    // Horizontal sum of vs2
    let vs2_1 = _mm_shuffle_epi32(vs2, 0b00_00_11_10);
    let vs2_2 = _mm_add_epi32(vs2, vs2_1);
    let vs2_3 = _mm_shuffle_epi32(vs2_2, 0b00_00_00_01);
    let vs2_sum = _mm_add_epi32(vs2_2, vs2_3);
    s2 += _mm_cvtsi128_si32(vs2_sum) as u32;

    (s1, s2)
}

/// Compute CRC32 using hardware CRC32 instructions (SSE4.2).
///
/// # Safety
/// Caller must ensure SSE4.2 is available on the current CPU.
#[target_feature(enable = "sse4.2")]
pub unsafe fn crc32_hw(data: &[u8]) -> u32 {
    let mut crc = !0u32;
    let mut remaining = data;

    // Process 8 bytes at a time
    while remaining.len() >= 8 {
        let val = u64::from_le_bytes(remaining[..8].try_into().unwrap());
        crc = _mm_crc32_u64(crc as u64, val) as u32;
        remaining = &remaining[8..];
    }

    // Process 4 bytes
    if remaining.len() >= 4 {
        let val = u32::from_le_bytes(remaining[..4].try_into().unwrap());
        crc = _mm_crc32_u32(crc, val);
        remaining = &remaining[4..];
    }

    // Process 2 bytes
    if remaining.len() >= 2 {
        let val = u16::from_le_bytes(remaining[..2].try_into().unwrap());
        crc = _mm_crc32_u16(crc, val);
        remaining = &remaining[2..];
    }

    // Process remaining byte
    if !remaining.is_empty() {
        crc = _mm_crc32_u8(crc, remaining[0]);
    }

    !crc
}

/// Compute match length using SSE2 16-byte comparison.
///
/// # Safety
/// Caller must ensure SSE2 is available on the current CPU.
#[target_feature(enable = "sse2")]
pub unsafe fn match_length_sse2(data: &[u8], pos1: usize, pos2: usize, max_len: usize) -> usize {
    let mut length = 0;

    // Compare 16 bytes at a time
    while length + 16 <= max_len {
        let a = _mm_loadu_si128(data[pos1 + length..].as_ptr() as *const __m128i);
        let b = _mm_loadu_si128(data[pos2 + length..].as_ptr() as *const __m128i);
        let cmp = _mm_cmpeq_epi8(a, b);
        let mask = _mm_movemask_epi8(cmp) as u32;

        if mask != 0xFFFF {
            // Found a mismatch - count trailing ones (matching bytes)
            return length + (!mask).trailing_zeros() as usize;
        }
        length += 16;
    }

    // Handle remaining bytes with u64 comparison
    while length + 8 <= max_len {
        let a = u64::from_ne_bytes(data[pos1 + length..pos1 + length + 8].try_into().unwrap());
        let b = u64::from_ne_bytes(data[pos2 + length..pos2 + length + 8].try_into().unwrap());
        if a != b {
            let xor = a ^ b;
            #[cfg(target_endian = "little")]
            {
                return length + (xor.trailing_zeros() / 8) as usize;
            }
            #[cfg(target_endian = "big")]
            {
                return length + (xor.leading_zeros() / 8) as usize;
            }
        }
        length += 8;
    }

    // Handle remaining bytes
    while length < max_len && data[pos1 + length] == data[pos2 + length] {
        length += 1;
    }

    length
}

/// Compute match length using AVX2 32-byte comparison.
///
/// # Safety
/// Caller must ensure AVX2 is available on the current CPU.
#[target_feature(enable = "avx2")]
pub unsafe fn match_length_avx2(data: &[u8], pos1: usize, pos2: usize, max_len: usize) -> usize {
    let mut length = 0;

    // Compare 32 bytes at a time
    while length + 32 <= max_len {
        let a = _mm256_loadu_si256(data[pos1 + length..].as_ptr() as *const __m256i);
        let b = _mm256_loadu_si256(data[pos2 + length..].as_ptr() as *const __m256i);
        let cmp = _mm256_cmpeq_epi8(a, b);
        let mask = _mm256_movemask_epi8(cmp) as u32;

        if mask != 0xFFFF_FFFF {
            // Find first differing byte
            let diff = !mask;
            return length + diff.trailing_zeros() as usize;
        }
        length += 32;
    }

    // Fall back to SSE2 for remaining bytes (at most 31 bytes)
    if length < max_len {
        length + match_length_sse2(data, pos1 + length, pos2 + length, max_len - length)
    } else {
        length
    }
}

/// Compute Adler-32 checksum using AVX2 instructions (32-byte chunks).
///
/// # Safety
/// Caller must ensure AVX2 is available on the current CPU.
#[target_feature(enable = "avx2")]
pub unsafe fn adler32_avx2(data: &[u8]) -> u32 {
    const MOD_ADLER: u32 = 65_521;
    const NMAX: usize = 5552; // same as scalar/SSSE3 path

    let mut s1: u32 = 1;
    let mut s2: u32 = 0;
    let mut processed = 0usize;

    let zeros = _mm256_setzero_si256();
    // weights 32..1
    let weights = _mm256_setr_epi8(
        32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,
        9, 8, 7, 6, 5, 4, 3, 2, 1,
    );

    let mut chunks = data.chunks_exact(32);
    for chunk in &mut chunks {
        // Load chunk
        let v = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);

        // Sum of bytes via SAD
        let sad = _mm256_sad_epu8(v, zeros);
        let mut sad_buf = [0i64; 4];
        _mm256_storeu_si256(sad_buf.as_mut_ptr() as *mut __m256i, sad);
        let chunk_sum = (sad_buf[0] + sad_buf[1] + sad_buf[2] + sad_buf[3]) as u32;

        // Weighted sum for s2
        let v_lo = _mm256_unpacklo_epi8(v, zeros);
        let v_hi = _mm256_unpackhi_epi8(v, zeros);
        let w_lo = _mm256_unpacklo_epi8(weights, zeros);
        let w_hi = _mm256_unpackhi_epi8(weights, zeros);

        let prod_lo = _mm256_madd_epi16(v_lo, w_lo); // 8 i32 lanes
        let prod_hi = _mm256_madd_epi16(v_hi, w_hi); // 8 i32 lanes
        let sum_prod = _mm256_add_epi32(prod_lo, prod_hi);

        // Horizontal sum of sum_prod
        let tmp1 = _mm256_hadd_epi32(sum_prod, sum_prod);
        let tmp2 = _mm256_hadd_epi32(tmp1, tmp1);
        let mut prod_buf = [0i32; 8];
        _mm256_storeu_si256(prod_buf.as_mut_ptr() as *mut __m256i, tmp2);
        let weighted_sum = (prod_buf[0] as i64 + prod_buf[4] as i64) as u32;

        s2 = s2.wrapping_add(s1.wrapping_mul(32));
        s2 = s2.wrapping_add(weighted_sum);
        s1 = s1.wrapping_add(chunk_sum);

        processed += 32;
        if processed >= NMAX {
            s1 %= MOD_ADLER;
            s2 %= MOD_ADLER;
            processed = 0;
        }
    }

    // Remainder (less than 32 bytes) scalar
    for &b in chunks.remainder() {
        s1 = s1.wrapping_add(b as u32);
        s2 = s2.wrapping_add(s1);
        processed += 1;
        if processed >= NMAX {
            s1 %= MOD_ADLER;
            s2 %= MOD_ADLER;
            processed = 0;
        }
    }

    s1 %= MOD_ADLER;
    s2 %= MOD_ADLER;

    (s2 << 16) | s1
}

/// Score a filtered row using SSE2 SAD instruction.
///
/// # Safety
/// Caller must ensure SSE2 is available on the current CPU.
#[target_feature(enable = "sse2")]
pub unsafe fn score_filter_sse2(filtered: &[u8]) -> u64 {
    let mut sum = _mm_setzero_si128();
    let mut remaining = filtered;

    // Process 16 bytes at a time using SAD
    while remaining.len() >= 16 {
        let v = _mm_loadu_si128(remaining.as_ptr() as *const __m128i);

        // For sum of absolute values where values are treated as signed:
        // We need |x| for signed interpretation. For bytes 0-127, |x| = x.
        // For bytes 128-255 (signed -128 to -1), |x| = 256 - x.
        // _mm_sad_epu8 computes sum of |a-b| treating as unsigned.
        // If we use zeros, we get sum of values (treating as unsigned 0-255).
        // But we want signed absolute values.

        // Convert to signed interpretation:
        // For values 0-127: keep as is
        // For values 128-255: negate (256 - x)
        let high_bit = _mm_set1_epi8(-128i8); // 0x80
        let _is_negative = _mm_cmpgt_epi8(high_bit, v); // true for 0-127, false for 128-255

        // Compute absolute value using: abs(x) = (x XOR mask) - mask where mask = x >> 7
        // But for bytes this is: for negative bytes, flip and add 1
        // Simpler: abs(x) = max(x, -x) but we don't have signed max easily

        // Alternative: treat as unsigned, values 128-255 become their unsigned value
        // The "absolute value" for filter scoring typically uses unsigned interpretation
        // where 128-255 are considered large positive, not negative.
        // Actually, looking at the original code, it treats bytes as signed i8
        // and takes unsigned_abs(). So 255 as i8 is -1, abs is 1.

        // Use a different approach: XOR with 0x80 to convert to "signed magnitude"
        // then SAD against 0x80
        let offset = _mm_set1_epi8(-128i8); // 0x80
        let adjusted = _mm_xor_si128(v, offset);
        let sad = _mm_sad_epu8(adjusted, offset);
        sum = _mm_add_epi64(sum, sad);

        remaining = &remaining[16..];
    }

    // Horizontal sum
    let high = _mm_shuffle_epi32(sum, 0b00_00_11_10);
    let total = _mm_add_epi64(sum, high);
    let mut result = _mm_cvtsi128_si64(total) as u64;

    // Process remaining bytes with scalar
    for &b in remaining {
        result += (b as i8).unsigned_abs() as u64;
    }

    result
}

/// Score a filtered row using AVX2 SAD instruction (32-byte chunks).
///
/// # Safety
/// Caller must ensure AVX2 is available on the current CPU.
#[target_feature(enable = "avx2")]
pub unsafe fn score_filter_avx2(filtered: &[u8]) -> u64 {
    let offset = _mm256_set1_epi8(-128i8); // 0x80
    let mut acc = _mm256_setzero_si256();
    let mut remaining = filtered;

    while remaining.len() >= 32 {
        let v = _mm256_loadu_si256(remaining.as_ptr() as *const __m256i);
        // Convert signed to unsigned magnitude by XORing with 0x80, then SAD vs 0x80.
        let adjusted = _mm256_xor_si256(v, offset);
        let sad = _mm256_sad_epu8(adjusted, offset); // produces four u64 lanes
        acc = _mm256_add_epi64(acc, sad);
        remaining = &remaining[32..];
    }

    // Horizontal sum of acc
    let mut buf = [0u64; 4];
    _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, acc);
    let mut result = buf.iter().sum::<u64>();

    // Remainder scalar
    for &b in remaining {
        result += (b as i8).unsigned_abs() as u64;
    }

    result
}

/// Apply Sub filter using SSE2.
///
/// # Safety
/// Caller must ensure SSE2 is available on the current CPU.
#[target_feature(enable = "sse2")]
pub unsafe fn filter_sub_sse2(row: &[u8], bpp: usize, output: &mut Vec<u8>) {
    // First bpp bytes have no left neighbor
    for &byte in &row[..bpp] {
        output.push(byte);
    }

    // For remaining bytes, we need row[i] - row[i-bpp]
    // This is tricky for arbitrary bpp because we need to shift by bpp bytes
    // For now, use scalar for the general case but optimize common cases

    let remaining = &row[bpp..];
    let left = &row[..row.len() - bpp];

    // Process 16 bytes at a time when possible
    let mut i = 0;
    let len = remaining.len();

    while i + 16 <= len {
        let curr = _mm_loadu_si128(remaining[i..].as_ptr() as *const __m128i);
        let prev = _mm_loadu_si128(left[i..].as_ptr() as *const __m128i);
        let diff = _mm_sub_epi8(curr, prev);

        // Store result
        let mut buf = [0u8; 16];
        _mm_storeu_si128(buf.as_mut_ptr() as *mut __m128i, diff);
        output.extend_from_slice(&buf);
        i += 16;
    }

    // Handle remaining bytes
    while i < len {
        output.push(remaining[i].wrapping_sub(left[i]));
        i += 1;
    }
}

/// Apply Up filter using SSE2.
///
/// # Safety
/// Caller must ensure SSE2 is available on the current CPU.
#[target_feature(enable = "sse2")]
pub unsafe fn filter_up_sse2(row: &[u8], prev_row: &[u8], output: &mut Vec<u8>) {
    let len = row.len();
    let mut i = 0;

    // Process 16 bytes at a time
    while i + 16 <= len {
        let curr = _mm_loadu_si128(row[i..].as_ptr() as *const __m128i);
        let prev = _mm_loadu_si128(prev_row[i..].as_ptr() as *const __m128i);
        let diff = _mm_sub_epi8(curr, prev);

        let mut buf = [0u8; 16];
        _mm_storeu_si128(buf.as_mut_ptr() as *mut __m128i, diff);
        output.extend_from_slice(&buf);
        i += 16;
    }

    // Handle remaining bytes
    while i < len {
        output.push(row[i].wrapping_sub(prev_row[i]));
        i += 1;
    }
}

/// Apply Sub filter using AVX2 (32 bytes at a time).
///
/// # Safety
/// Caller must ensure AVX2 is available on the current CPU.
#[target_feature(enable = "avx2")]
pub unsafe fn filter_sub_avx2(row: &[u8], bpp: usize, output: &mut Vec<u8>) {
    let len = row.len();
    output.reserve(len);

    // First bpp bytes unchanged
    output.extend_from_slice(&row[..bpp.min(len)]);

    if len <= bpp {
        return;
    }

    let remaining = &row[bpp..];
    let left = &row[..len - bpp];

    let mut i = 0;
    let rem_len = remaining.len();

    while i + 32 <= rem_len {
        let curr = _mm256_loadu_si256(remaining[i..].as_ptr() as *const __m256i);
        let prev = _mm256_loadu_si256(left[i..].as_ptr() as *const __m256i);
        let diff = _mm256_sub_epi8(curr, prev);

        let mut buf = [0u8; 32];
        _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, diff);
        output.extend_from_slice(&buf);
        i += 32;
    }

    while i < rem_len {
        output.push(remaining[i].wrapping_sub(left[i]));
        i += 1;
    }
}

#[inline]
unsafe fn abs_epi16_sse2(v: __m128i) -> __m128i {
    let sign = _mm_srai_epi16(v, 15);
    let xor = _mm_xor_si128(v, sign);
    _mm_sub_epi16(xor, sign)
}

#[inline]
unsafe fn paeth_predict_128(left: __m128i, above: __m128i, upper_left: __m128i) -> __m128i {
    let zero = _mm_setzero_si128();

    let a_lo = _mm_unpacklo_epi8(left, zero);
    let b_lo = _mm_unpacklo_epi8(above, zero);
    let c_lo = _mm_unpacklo_epi8(upper_left, zero);

    let a_hi = _mm_unpackhi_epi8(left, zero);
    let b_hi = _mm_unpackhi_epi8(above, zero);
    let c_hi = _mm_unpackhi_epi8(upper_left, zero);

    let p_lo = _mm_sub_epi16(_mm_add_epi16(a_lo, b_lo), c_lo);
    let p_hi = _mm_sub_epi16(_mm_add_epi16(a_hi, b_hi), c_hi);

    let pa_lo = abs_epi16_sse2(_mm_sub_epi16(p_lo, a_lo));
    let pb_lo = abs_epi16_sse2(_mm_sub_epi16(p_lo, b_lo));
    let pc_lo = abs_epi16_sse2(_mm_sub_epi16(p_lo, c_lo));

    let pa_hi = abs_epi16_sse2(_mm_sub_epi16(p_hi, a_hi));
    let pb_hi = abs_epi16_sse2(_mm_sub_epi16(p_hi, b_hi));
    let pc_hi = abs_epi16_sse2(_mm_sub_epi16(p_hi, c_hi));

    // PNG Paeth: choose a if pa <= pb && pa <= pc; else if pb <= pc choose b; else c
    // SSE2 has cmpgt but not cmple, so we use: (a <= b) = NOT(a > b)
    // mask_a = (pa <= pb) && (pa <= pc) = NOT(pa > pb) && NOT(pa > pc)
    let pa_le_pb_lo = _mm_andnot_si128(_mm_cmpgt_epi16(pa_lo, pb_lo), _mm_set1_epi16(-1));
    let pa_le_pc_lo = _mm_andnot_si128(_mm_cmpgt_epi16(pa_lo, pc_lo), _mm_set1_epi16(-1));
    let mask_a_lo = _mm_and_si128(pa_le_pb_lo, pa_le_pc_lo);

    // mask_b = NOT(mask_a) && (pb <= pc)
    let pb_le_pc_lo = _mm_andnot_si128(_mm_cmpgt_epi16(pb_lo, pc_lo), _mm_set1_epi16(-1));
    let mask_b_lo = _mm_andnot_si128(mask_a_lo, pb_le_pc_lo);

    // mask_c = NOT(mask_a) && NOT(mask_b)
    let mask_c_lo = _mm_andnot_si128(_mm_or_si128(mask_a_lo, mask_b_lo), _mm_set1_epi16(-1));

    let pa_le_pb_hi = _mm_andnot_si128(_mm_cmpgt_epi16(pa_hi, pb_hi), _mm_set1_epi16(-1));
    let pa_le_pc_hi = _mm_andnot_si128(_mm_cmpgt_epi16(pa_hi, pc_hi), _mm_set1_epi16(-1));
    let mask_a_hi = _mm_and_si128(pa_le_pb_hi, pa_le_pc_hi);

    let pb_le_pc_hi = _mm_andnot_si128(_mm_cmpgt_epi16(pb_hi, pc_hi), _mm_set1_epi16(-1));
    let mask_b_hi = _mm_andnot_si128(mask_a_hi, pb_le_pc_hi);

    let mask_c_hi = _mm_andnot_si128(_mm_or_si128(mask_a_hi, mask_b_hi), _mm_set1_epi16(-1));

    let pred_lo = _mm_or_si128(
        _mm_or_si128(
            _mm_and_si128(mask_a_lo, a_lo),
            _mm_and_si128(mask_b_lo, b_lo),
        ),
        _mm_and_si128(mask_c_lo, c_lo),
    );
    let pred_hi = _mm_or_si128(
        _mm_or_si128(
            _mm_and_si128(mask_a_hi, a_hi),
            _mm_and_si128(mask_b_hi, b_hi),
        ),
        _mm_and_si128(mask_c_hi, c_hi),
    );

    _mm_packus_epi16(pred_lo, pred_hi)
}

/// Apply Paeth filter using SSE2 (experimental; currently gated off in dispatch).
///
/// # Safety
///
/// The caller must ensure that the CPU supports SSE2 instructions.
#[target_feature(enable = "sse2")]
pub unsafe fn filter_paeth_sse2(row: &[u8], prev_row: &[u8], bpp: usize, output: &mut Vec<u8>) {
    let len = row.len();
    output.reserve(len);

    // First bpp bytes scalar
    for i in 0..bpp.min(len) {
        let left = 0;
        let above = prev_row[i];
        let upper_left = 0;
        let predicted = fallback_paeth_predictor(left, above, upper_left);
        output.push(row[i].wrapping_sub(predicted));
    }

    if len <= bpp {
        return;
    }

    let mut i = bpp;
    while i + 16 <= len {
        let curr = _mm_loadu_si128(row[i..].as_ptr() as *const __m128i);
        let left = _mm_loadu_si128(row[i - bpp..].as_ptr() as *const __m128i);
        let above = _mm_loadu_si128(prev_row[i..].as_ptr() as *const __m128i);
        let upper_left = _mm_loadu_si128(prev_row[i - bpp..].as_ptr() as *const __m128i);

        let predicted = paeth_predict_128(left, above, upper_left);
        let diff = _mm_sub_epi8(curr, predicted);

        let mut buf = [0u8; 16];
        _mm_storeu_si128(buf.as_mut_ptr() as *mut __m128i, diff);
        output.extend_from_slice(&buf);
        i += 16;
    }

    while i < len {
        let left = row[i - bpp];
        let above = prev_row[i];
        let upper_left = prev_row[i - bpp];
        let predicted = fallback_paeth_predictor(left, above, upper_left);
        output.push(row[i].wrapping_sub(predicted));
        i += 1;
    }
}

/// Apply Paeth filter using AVX2 (experimental; currently gated off in dispatch).
///
/// # Safety
///
/// The caller must ensure that the CPU supports AVX2 instructions.
#[target_feature(enable = "avx2")]
pub unsafe fn filter_paeth_avx2(row: &[u8], prev_row: &[u8], bpp: usize, output: &mut Vec<u8>) {
    let len = row.len();
    output.reserve(len);

    for i in 0..bpp.min(len) {
        let left = 0;
        let above = prev_row[i];
        let upper_left = 0;
        let predicted = fallback_paeth_predictor(left, above, upper_left);
        output.push(row[i].wrapping_sub(predicted));
    }

    if len <= bpp {
        return;
    }

    let mut i = bpp;
    while i + 32 <= len {
        let curr = _mm256_loadu_si256(row[i..].as_ptr() as *const __m256i);
        let left = _mm256_loadu_si256(row[i - bpp..].as_ptr() as *const __m256i);
        let above = _mm256_loadu_si256(prev_row[i..].as_ptr() as *const __m256i);
        let upper_left = _mm256_loadu_si256(prev_row[i - bpp..].as_ptr() as *const __m256i);

        let zero = _mm256_setzero_si256();
        let a_lo = _mm256_unpacklo_epi8(left, zero);
        let b_lo = _mm256_unpacklo_epi8(above, zero);
        let c_lo = _mm256_unpacklo_epi8(upper_left, zero);

        let a_hi = _mm256_unpackhi_epi8(left, zero);
        let b_hi = _mm256_unpackhi_epi8(above, zero);
        let c_hi = _mm256_unpackhi_epi8(upper_left, zero);

        let p_lo = _mm256_sub_epi16(_mm256_add_epi16(a_lo, b_lo), c_lo);
        let p_hi = _mm256_sub_epi16(_mm256_add_epi16(a_hi, b_hi), c_hi);

        let pa_lo = _mm256_abs_epi16(_mm256_sub_epi16(p_lo, a_lo));
        let pb_lo = _mm256_abs_epi16(_mm256_sub_epi16(p_lo, b_lo));
        let pc_lo = _mm256_abs_epi16(_mm256_sub_epi16(p_lo, c_lo));

        let pa_hi = _mm256_abs_epi16(_mm256_sub_epi16(p_hi, a_hi));
        let pb_hi = _mm256_abs_epi16(_mm256_sub_epi16(p_hi, b_hi));
        let pc_hi = _mm256_abs_epi16(_mm256_sub_epi16(p_hi, c_hi));

        // PNG Paeth: choose a if pa <= pb && pa <= pc; else if pb <= pc choose b; else c
        // AVX2 has cmpgt but not cmple, so we use: (a <= b) = NOT(a > b)
        // mask_a = (pa <= pb) && (pa <= pc)
        let pa_le_pb_lo =
            _mm256_andnot_si256(_mm256_cmpgt_epi16(pa_lo, pb_lo), _mm256_set1_epi16(-1));
        let pa_le_pc_lo =
            _mm256_andnot_si256(_mm256_cmpgt_epi16(pa_lo, pc_lo), _mm256_set1_epi16(-1));
        let mask_a_lo = _mm256_and_si256(pa_le_pb_lo, pa_le_pc_lo);

        // mask_b = NOT(mask_a) && (pb <= pc)
        let pb_le_pc_lo =
            _mm256_andnot_si256(_mm256_cmpgt_epi16(pb_lo, pc_lo), _mm256_set1_epi16(-1));
        let mask_b_lo = _mm256_andnot_si256(mask_a_lo, pb_le_pc_lo);

        let mask_c_lo =
            _mm256_andnot_si256(_mm256_or_si256(mask_a_lo, mask_b_lo), _mm256_set1_epi16(-1));

        let pa_le_pb_hi =
            _mm256_andnot_si256(_mm256_cmpgt_epi16(pa_hi, pb_hi), _mm256_set1_epi16(-1));
        let pa_le_pc_hi =
            _mm256_andnot_si256(_mm256_cmpgt_epi16(pa_hi, pc_hi), _mm256_set1_epi16(-1));
        let mask_a_hi = _mm256_and_si256(pa_le_pb_hi, pa_le_pc_hi);

        let pb_le_pc_hi =
            _mm256_andnot_si256(_mm256_cmpgt_epi16(pb_hi, pc_hi), _mm256_set1_epi16(-1));
        let mask_b_hi = _mm256_andnot_si256(mask_a_hi, pb_le_pc_hi);

        let mask_c_hi =
            _mm256_andnot_si256(_mm256_or_si256(mask_a_hi, mask_b_hi), _mm256_set1_epi16(-1));

        let pred_lo = _mm256_or_si256(
            _mm256_or_si256(
                _mm256_and_si256(mask_a_lo, a_lo),
                _mm256_and_si256(mask_b_lo, b_lo),
            ),
            _mm256_and_si256(mask_c_lo, c_lo),
        );
        let pred_hi = _mm256_or_si256(
            _mm256_or_si256(
                _mm256_and_si256(mask_a_hi, a_hi),
                _mm256_and_si256(mask_b_hi, b_hi),
            ),
            _mm256_and_si256(mask_c_hi, c_hi),
        );

        let predicted = _mm256_packus_epi16(pred_lo, pred_hi);
        let diff = _mm256_sub_epi8(curr, predicted);

        let mut buf = [0u8; 32];
        _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, diff);
        output.extend_from_slice(&buf);
        i += 32;
    }

    while i < len {
        let left = row[i - bpp];
        let above = prev_row[i];
        let upper_left = prev_row[i - bpp];
        let predicted = fallback_paeth_predictor(left, above, upper_left);
        output.push(row[i].wrapping_sub(predicted));
        i += 1;
    }
}

/// Apply Up filter using AVX2 (32 bytes at a time).
///
/// # Safety
/// Caller must ensure AVX2 is available on the current CPU.
#[target_feature(enable = "avx2")]
pub unsafe fn filter_up_avx2(row: &[u8], prev_row: &[u8], output: &mut Vec<u8>) {
    let len = row.len();
    output.reserve(len);

    let mut i = 0;
    while i + 32 <= len {
        let curr = _mm256_loadu_si256(row[i..].as_ptr() as *const __m256i);
        let prev = _mm256_loadu_si256(prev_row[i..].as_ptr() as *const __m256i);
        let diff = _mm256_sub_epi8(curr, prev);

        let mut buf = [0u8; 32];
        _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, diff);
        output.extend_from_slice(&buf);
        i += 32;
    }

    while i < len {
        output.push(row[i].wrapping_sub(prev_row[i]));
        i += 1;
    }
}

/// Apply Average filter using AVX2 (32 bytes at a time).
///
/// # Safety
/// Caller must ensure AVX2 is available on the current CPU.
#[target_feature(enable = "avx2")]
pub unsafe fn filter_average_avx2(row: &[u8], prev_row: &[u8], bpp: usize, output: &mut Vec<u8>) {
    let len = row.len();
    output.reserve(len);

    // First bpp bytes: use scalar
    for i in 0..bpp.min(len) {
        let left = 0u8;
        let above = prev_row[i];
        let avg = ((left as u16 + above as u16) / 2) as u8;
        output.push(row[i].wrapping_sub(avg));
    }

    if len <= bpp {
        return;
    }

    let mut i = bpp;
    while i + 32 <= len {
        let curr = _mm256_loadu_si256(row[i..].as_ptr() as *const __m256i);
        let above = _mm256_loadu_si256(prev_row[i..].as_ptr() as *const __m256i);
        let left = _mm256_loadu_si256(row[i - bpp..].as_ptr() as *const __m256i);

        // avg = (left + above) >> 1
        let left_lo = _mm256_unpacklo_epi8(left, _mm256_setzero_si256());
        let left_hi = _mm256_unpackhi_epi8(left, _mm256_setzero_si256());
        let above_lo = _mm256_unpacklo_epi8(above, _mm256_setzero_si256());
        let above_hi = _mm256_unpackhi_epi8(above, _mm256_setzero_si256());

        let avg_lo = _mm256_srli_epi16(_mm256_add_epi16(left_lo, above_lo), 1);
        let avg_hi = _mm256_srli_epi16(_mm256_add_epi16(left_hi, above_hi), 1);
        let avg = _mm256_packus_epi16(avg_lo, avg_hi);

        let diff = _mm256_sub_epi8(curr, avg);

        let mut buf = [0u8; 32];
        _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, diff);
        output.extend_from_slice(&buf);
        i += 32;
    }

    // Remainder scalar
    while i < len {
        let left = if i >= bpp { row[i - bpp] as u16 } else { 0 };
        let above = prev_row[i] as u16;
        let avg = ((left + above) / 2) as u8;
        output.push(row[i].wrapping_sub(avg));
        i += 1;
    }
}
