//! SIMD acceleration module for performance-critical operations.
//!
//! This module provides SIMD-accelerated implementations of:
//! - Adler-32 checksum
//! - CRC32 checksum (using hardware instructions)
//! - PNG filter operations
//! - LZ77 match length comparison
//!
//! The implementations are architecture-specific and fall back to scalar
//! implementations when SIMD is not available.

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

pub mod fallback;

// Re-export the best available implementations based on target architecture

/// Compute Adler-32 checksum using the best available implementation.
#[inline]
pub fn adler32(data: &[u8]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // Safety: We've verified AVX2 is available
            return unsafe { x86_64::adler32_avx2(data) };
        }
        if is_x86_feature_detected!("ssse3") {
            // Safety: We've verified SSSE3 is available
            return unsafe { x86_64::adler32_ssse3(data) };
        }
    }

    fallback::adler32(data)
}

/// Compute CRC32 checksum using the best available implementation.
///
/// Note: PNG uses CRC-32/ISO-HDLC (polynomial 0x04C11DB7), while the x86
/// hardware CRC32 instruction uses CRC32C (polynomial 0x1EDC6F41).
/// Therefore, we use the table-based implementation for correctness.
/// A future optimization could use PCLMULQDQ for the correct polynomial.
#[inline]
pub fn crc32(data: &[u8]) -> u32 {
    // The hardware CRC32 instruction uses a different polynomial (CRC32C)
    // than PNG's CRC32 (ISO-HDLC), so we use the table-based fallback.
    fallback::crc32(data)
}

/// Compute match length between two positions using the best available implementation.
#[inline]
pub fn match_length(data: &[u8], pos1: usize, pos2: usize, max_len: usize) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // Safety: We've verified AVX2 is available
            return unsafe { x86_64::match_length_avx2(data, pos1, pos2, max_len) };
        }
        if is_x86_feature_detected!("sse2") {
            // Safety: We've verified SSE2 is available
            return unsafe { x86_64::match_length_sse2(data, pos1, pos2, max_len) };
        }
    }

    fallback::match_length(data, pos1, pos2, max_len)
}

/// Score a filtered row (sum of absolute values) using the best available implementation.
#[inline]
pub fn score_filter(filtered: &[u8]) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // Safety: We've verified AVX2 is available
            return unsafe { x86_64::score_filter_avx2(filtered) };
        }
        if is_x86_feature_detected!("sse2") {
            // Safety: We've verified SSE2 is available
            return unsafe { x86_64::score_filter_sse2(filtered) };
        }
    }

    fallback::score_filter(filtered)
}

/// Apply Sub filter using the best available implementation.
#[inline]
pub fn filter_sub(row: &[u8], bpp: usize, output: &mut Vec<u8>) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // Safety: We've verified AVX2 is available
            unsafe { x86_64::filter_sub_avx2(row, bpp, output) };
            return;
        }
        if is_x86_feature_detected!("sse2") {
            // Safety: We've verified SSE2 is available
            unsafe { x86_64::filter_sub_sse2(row, bpp, output) };
            return;
        }
    }

    fallback::filter_sub(row, bpp, output)
}

/// Apply Up filter using the best available implementation.
#[inline]
pub fn filter_up(row: &[u8], prev_row: &[u8], output: &mut Vec<u8>) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // Safety: We've verified AVX2 is available
            unsafe { x86_64::filter_up_avx2(row, prev_row, output) };
            return;
        }
        if is_x86_feature_detected!("sse2") {
            // Safety: We've verified SSE2 is available
            unsafe { x86_64::filter_up_sse2(row, prev_row, output) };
            return;
        }
    }

    fallback::filter_up(row, prev_row, output)
}

/// Apply Average filter using the best available implementation.
#[inline]
pub fn filter_average(row: &[u8], prev_row: &[u8], bpp: usize, output: &mut Vec<u8>) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // Safety: We've verified AVX2 is available
            unsafe { x86_64::filter_average_avx2(row, prev_row, bpp, output) };
            return;
        }
    }

    fallback::filter_average(row, prev_row, bpp, output)
}

/// Apply Paeth filter using the best available implementation.
#[inline]
pub fn filter_paeth(row: &[u8], prev_row: &[u8], bpp: usize, output: &mut Vec<u8>) {
    // Paeth predictor is branchy; keep scalar for correctness and portability.
    // SIMD version exists but remains experimental until fully validated.
    fallback::filter_paeth(row, prev_row, bpp, output)
}
