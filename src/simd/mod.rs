//! SIMD acceleration module for performance-critical operations.
//!
//! This module provides SIMD-accelerated implementations of:
//! - Adler-32 checksum
//! - CRC32 checksum (using PCLMULQDQ for hardware acceleration)
//! - PNG filter operations
//! - LZ77 match length comparison
//!
//! The implementations are architecture-specific and fall back to scalar
//! implementations when SIMD is not available. Feature detection is cached
//! at startup to eliminate runtime overhead.

#[cfg(target_arch = "x86_64")]
use std::sync::LazyLock;

#[cfg(target_arch = "x86_64")]
pub mod x86_64;

#[cfg(target_arch = "aarch64")]
pub mod aarch64;

pub mod fallback;

// ============================================================================
// Cached SIMD Feature Detection
// ============================================================================

/// SIMD capability level for x86_64, detected once at startup.
#[cfg(target_arch = "x86_64")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum X86SimdLevel {
    /// AVX2 available (best performance).
    Avx2,
    /// SSSE3 available.
    Ssse3,
    /// SSE2 available (baseline for x86_64).
    Sse2,
    /// Fallback to scalar code.
    Scalar,
}

/// Detect the best available SIMD level on x86_64.
#[cfg(target_arch = "x86_64")]
fn detect_x86_simd_level() -> X86SimdLevel {
    if is_x86_feature_detected!("avx2") {
        X86SimdLevel::Avx2
    } else if is_x86_feature_detected!("ssse3") {
        X86SimdLevel::Ssse3
    } else if is_x86_feature_detected!("sse2") {
        X86SimdLevel::Sse2
    } else {
        X86SimdLevel::Scalar
    }
}

/// Cached SIMD level for x86_64, detected once at program startup.
#[cfg(target_arch = "x86_64")]
static X86_SIMD_LEVEL: LazyLock<X86SimdLevel> = LazyLock::new(detect_x86_simd_level);

/// Check if PCLMULQDQ is available (for CRC32 acceleration).
#[cfg(target_arch = "x86_64")]
static HAS_PCLMULQDQ: LazyLock<bool> = LazyLock::new(|| is_x86_feature_detected!("pclmulqdq"));

// ============================================================================
// Public API Functions
// ============================================================================

/// Compute Adler-32 checksum using the best available implementation.
#[inline]
pub fn adler32(data: &[u8]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        match *X86_SIMD_LEVEL {
            X86SimdLevel::Avx2 => unsafe { x86_64::adler32_avx2(data) },
            X86SimdLevel::Ssse3 | X86SimdLevel::Sse2 => unsafe { x86_64::adler32_ssse3(data) },
            X86SimdLevel::Scalar => fallback::adler32(data),
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on aarch64
        unsafe { aarch64::adler32_neon(data) }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fallback::adler32(data)
}

/// Compute CRC32 checksum using the best available implementation.
///
/// Uses PCLMULQDQ-accelerated CRC32 on x86_64 when available, which provides
/// significant speedup over the table-based implementation while using the
/// correct ISO-HDLC polynomial (0x04C11DB7) required by PNG.
#[inline]
pub fn crc32(data: &[u8]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if *HAS_PCLMULQDQ {
            return unsafe { x86_64::crc32_pclmulqdq(data) };
        }
        fallback::crc32(data)
    }

    #[cfg(not(target_arch = "x86_64"))]
    fallback::crc32(data)
}

/// Compute match length between two positions using the best available implementation.
#[inline]
pub fn match_length(data: &[u8], pos1: usize, pos2: usize, max_len: usize) -> usize {
    #[cfg(target_arch = "x86_64")]
    {
        match *X86_SIMD_LEVEL {
            X86SimdLevel::Avx2 => unsafe { x86_64::match_length_avx2(data, pos1, pos2, max_len) },
            X86SimdLevel::Ssse3 | X86SimdLevel::Sse2 => unsafe {
                x86_64::match_length_sse2(data, pos1, pos2, max_len)
            },
            X86SimdLevel::Scalar => fallback::match_length(data, pos1, pos2, max_len),
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { aarch64::match_length_neon(data, pos1, pos2, max_len) }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fallback::match_length(data, pos1, pos2, max_len)
}

/// Score a filtered row (sum of absolute values) using the best available implementation.
#[inline]
pub fn score_filter(filtered: &[u8]) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        match *X86_SIMD_LEVEL {
            X86SimdLevel::Avx2 => unsafe { x86_64::score_filter_avx2(filtered) },
            X86SimdLevel::Ssse3 | X86SimdLevel::Sse2 => unsafe {
                x86_64::score_filter_sse2(filtered)
            },
            X86SimdLevel::Scalar => fallback::score_filter(filtered),
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { aarch64::score_filter_neon(filtered) }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fallback::score_filter(filtered)
}

/// Apply Sub filter using the best available implementation.
#[inline]
pub fn filter_sub(row: &[u8], bpp: usize, output: &mut Vec<u8>) {
    #[cfg(target_arch = "x86_64")]
    {
        match *X86_SIMD_LEVEL {
            X86SimdLevel::Avx2 => unsafe { x86_64::filter_sub_avx2(row, bpp, output) },
            X86SimdLevel::Ssse3 | X86SimdLevel::Sse2 => unsafe {
                x86_64::filter_sub_sse2(row, bpp, output)
            },
            X86SimdLevel::Scalar => fallback::filter_sub(row, bpp, output),
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { aarch64::filter_sub_neon(row, bpp, output) };
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fallback::filter_sub(row, bpp, output)
}

/// Apply Up filter using the best available implementation.
#[inline]
pub fn filter_up(row: &[u8], prev_row: &[u8], output: &mut Vec<u8>) {
    #[cfg(target_arch = "x86_64")]
    {
        match *X86_SIMD_LEVEL {
            X86SimdLevel::Avx2 => unsafe { x86_64::filter_up_avx2(row, prev_row, output) },
            X86SimdLevel::Ssse3 | X86SimdLevel::Sse2 => unsafe {
                x86_64::filter_up_sse2(row, prev_row, output)
            },
            X86SimdLevel::Scalar => fallback::filter_up(row, prev_row, output),
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { aarch64::filter_up_neon(row, prev_row, output) };
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fallback::filter_up(row, prev_row, output)
}

/// Apply Average filter using the best available implementation.
#[inline]
pub fn filter_average(row: &[u8], prev_row: &[u8], bpp: usize, output: &mut Vec<u8>) {
    #[cfg(target_arch = "x86_64")]
    {
        match *X86_SIMD_LEVEL {
            X86SimdLevel::Avx2 => unsafe {
                x86_64::filter_average_avx2(row, prev_row, bpp, output)
            },
            _ => fallback::filter_average(row, prev_row, bpp, output),
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { aarch64::filter_average_neon(row, prev_row, bpp, output) };
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fallback::filter_average(row, prev_row, bpp, output)
}

/// Apply Paeth filter using the best available implementation.
#[inline]
pub fn filter_paeth(row: &[u8], prev_row: &[u8], bpp: usize, output: &mut Vec<u8>) {
    #[cfg(target_arch = "aarch64")]
    {
        unsafe { aarch64::filter_paeth_neon(row, prev_row, bpp, output) };
    }

    // Paeth predictor is branchy; keep scalar for correctness and portability on x86_64.
    #[cfg(not(target_arch = "aarch64"))]
    fallback::filter_paeth(row, prev_row, bpp, output)
}
