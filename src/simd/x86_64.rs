//! x86_64 SIMD implementations using SSE2, SSSE3, SSE4.2, and AVX2.

use std::arch::x86_64::*;

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

        // For sum of absolute values where values are treated as signed i8:
        // We need |x| for signed interpretation. For bytes 0-127, |x| = x.
        // For bytes 128-255 (signed -128 to -1), |x| = 256 - x.
        //
        // Use XOR with 0x80 to convert to "signed magnitude" then SAD against 0x80
        // This computes the sum of absolute values of signed bytes.
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

/// Compute Adler-32 checksum using AVX2 instructions.
/// Processes 32 bytes at a time for improved throughput.
///
/// # Safety
/// Caller must ensure AVX2 is available on the current CPU.
#[target_feature(enable = "avx2")]
pub unsafe fn adler32_avx2(data: &[u8]) -> u32 {
    const MOD_ADLER: u32 = 65_521;
    // Block size chosen to avoid overflow: 32 bytes * 172 iterations = 5504 bytes
    const BLOCK_SIZE: usize = 5504 / 32 * 32;

    let mut s1: u32 = 1;
    let mut s2: u32 = 0;

    let mut remaining = data;

    while remaining.len() >= BLOCK_SIZE {
        let (block, rest) = remaining.split_at(BLOCK_SIZE);
        let (new_s1, new_s2) = adler32_block_avx2(block, s1, s2);
        s1 = new_s1 % MOD_ADLER;
        s2 = new_s2 % MOD_ADLER;
        remaining = rest;
    }

    // Process remaining complete 32-byte chunks
    if remaining.len() >= 32 {
        let chunk_count = remaining.len() / 32 * 32;
        let (block, rest) = remaining.split_at(chunk_count);
        let (new_s1, new_s2) = adler32_block_avx2(block, s1, s2);
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

/// Process a block of data for Adler-32 using AVX2.
#[target_feature(enable = "avx2")]
unsafe fn adler32_block_avx2(data: &[u8], mut s1: u32, mut s2: u32) -> (u32, u32) {
    // Weights for s2 accumulation within a 32-byte chunk
    let weights = _mm256_setr_epi8(
        32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
        16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
    );
    let zeros = _mm256_setzero_si256();

    let mut vs1 = _mm256_setzero_si256();
    let mut vs2 = _mm256_setzero_si256();
    let mut vs1_total = _mm256_setzero_si256();

    for chunk in data.chunks_exact(32) {
        let v = _mm256_loadu_si256(chunk.as_ptr() as *const __m256i);

        // Add this chunk's contribution to s2 based on previous s1 total
        vs2 = _mm256_add_epi32(vs2, _mm256_slli_epi32(vs1_total, 5)); // *32

        // Compute sum of bytes for s1 using SAD against zero
        let sad = _mm256_sad_epu8(v, zeros);
        vs1 = _mm256_add_epi32(vs1, sad);
        vs1_total = _mm256_add_epi32(vs1_total, sad);

        // Compute weighted sum for s2
        let v_lo = _mm256_unpacklo_epi8(v, zeros);
        let v_hi = _mm256_unpackhi_epi8(v, zeros);
        let w_lo = _mm256_unpacklo_epi8(weights, zeros);
        let w_hi = _mm256_unpackhi_epi8(weights, zeros);

        let prod_lo = _mm256_madd_epi16(v_lo, w_lo);
        let prod_hi = _mm256_madd_epi16(v_hi, w_hi);
        let weighted_sum = _mm256_add_epi32(prod_lo, prod_hi);
        vs2 = _mm256_add_epi32(vs2, weighted_sum);
    }

    // Horizontal sum of vs1 (4 x 64-bit values)
    let vs1_128 = _mm_add_epi64(
        _mm256_castsi256_si128(vs1),
        _mm256_extracti128_si256(vs1, 1),
    );
    let vs1_sum = _mm_add_epi64(vs1_128, _mm_shuffle_epi32(vs1_128, 0b00_00_11_10));
    s1 += _mm_cvtsi128_si64(vs1_sum) as u32;

    // Horizontal sum of vs2 (8 x 32-bit values)
    let vs2_128 = _mm_add_epi32(
        _mm256_castsi256_si128(vs2),
        _mm256_extracti128_si256(vs2, 1),
    );
    let vs2_2 = _mm_add_epi32(vs2_128, _mm_shuffle_epi32(vs2_128, 0b00_00_11_10));
    let vs2_3 = _mm_add_epi32(vs2_2, _mm_shuffle_epi32(vs2_2, 0b00_00_00_01));
    s2 += _mm_cvtsi128_si32(vs2_3) as u32;

    (s1, s2)
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

        if mask != 0xFFFFFFFF {
            // Found a mismatch - count trailing ones (matching bytes)
            return length + (!mask).trailing_zeros() as usize;
        }
        length += 32;
    }

    // Handle remaining bytes with SSE2 (16 bytes at a time)
    while length + 16 <= max_len {
        let a = _mm_loadu_si128(data[pos1 + length..].as_ptr() as *const __m128i);
        let b = _mm_loadu_si128(data[pos2 + length..].as_ptr() as *const __m128i);
        let cmp = _mm_cmpeq_epi8(a, b);
        let mask = _mm_movemask_epi8(cmp) as u32;

        if mask != 0xFFFF {
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

/// Apply Up filter using AVX2 (32 bytes at a time).
///
/// # Safety
/// Caller must ensure AVX2 is available on the current CPU.
#[target_feature(enable = "avx2")]
pub unsafe fn filter_up_avx2(row: &[u8], prev_row: &[u8], output: &mut Vec<u8>) {
    let len = row.len();
    let mut i = 0;

    // Process 32 bytes at a time
    while i + 32 <= len {
        let curr = _mm256_loadu_si256(row[i..].as_ptr() as *const __m256i);
        let prev = _mm256_loadu_si256(prev_row[i..].as_ptr() as *const __m256i);
        let diff = _mm256_sub_epi8(curr, prev);

        let mut buf = [0u8; 32];
        _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, diff);
        output.extend_from_slice(&buf);
        i += 32;
    }

    // Handle remaining bytes with SSE2
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
    // First bpp bytes have no left neighbor
    for &byte in &row[..bpp] {
        output.push(byte);
    }

    let remaining = &row[bpp..];
    let left = &row[..row.len() - bpp];

    let mut i = 0;
    let len = remaining.len();

    // Process 32 bytes at a time
    while i + 32 <= len {
        let curr = _mm256_loadu_si256(remaining[i..].as_ptr() as *const __m256i);
        let prev = _mm256_loadu_si256(left[i..].as_ptr() as *const __m256i);
        let diff = _mm256_sub_epi8(curr, prev);

        let mut buf = [0u8; 32];
        _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, diff);
        output.extend_from_slice(&buf);
        i += 32;
    }

    // Handle remaining with SSE2
    while i + 16 <= len {
        let curr = _mm_loadu_si128(remaining[i..].as_ptr() as *const __m128i);
        let prev = _mm_loadu_si128(left[i..].as_ptr() as *const __m128i);
        let diff = _mm_sub_epi8(curr, prev);

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
