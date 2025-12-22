//! Scalar fallback implementations for when SIMD is not available.
//!
//! These are the same implementations used in the non-SIMD code paths,
//! extracted here for use as fallbacks.

/// Compute Adler-32 checksum (scalar fallback).
#[inline]
pub fn adler32(data: &[u8]) -> u32 {
    const MOD_ADLER: u32 = 65_521;
    const NMAX: usize = 5552;

    let mut s1: u32 = 1;
    let mut s2: u32 = 0;

    for chunk in data.chunks(NMAX) {
        for &b in chunk {
            s1 += b as u32;
            s2 += s1;
        }
        s1 %= MOD_ADLER;
        s2 %= MOD_ADLER;
    }

    (s2 << 16) | s1
}

/// CRC32 lookup table.
const CRC_TABLE: [u32; 256] = {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
};

/// Compute CRC32 checksum (scalar fallback).
#[inline]
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFFFFFF_u32;
    for &byte in data {
        let index = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC_TABLE[index];
    }
    crc ^ 0xFFFFFFFF
}

/// Compute match length between two positions (scalar fallback with u64 optimization).
#[inline]
pub fn match_length(data: &[u8], pos1: usize, pos2: usize, max_len: usize) -> usize {
    let mut length = 0;

    // Compare 8 bytes at a time using u64
    while length + 8 <= max_len {
        let a = u64::from_ne_bytes(data[pos1 + length..pos1 + length + 8].try_into().unwrap());
        let b = u64::from_ne_bytes(data[pos2 + length..pos2 + length + 8].try_into().unwrap());
        if a != b {
            let xor = a ^ b;
            #[cfg(target_endian = "little")]
            {
                length += (xor.trailing_zeros() / 8) as usize;
            }
            #[cfg(target_endian = "big")]
            {
                length += (xor.leading_zeros() / 8) as usize;
            }
            return length;
        }
        length += 8;
    }

    // Handle remaining bytes
    while length < max_len && data[pos1 + length] == data[pos2 + length] {
        length += 1;
    }

    length
}

/// Score a filtered row using sum of absolute values (scalar fallback).
#[inline]
pub fn score_filter(filtered: &[u8]) -> u64 {
    filtered
        .iter()
        .map(|&b| (b as i8).unsigned_abs() as u64)
        .sum()
}

/// Apply Sub filter (scalar fallback).
#[inline]
pub fn filter_sub(row: &[u8], bpp: usize, output: &mut Vec<u8>) {
    for (i, &byte) in row.iter().enumerate() {
        let left = if i >= bpp { row[i - bpp] } else { 0 };
        output.push(byte.wrapping_sub(left));
    }
}

/// Apply Up filter (scalar fallback).
#[inline]
pub fn filter_up(row: &[u8], prev_row: &[u8], output: &mut Vec<u8>) {
    for (i, &byte) in row.iter().enumerate() {
        output.push(byte.wrapping_sub(prev_row[i]));
    }
}

/// Apply Average filter (scalar fallback).
#[inline]
pub fn filter_average(row: &[u8], prev_row: &[u8], bpp: usize, output: &mut Vec<u8>) {
    for (i, &byte) in row.iter().enumerate() {
        let left = if i >= bpp { row[i - bpp] as u16 } else { 0 };
        let above = prev_row[i] as u16;
        let avg = ((left + above) / 2) as u8;
        output.push(byte.wrapping_sub(avg));
    }
}

/// Apply Paeth filter (scalar fallback).
#[inline]
pub fn filter_paeth(row: &[u8], prev_row: &[u8], bpp: usize, output: &mut Vec<u8>) {
    for (i, &byte) in row.iter().enumerate() {
        let left = if i >= bpp { row[i - bpp] } else { 0 };
        let above = prev_row[i];
        let upper_left = if i >= bpp { prev_row[i - bpp] } else { 0 };
        let predicted = fallback_paeth_predictor(left, above, upper_left);
        output.push(byte.wrapping_sub(predicted));
    }
}

/// Scalar Paeth predictor used by both fallback and tests.
#[inline]
pub fn fallback_paeth_predictor(a: u8, b: u8, c: u8) -> u8 {
    let a = a as i16;
    let b = b as i16;
    let c = c as i16;

    let p = a + b - c;
    let pa = (p - a).abs();
    let pb = (p - b).abs();
    let pc = (p - c).abs();

    if pa <= pb && pa <= pc {
        a as u8
    } else if pb <= pc {
        b as u8
    } else {
        c as u8
    }
}
