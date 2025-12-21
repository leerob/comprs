//! DEFLATE compression algorithm (RFC 1951).
//!
//! Combines LZ77 compression with Huffman coding.

use crate::bits::BitWriter;
use crate::compress::{adler32::adler32, huffman};
use crate::compress::lz77::{Lz77Compressor, Token, MAX_MATCH_LENGTH, MIN_MATCH_LENGTH};

/// Length code base values (codes 257-285).
const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];

/// Extra bits for length codes.
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];

/// Distance code base values (codes 0-29).
const DISTANCE_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];

/// Extra bits for distance codes.
const DISTANCE_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

/// Get the length code (257-285) for a match length.
fn length_code(length: u16) -> (u16, u8, u16) {
    debug_assert!(
        (MIN_MATCH_LENGTH as u16..=MAX_MATCH_LENGTH as u16).contains(&length),
        "Invalid length: {}",
        length
    );

    for (i, &base) in LENGTH_BASE.iter().enumerate() {
        let next_base = if i + 1 < LENGTH_BASE.len() {
            LENGTH_BASE[i + 1]
        } else {
            259
        };
        if length >= base && length < next_base {
            let extra_bits = LENGTH_EXTRA[i];
            let extra_value = length - base;
            return (257 + i as u16, extra_bits, extra_value);
        }
    }

    // Length 258
    (285, 0, 0)
}

/// Get the distance code (0-29) for a match distance.
fn distance_code(distance: u16) -> (u16, u8, u16) {
    debug_assert!(distance >= 1 && distance <= 32768, "Invalid distance");

    for (i, &base) in DISTANCE_BASE.iter().enumerate() {
        let next_base = if i + 1 < DISTANCE_BASE.len() {
            DISTANCE_BASE[i + 1]
        } else {
            32769
        };
        if distance >= base && distance < next_base {
            let extra_bits = DISTANCE_EXTRA[i];
            let extra_value = distance - base;
            return (i as u16, extra_bits, extra_value);
        }
    }

    unreachable!()
}

/// Compress data using DEFLATE algorithm.
///
/// # Arguments
/// * `data` - Raw data to compress
/// * `level` - Compression level 1-9
///
/// # Returns
/// Compressed data in raw DEFLATE format (no zlib/gzip wrapper).
pub fn deflate(data: &[u8], level: u8) -> Vec<u8> {
    if data.is_empty() {
        // Empty input: just output empty final block
        let mut writer = BitWriter::new();
        writer.write_bits(1, 1); // BFINAL = 1
        writer.write_bits(1, 2); // BTYPE = 01 (fixed Huffman)

        // Write end-of-block symbol (256)
        let lit_codes = huffman::fixed_literal_codes();
        let code = lit_codes[256];
        writer.write_bits(reverse_bits(code.code, code.length), code.length);

        return writer.finish();
    }

    // Use LZ77 to find matches
    let mut lz77 = Lz77Compressor::new(level);
    let tokens = lz77.compress(data);

    // Choose between fixed and dynamic Huffman based on output size.
    let fixed = encode_fixed_huffman(&tokens);
    let dynamic = encode_dynamic_huffman(&tokens);

    if dynamic.len() < fixed.len() {
        dynamic
    } else {
        fixed
    }
}

/// Compress data and wrap it in a zlib container (RFC 1950).
///
/// Produces: zlib header (CMF/FLG), deflate stream, Adler-32 checksum.
pub fn deflate_zlib(data: &[u8], level: u8) -> Vec<u8> {
    // For empty input, keep the fixed-Huffman minimal block.
    if data.is_empty() {
        let mut output = Vec::with_capacity(8);
        output.extend_from_slice(&zlib_header(level));
        output.extend_from_slice(&deflate(data, level));
        output.extend_from_slice(&adler32(data).to_be_bytes());
        return output;
    }

    let deflated = deflate(data, level);

    let use_stored = should_use_stored(data.len(), deflated.len());

    let mut output = Vec::with_capacity(deflated.len().min(data.len()) + 32);
    output.extend_from_slice(&zlib_header(level));

    if use_stored {
        let stored_blocks = deflate_stored(data);
        output.extend_from_slice(&stored_blocks);
    } else {
        output.extend_from_slice(&deflated);
    }

    output.extend_from_slice(&adler32(data).to_be_bytes());
    output
}

/// Decide whether stored blocks would be smaller than the compressed stream.
fn should_use_stored(data_len: usize, deflated_len: usize) -> bool {
    // Stored block size: data + 5 bytes per 65535 chunk
    let stored_overhead = (data_len / 65_535 + 1) * 5;
    let stored_total = data_len + stored_overhead + 2 /*zlib hdr*/ + 4 /*adler*/;
    let deflated_total = deflated_len + 2 /*zlib hdr*/ + 4 /*adler*/;
    deflated_total >= stored_total
}

/// Encode tokens using fixed Huffman codes.
fn encode_fixed_huffman(tokens: &[Token]) -> Vec<u8> {
    let lit_codes = huffman::fixed_literal_codes();
    let dist_codes = huffman::fixed_distance_codes();

    let mut writer = BitWriter::new();

    // Block header: BFINAL=1 (last block), BTYPE=01 (fixed Huffman)
    writer.write_bits(1, 1); // BFINAL
    writer.write_bits(1, 2); // BTYPE (01 = fixed Huffman, LSB first)

    for token in tokens {
        match *token {
            Token::Literal(byte) => {
                let code = lit_codes[byte as usize];
                writer.write_bits(reverse_bits(code.code, code.length), code.length);
            }
            Token::Match { length, distance } => {
                // Encode length
                let (len_symbol, len_extra_bits, len_extra_value) = length_code(length);
                let len_code = lit_codes[len_symbol as usize];
                writer.write_bits(reverse_bits(len_code.code, len_code.length), len_code.length);

                if len_extra_bits > 0 {
                    writer.write_bits(len_extra_value as u32, len_extra_bits);
                }

                // Encode distance
                let (dist_symbol, dist_extra_bits, dist_extra_value) = distance_code(distance);
                let dist_code = dist_codes[dist_symbol as usize];
                writer.write_bits(
                    reverse_bits(dist_code.code, dist_code.length),
                    dist_code.length,
                );

                if dist_extra_bits > 0 {
                    writer.write_bits(dist_extra_value as u32, dist_extra_bits);
                }
            }
        }
    }

    // End of block symbol (256)
    let eob_code = lit_codes[256];
    writer.write_bits(reverse_bits(eob_code.code, eob_code.length), eob_code.length);

    writer.finish()
}

/// Encode tokens using dynamic Huffman codes (RFC 1951).
fn encode_dynamic_huffman(tokens: &[Token]) -> Vec<u8> {
    // Frequencies
    let mut lit_freqs = vec![0u32; 286]; // 0-285
    let mut dist_freqs = vec![0u32; 30]; // 0-29

    for token in tokens {
        match *token {
            Token::Literal(b) => lit_freqs[b as usize] += 1,
            Token::Match { length, distance } => {
                let (len_symbol, _, _) = length_code(length);
                lit_freqs[len_symbol as usize] += 1;

                let (dist_symbol, _, _) = distance_code(distance);
                dist_freqs[dist_symbol as usize] += 1;
            }
        }
    }
    // End-of-block
    lit_freqs[256] += 1;

    // Ensure at least one distance code per spec
    if dist_freqs.iter().all(|&f| f == 0) {
        dist_freqs[0] = 1;
    }

    let lit_codes = huffman::build_codes(&lit_freqs, huffman::MAX_CODE_LENGTH);
    let dist_codes = huffman::build_codes(&dist_freqs, huffman::MAX_CODE_LENGTH);

    // Code lengths
    let mut lit_lengths: Vec<u8> = lit_codes.iter().map(|c| c.length).collect();
    let mut dist_lengths: Vec<u8> = dist_codes.iter().map(|c| c.length).collect();

    // Trim trailing zeros for HLIT/HDIST
    let hlit = (last_nonzero(&lit_lengths).saturating_sub(257)).min(29);
    let hdist = (last_nonzero(&dist_lengths).saturating_sub(1)).min(29);

    lit_lengths.truncate(257 + hlit as usize);
    dist_lengths.truncate(1 + hdist as usize);

    // RLE encode code lengths
    let mut cl_freqs = vec![0u32; 19];
    let rle = rle_code_lengths(&lit_lengths, &dist_lengths, &mut cl_freqs);

    // Build code length codes (max len 7)
    let cl_codes = huffman::build_codes(&cl_freqs, 7);

    // Determine HCLEN (last non-zero in order)
    let cl_order: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];
    let mut hclen = 0;
    for (i, &idx) in cl_order.iter().enumerate().rev() {
        if cl_codes[idx].length > 0 {
            hclen = i as u8;
            break;
        }
    }

    let mut writer = BitWriter::new();
    writer.write_bits(1, 1); // BFINAL (single block)
    writer.write_bits(2, 2); // BTYPE=10 (dynamic)

    writer.write_bits(hlit as u32, 5); // HLIT
    writer.write_bits(hdist as u32, 5); // HDIST
    writer.write_bits(hclen as u32, 4); // HCLEN (number of code length codes - 4)

    // Write code length code lengths in order
    for &idx in cl_order.iter().take(hclen as usize + 4) {
        writer.write_bits(cl_codes[idx].length as u32, 3);
    }

    // Write the RLE-encoded code lengths
    for (sym, extra_bits, extra_len) in rle {
        let code = cl_codes[sym as usize];
        writer.write_bits(reverse_bits(code.code, code.length), code.length);
        if extra_len > 0 {
            writer.write_bits(extra_bits as u32, extra_len);
        }
    }

    // Data block using dynamic codes
    for token in tokens {
        match *token {
            Token::Literal(byte) => {
                let code = lit_codes[byte as usize];
                writer.write_bits(reverse_bits(code.code, code.length), code.length);
            }
            Token::Match { length, distance } => {
                let (len_symbol, len_extra_bits, len_extra_value) = length_code(length);
                let len_code = lit_codes[len_symbol as usize];
                writer.write_bits(reverse_bits(len_code.code, len_code.length), len_code.length);
                if len_extra_bits > 0 {
                    writer.write_bits(len_extra_value as u32, len_extra_bits);
                }

                let (dist_symbol, dist_extra_bits, dist_extra_value) = distance_code(distance);
                let dist_code = dist_codes[dist_symbol as usize];
                writer.write_bits(reverse_bits(dist_code.code, dist_code.length), dist_code.length);
                if dist_extra_bits > 0 {
                    writer.write_bits(dist_extra_value as u32, dist_extra_bits);
                }
            }
        }
    }

    // End of block
    let eob_code = lit_codes[256];
    writer.write_bits(reverse_bits(eob_code.code, eob_code.length), eob_code.length);

    writer.finish()
}

fn last_nonzero(lengths: &[u8]) -> usize {
    lengths
        .iter()
        .rposition(|&l| l != 0)
        .map(|i| i + 1)
        .unwrap_or(1) // minimum 1 code
}

/// RLE encode literal/dist code lengths and collect code length code frequencies.
fn rle_code_lengths(
    lit_lengths: &[u8],
    dist_lengths: &[u8],
    cl_freqs: &mut [u32],
) -> Vec<(u8, u8, u8)> {
    let mut seq = Vec::new();
    seq.extend_from_slice(lit_lengths);
    seq.extend_from_slice(dist_lengths);

    let mut encoded = Vec::new();
    let mut i = 0;
    while i < seq.len() {
        let curr = seq[i];
        let mut run = 1;
        while i + run < seq.len() && seq[i + run] == curr {
            run += 1;
        }

        if curr == 0 {
            let mut rem = run;
            while rem > 0 {
                if rem >= 11 {
                    let take = rem.min(138);
                    encoded.push((18, (take - 11) as u8, 7));
                    cl_freqs[18] += 1;
                    rem -= take;
                } else if rem >= 3 {
                    let take = rem.min(10);
                    encoded.push((17, (take - 3) as u8, 3));
                    cl_freqs[17] += 1;
                    rem -= take;
                } else {
                    encoded.push((0, 0, 0));
                    cl_freqs[0] += 1;
                    rem -= 1;
                }
            }
        } else {
            // emit first occurrence
            encoded.push((curr, 0, 0));
            cl_freqs[curr as usize] += 1;
            let mut rem = run - 1;
            while rem >= 3 {
                let take = rem.min(6);
                encoded.push((16, (take - 3) as u8, 2));
                cl_freqs[16] += 1;
                rem -= take;
            }
            while rem > 0 {
                encoded.push((curr, 0, 0));
                cl_freqs[curr as usize] += 1;
                rem -= 1;
            }
        }

        i += run;
    }

    encoded
}

/// Reverse bits in a code (DEFLATE uses reversed bit order for Huffman codes).
#[inline]
fn reverse_bits(code: u16, length: u8) -> u32 {
    let mut result = 0u32;
    let mut code = code as u32;
    for _ in 0..length {
        result = (result << 1) | (code & 1);
        code >>= 1;
    }
    result
}

/// Build the two-byte zlib header for the given compression level.
fn zlib_header(level: u8) -> [u8; 2] {
    // CMF: 0b0111_1000 (Deflate, 32K window)
    let cmf: u8 = 0x78;

    // Map level to FLEVEL (informative only)
    let flevel = match level {
        0 | 1 | 2 => 1,        // fast
        3..=6 => 2,            // default
        _ => 3,                // maximum
    };

    let mut flg: u8 = flevel << 6; // FDICT=0
    let fcheck = (31 - (((cmf as u16) << 8 | flg as u16) % 31)) % 31;
    flg |= fcheck as u8;

    [cmf, flg]
}

/// Compress data using DEFLATE with stored blocks (no compression).
/// Useful for already-compressed data or when speed is critical.
#[allow(dead_code)]
pub fn deflate_stored(data: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(data.len() + data.len() / 65535 * 5 + 10);
    let chunks = data.chunks(65535);
    let num_chunks = chunks.len();

    for (i, chunk) in data.chunks(65535).enumerate() {
        let is_final = i == num_chunks - 1;
        let len = chunk.len() as u16;
        let nlen = !len;

        // Block header
        output.push(if is_final { 0x01 } else { 0x00 }); // BFINAL + BTYPE=00

        // LEN and NLEN (little-endian)
        output.push(len as u8);
        output.push((len >> 8) as u8);
        output.push(nlen as u8);
        output.push((nlen >> 8) as u8);

        // Data
        output.extend_from_slice(chunk);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::read::ZlibDecoder;
    use rand::{Rng, SeedableRng};
    use std::io::Read;

    #[test]
    fn test_length_code() {
        assert_eq!(length_code(3), (257, 0, 0));
        assert_eq!(length_code(4), (258, 0, 0));
        assert_eq!(length_code(10), (264, 0, 0));
        assert_eq!(length_code(11), (265, 1, 0));
        assert_eq!(length_code(12), (265, 1, 1));
        assert_eq!(length_code(258), (285, 0, 0));
    }

    #[test]
    fn test_distance_code() {
        assert_eq!(distance_code(1), (0, 0, 0));
        assert_eq!(distance_code(2), (1, 0, 0));
        assert_eq!(distance_code(5), (4, 1, 0));
        assert_eq!(distance_code(6), (4, 1, 1));
    }

    #[test]
    fn test_deflate_empty() {
        let compressed = deflate(&[], 6);
        assert!(!compressed.is_empty());
    }

    #[test]
    fn test_deflate_simple() {
        let data = b"Hello, World!";
        let compressed = deflate(data, 6);

        // Should produce some output
        assert!(!compressed.is_empty());
        // For short data, compression might not reduce size much
    }

    #[test]
    fn test_deflate_repetitive() {
        let data = b"abcabcabcabcabcabcabcabcabcabc";
        let compressed = deflate(data, 6);

        // Repetitive data should compress well
        assert!(compressed.len() < data.len());
    }

    #[test]
    fn test_deflate_zlib_header_checksum() {
        let data = b"hello";
        let compressed = deflate_zlib(data, 6);

        // Header should be 0x78 0x9C for default-ish compression
        assert_eq!(&compressed[0..2], &[0x78, 0x9C]);

        let checksum = u32::from_be_bytes(compressed[compressed.len() - 4..].try_into().unwrap());
        assert_eq!(checksum, 0x062C0215);
    }

    #[test]
    fn test_deflate_stored() {
        let data = b"Hello, World!";
        let compressed = deflate_stored(data);

        // Stored blocks have 5 bytes overhead per 65535 bytes
        assert_eq!(compressed.len(), data.len() + 5);
    }

    #[test]
    fn test_reverse_bits() {
        assert_eq!(reverse_bits(0b101, 3), 0b101);
        assert_eq!(reverse_bits(0b100, 3), 0b001);
        assert_eq!(reverse_bits(0b11110000, 8), 0b00001111);
    }

    #[test]
    fn test_should_use_stored_threshold() {
        // Deflated larger than stored -> use stored
        assert!(should_use_stored(1000, 1200));
        // Deflated smaller -> keep deflated
        assert!(!should_use_stored(1000, 400));
        // Near-equal totals prefer stored
        assert!(should_use_stored(1000, 1010));
    }

    fn decompress_zlib(data: &[u8]) -> Vec<u8> {
        let mut decoder = ZlibDecoder::new(data);
        let mut out = Vec::new();
        decoder.read_to_end(&mut out).expect("zlib decode");
        out
    }

    #[test]
    fn test_deflate_zlib_empty_decode() {
        let encoded = deflate_zlib(&[], 6);
        let decoded = decompress_zlib(&encoded);
        assert!(decoded.is_empty());
        // Header for default compression and empty body should be minimal
        assert!(encoded.len() <= 11); // 2 header + 5 stored + 4 adler
    }

    #[test]
    fn test_deflate_zlib_roundtrip_random_small() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(999);
        for len in [0usize, 1, 2, 5, 32, 128, 1024, 4096] {
            let mut data = vec![0u8; len];
            rng.fill(data.as_mut_slice());
            let encoded = deflate_zlib(&data, 6);
            let decoded = decompress_zlib(&encoded);
            assert_eq!(decoded, data, "mismatch at len={}", len);
        }
    }

    #[test]
    fn test_deflate_zlib_incompressible_prefers_stored() {
        let mut data = vec![0u8; 10_000];
        // High-entropy pattern to discourage compression
        let mut rng = rand::rngs::StdRng::seed_from_u64(1234);
        rng.fill(data.as_mut_slice());

        let encoded = deflate_zlib(&data, 6);
        let decoded = decompress_zlib(&encoded);
        assert_eq!(decoded, data);
        // stored overhead per 65535 block is 5 bytes + header/adler
        let stored_overhead = (data.len() / 65_535 + 1) * 5 + 2 + 4;
        assert!(encoded.len() <= data.len() + stored_overhead);
    }
}
