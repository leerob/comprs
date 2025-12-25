//! DEFLATE (RFC 1951) + zlib (RFC 1950) decoder.
//!
//! This module provides a minimal, dependency-free inflate implementation
//! sufficient for PNG decoding:
//! - Supports zlib-wrapped DEFLATE streams (CM=8).
//! - Handles stored, fixed Huffman, and dynamic Huffman blocks.
//! - Rejects unsupported/invalid streams with descriptive errors.
//! - Verifies zlib FCHECK/FDICT and Adler32.
//!
//! The implementation favors clarity and small size over maximum throughput,
//! but is structured to avoid excessive allocations and to pre-allocate output
//! when an expected size is provided.

use crate::compress::adler32::adler32;
use crate::error::{Error, Result};

const MAX_CODE_BITS: u8 = 15;
const MAX_DISTANCE: usize = 32 * 1024;

// Length and distance tables from RFC 1951.
const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];

const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

// Order for code-length codes in dynamic Huffman blocks.
const CODE_LENGTH_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

/// Inflate a zlib-wrapped DEFLATE stream.
///
/// If `expected_size` is provided, the output buffer will be pre-allocated and
/// the final size validated against it.
pub fn inflate_zlib(data: &[u8], expected_size: Option<usize>) -> Result<Vec<u8>> {
    if data.len() < 6 {
        return Err(Error::InvalidDecode("zlib stream too short".into()));
    }

    // Parse zlib header
    let cmf = data[0];
    let flg = data[1];
    if (cmf & 0x0F) != 8 {
        return Err(Error::InvalidDecode(
            "unsupported compression method (expect DEFLATE)".into(),
        ));
    }
    let cinfo = cmf >> 4;
    if cinfo > 7 {
        return Err(Error::InvalidDecode("invalid CINFO/window size".into()));
    }
    if (u16::from(cmf) * 256 + u16::from(flg)) % 31 != 0 {
        return Err(Error::InvalidDecode("zlib FCHECK failed".into()));
    }
    if (flg & 0b0010_0000) != 0 {
        return Err(Error::UnsupportedDecode(
            "zlib FDICT preset dictionary not supported".into(),
        ));
    }

    // Split payload and Adler32
    if data.len() < 6 {
        return Err(Error::InvalidDecode("zlib missing Adler32".into()));
    }
    let adler_off = data.len() - 4;
    let payload = &data[2..adler_off];
    let expected_adler = u32::from_be_bytes(data[adler_off..].try_into().unwrap());

    let mut out = Vec::with_capacity(expected_size.unwrap_or(0));
    inflate_deflate(payload, &mut out)?;

    let actual_adler = adler32(&out);
    if actual_adler != expected_adler {
        return Err(Error::InvalidDecode(format!(
            "Adler32 mismatch: expected {expected_adler:#010x}, got {actual_adler:#010x}"
        )));
    }

    if let Some(exp) = expected_size {
        if out.len() != exp {
            return Err(Error::InvalidDecode(format!(
                "decompressed size mismatch: expected {exp}, got {}",
                out.len()
            )));
        }
    }

    Ok(out)
}

fn inflate_deflate(input: &[u8], out: &mut Vec<u8>) -> Result<()> {
    let mut br = BitReader::new(input);
    loop {
        let bfinal = br.read_bits(1)? != 0;
        let btype = br.read_bits(2)?;
        match btype {
            0 => inflate_stored(&mut br, out)?,
            1 => inflate_compressed(&mut br, out, FixedTables::fixed())?,
            2 => {
                let tables = FixedTables::dynamic(&mut br)?;
                inflate_compressed(&mut br, out, tables)?;
            }
            _ => return Err(Error::InvalidDecode("reserved BTYPE encountered".into())),
        }
        if bfinal {
            break;
        }
    }
    Ok(())
}

fn inflate_stored(br: &mut BitReader<'_>, out: &mut Vec<u8>) -> Result<()> {
    br.align_byte();
    let len = br.read_u16_le()?;
    let nlen = br.read_u16_le()?;
    if len != !nlen {
        return Err(Error::InvalidDecode(
            "stored block LEN/NLEN mismatch".into(),
        ));
    }
    let start = br.byte_pos;
    let end = start
        .checked_add(len as usize)
        .ok_or_else(|| Error::InvalidDecode("stored block length overflow".into()))?;
    if end > br.data.len() {
        return Err(Error::InvalidDecode("stored block overruns input".into()));
    }
    out.extend_from_slice(&br.data[start..end]);
    br.byte_pos = end;
    Ok(())
}

fn inflate_compressed(br: &mut BitReader<'_>, out: &mut Vec<u8>, tables: Tables) -> Result<()> {
    loop {
        let sym = tables.litlen.decode(br)?;
        match sym {
            0..=255 => out.push(sym as u8),
            256 => break,
            257..=285 => {
                let len_idx = (sym - 257) as usize;
                let base = LENGTH_BASE[len_idx];
                let extra = LENGTH_EXTRA[len_idx];
                let length = base as usize + br.read_bits(extra)? as usize;

                let dist_sym = tables.dist.decode(br)?;
                if dist_sym >= 30 {
                    return Err(Error::InvalidDecode("distance symbol out of range".into()));
                }
                let dist_base = DIST_BASE[dist_sym as usize] as usize;
                let dist_extra = DIST_EXTRA[dist_sym as usize];
                let distance = dist_base + br.read_bits(dist_extra)? as usize;

                if distance == 0 || distance > out.len() {
                    return Err(Error::InvalidDecode(format!(
                        "invalid back-reference distance: dist={} out_len={}",
                        distance,
                        out.len()
                    )));
                }
                if distance > MAX_DISTANCE {
                    return Err(Error::InvalidDecode(
                        "distance exceeds 32 KiB window".into(),
                    ));
                }

                // Copy with possible overlap (use growing output for repeats)
                let target_len = out.len() + length;
                while out.len() < target_len {
                    let b = out[out.len() - distance];
                    out.push(b);
                }
            }
            _ => return Err(Error::InvalidDecode("invalid literal/length symbol".into())),
        }
    }
    Ok(())
}

struct BitReader<'a> {
    data: &'a [u8],
    bit_buf: u64,
    bits_in_buf: u8,
    byte_pos: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            bit_buf: 0,
            bits_in_buf: 0,
            byte_pos: 0,
        }
    }

    fn ensure(&mut self, bits: u8) -> Result<()> {
        while self.bits_in_buf < bits {
            if self.byte_pos >= self.data.len() {
                return Err(Error::InvalidDecode("unexpected end of input".into()));
            }
            self.bit_buf |= (self.data[self.byte_pos] as u64) << self.bits_in_buf;
            self.bits_in_buf += 8;
            self.byte_pos += 1;
        }
        Ok(())
    }

    fn read_bits(&mut self, bits: u8) -> Result<u32> {
        if bits == 0 {
            return Ok(0);
        }
        self.ensure(bits)?;
        let mask = if bits == 32 {
            u32::MAX
        } else {
            (1u32 << bits) - 1
        };
        let val = (self.bit_buf as u32) & mask;
        self.bit_buf >>= bits;
        self.bits_in_buf -= bits;
        Ok(val)
    }

    fn align_byte(&mut self) {
        let drop_bits = self.bits_in_buf % 8;
        self.bit_buf >>= drop_bits;
        self.bits_in_buf -= drop_bits;
    }

    fn read_u16_le(&mut self) -> Result<u16> {
        self.align_byte();
        if self.byte_pos + 2 > self.data.len() {
            return Err(Error::InvalidDecode("unexpected end of input".into()));
        }
        let v = u16::from_le_bytes([self.data[self.byte_pos], self.data[self.byte_pos + 1]]);
        self.byte_pos += 2;
        Ok(v)
    }
}

struct Tables {
    litlen: Huffman,
    dist: Huffman,
}

struct FixedTables;

impl FixedTables {
    fn fixed() -> Tables {
        let mut lengths = [0u8; 288];
        for i in 0..=143 {
            lengths[i] = 8;
        }
        for i in 144..=255 {
            lengths[i] = 9;
        }
        for i in 256..=279 {
            lengths[i] = 7;
        }
        for i in 280..=287 {
            lengths[i] = 8;
        }
        let litlen = Huffman::from_lengths(&lengths).expect("fixed litlen table");

        let dist_lengths = [5u8; 30];
        let dist = Huffman::from_lengths(&dist_lengths).expect("fixed dist table");

        Tables {
            litlen,
            dist,
        }
    }

    fn dynamic(br: &mut BitReader<'_>) -> Result<Tables> {
        let hlit = br.read_bits(5)? + 257;
        let hdist = br.read_bits(5)? + 1;
        let hclen = br.read_bits(4)? + 4;

        let mut code_len_lengths = [0u8; 19];
        for i in 0..hclen {
            let idx = CODE_LENGTH_ORDER[i as usize];
            code_len_lengths[idx] = br.read_bits(3)? as u8;
        }
        let code_len_huff = Huffman::from_lengths(&code_len_lengths)?;

        let total = (hlit + hdist) as usize;
        let mut lengths = Vec::with_capacity(total);
        let mut i = 0;
        while i < total {
            let sym = code_len_huff.decode(br)?;
            match sym {
                0..=15 => {
                    lengths.push(sym as u8);
                    i += 1;
                }
                16 => {
                    let repeat = 3 + br.read_bits(2)? as usize;
                    if lengths.is_empty() {
                        return Err(Error::InvalidDecode(
                            "repeat with no previous length".into(),
                        ));
                    }
                    let last = *lengths.last().unwrap();
                    lengths.extend(std::iter::repeat(last).take(repeat));
                    i += repeat;
                }
                17 => {
                    let repeat = 3 + br.read_bits(3)? as usize;
                    lengths.extend(std::iter::repeat(0u8).take(repeat));
                    i += repeat;
                }
                18 => {
                    let repeat = 11 + br.read_bits(7)? as usize;
                    lengths.extend(std::iter::repeat(0u8).take(repeat));
                    i += repeat;
                }
                _ => return Err(Error::InvalidDecode("invalid code length symbol".into())),
            }
            if lengths.len() > total {
                return Err(Error::InvalidDecode("code lengths overrun".into()));
            }
        }

        if lengths.len() != total {
            return Err(Error::InvalidDecode(
                "code lengths did not fill required entries".into(),
            ));
        }

        let litlen_lengths = &lengths[..hlit as usize];
        let dist_lengths = &lengths[hlit as usize..];

        // dist code lengths cannot be all zero
        if dist_lengths.iter().all(|&l| l == 0) {
            return Err(Error::InvalidDecode(
                "distance tree has all zero lengths".into(),
            ));
        }

        let litlen = Huffman::from_lengths(litlen_lengths)?;
        let dist = Huffman::from_lengths(dist_lengths)?;

        Ok(Tables { litlen, dist })
    }
}

#[derive(Clone)]
struct Huffman {
    counts: [u16; (MAX_CODE_BITS as usize) + 1],
    first_code: [u16; (MAX_CODE_BITS as usize) + 1],
    first_symbol: [u16; (MAX_CODE_BITS as usize) + 1],
    symbols: Vec<u16>,
    max_bits: u8,
}

impl Huffman {
    fn from_lengths(lengths: &[u8]) -> Result<Self> {
        let mut counts = [0u16; (MAX_CODE_BITS as usize) + 1];
        for &len in lengths {
            if len as usize > MAX_CODE_BITS as usize {
                return Err(Error::InvalidDecode("code length exceeds 15".into()));
            }
            if len > 0 {
                counts[len as usize] += 1;
            }
        }
        let max_bits = (1..=MAX_CODE_BITS)
            .rev()
            .find(|&b| counts[b as usize] > 0)
            .unwrap_or(0);
        if max_bits == 0 {
            return Err(Error::InvalidDecode("Huffman table has no codes".into()));
        }

        // Determine first_code and first_symbol per length (canonical codes)
        let mut first_code = [0u16; (MAX_CODE_BITS as usize) + 1];
        let mut next_code = [0u16; (MAX_CODE_BITS as usize) + 1];
        let mut code = 0u16;
        for bits in 1..=MAX_CODE_BITS as usize {
            code = (code + counts[bits - 1]) << 1;
            first_code[bits] = code;
            next_code[bits] = code;
        }

        let mut first_symbol = [0u16; (MAX_CODE_BITS as usize) + 1];
        {
            let mut sum = 0u16;
            for bits in 1..=MAX_CODE_BITS as usize {
                first_symbol[bits] = sum;
                sum += counts[bits];
            }
        }

        // Symbols ordered by length then code
        let mut symbols = vec![0u16; lengths.len()];
        for (symbol, &len) in lengths.iter().enumerate() {
            let len_usize = len as usize;
            if len_usize == 0 {
                continue;
            }
            let idx = first_symbol[len_usize] as usize;
            symbols[idx + (next_code[len_usize] - first_code[len_usize]) as usize] = symbol as u16;
            next_code[len_usize] += 1;
        }

        Ok(Self {
            counts,
            first_code,
            first_symbol,
            symbols,
            max_bits,
        })
    }

    fn decode(&self, br: &mut BitReader<'_>) -> Result<u16> {
        let mut code: u16 = 0;
        for len in 1..=self.max_bits as usize {
            let bit = br.read_bits(1)? as u16;
            code = (code << 1) | bit;
            let count = self.counts[len];
            if count == 0 {
                continue;
            }
            let first = self.first_code[len];
            if code >= first && code < first + count {
                let idx = self.first_symbol[len] as usize + (code - first) as usize;
                return Ok(self.symbols[idx]);
            }
        }
        Err(Error::InvalidDecode("invalid Huffman code".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compress::deflate::deflate_zlib;
    use rand::{Rng, SeedableRng};

    #[test]
    fn roundtrip_small_literals() {
        let data = b"hello world, hello PNG";
        let compressed = deflate_zlib(data, 6);
        let decoded = inflate_zlib(&compressed, Some(data.len())).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn roundtrip_random_block() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut data = vec![0u8; 4096];
        rng.fill(data.as_mut_slice());
        let compressed = deflate_zlib(&data, 6);
        let decoded = inflate_zlib(&compressed, Some(data.len())).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn detects_bad_adler() {
        let data = b"hello";
        let mut compressed = deflate_zlib(data, 6);
        // Corrupt last byte of adler
        let len = compressed.len();
        compressed[len - 1] ^= 0xFF;
        let err = inflate_zlib(&compressed, Some(data.len())).unwrap_err();
        assert!(matches!(err, Error::InvalidDecode(_)));
    }
}
