//! JPEG Huffman encoding.
//!
//! Implements the entropy coding stage of JPEG compression.

use crate::bits::BitWriterMsb;
use crate::jpeg::quantize::zigzag_reorder;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Standard DC luminance Huffman table (number of codes per bit length).
const DC_LUM_BITS: [u8; 16] = [0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0];

/// Standard DC luminance Huffman values.
const DC_LUM_VALS: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

/// Standard DC chrominance Huffman table.
const DC_CHROM_BITS: [u8; 16] = [0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0];

/// Standard DC chrominance Huffman values.
const DC_CHROM_VALS: [u8; 12] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];

/// Standard AC luminance Huffman table.
const AC_LUM_BITS: [u8; 16] = [0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 125];

/// Standard AC luminance Huffman values.
const AC_LUM_VALS: [u8; 162] = [
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
    0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
    0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
    0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
    0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
    0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
    0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
    0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
    0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
    0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
];

/// Standard AC chrominance Huffman table.
const AC_CHROM_BITS: [u8; 16] = [0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 119];

/// Standard AC chrominance Huffman values.
const AC_CHROM_VALS: [u8; 162] = [
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
    0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
    0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
    0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
    0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
    0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
    0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
    0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
    0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
    0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
    0xf9, 0xfa,
];

/// Huffman code: (code, length in bits).
#[derive(Debug, Clone, Copy, Default)]
struct HuffCode {
    code: u16,
    length: u8,
}

/// Precomputed Huffman tables for fast encoding.
pub struct HuffmanTables {
    /// DC luminance bits (for header output).
    pub dc_lum_bits: [u8; 16],
    /// DC luminance values (for header output).
    pub dc_lum_vals: Vec<u8>,
    /// DC chrominance bits.
    pub dc_chrom_bits: [u8; 16],
    /// DC chrominance values.
    pub dc_chrom_vals: Vec<u8>,
    /// AC luminance bits.
    pub ac_lum_bits: [u8; 16],
    /// AC luminance values.
    pub ac_lum_vals: Vec<u8>,
    /// AC chrominance bits.
    pub ac_chrom_bits: [u8; 16],
    /// AC chrominance values.
    pub ac_chrom_vals: Vec<u8>,

    // Lookup tables for encoding
    dc_lum_codes: [HuffCode; 12],
    dc_chrom_codes: [HuffCode; 12],
    ac_lum_codes: [HuffCode; 256],
    ac_chrom_codes: [HuffCode; 256],
}

impl HuffmanTables {
    /// Create Huffman tables from standard JPEG tables.
    pub fn new() -> Self {
        let dc_lum_codes = build_codes(&DC_LUM_BITS, &DC_LUM_VALS);
        let dc_chrom_codes = build_codes(&DC_CHROM_BITS, &DC_CHROM_VALS);
        let ac_lum_codes = build_codes_256(&AC_LUM_BITS, &AC_LUM_VALS);
        let ac_chrom_codes = build_codes_256(&AC_CHROM_BITS, &AC_CHROM_VALS);

        Self {
            dc_lum_bits: DC_LUM_BITS,
            dc_lum_vals: DC_LUM_VALS.to_vec(),
            dc_chrom_bits: DC_CHROM_BITS,
            dc_chrom_vals: DC_CHROM_VALS.to_vec(),
            ac_lum_bits: AC_LUM_BITS,
            ac_lum_vals: AC_LUM_VALS.to_vec(),
            ac_chrom_bits: AC_CHROM_BITS,
            ac_chrom_vals: AC_CHROM_VALS.to_vec(),
            dc_lum_codes,
            dc_chrom_codes,
            ac_lum_codes,
            ac_chrom_codes,
        }
    }

    fn get_dc_code(&self, category: u8, is_luminance: bool) -> HuffCode {
        if is_luminance {
            self.dc_lum_codes[category as usize]
        } else {
            self.dc_chrom_codes[category as usize]
        }
    }

    fn get_ac_code(&self, rs: u8, is_luminance: bool) -> HuffCode {
        if is_luminance {
            self.ac_lum_codes[rs as usize]
        } else {
            self.ac_chrom_codes[rs as usize]
        }
    }

    /// Create Huffman tables from per-symbol code lengths (bits/vals).
    #[allow(clippy::too_many_arguments)]
    fn from_bits_vals(
        dc_lum_bits: [u8; 16],
        dc_lum_vals: Vec<u8>,
        dc_chrom_bits: [u8; 16],
        dc_chrom_vals: Vec<u8>,
        ac_lum_bits: [u8; 16],
        ac_lum_vals: Vec<u8>,
        ac_chrom_bits: [u8; 16],
        ac_chrom_vals: Vec<u8>,
    ) -> Option<Self> {
        let dc_lum_codes = build_code_table::<12>(&dc_lum_bits, &dc_lum_vals, 12)?;
        let dc_chrom_codes = build_code_table::<12>(&dc_chrom_bits, &dc_chrom_vals, 12)?;
        let ac_lum_codes = build_code_table::<256>(&ac_lum_bits, &ac_lum_vals, 256)?;
        let ac_chrom_codes = build_code_table::<256>(&ac_chrom_bits, &ac_chrom_vals, 256)?;

        Some(Self {
            dc_lum_bits,
            dc_lum_vals,
            dc_chrom_bits,
            dc_chrom_vals,
            ac_lum_bits,
            ac_lum_vals,
            ac_chrom_bits,
            ac_chrom_vals,
            dc_lum_codes,
            dc_chrom_codes,
            ac_lum_codes,
            ac_chrom_codes,
        })
    }

    /// Build optimized Huffman tables from symbol frequency counts.
    /// Falls back to defaults if lengths exceed the JPEG limit (16 bits) or counts are empty.
    pub fn optimized_from_counts(
        dc_lum_counts: &[u64; 12],
        dc_chrom_counts: Option<&[u64; 12]>,
        ac_lum_counts: &[u64; 256],
        ac_chrom_counts: Option<&[u64; 256]>,
    ) -> Option<Self> {
        let (dc_lum_bits, dc_lum_vals) = build_bits_vals(dc_lum_counts)?;
        let (ac_lum_bits, ac_lum_vals) = build_bits_vals(ac_lum_counts)?;

        let (dc_chrom_bits, dc_chrom_vals) = if let Some(counts) = dc_chrom_counts {
            build_bits_vals(counts).unwrap_or((DC_CHROM_BITS, DC_CHROM_VALS.to_vec()))
        } else {
            (DC_CHROM_BITS, DC_CHROM_VALS.to_vec())
        };
        let (ac_chrom_bits, ac_chrom_vals) = if let Some(counts) = ac_chrom_counts {
            build_bits_vals(counts).unwrap_or((AC_CHROM_BITS, AC_CHROM_VALS.to_vec()))
        } else {
            (AC_CHROM_BITS, AC_CHROM_VALS.to_vec())
        };

        HuffmanTables::from_bits_vals(
            dc_lum_bits,
            dc_lum_vals,
            dc_chrom_bits,
            dc_chrom_vals,
            ac_lum_bits,
            ac_lum_vals,
            ac_chrom_bits,
            ac_chrom_vals,
        )
    }
}

impl Default for HuffmanTables {
    fn default() -> Self {
        Self::new()
    }
}

/// Build Huffman codes from bits/vals specification (for 12 DC symbols).
fn build_codes(bits: &[u8; 16], vals: &[u8]) -> [HuffCode; 12] {
    let mut codes = [HuffCode::default(); 12];
    let mut code = 0u16;
    let mut val_idx = 0;

    for (length, &count) in bits.iter().enumerate() {
        for _ in 0..count {
            if val_idx < vals.len() {
                let symbol = vals[val_idx] as usize;
                if symbol < 12 {
                    codes[symbol] = HuffCode {
                        code,
                        length: (length + 1) as u8,
                    };
                }
                val_idx += 1;
            }
            code += 1;
        }
        code <<= 1;
    }

    codes
}

fn build_codes_256(bits: &[u8; 16], vals: &[u8]) -> [HuffCode; 256] {
    let mut codes = [HuffCode::default(); 256];
    let mut code = 0u16;
    let mut val_idx = 0;

    for (length, &count) in bits.iter().enumerate() {
        for _ in 0..count {
            if val_idx < vals.len() {
                let symbol = vals[val_idx] as usize;
                codes[symbol] = HuffCode {
                    code,
                    length: (length + 1) as u8,
                };
                val_idx += 1;
            }
            code += 1;
        }
        code <<= 1;
    }

    codes
}

/// Build canonical code table for arbitrary symbol count using JPEG bits/vals format.
fn build_code_table<const N: usize>(
    bits: &[u8; 16],
    vals: &[u8],
    table_len: usize,
) -> Option<[HuffCode; N]> {
    let mut codes = [HuffCode::default(); N];
    let mut code: u16 = 0;
    let mut val_idx = 0usize;
    for (length, &count) in bits.iter().enumerate() {
        for _ in 0..count {
            if val_idx >= vals.len() {
                return None;
            }
            let symbol = vals[val_idx] as usize;
            if symbol >= table_len {
                return None;
            }
            codes[symbol] = HuffCode {
                code,
                length: (length + 1) as u8,
            };
            val_idx += 1;
            code += 1;
        }
        code <<= 1;
    }
    Some(codes)
}

/// Build bits/vals arrays from symbol frequencies. Returns None if all zero or depths exceed 16.
fn build_bits_vals(counts: &[u64]) -> Option<([u8; 16], Vec<u8>)> {
    let lengths = build_code_lengths(counts)?;
    let mut bits = [0u8; 16];
    for &len in &lengths {
        if len == 0 {
            continue;
        }
        if len as usize > bits.len() {
            return None;
        }
        bits[(len - 1) as usize] += 1;
    }
    // vals ordered by increasing length then symbol id
    let mut vals: Vec<u8> = (0..lengths.len())
        .filter(|&i| lengths[i] > 0)
        .collect::<Vec<_>>()
        .into_iter()
        .map(|i| i as u8)
        .collect();
    vals.sort_by_key(|&sym| (lengths[sym as usize], sym));
    Some((bits, vals))
}

/// Build canonical code lengths via Huffman tree (no length limiting beyond 16; returns None on overflow).
fn build_code_lengths(counts: &[u64]) -> Option<Vec<u8>> {
    #[derive(Clone)]
    struct Node {
        _freq: u64,
        left: Option<usize>,
        right: Option<usize>,
        symbol: Option<usize>,
    }

    let mut heap: BinaryHeap<Reverse<(u64, usize)>> = BinaryHeap::new();
    let mut nodes: Vec<Node> = Vec::new();

    for (sym, &freq) in counts.iter().enumerate() {
        if freq == 0 {
            continue;
        }
        let idx = nodes.len();
        nodes.push(Node {
            _freq: freq,
            left: None,
            right: None,
            symbol: Some(sym),
        });
        heap.push(Reverse((freq, idx)));
    }

    if heap.is_empty() {
        return None;
    }

    // Special case: only one symbol
    if heap.len() == 1 {
        let mut lengths = vec![0u8; counts.len()];
        let sym = nodes[heap.peek()?.0 .1].symbol?;
        lengths[sym] = 1;
        return Some(lengths);
    }

    // Build tree
    while heap.len() > 1 {
        let Reverse((f1, i1)) = heap.pop().unwrap();
        let Reverse((f2, i2)) = heap.pop().unwrap();
        let idx = nodes.len();
        nodes.push(Node {
            _freq: f1 + f2,
            left: Some(i1),
            right: Some(i2),
            symbol: None,
        });
        heap.push(Reverse((f1 + f2, idx)));
    }

    let root = heap.pop().unwrap().0 .1;
    let mut lengths = vec![0u8; counts.len()];
    let mut stack = vec![(root, 0u8)];
    while let Some((idx, depth)) = stack.pop() {
        let node = &nodes[idx];
        if let Some(sym) = node.symbol {
            let len = depth + 1;
            if len > 16 {
                return None;
            }
            lengths[sym] = len;
        } else {
            if let Some(l) = node.left {
                stack.push((l, depth + 1));
            }
            if let Some(r) = node.right {
                stack.push((r, depth + 1));
            }
        }
    }
    Some(lengths)
}

/// Get the category (number of bits needed) for a value.
fn category(value: i16) -> u8 {
    let abs_val = value.unsigned_abs();
    if abs_val == 0 {
        0
    } else {
        16 - abs_val.leading_zeros() as u8
    }
}

/// Encode a coefficient value (after category is known).
fn encode_value(value: i16) -> (u16, u8) {
    let cat = category(value);
    if cat == 0 {
        return (0, 0);
    }

    let bits = if value < 0 {
        // Negative values: use one's complement
        (value - 1) as u16
    } else {
        value as u16
    };

    (bits & ((1 << cat) - 1), cat)
}

/// Encode a quantized 8x8 block.
///
/// Returns the new DC value (for differential encoding of next block).
pub fn encode_block(
    writer: &mut BitWriterMsb,
    block: &[i16; 64],
    prev_dc: i16,
    is_luminance: bool,
    tables: &HuffmanTables,
) -> i16 {
    // Reorder to zigzag
    let zigzag = zigzag_reorder(block);

    // Encode DC coefficient (differential)
    let dc = zigzag[0];
    let dc_diff = dc - prev_dc;
    let dc_cat = category(dc_diff);

    let dc_code = tables.get_dc_code(dc_cat, is_luminance);
    writer.write_bits(dc_code.code as u32, dc_code.length);

    if dc_cat > 0 {
        let (val_bits, val_len) = encode_value(dc_diff);
        writer.write_bits(val_bits as u32, val_len);
    }

    // Encode AC coefficients
    let mut zero_run = 0;

    for &ac in zigzag.iter().skip(1) {
        if ac == 0 {
            zero_run += 1;
        } else {
            // Output ZRL (16 zeros) codes if needed
            while zero_run >= 16 {
                let zrl_code = tables.get_ac_code(0xF0, is_luminance);
                writer.write_bits(zrl_code.code as u32, zrl_code.length);
                zero_run -= 16;
            }

            // Output (run, size) code
            let ac_cat = category(ac);
            let rs = ((zero_run as u8) << 4) | ac_cat;

            let ac_code = tables.get_ac_code(rs, is_luminance);
            writer.write_bits(ac_code.code as u32, ac_code.length);

            let (val_bits, val_len) = encode_value(ac);
            writer.write_bits(val_bits as u32, val_len);

            zero_run = 0;
        }
    }

    // Output EOB if there are trailing zeros
    if zero_run > 0 {
        let eob_code = tables.get_ac_code(0x00, is_luminance);
        writer.write_bits(eob_code.code as u32, eob_code.length);
    }

    dc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_category() {
        assert_eq!(category(0), 0);
        assert_eq!(category(1), 1);
        assert_eq!(category(-1), 1);
        assert_eq!(category(2), 2);
        assert_eq!(category(3), 2);
        assert_eq!(category(-3), 2);
        assert_eq!(category(127), 7);
        assert_eq!(category(-127), 7);
        assert_eq!(category(255), 8);
    }

    #[test]
    fn test_encode_value() {
        assert_eq!(encode_value(0), (0, 0));
        assert_eq!(encode_value(1), (1, 1));
        assert_eq!(encode_value(-1), (0, 1));
        assert_eq!(encode_value(3), (3, 2));
        assert_eq!(encode_value(-3), (0, 2));
    }

    #[test]
    fn test_huffman_tables() {
        let tables = HuffmanTables::new();

        // DC category 0 should have a code
        let dc0 = tables.get_dc_code(0, true);
        assert!(dc0.length > 0);

        // AC EOB (0x00) should have a code
        let eob = tables.get_ac_code(0x00, true);
        assert!(eob.length > 0);
    }

    #[test]
    fn test_encode_block_zeros() {
        let tables = HuffmanTables::new();
        let mut writer = BitWriterMsb::new();
        let block = [0i16; 64];

        let new_dc = encode_block(&mut writer, &block, 0, true, &tables);

        assert_eq!(new_dc, 0);
        assert!(!writer.is_empty());
    }

    #[test]
    fn test_build_code_lengths_single_symbol() {
        // Only one symbol with non-zero frequency
        let counts = [0, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let lengths = build_code_lengths(&counts);
        assert!(lengths.is_some());
        let lengths = lengths.unwrap();
        // Single symbol should have length 1
        assert_eq!(lengths[2], 1);
        // Other symbols should have length 0
        assert_eq!(lengths[0], 0);
        assert_eq!(lengths[1], 0);
    }

    #[test]
    fn test_build_code_lengths_two_symbols() {
        // Two symbols with equal frequency
        let counts = [100, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let lengths = build_code_lengths(&counts);
        assert!(lengths.is_some());
        let lengths = lengths.unwrap();
        // Both should have equal length (the implementation uses depth+1 in tree traversal)
        assert_eq!(lengths[0], lengths[1]);
        assert!(lengths[0] > 0);
    }

    #[test]
    fn test_build_code_lengths_empty() {
        // All zeros - should return None
        let counts = [0u64; 12];
        let lengths = build_code_lengths(&counts);
        assert!(lengths.is_none());
    }

    #[test]
    fn test_build_code_lengths_unequal_frequencies() {
        // Unequal frequencies - common symbols should get shorter codes
        let counts = [1000, 100, 10, 1, 0, 0, 0, 0, 0, 0, 0, 0];
        let lengths = build_code_lengths(&counts);
        assert!(lengths.is_some());
        let lengths = lengths.unwrap();
        // Most common symbol should have shortest code
        assert!(lengths[0] <= lengths[1]);
        assert!(lengths[1] <= lengths[2]);
        assert!(lengths[2] <= lengths[3]);
    }

    #[test]
    fn test_build_bits_vals_basic() {
        // Simple case: two symbols
        let counts = [100u64, 50, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let result = build_bits_vals(&counts);
        assert!(result.is_some());
        let (bits, vals) = result.unwrap();

        // Should have exactly 2 codes (total)
        assert_eq!(vals.len(), 2);
        // bits array should contain 2 total codes across all lengths
        assert_eq!(bits.iter().map(|&b| b as usize).sum::<usize>(), 2);
    }

    #[test]
    fn test_build_bits_vals_empty() {
        let counts = [0u64; 12];
        let result = build_bits_vals(&counts);
        assert!(result.is_none());
    }

    #[test]
    fn test_build_code_table_basic() {
        // Standard DC luminance table
        let codes: Option<[HuffCode; 12]> = build_code_table::<12>(&DC_LUM_BITS, &DC_LUM_VALS, 12);
        assert!(codes.is_some());
        let codes = codes.unwrap();

        // All 12 symbols should have codes
        for i in 0..12 {
            assert!(codes[i].length > 0, "Symbol {i} should have a code");
        }
    }

    #[test]
    fn test_build_code_table_invalid_symbol() {
        // Invalid: symbol >= table_len
        let bits = [1u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let vals = [20u8]; // Symbol 20 is >= table_len of 12
        let result: Option<[HuffCode; 12]> = build_code_table::<12>(&bits, &vals, 12);
        assert!(result.is_none());
    }

    #[test]
    fn test_optimized_from_counts_basic() {
        // Create frequency counts that should produce valid tables
        let mut dc_lum_counts = [0u64; 12];
        dc_lum_counts[0] = 100; // Category 0 (DC = 0)
        dc_lum_counts[1] = 50; // Category 1

        let mut ac_lum_counts = [0u64; 256];
        ac_lum_counts[0x00] = 200; // EOB
        ac_lum_counts[0x01] = 100; // (0,1) - run=0, size=1

        let result =
            HuffmanTables::optimized_from_counts(&dc_lum_counts, None, &ac_lum_counts, None);
        assert!(result.is_some());
        let tables = result.unwrap();

        // Should have valid codes
        let dc0 = tables.get_dc_code(0, true);
        assert!(dc0.length > 0);
        let eob = tables.get_ac_code(0x00, true);
        assert!(eob.length > 0);
    }

    #[test]
    fn test_optimized_from_counts_with_chroma() {
        let mut dc_lum_counts = [0u64; 12];
        dc_lum_counts[0] = 100;
        dc_lum_counts[1] = 50;

        let mut dc_chrom_counts = [0u64; 12];
        dc_chrom_counts[0] = 80;
        dc_chrom_counts[1] = 40;

        let mut ac_lum_counts = [0u64; 256];
        ac_lum_counts[0x00] = 200;
        ac_lum_counts[0x01] = 100;

        let mut ac_chrom_counts = [0u64; 256];
        ac_chrom_counts[0x00] = 150;
        ac_chrom_counts[0x01] = 75;

        let result = HuffmanTables::optimized_from_counts(
            &dc_lum_counts,
            Some(&dc_chrom_counts),
            &ac_lum_counts,
            Some(&ac_chrom_counts),
        );
        assert!(result.is_some());
        let tables = result.unwrap();

        // Check both luminance and chrominance codes
        let dc_lum = tables.get_dc_code(0, true);
        let dc_chrom = tables.get_dc_code(0, false);
        assert!(dc_lum.length > 0);
        assert!(dc_chrom.length > 0);
    }

    #[test]
    fn test_category_large_values() {
        // Test larger values
        assert_eq!(category(256), 9);
        assert_eq!(category(512), 10);
        assert_eq!(category(1023), 10);
        assert_eq!(category(1024), 11);
        assert_eq!(category(-1024), 11);
        // Max i16 value
        assert_eq!(category(i16::MAX), 15);
        assert_eq!(category(i16::MIN + 1), 15);
    }

    #[test]
    fn test_encode_value_negative() {
        // Test negative value encoding (ones' complement)
        let (bits, len) = encode_value(-2);
        assert_eq!(len, 2);
        assert_eq!(bits, 1); // -2 in ones' complement for 2 bits = 01

        let (bits, len) = encode_value(-4);
        assert_eq!(len, 3);
        assert_eq!(bits, 3); // -4 in ones' complement for 3 bits = 011

        let (bits, len) = encode_value(-127);
        assert_eq!(len, 7);
        // -127 -> ones' complement is 0 (since -127 - 1 = -128 = 0x80, masked to 7 bits = 0)
        assert_eq!(bits, 0);
    }

    #[test]
    fn test_encode_block_with_dc_diff() {
        let tables = HuffmanTables::new();
        let mut writer = BitWriterMsb::new();

        let mut block = [0i16; 64];
        block[0] = 100; // DC coefficient

        let prev_dc = 50;
        let new_dc = encode_block(&mut writer, &block, prev_dc, true, &tables);

        // DC should return the block's DC value
        assert_eq!(new_dc, 100);
        // Should have written some data
        assert!(!writer.is_empty());
    }

    #[test]
    fn test_encode_block_with_ac_coefficients() {
        let tables = HuffmanTables::new();
        let mut writer = BitWriterMsb::new();

        let mut block = [0i16; 64];
        block[0] = 50; // DC
        block[1] = 10; // AC coefficient at position 1 in natural order
        block[8] = 5; // AC coefficient at position 8 in natural order

        let new_dc = encode_block(&mut writer, &block, 50, true, &tables);

        assert_eq!(new_dc, 50);
        // Should produce more output than a zero block
        let zero_block = [0i16; 64];
        let mut writer2 = BitWriterMsb::new();
        encode_block(&mut writer2, &zero_block, 0, true, &tables);

        // Block with AC coefficients should produce more data
        assert!(writer.len() > writer2.len());
    }

    #[test]
    fn test_encode_block_chrominance() {
        let tables = HuffmanTables::new();

        // Test luminance
        let mut writer_lum = BitWriterMsb::new();
        let mut block = [0i16; 64];
        block[0] = 30;
        encode_block(&mut writer_lum, &block, 0, true, &tables);

        // Test chrominance
        let mut writer_chrom = BitWriterMsb::new();
        encode_block(&mut writer_chrom, &block, 0, false, &tables);

        // Both should produce output (may be different sizes due to different tables)
        assert!(!writer_lum.is_empty());
        assert!(!writer_chrom.is_empty());
    }

    #[test]
    fn test_build_codes_standard_tables() {
        // Test that standard DC luminance codes are built correctly
        let codes = build_codes(&DC_LUM_BITS, &DC_LUM_VALS);

        // All 12 DC categories should have codes
        for i in 0..12 {
            assert!(codes[i].length > 0, "DC category {i} missing code");
            assert!(codes[i].length <= 16, "DC category {i} code too long");
        }
    }

    #[test]
    fn test_build_codes_256_standard_ac() {
        // Test that standard AC luminance codes are built correctly
        let codes = build_codes_256(&AC_LUM_BITS, &AC_LUM_VALS);

        // EOB (0x00) and common AC symbols should have codes
        assert!(codes[0x00].length > 0, "EOB missing");
        assert!(codes[0x01].length > 0, "AC (0,1) missing");
        assert!(codes[0xF0].length > 0, "ZRL missing");
    }

    #[test]
    fn test_encode_block_long_zero_run() {
        let tables = HuffmanTables::new();
        let mut writer = BitWriterMsb::new();

        // Create a block with zeros followed by a non-zero AC at the end
        let mut block = [0i16; 64];
        block[0] = 10; // DC
                       // Place a non-zero coefficient late in zigzag order
                       // Zigzag index 63 corresponds to position 63 in natural order
        block[63] = 5;

        let new_dc = encode_block(&mut writer, &block, 10, true, &tables);
        assert_eq!(new_dc, 10);
        // Should encode ZRL codes for long zero runs
        assert!(!writer.is_empty());
    }

    #[test]
    fn test_huffman_tables_default() {
        let tables = HuffmanTables::default();
        // Should be equivalent to new()
        let dc0 = tables.get_dc_code(0, true);
        assert!(dc0.length > 0);
    }
}
