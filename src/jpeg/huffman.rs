//! JPEG Huffman encoding.
//!
//! Implements the entropy coding stage of JPEG compression.

use crate::bits::BitWriterMsb;
use crate::jpeg::quantize::zigzag_reorder;

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

    /// Get DC code for a category.
    fn get_dc_code(&self, category: u8, is_luminance: bool) -> HuffCode {
        if is_luminance {
            self.dc_lum_codes[category as usize]
        } else {
            self.dc_chrom_codes[category as usize]
        }
    }

    /// Get AC code for a (run, size) pair.
    fn get_ac_code(&self, rs: u8, is_luminance: bool) -> HuffCode {
        if is_luminance {
            self.ac_lum_codes[rs as usize]
        } else {
            self.ac_chrom_codes[rs as usize]
        }
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

/// Build Huffman codes for 256 AC symbols.
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
}
