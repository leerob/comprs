//! Progressive JPEG encoding support.
//!
//! Implements progressive DCT-based JPEG encoding with multiple scans.
//! Progressive encoding can achieve 5-15% better compression than baseline
//! by using spectral selection and successive approximation.

use crate::bits::BitWriterMsb;
use crate::jpeg::huffman::HuffmanTables;
use crate::jpeg::quantize::zigzag_reorder;

/// Specification for a single scan in progressive JPEG.
#[derive(Debug, Clone)]
pub struct ScanSpec {
    /// Component indices to include in this scan (0=Y, 1=Cb, 2=Cr)
    pub components: Vec<u8>,
    /// Start of spectral selection (0-63)
    pub ss: u8,
    /// End of spectral selection (0-63)
    pub se: u8,
    /// Successive approximation high bit position
    pub ah: u8,
    /// Successive approximation low bit position
    pub al: u8,
}

impl ScanSpec {
    pub fn new(components: Vec<u8>, ss: u8, se: u8, ah: u8, al: u8) -> Self {
        Self {
            components,
            ss,
            se,
            ah,
            al,
        }
    }

    pub fn is_dc_scan(&self) -> bool {
        self.ss == 0 && self.se == 0
    }

    pub fn is_first_scan(&self) -> bool {
        self.ah == 0
    }

    pub fn is_refinement_scan(&self) -> bool {
        self.ah > 0
    }
}

/// Default progressive scan script optimized for compression.
/// Based on mozjpeg's default progressive script.
pub fn default_progressive_script() -> Vec<ScanSpec> {
    vec![
        // DC scans with successive approximation
        ScanSpec::new(vec![0], 0, 0, 0, 1), // Y DC, initial
        ScanSpec::new(vec![1], 0, 0, 0, 1), // Cb DC, initial
        ScanSpec::new(vec![2], 0, 0, 0, 1), // Cr DC, initial
        // Y AC scans - low frequencies first
        ScanSpec::new(vec![0], 1, 5, 0, 2),   // Y AC 1-5, initial
        ScanSpec::new(vec![0], 6, 14, 0, 2),  // Y AC 6-14, initial
        ScanSpec::new(vec![0], 15, 63, 0, 1), // Y AC 15-63, initial
        // Chroma AC scans
        ScanSpec::new(vec![1], 1, 63, 0, 1), // Cb AC, initial
        ScanSpec::new(vec![2], 1, 63, 0, 1), // Cr AC, initial
        // DC refinement
        ScanSpec::new(vec![0], 0, 0, 1, 0), // Y DC refine
        ScanSpec::new(vec![1], 0, 0, 1, 0), // Cb DC refine
        ScanSpec::new(vec![2], 0, 0, 1, 0), // Cr DC refine
        // Y AC refinement scans
        ScanSpec::new(vec![0], 1, 5, 2, 1),   // Y AC 1-5 refine
        ScanSpec::new(vec![0], 1, 5, 1, 0),   // Y AC 1-5 final
        ScanSpec::new(vec![0], 6, 14, 2, 1),  // Y AC 6-14 refine
        ScanSpec::new(vec![0], 6, 14, 1, 0),  // Y AC 6-14 final
        ScanSpec::new(vec![0], 15, 63, 1, 0), // Y AC 15-63 final
        // Chroma AC refinement
        ScanSpec::new(vec![1], 1, 63, 1, 0), // Cb AC final
        ScanSpec::new(vec![2], 1, 63, 1, 0), // Cr AC final
    ]
}

/// Simple progressive scan script for faster encoding with good compression.
pub fn simple_progressive_script() -> Vec<ScanSpec> {
    vec![
        // DC scans (all components together for simpler encoding)
        ScanSpec::new(vec![0], 0, 0, 0, 0), // Y DC
        ScanSpec::new(vec![1], 0, 0, 0, 0), // Cb DC
        ScanSpec::new(vec![2], 0, 0, 0, 0), // Cr DC
        // AC scans - spectral selection only (no successive approximation)
        ScanSpec::new(vec![0], 1, 10, 0, 0),  // Y AC low
        ScanSpec::new(vec![0], 11, 63, 0, 0), // Y AC high
        ScanSpec::new(vec![1], 1, 63, 0, 0),  // Cb AC
        ScanSpec::new(vec![2], 1, 63, 0, 0),  // Cr AC
    ]
}

/// Encode DC coefficient for progressive scan (first scan only).
pub fn encode_dc_first(
    writer: &mut BitWriterMsb,
    dc_diff: i16,
    al: u8,
    dc_code: (u16, u8), // (code, length) from Huffman table
) {
    // For successive approximation, we only encode the high bits
    let shifted_dc = dc_diff >> al;

    // Write the Huffman code for the category
    writer.write_bits(dc_code.0 as u32, dc_code.1);

    // Write the value bits if non-zero
    if shifted_dc != 0 {
        let cat = category(shifted_dc);
        let (val_bits, val_len) = encode_value(shifted_dc);
        writer.write_bits(val_bits as u32, val_len);
        let _ = cat; // Category was used to get the Huffman code
    }
}

pub fn encode_dc_refine(writer: &mut BitWriterMsb, dc_coef: i16, al: u8) {
    // Output the bit at position `al`
    let bit = ((dc_coef.unsigned_abs() >> al) & 1) as u32;
    writer.write_bits(bit, 1);
}

/// Encode AC coefficients for first progressive scan.
#[allow(clippy::too_many_arguments)]
pub fn encode_ac_first(
    writer: &mut BitWriterMsb,
    block: &[i16; 64],
    ss: u8,
    se: u8,
    al: u8,
    eob_run: &mut u16,
    tables: &HuffmanTables,
    is_luminance: bool,
) {
    let zigzag = zigzag_reorder(block);

    // Find last non-zero coefficient in range [ss, se]
    let mut k = se as usize;
    while k >= ss as usize && (zigzag[k] >> al) == 0 {
        if k == ss as usize {
            break;
        }
        k -= 1;
    }

    // If all zeros, accumulate EOB run
    let last_nonzero = k;
    let all_zero = last_nonzero == ss as usize && (zigzag[ss as usize] >> al) == 0;

    if all_zero {
        *eob_run += 1;
        // Flush if EOB run reaches maximum
        if *eob_run == 0x7FFF {
            flush_eob_run(writer, eob_run, tables, is_luminance);
        }
        return;
    }

    // Flush pending EOB run before encoding non-zero block
    if *eob_run > 0 {
        flush_eob_run(writer, eob_run, tables, is_luminance);
    }

    // Encode coefficients
    let mut zero_run = 0u8;
    for k in (ss as usize)..=(last_nonzero) {
        let coef = zigzag[k] >> al;

        if coef == 0 {
            zero_run += 1;
            continue;
        }

        // Emit ZRL codes for runs of 16+ zeros
        while zero_run >= 16 {
            let zrl_code = get_ac_code(tables, 0xF0, is_luminance);
            writer.write_bits(zrl_code.0 as u32, zrl_code.1);
            zero_run -= 16;
        }

        // Emit (run, size) code and value
        let ac_cat = category(coef);
        let rs = (zero_run << 4) | ac_cat;
        let ac_code = get_ac_code(tables, rs, is_luminance);
        writer.write_bits(ac_code.0 as u32, ac_code.1);

        let (val_bits, val_len) = encode_value(coef);
        writer.write_bits(val_bits as u32, val_len);

        zero_run = 0;
    }

    // If we ended before se, start an EOB run
    if last_nonzero < se as usize {
        *eob_run = 1;
    }
}

#[allow(clippy::too_many_arguments)]
pub fn encode_ac_refine(
    writer: &mut BitWriterMsb,
    block: &[i16; 64],
    ss: u8,
    se: u8,
    al: u8,
    eob_run: &mut u16,
    tables: &HuffmanTables,
    is_luminance: bool,
) {
    let zigzag = zigzag_reorder(block);

    // Collect correction bits for previously non-zero coefficients
    let mut correction_bits: Vec<u8> = Vec::new();
    let mut zero_run = 0u8;
    let mut pending_eob = false;

    // Find end of band
    let mut end = se as usize;
    while end > ss as usize {
        let coef = zigzag[end];
        if coef.abs() > (1 << al) {
            break;
        }
        if (coef.unsigned_abs() >> al) & 1 != 0 {
            break;
        }
        end -= 1;
    }

    for k in (ss as usize)..=(se as usize) {
        let coef = zigzag[k];
        let abs_coef = coef.unsigned_abs();

        if abs_coef > (1 << al) {
            // Previously non-zero: emit correction bit later
            correction_bits.push(((abs_coef >> al) & 1) as u8);
        } else if (abs_coef >> al) & 1 != 0 {
            // Newly significant coefficient
            if *eob_run > 0 {
                flush_eob_run(writer, eob_run, tables, is_luminance);
            }

            // Emit zero run and new coefficient
            while zero_run >= 16 {
                // ZRL with correction bits
                let zrl_code = get_ac_code(tables, 0xF0, is_luminance);
                writer.write_bits(zrl_code.0 as u32, zrl_code.1);
                // Output pending correction bits
                for &bit in &correction_bits {
                    writer.write_bits(bit as u32, 1);
                }
                correction_bits.clear();
                zero_run -= 16;
            }

            // (run, 1) code
            let rs = (zero_run << 4) | 1;
            let ac_code = get_ac_code(tables, rs, is_luminance);
            writer.write_bits(ac_code.0 as u32, ac_code.1);

            // Sign bit
            let sign_bit = if coef < 0 { 0u32 } else { 1u32 };
            writer.write_bits(sign_bit, 1);

            // Output pending correction bits
            for &bit in &correction_bits {
                writer.write_bits(bit as u32, 1);
            }
            correction_bits.clear();
            zero_run = 0;
        } else {
            // Zero coefficient
            zero_run += 1;
        }
    }

    // Handle end of band
    if zero_run > 0 || !correction_bits.is_empty() {
        pending_eob = true;
    }

    if pending_eob {
        *eob_run += 1;
        if *eob_run == 0x7FFF {
            flush_eob_run(writer, eob_run, tables, is_luminance);
        }
    }

    // Output remaining correction bits
    for &bit in &correction_bits {
        writer.write_bits(bit as u32, 1);
    }
}

pub fn flush_eob_run_public(
    writer: &mut BitWriterMsb,
    eob_run: &mut u16,
    tables: &HuffmanTables,
    is_luminance: bool,
) {
    flush_eob_run(writer, eob_run, tables, is_luminance);
}

fn flush_eob_run(
    writer: &mut BitWriterMsb,
    eob_run: &mut u16,
    tables: &HuffmanTables,
    is_luminance: bool,
) {
    if *eob_run == 0 {
        return;
    }

    // Encode EOB run length
    // EOB runs use symbols 0x00-0x0E where the value is log2(run_length)
    let mut temp = *eob_run;
    let mut nbits = 0u8;
    while temp > 0 {
        temp >>= 1;
        nbits += 1;
    }
    nbits = nbits.saturating_sub(1);

    // Symbol is (nbits << 4) for EOB runs
    let symbol = nbits << 4;
    let ac_code = get_ac_code(tables, symbol, is_luminance);
    writer.write_bits(ac_code.0 as u32, ac_code.1);

    // Write extra bits for run length if nbits > 0
    if nbits > 0 {
        let extra_bits = *eob_run - (1 << nbits);
        writer.write_bits(extra_bits as u32, nbits);
    }

    *eob_run = 0;
}

pub fn get_dc_code(tables: &HuffmanTables, category: u8, is_luminance: bool) -> (u16, u8) {
    if is_luminance {
        get_code_from_table(&tables.dc_lum_bits, &tables.dc_lum_vals, category)
    } else {
        get_code_from_table(&tables.dc_chrom_bits, &tables.dc_chrom_vals, category)
    }
}

fn get_ac_code(tables: &HuffmanTables, rs: u8, is_luminance: bool) -> (u16, u8) {
    if is_luminance {
        get_code_from_table(&tables.ac_lum_bits, &tables.ac_lum_vals, rs)
    } else {
        get_code_from_table(&tables.ac_chrom_bits, &tables.ac_chrom_vals, rs)
    }
}

fn get_code_from_table(bits: &[u8; 16], vals: &[u8], symbol: u8) -> (u16, u8) {
    let mut code = 0u16;
    let mut val_idx = 0;

    for (length, &count) in bits.iter().enumerate() {
        for _ in 0..count {
            if val_idx < vals.len() && vals[val_idx] == symbol {
                return (code, (length + 1) as u8);
            }
            val_idx += 1;
            code += 1;
        }
        code <<= 1;
    }

    // Fallback: return EOB code if symbol not found
    (0, 4)
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
        (value - 1) as u16
    } else {
        value as u16
    };

    (bits & ((1 << cat) - 1), cat)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan_spec_properties() {
        let dc_scan = ScanSpec::new(vec![0], 0, 0, 0, 1);
        assert!(dc_scan.is_dc_scan());
        assert!(dc_scan.is_first_scan());
        assert!(!dc_scan.is_refinement_scan());

        let ac_refine = ScanSpec::new(vec![0], 1, 63, 1, 0);
        assert!(!ac_refine.is_dc_scan());
        assert!(ac_refine.is_refinement_scan());
    }

    #[test]
    fn test_default_script_coverage() {
        let script = default_progressive_script();

        // All scans should have valid component indices
        for scan in &script {
            for &c in &scan.components {
                assert!(c < 3, "Invalid component index");
            }
            assert!(scan.ss <= scan.se);
            assert!(scan.se <= 63);
        }
    }

    #[test]
    fn test_simple_script_coverage() {
        let script = simple_progressive_script();

        // All scans should have valid component indices
        for scan in &script {
            for &c in &scan.components {
                assert!(c < 3, "Invalid component index");
            }
            assert!(scan.ss <= scan.se);
            assert!(scan.se <= 63);
        }
        // Simple script has fewer scans than default
        assert!(script.len() < default_progressive_script().len());
    }

    #[test]
    fn test_category() {
        assert_eq!(category(0), 0);
        assert_eq!(category(1), 1);
        assert_eq!(category(-1), 1);
        assert_eq!(category(127), 7);
        assert_eq!(category(-128), 8);
    }

    #[test]
    fn test_encode_value() {
        // Zero value
        assert_eq!(encode_value(0), (0, 0));

        // Positive values
        let (bits, len) = encode_value(1);
        assert_eq!(len, 1);
        assert_eq!(bits, 1);

        let (bits, len) = encode_value(127);
        assert_eq!(len, 7);
        assert_eq!(bits, 127);

        // Negative values (encoded as ones' complement)
        let (bits, len) = encode_value(-1);
        assert_eq!(len, 1);
        assert_eq!(bits, 0); // -1 -> 0 in ones' complement for 1 bit

        let (_bits, len) = encode_value(-127);
        assert_eq!(len, 7);
    }

    #[test]
    fn test_scan_spec_first_vs_refinement() {
        // First scan has ah=0
        let first = ScanSpec::new(vec![0], 0, 0, 0, 1);
        assert!(first.is_first_scan());
        assert!(!first.is_refinement_scan());

        // Refinement scan has ah>0
        let refine = ScanSpec::new(vec![0], 0, 0, 1, 0);
        assert!(!refine.is_first_scan());
        assert!(refine.is_refinement_scan());
    }

    #[test]
    fn test_get_code_from_table_fallback() {
        // Test with empty/minimal tables to exercise fallback path
        let bits = [0u8; 16];
        let vals: [u8; 0] = [];
        let (code, len) = get_code_from_table(&bits, &vals, 0);
        // Fallback returns EOB code
        assert_eq!((code, len), (0, 4));
    }
}
