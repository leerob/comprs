//! JPEG quantization tables and functions.

/// Standard JPEG luminance quantization table.
const STD_LUMINANCE_TABLE: [u8; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
    92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
];

/// Standard JPEG chrominance quantization table.
const STD_CHROMINANCE_TABLE: [u8; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99, 24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
];

/// Zigzag scan order for 8x8 block.
pub const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20,
    13, 6, 7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59,
    52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// Quantization tables for JPEG encoding.
pub struct QuantizationTables {
    /// Luminance quantization table (zigzag order for output).
    pub luminance: [u8; 64],
    /// Chrominance quantization table (zigzag order for output).
    pub chrominance: [u8; 64],
    /// Luminance table in natural order (for float computation).
    pub luminance_table: [f32; 64],
    /// Chrominance table in natural order (for float computation).
    pub chrominance_table: [f32; 64],
    /// Luminance table in natural order (for integer computation).
    pub luminance_table_int: [u16; 64],
    /// Chrominance table in natural order (for integer computation).
    pub chrominance_table_int: [u16; 64],
}

impl QuantizationTables {
    /// Create quantization tables with the given quality (1-100).
    pub fn with_quality(quality: u8) -> Self {
        let quality = quality.clamp(1, 100);

        // Calculate scale factor (same formula as libjpeg)
        let scale = if quality < 50 {
            5000 / quality as u32
        } else {
            200 - 2 * quality as u32
        };

        let mut luminance = [0u8; 64];
        let mut chrominance = [0u8; 64];
        let mut luminance_table = [0.0f32; 64];
        let mut chrominance_table = [0.0f32; 64];
        let mut luminance_table_int = [0u16; 64];
        let mut chrominance_table_int = [0u16; 64];

        for i in 0..64 {
            // Scale and clamp to 1-255
            let lum_val =
                ((STD_LUMINANCE_TABLE[ZIGZAG[i]] as u32 * scale + 50) / 100).clamp(1, 255);
            let chrom_val =
                ((STD_CHROMINANCE_TABLE[ZIGZAG[i]] as u32 * scale + 50) / 100).clamp(1, 255);

            luminance[i] = lum_val as u8;
            chrominance[i] = chrom_val as u8;
        }

        // Create tables in natural (non-zigzag) order for computation
        for i in 0..64 {
            let lum_val = ((STD_LUMINANCE_TABLE[i] as u32 * scale + 50) / 100).clamp(1, 255);
            let chrom_val = ((STD_CHROMINANCE_TABLE[i] as u32 * scale + 50) / 100).clamp(1, 255);

            luminance_table[i] = lum_val as f32;
            chrominance_table[i] = chrom_val as f32;
            luminance_table_int[i] = lum_val as u16;
            chrominance_table_int[i] = chrom_val as u16;
        }

        Self {
            luminance,
            chrominance,
            luminance_table,
            chrominance_table,
            luminance_table_int,
            chrominance_table_int,
        }
    }
}

impl Default for QuantizationTables {
    fn default() -> Self {
        Self::with_quality(75)
    }
}

/// Divides each coefficient by the corresponding quantization value.
pub fn quantize_block(dct: &[f32; 64], quant_table: &[f32; 64]) -> [i16; 64] {
    let mut result = [0i16; 64];
    for i in 0..64 {
        result[i] = (dct[i] / quant_table[i]).round() as i16;
    }
    result
}

pub fn zigzag_reorder(block: &[i16; 64]) -> [i16; 64] {
    let mut result = [0i16; 64];
    for i in 0..64 {
        result[i] = block[ZIGZAG[i]];
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_block() {
        let mut dct = [0.0f32; 64];
        dct[0] = 160.0; // DC component

        let tables = QuantizationTables::with_quality(75);
        let quantized = quantize_block(&dct, &tables.luminance_table);

        // DC should be quantized
        assert_ne!(quantized[0], 0);
    }

    #[test]
    fn test_quality_scaling() {
        let q50 = QuantizationTables::with_quality(50);
        let q90 = QuantizationTables::with_quality(90);

        // Higher quality = smaller quantization values = less loss
        assert!(q90.luminance[0] < q50.luminance[0]);
    }

    #[test]
    fn test_zigzag_order() {
        // First few zigzag indices should be: 0, 1, 8, 16, 9, 2, ...
        assert_eq!(ZIGZAG[0], 0);
        assert_eq!(ZIGZAG[1], 1);
        assert_eq!(ZIGZAG[2], 8);
        assert_eq!(ZIGZAG[3], 16);
        assert_eq!(ZIGZAG[4], 9);
        assert_eq!(ZIGZAG[5], 2);
    }

    #[test]
    fn test_zigzag_reorder() {
        let mut block = [0i16; 64];
        block[0] = 100; // DC
        block[1] = 50;
        block[8] = 30;

        let reordered = zigzag_reorder(&block);

        // After zigzag, positions should be rearranged
        assert_eq!(reordered[0], 100); // DC stays at 0
        assert_eq!(reordered[1], 50); // Position 1 in zigzag is position 1 in natural
        assert_eq!(reordered[2], 30); // Position 2 in zigzag is position 8 in natural
    }

    #[test]
    fn test_quantization_tables_quality_1() {
        let tables = QuantizationTables::with_quality(1);
        // Quality 1 should produce high quantization values
        assert!(tables.luminance[0] > 100); // DC quantizer should be high
    }

    #[test]
    fn test_quantization_tables_quality_100() {
        let tables = QuantizationTables::with_quality(100);
        // Quality 100 should produce low quantization values (least loss)
        assert!(tables.luminance[0] < 20); // DC quantizer should be low
    }

    #[test]
    fn test_quantization_tables_quality_clamping() {
        // Quality 0 should be clamped to 1
        let tables_q0 = QuantizationTables::with_quality(0);
        let tables_q1 = QuantizationTables::with_quality(1);
        assert_eq!(tables_q0.luminance, tables_q1.luminance);

        // Quality 101 should be clamped to 100
        let tables_q101 = QuantizationTables::with_quality(101);
        let tables_q100 = QuantizationTables::with_quality(100);
        assert_eq!(tables_q101.luminance, tables_q100.luminance);
    }

    #[test]
    fn test_quantization_tables_default() {
        let tables = QuantizationTables::default();
        let tables_q75 = QuantizationTables::with_quality(75);
        // Default should be quality 75
        assert_eq!(tables.luminance, tables_q75.luminance);
    }

    #[test]
    fn test_quantization_tables_all_formats() {
        let tables = QuantizationTables::with_quality(85);

        // Check that all table formats are populated
        assert_eq!(tables.luminance.len(), 64);
        assert_eq!(tables.chrominance.len(), 64);
        assert_eq!(tables.luminance_table.len(), 64);
        assert_eq!(tables.chrominance_table.len(), 64);
        assert_eq!(tables.luminance_table_int.len(), 64);
        assert_eq!(tables.chrominance_table_int.len(), 64);
    }

    #[test]
    fn test_quantization_values_range() {
        for q in [1, 25, 50, 75, 100] {
            let tables = QuantizationTables::with_quality(q);
            for &val in &tables.luminance {
                assert!(val >= 1, "Quality {q}: value {val} is zero");
            }
            for &val in &tables.chrominance {
                assert!(val >= 1, "Quality {q}: chrom value {val} is zero");
            }
        }
    }

    #[test]
    fn test_quantize_block_rounding() {
        let mut dct = [0.0f32; 64];
        dct[0] = 16.5; // Should round to 1 when divided by 16
        dct[1] = 16.4; // Should round to 1 when divided by 16
        dct[2] = 16.6; // Should round to 1 when divided by 16

        let mut quant = [16.0f32; 64];
        quant[0] = 16.0;

        let result = quantize_block(&dct, &quant);

        assert_eq!(result[0], 1); // 16.5 / 16 = 1.03125 -> 1
        assert_eq!(result[1], 1); // 16.4 / 16 = 1.025 -> 1
        assert_eq!(result[2], 1); // 16.6 / 16 = 1.0375 -> 1
    }

    #[test]
    fn test_quantize_block_negative() {
        let mut dct = [0.0f32; 64];
        dct[0] = -160.0;

        let quant = [16.0f32; 64];
        let result = quantize_block(&dct, &quant);

        assert_eq!(result[0], -10); // -160 / 16 = -10
    }

    #[test]
    fn test_quantize_block_zeros() {
        let dct = [0.0f32; 64];
        let quant = [16.0f32; 64];
        let result = quantize_block(&dct, &quant);

        for &val in &result {
            assert_eq!(val, 0);
        }
    }

    #[test]
    fn test_zigzag_complete() {
        // Verify zigzag covers all 64 positions exactly once
        let mut seen = [false; 64];
        for &pos in &ZIGZAG {
            assert!(!seen[pos], "Duplicate position {pos} in zigzag");
            seen[pos] = true;
        }
        for (i, &s) in seen.iter().enumerate() {
            assert!(s, "Position {i} missing from zigzag");
        }
    }

    #[test]
    fn test_zigzag_reorder_all_values() {
        // Create block with unique values at each position
        let mut block = [0i16; 64];
        for i in 0..64 {
            block[i] = i as i16;
        }

        let reordered = zigzag_reorder(&block);

        // Verify all values appear (may be in different order)
        let mut seen = [false; 64];
        for &val in &reordered {
            seen[val as usize] = true;
        }
        for (i, &s) in seen.iter().enumerate() {
            assert!(s, "Value {i} missing after zigzag reorder");
        }
    }

    #[test]
    fn test_quality_50_is_identity() {
        // Quality 50 should use scale factor 100 (no change to standard tables)
        let tables = QuantizationTables::with_quality(50);

        // DC luminance should be exactly 16 (from standard table)
        assert_eq!(tables.luminance[0], 16);
    }

    #[test]
    fn test_float_and_int_tables_consistent() {
        let tables = QuantizationTables::with_quality(85);

        // Float and int tables should have same values
        for i in 0..64 {
            assert_eq!(
                tables.luminance_table[i] as u16, tables.luminance_table_int[i],
                "Mismatch at position {i}"
            );
            assert_eq!(
                tables.chrominance_table[i] as u16, tables.chrominance_table_int[i],
                "Chroma mismatch at position {i}"
            );
        }
    }
}
