//! CRC32 checksum implementation (PNG uses CRC-32/ISO-HDLC).
//!
//! Uses slice-by-8 algorithm for improved performance: processes 8 bytes
//! per iteration with 8 parallel table lookups.

/// CRC32 lookup tables for slice-by-8 algorithm.
/// 8 tables of 256 entries each, using polynomial 0xEDB88320.
const CRC_TABLES: [[u32; 256]; 8] = {
    // First, generate the base table
    let mut tables = [[0u32; 256]; 8];
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
        tables[0][i] = crc;
        i += 1;
    }

    // Generate tables 1-7 from table 0
    i = 0;
    while i < 256 {
        let mut crc = tables[0][i];
        let mut j = 1;
        while j < 8 {
            crc = (crc >> 8) ^ tables[0][(crc & 0xFF) as usize];
            tables[j][i] = crc;
            j += 1;
        }
        i += 1;
    }

    tables
};

/// Simple CRC32 table for scalar fallback.
const CRC_TABLE: [u32; 256] = CRC_TABLES[0];

/// Calculate CRC32 checksum of data.
///
/// Uses the slice-by-8 algorithm for improved performance.
/// Processes 8 bytes per iteration instead of 1.
#[inline]
pub fn crc32(data: &[u8]) -> u32 {
    crc32_slice8(data)
}

/// Slice-by-8 CRC32 implementation.
/// Processes 8 bytes per iteration with 8 parallel table lookups.
#[inline]
fn crc32_slice8(data: &[u8]) -> u32 {
    let mut crc = !0u32;
    let mut remaining = data;

    // Process 8 bytes at a time
    while remaining.len() >= 8 {
        // Load 8 bytes as two u32s
        let lo = u32::from_le_bytes([remaining[0], remaining[1], remaining[2], remaining[3]]);
        let hi = u32::from_le_bytes([remaining[4], remaining[5], remaining[6], remaining[7]]);

        // XOR with current CRC
        let val_lo = crc ^ lo;
        let val_hi = hi;

        // Perform 8 parallel table lookups and XOR together
        crc = CRC_TABLES[7][(val_lo & 0xFF) as usize]
            ^ CRC_TABLES[6][((val_lo >> 8) & 0xFF) as usize]
            ^ CRC_TABLES[5][((val_lo >> 16) & 0xFF) as usize]
            ^ CRC_TABLES[4][((val_lo >> 24) & 0xFF) as usize]
            ^ CRC_TABLES[3][(val_hi & 0xFF) as usize]
            ^ CRC_TABLES[2][((val_hi >> 8) & 0xFF) as usize]
            ^ CRC_TABLES[1][((val_hi >> 16) & 0xFF) as usize]
            ^ CRC_TABLES[0][((val_hi >> 24) & 0xFF) as usize];

        remaining = &remaining[8..];
    }

    // Handle remaining bytes (0-7) with simple table lookup
    for &byte in remaining {
        let index = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC_TABLE[index];
    }

    !crc
}

/// Calculate CRC32 incrementally.
pub struct Crc32 {
    crc: u32,
}

impl Crc32 {
    /// Create a new CRC32 calculator.
    pub fn new() -> Self {
        Self { crc: 0xFFFFFFFF }
    }

    /// Update the CRC with more data.
    /// Uses slice-by-8 for improved performance on larger chunks.
    #[inline]
    pub fn update(&mut self, data: &[u8]) {
        let mut remaining = data;

        // Process 8 bytes at a time
        while remaining.len() >= 8 {
            let lo = u32::from_le_bytes([remaining[0], remaining[1], remaining[2], remaining[3]]);
            let hi = u32::from_le_bytes([remaining[4], remaining[5], remaining[6], remaining[7]]);

            let val_lo = self.crc ^ lo;
            let val_hi = hi;

            self.crc = CRC_TABLES[7][(val_lo & 0xFF) as usize]
                ^ CRC_TABLES[6][((val_lo >> 8) & 0xFF) as usize]
                ^ CRC_TABLES[5][((val_lo >> 16) & 0xFF) as usize]
                ^ CRC_TABLES[4][((val_lo >> 24) & 0xFF) as usize]
                ^ CRC_TABLES[3][(val_hi & 0xFF) as usize]
                ^ CRC_TABLES[2][((val_hi >> 8) & 0xFF) as usize]
                ^ CRC_TABLES[1][((val_hi >> 16) & 0xFF) as usize]
                ^ CRC_TABLES[0][((val_hi >> 24) & 0xFF) as usize];

            remaining = &remaining[8..];
        }

        // Handle remaining bytes
        for &byte in remaining {
            let index = ((self.crc ^ byte as u32) & 0xFF) as usize;
            self.crc = (self.crc >> 8) ^ CRC_TABLE[index];
        }
    }

    /// Finalize and return the CRC value.
    #[inline]
    pub fn finalize(self) -> u32 {
        self.crc ^ 0xFFFFFFFF
    }
}

impl Default for Crc32 {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crc32_empty() {
        assert_eq!(crc32(&[]), 0x00000000);
    }

    #[test]
    fn test_crc32_check_value() {
        // Standard test: CRC32 of "123456789" should be 0xCBF43926
        let data = b"123456789";
        assert_eq!(crc32(data), 0xCBF43926);
    }

    #[test]
    fn test_crc32_incremental() {
        let data = b"123456789";

        // Full calculation
        let full_crc = crc32(data);

        // Incremental calculation
        let mut crc = Crc32::new();
        crc.update(&data[..4]);
        crc.update(&data[4..]);
        let incremental_crc = crc.finalize();

        assert_eq!(full_crc, incremental_crc);
    }

    #[test]
    fn test_crc32_png_iend() {
        // PNG IEND chunk has type "IEND" (no data)
        // CRC should be 0xAE426082
        let chunk_type = b"IEND";
        assert_eq!(crc32(chunk_type), 0xAE426082);
    }
}
