//! CRC32 checksum implementation (PNG uses CRC-32/ISO-HDLC).

/// Slicing-by-8 tables for CRC32 polynomial 0xEDB88320 (reflected 0x04C11DB7).
/// Built once at runtime; zero-cost thereafter.
static CRC_TABLES: std::sync::LazyLock<[[u32; 256]; 8]> = std::sync::LazyLock::new(|| {
    let mut tables = [[0u32; 256]; 8];

    // Table 0: classic byte-at-a-time.
    for (i, entry) in tables[0].iter_mut().enumerate() {
        let mut crc = i as u32;
        for _ in 0..8 {
            crc = if (crc & 1) != 0 {
                (crc >> 1) ^ 0xEDB88320
            } else {
                crc >> 1
            };
        }
        *entry = crc;
    }

    // Tables 1..7 derived from table 0.
    for t in 1..8 {
        for i in 0..256 {
            let prev = tables[t - 1][i];
            tables[t][i] = (prev >> 8) ^ tables[0][(prev & 0xFF) as usize];
        }
    }

    tables
});

/// Calculate CRC32 checksum of data.
///
/// Uses the CRC-32/ISO-HDLC algorithm (polynomial 0x04C11DB7 reflected).
/// This is the CRC used by PNG, gzip, and many other formats.
#[inline]
#[must_use]
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    let tables = &*CRC_TABLES;

    // Process 8 bytes at a time using slicing-by-8.
    let mut chunks = data.chunks_exact(8);
    for chunk in &mut chunks {
        let low = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        let high = u32::from_le_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);

        crc ^= low;

        crc = tables[7][(crc & 0xFF) as usize]
            ^ tables[6][((crc >> 8) & 0xFF) as usize]
            ^ tables[5][((crc >> 16) & 0xFF) as usize]
            ^ tables[4][((crc >> 24) & 0xFF) as usize]
            ^ tables[3][(high & 0xFF) as usize]
            ^ tables[2][((high >> 8) & 0xFF) as usize]
            ^ tables[1][((high >> 16) & 0xFF) as usize]
            ^ tables[0][((high >> 24) & 0xFF) as usize];
    }

    for &b in chunks.remainder() {
        let idx = ((crc ^ b as u32) & 0xFF) as usize;
        crc = (crc >> 8) ^ tables[0][idx];
    }

    crc ^ 0xFFFF_FFFF
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
    #[inline]
    pub fn update(&mut self, data: &[u8]) {
        let tables = &*CRC_TABLES;
        for &byte in data {
            let index = ((self.crc ^ byte as u32) & 0xFF) as usize;
            self.crc = (self.crc >> 8) ^ tables[0][index];
        }
    }

    /// Finalize and return the CRC value.
    #[inline]
    #[must_use]
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
