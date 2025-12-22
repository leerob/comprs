//! PNG chunk handling.

/// Write a PNG chunk to the output buffer.
///
/// Chunk format:
/// - 4 bytes: data length (big-endian)
/// - 4 bytes: chunk type (ASCII)
/// - N bytes: chunk data
/// - 4 bytes: CRC32 of type + data
pub fn write_chunk(output: &mut Vec<u8>, chunk_type: &[u8; 4], data: &[u8]) {
    // Reserve space upfront: length (4) + type (4) + data + crc (4)
    output.reserve(12 + data.len());

    // Compute CRC without extra allocation
    let mut crc = crate::compress::crc32::Crc32::new();
    crc.update(chunk_type);
    crc.update(data);
    let crc = crc.finalize();

    // Length (big-endian)
    output.extend_from_slice(&(data.len() as u32).to_be_bytes());

    // Chunk type
    output.extend_from_slice(chunk_type);

    // Data
    output.extend_from_slice(data);

    // CRC32 of type + data
    output.extend_from_slice(&crc.to_be_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_iend_chunk() {
        let mut output = Vec::new();
        write_chunk(&mut output, b"IEND", &[]);

        // IEND chunk should be 12 bytes (4 length + 4 type + 0 data + 4 CRC)
        assert_eq!(output.len(), 12);

        // Length should be 0
        assert_eq!(&output[0..4], &[0, 0, 0, 0]);

        // Type should be IEND
        assert_eq!(&output[4..8], b"IEND");

        // CRC of "IEND" should be 0xAE426082
        assert_eq!(&output[8..12], &0xAE426082_u32.to_be_bytes());
    }

    #[test]
    fn test_write_chunk_with_data() {
        let mut output = Vec::new();
        write_chunk(&mut output, b"tEXt", b"hello");

        // Should be 4 + 4 + 5 + 4 = 17 bytes
        assert_eq!(output.len(), 17);

        // Length should be 5
        assert_eq!(&output[0..4], &[0, 0, 0, 5]);

        // Type
        assert_eq!(&output[4..8], b"tEXt");

        // Data
        assert_eq!(&output[8..13], b"hello");
    }
}
