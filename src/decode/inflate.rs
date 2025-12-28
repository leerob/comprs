//! DEFLATE decompression (RFC 1951).
//!
//! Implements the INFLATE algorithm for decompressing DEFLATE streams,
//! used by PNG's IDAT chunks (with zlib wrapper).

use super::bit_reader::BitReader;
use crate::compress::adler32::adler32;
use crate::error::{Error, Result};

/// Length code base values (codes 257-285) - same as encoder.
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

/// Order of code length codes for dynamic Huffman.
const CODE_LENGTH_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

/// Maximum code length for DEFLATE Huffman codes.
const MAX_BITS: u8 = 15;

/// Lookup table entry: (symbol, length) packed.
/// If length > LOOKUP_BITS, need slow path.
const LOOKUP_BITS: u8 = 9;

/// Huffman decoding table with fast lookup.
struct HuffmanTable {
    /// Fast lookup table: index by peeking LOOKUP_BITS bits.
    /// Entry format: low 12 bits = symbol, bits 12-15 = code length.
    /// If length > LOOKUP_BITS, use slow path.
    lookup: Vec<u16>,
    /// Code lengths for each symbol (for building and slow path).
    lengths: Vec<u8>,
    /// Maximum code length in this table.
    max_len: u8,
}

impl HuffmanTable {
    /// Build a Huffman table from code lengths.
    fn from_lengths(lengths: &[u8]) -> Result<Self> {
        let max_len = *lengths.iter().max().unwrap_or(&0);
        if max_len > MAX_BITS {
            return Err(Error::InvalidDecode("code length too large".into()));
        }
        if max_len == 0 {
            // All zeros - empty table
            return Ok(Self {
                lookup: vec![0; 1 << LOOKUP_BITS],
                lengths: lengths.to_vec(),
                max_len: 0,
            });
        }

        // Count codes of each length
        let mut bl_count = [0u32; 16];
        for &len in lengths {
            if len > 0 {
                bl_count[len as usize] += 1;
            }
        }

        // Find the starting code for each length
        let mut next_code = [0u32; 16];
        let mut code = 0u32;
        for bits in 1..=MAX_BITS {
            code = (code + bl_count[bits as usize - 1]) << 1;
            next_code[bits as usize] = code;
        }

        // Assign codes to symbols
        let mut codes = vec![0u32; lengths.len()];
        for (symbol, &len) in lengths.iter().enumerate() {
            if len > 0 {
                codes[symbol] = next_code[len as usize];
                next_code[len as usize] += 1;
            }
        }

        // Build lookup table
        let mut lookup = vec![0u16; 1 << LOOKUP_BITS];

        for (symbol, &len) in lengths.iter().enumerate() {
            if len == 0 || len > LOOKUP_BITS {
                continue;
            }

            // Reverse the code for LSB-first reading
            let code = codes[symbol];
            let reversed = reverse_bits(code as u16, len);

            // Fill all entries that have this code as prefix
            let fill_count = 1 << (LOOKUP_BITS - len);
            for i in 0..fill_count {
                let index = reversed as usize | (i << len);
                // Pack symbol and length: symbol in low 12 bits, length in high 4 bits
                lookup[index] = (symbol as u16) | ((len as u16) << 12);
            }
        }

        // Mark entries for codes longer than LOOKUP_BITS
        // (These use slow path, marked with length = 0)

        Ok(Self {
            lookup,
            lengths: lengths.to_vec(),
            max_len,
        })
    }

    /// Decode a symbol using the Huffman table.
    fn decode(&self, reader: &mut BitReader) -> Result<u16> {
        if self.max_len == 0 {
            return Err(Error::InvalidDecode("empty Huffman table".into()));
        }

        // Try to peek LOOKUP_BITS bits for fast path.
        // If we're near end of stream, we may have fewer bits available.
        let (peek, available) = reader.try_peek_bits(LOOKUP_BITS)?;

        if available >= LOOKUP_BITS {
            // Fast path: full lookup table available
            let entry = self.lookup[peek as usize];
            let len = (entry >> 12) as u8;

            if len > 0 && len <= LOOKUP_BITS {
                reader.consume(len);
                return Ok(entry & 0xFFF);
            }

            // Code is longer than LOOKUP_BITS, use slow path
            return self.decode_slow(reader);
        }

        // Near end of stream: fewer than LOOKUP_BITS bits available.
        // Check if any short code matches the available bits.
        if available > 0 {
            let entry = self.lookup[peek as usize];
            let len = (entry >> 12) as u8;

            if len > 0 && len <= available {
                reader.consume(len);
                return Ok(entry & 0xFFF);
            }
        }

        // Fall back to slow path for remaining bits
        self.decode_slow(reader)
    }

    /// Slow path for codes longer than LOOKUP_BITS.
    fn decode_slow(&self, reader: &mut BitReader) -> Result<u16> {
        let mut code = 0u32;
        for len in 1..=self.max_len {
            code = (code << 1) | reader.read_bits(1)?;
            // Check if this code matches any symbol
            for (symbol, &sym_len) in self.lengths.iter().enumerate() {
                if sym_len == len {
                    // Compare against the expected code for this symbol
                    let expected = self.code_for_symbol(symbol);
                    if code == expected {
                        return Ok(symbol as u16);
                    }
                }
            }
        }
        Err(Error::InvalidDecode("invalid Huffman code".into()))
    }

    /// Get the code for a symbol (used in slow path).
    fn code_for_symbol(&self, symbol: usize) -> u32 {
        let len = self.lengths[symbol];
        if len == 0 {
            return u32::MAX;
        }

        // Recompute the code (could cache this)
        let mut bl_count = [0u32; 16];
        for &l in &self.lengths {
            if l > 0 {
                bl_count[l as usize] += 1;
            }
        }

        let mut next_code = [0u32; 16];
        let mut code = 0u32;
        for bits in 1..=MAX_BITS {
            code = (code + bl_count[bits as usize - 1]) << 1;
            next_code[bits as usize] = code;
        }

        for (sym, &l) in self.lengths.iter().enumerate() {
            if l > 0 {
                if sym == symbol {
                    return next_code[l as usize];
                }
                next_code[l as usize] += 1;
            }
        }
        u32::MAX
    }
}

/// Reverse bits in a value (for LSB-first code reconstruction).
fn reverse_bits(value: u16, length: u8) -> u16 {
    let mut result = 0u16;
    let mut v = value;
    for _ in 0..length {
        result = (result << 1) | (v & 1);
        v >>= 1;
    }
    result
}

/// Build fixed Huffman tables per RFC 1951.
fn fixed_literal_table() -> Result<HuffmanTable> {
    let mut lengths = vec![0u8; 288];
    // Codes 0-143: 8 bits
    for len in lengths.iter_mut().take(144) {
        *len = 8;
    }
    // Codes 144-255: 9 bits
    for len in lengths.iter_mut().take(256).skip(144) {
        *len = 9;
    }
    // Codes 256-279: 7 bits
    for len in lengths.iter_mut().take(280).skip(256) {
        *len = 7;
    }
    // Codes 280-287: 8 bits
    for len in lengths.iter_mut().take(288).skip(280) {
        *len = 8;
    }
    HuffmanTable::from_lengths(&lengths)
}

fn fixed_distance_table() -> Result<HuffmanTable> {
    // All 32 distance codes use 5 bits
    let lengths = vec![5u8; 32];
    HuffmanTable::from_lengths(&lengths)
}

/// Inflate a raw DEFLATE stream with optional expected output size.
///
/// When `expected_size` is provided, the output buffer is pre-allocated
/// to the exact size for better memory efficiency.
fn inflate_with_size(data: &[u8], expected_size: Option<usize>) -> Result<Vec<u8>> {
    let mut reader = BitReader::new(data);
    let mut output = Vec::with_capacity(expected_size.unwrap_or(data.len() * 4));

    loop {
        let bfinal = reader.read_bits(1)?;
        let btype = reader.read_bits(2)?;

        match btype {
            0 => inflate_stored(&mut reader, &mut output)?,
            1 => inflate_fixed(&mut reader, &mut output)?,
            2 => inflate_dynamic(&mut reader, &mut output)?,
            3 => return Err(Error::InvalidDecode("reserved block type".into())),
            _ => unreachable!(),
        }

        if bfinal == 1 {
            break;
        }
    }

    Ok(output)
}

/// Inflate a zlib-wrapped stream with optional expected output size.
///
/// When `expected_size` is provided:
/// - Output buffer is pre-allocated to the exact size
/// - Final size is validated against the expected size
pub fn inflate_zlib_with_size(data: &[u8], expected_size: Option<usize>) -> Result<Vec<u8>> {
    if data.len() < 6 {
        return Err(Error::InvalidDecode("zlib stream too short".into()));
    }

    // Parse zlib header
    let cmf = data[0];
    let flg = data[1];

    // Check compression method (must be 8 = deflate)
    if cmf & 0x0F != 8 {
        return Err(Error::InvalidDecode(
            "invalid zlib compression method".into(),
        ));
    }

    // Check header checksum
    if (((cmf as u16) << 8) | (flg as u16)) % 31 != 0 {
        return Err(Error::InvalidDecode("invalid zlib header checksum".into()));
    }

    // Check for preset dictionary (not supported)
    if flg & 0x20 != 0 {
        return Err(Error::UnsupportedDecode(
            "preset dictionary not supported".into(),
        ));
    }

    // Inflate the deflate stream (skip 2-byte header)
    let deflate_end = data.len() - 4; // 4 bytes for Adler32
    let output = inflate_with_size(&data[2..deflate_end], expected_size)?;

    // Verify Adler32 checksum
    let stored_checksum = u32::from_be_bytes([
        data[deflate_end],
        data[deflate_end + 1],
        data[deflate_end + 2],
        data[deflate_end + 3],
    ]);
    let computed_checksum = adler32(&output);

    if stored_checksum != computed_checksum {
        return Err(Error::InvalidDecode(format!(
            "Adler32 mismatch: expected {stored_checksum:08X}, got {computed_checksum:08X}"
        )));
    }

    // Validate size if expected
    if let Some(expected) = expected_size {
        if output.len() != expected {
            return Err(Error::InvalidDecode(format!(
                "decompressed size mismatch: expected {expected}, got {}",
                output.len()
            )));
        }
    }

    Ok(output)
}

/// Inflate a stored (uncompressed) block.
fn inflate_stored(reader: &mut BitReader, output: &mut Vec<u8>) -> Result<()> {
    // Align to byte boundary
    reader.align_to_byte();

    // Read LEN and NLEN
    let len = reader.read_bits(16)? as u16;
    let nlen = reader.read_bits(16)? as u16;

    // Verify NLEN is one's complement of LEN
    if len != !nlen {
        return Err(Error::InvalidDecode(
            "stored block LEN/NLEN mismatch".into(),
        ));
    }

    // Read raw bytes
    let start = output.len();
    output.resize(start + len as usize, 0);
    reader.read_bytes(&mut output[start..])?;

    Ok(())
}

/// Inflate a block with fixed Huffman codes.
fn inflate_fixed(reader: &mut BitReader, output: &mut Vec<u8>) -> Result<()> {
    let lit_table = fixed_literal_table()?;
    let dist_table = fixed_distance_table()?;
    inflate_block(reader, output, &lit_table, &dist_table)
}

/// Inflate a block with dynamic Huffman codes.
fn inflate_dynamic(reader: &mut BitReader, output: &mut Vec<u8>) -> Result<()> {
    // Read code counts
    let hlit = reader.read_bits(5)? as usize + 257; // 257-286
    let hdist = reader.read_bits(5)? as usize + 1; // 1-32
    let hclen = reader.read_bits(4)? as usize + 4; // 4-19

    // Read code length code lengths
    let mut cl_lengths = [0u8; 19];
    for i in 0..hclen {
        cl_lengths[CODE_LENGTH_ORDER[i]] = reader.read_bits(3)? as u8;
    }

    // Build code length Huffman table
    let cl_table = HuffmanTable::from_lengths(&cl_lengths)?;

    // Read literal/length and distance code lengths
    let mut lengths = vec![0u8; hlit + hdist];
    let mut i = 0;
    while i < lengths.len() {
        let symbol = cl_table.decode(reader)?;

        match symbol {
            0..=15 => {
                lengths[i] = symbol as u8;
                i += 1;
            }
            16 => {
                // Repeat previous length 3-6 times
                if i == 0 {
                    return Err(Error::InvalidDecode("repeat code at start".into()));
                }
                let repeat = reader.read_bits(2)? as usize + 3;
                let prev = lengths[i - 1];
                for _ in 0..repeat {
                    if i >= lengths.len() {
                        return Err(Error::InvalidDecode("too many code lengths".into()));
                    }
                    lengths[i] = prev;
                    i += 1;
                }
            }
            17 => {
                // Repeat zero 3-10 times
                let repeat = reader.read_bits(3)? as usize + 3;
                for _ in 0..repeat {
                    if i >= lengths.len() {
                        return Err(Error::InvalidDecode("too many code lengths".into()));
                    }
                    lengths[i] = 0;
                    i += 1;
                }
            }
            18 => {
                // Repeat zero 11-138 times
                let repeat = reader.read_bits(7)? as usize + 11;
                for _ in 0..repeat {
                    if i >= lengths.len() {
                        return Err(Error::InvalidDecode("too many code lengths".into()));
                    }
                    lengths[i] = 0;
                    i += 1;
                }
            }
            _ => return Err(Error::InvalidDecode("invalid code length code".into())),
        }
    }

    // Build literal/length and distance tables
    let lit_table = HuffmanTable::from_lengths(&lengths[..hlit])?;
    let dist_table = HuffmanTable::from_lengths(&lengths[hlit..])?;

    inflate_block(reader, output, &lit_table, &dist_table)
}

/// Inflate a block using the given Huffman tables.
fn inflate_block(
    reader: &mut BitReader,
    output: &mut Vec<u8>,
    lit_table: &HuffmanTable,
    dist_table: &HuffmanTable,
) -> Result<()> {
    loop {
        let symbol = lit_table.decode(reader)?;

        match symbol {
            0..=255 => {
                // Literal byte
                output.push(symbol as u8);
            }
            256 => {
                // End of block
                break;
            }
            257..=285 => {
                // Length/distance pair
                let len_idx = (symbol - 257) as usize;
                let length = LENGTH_BASE[len_idx] as usize
                    + reader.read_bits(LENGTH_EXTRA[len_idx])? as usize;

                let dist_symbol = dist_table.decode(reader)?;
                if dist_symbol >= 30 {
                    return Err(Error::InvalidDecode("invalid distance code".into()));
                }
                let dist_idx = dist_symbol as usize;
                let distance = DISTANCE_BASE[dist_idx] as usize
                    + reader.read_bits(DISTANCE_EXTRA[dist_idx])? as usize;

                if distance > output.len() {
                    return Err(Error::InvalidDecode("distance too far back".into()));
                }

                // Copy from output buffer (may overlap)
                let start = output.len() - distance;
                for i in 0..length {
                    let byte = output[start + (i % distance)];
                    output.push(byte);
                }
            }
            _ => {
                return Err(Error::InvalidDecode(format!(
                    "invalid literal/length code: {symbol}"
                )));
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reverse_bits() {
        assert_eq!(reverse_bits(0b101, 3), 0b101);
        assert_eq!(reverse_bits(0b100, 3), 0b001);
        assert_eq!(reverse_bits(0b1100, 4), 0b0011);
    }

    #[test]
    fn test_inflate_stored() {
        // Create a stored block manually:
        // BFINAL=1, BTYPE=00 (stored)
        // LEN = 5, NLEN = !5 = 0xFFFA
        // Data: "hello"
        let mut data = vec![0b00000001]; // BFINAL=1, BTYPE=00, aligned to byte
        data.extend_from_slice(&[5, 0]); // LEN = 5
        data.extend_from_slice(&[0xFA, 0xFF]); // NLEN = 0xFFFA
        data.extend_from_slice(b"hello");

        let output = inflate_with_size(&data, None).unwrap();
        assert_eq!(output, b"hello");
    }

    #[test]
    fn test_inflate_zlib_roundtrip() {
        // Compress with our encoder, decompress with our decoder
        use crate::compress::deflate::deflate_zlib;

        let original = b"The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.";
        let compressed = deflate_zlib(original, 6);
        let decompressed = inflate_zlib_with_size(&compressed, None).unwrap();

        assert_eq!(decompressed, original.to_vec());
    }

    #[test]
    fn test_inflate_zlib_empty() {
        use crate::compress::deflate::deflate_zlib;

        let original: &[u8] = &[];
        let compressed = deflate_zlib(original, 6);
        let decompressed = inflate_zlib_with_size(&compressed, None).unwrap();

        assert_eq!(decompressed, original.to_vec());
    }

    #[test]
    fn test_inflate_zlib_various_sizes() {
        use crate::compress::deflate::deflate_zlib;

        for size in [1, 10, 100, 1000, 10000] {
            let original: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
            let compressed = deflate_zlib(&original, 6);
            let decompressed = inflate_zlib_with_size(&compressed, None).unwrap();
            assert_eq!(decompressed, original, "failed at size {size}");
        }
    }

    #[test]
    fn test_inflate_zlib_repetitive() {
        use crate::compress::deflate::deflate_zlib;

        // Test various sizes of repetitive data
        // Note: Very large repetitive data may trigger stored blocks which
        // have different characteristics. Keep test sizes reasonable.
        for size in [100, 500, 1000] {
            let original = vec![42u8; size];
            let compressed = deflate_zlib(&original, 6);
            let decompressed = inflate_zlib_with_size(&compressed, None)
                .unwrap_or_else(|_| panic!("failed at size {size}"));
            assert_eq!(decompressed, original, "mismatch at size {size}");
        }
    }

    #[test]
    fn test_fixed_huffman_tables() {
        let lit = fixed_literal_table().unwrap();
        let dist = fixed_distance_table().unwrap();

        // Verify table sizes
        assert_eq!(lit.lengths.len(), 288);
        assert_eq!(dist.lengths.len(), 32);
    }

    #[test]
    fn test_inflate_bad_checksum() {
        use crate::compress::deflate::deflate_zlib;

        let original = b"test data";
        let mut compressed = deflate_zlib(original, 6);

        // Corrupt the checksum
        let len = compressed.len();
        compressed[len - 1] ^= 0xFF;

        assert!(inflate_zlib_with_size(&compressed, None).is_err());
    }

    #[test]
    fn test_huffman_table_from_lengths() {
        // Simple table: two symbols with lengths 1 each
        let lengths = vec![1, 1];
        let table = HuffmanTable::from_lengths(&lengths).unwrap();

        assert_eq!(table.max_len, 1);
        assert_eq!(table.lengths, lengths);
    }

    #[test]
    fn test_inflate_dynamic_block() {
        use crate::compress::deflate::deflate_zlib;

        // Large enough data to trigger dynamic Huffman
        let original: Vec<u8> = (0..5000).map(|i| (i * 17 % 256) as u8).collect();
        let compressed = deflate_zlib(&original, 9);
        let decompressed = inflate_zlib_with_size(&compressed, None).unwrap();

        assert_eq!(decompressed, original);
    }

    #[test]
    fn test_inflate_zlib_with_correct_expected_size() {
        use crate::compress::deflate::deflate_zlib;

        // Use a longer string that compresses well
        let original = b"The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.";
        let compressed = deflate_zlib(original, 6);

        // First verify the regular inflate works
        let decompressed_no_size = inflate_zlib_with_size(&compressed, None).unwrap();
        assert_eq!(decompressed_no_size.len(), original.len());

        // Now test with expected size
        let decompressed = inflate_zlib_with_size(&compressed, Some(original.len())).unwrap();

        assert_eq!(decompressed, original.to_vec());
    }

    #[test]
    fn test_inflate_zlib_with_wrong_expected_size() {
        use crate::compress::deflate::deflate_zlib;

        let original = b"hello world";
        let compressed = deflate_zlib(original, 6);

        // Wrong expected size should error
        let result = inflate_zlib_with_size(&compressed, Some(original.len() + 10));
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(
            err_msg.contains("size mismatch"),
            "Error should mention size mismatch: {err_msg}"
        );
    }

    #[test]
    fn test_inflate_zlib_without_expected_size() {
        use crate::compress::deflate::deflate_zlib;

        let original = b"test data without expected size";
        let compressed = deflate_zlib(original, 6);

        // None expected size should work
        let decompressed = inflate_zlib_with_size(&compressed, None).unwrap();
        assert_eq!(decompressed, original.to_vec());
    }

    #[test]
    fn test_inflate_empty_data() {
        let result = inflate_zlib_with_size(&[], None);
        assert!(result.is_err());
    }

    #[test]
    fn test_inflate_truncated_header() {
        // Only 1 byte of zlib header
        let data = [0x78];
        let result = inflate_zlib_with_size(&data, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_inflate_invalid_zlib_header() {
        // Invalid CMF (not deflate compression)
        let data = [0x00, 0x00, 0x00, 0x00, 0x00];
        let result = inflate_zlib_with_size(&data, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_inflate_invalid_block_type() {
        // Valid zlib header followed by invalid block type (BTYPE=11)
        let data = [
            0x78, 0x9C, // zlib header
            0x07, // BFINAL=1, BTYPE=11 (reserved)
        ];
        let result = inflate_zlib_with_size(&data, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_inflate_stored_block() {
        use crate::compress::adler32::adler32;
        use crate::compress::deflate::deflate_stored;

        // Create a stored block manually with zlib wrapper
        let original = b"stored block test data";
        let compressed = deflate_stored(original);

        // Add zlib header and adler32
        let mut zlib_data = vec![0x78, 0x01]; // zlib header (no compression)
        zlib_data.extend_from_slice(&compressed);
        let checksum = adler32(original);
        zlib_data.extend_from_slice(&checksum.to_be_bytes());

        let decompressed = inflate_zlib_with_size(&zlib_data, None).unwrap();
        assert_eq!(decompressed, original.to_vec());
    }

    #[test]
    fn test_inflate_multiple_stored_blocks() {
        use crate::compress::adler32::adler32;
        use crate::compress::deflate::deflate_stored;

        // Data larger than 65535 bytes to force multiple stored blocks
        let original = vec![42u8; 70000];
        let compressed = deflate_stored(&original);

        let mut zlib_data = vec![0x78, 0x01];
        zlib_data.extend_from_slice(&compressed);
        let checksum = adler32(&original);
        zlib_data.extend_from_slice(&checksum.to_be_bytes());

        let decompressed = inflate_zlib_with_size(&zlib_data, None).unwrap();
        assert_eq!(decompressed, original);
    }

    #[test]
    fn test_huffman_table_empty() {
        // All zero lengths - empty table
        let lengths: Vec<u8> = vec![];
        let table = HuffmanTable::from_lengths(&lengths).unwrap();
        assert_eq!(table.max_len, 0);
    }

    #[test]
    fn test_huffman_table_single_symbol() {
        // Single symbol with length 1
        let lengths = vec![1];
        let table = HuffmanTable::from_lengths(&lengths).unwrap();
        assert_eq!(table.max_len, 1);
    }

    #[test]
    fn test_huffman_table_complex() {
        // More complex Huffman table
        let lengths = vec![3, 3, 3, 3, 3, 2, 4, 4];
        let table = HuffmanTable::from_lengths(&lengths).unwrap();
        assert_eq!(table.max_len, 4);
        assert_eq!(table.lengths.len(), 8);
    }

    #[test]
    fn test_inflate_fixed_huffman_block() {
        use crate::compress::deflate::deflate_zlib;

        // Small data that should use fixed Huffman
        let original = b"abc";
        let compressed = deflate_zlib(original, 1); // Low level for fixed Huffman
        let decompressed = inflate_zlib_with_size(&compressed, None).unwrap();
        assert_eq!(decompressed, original.to_vec());
    }

    #[test]
    fn test_inflate_with_back_reference() {
        use crate::compress::deflate::deflate_zlib;

        // Repeating "abc" pattern that uses back-references.
        // This was previously failing with "unexpected end of stream" because the
        // Huffman decoder required LOOKUP_BITS (9) bits even when fewer bits were
        // sufficient for the end-of-block code (7 bits).
        let original = b"abcabcabcabcabcabcabcabcabcabc";
        let compressed = deflate_zlib(original, 6);
        let decompressed = inflate_zlib_with_size(&compressed, None).unwrap();
        assert_eq!(decompressed.as_slice(), original);
    }

    #[test]
    fn test_inflate_long_match() {
        use crate::compress::deflate::deflate_zlib;

        // Long repetitive pattern for maximum length matches
        let original = vec![b'a'; 300];
        let compressed = deflate_zlib(&original, 6);
        let decompressed = inflate_zlib_with_size(&compressed, None).unwrap();
        assert_eq!(decompressed, original);
    }

    #[test]
    fn test_inflate_max_distance_match() {
        use crate::compress::deflate::deflate_zlib;

        // Pattern that requires max distance back-reference
        let mut original = vec![b'x'; 32768 + 10];
        original[0] = b'a';
        let compressed = deflate_zlib(&original, 6);
        let decompressed = inflate_zlib_with_size(&compressed, None).unwrap();
        assert_eq!(decompressed, original);
    }
}
