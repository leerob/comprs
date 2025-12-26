//! Fuzz target for DEFLATE/zlib compression.
//!
//! Tests that DEFLATE compression handles arbitrary input without panicking.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

/// Structured input for DEFLATE fuzzing.
#[derive(Arbitrary, Debug)]
struct DeflateInput {
    /// Compression level (1-9)
    level: u8,
    /// Raw data to compress
    data: Vec<u8>,
}

fuzz_target!(|input: DeflateInput| {
    // Skip empty inputs
    if input.data.is_empty() {
        return;
    }

    // Limit input size to avoid OOM
    if input.data.len() > 1024 * 1024 {
        return;
    }

    // Clamp compression level
    let level = (input.level % 9).max(1);

    // Compress using our implementation
    let compressed = pixo::compress::deflate::deflate_zlib(&input.data, level);

    // Verify zlib header
    assert!(compressed.len() >= 6, "Compressed data too short");

    // Check zlib header (CMF, FLG)
    let cmf = compressed[0];
    let flg = compressed[1];

    // CMF should indicate DEFLATE (method 8) with window size
    assert_eq!(cmf & 0x0F, 8, "Invalid compression method");

    // Header checksum
    assert_eq!((cmf as u16 * 256 + flg as u16) % 31, 0, "Invalid header checksum");
});
