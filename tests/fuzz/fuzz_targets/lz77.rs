//! Fuzz target for LZ77 compression.
//!
//! Tests that LZ77 compression handles arbitrary input without panicking
//! and produces valid tokens.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

/// Structured input for LZ77 fuzzing.
#[derive(Arbitrary, Debug)]
struct Lz77Input {
    /// Compression level (1-9)
    level: u8,
    /// Raw data to compress
    data: Vec<u8>,
}

fuzz_target!(|input: Lz77Input| {
    // Skip empty inputs
    if input.data.is_empty() {
        return;
    }

    // Limit input size to avoid OOM
    if input.data.len() > 256 * 1024 {
        return;
    }

    // Clamp compression level
    let level = (input.level % 9).max(1);

    // Create compressor
    let mut compressor = pixo::compress::lz77::Lz77Compressor::new(level);

    // Compress using token output
    let tokens = compressor.compress(&input.data);

    // Verify tokens can reconstruct original data
    let mut reconstructed = Vec::with_capacity(input.data.len());

    for token in &tokens {
        match token {
            pixo::compress::lz77::Token::Literal(byte) => {
                reconstructed.push(*byte);
            }
            pixo::compress::lz77::Token::Match { length, distance } => {
                let start = reconstructed.len() - *distance as usize;
                for i in 0..*length as usize {
                    let byte = reconstructed[start + i];
                    reconstructed.push(byte);
                }
            }
        }
    }

    // Verify reconstruction matches original
    assert_eq!(
        reconstructed, input.data,
        "LZ77 tokens do not reconstruct original data"
    );
});
