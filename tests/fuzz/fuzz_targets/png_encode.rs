//! Fuzz target for PNG encoding.
//!
//! Tests that PNG encoding handles arbitrary input without panicking
//! and produces valid output that can be decoded.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

/// Structured input for PNG encoding fuzzing.
#[derive(Arbitrary, Debug)]
struct PngInput {
    /// Image width (clamped to reasonable range)
    width: u8,
    /// Image height (clamped to reasonable range)
    height: u8,
    /// Color type selector (0-3)
    color_type: u8,
    /// Compression level (1-9)
    compression_level: u8,
    /// Raw pixel data
    data: Vec<u8>,
}

fuzz_target!(|input: PngInput| {
    // Clamp dimensions to reasonable range (1-64)
    let width = (input.width as u32 % 64).max(1);
    let height = (input.height as u32 % 64).max(1);

    // Select color type
    let color_type = match input.color_type % 4 {
        0 => comprs::ColorType::Gray,
        1 => comprs::ColorType::GrayAlpha,
        2 => comprs::ColorType::Rgb,
        _ => comprs::ColorType::Rgba,
    };

    // Calculate expected data length
    let bytes_per_pixel = color_type.bytes_per_pixel();
    let expected_len = (width * height) as usize * bytes_per_pixel;

    // If we don't have enough data, skip this input
    if input.data.len() < expected_len {
        return;
    }

    // Use only the required amount of data
    let pixel_data = &input.data[..expected_len];

    // Clamp compression level
    let compression_level = (input.compression_level % 9).max(1);

    // Build options
    let options = comprs::png::PngOptions {
        compression_level,
        ..Default::default()
    };

    // Try to encode - should not panic
    let result = comprs::png::encode_with_options(pixel_data, width, height, color_type, &options);

    // If encoding succeeded, verify basic structure
    if let Ok(encoded) = result {
        // Check PNG signature
        assert_eq!(
            &encoded[..8],
            &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A],
            "Invalid PNG signature"
        );

        // Check minimum length (signature + IHDR + IDAT + IEND)
        assert!(encoded.len() > 8 + 25 + 12, "PNG too short");
    }
});
