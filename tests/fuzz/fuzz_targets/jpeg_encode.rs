//! Fuzz target for JPEG encoding.
//!
//! Tests that JPEG encoding handles arbitrary input without panicking
//! and produces valid output.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

/// Structured input for JPEG encoding fuzzing.
#[derive(Arbitrary, Debug)]
struct JpegInput {
    /// Image width (clamped to reasonable range)
    width: u8,
    /// Image height (clamped to reasonable range)
    height: u8,
    /// Quality level (1-100)
    quality: u8,
    /// Use grayscale
    grayscale: bool,
    /// Enable progressive encoding
    progressive: bool,
    /// Raw pixel data
    data: Vec<u8>,
}

fuzz_target!(|input: JpegInput| {
    // Clamp dimensions to reasonable range (1-64)
    let width = (input.width as u32 % 64).max(1);
    let height = (input.height as u32 % 64).max(1);

    // Select color type
    let color_type = if input.grayscale {
        pixo::ColorType::Gray
    } else {
        pixo::ColorType::Rgb
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

    // Clamp quality to valid range
    let quality = (input.quality % 100).max(1);

    // Build options
    let options = pixo::jpeg::JpegOptions {
        quality,
        progressive: input.progressive,
        ..Default::default()
    };

    // Try to encode - should not panic
    let result =
        pixo::jpeg::encode_with_options(pixel_data, width, height, color_type, &options);

    // If encoding succeeded, verify basic structure
    if let Ok(encoded) = result {
        // Check JPEG SOI marker
        assert_eq!(&encoded[..2], &[0xFF, 0xD8], "Invalid JPEG SOI marker");

        // Check JPEG EOI marker at end
        assert_eq!(
            &encoded[encoded.len() - 2..],
            &[0xFF, 0xD9],
            "Invalid JPEG EOI marker"
        );
    }
});
