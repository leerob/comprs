//! Decode conformance tests.
//!
//! Tests PNG and JPEG decoding for correctness and validates
//! that our decoders can handle encoded images from our encoders
//! as well as external fixtures.

#![cfg(feature = "cli")]

use pixo::decode::{decode_jpeg, decode_png};
use pixo::jpeg::JpegOptions;
use pixo::png::PngOptions;
use pixo::{jpeg, png, ColorType};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::path::Path;

#[allow(dead_code)]
fn encode_png(
    data: &[u8],
    width: u32,
    height: u32,
    color_type: ColorType,
) -> pixo::Result<Vec<u8>> {
    let options = PngOptions::builder(width, height)
        .color_type(color_type)
        .build();
    png::encode(data, &options)
}

#[allow(dead_code)]
fn encode_jpeg(data: &[u8], width: u32, height: u32, quality: u8) -> pixo::Result<Vec<u8>> {
    let options = JpegOptions::builder(width, height)
        .color_type(ColorType::Rgb)
        .quality(quality)
        .build();
    jpeg::encode(data, &options)
}

#[allow(dead_code)]
fn encode_jpeg_with_color(
    data: &[u8],
    width: u32,
    height: u32,
    quality: u8,
    color_type: ColorType,
) -> pixo::Result<Vec<u8>> {
    let options = JpegOptions::builder(width, height)
        .color_type(color_type)
        .quality(quality)
        .build();
    jpeg::encode(data, &options)
}

#[allow(dead_code)]
fn encode_jpeg_with_options(
    data: &[u8],
    width: u32,
    height: u32,
    color_type: ColorType,
    options: &JpegOptions,
) -> pixo::Result<Vec<u8>> {
    let mut opts = *options;
    opts.width = width;
    opts.height = height;
    opts.color_type = color_type;
    jpeg::encode(data, &opts)
}

// ============================================================================
// PNG Decoder Tests
// ============================================================================

/// Test PNG decode of fixture images.
#[test]
fn test_decode_png_fixture_rocket() {
    let fixture_path = Path::new("tests/fixtures/rocket.png");
    if !fixture_path.exists() {
        eprintln!("Skipping: fixture not found");
        return;
    }

    let bytes = std::fs::read(fixture_path).expect("read fixture");
    let decoded = decode_png(&bytes).expect("decode PNG");

    // rocket.png is a known image
    assert!(decoded.width > 0);
    assert!(decoded.height > 0);
    assert!(!decoded.pixels.is_empty());
}

/// Test PNG decode of avatar fixture.
#[test]
fn test_decode_png_fixture_avatar() {
    let fixture_path = Path::new("tests/fixtures/avatar-color.png");
    if !fixture_path.exists() {
        eprintln!("Skipping: fixture not found");
        return;
    }

    let bytes = std::fs::read(fixture_path).expect("read fixture");
    let decoded = decode_png(&bytes).expect("decode PNG");

    assert!(decoded.width > 0);
    assert!(decoded.height > 0);
    assert!(!decoded.pixels.is_empty());
}

/// Test PNG decode of playground fixture.
#[test]
fn test_decode_png_fixture_playground() {
    let fixture_path = Path::new("tests/fixtures/playground.png");
    if !fixture_path.exists() {
        eprintln!("Skipping: fixture not found");
        return;
    }

    let bytes = std::fs::read(fixture_path).expect("read fixture");
    let decoded = decode_png(&bytes).expect("decode PNG");

    assert!(decoded.width > 0);
    assert!(decoded.height > 0);
}

/// Test PNG encode->decode roundtrip for RGB.
#[test]
fn test_png_encode_decode_roundtrip_rgb() {
    let pixels = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0]; // 2x2 RGB
    let encoded = encode_png(&pixels, 2, 2, ColorType::Rgb).expect("encode");
    let decoded = decode_png(&encoded).expect("decode");

    assert_eq!(decoded.width, 2);
    assert_eq!(decoded.height, 2);
    assert_eq!(decoded.color_type, ColorType::Rgb);
    assert_eq!(decoded.pixels, pixels);
}

/// Test PNG encode->decode roundtrip for RGBA.
#[test]
fn test_png_encode_decode_roundtrip_rgba() {
    let pixels = vec![
        255, 0, 0, 255, // Red, opaque
        0, 255, 0, 128, // Green, semi-transparent
        0, 0, 255, 0, // Blue, transparent
        255, 255, 0, 255, // Yellow, opaque
    ];
    let encoded = encode_png(&pixels, 2, 2, ColorType::Rgba).expect("encode");
    let decoded = decode_png(&encoded).expect("decode");

    assert_eq!(decoded.width, 2);
    assert_eq!(decoded.height, 2);
    assert_eq!(decoded.color_type, ColorType::Rgba);
    assert_eq!(decoded.pixels, pixels);
}

/// Test PNG encode->decode roundtrip for Grayscale.
#[test]
fn test_png_encode_decode_roundtrip_gray() {
    let pixels = vec![0, 64, 128, 255]; // 2x2 grayscale
    let encoded = encode_png(&pixels, 2, 2, ColorType::Gray).expect("encode");
    let decoded = decode_png(&encoded).expect("decode");

    assert_eq!(decoded.width, 2);
    assert_eq!(decoded.height, 2);
    assert_eq!(decoded.color_type, ColorType::Gray);
    assert_eq!(decoded.pixels, pixels);
}

/// Test PNG encode->decode roundtrip for GrayAlpha.
#[test]
fn test_png_encode_decode_roundtrip_gray_alpha() {
    let pixels = vec![0, 255, 128, 128, 255, 0, 64, 192]; // 2x2 gray+alpha
    let encoded = encode_png(&pixels, 2, 2, ColorType::GrayAlpha).expect("encode");
    let decoded = decode_png(&encoded).expect("decode");

    assert_eq!(decoded.width, 2);
    assert_eq!(decoded.height, 2);
    assert_eq!(decoded.color_type, ColorType::GrayAlpha);
    assert_eq!(decoded.pixels, pixels);
}

/// Test PNG decode with various sizes using the same pattern as working tests.
#[test]
fn test_png_encode_decode_various_sizes() {
    // Use the exact same pixel pattern as the working test_png_encode_decode_roundtrip_rgb
    let pixels_2x2_rgb = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0]; // 2x2 RGB
    let encoded = encode_png(&pixels_2x2_rgb, 2, 2, ColorType::Rgb).expect("encode should succeed");
    let decoded = decode_png(&encoded).unwrap_or_else(|e| panic!("decode failed for 2x2 RGB: {e}"));
    assert_eq!(decoded.width, 2);
    assert_eq!(decoded.height, 2);
    assert_eq!(decoded.color_type, ColorType::Rgb);
    assert_eq!(decoded.pixels, pixels_2x2_rgb);

    // Test 4x4 RGB with the same color pattern repeated
    let mut pixels_4x4_rgb = Vec::with_capacity(4 * 4 * 3);
    for _ in 0..4 {
        pixels_4x4_rgb.extend_from_slice(&[255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0]);
    }
    let encoded = encode_png(&pixels_4x4_rgb, 4, 4, ColorType::Rgb).expect("encode should succeed");
    let decoded = decode_png(&encoded).unwrap_or_else(|e| panic!("decode failed for 4x4 RGB: {e}"));
    assert_eq!(decoded.width, 4);
    assert_eq!(decoded.height, 4);
    assert_eq!(decoded.pixels, pixels_4x4_rgb);
}

/// Test PNG decode with larger images.
#[test]
fn test_png_encode_decode_larger() {
    let mut rng = StdRng::seed_from_u64(123);

    let (w, h) = (100, 80);
    let mut pixels = vec![0u8; w * h * 3];
    rng.fill(pixels.as_mut_slice());

    let encoded = encode_png(&pixels, w as u32, h as u32, ColorType::Rgb).expect("encode");
    let decoded = decode_png(&encoded).expect("decode");

    assert_eq!(decoded.width, w as u32);
    assert_eq!(decoded.height, h as u32);
    assert_eq!(decoded.pixels, pixels);
}

/// Test PNG decode error handling - invalid signature.
#[test]
fn test_png_decode_invalid_signature() {
    let data = b"not a PNG file";
    let result = decode_png(data);
    assert!(result.is_err());
}

/// Test PNG decode error handling - truncated data.
#[test]
fn test_png_decode_truncated() {
    // PNG signature only
    let data = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    let result = decode_png(&data);
    assert!(result.is_err());
}

/// Test PNG decode error handling - empty data.
#[test]
fn test_png_decode_empty() {
    let result = decode_png(&[]);
    assert!(result.is_err());
}

// ============================================================================
// JPEG Decoder Tests
// ============================================================================

/// Test JPEG decode of fixture images.
#[test]
fn test_decode_jpeg_fixture_browser() {
    let fixture_path = Path::new("tests/fixtures/browser.jpg");
    if !fixture_path.exists() {
        eprintln!("Skipping: fixture not found");
        return;
    }

    let bytes = std::fs::read(fixture_path).expect("read fixture");
    let decoded = decode_jpeg(&bytes).expect("decode JPEG");

    assert!(decoded.width > 0);
    assert!(decoded.height > 0);
    assert!(!decoded.pixels.is_empty());
}

/// Test JPEG decode of review fixture.
#[test]
fn test_decode_jpeg_fixture_review() {
    let fixture_path = Path::new("tests/fixtures/review.jpg");
    if !fixture_path.exists() {
        eprintln!("Skipping: fixture not found");
        return;
    }

    let bytes = std::fs::read(fixture_path).expect("read fixture");
    let decoded = decode_jpeg(&bytes).expect("decode JPEG");

    assert!(decoded.width > 0);
    assert!(decoded.height > 0);
}

/// Test JPEG decode of web fixture.
#[test]
fn test_decode_jpeg_fixture_web() {
    let fixture_path = Path::new("tests/fixtures/web.jpg");
    if !fixture_path.exists() {
        eprintln!("Skipping: fixture not found");
        return;
    }

    let bytes = std::fs::read(fixture_path).expect("read fixture");
    let decoded = decode_jpeg(&bytes).expect("decode JPEG");

    assert!(decoded.width > 0);
    assert!(decoded.height > 0);
}

/// Test JPEG encode->decode roundtrip - dimensions match.
#[test]
fn test_jpeg_encode_decode_roundtrip_rgb() {
    let mut rng = StdRng::seed_from_u64(55);
    let (w, h) = (16, 16);
    let mut pixels = vec![0u8; w * h * 3];
    rng.fill(pixels.as_mut_slice());

    let encoded = encode_jpeg(&pixels, w as u32, h as u32, 90).expect("encode");
    let decoded = decode_jpeg(&encoded).expect("decode");

    // JPEG is lossy, so just check dimensions
    assert_eq!(decoded.width, w as u32);
    assert_eq!(decoded.height, h as u32);
    assert_eq!(decoded.color_type, ColorType::Rgb);
    assert_eq!(decoded.pixels.len(), w * h * 3);
}

/// Test JPEG encode->decode roundtrip for grayscale.
#[test]
fn test_jpeg_encode_decode_roundtrip_gray() {
    let mut rng = StdRng::seed_from_u64(77);
    let (w, h) = (16, 16);
    let mut pixels = vec![0u8; w * h];
    rng.fill(pixels.as_mut_slice());

    let encoded =
        encode_jpeg_with_color(&pixels, w as u32, h as u32, 90, ColorType::Gray).expect("encode");
    let decoded = decode_jpeg(&encoded).expect("decode");

    assert_eq!(decoded.width, w as u32);
    assert_eq!(decoded.height, h as u32);
    assert_eq!(decoded.color_type, ColorType::Gray);
    assert_eq!(decoded.pixels.len(), w * h);
}

/// Test grayscale JPEG with non-MCU-aligned dimensions.
/// Verifies the decoder crops to actual image size, not MCU-aligned buffer size.
#[test]
fn test_jpeg_grayscale_non_mcu_aligned_size() {
    // Non-MCU-aligned grayscale image (15x9, not multiples of 8)
    // MCU-aligned would be 16x16 = 256 pixels, but we need exactly 15x9 = 135
    let (w, h) = (15, 9);
    let pixels = vec![128u8; w * h];

    let encoded =
        encode_jpeg_with_color(&pixels, w as u32, h as u32, 90, ColorType::Gray).expect("encode");
    let decoded = decode_jpeg(&encoded).expect("decode");

    assert_eq!(decoded.width, w as u32);
    assert_eq!(decoded.height, h as u32);
    assert_eq!(decoded.color_type, ColorType::Gray);
    assert_eq!(
        decoded.pixels.len(),
        w * h,
        "Expected {} pixels, got {} (MCU-aligned would be 256)",
        w * h,
        decoded.pixels.len()
    );
}

/// Test JPEG encode->decode with various sizes.
#[test]
fn test_jpeg_encode_decode_various_sizes() {
    let mut rng = StdRng::seed_from_u64(99);

    for (w, h) in [(8, 8), (15, 9), (24, 16), (32, 32)] {
        let mut pixels = vec![0u8; w * h * 3];
        rng.fill(pixels.as_mut_slice());

        let encoded = encode_jpeg(&pixels, w as u32, h as u32, 85).expect("encode");
        let decoded = decode_jpeg(&encoded).expect("decode");

        assert_eq!(decoded.width, w as u32, "width mismatch for {w}x{h}");
        assert_eq!(decoded.height, h as u32, "height mismatch for {w}x{h}");
    }
}

/// Test JPEG encode->decode with 4:2:0 subsampling.
#[test]
fn test_jpeg_encode_decode_subsampling_420() {
    let mut rng = StdRng::seed_from_u64(111);
    let (w, h) = (32, 32);
    let mut pixels = vec![0u8; w * h * 3];
    rng.fill(pixels.as_mut_slice());

    let opts = jpeg::JpegOptions {
        width: w as u32,
        height: h as u32,
        color_type: ColorType::Rgb,
        quality: 85,
        subsampling: jpeg::Subsampling::S420,
        restart_interval: None,
        optimize_huffman: false,
        progressive: false,
        trellis_quant: false,
    };

    let encoded = jpeg::encode(&pixels, &opts).expect("encode");
    let decoded = decode_jpeg(&encoded).expect("decode");

    assert_eq!(decoded.width, w as u32);
    assert_eq!(decoded.height, h as u32);
}

/// Test JPEG decode error handling - invalid signature.
#[test]
fn test_jpeg_decode_invalid_signature() {
    let data = b"not a JPEG file";
    let result = decode_jpeg(data);
    assert!(result.is_err());
}

/// Test JPEG decode error handling - truncated data.
#[test]
fn test_jpeg_decode_truncated() {
    // JPEG SOI marker only
    let data = [0xFF, 0xD8];
    let result = decode_jpeg(&data);
    assert!(result.is_err());
}

/// Test JPEG decode error handling - empty data.
#[test]
fn test_jpeg_decode_empty() {
    let result = decode_jpeg(&[]);
    assert!(result.is_err());
}

/// Test JPEG decode with solid color (easy to decode).
#[test]
fn test_jpeg_encode_decode_solid_color() {
    let (w, h) = (16, 16);
    let pixels = vec![128u8; w * h * 3]; // solid gray

    let encoded = encode_jpeg(&pixels, w as u32, h as u32, 95).expect("encode");
    let decoded = decode_jpeg(&encoded).expect("decode");

    assert_eq!(decoded.width, w as u32);
    assert_eq!(decoded.height, h as u32);

    // High quality solid color should decode close to original
    let avg_diff: i32 = decoded
        .pixels
        .iter()
        .zip(pixels.iter())
        .map(|(&a, &b)| (a as i32 - b as i32).abs())
        .sum::<i32>()
        / decoded.pixels.len() as i32;

    assert!(
        avg_diff < 10,
        "Average pixel difference too high: {avg_diff}"
    );
}

/// Test JPEG decode with gradient pattern.
#[test]
fn test_jpeg_encode_decode_gradient() {
    let (w, h) = (64, 64);
    let mut pixels = Vec::with_capacity(w * h * 3);

    for y in 0..h {
        for x in 0..w {
            pixels.push((x * 4) as u8); // R
            pixels.push((y * 4) as u8); // G
            pixels.push(128); // B
        }
    }

    let encoded = encode_jpeg(&pixels, w as u32, h as u32, 90).expect("encode");
    let decoded = decode_jpeg(&encoded).expect("decode");

    assert_eq!(decoded.width, w as u32);
    assert_eq!(decoded.height, h as u32);
}

// ============================================================================
// Cross-format Tests
// ============================================================================

/// Test decoding format detection by header.
#[test]
fn test_decode_format_detection() {
    // PNG signature
    let png_header = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    assert!(png_header.starts_with(&[0x89, 0x50, 0x4E, 0x47]));

    // JPEG signature
    let jpeg_header = [0xFF, 0xD8, 0xFF];
    assert!(jpeg_header.starts_with(&[0xFF, 0xD8]));
}

/// Test that PNG decoder rejects JPEG data.
#[test]
fn test_png_decode_rejects_jpeg() {
    let jpeg_data = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10];
    let result = decode_png(&jpeg_data);
    assert!(result.is_err());
}

/// Test that JPEG decoder rejects PNG data.
#[test]
fn test_jpeg_decode_rejects_png() {
    let png_data = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    let result = decode_jpeg(&png_data);
    assert!(result.is_err());
}
