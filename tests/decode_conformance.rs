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

#[test]
fn png_fixture_rocket_decodes_pixels() {
    let fixture_path = Path::new("tests/fixtures/rocket.png");
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

#[test]
fn png_fixture_avatar_decodes_pixels() {
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

#[test]
fn png_fixture_playground_reports_dimensions() {
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

#[test]
fn png_roundtrip_rgb_2x2_exact_pixels() {
    let pixels = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0];
    let encoded = encode_png(&pixels, 2, 2, ColorType::Rgb).expect("encode");
    let decoded = decode_png(&encoded).expect("decode");

    assert_eq!(decoded.width, 2);
    assert_eq!(decoded.height, 2);
    assert_eq!(decoded.color_type, ColorType::Rgb);
    assert_eq!(decoded.pixels, pixels);
}

#[test]
fn png_roundtrip_rgba_2x2_opacity() {
    let pixels = vec![
        255, 0, 0, 255, 0, 255, 0, 128, 0, 0, 255, 0, 255, 255, 0, 255,
    ];
    let encoded = encode_png(&pixels, 2, 2, ColorType::Rgba).expect("encode");
    let decoded = decode_png(&encoded).expect("decode");

    assert_eq!(decoded.width, 2);
    assert_eq!(decoded.height, 2);
    assert_eq!(decoded.color_type, ColorType::Rgba);
    assert_eq!(decoded.pixels, pixels);
}

#[test]
fn png_roundtrip_gray_2x2_gradient() {
    let pixels = vec![0, 64, 128, 255];
    let encoded = encode_png(&pixels, 2, 2, ColorType::Gray).expect("encode");
    let decoded = decode_png(&encoded).expect("decode");

    assert_eq!(decoded.width, 2);
    assert_eq!(decoded.height, 2);
    assert_eq!(decoded.color_type, ColorType::Gray);
    assert_eq!(decoded.pixels, pixels);
}

#[test]
fn png_roundtrip_gray_alpha_2x2_masked() {
    let pixels = vec![0, 255, 128, 128, 255, 0, 64, 192];
    let encoded = encode_png(&pixels, 2, 2, ColorType::GrayAlpha).expect("encode");
    let decoded = decode_png(&encoded).expect("decode");

    assert_eq!(decoded.width, 2);
    assert_eq!(decoded.height, 2);
    assert_eq!(decoded.color_type, ColorType::GrayAlpha);
    assert_eq!(decoded.pixels, pixels);
}

#[test]
fn png_roundtrip_rgb_repeated_pattern_sizes() {
    let pixels_2x2_rgb = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0];
    let encoded = encode_png(&pixels_2x2_rgb, 2, 2, ColorType::Rgb).expect("encode should succeed");
    let decoded = decode_png(&encoded).unwrap_or_else(|e| panic!("decode failed for 2x2 RGB: {e}"));
    assert_eq!(decoded.width, 2);
    assert_eq!(decoded.height, 2);
    assert_eq!(decoded.color_type, ColorType::Rgb);
    assert_eq!(decoded.pixels, pixels_2x2_rgb);

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

#[test]
fn png_roundtrip_rgb_random_100x80() {
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

#[test]
fn png_rejects_invalid_signature() {
    let data = b"not a PNG file";
    let result = decode_png(data);
    assert!(result.is_err());
}

#[test]
fn png_rejects_signature_without_chunks() {
    let data = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    let result = decode_png(&data);
    assert!(result.is_err());
}

#[test]
fn png_rejects_empty_input() {
    let result = decode_png(&[]);
    assert!(result.is_err());
}

#[test]
fn jpeg_fixture_browser_decodes_pixels() {
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

#[test]
fn jpeg_fixture_review_has_dimensions() {
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

#[test]
fn jpeg_fixture_web_has_dimensions() {
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

#[test]
fn jpeg_roundtrip_rgb_preserves_dimensions() {
    let mut rng = StdRng::seed_from_u64(55);
    let (w, h) = (16, 16);
    let mut pixels = vec![0u8; w * h * 3];
    rng.fill(pixels.as_mut_slice());

    let encoded = encode_jpeg(&pixels, w as u32, h as u32, 90).expect("encode");
    let decoded = decode_jpeg(&encoded).expect("decode");

    assert_eq!(decoded.width, w as u32);
    assert_eq!(decoded.height, h as u32);
    assert_eq!(decoded.color_type, ColorType::Rgb);
    assert_eq!(decoded.pixels.len(), w * h * 3);
}

#[test]
fn jpeg_roundtrip_grayscale_preserves_dimensions() {
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

#[test]
fn jpeg_grayscale_non_mcu_aligned_respects_dimensions() {
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

#[test]
fn jpeg_roundtrip_rgb_various_sizes() {
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

#[test]
fn jpeg_roundtrip_rgb_subsampling_420() {
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

#[test]
fn jpeg_rejects_invalid_signature() {
    let data = b"not a JPEG file";
    let result = decode_jpeg(data);
    assert!(result.is_err());
}

#[test]
fn jpeg_rejects_soi_only() {
    let data = [0xFF, 0xD8];
    let result = decode_jpeg(&data);
    assert!(result.is_err());
}

#[test]
fn jpeg_rejects_empty_input() {
    let result = decode_jpeg(&[]);
    assert!(result.is_err());
}

#[test]
fn jpeg_roundtrip_solid_color_is_stable() {
    let (w, h) = (16, 16);
    let pixels = vec![128u8; w * h * 3];

    let encoded = encode_jpeg(&pixels, w as u32, h as u32, 95).expect("encode");
    let decoded = decode_jpeg(&encoded).expect("decode");

    assert_eq!(decoded.width, w as u32);
    assert_eq!(decoded.height, h as u32);

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

#[test]
fn jpeg_roundtrip_gradient_retains_dimensions() {
    let (w, h) = (64, 64);
    let mut pixels = Vec::with_capacity(w * h * 3);

    for y in 0..h {
        for x in 0..w {
            pixels.push((x * 4) as u8);
            pixels.push((y * 4) as u8);
            pixels.push(128);
        }
    }

    let encoded = encode_jpeg(&pixels, w as u32, h as u32, 90).expect("encode");
    let decoded = decode_jpeg(&encoded).expect("decode");

    assert_eq!(decoded.width, w as u32);
    assert_eq!(decoded.height, h as u32);
}

#[test]
fn png_decoder_rejects_jpeg_data() {
    let jpeg_data = [0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10];
    let result = decode_png(&jpeg_data);
    assert!(result.is_err());
}

#[test]
fn jpeg_decoder_rejects_png_data() {
    let png_data = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    let result = decode_jpeg(&png_data);
    assert!(result.is_err());
}
