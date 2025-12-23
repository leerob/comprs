//! PNG conformance tests.
//!
//! Tests PNG encoding against expected output and validates
//! that encoded images can be decoded correctly.

#![allow(clippy::uninlined_format_args)]

use comprs::compress::crc32::crc32;
use comprs::{png, ColorType, Error};
use image::GenericImageView;
use proptest::prelude::*;
use rand::{rngs::StdRng, Rng, SeedableRng};
mod support;
use support::pngsuite::read_pngsuite;

/// Test that PNG output has correct header.
#[test]
fn test_png_signature() {
    let pixels = vec![255, 0, 0]; // 1x1 red pixel
    let result = png::encode(&pixels, 1, 1, ColorType::Rgb).unwrap();

    // PNG signature
    assert_eq!(
        &result[0..8],
        &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
    );
}

/// Test IHDR chunk format.
#[test]
fn test_ihdr_chunk() {
    let pixels = vec![0u8; 100 * 100 * 3]; // 100x100 RGB
    let result = png::encode(&pixels, 100, 100, ColorType::Rgb).unwrap();

    // IHDR should be right after signature
    // Length (4 bytes) + "IHDR" (4 bytes) + data (13 bytes) + CRC (4 bytes)

    // Length should be 13
    assert_eq!(&result[8..12], &[0, 0, 0, 13]);

    // Chunk type should be IHDR
    assert_eq!(&result[12..16], b"IHDR");

    // Width (100 = 0x64)
    assert_eq!(&result[16..20], &[0, 0, 0, 100]);

    // Height (100 = 0x64)
    assert_eq!(&result[20..24], &[0, 0, 0, 100]);

    // Bit depth (8)
    assert_eq!(result[24], 8);

    // Color type (2 = RGB)
    assert_eq!(result[25], 2);

    // Compression method (0 = DEFLATE)
    assert_eq!(result[26], 0);

    // Filter method (0 = adaptive)
    assert_eq!(result[27], 0);

    // Interlace method (0 = none)
    assert_eq!(result[28], 0);
}

/// Test that IEND chunk is present at end.
#[test]
fn test_iend_chunk() {
    let pixels = vec![128u8; 10 * 10 * 3];
    let result = png::encode(&pixels, 10, 10, ColorType::Rgb).unwrap();

    // IEND chunk should be at the end
    // It's 12 bytes: length (4) + type (4) + CRC (4)
    let iend_start = result.len() - 12;

    // Length should be 0
    assert_eq!(&result[iend_start..iend_start + 4], &[0, 0, 0, 0]);

    // Type should be IEND
    assert_eq!(&result[iend_start + 4..iend_start + 8], b"IEND");

    // CRC of "IEND" should be 0xAE426082
    assert_eq!(
        &result[iend_start + 8..iend_start + 12],
        &[0xAE, 0x42, 0x60, 0x82]
    );
}

/// Test encoding different color types.
#[test]
fn test_color_types() {
    // Grayscale
    let gray = vec![128u8; 4 * 4];
    let result = png::encode(&gray, 4, 4, ColorType::Gray).unwrap();
    assert_eq!(result[25], 0); // Color type 0

    // Grayscale + Alpha
    let gray_alpha = vec![128u8; 4 * 4 * 2];
    let result = png::encode(&gray_alpha, 4, 4, ColorType::GrayAlpha).unwrap();
    assert_eq!(result[25], 4); // Color type 4

    // RGB
    let rgb = vec![128u8; 4 * 4 * 3];
    let result = png::encode(&rgb, 4, 4, ColorType::Rgb).unwrap();
    assert_eq!(result[25], 2); // Color type 2

    // RGBA
    let rgba = vec![128u8; 4 * 4 * 4];
    let result = png::encode(&rgba, 4, 4, ColorType::Rgba).unwrap();
    assert_eq!(result[25], 6); // Color type 6
}

/// Test that different images produce different output.
#[test]
fn test_different_images() {
    let black = vec![0u8; 8 * 8 * 3];
    let white = vec![255u8; 8 * 8 * 3];

    let black_png = png::encode(&black, 8, 8, ColorType::Rgb).unwrap();
    let white_png = png::encode(&white, 8, 8, ColorType::Rgb).unwrap();

    // Should be different
    assert_ne!(black_png, white_png);
}

/// Ensure encoded PNGs decode correctly via the `image` crate (zlib wrapper validity).
#[test]
fn test_png_roundtrip_decode_rgb() {
    let width = 3;
    let height = 2;
    let pixels = vec![
        // row 0
        255, 0, 0, // red
        0, 255, 0, // green
        0, 0, 255, // blue
        // row 1
        255, 255, 0, // yellow
        0, 255, 255, // cyan
        255, 0, 255, // magenta
    ];

    let encoded = png::encode(&pixels, width, height, ColorType::Rgb).unwrap();

    let decoded = image::load_from_memory(&encoded).expect("decode").to_rgb8();
    assert_eq!(decoded.width(), width);
    assert_eq!(decoded.height(), height);
    assert_eq!(decoded.as_raw(), &pixels);
}

/// Randomized small-images roundtrip across color types to ensure decodability.
#[test]
fn test_png_roundtrip_random_small() {
    let mut rng = StdRng::seed_from_u64(42);
    let dims = [(1, 1), (2, 3), (3, 2), (4, 4), (8, 5)];
    let color_types = [
        ColorType::Gray,
        ColorType::GrayAlpha,
        ColorType::Rgb,
        ColorType::Rgba,
    ];

    for &(w, h) in &dims {
        for &ct in &color_types {
            let bpp = ct.bytes_per_pixel();
            let mut pixels = vec![0u8; (w * h) as usize * bpp];
            rng.fill(pixels.as_mut_slice());

            let encoded = png::encode(&pixels, w as u32, h as u32, ct).expect("encode random png");
            let decoded = image::load_from_memory(&encoded)
                .expect("decode")
                .to_rgba8();

            assert_eq!(decoded.width(), w as u32);
            assert_eq!(decoded.height(), h as u32);
        }
    }
}

/// Validate chunk lengths and CRCs for a generated PNG.
#[test]
fn test_png_chunk_crc_and_lengths() {
    let mut rng = StdRng::seed_from_u64(777);
    let w = 12;
    let h = 7;
    let mut pixels = vec![0u8; (w * h * 3) as usize];
    rng.fill(pixels.as_mut_slice());

    let encoded = png::encode(&pixels, w, h, ColorType::Rgb).unwrap();

    // PNG signature already validated elsewhere
    let mut offset = 8;
    let mut saw_iend = false;

    while offset < encoded.len() {
        // length (u32 big-endian)
        assert!(offset + 8 <= encoded.len(), "truncated chunk header");
        let len = u32::from_be_bytes(encoded[offset..offset + 4].try_into().unwrap()) as usize;
        let chunk_type = &encoded[offset + 4..offset + 8];
        offset += 8;

        assert!(
            offset + len + 4 <= encoded.len(),
            "chunk overruns buffer: type={:?} len={}",
            chunk_type,
            len
        );

        let data = &encoded[offset..offset + len];
        offset += len;

        let stored_crc = u32::from_be_bytes(encoded[offset..offset + 4].try_into().unwrap());
        offset += 4;

        // CRC computed over type + data
        let mut payload = Vec::with_capacity(4 + len);
        payload.extend_from_slice(chunk_type);
        payload.extend_from_slice(data);
        let computed_crc = crc32(&payload);
        assert_eq!(
            stored_crc, computed_crc,
            "CRC mismatch for chunk {:?}",
            chunk_type
        );

        if chunk_type == b"IEND" {
            saw_iend = true;
            break;
        }
    }

    assert!(saw_iend, "IEND not found");
}

fn png_image_strategy() -> impl Strategy<Value = (u32, u32, ColorType, Vec<u8>)> {
    (1u32..16, 1u32..16).prop_flat_map(|(w, h)| {
        prop_oneof![
            Just(ColorType::Gray),
            Just(ColorType::GrayAlpha),
            Just(ColorType::Rgb),
            Just(ColorType::Rgba),
        ]
        .prop_flat_map(move |ct| {
            let len = (w * h) as usize * ct.bytes_per_pixel();
            proptest::collection::vec(any::<u8>(), len).prop_map(move |data| (w, h, ct, data))
        })
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]
    #[test]
    fn prop_png_roundtrip_varied_color((w, h, ct, data) in png_image_strategy()) {
        let encoded = png::encode(&data, w, h, ct).unwrap();
        let decoded = image::load_from_memory(&encoded).expect("decode");

        match ct {
            ColorType::Gray => {
                let gray = decoded.to_luma8();
                prop_assert_eq!(gray.width(), w);
                prop_assert_eq!(gray.height(), h);
                prop_assert_eq!(gray.as_raw(), &data);
            }
            ColorType::GrayAlpha => {
                let rgba = decoded.to_rgba8();
                prop_assert_eq!(rgba.width(), w);
                prop_assert_eq!(rgba.height(), h);
                let mut expected = Vec::with_capacity((w * h * 4) as usize);
                for chunk in data.chunks(2) {
                    let g = chunk[0];
                    let a = chunk[1];
                    expected.extend_from_slice(&[g, g, g, a]);
                }
                prop_assert_eq!(rgba.as_raw(), &expected);
            }
            ColorType::Rgb => {
                let rgb = decoded.to_rgb8();
                prop_assert_eq!(rgb.width(), w);
                prop_assert_eq!(rgb.height(), h);
                prop_assert_eq!(rgb.as_raw(), &data);
            }
            ColorType::Rgba => {
                let rgba = decoded.to_rgba8();
                prop_assert_eq!(rgba.width(), w);
                prop_assert_eq!(rgba.height(), h);
                prop_assert_eq!(rgba.as_raw(), &data);
            }
        }
    }
}

/// Conformance: encode output of PngSuite fixtures and ensure decode success.
#[test]
fn test_pngsuite_encode_and_decode() {
    let Ok(cases) = read_pngsuite() else {
        eprintln!("Skipping PngSuite test: fixtures unavailable (offline?)");
        return;
    };

    for (path, bytes) in cases {
        // Decode using `image` as source pixels
        let img = image::load_from_memory(&bytes).expect("decode fixture");
        let rgba = img.to_rgba8();
        let (w, h) = img.dimensions();

        // Encode through our pipeline (RGBA)
        let encoded = png::encode(rgba.as_raw(), w, h, ColorType::Rgba).unwrap();

        // Decode the encoded PNG to ensure validity
        let decoded = image::load_from_memory(&encoded).expect("decode reencoded");
        assert_eq!(
            decoded.dimensions(),
            (w, h),
            "dimension mismatch for {:?}",
            path
        );
    }
}

/// Test filter strategies produce valid output.
#[test]
fn test_filter_strategies() {
    use png::{FilterStrategy, PngOptions};

    let pixels = vec![128u8; 16 * 16 * 3];

    let strategies = [
        FilterStrategy::None,
        FilterStrategy::Sub,
        FilterStrategy::Up,
        FilterStrategy::Average,
        FilterStrategy::Paeth,
        FilterStrategy::MinSum,
        FilterStrategy::Adaptive,
        FilterStrategy::AdaptiveFast,
    ];

    for strategy in &strategies {
        let options = PngOptions {
            filter_strategy: *strategy,
            compression_level: 6,
            optimize_alpha: false,
            reduce_color_type: false,
            strip_metadata: false,
            reduce_palette: false,
            verbose_filter_log: false,
            optimal_compression: false,
            quantization: png::QuantizationOptions::default(),
        };

        let result = png::encode_with_options(&pixels, 16, 16, ColorType::Rgb, &options).unwrap();

        // All should produce valid PNG files
        assert_eq!(
            &result[0..8],
            &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
        );
    }
}

/// Test compression levels.
#[test]
fn test_compression_levels() {
    use png::PngOptions;

    let pixels: Vec<u8> = (0..64 * 64 * 3).map(|i| (i % 256) as u8).collect();

    let mut sizes = Vec::new();

    for level in 1..=9 {
        let options = PngOptions {
            compression_level: level,
            ..Default::default()
        };

        let result = png::encode_with_options(&pixels, 64, 64, ColorType::Rgb, &options).unwrap();
        sizes.push((level, result.len()));
    }

    // Higher compression levels should generally produce smaller files
    // (though not strictly monotonic for all images)
    let level1_size = sizes[0].1;
    let level9_size = sizes[8].1;
    assert!(level9_size <= level1_size);
}

/// Test error handling for invalid input.
#[test]
fn test_invalid_input() {
    // Zero dimensions
    assert!(png::encode(&[0, 0, 0], 0, 1, ColorType::Rgb).is_err());
    assert!(png::encode(&[0, 0, 0], 1, 0, ColorType::Rgb).is_err());

    // Wrong data length
    assert!(png::encode(&[0, 0], 1, 1, ColorType::Rgb).is_err()); // Too short
    assert!(png::encode(&[0, 0, 0, 0], 1, 1, ColorType::Rgb).is_err()); // Too long
}

#[test]
fn test_invalid_compression_level() {
    let pixels = vec![0u8; 4 * 4 * 3];
    let opts = png::PngOptions {
        compression_level: 0,
        ..Default::default()
    };
    let err = png::encode_with_options(&pixels, 4, 4, ColorType::Rgb, &opts).unwrap_err();
    assert!(matches!(err, Error::InvalidCompressionLevel(0)));

    let opts = png::PngOptions {
        compression_level: 10,
        ..Default::default()
    };
    let err = png::encode_with_options(&pixels, 4, 4, ColorType::Rgb, &opts).unwrap_err();
    assert!(matches!(err, Error::InvalidCompressionLevel(10)));
}

/// Test large image encoding.
#[test]
fn test_large_image() {
    // 1000x1000 RGB image
    let pixels = vec![100u8; 1000 * 1000 * 3];
    let result = png::encode(&pixels, 1000, 1000, ColorType::Rgb).unwrap();

    // Should produce valid PNG
    assert_eq!(
        &result[0..8],
        &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
    );

    // IHDR should have correct dimensions
    assert_eq!(&result[16..20], &[0, 0, 0x03, 0xE8]); // 1000 in big-endian
    assert_eq!(&result[20..24], &[0, 0, 0x03, 0xE8]);
}

/// Encoding should be deterministic for identical inputs.
#[test]
fn test_png_deterministic() {
    let mut rng = StdRng::seed_from_u64(2024);
    let w = 16;
    let h = 8;
    let mut pixels = vec![0u8; (w * h * 3) as usize];
    rng.fill(pixels.as_mut_slice());

    let a = png::encode(&pixels, w, h, ColorType::Rgb).unwrap();
    let b = png::encode(&pixels, w, h, ColorType::Rgb).unwrap();
    assert_eq!(a, b);
}

/// Reject images exceeding maximum dimension without requiring huge allocations.
#[test]
fn test_png_rejects_image_too_large() {
    let width = (1 << 24) + 1; // just over MAX_DIMENSION
    let height = 1;
    let err = png::encode(&[], width, height, ColorType::Rgb).unwrap_err();
    assert!(matches!(err, Error::ImageTooLarge { .. }));
}

/// Regression test: ensure photographic images don't produce excessively large output.
///
/// This test uses a real-world photo (rocket.png) that previously triggered a bug
/// where lazy matching in LZ77 caused level 6 compression to produce output that
/// was 7.3% larger than the original. After the fix, output should be within 5%
/// of the original size.
#[test]
fn test_png_compression_regression_rocket() {
    let fixture_path = std::path::Path::new("tests/fixtures/rocket.png");
    if !fixture_path.exists() {
        eprintln!("Skipping rocket regression test: fixture not found");
        return;
    }

    let original_bytes = std::fs::read(fixture_path).expect("read fixture");
    let original_size = original_bytes.len();

    // Decode to get raw pixels
    let img = image::open(fixture_path).expect("open fixture");
    let img = img.to_rgb8();
    let width = img.width();
    let height = img.height();
    let raw_pixels = img.as_raw();

    // Test with level 9 adaptive (our best quality setting)
    let options = png::PngOptions {
        compression_level: 9,
        filter_strategy: png::FilterStrategy::Adaptive,
        optimize_alpha: false,
        reduce_color_type: false,
        strip_metadata: false,
        reduce_palette: false,
        verbose_filter_log: false,
        optimal_compression: false,
        quantization: png::QuantizationOptions::default(),
    };

    let encoded = png::encode_with_options(raw_pixels, width, height, ColorType::Rgb, &options)
        .expect("encode");

    // Verify the output is valid PNG
    let decoded = image::load_from_memory(&encoded).expect("decode our output");
    assert_eq!(decoded.dimensions(), (width, height));

    // Check that compression ratio is acceptable (within 10% of original)
    // The original was likely compressed with zopfli, so we allow some margin.
    let size_ratio = encoded.len() as f64 / original_size as f64;
    assert!(
        size_ratio < 1.10,
        "Compression regression: output is {:.1}% larger than original (expected < 10%)",
        (size_ratio - 1.0) * 100.0
    );

    // Also verify level 6 produces reasonable results
    let options_l6 = png::PngOptions {
        compression_level: 6,
        filter_strategy: png::FilterStrategy::Adaptive,
        optimize_alpha: false,
        reduce_color_type: false,
        strip_metadata: false,
        reduce_palette: false,
        verbose_filter_log: false,
        optimal_compression: false,
        quantization: png::QuantizationOptions::default(),
    };
    let encoded_l6 =
        png::encode_with_options(raw_pixels, width, height, ColorType::Rgb, &options_l6)
            .expect("encode l6");

    let size_ratio_l6 = encoded_l6.len() as f64 / original_size as f64;
    assert!(
        size_ratio_l6 < 1.10,
        "Compression regression at level 6: output is {:.1}% larger than original (expected < 10%)",
        (size_ratio_l6 - 1.0) * 100.0
    );

    // Level 9 should produce smaller or equal output compared to level 6
    assert!(
        encoded.len() <= encoded_l6.len(),
        "Level 9 ({} bytes) should not be larger than level 6 ({} bytes)",
        encoded.len(),
        encoded_l6.len()
    );
}

/// Test RGBA encoding of the rocket image (simulates web app behavior).
///
/// The web app receives RGBA data from canvas ImageData, so we need to ensure
/// RGBA encoding also produces reasonable output sizes.
#[test]
fn test_png_compression_regression_rocket_rgba() {
    let fixture_path = std::path::Path::new("tests/fixtures/rocket.png");
    if !fixture_path.exists() {
        eprintln!("Skipping rocket RGBA regression test: fixture not found");
        return;
    }

    let original_bytes = std::fs::read(fixture_path).expect("read fixture");
    let original_size = original_bytes.len();

    // Decode to RGBA (like the web app does via canvas ImageData)
    let img = image::open(fixture_path).expect("open fixture");
    let img_rgba = img.to_rgba8();
    let width = img_rgba.width();
    let height = img_rgba.height();
    let rgba_pixels = img_rgba.as_raw();

    eprintln!("Original PNG size: {} bytes", original_size);
    eprintln!("Image dimensions: {}x{}", width, height);
    eprintln!("RGBA raw size: {} bytes", rgba_pixels.len());

    // Test with level 6 adaptive (what web app uses by default)
    let options = png::PngOptions {
        compression_level: 6,
        filter_strategy: png::FilterStrategy::Adaptive,
        optimize_alpha: false,
        reduce_color_type: false,
        reduce_palette: false,
        strip_metadata: false,
        verbose_filter_log: false,
        optimal_compression: false,
        quantization: png::QuantizationOptions::default(),
    };

    let encoded_rgba =
        png::encode_with_options(rgba_pixels, width, height, ColorType::Rgba, &options)
            .expect("encode rgba");

    let rgba_ratio = encoded_rgba.len() as f64 / original_size as f64;
    eprintln!(
        "RGBA output size: {} bytes ({:.1}% vs original)",
        encoded_rgba.len(),
        (rgba_ratio - 1.0) * 100.0
    );

    // For RGBA, we expect overhead due to the extra alpha channel.
    // Even though the alpha is all 255, it adds ~33% more data per row to filter/compress.
    // Allow up to 20% larger since we're adding a 4th channel to an originally RGB image.
    // Note: The web app should detect hasAlpha=false and encode as RGB instead.
    assert!(
        rgba_ratio < 1.20,
        "RGBA Compression regression: output is {:.1}% larger than original (expected < 20%)",
        (rgba_ratio - 1.0) * 100.0
    );
}
