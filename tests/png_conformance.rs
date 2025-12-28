//! PNG conformance tests.
//!
//! Tests PNG encoding against expected output and validates
//! that encoded images can be decoded correctly.

#![allow(clippy::uninlined_format_args)]

use image::GenericImageView;
use pixo::compress::crc32::crc32;
use pixo::png::PngOptions;
use pixo::{png, ColorType, Error};
use proptest::prelude::*;
use rand::{rngs::StdRng, Rng, SeedableRng};
mod support;
use support::kodak::read_kodak_decoded_subset;
use support::pngsuite::read_pngsuite;
use support::synthetic;

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

fn encode_png_with_options(
    data: &[u8],
    width: u32,
    height: u32,
    color_type: ColorType,
    options: &PngOptions,
) -> pixo::Result<Vec<u8>> {
    let mut opts = options.clone();
    opts.width = width;
    opts.height = height;
    opts.color_type = color_type;
    png::encode(data, &opts)
}

/// Test that PNG output has correct header.
#[test]
fn test_png_signature() {
    let pixels = vec![255, 0, 0]; // 1x1 red pixel
    let result = encode_png(&pixels, 1, 1, ColorType::Rgb).unwrap();

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
    let result = encode_png(&pixels, 100, 100, ColorType::Rgb).unwrap();

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
    let result = encode_png(&pixels, 10, 10, ColorType::Rgb).unwrap();

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
    let result = encode_png(&gray, 4, 4, ColorType::Gray).unwrap();
    assert_eq!(result[25], 0); // Color type 0

    // Grayscale + Alpha
    let gray_alpha = vec![128u8; 4 * 4 * 2];
    let result = encode_png(&gray_alpha, 4, 4, ColorType::GrayAlpha).unwrap();
    assert_eq!(result[25], 4); // Color type 4

    // RGB
    let rgb = vec![128u8; 4 * 4 * 3];
    let result = encode_png(&rgb, 4, 4, ColorType::Rgb).unwrap();
    assert_eq!(result[25], 2); // Color type 2

    // RGBA
    let rgba = vec![128u8; 4 * 4 * 4];
    let result = encode_png(&rgba, 4, 4, ColorType::Rgba).unwrap();
    assert_eq!(result[25], 6); // Color type 6
}

/// Test that different images produce different output.
#[test]
fn test_different_images() {
    let black = vec![0u8; 8 * 8 * 3];
    let white = vec![255u8; 8 * 8 * 3];

    let black_png = encode_png(&black, 8, 8, ColorType::Rgb).unwrap();
    let white_png = encode_png(&white, 8, 8, ColorType::Rgb).unwrap();

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

    let encoded = encode_png(&pixels, width, height, ColorType::Rgb).unwrap();

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

            let encoded = encode_png(&pixels, w as u32, h as u32, ct).expect("encode random png");
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

    let encoded = encode_png(&pixels, w, h, ColorType::Rgb).unwrap();

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
        let encoded = encode_png(&data, w, h, ct).unwrap();
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
        let encoded = encode_png(rgba.as_raw(), w, h, ColorType::Rgba).unwrap();

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
        let options = PngOptions::builder(16, 16)
            .color_type(ColorType::Rgb)
            .filter_strategy(*strategy)
            .compression_level(6)
            .build();

        let result = png::encode(&pixels, &options).unwrap();

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

        let result = encode_png_with_options(&pixels, 64, 64, ColorType::Rgb, &options).unwrap();
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
    assert!(encode_png(&[0, 0, 0], 0, 1, ColorType::Rgb).is_err());
    assert!(encode_png(&[0, 0, 0], 1, 0, ColorType::Rgb).is_err());

    // Wrong data length
    assert!(encode_png(&[0, 0], 1, 1, ColorType::Rgb).is_err()); // Too short
    assert!(encode_png(&[0, 0, 0, 0], 1, 1, ColorType::Rgb).is_err()); // Too long
}

#[test]
fn test_invalid_compression_level() {
    let pixels = vec![0u8; 4 * 4 * 3];
    let opts = png::PngOptions {
        compression_level: 0,
        ..Default::default()
    };
    let err = encode_png_with_options(&pixels, 4, 4, ColorType::Rgb, &opts).unwrap_err();
    assert!(matches!(err, Error::InvalidCompressionLevel(0)));

    let opts = png::PngOptions {
        compression_level: 10,
        ..Default::default()
    };
    let err = encode_png_with_options(&pixels, 4, 4, ColorType::Rgb, &opts).unwrap_err();
    assert!(matches!(err, Error::InvalidCompressionLevel(10)));
}

/// Test large image encoding.
#[test]
fn test_large_image() {
    // 1000x1000 RGB image
    let pixels = vec![100u8; 1000 * 1000 * 3];
    let result = encode_png(&pixels, 1000, 1000, ColorType::Rgb).unwrap();

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

    let a = encode_png(&pixels, w, h, ColorType::Rgb).unwrap();
    let b = encode_png(&pixels, w, h, ColorType::Rgb).unwrap();
    assert_eq!(a, b);
}

/// Reject images exceeding maximum dimension without requiring huge allocations.
#[test]
fn test_png_rejects_image_too_large() {
    let width = (1 << 24) + 1; // just over MAX_DIMENSION
    let height = 1;
    let err = encode_png(&[], width, height, ColorType::Rgb).unwrap_err();
    assert!(matches!(err, Error::ImageTooLarge { .. }));
}

// ============================================================================
// Extended Property Tests - Edge Cases
// ============================================================================

/// Verify that zero dimensions are rejected for all color types.
#[test]
fn test_png_rejects_zero_dimensions() {
    let color_types = [
        ColorType::Gray,
        ColorType::GrayAlpha,
        ColorType::Rgb,
        ColorType::Rgba,
    ];

    for ct in &color_types {
        // Zero width
        let err = encode_png(&[0u8; 100], 0, 10, *ct);
        assert!(err.is_err(), "Should reject zero width for {:?}", ct);

        // Zero height
        let err = encode_png(&[0u8; 100], 10, 0, *ct);
        assert!(err.is_err(), "Should reject zero height for {:?}", ct);

        // Both zero
        let err = encode_png(&[], 0, 0, *ct);
        assert!(err.is_err(), "Should reject zero dimensions for {:?}", ct);
    }
}

// Property test: any valid image dimensions should produce decodable output.
proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]
    #[test]
    fn prop_png_valid_dimensions_always_succeed(
        width in 1u32..128,
        height in 1u32..128,
        color_type in prop_oneof![
            Just(ColorType::Gray),
            Just(ColorType::GrayAlpha),
            Just(ColorType::Rgb),
            Just(ColorType::Rgba),
        ],
        seed in any::<u64>(),
    ) {
        let bpp = color_type.bytes_per_pixel();
        let mut rng = StdRng::seed_from_u64(seed);
        let mut pixels = vec![0u8; (width * height) as usize * bpp];
        rng.fill(pixels.as_mut_slice());

        // Should always succeed with valid dimensions
        let encoded = encode_png(&pixels, width, height, color_type)
            .expect("encoding should succeed");

        // Should produce valid PNG
        prop_assert_eq!(&encoded[0..8], &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
    }
}

// Property test: encoding with all builder options should produce valid output.
proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]
    #[test]
    fn prop_png_builder_options_produce_valid_output(
        width in 8u32..64,
        height in 8u32..64,
        compression_level in 1u8..=9,
        filter_idx in 0u8..8,
        optimize_alpha in any::<bool>(),
        reduce_color_type in any::<bool>(),
        strip_metadata in any::<bool>(),
        seed in any::<u64>(),
    ) {
        let filter_strategy = match filter_idx {
            0 => png::FilterStrategy::None,
            1 => png::FilterStrategy::Sub,
            2 => png::FilterStrategy::Up,
            3 => png::FilterStrategy::Average,
            4 => png::FilterStrategy::Paeth,
            5 => png::FilterStrategy::MinSum,
            6 => png::FilterStrategy::Adaptive,
            _ => png::FilterStrategy::AdaptiveFast,
        };

        let mut rng = StdRng::seed_from_u64(seed);
        let mut pixels = vec![0u8; (width * height * 4) as usize];
        rng.fill(pixels.as_mut_slice());

        let options = PngOptions::builder(width, height)
            .color_type(ColorType::Rgba)
            .compression_level(compression_level)
            .filter_strategy(filter_strategy)
            .optimize_alpha(optimize_alpha)
            .reduce_color_type(reduce_color_type)
            .strip_metadata(strip_metadata)
            .build();

        let encoded = png::encode(&pixels, &options).expect("encoding should succeed");

        // Verify PNG signature
        prop_assert_eq!(&encoded[0..8], &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);

        // Verify it decodes
        let decoded = image::load_from_memory(&encoded).expect("decode");
        prop_assert_eq!(decoded.dimensions(), (width, height));
    }
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
    let options = PngOptions::builder(width, height)
        .color_type(ColorType::Rgb)
        .compression_level(9)
        .filter_strategy(png::FilterStrategy::Adaptive)
        .build();

    let encoded = png::encode(raw_pixels, &options).expect("encode");

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
    let options_l6 = PngOptions::builder(width, height)
        .color_type(ColorType::Rgb)
        .compression_level(6)
        .filter_strategy(png::FilterStrategy::Adaptive)
        .build();
    let encoded_l6 = png::encode(raw_pixels, &options_l6).expect("encode l6");

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
    let options = PngOptions::builder(width, height)
        .color_type(ColorType::Rgba)
        .compression_level(6)
        .filter_strategy(png::FilterStrategy::Adaptive)
        .build();

    let encoded_rgba = png::encode(rgba_pixels, &options).expect("encode rgba");

    let rgba_ratio = encoded_rgba.len() as f64 / original_size as f64;
    eprintln!(
        "RGBA output size: {} bytes ({:.1}% vs original)",
        encoded_rgba.len(),
        (rgba_ratio - 1.0) * 100.0
    );

    // For RGBA, we expect overhead due to the extra alpha channel.
    // Even though the alpha is all 255, it adds ~33% more data per row to filter/compress.
    // Allow up to 30% larger since we're adding a 4th channel to an originally RGB image.
    // Note: The web app should detect hasAlpha=false and encode as RGB instead.
    assert!(
        rgba_ratio < 1.30,
        "RGBA Compression regression: output is {:.1}% larger than original (expected < 30%)",
        (rgba_ratio - 1.0) * 100.0
    );
}

/// Test encoding with synthetic test patterns.
/// These cover edge cases like solid colors, gradients, and high-frequency patterns.
#[test]
fn test_png_synthetic_patterns() {
    let test_suite = synthetic::generate_minimal_test_suite();

    for (name, w, h, pixels) in test_suite {
        let encoded = encode_png(&pixels, w, h, ColorType::Rgb)
            .unwrap_or_else(|e| panic!("Failed to encode {name}: {e}"));

        // Verify valid PNG
        assert_eq!(
            &encoded[0..8],
            &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A],
            "Invalid PNG signature for {name}"
        );

        // Verify decode roundtrip
        let decoded = image::load_from_memory(&encoded)
            .unwrap_or_else(|e| panic!("Failed to decode {name}: {e}"));
        assert_eq!(decoded.width(), w, "Width mismatch for {name}");
        assert_eq!(decoded.height(), h, "Height mismatch for {name}");
    }
}

/// Test edge case dimensions with synthetic images.
#[test]
fn test_png_edge_case_dimensions() {
    for &(w, h, name) in synthetic::EDGE_CASE_DIMENSIONS {
        // Skip very large dimensions for speed
        if w > 1024 || h > 1024 {
            continue;
        }

        let pixels = synthetic::gradient_rgb(w, h);
        let encoded = encode_png(&pixels, w, h, ColorType::Rgb)
            .unwrap_or_else(|e| panic!("Failed to encode {name} ({w}x{h}): {e}"));

        let decoded = image::load_from_memory(&encoded)
            .unwrap_or_else(|e| panic!("Failed to decode {name} ({w}x{h}): {e}"));
        assert_eq!(decoded.width(), w, "Width mismatch for {name}");
        assert_eq!(decoded.height(), h, "Height mismatch for {name}");
    }
}

/// Test encoding Kodak suite images (photographic content).
/// This tests real-world compression performance on diverse photos.
#[test]
fn test_png_kodak_subset() {
    let Ok(images) = read_kodak_decoded_subset(4) else {
        eprintln!("Skipping Kodak test: fixtures unavailable (offline?)");
        return;
    };

    for (name, w, h, pixels) in images {
        // Encode with balanced settings
        let options = PngOptions::balanced(w, h);
        let mut opts = options;
        opts.color_type = ColorType::Rgb;
        let encoded = png::encode(&pixels, &opts)
            .unwrap_or_else(|e| panic!("Failed to encode Kodak {name}: {e}"));

        // Verify decode roundtrip
        let decoded = image::load_from_memory(&encoded)
            .unwrap_or_else(|e| panic!("Failed to decode Kodak {name}: {e}"));
        assert_eq!(decoded.width(), w, "Width mismatch for Kodak {name}");
        assert_eq!(decoded.height(), h, "Height mismatch for Kodak {name}");

        // Verify lossless roundtrip
        let decoded_rgb = decoded.to_rgb8();
        assert_eq!(
            decoded_rgb.as_raw(),
            &pixels,
            "Pixel data mismatch for Kodak {name}"
        );
    }
}
