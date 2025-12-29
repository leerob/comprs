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

#[test]
fn png_signature_matches_magic_bytes() {
    let pixels = vec![255, 0, 0];
    let result = encode_png(&pixels, 1, 1, ColorType::Rgb).unwrap();

    assert_eq!(
        &result[0..8],
        &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
    );
}

#[test]
fn ihdr_chunk_fields_match_dimensions() {
    let pixels = vec![0u8; 100 * 100 * 3];
    let result = encode_png(&pixels, 100, 100, ColorType::Rgb).unwrap();

    assert_eq!(&result[8..12], &[0, 0, 0, 13]);
    assert_eq!(&result[12..16], b"IHDR");
    assert_eq!(&result[16..20], &[0, 0, 0, 100]);
    assert_eq!(&result[20..24], &[0, 0, 0, 100]);
    assert_eq!(result[24], 8);
    assert_eq!(result[25], 2);
    assert_eq!(result[26], 0);
    assert_eq!(result[27], 0);
    assert_eq!(result[28], 0);
}

#[test]
fn iend_chunk_present_at_end() {
    let pixels = vec![128u8; 10 * 10 * 3];
    let result = encode_png(&pixels, 10, 10, ColorType::Rgb).unwrap();

    let iend_start = result.len() - 12;
    assert_eq!(&result[iend_start..iend_start + 4], &[0, 0, 0, 0]);
    assert_eq!(&result[iend_start + 4..iend_start + 8], b"IEND");
    assert_eq!(
        &result[iend_start + 8..iend_start + 12],
        &[0xAE, 0x42, 0x60, 0x82]
    );
}

#[test]
fn color_type_field_matches_input() {
    let gray = vec![128u8; 4 * 4];
    let result = encode_png(&gray, 4, 4, ColorType::Gray).unwrap();
    assert_eq!(result[25], 0);

    let gray_alpha = vec![128u8; 4 * 4 * 2];
    let result = encode_png(&gray_alpha, 4, 4, ColorType::GrayAlpha).unwrap();
    assert_eq!(result[25], 4);

    let rgb = vec![128u8; 4 * 4 * 3];
    let result = encode_png(&rgb, 4, 4, ColorType::Rgb).unwrap();
    assert_eq!(result[25], 2);

    let rgba = vec![128u8; 4 * 4 * 4];
    let result = encode_png(&rgba, 4, 4, ColorType::Rgba).unwrap();
    assert_eq!(result[25], 6);
}

#[test]
fn distinct_inputs_produce_distinct_pngs() {
    let black = vec![0u8; 8 * 8 * 3];
    let white = vec![255u8; 8 * 8 * 3];

    let black_png = encode_png(&black, 8, 8, ColorType::Rgb).unwrap();
    let white_png = encode_png(&white, 8, 8, ColorType::Rgb).unwrap();

    assert_ne!(black_png, white_png);
}

#[test]
fn png_roundtrip_rgb_via_image_crate() {
    let width = 3;
    let height = 2;
    let pixels = vec![
        255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0, 0, 255, 255, 255, 0, 255,
    ];

    let encoded = encode_png(&pixels, width, height, ColorType::Rgb).unwrap();

    let decoded = image::load_from_memory(&encoded).expect("decode").to_rgb8();
    assert_eq!(decoded.width(), width);
    assert_eq!(decoded.height(), height);
    assert_eq!(decoded.as_raw(), &pixels);
}

#[test]
fn png_roundtrip_random_small_images() {
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

#[test]
fn png_chunk_crc_and_lengths_are_valid() {
    let mut rng = StdRng::seed_from_u64(777);
    let w = 12;
    let h = 7;
    let mut pixels = vec![0u8; (w * h * 3) as usize];
    rng.fill(pixels.as_mut_slice());

    let encoded = encode_png(&pixels, w, h, ColorType::Rgb).unwrap();

    let mut offset = 8;
    let mut saw_iend = false;

    while offset < encoded.len() {
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

#[test]
fn pngsuite_fixtures_reencode_and_decode() {
    let Ok(cases) = read_pngsuite() else {
        eprintln!("Skipping PngSuite test: fixtures unavailable (offline?)");
        return;
    };

    for (path, bytes) in cases {
        let img = image::load_from_memory(&bytes).expect("decode fixture");
        let rgba = img.to_rgba8();
        let (w, h) = img.dimensions();

        let encoded = encode_png(rgba.as_raw(), w, h, ColorType::Rgba).unwrap();

        let decoded = image::load_from_memory(&encoded).expect("decode reencoded");
        assert_eq!(
            decoded.dimensions(),
            (w, h),
            "dimension mismatch for {:?}",
            path
        );
    }
}

#[test]
fn filter_strategies_produce_valid_signatures() {
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

        assert_eq!(
            &result[0..8],
            &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
        );
    }
}

#[test]
fn compression_level_9_not_larger_than_level_1() {
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

    let level1_size = sizes[0].1;
    let level9_size = sizes[8].1;
    assert!(level9_size <= level1_size);
}

#[test]
fn invalid_dimensions_or_lengths_are_rejected() {
    assert!(encode_png(&[0, 0, 0], 0, 1, ColorType::Rgb).is_err());
    assert!(encode_png(&[0, 0, 0], 1, 0, ColorType::Rgb).is_err());

    assert!(encode_png(&[0, 0], 1, 1, ColorType::Rgb).is_err());
    assert!(encode_png(&[0, 0, 0, 0], 1, 1, ColorType::Rgb).is_err());
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

#[test]
fn large_image_preserves_signature_and_dimensions() {
    let pixels = vec![100u8; 1000 * 1000 * 3];
    let result = encode_png(&pixels, 1000, 1000, ColorType::Rgb).unwrap();

    assert_eq!(
        &result[0..8],
        &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
    );

    assert_eq!(&result[16..20], &[0, 0, 0x03, 0xE8]);
    assert_eq!(&result[20..24], &[0, 0, 0x03, 0xE8]);
}

#[test]
fn png_encoding_is_deterministic() {
    let mut rng = StdRng::seed_from_u64(2024);
    let w = 16;
    let h = 8;
    let mut pixels = vec![0u8; (w * h * 3) as usize];
    rng.fill(pixels.as_mut_slice());

    let a = encode_png(&pixels, w, h, ColorType::Rgb).unwrap();
    let b = encode_png(&pixels, w, h, ColorType::Rgb).unwrap();
    assert_eq!(a, b);
}

#[test]
fn png_rejects_dimensions_over_limit() {
    let width = (1 << 24) + 1;
    let height = 1;
    let err = encode_png(&[], width, height, ColorType::Rgb).unwrap_err();
    assert!(matches!(err, Error::ImageTooLarge { .. }));
}

#[test]
fn png_rejects_zero_dimensions_all_color_types() {
    let color_types = [
        ColorType::Gray,
        ColorType::GrayAlpha,
        ColorType::Rgb,
        ColorType::Rgba,
    ];

    for ct in &color_types {
        let err = encode_png(&[0u8; 100], 0, 10, *ct);
        assert!(err.is_err(), "Should reject zero width for {:?}", ct);

        let err = encode_png(&[0u8; 100], 10, 0, *ct);
        assert!(err.is_err(), "Should reject zero height for {:?}", ct);

        let err = encode_png(&[], 0, 0, *ct);
        assert!(err.is_err(), "Should reject zero dimensions for {:?}", ct);
    }
}

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

        let encoded = encode_png(&pixels, width, height, color_type)
            .expect("encoding should succeed");

        prop_assert_eq!(&encoded[0..8], &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
    }
}

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

        prop_assert_eq!(&encoded[0..8], &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
        let decoded = image::load_from_memory(&encoded).expect("decode");
        prop_assert_eq!(decoded.dimensions(), (width, height));
    }
}

#[test]
fn png_rocket_rgb_output_within_ten_percent() {
    let fixture_path = std::path::Path::new("tests/fixtures/rocket.png");
    if !fixture_path.exists() {
        eprintln!("Skipping rocket regression test: fixture not found");
        return;
    }

    let original_bytes = std::fs::read(fixture_path).expect("read fixture");
    let original_size = original_bytes.len();

    let img = image::open(fixture_path).expect("open fixture");
    let img = img.to_rgb8();
    let width = img.width();
    let height = img.height();
    let raw_pixels = img.as_raw();

    let options = PngOptions::builder(width, height)
        .color_type(ColorType::Rgb)
        .compression_level(9)
        .filter_strategy(png::FilterStrategy::Adaptive)
        .build();

    let encoded = png::encode(raw_pixels, &options).expect("encode");

    let decoded = image::load_from_memory(&encoded).expect("decode our output");
    assert_eq!(decoded.dimensions(), (width, height));

    let size_ratio = encoded.len() as f64 / original_size as f64;
    assert!(
        size_ratio < 1.10,
        "Compression regression: output is {:.1}% larger than original (expected < 10%)",
        (size_ratio - 1.0) * 100.0
    );

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

    assert!(
        encoded.len() <= encoded_l6.len(),
        "Level 9 ({} bytes) should not be larger than level 6 ({} bytes)",
        encoded.len(),
        encoded_l6.len()
    );
}

#[test]
fn png_rocket_rgba_output_within_thirty_percent() {
    let fixture_path = std::path::Path::new("tests/fixtures/rocket.png");
    if !fixture_path.exists() {
        eprintln!("Skipping rocket RGBA regression test: fixture not found");
        return;
    }

    let original_bytes = std::fs::read(fixture_path).expect("read fixture");
    let original_size = original_bytes.len();

    let img = image::open(fixture_path).expect("open fixture");
    let img_rgba = img.to_rgba8();
    let width = img_rgba.width();
    let height = img_rgba.height();
    let rgba_pixels = img_rgba.as_raw();

    eprintln!("Original PNG size: {} bytes", original_size);
    eprintln!("Image dimensions: {}x{}", width, height);
    eprintln!("RGBA raw size: {} bytes", rgba_pixels.len());

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

    assert!(
        rgba_ratio < 1.30,
        "RGBA Compression regression: output is {:.1}% larger than original (expected < 30%)",
        (rgba_ratio - 1.0) * 100.0
    );
}

#[test]
fn synthetic_patterns_roundtrip() {
    let test_suite = synthetic::generate_minimal_test_suite();

    for (name, w, h, pixels) in test_suite {
        let encoded = encode_png(&pixels, w, h, ColorType::Rgb)
            .unwrap_or_else(|e| panic!("Failed to encode {name}: {e}"));

        assert_eq!(
            &encoded[0..8],
            &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A],
            "Invalid PNG signature for {name}"
        );

        let decoded = image::load_from_memory(&encoded)
            .unwrap_or_else(|e| panic!("Failed to decode {name}: {e}"));
        assert_eq!(decoded.width(), w, "Width mismatch for {name}");
        assert_eq!(decoded.height(), h, "Height mismatch for {name}");
    }
}

#[test]
fn synthetic_edge_case_dimensions_roundtrip() {
    for &(w, h, name) in synthetic::EDGE_CASE_DIMENSIONS {
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

#[test]
fn kodak_subset_roundtrip_lossless() {
    let Ok(images) = read_kodak_decoded_subset(4) else {
        eprintln!("Skipping Kodak test: fixtures unavailable (offline?)");
        return;
    };

    for (name, w, h, pixels) in images {
        let options = PngOptions::balanced(w, h);
        let mut opts = options;
        opts.color_type = ColorType::Rgb;
        let encoded = png::encode(&pixels, &opts)
            .unwrap_or_else(|e| panic!("Failed to encode Kodak {name}: {e}"));

        let decoded = image::load_from_memory(&encoded)
            .unwrap_or_else(|e| panic!("Failed to decode Kodak {name}: {e}"));
        assert_eq!(decoded.width(), w, "Width mismatch for Kodak {name}");
        assert_eq!(decoded.height(), h, "Height mismatch for Kodak {name}");

        let decoded_rgb = decoded.to_rgb8();
        assert_eq!(
            decoded_rgb.as_raw(),
            &pixels,
            "Pixel data mismatch for Kodak {name}"
        );
    }
}
