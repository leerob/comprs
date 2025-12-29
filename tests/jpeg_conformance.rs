#![allow(
    clippy::uninlined_format_args,
    clippy::needless_range_loop,
    clippy::unnecessary_cast
)]

use image::GenericImageView;
use pixo::jpeg::JpegOptions;
use pixo::{jpeg, ColorType};
use proptest::prelude::*;
use rand::{rngs::StdRng, Rng, SeedableRng};
mod support;
use support::jpeg_corpus::read_jpeg_corpus;
use support::kodak::read_kodak_decoded_subset;
use support::synthetic;

fn encode_jpeg(data: &[u8], width: u32, height: u32, quality: u8) -> pixo::Result<Vec<u8>> {
    let options = JpegOptions::builder(width, height)
        .color_type(ColorType::Rgb)
        .quality(quality)
        .build();
    jpeg::encode(data, &options)
}

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

fn test_jpeg_fast(width: u32, height: u32, quality: u8) -> JpegOptions {
    JpegOptions::fast(width, height, quality)
}

fn test_jpeg_balanced(width: u32, height: u32, quality: u8) -> JpegOptions {
    JpegOptions::balanced(width, height, quality)
}

fn test_jpeg_max(width: u32, height: u32, quality: u8) -> JpegOptions {
    JpegOptions::max(width, height, quality)
}

#[test]
fn jpeg_output_contains_soi_and_eoi() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let result = encode_jpeg(&pixels, 8, 8, 85).unwrap();

    assert_eq!(&result[0..2], &[0xFF, 0xD8]);
    assert_eq!(&result[result.len() - 2..], &[0xFF, 0xD9]);
}

#[test]
fn jpeg_app0_marker_present() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let result = encode_jpeg(&pixels, 8, 8, 85).unwrap();

    assert_eq!(&result[2..4], &[0xFF, 0xE0]);
    assert_eq!(&result[6..11], b"JFIF\0");
}

#[test]
fn jpeg_quality_levels_increase_size() {
    let pixels: Vec<u8> = (0..64 * 64 * 3).map(|i| (i % 256) as u8).collect();

    let sizes: Vec<(u8, usize)> = [10, 25, 50, 75, 90, 100]
        .iter()
        .map(|&q| {
            let result = encode_jpeg(&pixels, 64, 64, q).unwrap();
            (q, result.len())
        })
        .collect();

    for i in 1..sizes.len() {
        assert!(
            sizes[i].1 >= sizes[i - 1].1,
            "Quality {} produced {} bytes, but quality {} produced {} bytes",
            sizes[i].0,
            sizes[i].1,
            sizes[i - 1].0,
            sizes[i - 1].1
        );
    }
}

#[test]
fn jpeg_restart_interval_decodes_with_image_crate() {
    use std::io::Cursor;

    let width = 16;
    let height = 8;
    let mut rng = StdRng::seed_from_u64(8888);
    let mut rgb = vec![0u8; (width * height * 3) as usize];
    rng.fill(rgb.as_mut_slice());

    let mut opts = test_jpeg_fast(width, height, 85);
    opts.restart_interval = Some(2);

    let jpeg_bytes = jpeg::encode(&rgb, &opts).unwrap();

    let reader = image::ImageReader::new(Cursor::new(jpeg_bytes))
        .with_guessed_format()
        .expect("cursor io never fails");
    let img = reader.decode().expect("decode restart-interval JPEG");

    assert_eq!(img.width() as usize, width as usize);
    assert_eq!(img.height() as usize, height as usize);
    assert_eq!(
        img.into_rgb8().into_raw().len(),
        (width * height * 3) as usize
    );
}

#[test]
fn jpeg_progressive_decodes_with_image_crate() {
    use std::io::Cursor;

    let width = 16;
    let height = 12;
    let mut rng = StdRng::seed_from_u64(9999);
    let mut rgb = vec![0u8; (width * height * 3) as usize];
    rng.fill(rgb.as_mut_slice());

    let opts = JpegOptions::builder(width, height)
        .color_type(ColorType::Rgb)
        .quality(85)
        .subsampling(jpeg::Subsampling::S444)
        .optimize_huffman(true)
        .progressive(true)
        .build();

    let jpeg_bytes = jpeg::encode(&rgb, &opts).unwrap();

    let reader = image::ImageReader::new(Cursor::new(jpeg_bytes))
        .with_guessed_format()
        .expect("cursor io never fails");
    let img = reader.decode().expect("decode progressive JPEG");

    assert_eq!(img.width() as usize, width as usize);
    assert_eq!(img.height() as usize, height as usize);
    assert_eq!(
        img.into_rgb8().into_raw().len(),
        (width * height * 3) as usize
    );
}

#[test]
fn jpeg_progressive_and_baseline_markers_are_correct() {
    let width = 8;
    let height = 8;
    let rgb = vec![128u8; width * height * 3];

    let baseline_opts = test_jpeg_fast(width as u32, height as u32, 80);
    let baseline = jpeg::encode(&rgb, &baseline_opts).unwrap();
    assert!(
        baseline.windows(2).any(|w| w == [0xFF, 0xC0]),
        "baseline SOF0 missing"
    );
    assert!(
        !baseline.windows(2).any(|w| w == [0xFF, 0xC2]),
        "baseline should not contain SOF2"
    );

    let progressive_opts = test_jpeg_max(width as u32, height as u32, 80);
    let progressive = jpeg::encode(&rgb, &progressive_opts).unwrap();
    assert!(
        progressive.windows(2).any(|w| w == [0xFF, 0xC2]),
        "progressive SOF2 missing"
    );
}
#[test]
fn jpeg_various_sizes_have_soi_and_eoi() {
    let sizes = [
        (1, 1),
        (7, 7),
        (8, 8),
        (9, 9),
        (16, 16),
        (100, 50),
        (50, 100),
    ];

    for (width, height) in sizes {
        let pixels = vec![128u8; (width * height * 3) as usize];
        let result = encode_jpeg(&pixels, width, height, 85);

        assert!(result.is_ok(), "Failed for size {}x{}", width, height);

        let data = result.unwrap();
        assert_eq!(
            &data[0..2],
            &[0xFF, 0xD8],
            "Missing SOI for {}x{}",
            width,
            height
        );
        assert_eq!(
            &data[data.len() - 2..],
            &[0xFF, 0xD9],
            "Missing EOI for {}x{}",
            width,
            height
        );
    }
}

#[test]
fn jpeg_grayscale_markers_and_size() {
    let pixels = vec![128u8; 32 * 32];
    let result = encode_jpeg_with_color(&pixels, 32, 32, 85, ColorType::Gray).unwrap();

    assert_eq!(&result[0..2], &[0xFF, 0xD8]);
    assert_eq!(&result[result.len() - 2..], &[0xFF, 0xD9]);

    let rgb_pixels = vec![128u8; 32 * 32 * 3];
    let rgb_result = encode_jpeg(&rgb_pixels, 32, 32, 85).unwrap();
    assert!(result.len() < rgb_result.len());
}

#[test]
fn jpeg_invalid_parameters_are_rejected() {
    let pixels = vec![0u8; 8 * 8 * 3];
    assert!(encode_jpeg(&pixels, 8, 8, 0).is_err());
    assert!(encode_jpeg(&pixels, 8, 8, 101).is_err());

    assert!(encode_jpeg(&pixels, 0, 8, 85).is_err());
    assert!(encode_jpeg(&pixels, 8, 0, 85).is_err());

    assert!(encode_jpeg(&[0, 0], 8, 8, 85).is_err());
}

#[test]
fn test_invalid_restart_interval() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let mut opts = test_jpeg_fast(8, 8, 85);
    opts.restart_interval = Some(0);
    let result = jpeg::encode(&pixels, &opts);
    assert!(result.is_err());
}

#[test]
fn test_unsupported_color_type_rejected() {
    let pixels = vec![0u8; 4 * 4 * 4];
    let result = encode_jpeg_with_color(&pixels, 4, 4, 85, ColorType::Rgba);
    assert!(result.is_err());
}

#[test]
fn test_image_too_large() {
    let width = 65_536;
    let height = 1;
    let pixels = vec![0u8; (width as usize * height as usize * 3) as usize];
    let err = encode_jpeg(&pixels, width, height, 85).unwrap_err();
    assert!(matches!(err, pixo::Error::ImageTooLarge { .. }));
}

#[test]
fn jpeg_rejects_zero_dimensions() {
    let color_types = [ColorType::Gray, ColorType::Rgb];

    for ct in &color_types {
        let err = encode_jpeg_with_color(&[0u8; 100], 0, 10, 85, *ct);
        assert!(err.is_err(), "Should reject zero width for {:?}", ct);

        let err = encode_jpeg_with_color(&[0u8; 100], 10, 0, 85, *ct);
        assert!(err.is_err(), "Should reject zero height for {:?}", ct);

        let err = encode_jpeg_with_color(&[], 0, 0, 85, *ct);
        assert!(err.is_err(), "Should reject zero dimensions for {:?}", ct);
    }
}

#[test]
fn jpeg_quality_edge_values_are_valid() {
    let pixels = vec![128u8; 8 * 8 * 3];

    let result = encode_jpeg(&pixels, 8, 8, 1);
    assert!(result.is_ok(), "Quality 1 should be valid");

    let result = encode_jpeg(&pixels, 8, 8, 100);
    assert!(result.is_ok(), "Quality 100 should be valid");
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]
    #[test]
    fn prop_jpeg_valid_params_always_succeed(
        width in 1u32..64,
        height in 1u32..64,
        quality in 1u8..=100,
        is_gray in any::<bool>(),
        seed in any::<u64>(),
    ) {
        let color_type = if is_gray { ColorType::Gray } else { ColorType::Rgb };
        let bpp = color_type.bytes_per_pixel();

        let mut rng = StdRng::seed_from_u64(seed);
        let mut pixels = vec![0u8; (width * height) as usize * bpp];
        rng.fill(pixels.as_mut_slice());

        let encoded = encode_jpeg_with_color(&pixels, width, height, quality, color_type)
            .expect("encoding should succeed");

        prop_assert_eq!(&encoded[0..2], &[0xFF, 0xD8], "Missing SOI");
        prop_assert_eq!(&encoded[encoded.len()-2..], &[0xFF, 0xD9], "Missing EOI");
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]
    #[test]
    fn prop_jpeg_builder_options_produce_valid_output(
        width in 8u32..32,
        height in 8u32..32,
        quality in 50u8..95,
        use_420 in any::<bool>(),
        optimize_huffman in any::<bool>(),
        restart in proptest::option::of(1u16..8),
        seed in any::<u64>(),
    ) {
        let subsampling = if use_420 {
            jpeg::Subsampling::S420
        } else {
            jpeg::Subsampling::S444
        };

        let mut rng = StdRng::seed_from_u64(seed);
        let mut pixels = vec![0u8; (width * height * 3) as usize];
        rng.fill(pixels.as_mut_slice());

        let options = JpegOptions::builder(width, height)
            .color_type(ColorType::Rgb)
            .quality(quality)
            .subsampling(subsampling)
            .optimize_huffman(optimize_huffman)
            .restart_interval(restart)
            .build();

        let encoded = encode_jpeg_with_options(&pixels, width, height, ColorType::Rgb, &options)
            .expect("encoding should succeed");

        prop_assert_eq!(&encoded[0..2], &[0xFF, 0xD8], "Missing SOI");
        prop_assert_eq!(&encoded[encoded.len()-2..], &[0xFF, 0xD9], "Missing EOI");
        let decoded = image::load_from_memory(&encoded).expect("decode");
        prop_assert_eq!(decoded.dimensions(), (width, height));
    }
}

#[test]
fn jpeg_encoding_is_deterministic() {
    let pixels = vec![100u8; 16 * 16 * 3];

    let result1 = encode_jpeg(&pixels, 16, 16, 85).unwrap();
    let result2 = encode_jpeg(&pixels, 16, 16, 85).unwrap();

    assert_eq!(result1, result2);
}

#[test]
fn jpeg_pattern_sizes_are_ordered_by_complexity() {
    let solid = vec![128u8; 64 * 64 * 3];
    let solid_result = encode_jpeg(&solid, 64, 64, 85).unwrap();

    let mut gradient = Vec::with_capacity(64 * 64 * 3);
    for y in 0..64 {
        for x in 0..64 {
            gradient.push(((x * 4) % 256) as u8);
            gradient.push(((y * 4) % 256) as u8);
            gradient.push((((x + y) * 2) % 256) as u8);
        }
    }
    let gradient_result = encode_jpeg(&gradient, 64, 64, 85).unwrap();

    let mut noisy = Vec::with_capacity(64 * 64 * 3);
    let mut seed = 42u32;
    for _ in 0..(64 * 64 * 3) {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        noisy.push((seed >> 16) as u8);
    }
    let noisy_result = encode_jpeg(&noisy, 64, 64, 85).unwrap();

    assert!(solid_result.len() < gradient_result.len());
    assert!(gradient_result.len() < noisy_result.len());
}

#[test]
fn jpeg_decode_via_image_rgb_and_gray() {
    let mut rgb = vec![0u8; 8 * 8 * 3];
    for i in 0..rgb.len() {
        rgb[i] = (i as u8).wrapping_mul(31);
    }
    let jpeg_rgb = encode_jpeg(&rgb, 8, 8, 85).unwrap();
    let decoded_rgb = image::load_from_memory(&jpeg_rgb).expect("decode rgb");
    assert_eq!(decoded_rgb.width(), 8);
    assert_eq!(decoded_rgb.height(), 8);

    let mut rng = StdRng::seed_from_u64(1337);
    let mut gray = vec![0u8; 7 * 5];
    rng.fill(gray.as_mut_slice());
    let jpeg_gray = encode_jpeg_with_color(&gray, 7, 5, 75, ColorType::Gray).unwrap();
    let decoded_gray = image::load_from_memory(&jpeg_gray).expect("decode gray");
    assert_eq!(decoded_gray.width(), 7);
    assert_eq!(decoded_gray.height(), 5);
}

#[test]
fn jpeg_decode_random_small_images() {
    let mut rng = StdRng::seed_from_u64(2025);
    let dims = [(1, 1), (2, 3), (5, 4), (8, 8), (16, 9)];
    let qualities = [50u8, 85u8, 95u8];

    for &(w, h) in &dims {
        let mut rgb = vec![0u8; w * h * 3];
        rng.fill(rgb.as_mut_slice());
        for &q in &qualities {
            let jpeg_rgb = encode_jpeg(&rgb, w as u32, h as u32, q).unwrap();
            let decoded = image::load_from_memory(&jpeg_rgb).expect("decode rgb");
            assert_eq!(decoded.width(), w as u32);
            assert_eq!(decoded.height(), h as u32);
        }

        let mut gray = vec![0u8; w * h];
        rng.fill(gray.as_mut_slice());
        for &q in &qualities {
            let jpeg_gray =
                encode_jpeg_with_color(&gray, w as u32, h as u32, q, ColorType::Gray).unwrap();
            let decoded = image::load_from_memory(&jpeg_gray).expect("decode gray");
            assert_eq!(decoded.width(), w as u32);
            assert_eq!(decoded.height(), h as u32);
        }
    }
}

#[test]
fn jpeg_subsampling_420_not_larger_than_444() {
    let width = 32;
    let height = 32;
    let mut rng = StdRng::seed_from_u64(4242);
    let mut rgb = vec![0u8; (width * height * 3) as usize];
    rng.fill(rgb.as_mut_slice());

    let opts_444 = test_jpeg_fast(width, height, 75);
    let mut opts_420 = test_jpeg_fast(width, height, 75);
    opts_420.subsampling = jpeg::Subsampling::S420;

    let jpeg_444 =
        encode_jpeg_with_options(&rgb, width, height, ColorType::Rgb, &opts_444).unwrap();
    let jpeg_420 =
        encode_jpeg_with_options(&rgb, width, height, ColorType::Rgb, &opts_420).unwrap();

    assert!(jpeg_420.len() <= jpeg_444.len());

    let decoded = image::load_from_memory(&jpeg_420).expect("decode 420");
    assert_eq!(decoded.dimensions(), (width, height));
}

#[test]
fn jpeg_restart_interval_marker_present_and_decodes() {
    let width = 16;
    let height = 16;
    let mut rng = StdRng::seed_from_u64(5151);
    let mut rgb = vec![0u8; (width * height * 3) as usize];
    rng.fill(rgb.as_mut_slice());

    let mut opts = test_jpeg_fast(width, height, 80);
    opts.restart_interval = Some(4);

    let jpeg_bytes = encode_jpeg_with_options(&rgb, width, height, ColorType::Rgb, &opts).unwrap();

    let mut found_dri = false;
    for w in jpeg_bytes.windows(2) {
        if w == [0xFF, 0xDD] {
            found_dri = true;
            break;
        }
    }
    assert!(found_dri, "DRI marker not found");

    let decoded = image::load_from_memory(&jpeg_bytes).expect("decode with restart interval");
    assert_eq!(decoded.dimensions(), (width, height));
}

#[test]
fn jpeg_marker_structure_with_restart_interval() {
    let width = 16;
    let height = 12;
    let mut rng = StdRng::seed_from_u64(6262);
    let mut rgb = vec![0u8; (width * height * 3) as usize];
    rng.fill(rgb.as_mut_slice());

    let mut opts = test_jpeg_fast(width, height, 85);
    opts.subsampling = jpeg::Subsampling::S420;
    opts.restart_interval = Some(4);

    let jpeg_bytes = encode_jpeg_with_options(&rgb, width, height, ColorType::Rgb, &opts).unwrap();

    assert!(jpeg_bytes.starts_with(&[0xFF, 0xD8]), "missing SOI");
    assert!(jpeg_bytes.ends_with(&[0xFF, 0xD9]), "missing EOI");

    let mut offset = 2;
    let mut saw_app0 = false;
    let mut saw_dqt = false;
    let mut saw_sof0 = false;
    let mut saw_dht = false;
    let mut saw_dri = false;
    let mut saw_sos = false;

    while offset + 4 <= jpeg_bytes.len() {
        assert_eq!(jpeg_bytes[offset], 0xFF, "marker sync lost at {offset}");
        let marker = jpeg_bytes[offset + 1];
        offset += 2;

        if marker == 0xD9 {
            break;
        }

        assert!(
            offset + 2 <= jpeg_bytes.len(),
            "truncated marker length for 0x{:02X}",
            marker
        );
        let len = u16::from_be_bytes([jpeg_bytes[offset], jpeg_bytes[offset + 1]]) as usize;
        assert!(len >= 2, "invalid length for marker 0x{:02X}", marker);
        offset += 2;
        assert!(
            offset + len - 2 <= jpeg_bytes.len(),
            "segment overruns buffer for marker 0x{:02X}",
            marker
        );

        match marker {
            0xE0 => saw_app0 = true,
            0xDB => saw_dqt = true,
            0xC0 => saw_sof0 = true,
            0xC4 => saw_dht = true,
            0xDD => saw_dri = true,
            0xDA => {
                saw_sos = true;
                break;
            }
            _ => {}
        }

        offset += len - 2;
    }

    assert!(saw_app0, "APP0 not found");
    assert!(saw_dqt, "DQT not found");
    assert!(saw_sof0, "SOF0 not found");
    assert!(saw_dht, "DHT not found");
    assert!(saw_sos, "SOS not found");
    assert!(saw_dri, "DRI not found despite restart_interval");
}

#[test]
fn jpeg_no_restart_marker_without_interval() {
    let width = 12;
    let height = 9;
    let mut rng = StdRng::seed_from_u64(7373);
    let mut rgb = vec![0u8; (width * height * 3) as usize];
    rng.fill(rgb.as_mut_slice());

    let opts = test_jpeg_fast(width, height, 80);

    let jpeg_bytes = encode_jpeg_with_options(&rgb, width, height, ColorType::Rgb, &opts).unwrap();

    assert!(
        !jpeg_bytes.windows(2).any(|w| w == [0xFF, 0xDD]),
        "Unexpected DRI marker when restart_interval is None"
    );
}

#[test]
fn jpeg_no_trailing_restart_marker_when_divisible_444() {
    let width = 16;
    let height = 16;
    let mut rng = StdRng::seed_from_u64(9999);
    let mut rgb = vec![0u8; (width * height * 3) as usize];
    rng.fill(rgb.as_mut_slice());

    let mut opts = test_jpeg_fast(width, height, 85);
    opts.restart_interval = Some(4);

    let jpeg_bytes = encode_jpeg_with_options(&rgb, width, height, ColorType::Rgb, &opts).unwrap();

    assert!(jpeg_bytes.ends_with(&[0xFF, 0xD9]), "missing EOI");

    let len = jpeg_bytes.len();
    if len >= 4 {
        let before_eoi = &jpeg_bytes[len - 4..len - 2];
        let is_restart_marker =
            before_eoi[0] == 0xFF && (before_eoi[1] >= 0xD0 && before_eoi[1] <= 0xD7);
        assert!(
            !is_restart_marker,
            "Found trailing restart marker {:02X}{:02X} before EOI - should not be present when MCU count is exactly divisible by interval",
            before_eoi[0], before_eoi[1]
        );
    }

    let decoded = image::load_from_memory(&jpeg_bytes).expect("decode should succeed");
    assert_eq!(decoded.dimensions(), (width, height));
}

#[test]
fn jpeg_no_trailing_restart_marker_420_exact_multiple() {
    let width = 32;
    let height = 32;
    let mut rng = StdRng::seed_from_u64(8888);
    let mut rgb = vec![0u8; (width * height * 3) as usize];
    rng.fill(rgb.as_mut_slice());

    let mut opts = test_jpeg_fast(width, height, 85);
    opts.subsampling = jpeg::Subsampling::S420;
    opts.restart_interval = Some(2);

    let jpeg_bytes = encode_jpeg_with_options(&rgb, width, height, ColorType::Rgb, &opts).unwrap();

    assert!(jpeg_bytes.ends_with(&[0xFF, 0xD9]), "missing EOI");

    let len = jpeg_bytes.len();
    if len >= 4 {
        let before_eoi = &jpeg_bytes[len - 4..len - 2];
        let is_restart_marker =
            before_eoi[0] == 0xFF && (before_eoi[1] >= 0xD0 && before_eoi[1] <= 0xD7);
        assert!(
            !is_restart_marker,
            "Found trailing restart marker before EOI in 4:2:0 mode"
        );
    }

    let decoded = image::load_from_memory(&jpeg_bytes).expect("decode should succeed");
    assert_eq!(decoded.dimensions(), (width, height));
}

fn jpeg_case_strategy() -> impl Strategy<
    Value = (
        u32,
        u32,
        u8,
        ColorType,
        jpeg::Subsampling,
        Option<u16>,
        Vec<u8>,
    ),
> {
    (1u32..24, 1u32..24, 30u8..96).prop_flat_map(|(w, h, q)| {
        prop_oneof![Just(ColorType::Rgb), Just(ColorType::Gray)].prop_flat_map(move |color_type| {
            let bytes_per_pixel = match color_type {
                ColorType::Rgb => 3,
                ColorType::Gray => 1,
                _ => unreachable!(),
            };
            let subsampling = if matches!(color_type, ColorType::Rgb) {
                prop_oneof![Just(jpeg::Subsampling::S444), Just(jpeg::Subsampling::S420),].boxed()
            } else {
                Just(jpeg::Subsampling::S444).boxed()
            };
            let restart = prop_oneof![Just(None), (1u16..8u16).prop_map(Some)];
            (subsampling, restart).prop_flat_map(move |(subsampling, restart_interval)| {
                let len = (w * h) as usize * bytes_per_pixel;
                proptest::collection::vec(any::<u8>(), len).prop_map(move |data| {
                    (w, h, q, color_type, subsampling, restart_interval, data)
                })
            })
        })
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(24))]
    #[test]
    fn prop_jpeg_decode_randomized_options(
        (w, h, quality, color_type, subsampling, restart_interval, data) in jpeg_case_strategy()
    ) {
        let mut opts = JpegOptions::fast(w, h, quality);
        opts.color_type = color_type;
        opts.subsampling = subsampling;
        opts.restart_interval = restart_interval;

        let encoded = jpeg::encode(&data, &opts).unwrap();

        if restart_interval.is_some() {
            prop_assert!(encoded.windows(2).any(|w| w == [0xFF, 0xDD]));
        }
        prop_assert!(encoded.ends_with(&[0xFF, 0xD9]));

        let decoded = image::load_from_memory(&encoded).expect("decode");
        prop_assert_eq!(decoded.dimensions(), (w, h));
    }
}

#[test]
fn jpeg_corpus_reencode_preserves_dimensions() {
    let Ok(cases) = read_jpeg_corpus() else {
        eprintln!("Skipping JPEG corpus test: fixtures unavailable (offline?)");
        return;
    };

    for (path, bytes) in cases {
        let img = image::load_from_memory(&bytes).expect("decode jpeg fixture");
        let rgb = img.to_rgb8();
        let (w, h) = img.dimensions();

        let encoded = encode_jpeg(rgb.as_raw(), w, h, 85).expect("encode jpeg");
        let decoded = image::load_from_memory(&encoded).expect("decode encoded");
        assert_eq!(
            decoded.dimensions(),
            (w, h),
            "dimension mismatch for {:?}",
            path
        );
    }
}

#[test]
fn jpeg_optimized_huffman_not_larger_on_structured_image() {
    let width = 16;
    let height = 16;
    let mut rgb = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            let v = ((x + y) % 256) as u8;
            rgb.extend_from_slice(&[v, v / 2, 255 - v]);
        }
    }

    let base_opts = test_jpeg_fast(width, height, 85);
    let opt_opts = test_jpeg_balanced(width, height, 85);

    let default_bytes =
        encode_jpeg_with_options(&rgb, width, height, ColorType::Rgb, &base_opts).unwrap();
    let optimized_bytes =
        encode_jpeg_with_options(&rgb, width, height, ColorType::Rgb, &opt_opts).unwrap();

    assert!(
        optimized_bytes.len() <= default_bytes.len(),
        "optimized Huffman larger than default ({} > {})",
        optimized_bytes.len(),
        default_bytes.len()
    );

    let dec_default = image::load_from_memory(&default_bytes).expect("decode default jpeg");
    let dec_opt = image::load_from_memory(&optimized_bytes).expect("decode optimized jpeg");
    assert_eq!(dec_default.dimensions(), (width, height));
    assert_eq!(dec_opt.dimensions(), (width, height));
}

#[test]
fn jpeg_dqt_marker_present() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let result = encode_jpeg(&pixels, 8, 8, 85).unwrap();

    let mut found_dqt = false;
    for i in 0..result.len() - 1 {
        if result[i] == 0xFF && result[i + 1] == 0xDB {
            found_dqt = true;
            break;
        }
    }
    assert!(found_dqt, "DQT marker not found");
}

#[test]
fn jpeg_sof0_marker_present() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let result = encode_jpeg(&pixels, 8, 8, 85).unwrap();

    let mut found_sof0 = false;
    for i in 0..result.len() - 1 {
        if result[i] == 0xFF && result[i + 1] == 0xC0 {
            found_sof0 = true;
            break;
        }
    }
    assert!(found_sof0, "SOF0 marker not found");
}

#[test]
fn jpeg_dht_marker_count_is_four() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let result = encode_jpeg(&pixels, 8, 8, 85).unwrap();

    let mut dht_count = 0;
    for i in 0..result.len() - 1 {
        if result[i] == 0xFF && result[i + 1] == 0xC4 {
            dht_count += 1;
        }
    }
    assert_eq!(dht_count, 4, "Expected 4 DHT markers, found {}", dht_count);
}

#[test]
fn jpeg_sos_marker_present() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let result = encode_jpeg(&pixels, 8, 8, 85).unwrap();

    let mut found_sos = false;
    for i in 0..result.len() - 1 {
        if result[i] == 0xFF && result[i + 1] == 0xDA {
            found_sos = true;
            break;
        }
    }
    assert!(found_sos, "SOS marker not found");
}

#[test]
fn jpeg_synthetic_patterns_roundtrip() {
    let test_suite = synthetic::generate_minimal_test_suite();

    for (name, w, h, pixels) in test_suite {
        let encoded = encode_jpeg(&pixels, w, h, 85)
            .unwrap_or_else(|e| panic!("Failed to encode {name}: {e}"));

        assert_eq!(&encoded[0..2], &[0xFF, 0xD8], "Missing SOI for {name}");
        assert_eq!(
            &encoded[encoded.len() - 2..],
            &[0xFF, 0xD9],
            "Missing EOI for {name}"
        );

        let decoded = image::load_from_memory(&encoded)
            .unwrap_or_else(|e| panic!("Failed to decode {name}: {e}"));
        assert_eq!(decoded.width(), w, "Width mismatch for {name}");
        assert_eq!(decoded.height(), h, "Height mismatch for {name}");
    }
}

#[test]
fn jpeg_edge_case_dimensions_roundtrip() {
    for &(w, h, name) in synthetic::EDGE_CASE_DIMENSIONS {
        if w > 512 || h > 512 {
            continue;
        }

        let pixels = synthetic::gradient_rgb(w, h);
        let encoded = encode_jpeg(&pixels, w, h, 85)
            .unwrap_or_else(|e| panic!("Failed to encode {name} ({w}x{h}): {e}"));

        let decoded = image::load_from_memory(&encoded)
            .unwrap_or_else(|e| panic!("Failed to decode {name} ({w}x{h}): {e}"));
        assert_eq!(decoded.width(), w, "Width mismatch for {name}");
        assert_eq!(decoded.height(), h, "Height mismatch for {name}");
    }
}

#[test]
fn jpeg_kodak_subset_dimensions_match() {
    let Ok(images) = read_kodak_decoded_subset(4) else {
        eprintln!("Skipping Kodak test: fixtures unavailable (offline?)");
        return;
    };

    for (name, w, h, pixels) in images {
        let opts = test_jpeg_balanced(w, h, 85);
        let encoded = encode_jpeg_with_options(&pixels, w, h, ColorType::Rgb, &opts)
            .unwrap_or_else(|e| panic!("Failed to encode Kodak {name}: {e}"));

        assert_eq!(
            &encoded[0..2],
            &[0xFF, 0xD8],
            "Missing SOI for Kodak {name}"
        );
        assert_eq!(
            &encoded[encoded.len() - 2..],
            &[0xFF, 0xD9],
            "Missing EOI for Kodak {name}"
        );

        let decoded = image::load_from_memory(&encoded)
            .unwrap_or_else(|e| panic!("Failed to decode Kodak {name}: {e}"));
        assert_eq!(decoded.width(), w, "Width mismatch for Kodak {name}");
        assert_eq!(decoded.height(), h, "Height mismatch for Kodak {name}");
    }
}

#[test]
fn jpeg_presets_on_gradient_produce_reasonable_sizes() {
    let (w, h) = (256, 256);
    let pixels = synthetic::gradient_rgb(w, h);

    let presets = [
        ("fast", test_jpeg_fast(w, h, 85)),
        ("balanced", test_jpeg_balanced(w, h, 85)),
    ];

    let mut sizes = Vec::new();

    for (name, opts) in presets {
        let encoded = encode_jpeg_with_options(&pixels, w, h, ColorType::Rgb, &opts)
            .unwrap_or_else(|e| panic!("Failed to encode with {name} preset: {e}"));

        let decoded = image::load_from_memory(&encoded)
            .unwrap_or_else(|e| panic!("Failed to decode {name} preset: {e}"));
        assert_eq!(decoded.dimensions(), (w, h));

        sizes.push((name, encoded.len()));
    }

    for (name, size) in &sizes {
        assert!(*size > 0, "{name} preset produced empty output");
        assert!(
            *size < (w * h * 3) as usize,
            "{name} preset output ({size} bytes) larger than raw pixels"
        );
    }
}
