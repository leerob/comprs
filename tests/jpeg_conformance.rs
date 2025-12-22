//! JPEG conformance tests.
//!
//! Tests JPEG encoding for correctness and validates
//! that encoded images contain proper markers.

#![allow(
    clippy::uninlined_format_args,
    clippy::needless_range_loop,
    clippy::unnecessary_cast
)]

use comprs::{jpeg, ColorType};
use image::GenericImageView;
use proptest::prelude::*;
use rand::{rngs::StdRng, Rng, SeedableRng};
mod support;
use support::jpeg_corpus::read_jpeg_corpus;

/// Test that JPEG output has correct markers.
#[test]
fn test_jpeg_markers() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let result = jpeg::encode(&pixels, 8, 8, 85).unwrap();

    // SOI marker
    assert_eq!(&result[0..2], &[0xFF, 0xD8]);

    // EOI marker at end
    assert_eq!(&result[result.len() - 2..], &[0xFF, 0xD9]);
}

/// Test APP0 (JFIF) marker.
#[test]
fn test_app0_marker() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let result = jpeg::encode(&pixels, 8, 8, 85).unwrap();

    // APP0 should be right after SOI
    assert_eq!(&result[2..4], &[0xFF, 0xE0]);

    // JFIF identifier
    assert_eq!(&result[6..11], b"JFIF\0");
}

/// Test different quality levels.
#[test]
fn test_quality_levels() {
    let pixels: Vec<u8> = (0..64 * 64 * 3).map(|i| (i % 256) as u8).collect();

    let sizes: Vec<(u8, usize)> = [10, 25, 50, 75, 90, 100]
        .iter()
        .map(|&q| {
            let result = jpeg::encode(&pixels, 64, 64, q).unwrap();
            (q, result.len())
        })
        .collect();

    // Higher quality should produce larger files
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

/// Test different image sizes.
#[test]
fn test_various_sizes() {
    let sizes = [
        (1, 1),
        (7, 7),   // Not multiple of 8
        (8, 8),   // Exact MCU
        (9, 9),   // Just over one MCU
        (16, 16), // Two MCUs
        (100, 50),
        (50, 100),
    ];

    for (width, height) in sizes {
        let pixels = vec![128u8; (width * height * 3) as usize];
        let result = jpeg::encode(&pixels, width, height, 85);

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

/// Test grayscale encoding.
#[test]
fn test_grayscale() {
    let pixels = vec![128u8; 32 * 32];
    let result = jpeg::encode_with_color(&pixels, 32, 32, 85, ColorType::Gray).unwrap();

    // Should have proper markers
    assert_eq!(&result[0..2], &[0xFF, 0xD8]);
    assert_eq!(&result[result.len() - 2..], &[0xFF, 0xD9]);

    // Should be smaller than RGB (1 component vs 3)
    let rgb_pixels = vec![128u8; 32 * 32 * 3];
    let rgb_result = jpeg::encode(&rgb_pixels, 32, 32, 85).unwrap();
    assert!(result.len() < rgb_result.len());
}

/// Test error handling.
#[test]
fn test_error_handling() {
    // Invalid quality
    let pixels = vec![0u8; 8 * 8 * 3];
    assert!(jpeg::encode(&pixels, 8, 8, 0).is_err());
    assert!(jpeg::encode(&pixels, 8, 8, 101).is_err());

    // Invalid dimensions
    assert!(jpeg::encode(&pixels, 0, 8, 85).is_err());
    assert!(jpeg::encode(&pixels, 8, 0, 85).is_err());

    // Wrong data length
    assert!(jpeg::encode(&[0, 0], 8, 8, 85).is_err());
}

#[test]
fn test_invalid_restart_interval() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let opts = jpeg::JpegOptions {
        quality: 85,
        subsampling: jpeg::Subsampling::S444,
        restart_interval: Some(0),
    };
    let result = jpeg::encode_with_options(&pixels, 8, 8, 85, ColorType::Rgb, &opts);
    assert!(result.is_err());
}

#[test]
fn test_unsupported_color_type_rejected() {
    let pixels = vec![0u8; 4 * 4 * 4]; // RGBA data
    let result = jpeg::encode_with_color(&pixels, 4, 4, 85, ColorType::Rgba);
    assert!(result.is_err());
}

#[test]
fn test_image_too_large() {
    // Just over the MAX_DIMENSION (65535)
    let width = 65_536;
    let height = 1;
    let pixels = vec![0u8; (width as usize * height as usize * 3) as usize];
    let err = jpeg::encode(&pixels, width, height, 85).unwrap_err();
    assert!(matches!(err, comprs::Error::ImageTooLarge { .. }));
}

/// Test that encoding produces deterministic output.
#[test]
fn test_deterministic() {
    let pixels = vec![100u8; 16 * 16 * 3];

    let result1 = jpeg::encode(&pixels, 16, 16, 85).unwrap();
    let result2 = jpeg::encode(&pixels, 16, 16, 85).unwrap();

    assert_eq!(result1, result2);
}

/// Test different patterns compress differently.
#[test]
fn test_pattern_compression() {
    // Solid color (should compress very well)
    let solid = vec![128u8; 64 * 64 * 3];
    let solid_result = jpeg::encode(&solid, 64, 64, 85).unwrap();

    // Gradient (compresses reasonably)
    let mut gradient = Vec::with_capacity(64 * 64 * 3);
    for y in 0..64 {
        for x in 0..64 {
            gradient.push(((x * 4) % 256) as u8);
            gradient.push(((y * 4) % 256) as u8);
            gradient.push((((x + y) * 2) % 256) as u8);
        }
    }
    let gradient_result = jpeg::encode(&gradient, 64, 64, 85).unwrap();

    // Random-ish (compresses poorly)
    let mut noisy = Vec::with_capacity(64 * 64 * 3);
    let mut seed = 42u32;
    for _ in 0..(64 * 64 * 3) {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        noisy.push((seed >> 16) as u8);
    }
    let noisy_result = jpeg::encode(&noisy, 64, 64, 85).unwrap();

    // Solid should be smallest, noisy should be largest
    assert!(solid_result.len() < gradient_result.len());
    assert!(gradient_result.len() < noisy_result.len());
}

/// Ensure encoded JPEGs decode via `image` for RGB and Gray.
#[test]
fn test_jpeg_decode_via_image() {
    // RGB pattern
    let mut rgb = vec![0u8; 8 * 8 * 3];
    for i in 0..rgb.len() {
        rgb[i] = (i as u8).wrapping_mul(31);
    }
    let jpeg_rgb = jpeg::encode(&rgb, 8, 8, 85).unwrap();
    let decoded_rgb = image::load_from_memory(&jpeg_rgb).expect("decode rgb");
    assert_eq!(decoded_rgb.width(), 8);
    assert_eq!(decoded_rgb.height(), 8);

    // Grayscale random
    let mut rng = StdRng::seed_from_u64(1337);
    let mut gray = vec![0u8; 7 * 5];
    rng.fill(gray.as_mut_slice());
    let jpeg_gray = jpeg::encode_with_color(&gray, 7, 5, 75, ColorType::Gray).unwrap();
    let decoded_gray = image::load_from_memory(&jpeg_gray).expect("decode gray");
    assert_eq!(decoded_gray.width(), 7);
    assert_eq!(decoded_gray.height(), 5);
}

/// Randomized small-image decode across RGB/Gray and multiple qualities.
#[test]
fn test_jpeg_decode_random_small() {
    let mut rng = StdRng::seed_from_u64(2025);
    let dims = [(1, 1), (2, 3), (5, 4), (8, 8), (16, 9)];
    let qualities = [50u8, 85u8, 95u8];

    for &(w, h) in &dims {
        // RGB
        let mut rgb = vec![0u8; w * h * 3];
        rng.fill(rgb.as_mut_slice());
        for &q in &qualities {
            let jpeg_rgb = jpeg::encode(&rgb, w as u32, h as u32, q).unwrap();
            let decoded = image::load_from_memory(&jpeg_rgb).expect("decode rgb");
            assert_eq!(decoded.width(), w as u32);
            assert_eq!(decoded.height(), h as u32);
        }

        // Grayscale
        let mut gray = vec![0u8; w * h];
        rng.fill(gray.as_mut_slice());
        for &q in &qualities {
            let jpeg_gray =
                jpeg::encode_with_color(&gray, w as u32, h as u32, q, ColorType::Gray).unwrap();
            let decoded = image::load_from_memory(&jpeg_gray).expect("decode gray");
            assert_eq!(decoded.width(), w as u32);
            assert_eq!(decoded.height(), h as u32);
        }
    }
}

/// Subsampling 4:2:0 should produce valid JPEG and smaller size.
#[test]
fn test_jpeg_subsampling_420() {
    let width = 32;
    let height = 32;
    let mut rng = StdRng::seed_from_u64(4242);
    let mut rgb = vec![0u8; (width * height * 3) as usize];
    rng.fill(rgb.as_mut_slice());

    let opts_444 = jpeg::JpegOptions {
        quality: 75,
        subsampling: jpeg::Subsampling::S444,
        restart_interval: None,
    };
    let opts_420 = jpeg::JpegOptions {
        quality: 75,
        subsampling: jpeg::Subsampling::S420,
        restart_interval: None,
    };

    let jpeg_444 =
        jpeg::encode_with_options(&rgb, width, height, 75, ColorType::Rgb, &opts_444).unwrap();
    let jpeg_420 =
        jpeg::encode_with_options(&rgb, width, height, 75, ColorType::Rgb, &opts_420).unwrap();

    // 4:2:0 should not be larger than 4:4:4 for the same image/quality.
    assert!(jpeg_420.len() <= jpeg_444.len());

    // Decode and verify dimensions
    let decoded = image::load_from_memory(&jpeg_420).expect("decode 420");
    assert_eq!(decoded.dimensions(), (width, height));
}

/// Restart interval should emit DRI and decode successfully.
#[test]
fn test_jpeg_restart_interval_marker_and_decode() {
    let width = 16;
    let height = 16;
    let mut rng = StdRng::seed_from_u64(5151);
    let mut rgb = vec![0u8; (width * height * 3) as usize];
    rng.fill(rgb.as_mut_slice());

    let opts = jpeg::JpegOptions {
        quality: 80,
        subsampling: jpeg::Subsampling::S444,
        restart_interval: Some(4),
    };

    let jpeg_bytes =
        jpeg::encode_with_options(&rgb, width, height, 80, ColorType::Rgb, &opts).unwrap();

    // Ensure DRI marker (0xFFDD) exists
    let mut found_dri = false;
    for w in jpeg_bytes.windows(2) {
        if w == [0xFF, 0xDD] {
            found_dri = true;
            break;
        }
    }
    assert!(found_dri, "DRI marker not found");

    // Decode to verify validity
    let decoded = image::load_from_memory(&jpeg_bytes).expect("decode with restart interval");
    assert_eq!(decoded.dimensions(), (width, height));
}

/// Structural marker walk to ensure required segments and restart interval are present.
#[test]
fn test_jpeg_marker_structure_with_restart() {
    let width = 16;
    let height = 12;
    let mut rng = StdRng::seed_from_u64(6262);
    let mut rgb = vec![0u8; (width * height * 3) as usize];
    rng.fill(rgb.as_mut_slice());

    let opts = jpeg::JpegOptions {
        quality: 85,
        subsampling: jpeg::Subsampling::S420,
        restart_interval: Some(4),
    };

    let jpeg_bytes =
        jpeg::encode_with_options(&rgb, width, height, 85, ColorType::Rgb, &opts).unwrap();

    assert!(jpeg_bytes.starts_with(&[0xFF, 0xD8]), "missing SOI");
    assert!(jpeg_bytes.ends_with(&[0xFF, 0xD9]), "missing EOI");

    let mut offset = 2; // after SOI
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
            break; // EOI (no length)
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
            0xE0 => saw_app0 = true, // APP0
            0xDB => saw_dqt = true,  // DQT
            0xC0 => saw_sof0 = true, // SOF0
            0xC4 => saw_dht = true,  // DHT
            0xDD => saw_dri = true,  // DRI
            0xDA => {
                saw_sos = true; // SOS
                break; // after SOS, entropy-coded data continues until EOI
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

/// Ensure DRI is absent when restart intervals are disabled.
#[test]
fn test_jpeg_no_restart_marker_without_interval() {
    let width = 12;
    let height = 9;
    let mut rng = StdRng::seed_from_u64(7373);
    let mut rgb = vec![0u8; (width * height * 3) as usize];
    rng.fill(rgb.as_mut_slice());

    let opts = jpeg::JpegOptions {
        quality: 80,
        subsampling: jpeg::Subsampling::S444,
        restart_interval: None,
    };

    let jpeg_bytes =
        jpeg::encode_with_options(&rgb, width, height, 80, ColorType::Rgb, &opts).unwrap();

    assert!(
        !jpeg_bytes.windows(2).any(|w| w == [0xFF, 0xDD]),
        "Unexpected DRI marker when restart_interval is None"
    );
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
        let opts = jpeg::JpegOptions {
            quality,
            subsampling,
            restart_interval,
        };

        let encoded =
            jpeg::encode_with_options(&data, w, h, quality, color_type, &opts).unwrap();

        if restart_interval.is_some() {
            prop_assert!(encoded.windows(2).any(|w| w == [0xFF, 0xDD]));
        }
        prop_assert!(encoded.ends_with(&[0xFF, 0xD9]));

        let decoded = image::load_from_memory(&encoded).expect("decode");
        prop_assert_eq!(decoded.dimensions(), (w, h));
    }
}

/// Conformance: re-encode curated JPEG corpus and ensure decode succeeds.
#[test]
fn test_jpeg_corpus_reencode_decode() {
    let Ok(cases) = read_jpeg_corpus() else {
        eprintln!("Skipping JPEG corpus test: fixtures unavailable (offline?)");
        return;
    };

    for (path, bytes) in cases {
        let img = image::load_from_memory(&bytes).expect("decode jpeg fixture");
        let rgb = img.to_rgb8();
        let (w, h) = img.dimensions();

        let encoded = jpeg::encode(rgb.as_raw(), w, h, 85).expect("encode jpeg");
        let decoded = image::load_from_memory(&encoded).expect("decode encoded");
        assert_eq!(
            decoded.dimensions(),
            (w, h),
            "dimension mismatch for {:?}",
            path
        );
    }
}

/// Test that DQT tables are present.
#[test]
fn test_dqt_present() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let result = jpeg::encode(&pixels, 8, 8, 85).unwrap();

    // Look for DQT marker (0xFFDB)
    let mut found_dqt = false;
    for i in 0..result.len() - 1 {
        if result[i] == 0xFF && result[i + 1] == 0xDB {
            found_dqt = true;
            break;
        }
    }
    assert!(found_dqt, "DQT marker not found");
}

/// Test that SOF0 marker is present.
#[test]
fn test_sof0_present() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let result = jpeg::encode(&pixels, 8, 8, 85).unwrap();

    // Look for SOF0 marker (0xFFC0)
    let mut found_sof0 = false;
    for i in 0..result.len() - 1 {
        if result[i] == 0xFF && result[i + 1] == 0xC0 {
            found_sof0 = true;
            break;
        }
    }
    assert!(found_sof0, "SOF0 marker not found");
}

/// Test that DHT markers are present.
#[test]
fn test_dht_present() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let result = jpeg::encode(&pixels, 8, 8, 85).unwrap();

    // Count DHT markers (0xFFC4) - should have 4 (DC lum, DC chrom, AC lum, AC chrom)
    let mut dht_count = 0;
    for i in 0..result.len() - 1 {
        if result[i] == 0xFF && result[i + 1] == 0xC4 {
            dht_count += 1;
        }
    }
    assert_eq!(dht_count, 4, "Expected 4 DHT markers, found {}", dht_count);
}

/// Test that SOS marker is present.
#[test]
fn test_sos_present() {
    let pixels = vec![128u8; 8 * 8 * 3];
    let result = jpeg::encode(&pixels, 8, 8, 85).unwrap();

    // Look for SOS marker (0xFFDA)
    let mut found_sos = false;
    for i in 0..result.len() - 1 {
        if result[i] == 0xFF && result[i + 1] == 0xDA {
            found_sos = true;
            break;
        }
    }
    assert!(found_sos, "SOS marker not found");
}
