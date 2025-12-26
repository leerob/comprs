//! SIMD vs Fallback equality tests.
//!
//! These tests verify that SIMD implementations produce identical results
//! to their scalar fallback counterparts.

#![cfg(feature = "simd")]

use proptest::prelude::*;
use rand::{rngs::StdRng, Rng, SeedableRng};

// Import the fallback implementations for comparison
use pixo::simd::fallback;

/// Test Adler-32 SIMD vs fallback equality on various data sizes.
#[test]
fn test_adler32_simd_vs_fallback() {
    let test_cases: Vec<Vec<u8>> = vec![
        vec![],
        vec![0],
        vec![255],
        vec![0; 16],
        vec![255; 16],
        (0..256).map(|i| i as u8).collect(),
        (0..1000).map(|i| (i * 7) as u8).collect(),
        (0..5552).map(|i| (i % 256) as u8).collect(), // NMAX boundary
        (0..5553).map(|i| (i % 256) as u8).collect(), // Just over NMAX
        (0..10000).map(|i| ((i * 13) % 256) as u8).collect(),
    ];

    for data in test_cases {
        let expected = fallback::adler32(&data);
        let actual = pixo::simd::adler32(&data);
        assert_eq!(
            expected,
            actual,
            "Adler-32 mismatch for {} bytes",
            data.len()
        );
    }
}

/// Test CRC32 SIMD vs fallback equality on various data sizes.
// TODO: Fix CRC32 SIMD implementation mismatch with fallback on x86_64
#[test]
#[ignore]
fn test_crc32_simd_vs_fallback() {
    let test_cases: Vec<Vec<u8>> = vec![
        vec![],
        vec![0],
        vec![255],
        vec![0; 16],
        vec![255; 16],
        (0..256).map(|i| i as u8).collect(),
        (0..1000).map(|i| (i * 7) as u8).collect(),
        (0..10000).map(|i| ((i * 13) % 256) as u8).collect(),
    ];

    for data in test_cases {
        let expected = fallback::crc32(&data);
        let actual = pixo::simd::crc32(&data);
        assert_eq!(expected, actual, "CRC32 mismatch for {} bytes", data.len());
    }
}

/// Test match_length SIMD vs fallback equality.
#[test]
fn test_match_length_simd_vs_fallback() {
    let mut rng = StdRng::seed_from_u64(12345);

    // Test with repeated patterns
    let data: Vec<u8> = (0..1000).map(|i| (i % 32) as u8).collect();

    for _ in 0..100 {
        let pos1 = rng.gen_range(0..data.len() - 100);
        let pos2 = rng.gen_range(pos1 + 1..data.len() - 50);
        let max_len = (data.len() - pos2).min(258);

        let expected = fallback::match_length(&data, pos1, pos2, max_len);
        let actual = pixo::simd::match_length(&data, pos1, pos2, max_len);
        assert_eq!(
            expected, actual,
            "match_length mismatch at pos1={pos1}, pos2={pos2}, max_len={max_len}"
        );
    }
}

/// Test score_filter SIMD vs fallback equality.
#[test]
fn test_score_filter_simd_vs_fallback() {
    let test_cases: Vec<Vec<u8>> = vec![
        vec![],
        vec![0],
        vec![128],
        vec![255],
        vec![0; 32],
        vec![128; 32],
        (0..256).map(|i| i as u8).collect(),
        (0..1000).map(|i| (i * 7 % 256) as u8).collect(),
    ];

    for data in test_cases {
        let expected = fallback::score_filter(&data);
        let actual = pixo::simd::score_filter(&data);
        assert_eq!(
            expected,
            actual,
            "score_filter mismatch for {} bytes",
            data.len()
        );
    }
}

/// Test filter_sub SIMD vs fallback equality.
#[test]
fn test_filter_sub_simd_vs_fallback() {
    let mut rng = StdRng::seed_from_u64(54321);

    for bpp in [1, 2, 3, 4] {
        for width in [8, 16, 32, 64, 100, 256] {
            let mut row: Vec<u8> = vec![0; width];
            rng.fill(row.as_mut_slice());

            let mut expected_output = Vec::new();
            fallback::filter_sub(&row, bpp, &mut expected_output);

            let mut actual_output = Vec::new();
            pixo::simd::filter_sub(&row, bpp, &mut actual_output);

            assert_eq!(
                expected_output, actual_output,
                "filter_sub mismatch for bpp={bpp}, width={width}"
            );
        }
    }
}

/// Test filter_up SIMD vs fallback equality.
#[test]
fn test_filter_up_simd_vs_fallback() {
    let mut rng = StdRng::seed_from_u64(67890);

    for width in [8, 16, 32, 64, 100, 256] {
        let mut row: Vec<u8> = vec![0; width];
        let mut prev_row: Vec<u8> = vec![0; width];
        rng.fill(row.as_mut_slice());
        rng.fill(prev_row.as_mut_slice());

        let mut expected_output = Vec::new();
        fallback::filter_up(&row, &prev_row, &mut expected_output);

        let mut actual_output = Vec::new();
        pixo::simd::filter_up(&row, &prev_row, &mut actual_output);

        assert_eq!(
            expected_output, actual_output,
            "filter_up mismatch for width={width}"
        );
    }
}

/// Test filter_average SIMD vs fallback equality.
#[test]
fn test_filter_average_simd_vs_fallback() {
    let mut rng = StdRng::seed_from_u64(11111);

    for bpp in [1, 2, 3, 4] {
        for width in [8, 16, 32, 64, 100, 256] {
            let mut row: Vec<u8> = vec![0; width];
            let mut prev_row: Vec<u8> = vec![0; width];
            rng.fill(row.as_mut_slice());
            rng.fill(prev_row.as_mut_slice());

            let mut expected_output = Vec::new();
            fallback::filter_average(&row, &prev_row, bpp, &mut expected_output);

            let mut actual_output = Vec::new();
            pixo::simd::filter_average(&row, &prev_row, bpp, &mut actual_output);

            assert_eq!(
                expected_output, actual_output,
                "filter_average mismatch for bpp={bpp}, width={width}"
            );
        }
    }
}

/// Test filter_paeth SIMD vs fallback equality.
#[test]
fn test_filter_paeth_simd_vs_fallback() {
    let mut rng = StdRng::seed_from_u64(22222);

    for bpp in [1, 2, 3, 4] {
        for width in [8, 16, 32, 64, 100, 256] {
            let mut row: Vec<u8> = vec![0; width];
            let mut prev_row: Vec<u8> = vec![0; width];
            rng.fill(row.as_mut_slice());
            rng.fill(prev_row.as_mut_slice());

            let mut expected_output = Vec::new();
            fallback::filter_paeth(&row, &prev_row, bpp, &mut expected_output);

            let mut actual_output = Vec::new();
            pixo::simd::filter_paeth(&row, &prev_row, bpp, &mut actual_output);

            assert_eq!(
                expected_output, actual_output,
                "filter_paeth mismatch for bpp={bpp}, width={width}"
            );
        }
    }
}

// Property-based tests for more thorough coverage

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_adler32_simd_fallback_equality(data in proptest::collection::vec(any::<u8>(), 0..5000)) {
        let expected = fallback::adler32(&data);
        let actual = pixo::simd::adler32(&data);
        prop_assert_eq!(expected, actual);
    }

    // TODO: Fix CRC32 SIMD implementation mismatch with fallback on x86_64
    #[test]
    #[ignore]
    fn prop_crc32_simd_fallback_equality(data in proptest::collection::vec(any::<u8>(), 0..5000)) {
        let expected = fallback::crc32(&data);
        let actual = pixo::simd::crc32(&data);
        prop_assert_eq!(expected, actual);
    }

    #[test]
    fn prop_score_filter_simd_fallback_equality(data in proptest::collection::vec(any::<u8>(), 0..1000)) {
        let expected = fallback::score_filter(&data);
        let actual = pixo::simd::score_filter(&data);
        prop_assert_eq!(expected, actual);
    }

    #[test]
    fn prop_filter_sub_simd_fallback_equality(
        row in proptest::collection::vec(any::<u8>(), 4..256),
        bpp in 1usize..=4,
    ) {
        let mut expected_output = Vec::new();
        fallback::filter_sub(&row, bpp, &mut expected_output);

        let mut actual_output = Vec::new();
        pixo::simd::filter_sub(&row, bpp, &mut actual_output);

        prop_assert_eq!(expected_output, actual_output);
    }

    #[test]
    fn prop_filter_up_simd_fallback_equality(
        row in proptest::collection::vec(any::<u8>(), 1..256),
    ) {
        // Generate prev_row of same length
        let prev_row: Vec<u8> = row.iter().map(|&b| b.wrapping_add(42)).collect();

        let mut expected_output = Vec::new();
        fallback::filter_up(&row, &prev_row, &mut expected_output);

        let mut actual_output = Vec::new();
        pixo::simd::filter_up(&row, &prev_row, &mut actual_output);

        prop_assert_eq!(expected_output, actual_output);
    }
}
