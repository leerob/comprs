#![cfg(feature = "simd")]

use proptest::prelude::*;
use rand::{rngs::StdRng, Rng, SeedableRng};

use pixo::simd::fallback;

#[test]
fn test_adler32_simd_vs_fallback() {
    let nmax = 5552;
    let test_cases: Vec<Vec<u8>> = vec![
        vec![],
        vec![0],
        vec![255],
        vec![0; 16],
        vec![255; 16],
        (0..256).map(|i| i as u8).collect(),
        (0..1000).map(|i| (i * 7) as u8).collect(),
        (0..nmax).map(|i| (i % 256) as u8).collect(),
        (0..nmax + 1).map(|i| (i % 256) as u8).collect(),
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

#[test]
fn test_match_length_simd_vs_fallback() {
    let mut rng = StdRng::seed_from_u64(12345);

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

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    #[test]
    fn prop_adler32_simd_fallback_equality(data in proptest::collection::vec(any::<u8>(), 0..5000)) {
        let expected = fallback::adler32(&data);
        let actual = pixo::simd::adler32(&data);
        prop_assert_eq!(expected, actual);
    }

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
        let prev_row: Vec<u8> = row.iter().map(|&b| b.wrapping_add(42)).collect();

        let mut expected_output = Vec::new();
        fallback::filter_up(&row, &prev_row, &mut expected_output);

        let mut actual_output = Vec::new();
        pixo::simd::filter_up(&row, &prev_row, &mut actual_output);

        prop_assert_eq!(expected_output, actual_output);
    }

    #[test]
    fn prop_filter_average_simd_fallback_equality(
        row in proptest::collection::vec(any::<u8>(), 4..256),
        bpp in 1usize..=4,
    ) {
        let prev_row: Vec<u8> = row.iter().map(|&b| b.wrapping_mul(3)).collect();

        let mut expected_output = Vec::new();
        fallback::filter_average(&row, &prev_row, bpp, &mut expected_output);

        let mut actual_output = Vec::new();
        pixo::simd::filter_average(&row, &prev_row, bpp, &mut actual_output);

        prop_assert_eq!(expected_output, actual_output);
    }

    #[test]
    fn prop_filter_paeth_simd_fallback_equality(
        row in proptest::collection::vec(any::<u8>(), 4..256),
        bpp in 1usize..=4,
    ) {
        let prev_row: Vec<u8> = row.iter().map(|&b| b.wrapping_add(17)).collect();

        let mut expected_output = Vec::new();
        fallback::filter_paeth(&row, &prev_row, bpp, &mut expected_output);

        let mut actual_output = Vec::new();
        pixo::simd::filter_paeth(&row, &prev_row, bpp, &mut actual_output);

        prop_assert_eq!(expected_output, actual_output);
    }

    #[test]
    fn prop_match_length_simd_fallback_equality(
        data in proptest::collection::vec(any::<u8>(), 100..500),
        pos1_offset in 0usize..50,
        pos2_offset in 51usize..90,
    ) {
        let pos1 = pos1_offset;
        let pos2 = pos2_offset.min(data.len().saturating_sub(10));
        if pos2 <= pos1 || pos2 >= data.len() {
            return Ok(());
        }
        let max_len = (data.len() - pos2).min(100);

        let expected = fallback::match_length(&data, pos1, pos2, max_len);
        let actual = pixo::simd::match_length(&data, pos1, pos2, max_len);
        prop_assert_eq!(expected, actual);
    }
}

#[test]
fn test_adler32_simd_large_data() {
    let test_sizes = [5552, 5553, 11104, 16656, 20000, 50000];

    for size in test_sizes {
        let data: Vec<u8> = (0..size).map(|i| (i * 17 % 256) as u8).collect();
        let expected = fallback::adler32(&data);
        let actual = pixo::simd::adler32(&data);
        assert_eq!(expected, actual, "Adler-32 mismatch for {size} bytes");
    }
}

#[test]
fn test_score_filter_simd_remainder_sizes() {
    let test_sizes = [1, 7, 15, 17, 31, 33, 47, 63, 65, 127, 129, 255];

    for size in test_sizes {
        let data: Vec<u8> = (0..size).map(|i| (i * 13) as u8).collect();
        let expected = fallback::score_filter(&data);
        let actual = pixo::simd::score_filter(&data);
        assert_eq!(expected, actual, "score_filter mismatch for {size} bytes");
    }
}

#[test]
fn test_filters_simd_large_rows() {
    let mut rng = StdRng::seed_from_u64(99999);

    let widths = [640, 800, 1024, 1280, 1920, 4096];

    for width in widths {
        let mut row: Vec<u8> = vec![0; width];
        let mut prev_row: Vec<u8> = vec![0; width];
        rng.fill(row.as_mut_slice());
        rng.fill(prev_row.as_mut_slice());

        for bpp in [1, 3, 4] {
            let mut expected = Vec::new();
            fallback::filter_sub(&row, bpp, &mut expected);
            let mut actual = Vec::new();
            pixo::simd::filter_sub(&row, bpp, &mut actual);
            assert_eq!(
                expected, actual,
                "filter_sub mismatch for width={width}, bpp={bpp}"
            );

            expected.clear();
            fallback::filter_up(&row, &prev_row, &mut expected);
            actual.clear();
            pixo::simd::filter_up(&row, &prev_row, &mut actual);
            assert_eq!(expected, actual, "filter_up mismatch for width={width}");

            expected.clear();
            fallback::filter_average(&row, &prev_row, bpp, &mut expected);
            actual.clear();
            pixo::simd::filter_average(&row, &prev_row, bpp, &mut actual);
            assert_eq!(
                expected, actual,
                "filter_average mismatch for width={width}, bpp={bpp}"
            );

            expected.clear();
            fallback::filter_paeth(&row, &prev_row, bpp, &mut expected);
            actual.clear();
            pixo::simd::filter_paeth(&row, &prev_row, bpp, &mut actual);
            assert_eq!(
                expected, actual,
                "filter_paeth mismatch for width={width}, bpp={bpp}"
            );
        }
    }
}

#[test]
fn test_match_length_simd_long_sequences() {
    let pattern: Vec<u8> = (0..32).map(|i| (i * 7) as u8).collect();
    let mut data = Vec::with_capacity(1024);
    for _ in 0..32 {
        data.extend_from_slice(&pattern);
    }

    for offset in [0, 32, 64, 128] {
        let pos1 = offset;
        let pos2 = offset + 32;
        if pos2 + 32 < data.len() {
            let expected = fallback::match_length(&data, pos1, pos2, 258);
            let actual = pixo::simd::match_length(&data, pos1, pos2, 258);
            assert_eq!(expected, actual, "match_length mismatch at offset={offset}");
        }
    }
}

#[test]
fn test_match_length_simd_boundaries() {
    let data: Vec<u8> = vec![42; 512];

    for max_len in [8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 128, 256] {
        let expected = fallback::match_length(&data, 0, 0, max_len);
        let actual = pixo::simd::match_length(&data, 0, 0, max_len);
        assert_eq!(
            expected, actual,
            "match_length mismatch for max_len={max_len}"
        );
        assert_eq!(expected, max_len);
    }
}

#[cfg(target_arch = "aarch64")]
#[test]
fn test_crc32_simd_vs_fallback_aarch64() {
    let test_cases: Vec<Vec<u8>> = vec![
        vec![],
        vec![0],
        vec![255],
        vec![0; 64],
        vec![255; 64],
        (0..256).map(|i| i as u8).collect(),
        (0..1000).map(|i| (i * 7) as u8).collect(),
    ];

    for data in test_cases {
        let expected = fallback::crc32(&data);
        let actual = pixo::simd::crc32(&data);
        assert_eq!(expected, actual, "CRC32 mismatch for {} bytes", data.len());
    }
}

#[test]
fn test_filters_simd_short_rows() {
    let short_rows = [vec![1], vec![1, 2], vec![1, 2, 3]];
    let prev_rows = [vec![10], vec![10, 20], vec![10, 20, 30]];

    for (row, prev_row) in short_rows.iter().zip(prev_rows.iter()) {
        for bpp in [1, 2, 3, 4] {
            let mut expected = Vec::new();
            fallback::filter_sub(row, bpp, &mut expected);
            let mut actual = Vec::new();
            pixo::simd::filter_sub(row, bpp, &mut actual);
            assert_eq!(expected, actual, "filter_sub short row mismatch");

            if row.len() == prev_row.len() {
                expected.clear();
                fallback::filter_up(row, prev_row, &mut expected);
                actual.clear();
                pixo::simd::filter_up(row, prev_row, &mut actual);
                assert_eq!(expected, actual, "filter_up short row mismatch");

                expected.clear();
                fallback::filter_average(row, prev_row, bpp, &mut expected);
                actual.clear();
                pixo::simd::filter_average(row, prev_row, bpp, &mut actual);
                assert_eq!(expected, actual, "filter_average short row mismatch");

                expected.clear();
                fallback::filter_paeth(row, prev_row, bpp, &mut expected);
                actual.clear();
                pixo::simd::filter_paeth(row, prev_row, bpp, &mut actual);
                assert_eq!(expected, actual, "filter_paeth short row mismatch");
            }
        }
    }
}

#[test]
fn test_score_filter_simd_extreme_values() {
    let zeros = vec![0u8; 1000];
    let expected = fallback::score_filter(&zeros);
    let actual = pixo::simd::score_filter(&zeros);
    assert_eq!(expected, actual);
    assert_eq!(expected, 0);

    let mid = vec![0x80u8; 1000];
    let expected = fallback::score_filter(&mid);
    let actual = pixo::simd::score_filter(&mid);
    assert_eq!(expected, actual);
    assert_eq!(expected, 128 * 1000);

    let max = vec![0xFFu8; 1000];
    let expected = fallback::score_filter(&max);
    let actual = pixo::simd::score_filter(&max);
    assert_eq!(expected, actual);
    assert_eq!(expected, 1000);
}
