//! PNG filtering implementation.
//!
//! PNG uses filtering to improve compression by exploiting correlations
//! between adjacent pixels.

use super::{FilterStrategy, PngOptions};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "simd")]
use crate::simd;

/// Scratch buffers reused for adaptive filtering to reduce per-row allocations.
struct AdaptiveScratch {
    none: Vec<u8>,
    sub: Vec<u8>,
    up: Vec<u8>,
    avg: Vec<u8>,
    paeth: Vec<u8>,
}

impl AdaptiveScratch {
    fn new(row_len: usize) -> Self {
        Self {
            none: Vec::with_capacity(row_len),
            sub: Vec::with_capacity(row_len),
            up: Vec::with_capacity(row_len),
            avg: Vec::with_capacity(row_len),
            paeth: Vec::with_capacity(row_len),
        }
    }

    fn clear(&mut self) {
        self.none.clear();
        self.sub.clear();
        self.up.clear();
        self.avg.clear();
        self.paeth.clear();
    }
}

/// Filter type bytes as defined by PNG specification.
const FILTER_NONE: u8 = 0;
const FILTER_SUB: u8 = 1;
const FILTER_UP: u8 = 2;
const FILTER_AVERAGE: u8 = 3;
const FILTER_PAETH: u8 = 4;

/// Apply PNG filtering to raw image data.
///
/// Returns filtered data with a filter type byte prepended to each row.
pub fn apply_filters(
    data: &[u8],
    width: u32,
    height: u32,
    bytes_per_pixel: usize,
    options: &PngOptions,
) -> Vec<u8> {
    let row_bytes = width as usize * bytes_per_pixel;
    let filtered_row_size = row_bytes + 1; // +1 for filter type byte
    let zero_row = vec![0u8; row_bytes];

    // Height-aware strategy tweaks.
    let mut strategy = options.filter_strategy;
    let area = (width as usize).saturating_mul(height as usize);
    // For very small images, prefer Sub filter to minimize CPU overhead.
    if area <= 4096
        && matches!(
            strategy,
            FilterStrategy::Adaptive | FilterStrategy::AdaptiveFast
        )
    {
        strategy = FilterStrategy::Sub;
    }

    // Note: High-entropy row detection was previously here to skip adaptive
    // filtering for noisy data. However, checking only the first row is not
    // sufficient to determine the optimal strategy for the entire image.
    // For now, we rely on per-row adaptive decisions instead.

    // Parallel path (only for adaptive; other strategies are trivial)
    #[cfg(feature = "parallel")]
    {
        // Parallel gains when rows are numerous; avoid overhead on tiny images.
        if height > 32
            && matches!(
                strategy,
                FilterStrategy::Adaptive | FilterStrategy::AdaptiveFast
            )
        {
            return apply_filters_parallel(
                data,
                height as usize,
                row_bytes,
                bytes_per_pixel,
                filtered_row_size,
                strategy,
            );
        }
    }

    // Sequential path
    let mut output = Vec::with_capacity(filtered_row_size * height as usize);
    let mut prev_row: &[u8] = &zero_row;
    let mut adaptive_scratch = AdaptiveScratch::new(row_bytes);
    let mut last_filter: u8 = FILTER_PAETH; // default guess for sampled reuse
                                            // Track last used filter to bias adaptive_fast toward recent winner.
    let mut last_adaptive_filter: Option<u8> = None;
    let mut filter_counts = [0usize; 5];

    for y in 0..height as usize {
        let row_start = y * row_bytes;
        let end = (row_start + row_bytes).min(data.len());
        let row = &data[row_start..end];
        match strategy {
            FilterStrategy::MinSum => {
                minsum_filter(
                    row,
                    if y == 0 { &zero_row[..] } else { prev_row },
                    bytes_per_pixel,
                    &mut output,
                    &mut adaptive_scratch,
                );
                if let Some(&f) = output.last() {
                    last_filter = f;
                }
            }
            FilterStrategy::AdaptiveFast => {
                let base = output.len();
                filter_row(
                    row,
                    if y == 0 { &zero_row[..] } else { prev_row },
                    bytes_per_pixel,
                    // Bias adaptive fast toward the previous winning filter.
                    match last_adaptive_filter {
                        Some(FILTER_SUB) => FilterStrategy::Sub,
                        Some(FILTER_UP) => FilterStrategy::Up,
                        Some(FILTER_PAETH) => FilterStrategy::Paeth,
                        _ => FilterStrategy::AdaptiveFast,
                    },
                    &mut output,
                    &mut adaptive_scratch,
                );
                if let Some(&f) = output.get(base) {
                    last_filter = f;
                    last_adaptive_filter = Some(f);
                }
            }
            _ => {
                let base = output.len();
                filter_row(
                    row,
                    if y == 0 { &zero_row[..] } else { prev_row },
                    bytes_per_pixel,
                    strategy,
                    &mut output,
                    &mut adaptive_scratch,
                );
                if let Some(&f) = output.get(base) {
                    last_filter = f;
                }
            }
        }

        // Update previous row reference
        prev_row = row;

        if options.verbose_filter_log && last_filter <= FILTER_PAETH {
            filter_counts[last_filter as usize] += 1;
        }
    }

    if options.verbose_filter_log {
        eprintln!(
            "PNG filters: strategy={:?}, rows={} counts={{None:{}, Sub:{}, Up:{}, Avg:{}, Paeth:{}}}",
            strategy,
            height,
            filter_counts[0],
            filter_counts[1],
            filter_counts[2],
            filter_counts[3],
            filter_counts[4]
        );
    }

    output
}

/// Sub filter: difference from left pixel.
fn filter_sub(row: &[u8], bpp: usize, output: &mut Vec<u8>) {
    #[cfg(feature = "simd")]
    {
        simd::filter_sub(row, bpp, output);
    }

    #[cfg(not(feature = "simd"))]
    {
        for (i, &byte) in row.iter().enumerate() {
            let left = if i >= bpp { row[i - bpp] } else { 0 };
            output.push(byte.wrapping_sub(left));
        }
    }
}

/// Up filter: difference from above pixel.
fn filter_up(row: &[u8], prev_row: &[u8], output: &mut Vec<u8>) {
    #[cfg(feature = "simd")]
    {
        simd::filter_up(row, prev_row, output);
    }

    #[cfg(not(feature = "simd"))]
    {
        for (i, &byte) in row.iter().enumerate() {
            output.push(byte.wrapping_sub(prev_row[i]));
        }
    }
}

/// Average filter: difference from average of left and above.
fn filter_average(row: &[u8], prev_row: &[u8], bpp: usize, output: &mut Vec<u8>) {
    #[cfg(feature = "simd")]
    {
        simd::filter_average(row, prev_row, bpp, output);
    }

    #[cfg(not(feature = "simd"))]
    {
        for (i, &byte) in row.iter().enumerate() {
            let left = if i >= bpp { row[i - bpp] as u16 } else { 0 };
            let above = prev_row[i] as u16;
            let avg = ((left + above) / 2) as u8;
            output.push(byte.wrapping_sub(avg));
        }
    }
}

/// Paeth filter: difference from Paeth predictor.
fn filter_paeth(row: &[u8], prev_row: &[u8], bpp: usize, output: &mut Vec<u8>) {
    #[cfg(feature = "simd")]
    {
        simd::filter_paeth(row, prev_row, bpp, output);
    }

    #[cfg(not(feature = "simd"))]
    {
        for (i, &byte) in row.iter().enumerate() {
            let left = if i >= bpp { row[i - bpp] } else { 0 };
            let above = prev_row[i];
            let upper_left = if i >= bpp { prev_row[i - bpp] } else { 0 };
            let predicted = paeth_predictor(left, above, upper_left);
            output.push(byte.wrapping_sub(predicted));
        }
    }
}

/// Paeth predictor function.
///
/// Selects the value (a, b, or c) closest to p = a + b - c.
#[allow(dead_code)]
#[inline]
fn paeth_predictor(a: u8, b: u8, c: u8) -> u8 {
    let a_i = a as i16;
    let b_i = b as i16;
    let c_i = c as i16;

    let p = a_i + b_i - c_i;
    let pa = (p - a_i).abs();
    let pb = (p - b_i).abs();
    let pc = (p - c_i).abs();

    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// Adaptive filter selection: try all filters and pick the best.
/// Optimized to track best score incrementally and potentially short-circuit.
fn adaptive_filter(
    row: &[u8],
    prev_row: &[u8],
    bpp: usize,
    output: &mut Vec<u8>,
    scratch: &mut AdaptiveScratch,
) {
    scratch.clear();

    let mut best_filter = FILTER_NONE;
    let mut best_score = u64::MAX;
    // Early-stop threshold: if a candidate beats this, skip remaining filters.
    // Bias toward speed: allow earlier exit.
    let early_stop = (row.len() as u64 / 4).saturating_add(1);

    // Try None filter first
    scratch.none.extend_from_slice(row);
    let score = score_filter(&scratch.none);
    if score < best_score {
        best_score = score;
        best_filter = FILTER_NONE;
        if best_score <= early_stop {
            output.push(best_filter);
            output.extend_from_slice(&scratch.none);
            return;
        }
    }

    // A score of 0 means all zeros - can't do better
    if best_score == 0 {
        output.push(best_filter);
        output.extend_from_slice(&scratch.none);
        return;
    }

    // Try Sub filter
    filter_sub(row, bpp, &mut scratch.sub);
    let score = score_filter(&scratch.sub);
    if score < best_score {
        best_score = score;
        best_filter = FILTER_SUB;
        if best_score == 0 || best_score <= early_stop {
            output.push(best_filter);
            output.extend_from_slice(&scratch.sub);
            return;
        }
    }

    // Try Up filter
    filter_up(row, prev_row, &mut scratch.up);
    let score = score_filter(&scratch.up);
    if score < best_score {
        best_score = score;
        best_filter = FILTER_UP;
        if best_score == 0 || best_score <= early_stop {
            output.push(best_filter);
            output.extend_from_slice(&scratch.up);
            return;
        }
    }

    // Try Average filter
    filter_average(row, prev_row, bpp, &mut scratch.avg);
    let score = score_filter(&scratch.avg);
    if score < best_score {
        best_score = score;
        best_filter = FILTER_AVERAGE;
        if best_score == 0 || best_score <= early_stop {
            output.push(best_filter);
            output.extend_from_slice(&scratch.avg);
            return;
        }
    }

    // Try Paeth filter
    filter_paeth(row, prev_row, bpp, &mut scratch.paeth);
    let score = score_filter(&scratch.paeth);
    if score < best_score {
        best_filter = FILTER_PAETH;
    }

    // Output the best filter result
    output.push(best_filter);
    match best_filter {
        FILTER_NONE => output.extend_from_slice(&scratch.none),
        FILTER_SUB => output.extend_from_slice(&scratch.sub),
        FILTER_UP => output.extend_from_slice(&scratch.up),
        FILTER_AVERAGE => output.extend_from_slice(&scratch.avg),
        FILTER_PAETH => output.extend_from_slice(&scratch.paeth),
        _ => unreachable!(),
    }
}

/// Min-sum filter selection (alias of adaptive using sum of absolute values).
fn minsum_filter(
    row: &[u8],
    prev_row: &[u8],
    bpp: usize,
    output: &mut Vec<u8>,
    scratch: &mut AdaptiveScratch,
) {
    adaptive_filter(row, prev_row, bpp, output, scratch);
}

/// Adaptive filtering with a faster heuristic and early cutoffs.
fn adaptive_filter_fast(
    row: &[u8],
    prev_row: &[u8],
    bpp: usize,
    output: &mut Vec<u8>,
    scratch: &mut AdaptiveScratch,
) {
    scratch.clear();

    // Try Sub first (good for high-frequency data)
    filter_sub(row, bpp, &mut scratch.sub);
    let mut best_filter = FILTER_SUB;
    let mut best_score = score_filter(&scratch.sub);

    // Early stop threshold: very low score, stop immediately. Bias toward speed.
    let early_stop = (row.len() as u64 / 8).saturating_add(1);
    if best_score <= early_stop {
        output.push(best_filter);
        output.extend_from_slice(&scratch.sub);
        return;
    }

    // Try Up (good for smooth gradients)
    filter_up(row, prev_row, &mut scratch.up);
    let up_score = score_filter(&scratch.up);
    if up_score < best_score {
        best_score = up_score;
        best_filter = FILTER_UP;
    }
    if best_score <= early_stop {
        output.push(best_filter);
        match best_filter {
            FILTER_SUB => output.extend_from_slice(&scratch.sub),
            FILTER_UP => output.extend_from_slice(&scratch.up),
            _ => {}
        }
        return;
    }

    // Try Paeth last (more expensive)
    filter_paeth(row, prev_row, bpp, &mut scratch.paeth);
    let paeth_score = score_filter(&scratch.paeth);
    if paeth_score < best_score {
        best_filter = FILTER_PAETH;
    }

    output.push(best_filter);
    match best_filter {
        FILTER_SUB => output.extend_from_slice(&scratch.sub),
        FILTER_UP => output.extend_from_slice(&scratch.up),
        FILTER_PAETH => output.extend_from_slice(&scratch.paeth),
        _ => unreachable!(),
    }
}

fn filter_row(
    row: &[u8],
    prev_row: &[u8],
    bpp: usize,
    strategy: FilterStrategy,
    output: &mut Vec<u8>,
    scratch: &mut AdaptiveScratch,
) {
    match strategy {
        FilterStrategy::None => {
            output.push(FILTER_NONE);
            output.extend_from_slice(row);
        }
        FilterStrategy::Sub => {
            output.push(FILTER_SUB);
            filter_sub(row, bpp, output);
        }
        FilterStrategy::Up => {
            output.push(FILTER_UP);
            filter_up(row, prev_row, output);
        }
        FilterStrategy::Average => {
            output.push(FILTER_AVERAGE);
            filter_average(row, prev_row, bpp, output);
        }
        FilterStrategy::Paeth => {
            output.push(FILTER_PAETH);
            filter_paeth(row, prev_row, bpp, output);
        }
        FilterStrategy::MinSum => {
            minsum_filter(row, prev_row, bpp, output, scratch);
        }
        FilterStrategy::Adaptive => {
            adaptive_filter(row, prev_row, bpp, output, scratch);
        }
        FilterStrategy::AdaptiveFast => {
            adaptive_filter_fast(row, prev_row, bpp, output, scratch);
        }
    }
}

#[cfg(feature = "parallel")]
fn apply_filters_parallel(
    data: &[u8],
    height: usize,
    row_bytes: usize,
    bpp: usize,
    filtered_row_size: usize,
    strategy: FilterStrategy,
) -> Vec<u8> {
    let zero_row = vec![0u8; row_bytes];
    let mut output = vec![0u8; filtered_row_size * height];

    output
        .par_chunks_mut(filtered_row_size)
        .enumerate()
        .for_each(|(y, out_row)| {
            let row_start = y * row_bytes;
            let row = &data[row_start..row_start + row_bytes];
            let prev = if y == 0 {
                &zero_row[..]
            } else {
                &data[(y - 1) * row_bytes..y * row_bytes]
            };
            let mut scratch = AdaptiveScratch::new(row_bytes);
            let mut row_buf = Vec::with_capacity(filtered_row_size);
            filter_row(row, prev, bpp, strategy, &mut row_buf, &mut scratch);
            debug_assert_eq!(
                row_buf.len(),
                filtered_row_size,
                "filtered row size mismatch"
            );
            out_row.copy_from_slice(&row_buf);
        });

    output
}

/// Score a filtered row using sum of absolute values.
///
/// Lower scores typically result in better compression.
#[inline]
fn score_filter(filtered: &[u8]) -> u64 {
    #[cfg(feature = "simd")]
    {
        simd::score_filter(filtered)
    }

    #[cfg(not(feature = "simd"))]
    {
        filtered
            .iter()
            .map(|&b| (b as i8).unsigned_abs() as u64)
            .sum()
    }
}

/// Simple high-entropy detector:
/// - Fewer than 1% of neighboring bytes are equal (no runs)
/// - The most common delta between neighbors accounts for <10% of positions
///
/// This avoids misclassifying smooth gradients (constant delta).
/// Guarded to rows >= 1024 bytes to avoid noise.
#[allow(dead_code)]
fn is_high_entropy_row(row: &[u8]) -> bool {
    if row.len() < 1024 {
        return false;
    }
    let mut equal_neighbors = 0usize;
    let mut delta_hist = [0u32; 256];
    let mut total_deltas = 0usize;
    for w in row.windows(2) {
        if w[0] == w[1] {
            equal_neighbors += 1;
        }
        let delta = w[1].wrapping_sub(w[0]);
        delta_hist[delta as usize] += 1;
        total_deltas += 1;
    }
    let ratio = equal_neighbors as f32 / (row.len().saturating_sub(1) as f32);
    let max_delta = delta_hist.iter().copied().max().unwrap_or(0);
    let max_delta_ratio = if total_deltas == 0 {
        1.0
    } else {
        max_delta as f32 / total_deltas as f32
    };
    ratio < 0.01 && max_delta_ratio < 0.10
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paeth_predictor() {
        // When all are equal, should return that value
        assert_eq!(paeth_predictor(100, 100, 100), 100);

        // When a=0, b=0, c=0, should return 0
        assert_eq!(paeth_predictor(0, 0, 0), 0);

        // Test typical case: a=10, b=20, c=15
        // p = a + b - c = 10 + 20 - 15 = 15
        // pa = |p - a| = |15 - 10| = 5
        // pb = |p - b| = |15 - 20| = 5
        // pc = |p - c| = |15 - 15| = 0
        // pc is smallest, so return c (15)
        assert_eq!(paeth_predictor(10, 20, 15), 15);
    }

    #[test]
    fn test_filter_sub() {
        let row = vec![10, 20, 30, 40, 50, 60];
        let mut output = Vec::new();
        filter_sub(&row, 3, &mut output);

        // First 3 bytes: no left pixel, so unchanged
        assert_eq!(output[0], 10);
        assert_eq!(output[1], 20);
        assert_eq!(output[2], 30);

        // Next 3 bytes: difference from 3 bytes back
        assert_eq!(output[3], 40u8.wrapping_sub(10)); // 30
        assert_eq!(output[4], 50u8.wrapping_sub(20)); // 30
        assert_eq!(output[5], 60u8.wrapping_sub(30)); // 30
    }

    #[test]
    fn test_filter_up() {
        let row = vec![50, 60, 70];
        let prev = vec![10, 20, 30];
        let mut output = Vec::new();
        filter_up(&row, &prev, &mut output);

        assert_eq!(output[0], 40); // 50 - 10
        assert_eq!(output[1], 40); // 60 - 20
        assert_eq!(output[2], 40); // 70 - 30
    }

    #[test]
    fn test_apply_filters_none() {
        let data = vec![100, 150, 200, 50, 100, 150];
        let options = PngOptions {
            filter_strategy: FilterStrategy::None,
            ..Default::default()
        };

        let filtered = apply_filters(&data, 2, 1, 3, &options);

        // Should be filter byte (0) + original data
        assert_eq!(filtered[0], FILTER_NONE);
        assert_eq!(&filtered[1..], &data[..]);
    }

    #[test]
    fn test_apply_filters_multiple_rows() {
        let data = vec![
            10, 20, 30, 40, 50, 60, // Row 1
            70, 80, 90, 100, 110, 120, // Row 2
        ];
        let options = PngOptions {
            filter_strategy: FilterStrategy::None,
            ..Default::default()
        };

        let filtered = apply_filters(&data, 2, 2, 3, &options);

        // Should have 2 rows, each with filter byte
        assert_eq!(filtered.len(), 2 * (1 + 6)); // 2 rows * (1 filter + 6 data)
        assert_eq!(filtered[0], FILTER_NONE);
        assert_eq!(filtered[7], FILTER_NONE);
    }

    #[test]
    fn test_apply_filters_adaptive_fast() {
        let data = vec![
            10, 20, 30, 40, 50, 60, // Row 1
            70, 80, 90, 100, 110, 120, // Row 2
        ];
        let options = PngOptions {
            filter_strategy: FilterStrategy::AdaptiveFast,
            ..Default::default()
        };

        let filtered = apply_filters(&data, 2, 2, 3, &options);

        // Two rows, each 1 filter byte + 6 bytes
        assert_eq!(filtered.len(), 2 * (1 + 6));
        // Filter bytes should be one of the defined filters
        assert!(matches!(filtered[0], FILTER_SUB | FILTER_UP | FILTER_PAETH));
        assert!(matches!(filtered[7], FILTER_SUB | FILTER_UP | FILTER_PAETH));
    }

    #[test]
    fn test_filter_average() {
        let row = vec![100, 100, 100];
        let prev = vec![50, 50, 50];
        let mut output = Vec::new();
        filter_average(&row, &prev, 1, &mut output);

        // First byte: left=0, above=50, avg=25
        assert_eq!(output[0], 100u8.wrapping_sub(25)); // 75
                                                       // Second byte: left=100, above=50, avg=75
        assert_eq!(output[1], 100u8.wrapping_sub(75)); // 25
                                                       // Third byte: left=100, above=50, avg=75
        assert_eq!(output[2], 100u8.wrapping_sub(75)); // 25
    }

    #[test]
    fn test_filter_paeth() {
        let row = vec![100, 100, 100];
        let prev = vec![50, 50, 50];
        let mut output = Vec::new();
        filter_paeth(&row, &prev, 1, &mut output);

        // First byte: left=0, above=50, upper_left=0
        // p = 0 + 50 - 0 = 50
        // pa = |50-0| = 50, pb = |50-50| = 0, pc = |50-0| = 50
        // pb is smallest, return b=50
        assert_eq!(output[0], 100u8.wrapping_sub(50)); // 50

        // All output should be valid filtered values
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn test_score_filter_all_zeros() {
        let data = vec![0u8; 100];
        let score = score_filter(&data);
        assert_eq!(score, 0);
    }

    #[test]
    fn test_score_filter_high_values() {
        // 0x80 as i8 is -128, abs = 128
        let data = vec![0x80u8; 10];
        let score = score_filter(&data);
        assert_eq!(score, 128 * 10);
    }

    #[test]
    fn test_score_filter_mixed() {
        // Mix of positive and negative values (as i8)
        let data = vec![1, 0xFF, 2, 0xFE]; // 1, -1, 2, -2 as i8
        let score = score_filter(&data);
        // abs values: 1, 1, 2, 2 = 6
        assert_eq!(score, 6);
    }

    #[test]
    fn test_is_high_entropy_row_short() {
        // Short rows should not be considered high entropy
        let row = vec![0u8; 100];
        assert!(!is_high_entropy_row(&row));
    }

    #[test]
    fn test_is_high_entropy_row_uniform() {
        // Uniform data has many equal neighbors - not high entropy
        let row = vec![42u8; 2000];
        assert!(!is_high_entropy_row(&row));
    }

    #[test]
    fn test_is_high_entropy_row_gradient() {
        // Gradient has constant delta - not high entropy
        let row: Vec<u8> = (0..2000).map(|i| (i % 256) as u8).collect();
        assert!(!is_high_entropy_row(&row));
    }

    #[test]
    fn test_paeth_predictor_edge_cases() {
        // Edge case: a closest to p
        assert_eq!(paeth_predictor(100, 0, 0), 100);
        // Edge case: b closest to p
        assert_eq!(paeth_predictor(0, 100, 0), 100);
        // Edge case: c closest to p
        assert_eq!(paeth_predictor(100, 100, 100), 100);
        // Edge case: boundary values
        assert_eq!(paeth_predictor(255, 0, 0), 255);
        assert_eq!(paeth_predictor(0, 255, 0), 255);
    }

    #[test]
    fn test_paeth_predictor_tie_breaking() {
        // When pa == pb, a should be chosen
        // p = a + b - c, if pa <= pb and pa <= pc, return a
        // a=100, b=100, c=100: p=100, pa=0, pb=0, pc=0
        // pa <= pb is true, pa <= pc is true, return a
        assert_eq!(paeth_predictor(100, 100, 100), 100);

        // a=50, b=100, c=75: p = 50+100-75 = 75
        // pa = |75-50| = 25, pb = |75-100| = 25, pc = |75-75| = 0
        // pc is smallest, return c
        assert_eq!(paeth_predictor(50, 100, 75), 75);
    }

    #[test]
    fn test_adaptive_scratch_reuse() {
        let mut scratch = AdaptiveScratch::new(100);
        scratch.none.extend_from_slice(&[1, 2, 3]);
        scratch.sub.extend_from_slice(&[4, 5, 6]);

        assert_eq!(scratch.none.len(), 3);
        assert_eq!(scratch.sub.len(), 3);

        scratch.clear();

        assert_eq!(scratch.none.len(), 0);
        assert_eq!(scratch.sub.len(), 0);
    }

    #[test]
    fn test_filter_sub_bpp_variations() {
        // Test with different bytes per pixel values
        for bpp in 1..=4 {
            let row: Vec<u8> = (0..20).collect();
            let mut output = Vec::new();
            filter_sub(&row, bpp, &mut output);
            assert_eq!(output.len(), row.len());

            // First bpp bytes should equal original (no left reference)
            for i in 0..bpp {
                assert_eq!(output[i], row[i]);
            }
        }
    }

    #[test]
    fn test_filter_up_first_row() {
        // First row uses zero as previous row
        let row = vec![10, 20, 30, 40];
        let zero_row = vec![0u8; 4];
        let mut output = Vec::new();
        filter_up(&row, &zero_row, &mut output);

        // Should just be the original row values (minus zero)
        assert_eq!(output, row);
    }

    #[test]
    fn test_apply_filters_sub_strategy() {
        let data = vec![10, 20, 30, 40, 50, 60]; // Single row
        let options = PngOptions {
            filter_strategy: FilterStrategy::Sub,
            ..Default::default()
        };

        let filtered = apply_filters(&data, 2, 1, 3, &options);

        assert_eq!(filtered[0], FILTER_SUB);
        // Check the filtered values
        assert_eq!(filtered.len(), 1 + 6); // 1 filter byte + 6 data bytes
    }

    #[test]
    fn test_apply_filters_up_strategy() {
        let data = vec![
            10, 20, 30, // Row 1
            50, 60, 70, // Row 2
        ];
        let options = PngOptions {
            filter_strategy: FilterStrategy::Up,
            ..Default::default()
        };

        let filtered = apply_filters(&data, 1, 2, 3, &options);

        // Both rows should use Up filter
        assert_eq!(filtered[0], FILTER_UP);
        assert_eq!(filtered[4], FILTER_UP);
    }

    #[test]
    fn test_apply_filters_average_strategy() {
        let data = vec![100, 100, 100];
        let options = PngOptions {
            filter_strategy: FilterStrategy::Average,
            ..Default::default()
        };

        let filtered = apply_filters(&data, 1, 1, 3, &options);

        assert_eq!(filtered[0], FILTER_AVERAGE);
    }

    #[test]
    fn test_apply_filters_paeth_strategy() {
        let data = vec![100, 100, 100];
        let options = PngOptions {
            filter_strategy: FilterStrategy::Paeth,
            ..Default::default()
        };

        let filtered = apply_filters(&data, 1, 1, 3, &options);

        assert_eq!(filtered[0], FILTER_PAETH);
    }

    #[test]
    fn test_apply_filters_minsum_strategy() {
        let data = vec![0u8; 100]; // All zeros should favor None filter
        let options = PngOptions {
            filter_strategy: FilterStrategy::MinSum,
            ..Default::default()
        };

        let filtered = apply_filters(&data, 10, 1, 10, &options);

        // Should produce valid output
        assert_eq!(filtered.len(), 1 + 100); // 1 row: 1 filter byte + 100 data bytes
    }

    #[test]
    fn test_apply_filters_adaptive_strategy() {
        let data: Vec<u8> = (0..200).map(|i| (i % 256) as u8).collect();
        let options = PngOptions {
            filter_strategy: FilterStrategy::Adaptive,
            ..Default::default()
        };

        let filtered = apply_filters(&data, 10, 2, 10, &options);

        // Should produce valid output with filter bytes
        assert_eq!(filtered.len(), 2 * (1 + 100)); // 2 rows
    }

    #[test]
    fn test_filter_wrapping() {
        // Test that filters handle wrapping correctly
        let row = vec![5, 10, 15];
        let prev = vec![10, 20, 30];
        let mut output = Vec::new();

        filter_up(&row, &prev, &mut output);

        // 5 - 10 = -5, wraps to 251
        assert_eq!(output[0], 5u8.wrapping_sub(10));
        // 10 - 20 = -10, wraps to 246
        assert_eq!(output[1], 10u8.wrapping_sub(20));
    }

    #[test]
    fn test_small_image_uses_sub() {
        // Small images (area <= 4096) should use Sub instead of Adaptive
        let data = vec![0u8; 64 * 3]; // 64 pixels = area < 4096
        let options = PngOptions {
            filter_strategy: FilterStrategy::Adaptive,
            ..Default::default()
        };

        let filtered = apply_filters(&data, 8, 8, 3, &options);

        // Should use Sub filter for small images
        // (Filter byte is the first byte)
        assert_eq!(filtered[0], FILTER_SUB);
    }
}
