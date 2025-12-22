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

    // Parallel path (only for adaptive; other strategies are trivial)
    #[cfg(feature = "parallel")]
    {
        if matches!(options.filter_strategy, FilterStrategy::Adaptive) && height > 1 {
            return apply_filters_parallel(
                data,
                height as usize,
                row_bytes,
                bytes_per_pixel,
                filtered_row_size,
                options.filter_strategy,
            );
        }
    }

    // Sequential path
    let mut output = Vec::with_capacity(filtered_row_size * height as usize);
    let mut prev_row = vec![0u8; row_bytes];
    let mut adaptive_scratch = AdaptiveScratch::new(row_bytes);
    let mut last_filter: u8 = FILTER_PAETH; // default guess for sampled reuse

    for y in 0..height as usize {
        let row_start = y * row_bytes;
        let row = &data[row_start..row_start + row_bytes];
        match options.filter_strategy {
            FilterStrategy::AdaptiveSampled { interval } if interval > 1 => {
                let interval = interval.max(1) as usize;
                let prev = if y == 0 { &zero_row[..] } else { &prev_row[..] };
                if y % interval == 0 {
                    let base = output.len();
                    adaptive_filter(
                        row,
                        prev,
                        bytes_per_pixel,
                        &mut output,
                        &mut adaptive_scratch,
                    );
                    if let Some(&f) = output.get(base) {
                        last_filter = f;
                    }
                } else {
                    output.push(last_filter);
                    apply_filter_type(
                        last_filter,
                        row,
                        prev,
                        bytes_per_pixel,
                        &mut output,
                        &mut adaptive_scratch,
                    );
                }
            }
            _ => {
                let base = output.len();
                filter_row(
                    row,
                    if y == 0 { &zero_row[..] } else { &prev_row[..] },
                    bytes_per_pixel,
                    options.filter_strategy,
                    &mut output,
                    &mut adaptive_scratch,
                );
                if matches!(options.filter_strategy, FilterStrategy::AdaptiveFast) {
                    if let Some(&f) = output.get(base) {
                        last_filter = f;
                    }
                }
            }
        }

        // Update previous row
        prev_row.copy_from_slice(row);
    }

    output
}

/// Sub filter: difference from left pixel.
fn filter_sub(row: &[u8], bpp: usize, output: &mut Vec<u8>) {
    #[cfg(feature = "simd")]
    {
        simd::filter_sub(row, bpp, output);
        return;
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
        return;
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
        return;
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
    for (i, &byte) in row.iter().enumerate() {
        let left = if i >= bpp { row[i - bpp] } else { 0 };
        let above = prev_row[i];
        let upper_left = if i >= bpp { prev_row[i - bpp] } else { 0 };
        let predicted = paeth_predictor(left, above, upper_left);
        output.push(byte.wrapping_sub(predicted));
    }
}

/// Paeth predictor function.
///
/// Selects the value (a, b, or c) closest to p = a + b - c.
#[inline]
fn paeth_predictor(a: u8, b: u8, c: u8) -> u8 {
    let a = a as i16;
    let b = b as i16;
    let c = c as i16;

    let p = a + b - c;
    let pa = (p - a).abs();
    let pb = (p - b).abs();
    let pc = (p - c).abs();

    if pa <= pb && pa <= pc {
        a as u8
    } else if pb <= pc {
        b as u8
    } else {
        c as u8
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
    let early_stop = (row.len() as u64 / 8).saturating_add(1);

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

    // Early stop threshold: very low score, stop immediately.
    let early_stop = (row.len() as u64 / 10).saturating_add(1);
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

/// Filter a single row with the configured strategy.
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
        FilterStrategy::Adaptive => {
            adaptive_filter(row, prev_row, bpp, output, scratch);
        }
        FilterStrategy::AdaptiveFast => {
            adaptive_filter_fast(row, prev_row, bpp, output, scratch);
        }
        FilterStrategy::AdaptiveSampled { .. } => {
            // Fallback to full adaptive; sampled handling lives in apply_filters loop.
            adaptive_filter(row, prev_row, bpp, output, scratch);
        }
    }
}

fn apply_filter_type(
    filter: u8,
    row: &[u8],
    prev_row: &[u8],
    bpp: usize,
    output: &mut Vec<u8>,
    scratch: &mut AdaptiveScratch,
) {
    match filter {
        FILTER_NONE => output.extend_from_slice(row),
        FILTER_SUB => filter_sub(row, bpp, output),
        FILTER_UP => filter_up(row, prev_row, output),
        FILTER_AVERAGE => filter_average(row, prev_row, bpp, output),
        FILTER_PAETH => filter_paeth(row, prev_row, bpp, output),
        _ => {
            adaptive_filter(row, prev_row, bpp, output, scratch);
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
    let rows: Vec<Vec<u8>> = (0..height)
        .into_par_iter()
        .map(|y| {
            let row_start = y * row_bytes;
            let row = &data[row_start..row_start + row_bytes];
            let prev = if y == 0 {
                &zero_row[..]
            } else {
                &data[(y - 1) * row_bytes..y * row_bytes]
            };
            let mut out = Vec::with_capacity(filtered_row_size);
            let mut scratch = AdaptiveScratch::new(row_bytes);
            filter_row(row, prev, bpp, strategy, &mut out, &mut scratch);
            out
        })
        .collect();

    let mut output = Vec::with_capacity(filtered_row_size * height);
    for row in rows {
        output.extend_from_slice(&row);
    }
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
    fn test_apply_filters_adaptive_sampled_reuses_filter() {
        let data = vec![
            10, 20, 30, 40, 50, 60, // Row 1
            11, 21, 31, 41, 51, 61, // Row 2
            12, 22, 32, 42, 52, 62, // Row 3
            13, 23, 33, 43, 53, 63, // Row 4
        ];
        let options = PngOptions {
            filter_strategy: FilterStrategy::AdaptiveSampled { interval: 2 },
            ..Default::default()
        };

        let filtered = apply_filters(&data, 2, 4, 3, &options);

        // 4 rows, each with filter byte + 6 data bytes
        assert_eq!(filtered.len(), 4 * (1 + 6));
        let f1 = filtered[0];
        let f2 = filtered[7];
        let f3 = filtered[14];
        let f4 = filtered[21];
        // Rows 2 and 4 reuse previous filters
        assert_eq!(f2, f1);
        assert_eq!(f4, f3);
    }
}
