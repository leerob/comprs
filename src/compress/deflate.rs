//! DEFLATE compression algorithm (RFC 1951).
//!
//! Combines LZ77 compression with Huffman coding.

use crate::bits::BitWriter64;
use crate::compress::lz77::{
    CostModel, Lz77Compressor, PackedToken, Token, MAX_MATCH_LENGTH, MIN_MATCH_LENGTH,
};
use crate::compress::{adler32::adler32, huffman};
use std::sync::{LazyLock, Mutex};
use std::time::Duration;
use std::time::Instant;

/// Length code base values (codes 257-285).
const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];

/// Extra bits for length codes.
const LENGTH_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];

/// Distance code base values (codes 0-29).
const DISTANCE_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];

/// Extra bits for distance codes.
const DISTANCE_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

/// Timing and accounting information for a single DEFLATE encode.
#[derive(Debug, Clone)]
#[must_use]
pub struct DeflateStats {
    /// Time spent in LZ77 match finding/tokenization.
    pub lz77_time: Duration,
    /// Time spent encoding with fixed Huffman codes.
    pub fixed_huffman_time: Duration,
    /// Time spent encoding with dynamic Huffman codes.
    pub dynamic_huffman_time: Duration,
    /// Time spent choosing the smaller of fixed vs dynamic outputs.
    pub choose_time: Duration,
    /// Number of tokens produced by the LZ77 stage.
    pub token_count: usize,
    /// Number of literal tokens.
    pub literal_count: usize,
    /// Number of match tokens.
    pub match_count: usize,
    /// Whether the final stream used dynamic Huffman codes.
    pub used_dynamic: bool,
    /// Whether the zlib wrapper selected stored (uncompressed) blocks.
    pub used_stored_block: bool,
}

impl Default for DeflateStats {
    fn default() -> Self {
        Self {
            lz77_time: Duration::ZERO,
            fixed_huffman_time: Duration::ZERO,
            dynamic_huffman_time: Duration::ZERO,
            choose_time: Duration::ZERO,
            token_count: 0,
            literal_count: 0,
            match_count: 0,
            used_dynamic: false,
            used_stored_block: false,
        }
    }
}

/// Lookup table for length codes: maps length (3-258) to (symbol, extra_bits).
/// Index is (length - 3), value is (symbol - 257, extra_bits).
const LENGTH_LOOKUP: [(u8, u8); 256] = {
    let mut table = [(0u8, 0u8); 256];
    let mut i = 0usize;
    while i < 256 {
        let length = i + 3;
        // Find the appropriate code
        let mut code_idx = 0usize;
        while code_idx < 28 {
            if length >= LENGTH_BASE[code_idx] as usize
                && length < LENGTH_BASE[code_idx + 1] as usize
            {
                break;
            }
            code_idx += 1;
        }
        // code_idx 28 is for length 258
        table[i] = (code_idx as u8, LENGTH_EXTRA[code_idx]);
        i += 1;
    }
    table
};

/// Threshold (in tokens) above which we skip fixed-Huffman encoding and go
/// straight to dynamic codes to avoid double-encoding overhead on large
/// payloads (common for PNG scanlines).
const DYNAMIC_ONLY_TOKEN_THRESHOLD: usize = 128;
/// Below this token count, prefer fixed Huffman only to avoid double encoding.
const FIXED_ONLY_TOKEN_THRESHOLD: usize = 128;
/// Below this byte length, favor a simpler path and optionally skip dynamic Huffman.
const SMALL_INPUT_BYTES: usize = 1 << 10; // 1 KiB
/// Above this size and with high-entropy detection, skip LZ77/Huffman and emit stored blocks.
const HIGH_ENTROPY_BAIL_BYTES: usize = 4 * 1024;
/// When input has only literals (no matches) and meets this size, prefer stored blocks immediately.
const STORED_LITERAL_ONLY_BYTES: usize = 8 * 1024;

/// Global pool of reusable deflaters keyed by compression level.
/// Mutex-protected to allow reuse across threads while avoiding RefCell/thread-local costs.
static DEFLATE_REUSE: LazyLock<Vec<Mutex<Deflater>>> = LazyLock::new(|| {
    (0..=9)
        .map(|level| Mutex::new(Deflater::new(level.max(1) as u8)))
        .collect()
});

#[inline]
fn with_reusable_deflater<T>(level: u8, f: impl FnOnce(&mut Deflater) -> T) -> T {
    let level = level.clamp(1, 9);
    let idx = level as usize;
    let pool = DEFLATE_REUSE
        .get(idx)
        .unwrap_or_else(|| panic!("deflater pool missing for level {level}"));
    let mut guard = pool.lock().expect("deflater mutex poisoned");
    // If pool was initialized with different level (future-proof), refresh it.
    if guard.level() != level {
        *guard = Deflater::new(level);
    }
    f(&mut guard)
}

#[inline]
fn encode_best_huffman(tokens: &[Token], est_bytes: usize) -> (Vec<u8>, bool) {
    if tokens.len() <= FIXED_ONLY_TOKEN_THRESHOLD {
        return (encode_fixed_huffman_with_capacity(tokens, est_bytes), false);
    }

    if tokens.len() >= DYNAMIC_ONLY_TOKEN_THRESHOLD {
        return (
            encode_dynamic_huffman_with_capacity(tokens, est_bytes),
            true,
        );
    }

    let fixed = encode_fixed_huffman_with_capacity(tokens, est_bytes);
    let dynamic = encode_dynamic_huffman_with_capacity(tokens, est_bytes);
    if dynamic.len() < fixed.len() {
        (dynamic, true)
    } else {
        (fixed, false)
    }
}

#[inline]
fn encode_best_huffman_packed(tokens: &[PackedToken], est_bytes: usize) -> (Vec<u8>, bool) {
    if tokens.len() <= FIXED_ONLY_TOKEN_THRESHOLD {
        return (
            encode_fixed_huffman_packed_with_capacity(tokens, est_bytes),
            false,
        );
    }

    if tokens.len() >= DYNAMIC_ONLY_TOKEN_THRESHOLD {
        return (
            encode_dynamic_huffman_packed_with_capacity(tokens, est_bytes),
            true,
        );
    }

    let fixed = encode_fixed_huffman_packed_with_capacity(tokens, est_bytes);
    let dynamic = encode_dynamic_huffman_packed_with_capacity(tokens, est_bytes);
    if dynamic.len() < fixed.len() {
        (dynamic, true)
    } else {
        (fixed, false)
    }
}

/// Lookup table for distance codes: maps distance (1-32768) to code index.
/// Uses a two-level approach for efficiency.
const DISTANCE_LOOKUP_SMALL: [u8; 512] = {
    let mut table = [0u8; 512];
    let mut i = 1usize;
    while i < 512 {
        let mut code_idx = 0usize;
        while code_idx < 29 {
            if i >= DISTANCE_BASE[code_idx] as usize && i < DISTANCE_BASE[code_idx + 1] as usize {
                break;
            }
            code_idx += 1;
        }
        table[i] = code_idx as u8;
        i += 1;
    }
    table
};

/// Get the length code (257-285) for a match length.
/// Uses a lookup table for O(1) performance.
#[inline]
fn length_code(length: u16) -> (u16, u8, u16) {
    debug_assert!(
        (MIN_MATCH_LENGTH as u16..=MAX_MATCH_LENGTH as u16).contains(&length),
        "Invalid length: {length}",
    );

    let idx = (length - 3) as usize;
    let (code_offset, extra_bits) = LENGTH_LOOKUP[idx];
    let symbol = 257 + code_offset as u16;
    let extra_value = length - LENGTH_BASE[code_offset as usize];
    (symbol, extra_bits, extra_value)
}

/// Get the distance code (0-29) for a match distance.
/// Uses lookup table for small distances, bit manipulation for large.
#[inline]
fn distance_code(distance: u16) -> (u16, u8, u16) {
    debug_assert!((1..=32768).contains(&distance), "Invalid distance");

    let code_idx = if distance < 512 {
        // Use direct lookup for small distances (covers codes 0-17)
        DISTANCE_LOOKUP_SMALL[distance as usize] as usize
    } else {
        // For larger distances (512+), use bit manipulation
        // The pattern for distance codes 4+ is:
        // code = 2 * floor(log2(distance - 1)) + second_highest_bit
        // where second_highest_bit is the bit below the MSB of (distance - 1)
        let d = distance as u32 - 1;
        let msb = 31 - d.leading_zeros(); // position of highest set bit
        let second_bit = (d >> (msb - 1)) & 1; // second highest bit
        let code = (2 * msb + second_bit) as usize;
        code.min(29)
    };

    let extra_bits = DISTANCE_EXTRA[code_idx];
    let extra_value = distance - DISTANCE_BASE[code_idx];
    (code_idx as u16, extra_bits, extra_value)
}

/// Compress data using DEFLATE algorithm.
///
/// # Arguments
/// * `data` - Raw data to compress
/// * `level` - Compression level 1-9
///
/// # Returns
/// Compressed data in raw DEFLATE format (no zlib/gzip wrapper).
#[must_use]
pub fn deflate(data: &[u8], level: u8) -> Vec<u8> {
    if data.is_empty() {
        return empty_deflate_fixed_block();
    }

    if data.len() <= SMALL_INPUT_BYTES {
        with_reusable_deflater(level, |deflater| deflater.compress_fixed_only(data))
    } else {
        with_reusable_deflater(level, |deflater| deflater.compress(data))
    }
}

/// Compress data using DEFLATE algorithm with packed tokens (non-reusable).
///
/// This is an experimental fast path that avoids `Token` allocations by
/// emitting packed tokens directly into the Huffman encoder.
#[must_use]
pub fn deflate_packed(data: &[u8], level: u8) -> Vec<u8> {
    if data.is_empty() {
        return empty_deflate_fixed_block();
    }

    if data.len() <= SMALL_INPUT_BYTES {
        with_reusable_deflater(level, |deflater| deflater.compress_fixed_only_packed(data))
    } else {
        with_reusable_deflater(level, |deflater| deflater.compress_packed(data))
    }
}

// ============================================================================
// Optimal DEFLATE with Iterative Huffman Refinement (Zopfli-style)
// ============================================================================

/// Default number of iterations for optimal DEFLATE compression.
/// Zopfli uses 15 by default; we use fewer for a balance of compression vs speed.
const DEFAULT_OPTIMAL_ITERATIONS: usize = 5;

/// Compress data using optimal DEFLATE with iterative Huffman refinement.
///
/// This implements the core Zopfli technique:
/// 1. Initial greedy LZ77 pass to get baseline statistics
/// 2. Compute bit costs from symbol frequencies (entropy)
/// 3. Re-parse using optimal LZ77 with the cost model
/// 4. Iterate steps 2-3 until convergence or max iterations
/// 5. Return the smallest result
///
/// This produces significantly better compression than greedy parsing,
/// at the cost of much slower encoding.
#[must_use]
pub fn deflate_optimal(data: &[u8], iterations: usize) -> Vec<u8> {
    if data.is_empty() {
        return empty_deflate_fixed_block();
    }

    let mut lz77 = Lz77Compressor::new(9); // Use max chain length for optimal parsing

    // Initial greedy pass for baseline statistics
    let initial_tokens = lz77.compress(data);
    let (mut lit_len_counts, mut dist_counts) = count_symbols(&initial_tokens);

    // Encode initial result
    let est_bytes = estimated_deflate_size(data.len(), 9);
    let mut best_output = encode_dynamic_huffman_with_capacity(&initial_tokens, est_bytes);
    let mut best_size = best_output.len();

    // Track previous cost for convergence detection
    let mut prev_cost = f32::MAX;

    for iter in 0..iterations {
        // Create cost model from current statistics
        let cost_model = CostModel::from_statistics(&lit_len_counts, &dist_counts);

        // Re-parse with optimal LZ77 using the cost model
        let tokens = lz77.compress_optimal(data, &cost_model);

        // Encode and check size
        let output = encode_dynamic_huffman_with_capacity(&tokens, est_bytes);
        if output.len() < best_size {
            best_size = output.len();
            best_output = output;
        }

        // Update statistics for next iteration
        let (new_lit_counts, new_dist_counts) = count_symbols(&tokens);

        // Check for convergence: if cost hasn't improved much, we're done
        let cost: f32 = tokens
            .iter()
            .map(|t| match t {
                Token::Literal(b) => cost_model.literal_cost(*b),
                Token::Match { length, distance } => cost_model.match_cost(*length, *distance),
            })
            .sum();

        if iter > 2 && (prev_cost - cost).abs() < cost * 0.001 {
            // Converged
            break;
        }
        prev_cost = cost;

        // Blend new stats with old stats for smoother convergence (Zopfli trick)
        for i in 0..286 {
            lit_len_counts[i] = (lit_len_counts[i] as f32 * 0.5 + new_lit_counts[i] as f32) as u32;
        }
        for i in 0..30 {
            dist_counts[i] = (dist_counts[i] as f32 * 0.5 + new_dist_counts[i] as f32) as u32;
        }
    }

    best_output
}

/// Compress data using optimal DEFLATE with default iteration count.
#[must_use]
pub fn deflate_optimal_default(data: &[u8]) -> Vec<u8> {
    deflate_optimal(data, DEFAULT_OPTIMAL_ITERATIONS)
}

/// Maximum data size (in bytes) for which block splitting is attempted.
/// Block splitting has O(n²) cost estimation, so we skip it for very large inputs.
const BLOCK_SPLIT_SIZE_LIMIT: usize = 512 * 1024; // 512KB

/// Compress data using optimal DEFLATE and wrap in zlib container.
/// Uses adaptive block splitting for improved compression on smaller inputs.
#[must_use]
pub fn deflate_optimal_zlib(data: &[u8], iterations: usize) -> Vec<u8> {
    if data.is_empty() {
        return empty_zlib(9);
    }

    // Skip block splitting for very large inputs to avoid O(n²) cost
    if data.len() > BLOCK_SPLIT_SIZE_LIMIT {
        // Use optimal DEFLATE without block splitting
        let deflated = deflate_optimal(data, iterations);
        let use_stored = should_use_stored(data.len(), deflated.len());

        let mut output = Vec::with_capacity(deflated.len().min(data.len()) + 32);
        output.extend_from_slice(&zlib_header(9));

        if use_stored {
            let stored_blocks = deflate_stored(data);
            output.extend_from_slice(&stored_blocks);
        } else {
            output.extend_from_slice(&deflated);
        }

        output.extend_from_slice(&adler32(data).to_be_bytes());
        return output;
    }

    // Use block splitting for smaller inputs where the overhead is acceptable
    deflate_optimal_split_zlib(data, iterations, DEFAULT_MAX_BLOCKS)
}

/// Count symbol frequencies from a token stream.
fn count_symbols(tokens: &[Token]) -> ([u32; 286], [u32; 30]) {
    let mut lit_len_counts = [0u32; 286];
    let mut dist_counts = [0u32; 30];

    for token in tokens {
        match *token {
            Token::Literal(b) => {
                lit_len_counts[b as usize] += 1;
            }
            Token::Match { length, distance } => {
                let (len_symbol, _, _) = length_code(length);
                lit_len_counts[len_symbol as usize] += 1;

                let (dist_symbol, _, _) = distance_code(distance);
                dist_counts[dist_symbol as usize] += 1;
            }
        }
    }
    // End of block
    lit_len_counts[256] += 1;

    // Ensure at least one distance code per spec
    if dist_counts.iter().all(|&f| f == 0) {
        dist_counts[0] = 1;
    }

    (lit_len_counts, dist_counts)
}

// ============================================================================
// Adaptive Block Splitting (Zopfli-style)
// ============================================================================

/// Overhead in bits for a dynamic Huffman block header.
/// This includes the tree encoding (~200-400 bits typically).
const BLOCK_HEADER_OVERHEAD_BITS: f64 = 300.0;

/// Minimum block size in tokens to consider splitting.
const MIN_BLOCK_SIZE: usize = 10;

/// Maximum number of blocks to split into.
const DEFAULT_MAX_BLOCKS: usize = 15;

/// Count symbol frequencies for a slice of tokens.
fn count_symbols_range(tokens: &[Token], start: usize, end: usize) -> ([u32; 286], [u32; 30]) {
    let mut lit_len_counts = [0u32; 286];
    let mut dist_counts = [0u32; 30];

    for token in &tokens[start..end] {
        match *token {
            Token::Literal(b) => {
                lit_len_counts[b as usize] += 1;
            }
            Token::Match { length, distance } => {
                let (len_symbol, _, _) = length_code(length);
                lit_len_counts[len_symbol as usize] += 1;

                let (dist_symbol, _, _) = distance_code(distance);
                dist_counts[dist_symbol as usize] += 1;
            }
        }
    }
    // End of block
    lit_len_counts[256] += 1;

    // Ensure at least one distance code per spec
    if dist_counts.iter().all(|&f| f == 0) {
        dist_counts[0] = 1;
    }

    (lit_len_counts, dist_counts)
}

/// Estimate the bit cost of encoding a range of tokens with dynamic Huffman.
/// Returns estimated bits including block header overhead.
fn estimate_block_cost(tokens: &[Token], start: usize, end: usize) -> f64 {
    if end <= start {
        return 0.0;
    }

    let (lit_len_counts, dist_counts) = count_symbols_range(tokens, start, end);

    // Calculate entropy-based costs
    let lit_total: u32 = lit_len_counts.iter().sum();
    let dist_total: u32 = dist_counts.iter().sum();

    if lit_total == 0 {
        return BLOCK_HEADER_OVERHEAD_BITS;
    }

    let log_lit_total = (lit_total as f64).log2();
    let log_dist_total = if dist_total > 0 {
        (dist_total as f64).log2()
    } else {
        0.0
    };

    let mut total_bits = BLOCK_HEADER_OVERHEAD_BITS;

    // Add entropy cost for literal/length symbols
    for &count in &lit_len_counts {
        if count > 0 {
            let bits_per_symbol = log_lit_total - (count as f64).log2();
            total_bits += count as f64 * bits_per_symbol;
        }
    }

    // Add entropy cost for distance symbols
    for &count in &dist_counts {
        if count > 0 {
            let bits_per_symbol = log_dist_total - (count as f64).log2();
            total_bits += count as f64 * bits_per_symbol;
        }
    }

    // Add extra bits for length and distance codes
    for token in &tokens[start..end] {
        if let Token::Match { length, distance } = token {
            let (_, len_extra, _) = length_code(*length);
            let (_, dist_extra, _) = distance_code(*distance);
            total_bits += (len_extra + dist_extra) as f64;
        }
    }

    total_bits
}

/// Find the optimal split point within a range that minimizes total cost.
/// Returns (split_point, cost_if_split) or None if no beneficial split found.
fn find_best_split(tokens: &[Token], start: usize, end: usize) -> Option<(usize, f64)> {
    if end - start < MIN_BLOCK_SIZE * 2 {
        return None;
    }

    let orig_cost = estimate_block_cost(tokens, start, end);
    let mut best_split = None;
    let mut best_cost = orig_cost;

    // Try split points using a coarse-to-fine search
    // First pass: sample at regular intervals
    let step = ((end - start) / 9).max(1);
    let mut candidates = Vec::new();

    for i in (start + MIN_BLOCK_SIZE..end - MIN_BLOCK_SIZE).step_by(step) {
        let left_cost = estimate_block_cost(tokens, start, i);
        let right_cost = estimate_block_cost(tokens, i, end);
        let total = left_cost + right_cost;
        candidates.push((i, total));
    }

    // Find best candidate
    if let Some(&(best_i, cost)) = candidates
        .iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    {
        if cost < best_cost {
            best_cost = cost;
            best_split = Some(best_i);
        }
    }

    // Fine-tune around the best candidate
    if let Some(approx_best) = best_split {
        let fine_start = approx_best.saturating_sub(step).max(start + MIN_BLOCK_SIZE);
        let fine_end = (approx_best + step).min(end - MIN_BLOCK_SIZE);

        for i in fine_start..=fine_end {
            let left_cost = estimate_block_cost(tokens, start, i);
            let right_cost = estimate_block_cost(tokens, i, end);
            let total = left_cost + right_cost;
            if total < best_cost {
                best_cost = total;
                best_split = Some(i);
            }
        }
    }

    // Only return if splitting actually helps
    if let Some(split) = best_split {
        if best_cost < orig_cost - 10.0 {
            // Require at least 10 bits improvement
            return Some((split, best_cost));
        }
    }

    None
}

/// Find optimal block split points for a token stream.
/// Returns token indices where blocks should start (not including 0).
fn find_block_splits(tokens: &[Token], max_blocks: usize) -> Vec<usize> {
    if tokens.len() < MIN_BLOCK_SIZE * 2 || max_blocks <= 1 {
        return Vec::new();
    }

    let mut splits: Vec<usize> = Vec::new();
    let mut done = vec![false; tokens.len()];
    let mut num_blocks = 1;

    loop {
        if num_blocks >= max_blocks {
            break;
        }

        // Find the largest splittable block
        let mut largest_block: Option<(usize, usize, usize)> = None; // (start, end, size)

        let block_boundaries: Vec<usize> = std::iter::once(0)
            .chain(splits.iter().copied())
            .chain(std::iter::once(tokens.len()))
            .collect();

        for i in 0..block_boundaries.len() - 1 {
            let start = block_boundaries[i];
            let end = block_boundaries[i + 1];
            let size = end - start;

            if !done[start]
                && size >= MIN_BLOCK_SIZE * 2
                && largest_block.is_none_or(|(_, _, s)| size > s)
            {
                largest_block = Some((start, end, size));
            }
        }

        let Some((start, end, _)) = largest_block else {
            break;
        };

        // Try to find optimal split for this block
        if let Some((split_point, _)) = find_best_split(tokens, start, end) {
            // Insert split point in sorted order
            let insert_pos = splits
                .iter()
                .position(|&s| s > split_point)
                .unwrap_or(splits.len());
            splits.insert(insert_pos, split_point);
            num_blocks += 1;
        } else {
            // No beneficial split found for this block
            done[start] = true;
        }
    }

    splits
}

/// Write a dynamic Huffman block to a BitWriter.
/// This allows multiple blocks to share a single bit stream.
fn write_dynamic_huffman_block(writer: &mut BitWriter64, tokens: &[Token], is_final: bool) {
    // Frequencies
    let mut lit_freqs = vec![0u32; 286];
    let mut dist_freqs = vec![0u32; 30];

    for token in tokens {
        match *token {
            Token::Literal(b) => lit_freqs[b as usize] += 1,
            Token::Match { length, distance } => {
                let (len_symbol, _, _) = length_code(length);
                lit_freqs[len_symbol as usize] += 1;

                let (dist_symbol, _, _) = distance_code(distance);
                dist_freqs[dist_symbol as usize] += 1;
            }
        }
    }
    lit_freqs[256] += 1; // End-of-block

    if dist_freqs.iter().all(|&f| f == 0) {
        dist_freqs[0] = 1;
    }

    let lit_codes = huffman::build_codes(&lit_freqs, huffman::MAX_CODE_LENGTH);
    let dist_codes = huffman::build_codes(&dist_freqs, huffman::MAX_CODE_LENGTH);
    let lit_rev = prepare_reversed_codes(&lit_codes);
    let dist_rev = prepare_reversed_codes(&dist_codes);

    let mut lit_lengths: Vec<u8> = lit_codes.iter().map(|c| c.length).collect();
    let mut dist_lengths: Vec<u8> = dist_codes.iter().map(|c| c.length).collect();

    let hlit = (last_nonzero(&lit_lengths).saturating_sub(257)).min(29);
    let hdist = (last_nonzero(&dist_lengths).saturating_sub(1)).min(29);

    lit_lengths.truncate(257 + hlit as usize);
    dist_lengths.truncate(1 + hdist as usize);

    let mut cl_freqs = vec![0u32; 19];
    let rle = rle_code_lengths(&lit_lengths, &dist_lengths, &mut cl_freqs);

    let cl_codes = huffman::build_codes(&cl_freqs, 7);
    let cl_rev = prepare_reversed_codes(&cl_codes);

    let cl_order: [usize; 19] = [
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    ];
    let mut hclen = 0u8;
    for (i, &idx) in cl_order.iter().enumerate().rev() {
        if cl_codes[idx].length > 0 {
            hclen = i.min(15) as u8;
            break;
        }
    }

    writer.write_bits(if is_final { 1 } else { 0 }, 1); // BFINAL
    writer.write_bits(2, 2); // BTYPE=10 (dynamic)

    writer.write_bits(hlit as u32, 5);
    writer.write_bits(hdist as u32, 5);
    writer.write_bits(hclen as u32, 4);

    for &idx in cl_order.iter().take(hclen as usize + 4) {
        writer.write_bits(cl_codes[idx].length as u32, 3);
    }

    for (sym, extra_bits, extra_len) in rle {
        let (code, len) = cl_rev[sym as usize];
        writer.write_bits(code, len);
        if extra_len > 0 {
            writer.write_bits(extra_bits as u32, extra_len);
        }
    }

    for token in tokens {
        match *token {
            Token::Literal(byte) => {
                let (code, len) = lit_rev[byte as usize];
                writer.write_bits(code, len);
            }
            Token::Match { length, distance } => {
                let (len_symbol, len_extra_bits, len_extra_value) = length_code(length);
                let (len_code, len_len) = lit_rev[len_symbol as usize];
                writer.write_bits(len_code, len_len);
                if len_extra_bits > 0 {
                    writer.write_bits(len_extra_value as u32, len_extra_bits);
                }

                let (dist_symbol, dist_extra_bits, dist_extra_value) = distance_code(distance);
                let (dist_code, dist_len) = dist_rev[dist_symbol as usize];
                writer.write_bits(dist_code, dist_len);
                if dist_extra_bits > 0 {
                    writer.write_bits(dist_extra_value as u32, dist_extra_bits);
                }
            }
        }
    }

    let (eob_code, eob_len) = lit_rev[256];
    writer.write_bits(eob_code, eob_len);
}

/// Compress data using optimal DEFLATE with adaptive block splitting.
///
/// This extends `deflate_optimal` by splitting the token stream into
/// multiple blocks when doing so reduces total encoded size. Each block
/// gets its own Huffman tables optimized for its content.
#[must_use]
pub fn deflate_optimal_split(data: &[u8], iterations: usize, max_blocks: usize) -> Vec<u8> {
    if data.is_empty() {
        return empty_deflate_fixed_block();
    }

    let mut lz77 = Lz77Compressor::new(9);

    // Initial greedy pass for baseline statistics
    let initial_tokens = lz77.compress(data);
    let (mut lit_len_counts, mut dist_counts) = count_symbols(&initial_tokens);

    let mut best_tokens = initial_tokens;
    let mut prev_cost = f32::MAX;

    // Iterative refinement
    for iter in 0..iterations {
        let cost_model = CostModel::from_statistics(&lit_len_counts, &dist_counts);
        let tokens = lz77.compress_optimal(data, &cost_model);

        // Update statistics
        let (new_lit_counts, new_dist_counts) = count_symbols(&tokens);

        // Check for convergence
        let cost: f32 = tokens
            .iter()
            .map(|t| match t {
                Token::Literal(b) => cost_model.literal_cost(*b),
                Token::Match { length, distance } => cost_model.match_cost(*length, *distance),
            })
            .sum();

        if iter > 2 && (prev_cost - cost).abs() < cost * 0.001 {
            best_tokens = tokens;
            break;
        }
        prev_cost = cost;
        best_tokens = tokens;

        // Blend statistics
        for i in 0..286 {
            lit_len_counts[i] = (lit_len_counts[i] as f32 * 0.5 + new_lit_counts[i] as f32) as u32;
        }
        for i in 0..30 {
            dist_counts[i] = (dist_counts[i] as f32 * 0.5 + new_dist_counts[i] as f32) as u32;
        }
    }

    // Find optimal block splits
    let splits = find_block_splits(&best_tokens, max_blocks);

    // Create a single BitWriter for the entire stream
    let est_bytes = best_tokens.len() * 2;
    let mut writer = BitWriter64::with_capacity(est_bytes);

    if splits.is_empty() {
        // No splitting beneficial, encode as single block
        write_dynamic_huffman_block(&mut writer, &best_tokens, true);
    } else {
        // Encode each block using the same bit writer
        let boundaries: Vec<usize> = std::iter::once(0)
            .chain(splits.iter().copied())
            .chain(std::iter::once(best_tokens.len()))
            .collect();

        for i in 0..boundaries.len() - 1 {
            let start = boundaries[i];
            let end = boundaries[i + 1];
            let is_final = i == boundaries.len() - 2;

            let block_tokens = &best_tokens[start..end];
            write_dynamic_huffman_block(&mut writer, block_tokens, is_final);
        }
    }

    writer.finish()
}

/// Compress data using optimal DEFLATE with block splitting and wrap in zlib container.
#[must_use]
pub fn deflate_optimal_split_zlib(data: &[u8], iterations: usize, max_blocks: usize) -> Vec<u8> {
    if data.is_empty() {
        return empty_zlib(9);
    }

    let deflated = deflate_optimal_split(data, iterations, max_blocks);
    let use_stored = should_use_stored(data.len(), deflated.len());

    let mut output = Vec::with_capacity(deflated.len().min(data.len()) + 32);
    output.extend_from_slice(&zlib_header(9));

    if use_stored {
        let stored_blocks = deflate_stored(data);
        output.extend_from_slice(&stored_blocks);
    } else {
        output.extend_from_slice(&deflated);
    }

    output.extend_from_slice(&adler32(data).to_be_bytes());
    output
}

/// Compress data using DEFLATE algorithm with packed tokens, returning stats.
#[cfg(feature = "timing")]
pub fn deflate_packed_with_stats(data: &[u8], level: u8) -> (Vec<u8>, DeflateStats) {
    if data.is_empty() {
        return (
            empty_deflate_fixed_block(),
            DeflateStats {
                used_dynamic: false,
                ..Default::default()
            },
        );
    }

    let t0 = Instant::now();
    let mut lz77 = Lz77Compressor::new(level);
    let tokens = lz77.compress_packed(data);
    let lz77_time = t0.elapsed();

    let (literal_count, match_count) = token_counts_packed(&tokens);

    // Literal-only fast path to skip Huffman work for incompressible data.
    if match_count == 0 && data.len() >= STORED_LITERAL_ONLY_BYTES {
        let stored = deflate_stored(data);
        let stats = DeflateStats {
            lz77_time,
            fixed_huffman_time: Duration::ZERO,
            dynamic_huffman_time: Duration::ZERO,
            choose_time: Duration::ZERO,
            token_count: tokens.len(),
            literal_count,
            match_count,
            used_dynamic: false,
            used_stored_block: true,
        };
        return (stored, stats);
    }

    let est_bytes = estimated_deflate_size(data.len(), level);

    let t1 = Instant::now();
    let fixed = encode_fixed_huffman_packed_with_capacity(&tokens, est_bytes);
    let fixed_time = t1.elapsed();

    let t2 = Instant::now();
    let dynamic = encode_dynamic_huffman_packed_with_capacity(&tokens, est_bytes);
    let dynamic_time = t2.elapsed();

    let choose_start = Instant::now();
    let use_dynamic = dynamic.len() < fixed.len();
    let encoded = if use_dynamic { dynamic } else { fixed };
    let choose_time = choose_start.elapsed();

    let stats = DeflateStats {
        lz77_time,
        fixed_huffman_time: fixed_time,
        dynamic_huffman_time: dynamic_time,
        choose_time,
        token_count: tokens.len(),
        literal_count,
        match_count,
        used_dynamic: use_dynamic,
        used_stored_block: false,
    };

    (encoded, stats)
}

fn token_counts_packed(tokens: &[PackedToken]) -> (usize, usize) {
    let mut literal_count = 0;
    let mut match_count = 0;
    for t in tokens {
        if t.is_literal() {
            literal_count += 1;
        } else {
            match_count += 1;
        }
    }
    (literal_count, match_count)
}

/// Reusable DEFLATE encoder that minimizes allocations by reusing buffers.
pub struct Deflater {
    level: u8,
    lz77: Lz77Compressor,
    tokens: Vec<Token>,
    packed_tokens: Vec<PackedToken>,
}

impl Deflater {
    /// Create a new reusable deflater for the given level.
    pub fn new(level: u8) -> Self {
        let level = level.clamp(1, 9);
        Self {
            level,
            lz77: Lz77Compressor::new(level),
            tokens: Vec::new(),
            packed_tokens: Vec::new(),
        }
    }

    /// Return the compression level configured for this deflater.
    #[inline]
    pub fn level(&self) -> u8 {
        self.level
    }

    /// Compress raw data into a DEFLATE stream.
    pub fn compress(&mut self, data: &[u8]) -> Vec<u8> {
        if data.is_empty() {
            return empty_deflate_fixed_block();
        }

        self.tokens.clear();
        self.tokens.reserve(data.len());
        self.lz77.compress_into(data, &mut self.tokens);

        let est_bytes = estimated_deflate_size(data.len(), self.level);
        let (encoded, _) = encode_best_huffman(&self.tokens, est_bytes);
        encoded
    }

    /// Compress using only fixed Huffman codes (for very small inputs).
    pub fn compress_fixed_only(&mut self, data: &[u8]) -> Vec<u8> {
        if data.is_empty() {
            return empty_deflate_fixed_block();
        }

        self.tokens.clear();
        self.tokens.reserve(data.len());
        self.lz77.compress_into(data, &mut self.tokens);

        let est_bytes = estimated_deflate_size(data.len(), self.level);
        encode_fixed_huffman_with_capacity(&self.tokens, est_bytes)
    }

    /// Compress data and wrap in a zlib container.
    pub fn compress_zlib(&mut self, data: &[u8]) -> Vec<u8> {
        if data.is_empty() {
            return empty_zlib(self.level);
        }

        self.tokens.clear();
        self.tokens.reserve(data.len());
        self.lz77.compress_into(data, &mut self.tokens);

        let est_bytes = estimated_deflate_size(data.len(), self.level);
        let (deflated, _) = encode_best_huffman(&self.tokens, est_bytes);

        let use_stored = should_use_stored(data.len(), deflated.len());

        let mut output = Vec::with_capacity(deflated.len().min(data.len()) + 32);
        output.extend_from_slice(&zlib_header(self.level));

        if use_stored {
            let stored_blocks = deflate_stored(data);
            output.extend_from_slice(&stored_blocks);
        } else {
            output.extend_from_slice(&deflated);
        }

        output.extend_from_slice(&adler32(data).to_be_bytes());
        output
    }

    /// Compress data using packed tokens into a raw DEFLATE stream.
    pub fn compress_packed(&mut self, data: &[u8]) -> Vec<u8> {
        if data.is_empty() {
            return empty_deflate_fixed_block();
        }

        self.packed_tokens.clear();
        self.packed_tokens.reserve(data.len());
        self.lz77
            .compress_packed_into(data, &mut self.packed_tokens);

        let (_literal_count, match_count) = token_counts_packed(&self.packed_tokens);

        if match_count == 0 && data.len() >= STORED_LITERAL_ONLY_BYTES {
            return deflate_stored(data);
        }

        let est_bytes = estimated_deflate_size(data.len(), self.level);
        let (encoded, _) = encode_best_huffman_packed(&self.packed_tokens, est_bytes);
        encoded
    }

    /// Compress using only fixed Huffman codes (packed) for very small inputs.
    pub fn compress_fixed_only_packed(&mut self, data: &[u8]) -> Vec<u8> {
        if data.is_empty() {
            return empty_deflate_fixed_block();
        }

        self.packed_tokens.clear();
        self.packed_tokens.reserve(data.len());
        self.lz77
            .compress_packed_into(data, &mut self.packed_tokens);

        let est_bytes = estimated_deflate_size(data.len(), self.level);
        encode_fixed_huffman_packed_with_capacity(&self.packed_tokens, est_bytes)
    }

    /// Compress data using packed tokens and wrap in a zlib container.
    pub fn compress_packed_zlib(&mut self, data: &[u8]) -> Vec<u8> {
        if data.is_empty() {
            return empty_zlib(self.level);
        }

        self.packed_tokens.clear();
        self.packed_tokens.reserve(data.len());
        self.lz77
            .compress_packed_into(data, &mut self.packed_tokens);

        let (_literal_count, match_count) = token_counts_packed(&self.packed_tokens);

        if match_count == 0 && data.len() >= STORED_LITERAL_ONLY_BYTES {
            let stored_blocks = deflate_stored(data);
            let mut output = Vec::with_capacity(stored_blocks.len().min(data.len()) + 32);
            output.extend_from_slice(&zlib_header(self.level));
            output.extend_from_slice(&stored_blocks);
            output.extend_from_slice(&adler32(data).to_be_bytes());
            return output;
        }

        let est_bytes = estimated_deflate_size(data.len(), self.level);
        let (deflated, _) = encode_best_huffman_packed(&self.packed_tokens, est_bytes);

        let use_stored = should_use_stored(data.len(), deflated.len());

        let mut output = Vec::with_capacity(deflated.len().min(data.len()) + 32);
        output.extend_from_slice(&zlib_header(self.level));

        if use_stored {
            let stored_blocks = deflate_stored(data);
            output.extend_from_slice(&stored_blocks);
        } else {
            output.extend_from_slice(&deflated);
        }

        output.extend_from_slice(&adler32(data).to_be_bytes());
        output
    }
}

/// Compress data using DEFLATE and return encoded bytes plus timing/accounting stats.
///
/// This leaves the main `deflate` fast path unchanged; callers opt-in to the
/// additional instrumentation by using this entrypoint.
pub fn deflate_with_stats(data: &[u8], level: u8) -> (Vec<u8>, DeflateStats) {
    if data.is_empty() {
        return (
            empty_deflate_fixed_block(),
            DeflateStats {
                used_dynamic: false,
                ..Default::default()
            },
        );
    }

    // LZ77 tokenization
    let t0 = Instant::now();
    let mut lz77 = Lz77Compressor::new(level);
    let tokens = lz77.compress(data);
    let lz77_time = t0.elapsed();

    let (literal_count, match_count) = token_counts(&tokens);

    // Fixed Huffman encode
    let t1 = Instant::now();
    let est_bytes = estimated_deflate_size(data.len(), level);

    let (encoded, fixed_time, dynamic_time, choose_time, use_dynamic) =
        if tokens.len() >= DYNAMIC_ONLY_TOKEN_THRESHOLD {
            // Skip fixed encode for large inputs to avoid double work.
            let t_dyn_start = Instant::now();
            let dynamic = encode_dynamic_huffman_with_capacity(&tokens, est_bytes);
            let dynamic_time = t_dyn_start.elapsed();
            (dynamic, Duration::ZERO, dynamic_time, Duration::ZERO, true)
        } else {
            let fixed = encode_fixed_huffman_with_capacity(&tokens, est_bytes);
            let fixed_time = t1.elapsed();

            // Dynamic Huffman encode
            let t2 = Instant::now();
            let dynamic = encode_dynamic_huffman_with_capacity(&tokens, est_bytes);
            let dynamic_time = t2.elapsed();

            // Choose smaller stream
            let choose_start = Instant::now();
            let use_dynamic = dynamic.len() < fixed.len();
            let encoded = if use_dynamic { dynamic } else { fixed };
            let choose_time = choose_start.elapsed();

            (encoded, fixed_time, dynamic_time, choose_time, use_dynamic)
        };

    let stats = DeflateStats {
        lz77_time,
        fixed_huffman_time: fixed_time,
        dynamic_huffman_time: dynamic_time,
        choose_time,
        token_count: tokens.len(),
        literal_count,
        match_count,
        used_dynamic: use_dynamic,
        used_stored_block: false,
    };

    (encoded, stats)
}

/// Compress data and wrap it in a zlib container (RFC 1950).
///
/// Produces: zlib header (CMF/FLG), deflate stream, Adler-32 checksum.
#[must_use]
pub fn deflate_zlib(data: &[u8], level: u8) -> Vec<u8> {
    if data.len() >= HIGH_ENTROPY_BAIL_BYTES && is_high_entropy_data(data) {
        return deflate_zlib_stored(data, level);
    }
    with_reusable_deflater(level, |d| d.compress_zlib(data))
}

/// Compress data with packed tokens and wrap it in a zlib container.
#[must_use]
pub fn deflate_zlib_packed(data: &[u8], level: u8) -> Vec<u8> {
    if data.len() >= HIGH_ENTROPY_BAIL_BYTES && is_high_entropy_data(data) {
        return deflate_zlib_stored(data, level);
    }
    with_reusable_deflater(level, |d| d.compress_packed_zlib(data))
}

/// Emit zlib wrapper with stored (uncompressed) DEFLATE blocks.
fn deflate_zlib_stored(data: &[u8], level: u8) -> Vec<u8> {
    let mut output = Vec::with_capacity(data.len() + 16);
    output.extend_from_slice(&zlib_header(level));
    let stored_blocks = deflate_stored(data);
    output.extend_from_slice(&stored_blocks);
    output.extend_from_slice(&adler32(data).to_be_bytes());
    output
}

/// Compress data using DEFLATE in a zlib container, returning encoded bytes plus stats.
pub fn deflate_zlib_with_stats(data: &[u8], level: u8) -> (Vec<u8>, DeflateStats) {
    // Empty input mirrors `deflate_zlib`
    if data.is_empty() {
        return (
            empty_zlib(level),
            DeflateStats {
                used_stored_block: false,
                ..Default::default()
            },
        );
    }

    // High-entropy fast path: skip LZ77/Huffman and emit stored blocks directly.
    if data.len() >= HIGH_ENTROPY_BAIL_BYTES && is_high_entropy_data(data) {
        let mut output = Vec::with_capacity(data.len() + 16);
        output.extend_from_slice(&zlib_header(level));
        let stored_blocks = deflate_stored(data);
        output.extend_from_slice(&stored_blocks);
        output.extend_from_slice(&adler32(data).to_be_bytes());
        let stats = DeflateStats {
            used_stored_block: true,
            ..Default::default()
        };
        return (output, stats);
    }

    let (deflated, mut stats) = deflate_with_stats(data, level);

    let use_stored = should_use_stored(data.len(), deflated.len());
    let mut output = Vec::with_capacity(deflated.len().min(data.len()) + 32);
    output.extend_from_slice(&zlib_header(level));

    if use_stored {
        let stored_blocks = deflate_stored(data);
        output.extend_from_slice(&stored_blocks);
    } else {
        output.extend_from_slice(&deflated);
    }

    output.extend_from_slice(&adler32(data).to_be_bytes());

    stats.used_stored_block = use_stored;
    (output, stats)
}

/// Compress data with packed tokens into a zlib container, returning stats.
#[cfg(feature = "timing")]
pub fn deflate_zlib_packed_with_stats(data: &[u8], level: u8) -> (Vec<u8>, DeflateStats) {
    if data.is_empty() {
        return (
            empty_zlib(level),
            DeflateStats {
                used_stored_block: false,
                ..Default::default()
            },
        );
    }

    if data.len() >= HIGH_ENTROPY_BAIL_BYTES && is_high_entropy_data(data) {
        let mut output = Vec::with_capacity(data.len() + 16);
        output.extend_from_slice(&zlib_header(level));
        let stored_blocks = deflate_stored(data);
        output.extend_from_slice(&stored_blocks);
        output.extend_from_slice(&adler32(data).to_be_bytes());
        let stats = DeflateStats {
            used_stored_block: true,
            ..Default::default()
        };
        return (output, stats);
    }

    let (deflated, mut stats) = deflate_packed_with_stats(data, level);

    let use_stored = should_use_stored(data.len(), deflated.len());
    let mut output = Vec::with_capacity(deflated.len().min(data.len()) + 32);
    output.extend_from_slice(&zlib_header(level));

    if use_stored {
        let stored_blocks = deflate_stored(data);
        output.extend_from_slice(&stored_blocks);
    } else {
        output.extend_from_slice(&deflated);
    }

    output.extend_from_slice(&adler32(data).to_be_bytes());

    stats.used_stored_block = use_stored;
    (output, stats)
}

/// Decide whether stored blocks would be smaller than the compressed stream.
fn should_use_stored(data_len: usize, deflated_len: usize) -> bool {
    // Stored block size: data + 5 bytes per 65535 chunk
    let stored_overhead = (data_len / 65_535 + 1) * 5;
    let stored_total = data_len + stored_overhead + 2 /*zlib hdr*/ + 4 /*adler*/;
    let deflated_total = deflated_len + 2 /*zlib hdr*/ + 4 /*adler*/;
    deflated_total >= stored_total
}

/// Detect high-entropy (likely incompressible) data by sampling for repeated n-grams.
///
/// The previous heuristic (equal neighbors + delta histogram) failed on repetitive text
/// because text has few adjacent equal bytes and many different character transitions,
/// even when highly compressible by LZ77.
///
/// This new heuristic samples 4-byte sequences (what LZ77 actually matches on) and checks
/// if we see repeated patterns. True high-entropy data (random bytes) will have very few
/// repeated 4-grams, while compressible data (text, structured data) will have many.
fn is_high_entropy_data(data: &[u8]) -> bool {
    if data.len() < 4096 {
        return false;
    }

    // Sample a portion of the data to avoid O(n) overhead on large inputs.
    // Check first 8KB for repeated 4-byte sequences using a simple hash set approach.
    let sample_len = data.len().min(8192);
    let sample = &data[..sample_len];

    // Use a simple hash table to count unique 4-grams
    // If we see many repeated 4-grams, data is likely compressible
    const HASH_SIZE: usize = 4096;
    let mut seen = [false; HASH_SIZE];
    let mut collisions = 0usize;

    for window in sample.windows(4) {
        let val = u32::from_le_bytes([window[0], window[1], window[2], window[3]]);
        let hash = ((val.wrapping_mul(0x1E35_A7BD)) >> 20) as usize & (HASH_SIZE - 1);

        if seen[hash] {
            collisions += 1;
        } else {
            seen[hash] = true;
        }
    }

    // For truly random data, we expect very few hash collisions in a 4K table
    // when sampling 8K 4-grams (birthday paradox gives ~50% fill).
    // Compressible data will have many repeated patterns causing high collision rate.
    //
    // If collision rate is < 5%, data is likely random/incompressible.
    // This threshold is conservative to avoid false positives on compressible data.
    let total_4grams = sample_len.saturating_sub(3);
    let collision_rate = collisions as f32 / total_4grams as f32;

    collision_rate < 0.05
}

/// Encode tokens using fixed Huffman codes.
pub fn encode_fixed_huffman(tokens: &[Token]) -> Vec<u8> {
    encode_fixed_huffman_with_capacity(tokens, 1024)
}

fn encode_fixed_huffman_with_capacity(tokens: &[Token], capacity_hint: usize) -> Vec<u8> {
    let lit_rev = fixed_literal_codes_rev();
    let dist_rev = fixed_distance_codes_rev();

    let mut writer = BitWriter64::with_capacity(capacity_hint);

    // Block header: BFINAL=1 (last block), BTYPE=01 (fixed Huffman)
    writer.write_bits(1, 1); // BFINAL
    writer.write_bits(1, 2); // BTYPE (01 = fixed Huffman, LSB first)

    for token in tokens {
        match *token {
            Token::Literal(byte) => {
                let (code, len) = lit_rev[byte as usize];
                writer.write_bits(code, len);
            }
            Token::Match { length, distance } => {
                // Encode length
                let (len_symbol, len_extra_bits, len_extra_value) = length_code(length);
                let (len_code, len_len) = lit_rev[len_symbol as usize];
                writer.write_bits(len_code, len_len);

                if len_extra_bits > 0 {
                    writer.write_bits(len_extra_value as u32, len_extra_bits);
                }

                // Encode distance
                let (dist_symbol, dist_extra_bits, dist_extra_value) = distance_code(distance);
                let (dist_code, dist_len) = dist_rev[dist_symbol as usize];
                writer.write_bits(dist_code, dist_len);

                if dist_extra_bits > 0 {
                    writer.write_bits(dist_extra_value as u32, dist_extra_bits);
                }
            }
        }
    }

    // End of block symbol (256)
    let (eob_code, eob_len) = lit_rev[256];
    writer.write_bits(eob_code, eob_len);

    writer.finish()
}

/// Encode packed tokens using fixed Huffman codes (fast path).
pub fn encode_fixed_huffman_packed(tokens: &[PackedToken]) -> Vec<u8> {
    encode_fixed_huffman_packed_with_capacity(tokens, 1024)
}

fn encode_fixed_huffman_packed_with_capacity(
    tokens: &[PackedToken],
    capacity_hint: usize,
) -> Vec<u8> {
    let lit_rev = fixed_literal_codes_rev();
    let dist_rev = fixed_distance_codes_rev();

    let mut writer = BitWriter64::with_capacity(capacity_hint);

    writer.write_bits(1, 1); // BFINAL
    writer.write_bits(1, 2); // BTYPE fixed

    for token in tokens {
        if let Some(b) = token.as_literal() {
            let (code, len) = lit_rev[b as usize];
            writer.write_bits(code, len);
        } else if let Some((length, distance)) = token.as_match() {
            let (len_symbol, len_extra_bits, len_extra_value) = length_code(length);
            let (len_code, len_len) = lit_rev[len_symbol as usize];
            writer.write_bits(len_code, len_len);
            if len_extra_bits > 0 {
                writer.write_bits(len_extra_value as u32, len_extra_bits);
            }

            let (dist_symbol, dist_extra_bits, dist_extra_value) = distance_code(distance);
            let (dist_code, dist_len) = dist_rev[dist_symbol as usize];
            writer.write_bits(dist_code, dist_len);
            if dist_extra_bits > 0 {
                writer.write_bits(dist_extra_value as u32, dist_extra_bits);
            }
        }
    }

    let (eob_code, eob_len) = lit_rev[256];
    writer.write_bits(eob_code, eob_len);

    writer.finish()
}

/// Encode tokens using dynamic Huffman codes (RFC 1951).
pub fn encode_dynamic_huffman(tokens: &[Token]) -> Vec<u8> {
    encode_dynamic_huffman_with_capacity(tokens, 1024)
}

fn encode_dynamic_huffman_with_capacity(tokens: &[Token], capacity_hint: usize) -> Vec<u8> {
    // Frequencies
    let mut lit_freqs = vec![0u32; 286]; // 0-285
    let mut dist_freqs = vec![0u32; 30]; // 0-29

    for token in tokens {
        match *token {
            Token::Literal(b) => lit_freqs[b as usize] += 1,
            Token::Match { length, distance } => {
                let (len_symbol, _, _) = length_code(length);
                lit_freqs[len_symbol as usize] += 1;

                let (dist_symbol, _, _) = distance_code(distance);
                dist_freqs[dist_symbol as usize] += 1;
            }
        }
    }
    // End-of-block
    lit_freqs[256] += 1;

    // Ensure at least one distance code per spec
    if dist_freqs.iter().all(|&f| f == 0) {
        dist_freqs[0] = 1;
    }

    let lit_codes = huffman::build_codes(&lit_freqs, huffman::MAX_CODE_LENGTH);
    let dist_codes = huffman::build_codes(&dist_freqs, huffman::MAX_CODE_LENGTH);
    let lit_rev = prepare_reversed_codes(&lit_codes);
    let dist_rev = prepare_reversed_codes(&dist_codes);

    // Code lengths
    let mut lit_lengths: Vec<u8> = lit_codes.iter().map(|c| c.length).collect();
    let mut dist_lengths: Vec<u8> = dist_codes.iter().map(|c| c.length).collect();

    // Trim trailing zeros for HLIT/HDIST
    let hlit = (last_nonzero(&lit_lengths).saturating_sub(257)).min(29);
    let hdist = (last_nonzero(&dist_lengths).saturating_sub(1)).min(29);

    lit_lengths.truncate(257 + hlit as usize);
    dist_lengths.truncate(1 + hdist as usize);

    // RLE encode code lengths
    let mut cl_freqs = vec![0u32; 19];
    let rle = rle_code_lengths(&lit_lengths, &dist_lengths, &mut cl_freqs);

    // Build code length codes (max len 7)
    let cl_codes = huffman::build_codes(&cl_freqs, 7);
    let cl_rev = prepare_reversed_codes(&cl_codes);

    // Determine HCLEN (last non-zero in order)
    let cl_order: [usize; 19] = [
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    ];
    // Find last non-zero code length in cl_order; HCLEN = index, clamped to [0, 15]
    // (HCLEN is 4 bits, and HCLEN + 4 gives number of code length codes, max 19)
    let mut hclen = 0u8;
    for (i, &idx) in cl_order.iter().enumerate().rev() {
        if cl_codes[idx].length > 0 {
            hclen = i.min(15) as u8; // clamp to 4-bit max
            break;
        }
    }

    let mut writer = BitWriter64::with_capacity(capacity_hint);
    writer.write_bits(1, 1); // BFINAL (single block)
    writer.write_bits(2, 2); // BTYPE=10 (dynamic)

    writer.write_bits(hlit as u32, 5); // HLIT
    writer.write_bits(hdist as u32, 5); // HDIST
    writer.write_bits(hclen as u32, 4); // HCLEN (number of code length codes - 4)

    // Write code length code lengths in order (3-bit values per RFC 1951)
    for &idx in cl_order.iter().take(hclen as usize + 4) {
        writer.write_bits(cl_codes[idx].length as u32, 3);
    }

    // Write the RLE-encoded code lengths
    for (sym, extra_bits, extra_len) in rle {
        let (code, len) = cl_rev[sym as usize];
        writer.write_bits(code, len);
        if extra_len > 0 {
            writer.write_bits(extra_bits as u32, extra_len);
        }
    }

    // Data block using dynamic codes
    for token in tokens {
        match *token {
            Token::Literal(byte) => {
                let (code, len) = lit_rev[byte as usize];
                writer.write_bits(code, len);
            }
            Token::Match { length, distance } => {
                let (len_symbol, len_extra_bits, len_extra_value) = length_code(length);
                let (len_code, len_len) = lit_rev[len_symbol as usize];
                writer.write_bits(len_code, len_len);
                if len_extra_bits > 0 {
                    writer.write_bits(len_extra_value as u32, len_extra_bits);
                }

                let (dist_symbol, dist_extra_bits, dist_extra_value) = distance_code(distance);
                let (dist_code, dist_len) = dist_rev[dist_symbol as usize];
                writer.write_bits(dist_code, dist_len);
                if dist_extra_bits > 0 {
                    writer.write_bits(dist_extra_value as u32, dist_extra_bits);
                }
            }
        }
    }

    // End of block
    let (eob_code, eob_len) = lit_rev[256];
    writer.write_bits(eob_code, eob_len);

    writer.finish()
}

/// Encode packed tokens using dynamic Huffman codes.
pub fn encode_dynamic_huffman_packed(tokens: &[PackedToken]) -> Vec<u8> {
    encode_dynamic_huffman_packed_with_capacity(tokens, 1024)
}

fn encode_dynamic_huffman_packed_with_capacity(
    tokens: &[PackedToken],
    capacity_hint: usize,
) -> Vec<u8> {
    // Frequencies
    let mut lit_freqs = vec![0u32; 286]; // 0-285
    let mut dist_freqs = vec![0u32; 30]; // 0-29

    for token in tokens {
        if let Some(b) = token.as_literal() {
            lit_freqs[b as usize] += 1;
        } else if let Some((length, distance)) = token.as_match() {
            let (len_symbol, _, _) = length_code(length);
            lit_freqs[len_symbol as usize] += 1;

            let (dist_symbol, _, _) = distance_code(distance);
            dist_freqs[dist_symbol as usize] += 1;
        }
    }
    // End-of-block
    lit_freqs[256] += 1;

    if dist_freqs.iter().all(|&f| f == 0) {
        dist_freqs[0] = 1;
    }

    let lit_codes = huffman::build_codes(&lit_freqs, huffman::MAX_CODE_LENGTH);
    let dist_codes = huffman::build_codes(&dist_freqs, huffman::MAX_CODE_LENGTH);
    let lit_rev = prepare_reversed_codes(&lit_codes);
    let dist_rev = prepare_reversed_codes(&dist_codes);

    let mut lit_lengths: Vec<u8> = lit_codes.iter().map(|c| c.length).collect();
    let mut dist_lengths: Vec<u8> = dist_codes.iter().map(|c| c.length).collect();

    let hlit = (last_nonzero(&lit_lengths).saturating_sub(257)).min(29);
    let hdist = (last_nonzero(&dist_lengths).saturating_sub(1)).min(29);

    lit_lengths.truncate(257 + hlit as usize);
    dist_lengths.truncate(1 + hdist as usize);

    let mut cl_freqs = vec![0u32; 19];
    let rle = rle_code_lengths(&lit_lengths, &dist_lengths, &mut cl_freqs);

    let cl_codes = huffman::build_codes(&cl_freqs, 7);
    let cl_rev = prepare_reversed_codes(&cl_codes);

    let cl_order: [usize; 19] = [
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    ];
    // Find last non-zero code length in cl_order; HCLEN = index, clamped to [0, 15]
    let mut hclen = 0u8;
    for (i, &idx) in cl_order.iter().enumerate().rev() {
        if cl_codes[idx].length > 0 {
            hclen = i.min(15) as u8; // clamp to 4-bit max
            break;
        }
    }

    let mut writer = BitWriter64::with_capacity(capacity_hint);
    writer.write_bits(1, 1); // BFINAL (single block)
    writer.write_bits(2, 2); // BTYPE=10 (dynamic)

    writer.write_bits(hlit as u32, 5); // HLIT
    writer.write_bits(hdist as u32, 5); // HDIST
    writer.write_bits(hclen as u32, 4); // HCLEN

    // Write code length code lengths in order (3-bit values per RFC 1951)
    for &idx in cl_order.iter().take(hclen as usize + 4) {
        writer.write_bits(cl_codes[idx].length as u32, 3);
    }

    for (sym, extra_bits, extra_len) in rle {
        let (code, len) = cl_rev[sym as usize];
        writer.write_bits(code, len);
        if extra_len > 0 {
            writer.write_bits(extra_bits as u32, extra_len);
        }
    }

    for token in tokens {
        if let Some(b) = token.as_literal() {
            let (code, len) = lit_rev[b as usize];
            writer.write_bits(code, len);
        } else if let Some((length, distance)) = token.as_match() {
            let (len_symbol, len_extra_bits, len_extra_value) = length_code(length);
            let (len_code, len_len) = lit_rev[len_symbol as usize];
            writer.write_bits(len_code, len_len);
            if len_extra_bits > 0 {
                writer.write_bits(len_extra_value as u32, len_extra_bits);
            }

            let (dist_symbol, dist_extra_bits, dist_extra_value) = distance_code(distance);
            let (dist_code, dist_len) = dist_rev[dist_symbol as usize];
            writer.write_bits(dist_code, dist_len);
            if dist_extra_bits > 0 {
                writer.write_bits(dist_extra_value as u32, dist_extra_bits);
            }
        }
    }

    let (eob_code, eob_len) = lit_rev[256];
    writer.write_bits(eob_code, eob_len);

    writer.finish()
}

fn last_nonzero(lengths: &[u8]) -> usize {
    lengths
        .iter()
        .rposition(|&l| l != 0)
        .map(|i| i + 1)
        .unwrap_or(1) // minimum 1 code
}

/// Count literal and match tokens for stats.
fn token_counts(tokens: &[Token]) -> (usize, usize) {
    let mut literal_count = 0;
    let mut match_count = 0;
    for token in tokens {
        match token {
            Token::Literal(_) => literal_count += 1,
            Token::Match { .. } => match_count += 1,
        }
    }
    (literal_count, match_count)
}

/// Heuristic estimate for compressed size in bytes to pre-allocate writers.
fn estimated_deflate_size(data_len: usize, level: u8) -> usize {
    let est = match level {
        1..=3 => data_len.saturating_mul(9) / 10, // fast modes ~90%
        4..=6 => data_len.saturating_mul(7) / 10, // default ~70%
        7..=9 => data_len.saturating_mul(6) / 10, // max ~60%
        _ => data_len,
    };
    est.max(1024)
}

/// RLE encode literal/dist code lengths and collect code length code frequencies.
fn rle_code_lengths(
    lit_lengths: &[u8],
    dist_lengths: &[u8],
    cl_freqs: &mut [u32],
) -> Vec<(u8, u8, u8)> {
    let mut seq = Vec::new();
    seq.extend_from_slice(lit_lengths);
    seq.extend_from_slice(dist_lengths);

    let mut encoded = Vec::new();
    let mut i = 0;
    while i < seq.len() {
        let curr = seq[i];
        let mut run = 1;
        while i + run < seq.len() && seq[i + run] == curr {
            run += 1;
        }

        if curr == 0 {
            let mut rem = run;
            while rem > 0 {
                if rem >= 11 {
                    let take = rem.min(138);
                    encoded.push((18, (take - 11) as u8, 7));
                    cl_freqs[18] += 1;
                    rem -= take;
                } else if rem >= 3 {
                    let take = rem.min(10);
                    encoded.push((17, (take - 3) as u8, 3));
                    cl_freqs[17] += 1;
                    rem -= take;
                } else {
                    encoded.push((0, 0, 0));
                    cl_freqs[0] += 1;
                    rem -= 1;
                }
            }
        } else {
            // emit first occurrence
            encoded.push((curr, 0, 0));
            cl_freqs[curr as usize] += 1;
            let mut rem = run - 1;
            while rem >= 3 {
                let take = rem.min(6);
                encoded.push((16, (take - 3) as u8, 2));
                cl_freqs[16] += 1;
                rem -= take;
            }
            while rem > 0 {
                encoded.push((curr, 0, 0));
                cl_freqs[curr as usize] += 1;
                rem -= 1;
            }
        }

        i += run;
    }

    encoded
}

/// Lookup table for reversing the bits in a byte.
const REVERSE_BYTE: [u8; 256] = {
    let mut table = [0u8; 256];
    let mut i = 0usize;
    while i < 256 {
        let mut b = i as u8;
        let mut r = 0u8;
        let mut j = 0;
        while j < 8 {
            r = (r << 1) | (b & 1);
            b >>= 1;
            j += 1;
        }
        table[i] = r;
        i += 1;
    }
    table
};

/// Reverse bits in a code (DEFLATE uses reversed bit order for Huffman codes).
/// Uses a lookup table for O(1) byte reversal.
#[inline]
fn reverse_bits(code: u16, length: u8) -> u32 {
    if length == 0 {
        return 0;
    }
    // Reverse all 16 bits using byte lookup table, then shift to correct position
    let low = REVERSE_BYTE[code as u8 as usize] as u16;
    let high = REVERSE_BYTE[(code >> 8) as u8 as usize] as u16;
    let reversed = (low << 8) | high;
    (reversed >> (16 - length)) as u32
}

/// Precompute reversed codes for a slice of Huffman codes.
#[inline]
fn prepare_reversed_codes(codes: &[huffman::HuffmanCode]) -> Vec<(u32, u8)> {
    codes
        .iter()
        .map(|c| (reverse_bits(c.code, c.length), c.length))
        .collect()
}

#[inline]
fn empty_deflate_fixed_block() -> Vec<u8> {
    let mut writer = BitWriter64::with_capacity(16);
    writer.write_bits(1, 1); // BFINAL = 1
    writer.write_bits(1, 2); // BTYPE = 01 (fixed Huffman)

    let (code, len) = fixed_literal_codes_rev()[256];
    writer.write_bits(code, len);
    writer.finish()
}

#[inline]
fn empty_zlib(level: u8) -> Vec<u8> {
    let mut output = Vec::with_capacity(8);
    output.extend_from_slice(&zlib_header(level));
    output.extend_from_slice(&empty_deflate_fixed_block());
    output.extend_from_slice(&adler32(&[]).to_be_bytes());
    output
}

/// Cached reversed fixed literal codes.
static FIXED_LIT_REV: LazyLock<[(u32, u8); 288]> = LazyLock::new(|| {
    let codes = huffman::fixed_literal_codes();
    let mut out = [(0u32, 0u8); 288];
    for (i, c) in codes.iter().enumerate() {
        out[i] = (reverse_bits(c.code, c.length), c.length);
    }
    out
});

/// Cached reversed fixed distance codes.
static FIXED_DIST_REV: LazyLock<[(u32, u8); 32]> = LazyLock::new(|| {
    let codes = huffman::fixed_distance_codes();
    let mut out = [(0u32, 0u8); 32];
    for (i, c) in codes.iter().enumerate() {
        out[i] = (reverse_bits(c.code, c.length), c.length);
    }
    out
});

#[inline]
fn fixed_literal_codes_rev() -> &'static [(u32, u8); 288] {
    &FIXED_LIT_REV
}

#[inline]
fn fixed_distance_codes_rev() -> &'static [(u32, u8); 32] {
    &FIXED_DIST_REV
}

/// Build the two-byte zlib header for the given compression level.
fn zlib_header(level: u8) -> [u8; 2] {
    // CMF: 0b0111_1000 (Deflate, 32K window)
    let cmf: u8 = 0x78;

    // Map level to FLEVEL (informative only)
    let flevel = match level {
        0..=2 => 1, // fast
        3..=6 => 2, // default
        _ => 3,     // maximum
    };

    let mut flg: u8 = flevel << 6; // FDICT=0
    let fcheck = (31 - (((cmf as u16) << 8 | flg as u16) % 31)) % 31;
    flg |= fcheck as u8;

    [cmf, flg]
}

/// Compress data using DEFLATE with stored blocks (no compression).
/// Useful for already-compressed data or when speed is critical.
#[allow(dead_code)]
#[must_use]
pub fn deflate_stored(data: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(data.len() + data.len() / 65535 * 5 + 10);
    let chunks = data.chunks(65535);
    let num_chunks = chunks.len();

    for (i, chunk) in data.chunks(65535).enumerate() {
        let is_final = i == num_chunks - 1;
        let len = chunk.len() as u16;
        let nlen = !len;

        // Block header
        output.push(if is_final { 0x01 } else { 0x00 }); // BFINAL + BTYPE=00

        // LEN and NLEN (little-endian)
        output.push(len as u8);
        output.push((len >> 8) as u8);
        output.push(nlen as u8);
        output.push((nlen >> 8) as u8);

        // Data
        output.extend_from_slice(chunk);
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::read::ZlibDecoder;
    use rand::{Rng, SeedableRng};
    use std::io::Read;

    #[test]
    fn test_length_code() {
        assert_eq!(length_code(3), (257, 0, 0));
        assert_eq!(length_code(4), (258, 0, 0));
        assert_eq!(length_code(10), (264, 0, 0));
        assert_eq!(length_code(11), (265, 1, 0));
        assert_eq!(length_code(12), (265, 1, 1));
        assert_eq!(length_code(258), (285, 0, 0));
    }

    #[test]
    fn test_distance_code() {
        assert_eq!(distance_code(1), (0, 0, 0));
        assert_eq!(distance_code(2), (1, 0, 0));
        assert_eq!(distance_code(5), (4, 1, 0));
        assert_eq!(distance_code(6), (4, 1, 1));
    }

    #[test]
    fn test_deflate_empty() {
        let compressed = deflate(&[], 6);
        assert!(!compressed.is_empty());
    }

    #[test]
    fn test_deflate_simple() {
        let data = b"Hello, World!";
        let compressed = deflate(data, 6);

        // Should produce some output
        assert!(!compressed.is_empty());
        // For short data, compression might not reduce size much
    }

    #[test]
    fn test_deflate_repetitive() {
        let data = b"abcabcabcabcabcabcabcabcabcabc";
        let compressed = deflate(data, 6);

        // Repetitive data should compress well
        assert!(compressed.len() < data.len());
    }

    #[test]
    fn test_deflate_zlib_header_checksum() {
        let data = b"hello";
        let compressed = deflate_zlib(data, 6);

        // Header should be 0x78 0x9C for default-ish compression
        assert_eq!(&compressed[0..2], &[0x78, 0x9C]);

        let checksum = u32::from_be_bytes(compressed[compressed.len() - 4..].try_into().unwrap());
        assert_eq!(checksum, 0x062C0215);
    }

    #[test]
    fn test_deflate_stored() {
        let data = b"Hello, World!";
        let compressed = deflate_stored(data);

        // Stored blocks have 5 bytes overhead per 65535 bytes
        assert_eq!(compressed.len(), data.len() + 5);
    }

    #[test]
    fn test_reverse_bits() {
        assert_eq!(reverse_bits(0b101, 3), 0b101);
        assert_eq!(reverse_bits(0b100, 3), 0b001);
        assert_eq!(reverse_bits(0b11110000, 8), 0b00001111);
    }

    #[test]
    fn test_should_use_stored_threshold() {
        // Deflated larger than stored -> use stored
        assert!(should_use_stored(1000, 1200));
        // Deflated smaller -> keep deflated
        assert!(!should_use_stored(1000, 400));
        // Near-equal totals prefer stored
        assert!(should_use_stored(1000, 1010));
    }

    fn decompress_zlib(data: &[u8]) -> Vec<u8> {
        let mut decoder = ZlibDecoder::new(data);
        let mut out = Vec::new();
        decoder.read_to_end(&mut out).expect("zlib decode");
        out
    }

    #[test]
    fn test_deflate_zlib_empty_decode() {
        let encoded = deflate_zlib(&[], 6);
        let decoded = decompress_zlib(&encoded);
        assert!(decoded.is_empty());
        // Header for default compression and empty body should be minimal
        assert!(encoded.len() <= 11); // 2 header + 5 stored + 4 adler
    }

    #[test]
    fn test_deflate_zlib_roundtrip_random_small() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(999);
        for len in [0usize, 1, 2, 5, 32, 128, 1024, 4096] {
            let mut data = vec![0u8; len];
            rng.fill(data.as_mut_slice());
            let encoded = deflate_zlib(&data, 6);
            let decoded = decompress_zlib(&encoded);
            assert_eq!(decoded, data, "mismatch at len={len}");
        }
    }

    #[test]
    fn test_deflate_zlib_packed_roundtrip_random_small() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1001);
        for len in [0usize, 1, 2, 5, 32, 128, 1024, 4096] {
            let mut data = vec![0u8; len];
            rng.fill(data.as_mut_slice());
            let encoded = deflate_zlib_packed(&data, 6);
            let decoded = decompress_zlib(&encoded);
            assert_eq!(decoded, data, "mismatch at len={len}");
        }
    }

    #[test]
    fn test_deflate_zlib_packed_matches_standard_output() {
        let data = b"The quick brown fox jumps over the lazy dog. Pack me tightly!";
        let std_encoded = deflate_zlib(data, 6);
        let packed_encoded = deflate_zlib_packed(data, 6);
        assert_eq!(std_encoded, packed_encoded);
    }

    #[test]
    fn test_deflate_zlib_incompressible_prefers_stored() {
        let mut data = vec![0u8; 10_000];
        // High-entropy pattern to discourage compression
        let mut rng = rand::rngs::StdRng::seed_from_u64(1234);
        rng.fill(data.as_mut_slice());

        let encoded = deflate_zlib(&data, 6);
        let decoded = decompress_zlib(&encoded);
        assert_eq!(decoded, data);
        // stored overhead per 65535 block is 5 bytes + header/adler
        let stored_overhead = (data.len() / 65_535 + 1) * 5 + 2 + 4;
        assert!(encoded.len() <= data.len() + stored_overhead);
    }

    #[test]
    fn test_packed_fixed_matches_standard() {
        let data = b"aaaaabbbbccddeeffgg";
        let mut lz = Lz77Compressor::new(6);
        let tokens = lz.compress(data);

        let mut packed = Vec::with_capacity(tokens.len());
        for t in &tokens {
            match t {
                Token::Literal(b) => packed.push(PackedToken::literal(*b)),
                Token::Match { length, distance } => {
                    packed.push(PackedToken::match_(*length, *distance))
                }
            }
        }

        let std_out = encode_fixed_huffman(&tokens);
        let packed_out = encode_fixed_huffman_packed(&packed);
        assert_eq!(std_out, packed_out);
    }

    #[test]
    fn test_packed_dynamic_matches_standard() {
        let data = b"The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.";
        let mut lz = Lz77Compressor::new(6);
        let tokens = lz.compress(data);

        let mut packed = Vec::with_capacity(tokens.len());
        for t in &tokens {
            match t {
                Token::Literal(b) => packed.push(PackedToken::literal(*b)),
                Token::Match { length, distance } => {
                    packed.push(PackedToken::match_(*length, *distance))
                }
            }
        }

        let std_out = encode_dynamic_huffman(&tokens);
        let packed_out = encode_dynamic_huffman_packed(&packed);
        assert_eq!(std_out, packed_out);
    }

    #[test]
    fn test_dynamic_huffman_decode() {
        use flate2::read::DeflateDecoder;
        use std::io::Read;

        // Data that produces dynamic Huffman output
        let data = b"The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.";
        let mut lz = Lz77Compressor::new(6);
        let tokens = lz.compress(data);

        let dynamic_out = encode_dynamic_huffman(&tokens);

        // Verify it decodes correctly
        let mut decoder = DeflateDecoder::new(&dynamic_out[..]);
        let mut decoded = Vec::new();
        decoder
            .read_to_end(&mut decoded)
            .expect("dynamic Huffman should decode");

        assert_eq!(decoded, data.to_vec(), "dynamic Huffman roundtrip failed");
    }

    #[test]
    fn test_deflate_optimal_empty() {
        let compressed = deflate_optimal(&[], 3);
        assert!(!compressed.is_empty());
    }

    #[test]
    fn test_deflate_optimal_roundtrip() {
        use flate2::read::DeflateDecoder;
        use std::io::Read;

        let data = b"The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.";
        let compressed = deflate_optimal(data, 3);

        // Verify it decodes correctly
        let mut decoder = DeflateDecoder::new(&compressed[..]);
        let mut decoded = Vec::new();
        decoder
            .read_to_end(&mut decoded)
            .expect("optimal deflate should decode");

        assert_eq!(decoded, data.to_vec(), "optimal deflate roundtrip failed");
    }

    #[test]
    fn test_deflate_optimal_compresses_better() {
        // For repetitive data, optimal parsing should be at least as good as greedy
        let data = b"abcdefghijklmnopqrstuvwxyz".repeat(100);

        let greedy = deflate(&data, 9);
        let optimal = deflate_optimal(&data, 5);

        // Optimal should be no larger than greedy (and usually smaller)
        assert!(
            optimal.len() <= greedy.len() + 10, // Allow small margin due to block header differences
            "optimal ({}) should not be much larger than greedy ({})",
            optimal.len(),
            greedy.len()
        );
    }

    #[test]
    fn test_deflate_optimal_zlib_roundtrip() {
        let data = b"This is a test of optimal DEFLATE compression with zlib wrapper. This is a test of optimal DEFLATE compression with zlib wrapper.";
        let compressed = deflate_optimal_zlib(data, 3);

        let decoded = decompress_zlib(&compressed);
        assert_eq!(decoded, data.to_vec(), "optimal zlib roundtrip failed");
    }

    #[test]
    fn test_count_symbols() {
        let tokens = vec![
            Token::Literal(b'a'),
            Token::Literal(b'b'),
            Token::Match {
                length: 3,
                distance: 2,
            },
        ];

        let (lit_counts, dist_counts) = count_symbols(&tokens);

        // Check that 'a' and 'b' are counted
        assert_eq!(lit_counts[b'a' as usize], 1);
        assert_eq!(lit_counts[b'b' as usize], 1);

        // Check that length code 257 (for length 3) is counted
        assert_eq!(lit_counts[257], 1);

        // Check that end of block is counted
        assert_eq!(lit_counts[256], 1);

        // Check distance code 1 (for distance 2) is counted
        assert_eq!(dist_counts[1], 1);
    }

    #[test]
    fn test_deflate_optimal_split_empty() {
        let compressed = deflate_optimal_split(&[], 3, 15);
        assert!(!compressed.is_empty());
    }

    #[test]
    fn test_deflate_optimal_split_roundtrip() {
        use flate2::read::DeflateDecoder;
        use std::io::Read;

        // Use a larger data set that might benefit from block splitting
        let data = b"The quick brown fox jumps over the lazy dog. ".repeat(100);
        let compressed = deflate_optimal_split(&data, 3, 15);

        // Verify it decodes correctly
        let mut decoder = DeflateDecoder::new(&compressed[..]);
        let mut decoded = Vec::new();
        decoder
            .read_to_end(&mut decoded)
            .expect("optimal split deflate should decode");

        assert_eq!(decoded, data, "optimal split deflate roundtrip failed");
    }

    #[test]
    fn test_deflate_optimal_split_zlib_roundtrip() {
        // Test with varying data that might benefit from block splitting
        let mut data = Vec::new();
        // First section: repetitive text
        data.extend_from_slice(&b"abcdefgh".repeat(200));
        // Second section: different pattern
        data.extend_from_slice(&b"12345678".repeat(200));
        // Third section: mixed content
        data.extend_from_slice(
            b"The quick brown fox jumps over the lazy dog. "
                .repeat(50)
                .as_slice(),
        );

        let compressed = deflate_optimal_split_zlib(&data, 3, 15);
        let decoded = decompress_zlib(&compressed);
        assert_eq!(decoded, data, "optimal split zlib roundtrip failed");
    }

    #[test]
    fn test_block_splitting_finds_splits_for_varied_data() {
        // Create data with distinct statistical regions
        let mut lz77 = Lz77Compressor::new(6);

        let mut data = Vec::new();
        // Region 1: lots of 'a'
        data.extend_from_slice(&[b'a'; 1000]);
        // Region 2: lots of 'z'
        data.extend_from_slice(&[b'z'; 1000]);

        let tokens = lz77.compress(&data);

        // Should find at least some split points for data with distinct regions
        // (though whether it actually splits depends on whether it saves bits)
        let splits = find_block_splits(&tokens, 10);

        // Just verify it doesn't crash and returns a valid result
        assert!(splits.len() <= 9); // At most max_blocks - 1 splits
    }
}
