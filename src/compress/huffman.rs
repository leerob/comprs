//! Huffman coding implementation for DEFLATE.
//!
//! This module provides Huffman tree construction and canonical code generation.

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::LazyLock;

/// Maximum code length for DEFLATE (15 bits for literals/lengths, 7 for distances).
pub const MAX_CODE_LENGTH: usize = 15;

/// Huffman code: (code bits, length in bits).
#[derive(Debug, Clone, Copy, Default)]
pub struct HuffmanCode {
    /// The code bits (right-aligned).
    pub code: u16,
    /// Number of bits in the code.
    pub length: u8,
}

/// Huffman tree node for construction.
#[derive(Debug, Clone, Eq, PartialEq)]
struct Node {
    frequency: u32,
    symbol: Option<u16>,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Compare by frequency, then by symbol for stability
        self.frequency
            .cmp(&other.frequency)
            .then_with(|| self.symbol.cmp(&other.symbol))
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Build Huffman codes from symbol frequencies.
///
/// Returns a vector where index is the symbol and value is its Huffman code.
pub fn build_codes(frequencies: &[u32], max_length: usize) -> Vec<HuffmanCode> {
    let num_symbols = frequencies.len();
    if num_symbols == 0 {
        return Vec::new();
    }

    // Count non-zero frequencies
    let non_zero: Vec<(u16, u32)> = frequencies
        .iter()
        .enumerate()
        .filter_map(|(i, &f)| (f > 0).then_some((i as u16, f)))
        .collect();

    if non_zero.is_empty() {
        return vec![HuffmanCode::default(); num_symbols];
    }

    // Special case: only one symbol
    if non_zero.len() == 1 {
        let mut codes = vec![HuffmanCode::default(); num_symbols];
        codes[non_zero[0].0 as usize] = HuffmanCode { code: 0, length: 1 };
        return codes;
    }

    // Build Huffman tree
    let mut heap: BinaryHeap<Reverse<Node>> = non_zero
        .iter()
        .map(|&(sym, freq)| {
            Reverse(Node {
                frequency: freq,
                symbol: Some(sym),
                left: None,
                right: None,
            })
        })
        .collect();

    while heap.len() > 1 {
        let Reverse(left) = heap.pop().unwrap();
        let Reverse(right) = heap.pop().unwrap();

        let parent = Node {
            frequency: left.frequency + right.frequency,
            symbol: None,
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
        };

        heap.push(Reverse(parent));
    }

    // Extract code lengths from tree
    let root = heap.pop().unwrap().0;
    let mut code_lengths = vec![0u8; num_symbols];
    extract_lengths(&root, 0, &mut code_lengths);

    // Limit code lengths if necessary
    limit_code_lengths(&mut code_lengths, max_length);

    // Generate canonical Huffman codes from lengths
    generate_canonical_codes(&code_lengths)
}

/// Extract code lengths from Huffman tree via DFS.
fn extract_lengths(node: &Node, depth: u8, lengths: &mut [u8]) {
    if let Some(symbol) = node.symbol {
        lengths[symbol as usize] = depth.max(1); // At least 1 bit
    } else {
        if let Some(ref left) = node.left {
            extract_lengths(left, depth + 1, lengths);
        }
        if let Some(ref right) = node.right {
            extract_lengths(right, depth + 1, lengths);
        }
    }
}

/// Limit code lengths to max_length using a simple redistribution algorithm.
///
/// This ensures the Huffman tree is complete (Kraft sum equals 2^max_length).
fn limit_code_lengths(lengths: &mut [u8], max_length: usize) {
    let max_length = max_length as u8;

    // Check if any codes exceed max length
    let has_overflow = lengths.iter().any(|&l| l > max_length);

    if !has_overflow {
        return;
    }

    // Truncate long codes to max_length
    for length in lengths.iter_mut() {
        if *length > max_length {
            *length = max_length;
        }
    }

    // Calculate current Kraft sum
    let kraft_limit = 1u32 << max_length;
    let mut kraft_sum: u32 = lengths
        .iter()
        .filter(|&&l| l > 0)
        .map(|&l| 1u32 << (max_length as u32 - l as u32))
        .sum();

    // If over-subscribed, make some codes longer until we're at or under the limit
    while kraft_sum > kraft_limit {
        // Find the shortest non-zero code that can be made longer
        let mut best_idx = None;
        let mut best_len = max_length;
        for (i, &len) in lengths.iter().enumerate() {
            if len > 0 && len < max_length && len < best_len {
                best_len = len;
                best_idx = Some(i);
            }
        }

        if let Some(idx) = best_idx {
            kraft_sum -= 1u32 << (max_length as u32 - lengths[idx] as u32);
            lengths[idx] += 1;
            kraft_sum += 1u32 << (max_length as u32 - lengths[idx] as u32);
        } else {
            // All codes are already at max_length, can't fix
            break;
        }
    }

    // If under-subscribed, make some codes shorter to fill the gap
    // We do this by finding codes at max_length and reducing them
    while kraft_sum < kraft_limit {
        // Find a code at the longest length that can be made shorter
        let mut best_idx = None;
        let mut best_len = 0u8;
        for (i, &len) in lengths.iter().enumerate() {
            if len > 1 && len > best_len {
                best_len = len;
                best_idx = Some(i);
            }
        }

        if let Some(idx) = best_idx {
            let old_contribution = 1u32 << (max_length as u32 - lengths[idx] as u32);
            let new_contribution = 1u32 << (max_length as u32 - (lengths[idx] - 1) as u32);

            // Check if making this code shorter would overshoot
            if kraft_sum - old_contribution + new_contribution <= kraft_limit {
                kraft_sum -= old_contribution;
                lengths[idx] -= 1;
                kraft_sum += new_contribution;
            } else {
                // Can't shorten without overshooting, we're done
                break;
            }
        } else {
            break;
        }
    }
}

/// Generate canonical Huffman codes from code lengths.
///
/// Canonical codes are generated such that:
/// 1. Shorter codes come before longer codes
/// 2. Codes of the same length are assigned in symbol order
pub fn generate_canonical_codes(lengths: &[u8]) -> Vec<HuffmanCode> {
    let num_symbols = lengths.len();
    let mut codes = vec![HuffmanCode::default(); num_symbols];

    // Count codes of each length
    let mut bl_count = [0u32; MAX_CODE_LENGTH + 1];
    for &length in lengths {
        if length > 0 {
            bl_count[length as usize] += 1;
        }
    }

    // Calculate starting code for each length
    let mut next_code = [0u16; MAX_CODE_LENGTH + 1];
    let mut code = 0u16;
    for bits in 1..=MAX_CODE_LENGTH {
        code = (code + bl_count[bits - 1] as u16) << 1;
        next_code[bits] = code;
    }

    // Assign codes to symbols
    for (symbol, &length) in lengths.iter().enumerate() {
        if length > 0 {
            codes[symbol] = HuffmanCode {
                code: next_code[length as usize],
                length,
            };
            next_code[length as usize] += 1;
        }
    }

    codes
}

/// Cached fixed Huffman codes for literal/length symbols (0-287).
static FIXED_LITERAL_CODES: LazyLock<Vec<HuffmanCode>> = LazyLock::new(|| {
    let mut lengths = vec![0u8; 288];

    // 0-143: 8 bits
    for length in lengths.iter_mut().take(144) {
        *length = 8;
    }
    // 144-255: 9 bits
    for length in lengths.iter_mut().take(256).skip(144) {
        *length = 9;
    }
    // 256-279: 7 bits
    for length in lengths.iter_mut().take(280).skip(256) {
        *length = 7;
    }
    // 280-287: 8 bits
    for length in lengths.iter_mut().take(288).skip(280) {
        *length = 8;
    }

    generate_canonical_codes(&lengths)
});

/// Cached fixed Huffman codes for distance symbols (0-31).
static FIXED_DISTANCE_CODES: LazyLock<Vec<HuffmanCode>> = LazyLock::new(|| {
    // All distance codes are 5 bits in fixed Huffman
    let lengths = vec![5u8; 32];
    generate_canonical_codes(&lengths)
});

/// DEFLATE fixed Huffman codes for literal/length symbols (0-287).
/// Returns a cached reference for O(1) access after first call.
#[inline]
pub fn fixed_literal_codes() -> &'static [HuffmanCode] {
    &FIXED_LITERAL_CODES
}

/// DEFLATE fixed Huffman codes for distance symbols (0-31).
/// Returns a cached reference for O(1) access after first call.
#[inline]
pub fn fixed_distance_codes() -> &'static [HuffmanCode] {
    &FIXED_DISTANCE_CODES
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_codes_simple() {
        // Frequencies: a=5, b=2, c=1, d=1
        let freqs = [5, 2, 1, 1];
        let codes = build_codes(&freqs, 15);

        // All symbols should have codes
        assert!(codes[0].length > 0);
        assert!(codes[1].length > 0);
        assert!(codes[2].length > 0);
        assert!(codes[3].length > 0);

        // More frequent symbols should have shorter codes
        assert!(codes[0].length <= codes[2].length);
        assert!(codes[0].length <= codes[3].length);
    }

    #[test]
    fn test_build_codes_single_symbol() {
        let freqs = [0, 0, 5, 0];
        let codes = build_codes(&freqs, 15);

        assert_eq!(codes[2].length, 1);
        assert_eq!(codes[0].length, 0);
        assert_eq!(codes[1].length, 0);
        assert_eq!(codes[3].length, 0);
    }

    #[test]
    fn test_canonical_codes_prefix_free() {
        let freqs = [10, 5, 3, 2, 1, 1, 1, 1];
        let codes = build_codes(&freqs, 15);

        // Verify prefix-free property
        for i in 0..codes.len() {
            for j in (i + 1)..codes.len() {
                if codes[i].length > 0 && codes[j].length > 0 {
                    let min_len = codes[i].length.min(codes[j].length);
                    let mask = (1u16 << min_len) - 1;
                    let prefix_i = codes[i].code >> (codes[i].length - min_len);
                    let prefix_j = codes[j].code >> (codes[j].length - min_len);
                    assert_ne!(
                        prefix_i & mask,
                        prefix_j & mask,
                        "Codes {i} and {j} share prefix",
                    );
                }
            }
        }
    }

    #[test]
    fn test_fixed_literal_codes() {
        let codes = fixed_literal_codes();
        assert_eq!(codes.len(), 288);

        // Check expected lengths
        assert_eq!(codes[0].length, 8); // 0-143 are 8 bits
        assert_eq!(codes[143].length, 8);
        assert_eq!(codes[144].length, 9); // 144-255 are 9 bits
        assert_eq!(codes[255].length, 9);
        assert_eq!(codes[256].length, 7); // 256-279 are 7 bits
        assert_eq!(codes[279].length, 7);
        assert_eq!(codes[280].length, 8); // 280-287 are 8 bits
    }

    #[test]
    fn test_fixed_distance_codes() {
        let codes = fixed_distance_codes();
        assert_eq!(codes.len(), 32);

        // All should be 5 bits
        for code in codes {
            assert_eq!(code.length, 5);
        }
    }

    #[test]
    fn test_code_lengths_respect_max_length() {
        // Many symbols with non-zero frequency should still produce codes
        // that do not exceed the specified max length (15 for DEFLATE literals).
        let freqs = vec![1u32; 400]; // more than the DEFLATE literal alphabet size
        let codes = build_codes(&freqs, 15);
        for code in codes {
            assert!(
                code.length <= 15,
                "code length exceeded max: {}",
                code.length
            );
        }
    }

    #[test]
    fn test_build_codes_empty() {
        let freqs: [u32; 0] = [];
        let codes = build_codes(&freqs, 15);
        assert!(codes.is_empty());
    }

    #[test]
    fn test_build_codes_all_zero() {
        let freqs = [0, 0, 0, 0];
        let codes = build_codes(&freqs, 15);
        assert_eq!(codes.len(), 4);
        for code in &codes {
            assert_eq!(code.length, 0);
        }
    }

    #[test]
    fn test_generate_canonical_codes_directly() {
        // Test the canonical code generation function directly
        let lengths = [2, 1, 3, 3];
        let codes = generate_canonical_codes(&lengths);

        assert_eq!(codes.len(), 4);
        // Symbol 1 has length 1, so should have code 0
        assert_eq!(codes[1].length, 1);
        assert_eq!(codes[1].code, 0);

        // Symbol 0 has length 2, should start after symbol 1
        assert_eq!(codes[0].length, 2);
    }

    #[test]
    fn test_huffman_code_default() {
        let code = HuffmanCode::default();
        assert_eq!(code.code, 0);
        assert_eq!(code.length, 0);
    }

    #[test]
    fn test_limit_code_lengths_no_overflow() {
        // When no codes exceed max length, should be unchanged
        let freqs = [10, 5, 3, 2];
        let codes = build_codes(&freqs, 15);
        for code in &codes {
            assert!(code.length <= 15);
        }
    }

    #[test]
    fn test_limit_code_lengths_with_overflow() {
        // Force a situation where initial tree would be too deep
        // by having exponentially growing frequencies
        let mut freqs = vec![1u32; 32];
        for i in 0..32 {
            freqs[i] = 1 << i;
        }
        // Use a small max length to force limiting
        let codes = build_codes(&freqs, 7);
        for code in &codes {
            if code.length > 0 {
                assert!(
                    code.length <= 7,
                    "code length {} exceeds max 7",
                    code.length
                );
            }
        }
    }

    #[test]
    fn test_build_codes_two_symbols() {
        let freqs = [10, 5];
        let codes = build_codes(&freqs, 15);

        assert_eq!(codes.len(), 2);
        // Both should have length 1 (perfect binary tree)
        assert_eq!(codes[0].length, 1);
        assert_eq!(codes[1].length, 1);
        // They should have different codes
        assert_ne!(codes[0].code, codes[1].code);
    }

    #[test]
    fn test_build_codes_equal_frequencies() {
        let freqs = [10, 10, 10, 10];
        let codes = build_codes(&freqs, 15);

        // All should have the same length (balanced tree)
        let lengths: Vec<u8> = codes.iter().map(|c| c.length).collect();
        assert!(lengths.iter().all(|&l| l == 2));
    }

    #[test]
    fn test_build_codes_large_alphabet() {
        // Test with DEFLATE literal alphabet size
        let mut freqs = vec![1u32; 288];
        freqs[0] = 100;
        freqs[32] = 50;
        freqs[101] = 25;

        let codes = build_codes(&freqs, 15);
        assert_eq!(codes.len(), 288);

        // More frequent symbols should have shorter codes
        assert!(codes[0].length <= codes[255].length);
    }

    #[test]
    fn test_build_codes_max_length_7() {
        // Test with DEFLATE distance max length
        let freqs = vec![1u32; 30];
        let codes = build_codes(&freqs, 7);

        for code in &codes {
            if code.length > 0 {
                assert!(code.length <= 7);
            }
        }
    }

    #[test]
    fn test_generate_canonical_codes_all_zeros() {
        let lengths = [0u8; 10];
        let codes = generate_canonical_codes(&lengths);

        assert_eq!(codes.len(), 10);
        for code in &codes {
            assert_eq!(code.length, 0);
            assert_eq!(code.code, 0);
        }
    }

    #[test]
    fn test_generate_canonical_codes_all_same_length() {
        let lengths = [3u8; 8];
        let codes = generate_canonical_codes(&lengths);

        // All have length 3
        for code in &codes {
            assert_eq!(code.length, 3);
        }

        // Codes should be sequential
        for i in 0..8 {
            assert_eq!(codes[i].code, i as u16);
        }
    }

    #[test]
    fn test_fixed_codes_cached() {
        // First access creates the codes
        let codes1 = fixed_literal_codes();
        // Second access should return the same reference
        let codes2 = fixed_literal_codes();
        assert!(std::ptr::eq(codes1, codes2));
    }

    #[test]
    fn test_fixed_distance_codes_cached() {
        let codes1 = fixed_distance_codes();
        let codes2 = fixed_distance_codes();
        assert!(std::ptr::eq(codes1, codes2));
    }

    #[test]
    fn test_build_codes_kraft_inequality() {
        // Kraft inequality: sum of 2^(-length) <= 1
        // For our representation: sum of 2^(max_length - length) <= 2^max_length
        let freqs = [10, 5, 3, 2, 1, 1, 1, 1];
        let codes = build_codes(&freqs, 15);

        let kraft_sum: u32 = codes
            .iter()
            .filter(|c| c.length > 0)
            .map(|c| 1u32 << (15 - c.length as u32))
            .sum();

        assert!(kraft_sum <= (1u32 << 15), "Kraft inequality violated");
    }

    #[test]
    fn test_build_codes_unique_codes() {
        let freqs = [10, 5, 3, 2, 1, 1, 1, 1];
        let codes = build_codes(&freqs, 15);

        // All non-zero codes should be unique (within same length)
        let mut length_codes: std::collections::HashMap<u8, Vec<u16>> =
            std::collections::HashMap::new();

        for code in &codes {
            if code.length > 0 {
                length_codes.entry(code.length).or_default().push(code.code);
            }
        }

        for (length, codes) in length_codes {
            let unique: std::collections::HashSet<_> = codes.iter().collect();
            assert_eq!(
                unique.len(),
                codes.len(),
                "Duplicate codes at length {length}"
            );
        }
    }

    #[test]
    fn test_build_codes_very_skewed_frequencies() {
        // One very common symbol, many rare ones
        let mut freqs = vec![1u32; 100];
        freqs[0] = 10000;

        let codes = build_codes(&freqs, 15);

        // The common symbol should have a short code
        assert!(codes[0].length < 5);
        // Rare symbols should have longer codes
        for code in codes.iter().skip(1) {
            if code.length > 0 {
                assert!(code.length >= codes[0].length);
            }
        }
    }

    #[test]
    fn test_fixed_literal_codes_eob() {
        let codes = fixed_literal_codes();
        // End of block (256) should have a valid code
        assert!(codes[256].length > 0);
    }

    #[test]
    fn test_fixed_literal_codes_length_symbols() {
        let codes = fixed_literal_codes();
        // Length symbols (257-285) should have valid codes
        for i in 257..=285 {
            assert!(codes[i].length > 0, "Length symbol {i} has no code");
        }
    }

    #[test]
    fn test_node_ordering() {
        // Test that nodes are ordered correctly for the priority queue
        let node1 = Node {
            frequency: 10,
            symbol: Some(1),
            left: None,
            right: None,
        };
        let node2 = Node {
            frequency: 20,
            symbol: Some(2),
            left: None,
            right: None,
        };

        // Lower frequency should come first
        assert!(node1 < node2);
    }

    #[test]
    fn test_node_ordering_same_frequency() {
        // Nodes with same frequency should be ordered by symbol
        let node1 = Node {
            frequency: 10,
            symbol: Some(1),
            left: None,
            right: None,
        };
        let node2 = Node {
            frequency: 10,
            symbol: Some(2),
            left: None,
            right: None,
        };

        // Lower symbol should come first when frequencies are equal
        assert!(node1 < node2);
    }
}
