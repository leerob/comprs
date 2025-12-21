//! LZ77 compression algorithm with sliding window.
//!
//! LZ77 finds repeated sequences in the input and replaces them with
//! (length, distance) pairs referring back to previous occurrences.

/// Maximum distance to look back for matches (32KB window).
pub const MAX_DISTANCE: usize = 32768;

/// Threshold for "good enough" match - skip lazy matching above this length.
/// This is a common optimization used by zlib to speed up compression.
const GOOD_MATCH_LENGTH: usize = 32;

/// Maximum match length (as per DEFLATE spec).
pub const MAX_MATCH_LENGTH: usize = 258;

/// Minimum match length worth encoding.
pub const MIN_MATCH_LENGTH: usize = 3;

/// Size of the hash table (power of 2 for fast modulo).
/// Larger table = fewer collisions = faster matching.
const HASH_SIZE: usize = 1 << 16; // 65536 entries (doubled from 32K)

/// LZ77 token representing either a literal or a match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Token {
    /// A literal byte that couldn't be compressed.
    Literal(u8),
    /// A back-reference: (length, distance).
    Match {
        /// Length of the match (3-258).
        length: u16,
        /// Distance back to the match (1-32768).
        distance: u16,
    },
}

/// Hash function for 4-byte sequences with better distribution.
/// Uses a multiplicative hash with a prime that provides good bit mixing.
#[inline]
fn hash4(data: &[u8], pos: usize) -> usize {
    if pos + 4 > data.len() {
        return 0;
    }
    // Load 4 bytes at once
    let val = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
    // Use a better multiplicative hash constant (0x1E35A7BD from xxHash)
    ((val.wrapping_mul(0x1E35A7BD)) >> 16) as usize & (HASH_SIZE - 1)
}

/// LZ77 compressor with hash chain for fast matching.
pub struct Lz77Compressor {
    /// Hash table: maps hash -> most recent position
    head: Vec<i32>,
    /// Chain links: prev[pos % window] -> previous position with same hash
    prev: Vec<i32>,
    /// Compression level (affects search depth)
    max_chain_length: usize,
    /// Lazy matching: check if next position has better match
    lazy_matching: bool,
    /// Nice match threshold - stop searching when we find a match this good
    nice_match: usize,
}

impl Lz77Compressor {
    /// Create a new LZ77 compressor.
    ///
    /// # Arguments
    /// * `level` - Compression level 1-9 (higher = better compression, slower)
    pub fn new(level: u8) -> Self {
        let level = level.clamp(1, 9);

        // Tune chain length, lazy matching, and nice match based on level
        let (max_chain_length, lazy_matching, nice_match) = match level {
            1 => (4, false, 32),
            2 => (8, false, 48),
            3 => (16, false, 64),
            4 => (32, true, 96),
            5 => (64, true, 128),
            6 => (128, true, 160),
            7 => (256, true, 192),
            8 => (512, true, 224),
            9 => (1024, true, MAX_MATCH_LENGTH),
            _ => (128, true, 160),
        };

        Self {
            head: vec![-1; HASH_SIZE],
            prev: vec![-1; MAX_DISTANCE],
            max_chain_length,
            lazy_matching,
            nice_match,
        }
    }

    /// Compress data and return LZ77 tokens.
    pub fn compress(&mut self, data: &[u8]) -> Vec<Token> {
        if data.is_empty() {
            return Vec::new();
        }

        // Pre-allocate with estimated capacity (tokens are ~60% of input for typical data)
        let mut tokens = Vec::with_capacity(data.len() * 2 / 3);
        let mut pos = 0;

        // Reset hash tables
        self.head.fill(-1);
        self.prev.fill(-1);

        while pos < data.len() {
            let best_match = self.find_best_match(data, pos);

            if let Some((length, distance)) = best_match {
                // Check for lazy match if enabled, but skip for "good enough" matches
                // This is a common optimization used by zlib
                if self.lazy_matching && length < GOOD_MATCH_LENGTH && pos + 1 < data.len() {
                    // Update hash for current position
                    self.update_hash(data, pos);

                    if let Some((next_length, _)) = self.find_best_match(data, pos + 1) {
                        if next_length > length + 1 {
                            // Better match at next position, emit literal
                            tokens.push(Token::Literal(data[pos]));
                            pos += 1;
                            continue;
                        }
                    }
                }

                tokens.push(Token::Match {
                    length: length as u16,
                    distance: distance as u16,
                });

                // Update hash for positions in the match
                // For longer matches, batch the hash updates
                self.update_hash_batch(data, pos, length);
                pos += length;
            } else {
                tokens.push(Token::Literal(data[pos]));
                self.update_hash(data, pos);
                pos += 1;
            }
        }

        tokens
    }

    /// Find the best match at the given position.
    /// Uses quick rejection filters and early exit thresholds for performance.
    fn find_best_match(&self, data: &[u8], pos: usize) -> Option<(usize, usize)> {
        if pos + MIN_MATCH_LENGTH > data.len() {
            return None;
        }

        let hash = hash4(data, pos);
        let mut chain_pos = self.head[hash];
        let mut best_length = MIN_MATCH_LENGTH - 1;
        let mut best_distance = 0;

        let max_distance = pos.min(MAX_DISTANCE);
        let mut chain_count = 0;
        let mut max_chain = self.max_chain_length;

        // Pre-load the first 4 bytes of target for quick rejection
        // Only if we have enough bytes
        let target_prefix = if pos + 4 <= data.len() {
            Some(u32::from_le_bytes([
                data[pos],
                data[pos + 1],
                data[pos + 2],
                data[pos + 3],
            ]))
        } else {
            None
        };

        while chain_pos >= 0 && chain_count < max_chain {
            let match_pos = chain_pos as usize;
            let distance = pos - match_pos;

            if distance > max_distance {
                break;
            }

            // Quick rejection: check if first 4 bytes can possibly beat current best
            // If we already have a match of length N, the first N bytes must match
            if let Some(target) = target_prefix {
                if match_pos + 4 <= data.len() {
                    let candidate = u32::from_le_bytes([
                        data[match_pos],
                        data[match_pos + 1],
                        data[match_pos + 2],
                        data[match_pos + 3],
                    ]);

                    // If best_length >= 4, all 4 bytes must match
                    if best_length >= 4 && candidate != target {
                        chain_pos = self.prev[match_pos % MAX_DISTANCE];
                        chain_count += 1;
                        continue;
                    }

                    // If best_length >= 3, first 3 bytes must match (check lower 24 bits)
                    if best_length >= 3 && (candidate ^ target) & 0x00FFFFFF != 0 {
                        chain_pos = self.prev[match_pos % MAX_DISTANCE];
                        chain_count += 1;
                        continue;
                    }
                }
            }

            // Also check that the byte at best_length position matches
            // (if we're going to beat the current best, this byte must match)
            if best_length >= MIN_MATCH_LENGTH
                && pos + best_length < data.len()
                && match_pos + best_length < data.len()
                && data[match_pos + best_length] != data[pos + best_length]
            {
                chain_pos = self.prev[match_pos % MAX_DISTANCE];
                chain_count += 1;
                continue;
            }

            // Full match length comparison
            let length = self.match_length(data, match_pos, pos);

            if length > best_length {
                best_length = length;
                best_distance = distance;

                // Early exit if we found a "nice" match - stop searching
                if length >= self.nice_match {
                    break;
                }

                // Early exit if we found max length
                if length >= MAX_MATCH_LENGTH {
                    break;
                }

                // After finding a good match, reduce remaining chain depth
                if length >= GOOD_MATCH_LENGTH {
                    max_chain = (chain_count + 4).min(max_chain);
                }
            }

            // Follow chain
            chain_pos = self.prev[match_pos % MAX_DISTANCE];
            chain_count += 1;
        }

        if best_length >= MIN_MATCH_LENGTH {
            Some((best_length, best_distance))
        } else {
            None
        }
    }

    /// Calculate match length between two positions.
    /// Uses SIMD (when available) or multi-byte comparison for better performance.
    #[inline]
    fn match_length(&self, data: &[u8], pos1: usize, pos2: usize) -> usize {
        let max_len = (data.len() - pos2).min(MAX_MATCH_LENGTH);

        #[cfg(feature = "simd")]
        {
            crate::simd::match_length(data, pos1, pos2, max_len)
        }

        #[cfg(not(feature = "simd"))]
        {
            Self::match_length_scalar(data, pos1, pos2, max_len)
        }
    }

    /// Scalar implementation of match length comparison.
    #[cfg(not(feature = "simd"))]
    #[inline]
    fn match_length_scalar(data: &[u8], pos1: usize, pos2: usize, max_len: usize) -> usize {
        let mut length = 0;

        // Compare 8 bytes at a time using u64
        while length + 8 <= max_len {
            let a = u64::from_ne_bytes(
                data[pos1 + length..pos1 + length + 8]
                    .try_into()
                    .unwrap(),
            );
            let b = u64::from_ne_bytes(
                data[pos2 + length..pos2 + length + 8]
                    .try_into()
                    .unwrap(),
            );
            if a != b {
                // Find the first differing byte using trailing zeros
                let xor = a ^ b;
                #[cfg(target_endian = "little")]
                {
                    length += (xor.trailing_zeros() / 8) as usize;
                }
                #[cfg(target_endian = "big")]
                {
                    length += (xor.leading_zeros() / 8) as usize;
                }
                return length;
            }
            length += 8;
        }

        // Handle remaining bytes one at a time
        while length < max_len && data[pos1 + length] == data[pos2 + length] {
            length += 1;
        }

        length
    }

    /// Update hash table for a position.
    #[inline]
    fn update_hash(&mut self, data: &[u8], pos: usize) {
        if pos + 4 > data.len() {
            return;
        }

        let hash = hash4(data, pos);
        self.prev[pos % MAX_DISTANCE] = self.head[hash];
        self.head[hash] = pos as i32;
    }

    /// Batch update hash table for multiple positions.
    /// More efficient than calling update_hash in a loop.
    #[inline]
    fn update_hash_batch(&mut self, data: &[u8], start: usize, count: usize) {
        // For short matches, just use the simple loop
        if count <= 4 {
            for i in 0..count {
                self.update_hash(data, start + i);
            }
            return;
        }

        // For longer matches, update hashes more efficiently
        // We only need to update hashes for positions that could start a future match
        // First, update the first few positions normally
        for i in 0..4.min(count) {
            self.update_hash(data, start + i);
        }

        // For remaining positions, we can skip some hash updates for very long matches
        // but we still need to maintain the chain structure
        for i in 4..count {
            let pos = start + i;
            if pos + 4 > data.len() {
                break;
            }
            let hash = hash4(data, pos);
            self.prev[pos % MAX_DISTANCE] = self.head[hash];
            self.head[hash] = pos as i32;
        }
    }
}

impl Default for Lz77Compressor {
    fn default() -> Self {
        Self::new(6)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lz77_no_matches() {
        let mut compressor = Lz77Compressor::new(6);
        let data = b"abcdefgh";
        let tokens = compressor.compress(data);

        // All literals
        assert_eq!(tokens.len(), 8);
        for (i, &token) in tokens.iter().enumerate() {
            assert_eq!(token, Token::Literal(data[i]));
        }
    }

    #[test]
    fn test_lz77_simple_repeat() {
        let mut compressor = Lz77Compressor::new(6);
        let data = b"abcabcabc";
        let tokens = compressor.compress(data);

        // Should have "abc" as literals, then matches
        assert!(tokens.len() < 9); // Less than all literals
    }

    #[test]
    fn test_lz77_long_repeat() {
        let mut compressor = Lz77Compressor::new(6);
        let data = b"abcdefghijabcdefghijabcdefghij";
        let tokens = compressor.compress(data);

        // Should compress well
        assert!(tokens.len() < 20);
    }

    #[test]
    fn test_lz77_empty() {
        let mut compressor = Lz77Compressor::new(6);
        let tokens = compressor.compress(&[]);
        assert!(tokens.is_empty());
    }
}
