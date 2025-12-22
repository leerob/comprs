//! LZ77 compression algorithm with sliding window.
//!
//! LZ77 finds repeated sequences in the input and replaces them with
//! (length, distance) pairs referring back to previous occurrences.

/// Maximum distance to look back for matches (32KB window).
pub const MAX_DISTANCE: usize = 32768;

/// Threshold for "good enough" match - skip lazy matching above this length.
/// This is a common optimization used by zlib to speed up compression.
const GOOD_MATCH_LENGTH: usize = 16;

/// Maximum match length (as per DEFLATE spec).
pub const MAX_MATCH_LENGTH: usize = 258;

/// Minimum match length worth encoding.
pub const MIN_MATCH_LENGTH: usize = 3;

/// Size of the hash table (power of 2 for fast modulo).
/// Enlarged to reduce collisions when using 4-byte hashes.
const HASH_SIZE: usize = 1 << 16; // 65536 entries

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

/// Packed token representation (4 bytes) for cache-friendly encoding paths.
///
/// Bit layout:
/// - Bit 31: LITERAL_FLAG (1 = literal, 0 = match)
/// - For literals: bits 0-7 contain the byte value
/// - For matches: bits 0-15 contain length, bits 16-30 contain (distance - 1)
///
/// Distance is stored as (distance - 1) so that the range 1-32768 maps to 0-32767,
/// which fits in 15 bits and avoids collision with the LITERAL_FLAG.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PackedToken(u32);

impl PackedToken {
    const LITERAL_FLAG: u32 = 0x8000_0000;

    #[inline]
    /// Create a packed literal token.
    pub fn literal(byte: u8) -> Self {
        Self(Self::LITERAL_FLAG | byte as u32)
    }

    #[inline]
    /// Create a packed match token.
    pub fn match_(length: u16, distance: u16) -> Self {
        debug_assert!(distance >= 1, "Distance must be at least 1");
        // Store (distance - 1) so range 1-32768 becomes 0-32767, fitting in 15 bits.
        // This ensures bit 31 is never set for matches, avoiding collision with LITERAL_FLAG.
        let dist_minus_one = (distance - 1) as u32;
        let val = dist_minus_one << 16 | (length as u32);
        Self(val)
    }

    #[inline]
    /// Returns true if this is a literal.
    pub fn is_literal(self) -> bool {
        (self.0 & Self::LITERAL_FLAG) != 0
    }

    #[inline]
    /// Get the literal byte if present.
    pub fn as_literal(self) -> Option<u8> {
        if self.is_literal() {
            Some(self.0 as u8)
        } else {
            None
        }
    }

    #[inline]
    /// Get (length, distance) if this is a match.
    pub fn as_match(self) -> Option<(u16, u16)> {
        if self.is_literal() {
            None
        } else {
            let length = (self.0 & 0xFFFF) as u16;
            // Recover original distance by adding 1 back
            let distance = ((self.0 >> 16) as u16) + 1;
            Some((length, distance))
        }
    }
}

/// Hash function for 4-byte sequences with better distribution.
#[inline]
fn hash4(data: &[u8], pos: usize) -> usize {
    if pos + 3 >= data.len() {
        return 0;
    }
    let val = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
    // Multiplicative hash; 0x1E35_A7BD is used in several LZ implementations.
    ((val.wrapping_mul(0x1E35_A7BD)) >> 16) as usize & (HASH_SIZE - 1)
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
}

impl Lz77Compressor {
    /// Create a new LZ77 compressor.
    ///
    /// # Arguments
    /// * `level` - Compression level 1-9 (higher = better compression, slower)
    pub fn new(level: u8) -> Self {
        let level = level.clamp(1, 9);

        // Tune chain length and lazy matching based on level
        let (max_chain_length, lazy_matching) = match level {
            1 => (4, false),
            2 => (6, false),
            3 => (10, false),
            4 => (24, true),
            5 => (48, true),
            6 => (64, true), // trimmed further for speed at default level
            7 => (256, true),
            8 => (512, true),
            9 => (1024, true),
            _ => (128, true),
        };

        Self {
            head: vec![-1; HASH_SIZE],
            prev: vec![-1; MAX_DISTANCE],
            max_chain_length,
            lazy_matching,
        }
    }

    /// Compress data and return LZ77 tokens.
    pub fn compress(&mut self, data: &[u8]) -> Vec<Token> {
        let mut tokens = Vec::with_capacity(data.len());
        self.compress_into(data, &mut tokens);
        tokens
    }

    /// Compress data into a provided token buffer, reusing allocations.
    pub fn compress_into(&mut self, data: &[u8], tokens: &mut Vec<Token>) {
        if data.is_empty() {
            tokens.clear();
            return;
        }

        tokens.clear();
        tokens.reserve(data.len());
        let mut pos = 0;
        let mut literal_streak = 0usize;
        let mut incompressible_mode = false;
        let mut probe_since_last = 0usize;
        let mut incompressible_updates = 0usize;

        // Reset hash tables
        self.head.fill(-1);
        self.prev.fill(-1);

        while pos < data.len() {
            if incompressible_mode {
                // Periodically probe for a match with a very shallow chain to exit early if data changes.
                if probe_since_last >= INCOMPRESSIBLE_PROBE_INTERVAL {
                    probe_since_last = 0;
                    if let Some((length, distance)) =
                        self.find_best_match(data, pos, INCOMPRESSIBLE_CHAIN_LIMIT)
                    {
                        incompressible_mode = false;
                        literal_streak = 0;

                        tokens.push(Token::Match {
                            length: length as u16,
                            distance: distance as u16,
                        });

                        for i in 0..length {
                            self.update_hash(data, pos + i);
                        }
                        pos += length;
                        continue;
                    }
                }

                // Stay in literal-only fast path
                tokens.push(Token::Literal(data[pos]));
                incompressible_updates += 1;
                if incompressible_updates >= INCOMPRESSIBLE_UPDATE_INTERVAL {
                    self.update_hash(data, pos);
                    incompressible_updates = 0;
                }
                pos += 1;
                literal_streak = literal_streak.saturating_add(1);
                probe_since_last = probe_since_last.saturating_add(1);
                continue;
            }

            let chain_limit = if literal_streak >= INCOMPRESSIBLE_LITERAL_THRESHOLD {
                incompressible_mode = true;
                probe_since_last = 0;
                INCOMPRESSIBLE_CHAIN_LIMIT
            } else {
                self.max_chain_length
            };

            let best_match = self.find_best_match(data, pos, chain_limit);

            if let Some((length, distance)) = best_match {
                literal_streak = 0;
                incompressible_mode = false;
                probe_since_last = 0;

                // Check for lazy match if enabled, but skip for "good enough" matches
                // This is a common optimization used by zlib
                if self.lazy_matching && length < GOOD_MATCH_LENGTH && pos + 1 < data.len() {
                    // Update hash for current position
                    self.update_hash(data, pos);

                    if let Some((next_length, _)) = self.find_best_match(data, pos + 1, chain_limit)
                    {
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

                // Update hash for all positions in the match
                for i in 0..length {
                    self.update_hash(data, pos + i);
                }
                pos += length;
            } else {
                literal_streak = literal_streak.saturating_add(1);
                if literal_streak >= INCOMPRESSIBLE_LITERAL_THRESHOLD {
                    incompressible_mode = true;
                    probe_since_last = 0;
                    incompressible_updates = 0;
                }
                tokens.push(Token::Literal(data[pos]));
                self.update_hash(data, pos);
                pos += 1;
            }
        }
    }

    /// Compress data and return packed tokens (4 bytes each) for cache-friendly encoding.
    /// This does not change the parsing—only the representation of tokens.
    pub fn compress_packed(&mut self, data: &[u8]) -> Vec<PackedToken> {
        let mut tokens = Vec::with_capacity(data.len());
        self.compress_packed_into(data, &mut tokens);
        tokens
    }

    /// Compress data into a provided packed token buffer, reusing allocations.
    /// Semantics match `compress`, but outputs `PackedToken` instead of `Token`.
    pub fn compress_packed_into(&mut self, data: &[u8], tokens: &mut Vec<PackedToken>) {
        if data.is_empty() {
            tokens.clear();
            return;
        }

        tokens.clear();
        tokens.reserve(data.len());
        let mut pos = 0;
        let mut literal_streak = 0usize;
        let mut incompressible_mode = false;
        let mut probe_since_last = 0usize;
        let mut incompressible_updates = 0usize;

        // Reset hash tables
        self.head.fill(-1);
        self.prev.fill(-1);

        while pos < data.len() {
            if incompressible_mode {
                if probe_since_last >= INCOMPRESSIBLE_PROBE_INTERVAL {
                    probe_since_last = 0;
                    if let Some((length, distance)) =
                        self.find_best_match(data, pos, INCOMPRESSIBLE_CHAIN_LIMIT)
                    {
                        incompressible_mode = false;
                        literal_streak = 0;

                        tokens.push(PackedToken::match_(length as u16, distance as u16));
                        for i in 0..length {
                            self.update_hash(data, pos + i);
                        }
                        pos += length;
                        continue;
                    }
                }

                tokens.push(PackedToken::literal(data[pos]));
                incompressible_updates += 1;
                if incompressible_updates >= INCOMPRESSIBLE_UPDATE_INTERVAL {
                    self.update_hash(data, pos);
                    incompressible_updates = 0;
                }
                pos += 1;
                literal_streak = literal_streak.saturating_add(1);
                probe_since_last = probe_since_last.saturating_add(1);
                continue;
            }

            let chain_limit = if literal_streak >= INCOMPRESSIBLE_LITERAL_THRESHOLD {
                incompressible_mode = true;
                probe_since_last = 0;
                INCOMPRESSIBLE_CHAIN_LIMIT
            } else {
                self.max_chain_length
            };

            let best_match = self.find_best_match(data, pos, chain_limit);

            if let Some((length, distance)) = best_match {
                literal_streak = 0;
                incompressible_mode = false;
                probe_since_last = 0;

                if self.lazy_matching && length < GOOD_MATCH_LENGTH && pos + 1 < data.len() {
                    self.update_hash(data, pos);

                    if let Some((next_length, _)) = self.find_best_match(data, pos + 1, chain_limit)
                    {
                        if next_length > length + 1 {
                            tokens.push(PackedToken::literal(data[pos]));
                            pos += 1;
                            continue;
                        }
                    }
                }

                tokens.push(PackedToken::match_(length as u16, distance as u16));

                for i in 0..length {
                    self.update_hash(data, pos + i);
                }
                pos += length;
            } else {
                literal_streak = literal_streak.saturating_add(1);
                if literal_streak >= INCOMPRESSIBLE_LITERAL_THRESHOLD {
                    incompressible_mode = true;
                    probe_since_last = 0;
                    incompressible_updates = 0;
                }
                tokens.push(PackedToken::literal(data[pos]));
                self.update_hash(data, pos);
                pos += 1;
            }
        }
    }

    /// Find the best match at the given position.
    fn find_best_match(
        &self,
        data: &[u8],
        pos: usize,
        chain_limit: usize,
    ) -> Option<(usize, usize)> {
        if pos + MIN_MATCH_LENGTH > data.len() {
            return None;
        }

        let hash = hash4(data, pos);
        let mut chain_pos = self.head[hash];
        let mut best_length = MIN_MATCH_LENGTH - 1;
        let mut best_distance = 0;

        let max_distance = pos.min(MAX_DISTANCE);
        let mut chain_remaining = chain_limit;

        // Quick-rejection prefix for candidates when we have at least 4 bytes ahead.
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

        while chain_pos >= 0 && chain_remaining > 0 {
            let match_pos = chain_pos as usize;
            let distance = pos - match_pos;

            if distance > max_distance {
                break;
            }

            if let Some(prefix) = target_prefix {
                if match_pos + 4 <= data.len() {
                    let cand = u32::from_le_bytes([
                        data[match_pos],
                        data[match_pos + 1],
                        data[match_pos + 2],
                        data[match_pos + 3],
                    ]);
                    if cand != prefix {
                        chain_pos = self.prev[match_pos % MAX_DISTANCE];
                        chain_remaining -= 1;
                        continue;
                    }
                }
            }

            // Compare strings
            let length = self.match_length(data, match_pos, pos);

            if length > best_length {
                best_length = length;
                best_distance = distance;

                // Early exit if we found max length.
                if length >= MAX_MATCH_LENGTH {
                    break;
                }
            }

            // Follow chain
            chain_pos = self.prev[match_pos % MAX_DISTANCE];
            chain_remaining -= 1;
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
            let a = u64::from_ne_bytes(data[pos1 + length..pos1 + length + 8].try_into().unwrap());
            let b = u64::from_ne_bytes(data[pos2 + length..pos2 + length + 8].try_into().unwrap());
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
        if pos + 3 >= data.len() {
            return;
        }

        let hash = hash4(data, pos);
        self.prev[pos % MAX_DISTANCE] = self.head[hash];
        self.head[hash] = pos as i32;
    }
}

/// After this many consecutive literals, assume data is mostly incompressible and
/// cap hash-chain traversal to a small constant to reduce wasted work.
const INCOMPRESSIBLE_LITERAL_THRESHOLD: usize = 512;
/// Chain depth to use once incompressible mode triggers.
const INCOMPRESSIBLE_CHAIN_LIMIT: usize = 1;
/// Probe interval (bytes) to attempt exiting incompressible mode with a shallow search.
const INCOMPRESSIBLE_PROBE_INTERVAL: usize = 256;
/// How often to update the hash tables while in incompressible mode (to keep some recency without
/// paying per-byte cost).
const INCOMPRESSIBLE_UPDATE_INTERVAL: usize = 64;

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

    #[test]
    fn test_packed_token_literal() {
        for byte in 0u8..=255 {
            let token = PackedToken::literal(byte);
            assert!(
                token.is_literal(),
                "Literal should be identified as literal"
            );
            assert_eq!(token.as_literal(), Some(byte), "Literal value mismatch");
            assert_eq!(token.as_match(), None, "Literal should not be a match");
        }
    }

    #[test]
    fn test_packed_token_match() {
        // Test various lengths and distances including edge cases
        let test_cases = [
            (3, 1),       // min length, min distance
            (258, 1),     // max length, min distance
            (3, 32768),   // min length, max distance (the bug case!)
            (258, 32768), // max length, max distance
            (100, 100),   // typical values
            (3, 32767),   // just below max distance
        ];

        for (length, distance) in test_cases {
            let token = PackedToken::match_(length, distance);
            assert!(
                !token.is_literal(),
                "Match with length={length}, distance={distance} should NOT be identified as literal",
            );
            assert_eq!(
                token.as_match(),
                Some((length, distance)),
                "Match values mismatch for length={length}, distance={distance}",
            );
            assert_eq!(
                token.as_literal(),
                None,
                "Match should not return a literal value for length={length}, distance={distance}",
            );
        }
    }

    #[test]
    fn test_packed_token_max_distance_not_literal() {
        // Regression test: distance=32768 previously collided with LITERAL_FLAG
        let token = PackedToken::match_(10, 32768);
        assert!(
            !token.is_literal(),
            "Match with max distance (32768) was incorrectly identified as literal"
        );
        assert_eq!(
            token.as_match(),
            Some((10, 32768)),
            "Match with max distance failed to roundtrip correctly"
        );
    }
}
