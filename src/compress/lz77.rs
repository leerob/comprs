//! LZ77 compression algorithm with sliding window.
//!
//! LZ77 finds repeated sequences in the input and replaces them with
//! (length, distance) pairs referring back to previous occurrences.
//!
//! This module supports two parsing strategies:
//! - **Greedy parsing**: Fast, finds longest match at each position
//! - **Optimal parsing**: Slower, uses dynamic programming to find globally optimal tokenization

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

/// Sink abstraction to emit tokens (literal or match) for both Token and PackedToken outputs.
trait TokenSink {
    fn clear(&mut self);
    fn reserve(&mut self, additional: usize);
    fn push_literal(&mut self, byte: u8);
    fn push_match(&mut self, length: u16, distance: u16);
}

impl TokenSink for Vec<Token> {
    fn clear(&mut self) {
        Vec::clear(self)
    }

    fn reserve(&mut self, additional: usize) {
        Vec::reserve(self, additional)
    }

    fn push_literal(&mut self, byte: u8) {
        self.push(Token::Literal(byte));
    }

    fn push_match(&mut self, length: u16, distance: u16) {
        self.push(Token::Match { length, distance });
    }
}

impl TokenSink for Vec<PackedToken> {
    fn clear(&mut self) {
        Vec::clear(self)
    }

    fn reserve(&mut self, additional: usize) {
        Vec::reserve(self, additional)
    }

    fn push_literal(&mut self, byte: u8) {
        self.push(PackedToken::literal(byte));
    }

    fn push_match(&mut self, length: u16, distance: u16) {
        self.push(PackedToken::match_(length, distance));
    }
}

/// Hash function for 4-byte sequences with better distribution.
#[inline(always)]
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

        // Tune chain length and lazy matching based on level.
        // Note: Lazy matching is disabled because empirical testing shows it hurts
        // compression ratio for PNG-style data with many short matches. The longer
        // chain lengths at higher levels provide better compression without it.
        let (max_chain_length, lazy_matching) = match level {
            1 => (4, false),
            2 => (8, false),
            3 => (16, false),
            4 => (32, false),
            5 => (64, false),
            6 => (128, false),
            7 => (256, false),
            8 => (1024, false),
            9 => (4096, false), // Exhaustive search for maximum compression
            _ => (128, false),
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
        self.compress_into_sink(data, &mut tokens);
        tokens
    }

    /// Compress data into a provided token buffer, reusing allocations.
    pub fn compress_into(&mut self, data: &[u8], tokens: &mut Vec<Token>) {
        self.compress_into_sink(data, tokens);
    }

    fn compress_into_sink<T: TokenSink>(&mut self, data: &[u8], sink: &mut T) {
        if data.is_empty() {
            sink.clear();
            return;
        }

        sink.clear();
        sink.reserve(data.len());
        let mut pos = 0;
        let mut literal_streak = 0usize;
        let mut incompressible_mode = false;
        let mut probe_since_last = 0usize;
        let mut incompressible_updates = 0usize;
        // Track pending match from lazy evaluation to prevent cascading deferrals
        let mut pending_match: Option<(usize, usize)> = None;

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

                        sink.push_match(length as u16, distance as u16);

                        for i in 0..length {
                            self.update_hash(data, pos + i);
                        }
                        pos += length;
                        continue;
                    }
                }

                // Stay in literal-only fast path
                sink.push_literal(data[pos]);
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

            // If we have a pending match from lazy evaluation, use it directly
            // (prevents cascading deferrals)
            let best_match = if let Some(pending) = pending_match.take() {
                Some(pending)
            } else {
                self.find_best_match(data, pos, chain_limit)
            };

            if let Some((length, distance)) = best_match {
                literal_streak = 0;
                incompressible_mode = false;
                probe_since_last = 0;

                // Check for lazy match if enabled, but skip for "good enough" matches.
                // Only defer if the next match is significantly better (>= 3 bytes longer)
                // to justify the cost of emitting a literal.
                if self.lazy_matching && length < GOOD_MATCH_LENGTH && pos + 1 < data.len() {
                    // Update hash for current position before looking ahead
                    self.update_hash(data, pos);

                    if let Some((next_length, next_distance)) =
                        self.find_best_match(data, pos + 1, chain_limit)
                    {
                        // Require significant improvement to justify deferral.
                        // A literal costs ~8-9 bits, so the next match should save more than that.
                        // Length difference of 3+ bytes typically saves 24+ bits of match data.
                        if next_length >= length + 3 {
                            // Better match at next position, emit literal and store pending match
                            sink.push_literal(data[pos]);
                            pending_match = Some((next_length, next_distance));
                            pos += 1;
                            continue;
                        }
                    }
                }

                sink.push_match(length as u16, distance as u16);

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
                sink.push_literal(data[pos]);
                self.update_hash(data, pos);
                pos += 1;
            }
        }
    }

    /// Compress data and return packed tokens (4 bytes each) for cache-friendly encoding.
    /// This does not change the parsing—only the representation of tokens.
    pub fn compress_packed(&mut self, data: &[u8]) -> Vec<PackedToken> {
        let mut tokens = Vec::with_capacity(data.len());
        self.compress_into_sink(data, &mut tokens);
        tokens
    }

    /// Compress data into a provided packed token buffer, reusing allocations.
    /// Semantics match `compress`, but outputs `PackedToken` instead of `Token`.
    pub fn compress_packed_into(&mut self, data: &[u8], tokens: &mut Vec<PackedToken>) {
        self.compress_into_sink(data, tokens);
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
    #[inline(always)]
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
    #[inline(always)]
    fn update_hash(&mut self, data: &[u8], pos: usize) {
        if pos + 3 >= data.len() {
            return;
        }

        let hash = hash4(data, pos);
        self.prev[pos % MAX_DISTANCE] = self.head[hash];
        self.head[hash] = pos as i32;
    }

    /// Find all matches at the given position and return the shortest distance for each length.
    ///
    /// Returns (sublen, max_length) where:
    /// - `sublen[len]` = shortest distance that achieves a match of length `len` (0 if none)
    /// - `max_length` = longest match found
    ///
    /// This is the key optimization from Zopfli: tracking sublen allows optimal parsing to
    /// consider all possible match lengths, not just the longest.
    fn find_match_with_sublen(&self, data: &[u8], pos: usize) -> ([u16; 259], usize) {
        let mut sublen = [0u16; 259]; // sublen[length] = best distance for that length
        let mut max_length = 0usize;

        if pos + MIN_MATCH_LENGTH > data.len() {
            return (sublen, 0);
        }

        let hash = hash4(data, pos);
        let mut chain_pos = self.head[hash];
        let max_distance = pos.min(MAX_DISTANCE);
        let mut chain_remaining = self.max_chain_length;

        while chain_pos >= 0 && chain_remaining > 0 {
            let match_pos = chain_pos as usize;
            let distance = pos - match_pos;

            if distance > max_distance {
                break;
            }

            let length = self.match_length(data, match_pos, pos);

            if length >= MIN_MATCH_LENGTH {
                // For each length from MIN_MATCH_LENGTH to length, if we haven't
                // seen a match of that length yet OR this distance is shorter,
                // record it. Shorter distances are better for compression.
                for len in MIN_MATCH_LENGTH..=length {
                    if sublen[len] == 0 || (distance as u16) < sublen[len] {
                        sublen[len] = distance as u16;
                    }
                }
                if length > max_length {
                    max_length = length;
                    if max_length >= MAX_MATCH_LENGTH {
                        break;
                    }
                }
            }

            chain_pos = self.prev[match_pos % MAX_DISTANCE];
            chain_remaining -= 1;
        }

        (sublen, max_length)
    }

    /// Optimal LZ77 parsing using forward dynamic programming.
    ///
    /// This implements the core Zopfli technique: instead of greedily choosing
    /// the longest match at each position, we use DP to find the globally optimal
    /// sequence of tokens that minimizes total bit cost.
    ///
    /// # Algorithm
    /// 1. Forward pass: For each position, try literal and all match lengths
    /// 2. Track minimum cost to reach each position and the length that achieved it
    /// 3. Backward pass: Trace back from end to reconstruct optimal token sequence
    pub fn compress_optimal(&mut self, data: &[u8], cost_model: &CostModel) -> Vec<Token> {
        if data.is_empty() {
            return Vec::new();
        }

        let n = data.len();

        // Reset hash tables
        self.head.fill(-1);
        self.prev.fill(-1);

        // costs[i] = minimum cost to encode bytes 0..i
        // length_array[i] = length of token that ends at position i in the optimal path
        // dist_array[i] = distance for that token (0 for literals)
        let mut costs = vec![f32::MAX; n + 1];
        let mut length_array = vec![0u16; n + 1];
        let mut dist_array = vec![0u16; n + 1];

        costs[0] = 0.0;

        // Forward pass: compute minimum cost to reach each position
        for i in 0..n {
            if costs[i] >= f32::MAX {
                continue;
            }

            // Update hash for this position
            self.update_hash(data, i);

            // Try emitting a literal
            let lit_cost = costs[i] + cost_model.literal_cost(data[i]);
            if lit_cost < costs[i + 1] {
                costs[i + 1] = lit_cost;
                length_array[i + 1] = 1;
                dist_array[i + 1] = 0;
            }

            // Find all matches at this position
            let (sublen, max_len) = self.find_match_with_sublen(data, i);

            // Try each possible match length
            for len in MIN_MATCH_LENGTH..=max_len {
                let dist = sublen[len];
                if dist == 0 {
                    continue;
                }

                let match_cost = costs[i] + cost_model.match_cost(len as u16, dist);
                let end_pos = i + len;
                if end_pos <= n && match_cost < costs[end_pos] {
                    costs[end_pos] = match_cost;
                    length_array[end_pos] = len as u16;
                    dist_array[end_pos] = dist;
                }
            }
        }

        // Backward pass: reconstruct optimal token sequence
        self.trace_backwards(&length_array, &dist_array, data)
    }

    /// Trace backwards through the DP solution to reconstruct optimal tokens.
    fn trace_backwards(&self, length_array: &[u16], dist_array: &[u16], data: &[u8]) -> Vec<Token> {
        let n = data.len();
        let mut tokens = Vec::new();

        // Count how many tokens we'll have by walking backwards
        let mut pos = n;
        while pos > 0 {
            let len = length_array[pos] as usize;
            if len == 0 {
                // This shouldn't happen if the DP completed correctly
                break;
            }
            pos -= len;
        }

        // We need to walk forward using the lengths stored at each endpoint
        // length_array[i] tells us the length of the token that ENDS at position i
        // So we need to find the sequence of endpoints

        // First, collect all the lengths by walking backward
        let mut lengths_rev = Vec::new();
        let mut p = n;
        while p > 0 {
            let len = length_array[p] as usize;
            let dist = dist_array[p];
            if len == 0 {
                break;
            }
            lengths_rev.push((len, dist));
            p -= len;
        }

        // Now emit tokens in forward order
        let mut data_pos = 0;
        for (len, dist) in lengths_rev.into_iter().rev() {
            if len == 1 && dist == 0 {
                // Literal
                tokens.push(Token::Literal(data[data_pos]));
            } else {
                // Match
                tokens.push(Token::Match {
                    length: len as u16,
                    distance: dist,
                });
            }
            data_pos += len;
        }

        tokens
    }
}

// ============================================================================
// Cost Model for Optimal Parsing
// ============================================================================

/// Cost model for computing bit costs of literals and matches.
///
/// The cost model can use fixed estimates or actual Huffman bit lengths
/// derived from symbol statistics. Using actual statistics enables
/// iterative refinement (the key to Zopfli's effectiveness).
#[derive(Clone)]
pub struct CostModel {
    /// Bit cost for each literal byte (0-255) and length symbol (256-285)
    pub lit_len_costs: [f32; 286],
    /// Bit cost for each distance symbol (0-29)
    pub dist_costs: [f32; 30],
}

impl CostModel {
    /// Create a cost model with fixed estimates.
    ///
    /// Uses approximate costs based on fixed Huffman code lengths.
    /// Good for initial pass before we have actual statistics.
    pub fn fixed() -> Self {
        let mut lit_len_costs = [0.0f32; 286];
        let mut dist_costs = [0.0f32; 30];

        // Fixed Huffman: literals 0-143 = 8 bits, 144-255 = 9 bits
        for i in 0..144 {
            lit_len_costs[i] = 8.0;
        }
        for i in 144..256 {
            lit_len_costs[i] = 9.0;
        }
        // End of block (256) and length codes (257-279) = 7 bits
        for i in 256..280 {
            lit_len_costs[i] = 7.0;
        }
        // Length codes 280-285 = 8 bits
        for i in 280..286 {
            lit_len_costs[i] = 8.0;
        }

        // Fixed Huffman: all distance codes = 5 bits
        for i in 0..30 {
            dist_costs[i] = 5.0;
        }

        Self {
            lit_len_costs,
            dist_costs,
        }
    }

    /// Create a cost model from symbol frequency counts.
    ///
    /// Computes bit costs using entropy: cost = -log2(frequency/total)
    /// This is the foundation of iterative refinement.
    pub fn from_statistics(lit_len_counts: &[u32; 286], dist_counts: &[u32; 30]) -> Self {
        let mut lit_len_costs = [0.0f32; 286];
        let mut dist_costs = [0.0f32; 30];

        // Compute literal/length costs from entropy
        let lit_total: u32 = lit_len_counts.iter().sum();
        if lit_total > 0 {
            let log_total = (lit_total as f32).log2();
            for (i, &count) in lit_len_counts.iter().enumerate() {
                if count > 0 {
                    // cost = -log2(p) = log2(total) - log2(count)
                    lit_len_costs[i] = log_total - (count as f32).log2();
                } else {
                    // Unseen symbols get high cost (but not infinite)
                    lit_len_costs[i] = 15.0;
                }
            }
        } else {
            // Fallback to fixed costs
            return Self::fixed();
        }

        // Compute distance costs from entropy
        let dist_total: u32 = dist_counts.iter().sum();
        if dist_total > 0 {
            let log_total = (dist_total as f32).log2();
            for (i, &count) in dist_counts.iter().enumerate() {
                if count > 0 {
                    dist_costs[i] = log_total - (count as f32).log2();
                } else {
                    dist_costs[i] = 15.0;
                }
            }
        } else {
            // No distance symbols seen - use fixed
            for i in 0..30 {
                dist_costs[i] = 5.0;
            }
        }

        Self {
            lit_len_costs,
            dist_costs,
        }
    }

    /// Compute the bit cost of emitting a literal.
    #[inline]
    pub fn literal_cost(&self, byte: u8) -> f32 {
        self.lit_len_costs[byte as usize]
    }

    /// Compute the bit cost of emitting a match (length + distance).
    #[inline]
    pub fn match_cost(&self, length: u16, distance: u16) -> f32 {
        // Get length symbol (257-285) and extra bits
        let (len_symbol, len_extra_bits) = length_to_symbol(length);
        let len_cost = self.lit_len_costs[len_symbol as usize] + len_extra_bits as f32;

        // Get distance symbol (0-29) and extra bits
        let (dist_symbol, dist_extra_bits) = distance_to_symbol(distance);
        let dist_cost = self.dist_costs[dist_symbol as usize] + dist_extra_bits as f32;

        len_cost + dist_cost
    }
}

impl Default for CostModel {
    fn default() -> Self {
        Self::fixed()
    }
}

// ============================================================================
// Symbol Encoding Tables (shared with deflate.rs)
// ============================================================================

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
/// Used for reference; actual lookup uses bit manipulation for efficiency.
#[allow(dead_code)]
const DISTANCE_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];

/// Extra bits for distance codes.
const DISTANCE_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

/// Convert a match length to its DEFLATE symbol and extra bits count.
#[inline]
fn length_to_symbol(length: u16) -> (u16, u8) {
    debug_assert!(
        (MIN_MATCH_LENGTH as u16..=MAX_MATCH_LENGTH as u16).contains(&length),
        "Invalid length: {length}"
    );

    // Binary search would be O(log n), but linear scan is fine for 29 entries
    // and often faster due to early termination
    for i in 0..28 {
        if length < LENGTH_BASE[i + 1] {
            return (257 + i as u16, LENGTH_EXTRA[i]);
        }
    }
    (285, 0) // length 258
}

/// Convert a match distance to its DEFLATE symbol and extra bits count.
#[inline]
fn distance_to_symbol(distance: u16) -> (u16, u8) {
    debug_assert!((1..=32768).contains(&distance), "Invalid distance");

    if distance < 5 {
        return (distance - 1, 0);
    }

    // For larger distances, use bit manipulation
    let d = distance as u32 - 1;
    let msb = 31 - d.leading_zeros();
    let second_bit = (d >> (msb - 1)) & 1;
    let code = (2 * msb + second_bit) as usize;
    let code = code.min(29);

    (code as u16, DISTANCE_EXTRA[code])
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

    #[test]
    fn test_optimal_lz77_empty() {
        let mut compressor = Lz77Compressor::new(6);
        let cost_model = CostModel::fixed();
        let tokens = compressor.compress_optimal(&[], &cost_model);
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_optimal_lz77_no_matches() {
        let mut compressor = Lz77Compressor::new(6);
        let cost_model = CostModel::fixed();
        let data = b"abcdefgh";
        let tokens = compressor.compress_optimal(data, &cost_model);

        // All literals
        assert_eq!(tokens.len(), 8);
        for (i, &token) in tokens.iter().enumerate() {
            assert_eq!(token, Token::Literal(data[i]));
        }
    }

    #[test]
    fn test_optimal_lz77_simple_repeat() {
        let mut compressor = Lz77Compressor::new(6);
        let cost_model = CostModel::fixed();
        let data = b"abcabcabc";
        let tokens = compressor.compress_optimal(data, &cost_model);

        // Should have "abc" as literals, then matches (same as greedy for this case)
        assert!(
            tokens.len() < 9,
            "Expected fewer than 9 tokens, got {}",
            tokens.len()
        );
    }

    #[test]
    fn test_optimal_lz77_produces_valid_tokens() {
        let mut compressor = Lz77Compressor::new(6);
        let cost_model = CostModel::fixed();
        let data = b"The quick brown fox jumps over the lazy dog. The quick brown fox jumps.";
        let tokens = compressor.compress_optimal(data, &cost_model);

        // Verify we can reconstruct the original data from tokens
        let mut reconstructed = Vec::new();
        for token in &tokens {
            match token {
                Token::Literal(b) => reconstructed.push(*b),
                Token::Match { length, distance } => {
                    let start = reconstructed.len() - *distance as usize;
                    for i in 0..*length as usize {
                        reconstructed.push(reconstructed[start + i]);
                    }
                }
            }
        }
        assert_eq!(reconstructed.as_slice(), data.as_slice());
    }

    #[test]
    fn test_cost_model_fixed() {
        let model = CostModel::fixed();

        // Literal 'a' (97) should cost 8 bits (it's in 0-143 range)
        assert!((model.literal_cost(b'a') - 8.0).abs() < 0.01);

        // Literal 200 should cost 9 bits (it's in 144-255 range)
        assert!((model.literal_cost(200) - 9.0).abs() < 0.01);

        // Match cost should include length symbol + extra bits + distance symbol + extra bits
        let match_cost = model.match_cost(3, 1);
        // Length 3 = symbol 257 (7 bits, 0 extra), distance 1 = symbol 0 (5 bits, 0 extra)
        assert!((match_cost - 12.0).abs() < 0.01);
    }

    #[test]
    fn test_cost_model_from_statistics() {
        let mut lit_counts = [0u32; 286];
        let mut dist_counts = [0u32; 30];

        // Simulate a distribution where 'a' is very common
        lit_counts[b'a' as usize] = 100;
        lit_counts[b'b' as usize] = 10;
        lit_counts[256] = 1; // end of block

        dist_counts[0] = 50;
        dist_counts[1] = 10;

        let model = CostModel::from_statistics(&lit_counts, &dist_counts);

        // 'a' should have lower cost than 'b' (more frequent)
        assert!(model.literal_cost(b'a') < model.literal_cost(b'b'));
    }

    #[test]
    fn test_length_to_symbol() {
        // Test boundary cases
        assert_eq!(length_to_symbol(3), (257, 0)); // min length
        assert_eq!(length_to_symbol(10), (264, 0)); // last 0-extra-bit code
        assert_eq!(length_to_symbol(11), (265, 1)); // first 1-extra-bit code
        assert_eq!(length_to_symbol(258), (285, 0)); // max length
    }

    #[test]
    fn test_distance_to_symbol() {
        // Test small distances (direct lookup)
        assert_eq!(distance_to_symbol(1), (0, 0));
        assert_eq!(distance_to_symbol(4), (3, 0));

        // Test larger distances (bit manipulation)
        assert_eq!(distance_to_symbol(5), (4, 1));
        assert_eq!(distance_to_symbol(6), (4, 1));
    }
}
