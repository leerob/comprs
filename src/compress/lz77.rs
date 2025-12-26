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
/// This is a common optimization used by zlib/libdeflate to speed up compression.
const GOOD_MATCH_LENGTH: usize = 16;

/// Maximum match length (as per DEFLATE spec).
pub const MAX_MATCH_LENGTH: usize = 258;

/// Minimum match length worth encoding.
pub const MIN_MATCH_LENGTH: usize = 3;

/// Size of the 4-byte hash table (power of 2 for fast modulo).
/// Enlarged to reduce collisions when using 4-byte hashes.
const HASH_SIZE: usize = 1 << 16; // 65536 entries

/// Size of the 3-byte hash table (singleton buckets).
const HASH3_SIZE: usize = 1 << 15;

/// Parameters for HT-style fast matchfinder (level 1).
const HT_HASH_BITS: usize = 15;
const HT_HASH_SIZE: usize = 1 << HT_HASH_BITS;
const HT_BUCKET_SIZE: usize = 2;

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

/// Parsing strategy for lazy matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LazyKind {
    None,
    Lazy,
    Lazy2,
}

/// Level-specific tuning parameters.
#[derive(Debug, Clone, Copy)]
struct LevelConfig {
    max_chain_length: usize,
    max_search_depth: usize,
    nice_length: usize,
    lazy: LazyKind,
    use_ht: bool,
}

impl PackedToken {
    const LITERAL_FLAG: u32 = 0x8000_0000;

    #[inline]
    pub fn literal(byte: u8) -> Self {
        Self(Self::LITERAL_FLAG | byte as u32)
    }

    #[inline]
    pub fn match_(length: u16, distance: u16) -> Self {
        debug_assert!(distance >= 1, "Distance must be at least 1");
        // Store (distance - 1) so range 1-32768 becomes 0-32767, fitting in 15 bits.
        // This ensures bit 31 is never set for matches, avoiding collision with LITERAL_FLAG.
        let dist_minus_one = (distance - 1) as u32;
        let val = dist_minus_one << 16 | (length as u32);
        Self(val)
    }

    #[inline]
    pub fn is_literal(self) -> bool {
        (self.0 & Self::LITERAL_FLAG) != 0
    }

    #[inline]
    pub fn as_literal(self) -> Option<u8> {
        if self.is_literal() {
            Some(self.0 as u8)
        } else {
            None
        }
    }

    #[inline]
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

/// Hash function for 3-byte sequences (singleton buckets).
#[inline(always)]
fn hash3(data: &[u8], pos: usize) -> usize {
    if pos + 2 >= data.len() {
        return 0;
    }
    let val = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], 0]);
    ((val.wrapping_mul(0x1E35_A7BD)) >> 17) as usize & (HASH3_SIZE - 1)
}

/// Hash for HT buckets (15-bit) based on 4-byte sequence.
#[inline(always)]
fn hash4_ht(data: &[u8], pos: usize) -> usize {
    if pos + 3 >= data.len() {
        return 0;
    }
    let val = u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
    (val.wrapping_mul(0x1E35_A7BD) >> (32 - HT_HASH_BITS)) as usize & (HT_HASH_SIZE - 1)
}

/// Heuristic: choose a minimum match length based on literal diversity and search depth.
fn calculate_min_match_len(data: &[u8], max_search_depth: usize) -> usize {
    let mut used = [false; 256];
    let mut num_used = 0usize;
    let scan = data.len().min(4096);
    for &b in &data[..scan] {
        if !used[b as usize] {
            used[b as usize] = true;
            num_used += 1;
        }
    }
    choose_min_match_len(num_used, max_search_depth)
}

fn choose_min_match_len(num_used_literals: usize, max_search_depth: usize) -> usize {
    if max_search_depth <= 4 {
        return MIN_MATCH_LENGTH;
    }

    let mut min_len = MIN_MATCH_LENGTH;

    if num_used_literals > 32 {
        min_len = 4;
    }
    if num_used_literals > 64 && max_search_depth >= 10 {
        min_len = 5;
    }
    if num_used_literals > 96 && max_search_depth >= 20 {
        min_len = 6;
    }

    min_len.min(MAX_MATCH_LENGTH)
}

/// LZ77 compressor with hash chain for fast matching.
pub struct Lz77Compressor {
    /// Hash table: maps hash -> most recent position
    head: Vec<i32>,
    /// Hash table for 3-byte matches (singleton buckets)
    head3: Vec<i32>,
    /// Buckets for the fast HT-style matchfinder (level 1)
    ht_buckets: Vec<[i32; HT_BUCKET_SIZE]>,
    /// Chain links: prev[pos % window] -> previous position with same hash
    prev: Vec<i32>,
    /// Level tuning knobs
    config: LevelConfig,
}

impl Lz77Compressor {
    /// Level 1-9: higher = better compression, slower.
    pub fn new(level: u8) -> Self {
        let level = level.clamp(1, 9);

        let config = Self::config_for_level(level);

        Self {
            head: vec![-1; HASH_SIZE],
            head3: vec![-1; HASH3_SIZE],
            ht_buckets: vec![[-1; HT_BUCKET_SIZE]; HT_HASH_SIZE],
            prev: vec![-1; MAX_DISTANCE],
            config,
        }
    }

    pub fn compress(&mut self, data: &[u8]) -> Vec<Token> {
        let mut tokens = Vec::with_capacity(data.len());
        self.compress_into_sink(data, &mut tokens);
        tokens
    }

    /// Reuses the provided token buffer allocation.
    pub fn compress_into(&mut self, data: &[u8], tokens: &mut Vec<Token>) {
        self.compress_into_sink(data, tokens);
    }

    fn compress_into_sink<T: TokenSink>(&mut self, data: &[u8], sink: &mut T) {
        if data.is_empty() {
            sink.clear();
            return;
        }

        // Heuristic minimum match length based on literal diversity and search depth.
        let min_match_len = calculate_min_match_len(data, self.config.max_search_depth);

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
        self.head3.fill(-1);
        for bucket in &mut self.ht_buckets {
            *bucket = [-1; HT_BUCKET_SIZE];
        }
        self.prev.fill(-1);

        while pos < data.len() {
            if incompressible_mode {
                // Periodically probe for a match with a very shallow chain to exit early if data changes.
                if probe_since_last >= INCOMPRESSIBLE_PROBE_INTERVAL {
                    probe_since_last = 0;
                    if let Some((length, distance)) = self.find_best_match(
                        data,
                        pos,
                        INCOMPRESSIBLE_CHAIN_LIMIT.min(self.config.max_search_depth),
                        self.config.nice_length,
                        min_match_len,
                    ) {
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
                self.config.max_chain_length
            };

            // If we have a pending match from lazy evaluation, use it directly
            // (prevents cascading deferrals)
            let best_match = if let Some(pending) = pending_match.take() {
                Some(pending)
            } else if self.config.use_ht {
                self.find_best_match_ht(data, pos, self.config.nice_length, min_match_len)
            } else {
                self.find_best_match(
                    data,
                    pos,
                    chain_limit.min(self.config.max_search_depth),
                    self.config.nice_length,
                    min_match_len,
                )
            };

            if let Some((length, distance)) = best_match {
                literal_streak = 0;
                incompressible_mode = false;
                probe_since_last = 0;

                if distance == 0 {
                    // Defensive: treat as literal if match distance is invalid.
                    sink.push_literal(data[pos]);
                    self.update_hash(data, pos);
                    pos += 1;
                    continue;
                }

                // Check for lazy match if enabled, but skip for "good enough" matches.
                // Only defer if the next match is significantly better (>= 3 bytes longer)
                // to justify the cost of emitting a literal.
                if self.config.lazy != LazyKind::None
                    && length < self.config.nice_length
                    && length < GOOD_MATCH_LENGTH
                    && pos + 1 < data.len()
                {
                    // Update hash for current position before looking ahead
                    self.update_hash(data, pos);

                    let next_chain = if self.config.lazy == LazyKind::Lazy2 {
                        (chain_limit / 2).max(1)
                    } else {
                        chain_limit
                    };

                    let next_match = if self.config.use_ht {
                        self.find_best_match_ht(
                            data,
                            pos + 1,
                            self.config.nice_length,
                            min_match_len,
                        )
                    } else {
                        self.find_best_match(
                            data,
                            pos + 1,
                            next_chain.min(self.config.max_search_depth),
                            self.config.nice_length,
                            min_match_len,
                        )
                    };

                    if let Some((next_length, next_distance)) = next_match {
                        // Require significant improvement to justify deferral.
                        // A literal costs ~8-9 bits, so the next match should save more than that.
                        // Length difference of 3+ bytes typically saves 24+ bits of match data.
                        if next_length >= length + 3 || next_length >= self.config.nice_length {
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

    /// Returns packed tokens (4 bytes each) for cache-friendly encoding.
    pub fn compress_packed(&mut self, data: &[u8]) -> Vec<PackedToken> {
        let mut tokens = Vec::with_capacity(data.len());
        self.compress_into_sink(data, &mut tokens);
        tokens
    }

    /// Reuses the provided packed token buffer allocation.
    pub fn compress_packed_into(&mut self, data: &[u8], tokens: &mut Vec<PackedToken>) {
        self.compress_into_sink(data, tokens);
    }

    fn find_best_match(
        &self,
        data: &[u8],
        pos: usize,
        chain_limit: usize,
        nice_length: usize,
        min_match_length: usize,
    ) -> Option<(usize, usize)> {
        if pos + MIN_MATCH_LENGTH > data.len() {
            return None;
        }

        // Check length-3 singleton hash first (cheap path)
        let mut best_length = min_match_length.saturating_sub(1);
        let mut best_distance = 0;

        let hash3 = hash3(data, pos);
        let cand3 = self.head3[hash3];
        if cand3 >= 0 {
            let match_pos = cand3 as usize;
            let distance = pos - match_pos;
            if distance == 0 {
                // Skip self-reference
            } else if distance <= MAX_DISTANCE && match_pos + 3 <= data.len() {
                let a = &data[pos..pos + 3];
                let b = &data[match_pos..match_pos + 3];
                if a == b {
                    let len = self
                        .match_length(data, match_pos, pos)
                        .min(MAX_MATCH_LENGTH);
                    // Gate short matches: reject length 3 with very long distance.
                    if len >= min_match_length && !(len == 3 && distance > 8192) {
                        best_length = len;
                        best_distance = distance;
                        if best_length >= nice_length {
                            return Some((best_length, best_distance));
                        }
                    }
                }
            }
        }

        let hash = hash4(data, pos);
        let mut chain_pos = self.head[hash];

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

            if distance == 0 {
                chain_pos = self.prev[match_pos % MAX_DISTANCE];
                chain_remaining -= 1;
                continue;
            }

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

            if length >= min_match_length
                && !(length == 3 && distance > 8192)
                && length > best_length
            {
                best_length = length;
                best_distance = distance;

                // Early exit if we found max length.
                if length >= MAX_MATCH_LENGTH || best_length >= nice_length {
                    break;
                }
            }

            // Follow chain
            chain_pos = self.prev[match_pos % MAX_DISTANCE];
            chain_remaining -= 1;
        }

        // Only return a match if it meets the minimum length requirement.
        // best_distance is only updated when we find a valid match, so if
        // best_length >= min_match_length, best_distance is guaranteed to be valid (> 0).
        if best_length >= min_match_length {
            Some((best_length, best_distance))
        } else {
            None
        }
    }

    /// Fast HT-style matchfinder for level 1 (2-entry buckets, shallow search).
    fn find_best_match_ht(
        &mut self,
        data: &[u8],
        pos: usize,
        nice_length: usize,
        min_match_length: usize,
    ) -> Option<(usize, usize)> {
        if pos + MIN_MATCH_LENGTH > data.len() {
            return None;
        }

        let hash = hash4_ht(data, pos);
        let bucket = &mut self.ht_buckets[hash];
        let cand0 = bucket[0];
        let cand1 = bucket[1];

        // Insert current position at head of bucket
        bucket[1] = cand0;
        bucket[0] = pos as i32;

        let mut best_len = min_match_length.saturating_sub(1);
        let mut best_dist = 0usize;

        for &cand in [cand0, cand1].iter() {
            if cand < 0 {
                continue;
            }
            let match_pos = cand as usize;
            let distance = pos - match_pos;
            if distance == 0 || distance > MAX_DISTANCE || match_pos + 3 > data.len() {
                continue;
            }
            let a = &data[pos..pos + 3];
            let b = &data[match_pos..match_pos + 3];
            if a != b {
                continue;
            }
            let len = self
                .match_length(data, match_pos, pos)
                .min(MAX_MATCH_LENGTH);
            if len < min_match_length || (len == 3 && distance > 8192) {
                continue;
            }
            if len > best_len {
                best_len = len;
                best_dist = distance;
                if best_len >= nice_length {
                    break;
                }
            }
        }

        // Only return a match if it meets the minimum length requirement.
        // best_dist is only updated when we find a valid match, so if
        // best_len >= min_match_length, best_dist is guaranteed to be valid (> 0).
        if best_len >= min_match_length {
            Some((best_len, best_dist))
        } else {
            None
        }
    }

    /// Uses SIMD when available.
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

    #[inline(always)]
    fn update_hash(&mut self, data: &[u8], pos: usize) {
        if pos + 3 >= data.len() {
            return;
        }

        // Update 3-byte singleton hash
        let h3 = hash3(data, pos);
        self.head3[h3] = pos as i32;

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

        // Cheap length-3 singleton hash probe
        let hash3 = hash3(data, pos);
        let cand3 = self.head3[hash3];
        if cand3 >= 0 {
            let match_pos = cand3 as usize;
            let distance = pos - match_pos;
            if distance == 0 {
                // Skip self-reference
            } else if distance <= MAX_DISTANCE && match_pos + 3 <= data.len() {
                let a = &data[pos..pos + 3];
                let b = &data[match_pos..match_pos + 3];
                if a == b {
                    sublen[3] = distance as u16;
                    max_length = 3;
                }
            }
        }

        let hash = hash4(data, pos);
        let mut chain_pos = self.head[hash];
        let max_distance = pos.min(MAX_DISTANCE);
        let mut chain_remaining = self
            .config
            .max_chain_length
            .min(self.config.max_search_depth);

        while chain_pos >= 0 && chain_remaining > 0 {
            let match_pos = chain_pos as usize;
            let distance = pos - match_pos;

            if distance == 0 {
                chain_pos = self.prev[match_pos % MAX_DISTANCE];
                chain_remaining -= 1;
                continue;
            }

            if distance > max_distance {
                break;
            }

            let length = self.match_length(data, match_pos, pos);

            if length >= MIN_MATCH_LENGTH && !(length == 3 && distance > 8192) {
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
        self.head3.fill(-1);
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

            // Insert this position into hash tables after using prior history.
            self.update_hash(data, i);
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
                // Defensive: distance must be at least 1; otherwise treat as literal.
                if dist == 0 {
                    tokens.push(Token::Literal(data[data_pos]));
                    data_pos += len;
                    continue;
                }
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

    #[inline]
    pub fn literal_cost(&self, byte: u8) -> f32 {
        self.lit_len_costs[byte as usize]
    }

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

impl Lz77Compressor {
    fn config_for_level(level: u8) -> LevelConfig {
        match level {
            1 => LevelConfig {
                max_chain_length: 4,
                max_search_depth: 4,
                nice_length: 32,
                lazy: LazyKind::None,
                use_ht: true,
            },
            2 => LevelConfig {
                max_chain_length: 8,
                max_search_depth: 6,
                nice_length: 10,
                lazy: LazyKind::None,
                use_ht: false,
            },
            3 => LevelConfig {
                max_chain_length: 16,
                max_search_depth: 12,
                nice_length: 14,
                lazy: LazyKind::None,
                use_ht: false,
            },
            4 => LevelConfig {
                max_chain_length: 32,
                max_search_depth: 16,
                nice_length: 30,
                lazy: LazyKind::None,
                use_ht: false,
            },
            5 => LevelConfig {
                max_chain_length: 64,
                max_search_depth: 16,
                nice_length: 30,
                lazy: LazyKind::Lazy,
                use_ht: false,
            },
            6 => LevelConfig {
                max_chain_length: 128,
                max_search_depth: 35,
                nice_length: 65,
                lazy: LazyKind::Lazy,
                use_ht: false,
            },
            7 => LevelConfig {
                max_chain_length: 256,
                max_search_depth: 100,
                nice_length: 130,
                lazy: LazyKind::Lazy,
                use_ht: false,
            },
            8 => LevelConfig {
                max_chain_length: 1024,
                max_search_depth: 300,
                nice_length: MAX_MATCH_LENGTH,
                lazy: LazyKind::Lazy2,
                use_ht: false,
            },
            9 => LevelConfig {
                max_chain_length: 4096,
                max_search_depth: 600,
                nice_length: MAX_MATCH_LENGTH,
                lazy: LazyKind::Lazy2,
                use_ht: false,
            },
            _ => LevelConfig {
                max_chain_length: 4096,
                max_search_depth: 600,
                nice_length: MAX_MATCH_LENGTH,
                lazy: LazyKind::Lazy2,
                use_ht: false,
            },
        }
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

    #[test]
    fn test_find_best_match_no_zero_distance() {
        // Regression test: find_best_match should never return distance=0
        // This bug manifests when min_match_length > MIN_MATCH_LENGTH and no match is found.
        // The best_length would be initialized to min_match_length - 1 (e.g., 3) while
        // best_distance stays at 0, causing an invalid (3, 0) match to be returned.
        let mut compressor = Lz77Compressor::new(6);

        // Create high-entropy data to trigger min_match_length > 3
        let mut unique_data: Vec<u8> = (0..=255).collect();
        unique_data.extend_from_slice(b"xyz"); // Short unrepeated sequence at end

        let result = compressor.compress(&unique_data);

        // Verify no zero-distance matches in result
        for token in &result {
            if let Token::Match { distance, .. } = token {
                assert!(*distance > 0, "Found invalid match with distance 0");
            }
        }
    }

    #[test]
    fn test_find_best_match_returns_none_for_no_match() {
        // Direct test of find_best_match with min_match_length > MIN_MATCH_LENGTH
        let compressor = Lz77Compressor::new(6);

        // Data with no possible matches (all unique bytes, too short for matches)
        let data: Vec<u8> = (0..10).collect();

        // With min_match_length = 4, best_length is initialized to 3
        // If no match is found, it should return None, not Some((3, 0))
        let result = compressor.find_best_match(&data, 5, 128, 65, 4);

        // Should be None since there's no valid match
        assert!(
            result.is_none() || result.unwrap().1 > 0,
            "find_best_match returned invalid match with distance 0: {:?}",
            result
        );
    }

    #[test]
    fn test_find_best_match_respects_min_match_length() {
        // Regression test: find_best_match should never return a match shorter than
        // min_match_length, and should never return a zero distance.
        let mut compressor = Lz77Compressor::new(6);

        // Create data where the only possible match is exactly 4 bytes.
        // "abcdXYZWabcdPQRS" - "abcd" appears at position 0 and 8, but
        // the bytes after differ, limiting match to 4 bytes.
        let data = b"abcdXYZWabcdPQRS";

        // Reset hash tables
        compressor.head.fill(-1);
        compressor.head3.fill(-1);
        compressor.prev.fill(-1);

        // Insert the first occurrence into hash tables (positions 0-7)
        for i in 0..8 {
            compressor.update_hash(data, i);
        }

        // At position 8, we should find a 4-byte match (abcd).
        // With min_match_length = 5, we should get None (not Some((4, 0)))
        let result = compressor.find_best_match(data, 8, 100, 258, 5);
        assert!(
            result.is_none(),
            "find_best_match returned {:?} when no match >= min_match_length exists",
            result
        );

        // With min_match_length = 4, we should get a valid match
        let result = compressor.find_best_match(data, 8, 100, 258, 4);
        assert!(result.is_some(), "Expected a match with min_match_length=4");
        let (len, dist) = result.unwrap();
        assert!(len >= 4, "Match length should be >= min_match_length");
        assert!(dist > 0, "Match distance must be > 0");
        assert_eq!(dist, 8, "Distance should be 8 (back to position 0)");
    }

    #[test]
    fn test_find_best_match_ht_respects_min_match_length() {
        // Same regression test for the HT-style matchfinder
        let mut compressor = Lz77Compressor::new(1); // Level 1 uses HT

        // Same test data: only a 4-byte match is possible
        let data = b"abcdXYZWabcdPQRS";

        // Reset HT buckets
        for bucket in &mut compressor.ht_buckets {
            *bucket = [-1; HT_BUCKET_SIZE];
        }

        // Insert first occurrence at position 0
        let hash = hash4_ht(data, 0);
        compressor.ht_buckets[hash][0] = 0;

        // At position 8, with min_match_length = 5, we should get None
        let result = compressor.find_best_match_ht(data, 8, 258, 5);
        assert!(
            result.is_none(),
            "find_best_match_ht returned {:?} when no match >= min_match_length exists",
            result
        );

        // With min_match_length = 4, we should get a valid match
        let result = compressor.find_best_match_ht(data, 8, 258, 4);
        assert!(result.is_some(), "Expected a match with min_match_length=4");
        let (len, dist) = result.unwrap();
        assert!(len >= 4, "Match length should be >= min_match_length");
        assert!(dist > 0, "Match distance must be > 0");
        assert_eq!(dist, 8, "Distance should be 8 (back to position 0)");
    }

    #[test]
    fn test_find_best_match_never_returns_zero_distance() {
        // Additional regression test: verify that matches always have valid distances.
        // This specifically tests the edge case where min_match_length > MIN_MATCH_LENGTH.
        let mut compressor = Lz77Compressor::new(6);

        // Data with no repeating patterns - should return None for any min_match_length
        let data = b"abcdefghijklmnop";

        compressor.head.fill(-1);
        compressor.head3.fill(-1);
        compressor.prev.fill(-1);

        // Insert early positions
        for i in 0..8 {
            compressor.update_hash(data, i);
        }

        // No matches should be found
        for min_len in 3..=6 {
            let result = compressor.find_best_match(data, 8, 100, 258, min_len);
            if let Some((len, dist)) = result {
                assert!(
                    dist > 0,
                    "find_best_match returned zero distance: len={}, dist={}, min_len={}",
                    len,
                    dist,
                    min_len
                );
                assert!(
                    len >= min_len,
                    "find_best_match returned length {} < min_match_length {}",
                    len,
                    min_len
                );
            }
        }
    }

    #[test]
    fn test_lz77_default() {
        let compressor = Lz77Compressor::default();
        // Default level is 6
        assert_eq!(compressor.config.max_chain_length, 128);
    }

    #[test]
    fn test_lz77_compress_into() {
        let mut compressor = Lz77Compressor::new(6);
        let data = b"abcdefgh";
        let mut tokens = Vec::new();
        compressor.compress_into(data, &mut tokens);
        assert_eq!(tokens.len(), 8);
    }

    #[test]
    fn test_lz77_compress_packed() {
        let mut compressor = Lz77Compressor::new(6);
        let data = b"abcabcabc";
        let tokens = compressor.compress_packed(data);
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_lz77_compress_packed_into() {
        let mut compressor = Lz77Compressor::new(6);
        let data = b"abcabcabc";
        let mut tokens = Vec::new();
        compressor.compress_packed_into(data, &mut tokens);
        assert!(!tokens.is_empty());
    }

    #[test]
    fn test_cost_model_default() {
        let model = CostModel::default();
        // Should be same as fixed
        assert!((model.literal_cost(b'a') - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_cost_model_from_empty_statistics() {
        let lit_counts = [0u32; 286];
        let dist_counts = [0u32; 30];

        // Empty statistics should fall back to fixed costs
        let model = CostModel::from_statistics(&lit_counts, &dist_counts);
        assert!((model.literal_cost(b'a') - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_cost_model_from_statistics_no_distances() {
        let mut lit_counts = [0u32; 286];
        let dist_counts = [0u32; 30];

        lit_counts[b'a' as usize] = 100;
        lit_counts[256] = 1;

        // No distance symbols - should use fixed distance costs
        let model = CostModel::from_statistics(&lit_counts, &dist_counts);
        assert!(model.dist_costs[0] > 0.0);
    }

    #[test]
    fn test_length_to_symbol_all_ranges() {
        // Test a representative value from each length code range
        for length in 3..=258u16 {
            let (symbol, extra_bits) = length_to_symbol(length);
            assert!((257..=285).contains(&symbol));
            assert!(extra_bits <= 5);
        }
    }

    #[test]
    fn test_distance_to_symbol_all_ranges() {
        // Test various distances
        for &dist in &[1, 2, 3, 4, 5, 10, 100, 1000, 10000, 32768] {
            let (symbol, extra_bits) = distance_to_symbol(dist);
            assert!(symbol < 30);
            assert!(extra_bits <= 13);
        }
    }

    #[test]
    fn test_lz77_level_clamping() {
        // Test that level is clamped to valid range
        let compressor = Lz77Compressor::new(0); // Below minimum
        assert!(compressor.config.max_chain_length > 0);

        let compressor = Lz77Compressor::new(100); // Above maximum
        assert!(compressor.config.max_chain_length > 0);
    }

    #[test]
    fn test_lz77_all_levels() {
        let data = b"abcabcabcabcabc";
        for level in 1..=9 {
            let mut compressor = Lz77Compressor::new(level);
            let tokens = compressor.compress(data);
            assert!(!tokens.is_empty());
        }
    }

    #[test]
    fn test_cost_model_match_cost_various_lengths() {
        let model = CostModel::fixed();

        // Test various lengths
        let cost_3 = model.match_cost(3, 1);
        let cost_10 = model.match_cost(10, 1);
        let cost_100 = model.match_cost(100, 1);
        let cost_258 = model.match_cost(258, 1);

        // All should be positive
        assert!(cost_3 > 0.0);
        assert!(cost_10 > 0.0);
        assert!(cost_100 > 0.0);
        assert!(cost_258 > 0.0);
    }

    #[test]
    fn test_cost_model_match_cost_various_distances() {
        let model = CostModel::fixed();

        // Test various distances
        let cost_1 = model.match_cost(3, 1);
        let cost_100 = model.match_cost(3, 100);
        let cost_1000 = model.match_cost(3, 1000);
        let cost_max = model.match_cost(3, 32768);

        // All should be positive
        assert!(cost_1 > 0.0);
        assert!(cost_100 > 0.0);
        assert!(cost_1000 > 0.0);
        assert!(cost_max > 0.0);

        // Larger distances should cost more (more extra bits)
        assert!(cost_max >= cost_1);
    }

    #[test]
    fn test_cost_model_literal_all_bytes() {
        let model = CostModel::fixed();

        for byte in 0u8..=255 {
            let cost = model.literal_cost(byte);
            assert!(cost > 0.0, "Literal {byte} should have positive cost");
            assert!(cost <= 16.0, "Literal {byte} cost should be reasonable");
        }
    }

    #[test]
    fn test_lz77_single_byte() {
        let mut compressor = Lz77Compressor::new(6);
        let data = b"a";
        let tokens = compressor.compress(data);

        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0], Token::Literal(b'a'));
    }

    #[test]
    fn test_lz77_two_bytes() {
        let mut compressor = Lz77Compressor::new(6);
        let data = b"ab";
        let tokens = compressor.compress(data);

        assert_eq!(tokens.len(), 2);
    }

    #[test]
    fn test_lz77_all_same_bytes() {
        let mut compressor = Lz77Compressor::new(6);
        let data = vec![b'a'; 100];
        let tokens = compressor.compress(&data);

        // Should compress very well
        assert!(tokens.len() < 10, "All same bytes should compress well");
    }

    #[test]
    fn test_lz77_run_length_encoding() {
        let mut compressor = Lz77Compressor::new(6);
        // Long run of same byte
        let data: Vec<u8> = vec![0xFF; 1000];
        let tokens = compressor.compress(&data);

        // Should use matches for RLE-like compression
        assert!(tokens.len() < 50);
    }

    #[test]
    fn test_packed_token_boundary_lengths() {
        // Test minimum and maximum lengths
        let min = PackedToken::match_(3, 1);
        assert_eq!(min.as_match(), Some((3, 1)));

        let max_len = PackedToken::match_(258, 1);
        assert_eq!(max_len.as_match(), Some((258, 1)));
    }

    #[test]
    fn test_packed_token_boundary_distances() {
        // Test minimum and maximum distances
        let min = PackedToken::match_(3, 1);
        assert_eq!(min.as_match(), Some((3, 1)));

        let max_dist = PackedToken::match_(3, 32768);
        assert_eq!(max_dist.as_match(), Some((3, 32768)));
    }

    #[test]
    fn test_lz77_compress_deterministic() {
        let data = b"hello world hello world";

        let mut compressor1 = Lz77Compressor::new(6);
        let tokens1 = compressor1.compress(data);

        let mut compressor2 = Lz77Compressor::new(6);
        let tokens2 = compressor2.compress(data);

        assert_eq!(tokens1, tokens2, "Compression should be deterministic");
    }

    #[test]
    fn test_optimal_vs_greedy() {
        let mut compressor = Lz77Compressor::new(6);
        let cost_model = CostModel::fixed();

        // Some inputs where optimal might differ from greedy
        let data = b"abcdeabcdfabcde";

        let greedy_tokens = compressor.compress(data);
        let optimal_tokens = compressor.compress_optimal(data, &cost_model);

        // Both should produce valid output (reconstruct same data)
        let greedy_output = reconstruct_from_tokens(&greedy_tokens);
        let optimal_output = reconstruct_from_tokens(&optimal_tokens);

        assert_eq!(greedy_output, data.to_vec());
        assert_eq!(optimal_output, data.to_vec());
    }

    fn reconstruct_from_tokens(tokens: &[Token]) -> Vec<u8> {
        let mut result = Vec::new();
        for token in tokens {
            match token {
                Token::Literal(b) => result.push(*b),
                Token::Match { length, distance } => {
                    let start = result.len() - *distance as usize;
                    for i in 0..*length as usize {
                        result.push(result[start + i]);
                    }
                }
            }
        }
        result
    }

    #[test]
    fn test_cost_model_short_match_cost() {
        let model = CostModel::fixed();

        // Short match cost
        let short = model.match_cost(3, 1);
        let long = model.match_cost(100, 1);

        // Both should be reasonable
        assert!(short > 0.0);
        assert!(long > 0.0);
    }

    #[test]
    fn test_cost_model_frequency_based() {
        let mut lit_counts = [0u32; 286];
        let mut dist_counts = [0u32; 30];

        // Very skewed distribution
        lit_counts[0] = 1000; // null byte very common
        lit_counts[255] = 1; // 0xFF very rare

        dist_counts[0] = 100; // Short distances common

        let model = CostModel::from_statistics(&lit_counts, &dist_counts);

        // Common byte should cost less
        assert!(model.literal_cost(0) < model.literal_cost(255));
    }

    #[test]
    fn test_lz77_window_size() {
        // Test that matches respect the window size
        let mut data = vec![0u8; 40000];
        // Place pattern at start
        for i in 0..10 {
            data[i] = (i + 1) as u8;
        }
        // Repeat pattern beyond window size
        for i in 0..10 {
            data[35000 + i] = (i + 1) as u8;
        }

        let mut compressor = Lz77Compressor::new(6);
        let tokens = compressor.compress(&data);

        // Should produce valid output
        let reconstructed = reconstruct_from_tokens(&tokens);
        assert_eq!(reconstructed, data);
    }

    #[test]
    fn test_length_to_symbol_edge_cases() {
        // Test exact boundaries between length codes
        assert_eq!(length_to_symbol(3).0, 257);
        assert_eq!(length_to_symbol(4).0, 258);
        assert_eq!(length_to_symbol(5).0, 259);
        assert_eq!(length_to_symbol(6).0, 260);
        assert_eq!(length_to_symbol(7).0, 261);
        assert_eq!(length_to_symbol(8).0, 262);
        assert_eq!(length_to_symbol(9).0, 263);
        assert_eq!(length_to_symbol(10).0, 264);
    }

    #[test]
    fn test_distance_to_symbol_edge_cases() {
        // Small distances have direct codes
        assert_eq!(distance_to_symbol(1).0, 0);
        assert_eq!(distance_to_symbol(2).0, 1);
        assert_eq!(distance_to_symbol(3).0, 2);
        assert_eq!(distance_to_symbol(4).0, 3);
    }

    #[test]
    fn test_optimal_lz77_long_input() {
        let mut compressor = Lz77Compressor::new(6);
        let cost_model = CostModel::fixed();

        // Create longer input with repeating patterns
        let mut data = Vec::new();
        for _ in 0..100 {
            data.extend_from_slice(b"pattern");
        }

        let tokens = compressor.compress_optimal(&data, &cost_model);
        let reconstructed = reconstruct_from_tokens(&tokens);
        assert_eq!(reconstructed, data);
    }

    #[test]
    fn test_hash_collisions() {
        // Test that hash collisions are handled correctly
        let mut compressor = Lz77Compressor::new(6);

        // Data that might cause hash collisions
        let mut data = Vec::new();
        for i in 0..1000 {
            data.push((i % 4) as u8);
        }

        let tokens = compressor.compress(&data);
        let reconstructed = reconstruct_from_tokens(&tokens);
        assert_eq!(reconstructed, data);
    }
}

#[cfg(test)]
mod trace_backwards_tests {
    use super::*;

    #[test]
    fn test_trace_backwards_zero_distance_match_multibyte() {
        let compressor = Lz77Compressor::new(6);

        // Create DP arrays representing:
        // - Token 1: length=5 ending at position 5 (covers bytes 0-4), distance=0 (invalid match)
        // - Token 2: length=2 ending at position 7 (covers bytes 5-6), distance=0 (invalid match)
        // Total: 7 bytes of input data
        //
        // The DP arrays are indexed by end position (0 to n inclusive).
        // length_array[i] = length of token ending at position i
        // dist_array[i] = distance for that token (0 for literals or invalid matches)
        let length_array = vec![0u16, 0, 0, 0, 0, 5, 0, 2];
        let dist_array = vec![0u16, 0, 0, 0, 0, 0, 0, 0];

        let data = b"abcdefg"; // 7 bytes

        let tokens = compressor.trace_backwards(&length_array, &dist_array, data);

        // With the bug:
        // - Token (5, 0): emits literal 'a', data_pos += 1 (should be += 5)
        // - Token (2, 0): emits literal 'b', data_pos += 1 (should be += 2)
        // Result: 2 tokens producing 2 bytes
        //
        // Without the bug:
        // - Token (5, 0): emits literal 'a', data_pos += 5
        // - Token (2, 0): emits literal 'f', data_pos += 2
        // Result: 2 tokens producing 2 bytes, but from correct positions

        // The bug causes data_pos to be incorrect, so subsequent literals read wrong bytes.
        // With bug: tokens are [Literal('a'), Literal('b')]
        // Without bug: tokens are [Literal('a'), Literal('f')]

        assert_eq!(tokens.len(), 2, "Should produce 2 tokens");

        // Verify the second token reads from the correct position (byte 5 = 'f', not byte 1 = 'b')
        match tokens[1] {
            Token::Literal(b) => {
                assert_eq!(
                    b, b'f',
                    "Second literal should be 'f' (byte at position 5), got '{}'. \
                     This indicates data_pos was not advanced correctly.",
                    b as char
                );
            }
            Token::Match { .. } => {
                panic!("Expected literal, got match");
            }
        }
    }
}
