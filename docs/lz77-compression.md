# LZ77 Compression

LZ77, invented by Abraham Lempel and Jacob Ziv in 1977, is the foundation of many compression formats including DEFLATE (used in PNG, gzip, and ZIP). It exploits a fundamental observation: **data often contains repeated patterns**.

## The Core Insight

Consider this text:

```
"the cat sat on the mat"
```

Notice anything? The word "the" appears twice, and "at" appears three times! Instead of spelling everything out, we could say:

```
"the cat sat on (copy 13 chars back, length 4)mat"
                 └── refers to "the " ──┘
```

This is the essence of LZ77: replace repeated sequences with **back-references** to earlier occurrences.

## How It Works

LZ77 uses a **sliding window** that looks back at recently processed data. When it finds a match between current data and something in the window, it outputs a (length, distance) pair instead of the literal bytes.

### The Sliding Window

```
Already processed (can reference)    Current position    Not yet seen
         ◄──── WINDOW ────►                │
┌─────────────────────────────────────────────────────────────────────┐
│ t h e   c a t   s a t   o n   | t h e   m a t                       │
└─────────────────────────────────────────────────────────────────────┘
                                  ▲
                                  └── Looking for matches here
```

At the current position, we see "the ". Looking back in the window, we find "the " at distance 14. Instead of encoding 4 bytes, we encode a reference: (length=4, distance=14).

### Token Types

LZ77 produces a stream of **tokens**, each being either:

1. **Literal**: A single byte that couldn't be matched
2. **Match**: A (length, distance) pair referring to previous data

```rust
// From src/compress/lz77.rs
pub enum Token {
    /// A literal byte that couldn't be compressed.
    Literal(u8),
    /// A back-reference: (length, distance).
    Match {
        length: u16,   // 3-258 bytes
        distance: u16, // 1-32768 bytes back
    },
}
```

## Step-by-Step Example

Let's compress: `ABRACADABRA`

```
Position 0: A
  Window: (empty)
  No match found → Literal 'A'

Position 1: B
  Window: A
  No match found → Literal 'B'

Position 2: R
  Window: AB
  No match found → Literal 'R'

Position 3: A
  Window: ABR
  Match found: 'A' at distance 3, length 1
  But minimum match length is 3, so → Literal 'A'

Position 4: C
  Window: ABRA
  No match found → Literal 'C'

Position 5: A
  Window: ABRAC
  No match found (no 3+ char match) → Literal 'A'

Position 6: D
  Window: ABRACA
  No match found → Literal 'D'

Position 7: A
  Window: ABRACAD
  Looking for match starting with 'A'...
  Found "ABRA" at distance 7, length 4!
  → Match(length=4, distance=7)

Final output: A B R A C A D [4,7]
              7 literals + 1 match = much smaller than 11 literals!
```

## Window Size Matters

The window size is a trade-off:

| Larger Window        | Smaller Window        |
| -------------------- | --------------------- |
| ✓ Find more matches  | ✓ Less memory needed  |
| ✓ Better compression | ✓ Faster searching    |
| ✗ More memory        | ✗ Fewer matches found |
| ✗ Slower searching   | ✗ Worse compression   |

DEFLATE uses a **32KB window** (32,768 bytes), a well-chosen balance that works great for most data.

```rust
// From src/compress/lz77.rs
/// Maximum distance to look back for matches (32KB window).
pub const MAX_DISTANCE: usize = 32768;

/// Maximum match length (as per DEFLATE spec).
pub const MAX_MATCH_LENGTH: usize = 258;

/// Minimum match length worth encoding.
pub const MIN_MATCH_LENGTH: usize = 3;
```

## Finding Matches Efficiently: Hash Chains

Naive matching (comparing every position) is O(n × window_size) — too slow!

The solution: **hash chains**. We hash the first 3 bytes at each position and maintain chains of positions with the same hash.

### How Hash Chains Work

```
Data: "the cat sat on the mat"
                      ▲
                      └── Current position, looking at "the"

Hash table (indexed by hash of 3 bytes):
  hash("the") → [14] → [0] → [-1]  (positions with "the" or hash collision)
  hash("cat") → [4] → [-1]
  hash("sat") → [8] → [-1]
  ...
```

When we need to find a match for "the" at position 14:

1. Compute hash("the")
2. Look up the chain for that hash
3. Check each position in the chain for actual matches
4. Follow the chain until we find the best match or reach the limit

```rust
// From src/compress/lz77.rs
/// Hash function for 3-byte sequences.
fn hash3(data: &[u8], pos: usize) -> usize {
    if pos + 2 >= data.len() {
        return 0;
    }
    let h = (data[pos] as u32)
        | ((data[pos + 1] as u32) << 8)
        | ((data[pos + 2] as u32) << 16);
    // Multiply by a prime and take high bits
    ((h.wrapping_mul(2654435769)) >> 17) as usize & (HASH_SIZE - 1)
}
```

The hash function uses a **multiplicative hash** (0x1E35A7BD, a Knuth-style constant) to distribute 4-byte sequences evenly across the table.

## Compression Levels

Different compression levels trade speed for compression ratio by adjusting chain depth; lazy matching is disabled in our parser because it slightly worsened PNG-style data:

| Level | Chain Length | Lazy Matching |
| ----- | ------------ | ------------- |
| 1     | 4            | No            |
| 2     | 8            | No            |
| 3     | 16           | No            |
| 4     | 32           | No            |
| 5     | 64           | No            |
| 6     | 128          | No            |
| 7     | 256          | No            |
| 8     | 1024         | No            |
| 9     | 4096         | No            |

```rust
// From src/compress/lz77.rs
let (max_chain_length, lazy_matching) = match level {
    1 => (4, false),
    2 => (8, false),
    3 => (16, false),
    4 => (32, false),
    5 => (64, false),
    6 => (128, false),
    7 => (256, false),
    8 => (1024, false),
    9 => (4096, false),
    _ => (128, false),
};
```

> Note: Lazy matching can improve some compressors, but our profiling on PNG-like data favored deeper chains without deferrals.

## Why Minimum Length 3?

Encoding a match requires:

- Length code: variable bits (at least 1)
- Distance code: 5+ bits

A 2-byte match would need similar bits to just sending 2 literal bytes. The break-even point is around 3 bytes, so shorter matches aren't worth encoding.

## Special Cases

### Overlapping Matches

LZ77 can reference data that overlaps the current position! This enables powerful run-length encoding.

```
Data: "AAAAAAAAAA" (10 A's)

Position 0: Literal 'A'
Position 1: Match(length=9, distance=1)
            ← This copies from position 0, but extends beyond current position!

During decompression:
  Output: A
  Copy from distance 1: A → Output: AA
  Copy continues:       A → Output: AAA
  ...until length 9 is reached
```

This is why LZ77 can compress runs extremely efficiently!

### The Empty Match

What about the very beginning when the window is empty? We simply can't find any matches — everything starts as literals.

## LZ77 vs. Other Algorithms

| Algorithm | Year | Approach                                 |
| --------- | ---- | ---------------------------------------- |
| LZ77      | 1977 | Sliding window, explicit back-references |
| LZ78      | 1978 | Build dictionary of phrases              |
| LZW       | 1984 | LZ78 variant, used in GIF                |
| LZSS      | 1982 | LZ77 variant, flag literals vs matches   |
| LZMA      | 1998 | Advanced LZ77 + range coding             |

DEFLATE uses a variant of LZSS combined with Huffman coding.

## Practical Compression Ratios

LZ77 alone achieves:

- **English text**: 40-50% of original
- **Source code**: 30-40% of original
- **Already compressed**: ~100% (no benefit)
- **Random data**: ~100% (no patterns to find)

When combined with Huffman coding (as in DEFLATE), these improve to:

- **English text**: 30-40% of original
- **Source code**: 20-30% of original

## The Algorithm in Full

Here's the complete compression loop:

```rust
// From src/compress/lz77.rs - simplified
pub fn compress(&mut self, data: &[u8]) -> Vec<Token> {
    let mut tokens = Vec::with_capacity(data.len());
    let mut pos = 0;

    while pos < data.len() {
        let best_match = self.find_best_match(data, pos);

        if let Some((length, distance)) = best_match {
            // Optional: Check for lazy match
            if self.lazy_matching && pos + 1 < data.len() {
                if let Some((next_length, _)) = self.find_best_match(data, pos + 1) {
                    if next_length > length + 1 {
                        tokens.push(Token::Literal(data[pos]));
                        self.update_hash(data, pos);
                        pos += 1;
                        continue;
                    }
                }
            }

            // Emit match
            tokens.push(Token::Match {
                length: length as u16,
                distance: distance as u16,
            });

            // Update hash for all matched positions
            for i in 0..length {
                self.update_hash(data, pos + i);
            }
            pos += length;
        } else {
            // No match found, emit literal
            tokens.push(Token::Literal(data[pos]));
            self.update_hash(data, pos);
            pos += 1;
        }
    }

    tokens
}
```

## Memory Usage

Our implementation uses:

- **Hash table**: 32K entries × 4 bytes = 128 KB
- **Chain links**: 32K entries × 4 bytes = 128 KB
- **Total**: ~256 KB for the compressor state

This is modest by modern standards, but was a significant consideration when LZ77 was invented in 1977!

## Summary

LZ77 provides:

- **Dictionary compression** via back-references
- **Sliding window** for bounded memory
- **Hash chains** for fast matching
- **Compression levels** for speed/quality trade-offs
- Foundation for DEFLATE, gzip, ZIP, PNG, and more

The genius of LZ77 is its simplicity: just look for repeating patterns and point to them. This simple idea, combined with Huffman coding, powers most of the compression we use daily.

## Next Steps

Continue to [DEFLATE Algorithm](./deflate.md) to see how LZ77 tokens are efficiently encoded using Huffman codes.

---

## References

- Ziv, J. and Lempel, A. (1977). "A Universal Algorithm for Sequential Data Compression"
- [RFC 1951 Section 4 - Compression algorithm details](https://www.rfc-editor.org/rfc/rfc1951#section-4)
- See implementation: `src/compress/lz77.rs`
