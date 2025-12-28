# Performance Optimization Strategies

This document explains the performance optimization techniques used in pixo. These are general principles that apply to any high-performance code, illustrated with concrete examples from this library.

## The Optimization Mindset

Before diving into techniques, it's worth understanding the hierarchy of optimization impact:

```text
┌─────────────────────────────────────────────────────────────────┐
│  1. Algorithm Choice           (10x-1000x improvement possible) │
├─────────────────────────────────────────────────────────────────┤
│  2. Data Structure Selection   (2x-100x improvement possible)   │
├─────────────────────────────────────────────────────────────────┤
│  3. Memory Access Patterns     (2x-10x improvement possible)    │
├─────────────────────────────────────────────────────────────────┤
│  4. Low-Level Optimizations    (1.1x-2x improvement possible)   │
└─────────────────────────────────────────────────────────────────┘
```

Always start at the top. A better algorithm beats a perfectly-optimized bad algorithm every time.

## Core Principle: Do Less Work

The fastest code is code that doesn't run. Every optimization ultimately reduces the amount of work the CPU must do.

### Example: Deferred Operations

The Adler-32 checksum algorithm requires a modulo operation (`% 65521`) to prevent overflow. The naive approach applies it after every byte:

```text
// Slow: modulo per byte
for byte in data {
    s1 = (s1 + byte) % 65521;
    s2 = (s2 + s1) % 65521;
}
```

But modulo is expensive! Our optimization: do the math to find how many bytes we can process before overflow, then only apply modulo at chunk boundaries:

```rust,ignore
// From src/compress/adler32.rs
const NMAX: usize = 5552;  // Max bytes before overflow

for chunk in data.chunks(NMAX) {
    for &b in chunk {
        s1 += b as u32;
        s2 += s1;
    }
    // Modulo once per chunk instead of per byte
    s1 %= MOD_ADLER;
    s2 %= MOD_ADLER;
}
```

**Result**: We reduced 5552 modulo operations to just 2 per chunk, a 2776x reduction in the most expensive operation.

### Example: Early Exit

When selecting the best PNG filter, we try all 5 filters and pick the one with the lowest "score" (sum of absolute values). But if we find a perfect score of 0, we can stop immediately:

```rust,ignore
// From src/png/filter.rs
let score = score_filter(&scratch.sub);
if score < best_score {
    best_score = score;
    best_filter = FILTER_SUB;
    if best_score == 0 {  // Can't do better than 0!
        return;
    }
}
```

Similarly, for LZ77 matching, if we find a "good enough" match, we skip expensive lazy matching:

```rust,ignore
// From src/compress/lz77.rs
const GOOD_MATCH_LENGTH: usize = 32;

// Skip lazy matching for long matches - they're already good enough
if self.lazy_matching && length < GOOD_MATCH_LENGTH {
    // ... try next position
}
```

## Lookup Tables: Trading Memory for Speed

Computation takes time. Memory lookups are often faster. When you repeatedly compute the same function on a small domain, precompute all results.

### Example: Length and Distance Code Lookup

DEFLATE needs to convert match lengths (3-258) to symbol codes. The naive approach searches through a table:

```text
// Slow: linear search
fn length_code(length: u16) -> u16 {
    for (i, &base) in LENGTH_BASE.iter().enumerate() {
        if length < LENGTH_BASE[i + 1] {
            return 257 + i as u16;
        }
    }
}
```

Our optimization: precompute a direct lookup table at compile time:

```rust,ignore
// From src/compress/deflate.rs
const LENGTH_LOOKUP: [(u8, u8); 256] = {
    let mut table = [(0u8, 0u8); 256];
    // ... populate at compile time
    table
};

#[inline]
fn length_code(length: u16) -> (u16, u8, u16) {
    let idx = (length - 3) as usize;
    let (code_offset, extra_bits) = LENGTH_LOOKUP[idx];  // O(1) lookup
    // ...
}
```

**Result**: O(n) search becomes O(1) lookup. For hot paths called millions of times, this matters.

### Example: Bit Reversal Table

DEFLATE requires reversing the bit order of Huffman codes. Computing this involves a loop:

```text
// Slow: bit-by-bit reversal
fn reverse_bits_slow(code: u16, length: u8) -> u32 {
    let mut result = 0;
    for _ in 0..length {
        result = (result << 1) | (code & 1);
        code >>= 1;
    }
    result
}
```

Our optimization: precompute all 256 byte reversals, then combine them:

```rust,ignore
// From src/compress/deflate.rs
const REVERSE_BYTE: [u8; 256] = { /* computed at compile time */ };

#[inline]
fn reverse_bits(code: u16, length: u8) -> u32 {
    let low = REVERSE_BYTE[code as u8 as usize] as u16;
    let high = REVERSE_BYTE[(code >> 8) as u8 as usize] as u16;
    let reversed = (low << 8) | high;
    (reversed >> (16 - length)) as u32
}
```

**Result**: 16 loop iterations become 2 table lookups and some bit shifts.

## Integer Arithmetic vs Floating Point

Integer operations are generally faster and more predictable than floating-point. When you don't need the precision of floats, use integers with fixed-point arithmetic.

### Example: Color Space Conversion

RGB to YCbCr conversion uses these floating-point formulas:

```text
Y  = 0.299×R + 0.587×G + 0.114×B
Cb = -0.169×R - 0.331×G + 0.5×B + 128
Cr = 0.5×R - 0.419×G - 0.081×B + 128
```

The naive approach uses `f32` with `round()` and `clamp()`. Our optimization: scale coefficients by 256 and use integer math with bit shifts:

```rust,ignore
// From src/color.rs
#[inline]
pub fn rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let r = r as i32;
    let g = g as i32;
    let b = b as i32;

    // Fixed-point coefficients (scaled by 256)
    // +128 for rounding before right shift
    let y = (77 * r + 150 * g + 29 * b + 128) >> 8;
    let cb = ((-43 * r - 85 * g + 128 * b + 128) >> 8) + 128;
    let cr = ((128 * r - 107 * g - 21 * b + 128) >> 8) + 128;

    (y as u8, cb as u8, cr as u8)
}
```

**Key insight**: `>> 8` (right shift by 8) is equivalent to `/ 256`, but much faster. We've eliminated all floating-point operations.

## Choosing the Right Data Type

Using smaller data types can improve performance through better cache utilization and SIMD vectorization.

### Memory Hierarchy Matters

```text
┌─────────────────────────────────────────────────────────────┐
│  L1 Cache:   ~1 ns access,   64 KB                          │
├─────────────────────────────────────────────────────────────┤
│  L2 Cache:   ~4 ns access,   256 KB                         │
├─────────────────────────────────────────────────────────────┤
│  L3 Cache:   ~12 ns access,  8 MB                           │
├─────────────────────────────────────────────────────────────┤
│  RAM:        ~100 ns access, GBs                            │
└─────────────────────────────────────────────────────────────┘
```

If your data fits in a smaller cache level, everything runs faster. Using `u8` instead of `u32` means 4x more data fits in cache.

### Example: Hash Table Sizing

In LZ77, we store positions as `i32` (4 bytes) instead of `usize` (8 bytes on 64-bit):

```rust,ignore
// From src/compress/lz77.rs
pub struct Lz77Compressor {
    head: Vec<i32>,  // 4 bytes, not 8
    prev: Vec<i32>,  // 4 bytes, not 8
    // ...
}
```

This halves memory usage and doubles cache efficiency. We use `i32` (signed) so we can use `-1` as a sentinel for "no entry" — a common pattern.

## Batching Operations

Processing items one at a time has overhead. Processing them in batches amortizes that overhead.

### Example: Multi-Byte Comparison

When finding LZ77 match lengths, the naive approach compares byte by byte:

```text
// Slow: byte-by-byte
while length < max && data[pos1 + length] == data[pos2 + length] {
    length += 1;
}
```

Our optimization: compare 8 bytes at a time using `u64`:

```rust,ignore
// From src/compress/lz77.rs
// Compare 8 bytes at a time using u64
while length + 8 <= max_len {
    let a = u64::from_ne_bytes(data[pos1+length..pos1+length+8].try_into().unwrap());
    let b = u64::from_ne_bytes(data[pos2+length..pos2+length+8].try_into().unwrap());
    if a != b {
        // Find first differing byte using trailing zeros
        let xor = a ^ b;
        length += (xor.trailing_zeros() / 8) as usize;
        return length;
    }
    length += 8;
}
```

**Key insight**: `trailing_zeros()` on the XOR tells us exactly where the first difference is, without a loop.

### Example: Batch Bit Writing

The JPEG bit writer was originally implemented as a bit-by-bit loop. Our optimization processes multiple bits per iteration:

```rust,ignore
// From src/bits.rs (BitWriterMsb)
while remaining > 0 {
    let space = self.bit_position;
    let to_write = remaining.min(space);

    // Write up to 8 bits at once
    let shift = remaining - to_write;
    let mask = (1u32 << to_write) - 1;
    let bits = ((val >> shift) & mask) as u8;

    self.bit_position -= to_write;
    self.current_byte |= bits << self.bit_position;
    remaining -= to_write;
    // ...
}
```

## SIMD: Single Instruction, Multiple Data

SIMD instructions process multiple values simultaneously. A 128-bit register can hold 16 bytes, and one instruction operates on all 16.

```text
┌────────────────────────────────────────────────────────┐
│                   Scalar Addition                       │
│                                                         │
│   a[0] + b[0] = c[0]  (1 operation)                    │
│   a[1] + b[1] = c[1]  (1 operation)                    │
│   a[2] + b[2] = c[2]  (1 operation)                    │
│   ...                                                   │
│   a[15] + b[15] = c[15]  (1 operation)                 │
│                                                         │
│   Total: 16 operations                                  │
└────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────┐
│                   SIMD Addition                         │
│                                                         │
│   ┌─────────────────────────────────────────────────┐  │
│   │ a[0] a[1] a[2] ... a[15] │  (128-bit register)  │  │
│   └─────────────────────────────────────────────────┘  │
│                      +                                  │
│   ┌─────────────────────────────────────────────────┐  │
│   │ b[0] b[1] b[2] ... b[15] │  (128-bit register)  │  │
│   └─────────────────────────────────────────────────┘  │
│                      =                                  │
│   ┌─────────────────────────────────────────────────┐  │
│   │ c[0] c[1] c[2] ... c[15] │  (128-bit register)  │  │
│   └─────────────────────────────────────────────────┘  │
│                                                         │
│   Total: 1 operation (16x speedup potential)           │
└────────────────────────────────────────────────────────┘
```

### Example: SIMD Adler-32

The standard Adler-32 loop processes one byte at a time. With SSSE3, we process 16 bytes at once:

```rust,ignore
// From src/simd/x86_64.rs (simplified)
pub unsafe fn adler32_ssse3(data: &[u8]) -> u32 {
    // Process 16 bytes at a time
    let chunk = _mm_loadu_si128(ptr as *const __m128i);

    // Use SIMD multiply-add for weighted sums
    // (the actual implementation is more complex)
}
```

### Example: SIMD PNG Filter Scoring

Calculating the sum of absolute values for filter selection:

```rust,ignore
// From src/simd/x86_64.rs
pub unsafe fn score_filter_sse2(filtered: &[u8]) -> u64 {
    let mut sum = _mm_setzero_si128();

    // Process 16 bytes at a time
    while remaining >= 16 {
        let chunk = _mm_loadu_si128(ptr as *const __m128i);
        // Sum absolute differences from zero
        let sad = _mm_sad_epu8(chunk, _mm_setzero_si128());
        sum = _mm_add_epi64(sum, sad);
    }
    // ...
}
```

**Key SIMD operations used**:

- `_mm_loadu_si128`: Load 16 bytes into a register
- `_mm_sad_epu8`: Sum of Absolute Differences (perfect for filter scoring)
- `_mm_cmpeq_epi8`: Compare 16 bytes at once (for match length)
- `_mm_movemask_epi8`: Convert comparison results to a bitmask

### Runtime Feature Detection

Different CPUs support different SIMD instruction sets. We detect at runtime:

```rust,ignore
// From src/simd/mod.rs
pub fn adler32(data: &[u8]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("ssse3") {
            return unsafe { x86_64::adler32_ssse3(data) };
        }
    }

    // Fallback to scalar implementation
    fallback::adler32(data)
}
```

This pattern ensures the code runs everywhere while taking advantage of faster instructions when available.

## Caching Computed Results

When you compute the same thing repeatedly, cache the result.

### Example: Fixed Huffman Code Caching

DEFLATE's fixed Huffman codes are the same for every compression. Instead of regenerating them each time:

```rust,ignore
// From src/compress/huffman.rs
use std::sync::LazyLock;

static FIXED_LITERAL_CODES: LazyLock<Vec<HuffmanCode>> = LazyLock::new(|| {
    // Generate codes once, on first access
    let mut lengths = vec![0u8; 288];
    // ... set up lengths
    generate_canonical_codes(&lengths)
});

#[inline]
pub fn fixed_literal_codes() -> &'static [HuffmanCode] {
    &FIXED_LITERAL_CODES  // O(1) after first call
}
```

`LazyLock` ensures thread-safe, one-time initialization.

### Example: Reusing Scratch Buffers

When processing PNG rows, we need temporary buffers for each filter type. Instead of allocating new buffers per row:

```rust,ignore
// From src/png/filter.rs
struct AdaptiveScratch {
    none: Vec<u8>,
    sub: Vec<u8>,
    up: Vec<u8>,
    avg: Vec<u8>,
    paeth: Vec<u8>,
}

impl AdaptiveScratch {
    fn clear(&mut self) {
        // Just reset length, keep capacity
        self.none.clear();
        self.sub.clear();
        // ...
    }
}
```

We allocate once and reuse. The vectors keep their capacity between rows, avoiding repeated allocation.

## Transform Before Processing

Sometimes the best optimization isn't in the algorithm—it's in reshaping the data before the algorithm runs.

### The Pattern

```text
┌─────────────────────────────────────────────────────────────────┐
│  Input Data          Transform           Optimized Data         │
│  ──────────    →    ───────────    →    ──────────────          │
│  [scattered]        [reorder for        [spatially              │
│                      locality]           coherent]              │
│                                                                 │
│  The same algorithm runs faster on well-structured data.        │
└─────────────────────────────────────────────────────────────────┘
```

### Example: Palette Reordering (Zeng Algorithm)

PNG palette images assign an index (0-255) to each pixel. The naive approach assigns indices in first-occurrence order. But compression works better when similar values are adjacent.

```text
Before: Pixel sequence uses scattered indices
┌─────────────────────────────────────────┐
│  3  7  3  12  7  3  45  12  7  3  ...  │  Indices jump around
└─────────────────────────────────────────┘

After: Reorder palette so spatially adjacent pixels have adjacent indices
┌─────────────────────────────────────────┐
│  1  2  1  3   2  1  4   3   2  1  ...  │  Smooth transitions
└─────────────────────────────────────────┘
```

The Zeng algorithm builds a **co-occurrence matrix** counting which colors appear next to each other, then reorders the palette to minimize index differences between neighbors.

**Result**: 20-40% smaller files for palette images, with zero change to the compression algorithm itself.

### Example: Progressive JPEG Scan Order

Instead of encoding each 8×8 block completely before moving to the next, progressive JPEG groups coefficients by frequency:

```text
Sequential: Block₁[all 64 coeffs] → Block₂[all 64] → Block₃[all 64] → ...

Progressive: All DC coeffs → All low AC → All high AC
             (similar data grouped together = better compression)
```

Same data, different order — 5-15% smaller files.

### When to Apply This Pattern

Ask: "Are there relationships in my data that the algorithm can't see?"

- **Spatial locality**: Adjacent pixels often have similar values
- **Frequency correlation**: Similar-frequency DCT coefficients have similar distributions
- **Temporal patterns**: Sequential data often has predictable deltas

Transform the data to expose these patterns before compression.

---

## Structure the Search Space

When facing a combinatorial decision problem, don't enumerate all possibilities. Model it as a graph and find the shortest path.

### The Problem: Exponential Choices

Many optimization problems seem to require trying every combination:

```text
Quantization example:
─────────────────────
64 coefficients, each could round up or down
Brute force: 2^64 possibilities = 18 quintillion combinations

LZ77 parsing example:
─────────────────────
At each position: literal OR match of length 3, 4, 5, ... 258
10,000 byte file = astronomical combinations
```

### The Solution: Model as a Graph

Instead of brute force, observe that:

1. **Decisions are sequential** — you make choices one position at a time
2. **Future costs don't depend on how you got here** — only on your current state
3. **You can compute optimal paths efficiently** — using DP or shortest-path algorithms

```text
Trellis Quantization Graph
══════════════════════════

Position 0        Position 1        Position 2
   [4]───────────────[2]───────────────[0]
    │╲               │╲               │
    │ ╲cost          │ ╲              │
   [5]──╲───────────[3]──╲───────────[1]
          ╲               ╲
          [6]─────────────[4]─────────────[2]

Each node = a possible quantized value
Each edge = cost (rate + λ × distortion) to transition
Solution  = shortest path from start to end
```

**Key insight**: You only need to track the _best way to reach each state_, not all possible paths. This reduces 2^N to O(N × S²) where S is states per position.

### Example: Optimal LZ77 Parsing

Greedy LZ77 takes the longest match at each position. But sometimes a shorter match enables a better match later:

```text
Data: "ABCABCABCD"

Greedy:     ABC (match 3) + ABCD (match 4) = suboptimal
Optimal:    ABCABC (match 6) + D (literal) = better

The greedy choice at position 3 blocks a longer match at position 0.
```

The optimal approach:

1. Build a graph: each position has edges for "literal" and all valid match lengths
2. Edge costs = actual bit cost (from Huffman statistics)
3. Find shortest path with forward DP

```rust,ignore
// Forward DP: cost[i] = minimum bits to encode bytes 0..i
for i in 0..n {
    // Try literal
    let lit_cost = cost[i] + literal_bits(data[i]);
    if lit_cost < cost[i + 1] {
        cost[i + 1] = lit_cost;
    }

    // Try each match length
    for (len, dist) in find_matches(data, i) {
        let match_cost = cost[i] + match_bits(len, dist);
        if match_cost < cost[i + len] {
            cost[i + len] = match_cost;
        }
    }
}
```

**Result**: 3-8% smaller files by finding globally optimal parse instead of locally greedy.

### When to Apply This Pattern

Ask: "Am I making sequential decisions where each choice affects future options?"

If yes, model it as a graph:

- **Nodes** = states (what information do I need to make the next decision?)
- **Edges** = choices with costs
- **Solution** = shortest path (Viterbi, Dijkstra, or forward DP)

The key properties that make this work:

- **Optimal substructure**: Best solution contains best sub-solutions
- **Limited state space**: You don't need to track everything—just enough to evaluate future costs

---

## Multi-Pass Processing

Sometimes you need information about the whole input to make optimal decisions. The solution: analyze first, then process.

### The Trade-off

```text
┌─────────────────────────────────────────────────────────────────┐
│                    Single-Pass vs Multi-Pass                     │
│                                                                  │
│  Single-pass:  Faster, streaming-friendly, but uses estimates   │
│  Multi-pass:   Slower, requires buffering, but optimal choices  │
│                                                                  │
│  When encoding time matters less than output quality,           │
│  multi-pass wins.                                               │
└─────────────────────────────────────────────────────────────────┘
```

### Example: Optimized Huffman Tables

DEFLATE can use fixed Huffman codes (single-pass) or build optimal codes for the specific data (two-pass):

```text
Pass 1: Count symbol frequencies
        literals: {0: 5, 1: 12, 2: 8, ...}
        distances: {1: 20, 2: 15, ...}

Pass 2: Build optimal Huffman tree from frequencies
        Encode with custom codes
```

**Result**: 2-5% smaller files. The table overhead is ~300 bits; savings are usually much larger.

### Example: Iterative Refinement

Some problems have circular dependencies—you need to solve A to solve B, but B affects A:

```text
LZ77 + Huffman Chicken-and-Egg:
──────────────────────────────
Best LZ77 parse depends on Huffman bit costs
Huffman bit costs depend on LZ77 symbol frequencies
Which do you compute first?
```

The solution: iterate until convergence.

```text
Iteration 1: Greedy LZ77 → rough frequencies → initial Huffman costs
Iteration 2: Optimal LZ77 with costs → better frequencies → refined costs
Iteration 3: Optimal LZ77 with refined costs → even better frequencies
...
Iteration N: Converged to (near-)optimal solution
```

Zopfli uses 15 iterations by default. Each iteration improves by smaller amounts until the solution stabilizes.

### When to Apply This Pattern

Multi-pass is worth it when:

- **Output size matters more than encoding speed** (storage, bandwidth)
- **The input fits in memory** (can buffer for second pass)
- **Decisions early in the stream affect optimal choices later**

---

## Algorithm Selection: The AAN Fast DCT

Sometimes the biggest wins come from choosing a better algorithm entirely.

### The Problem: DCT is Expensive

The naive 1D 8-point DCT needs 64 multiplications; a fully naive 2D 8×8 would take 4096. Separability drops this to 2×8³ = 1024 multiplications, but that's still costly:

```text
// Naive 1D DCT
for k in 0..8 {
    for n in 0..8 {
        output[k] += input[n] * cos((2n+1) * k * π / 16);
    }
}
```

That's O(n²) — 64 multiplications for 8 points.

### The Solution: AAN Algorithm

The Arai-Agui-Nakajima (AAN) algorithm reduces this to just **5 multiplications and 29 additions**:

```rust,ignore
// From src/jpeg/dct.rs
fn aan_dct_1d(data: &mut [f32; 8]) {
    // Stage 1: Butterfly operations (additions only)
    let tmp0 = data[0] + data[7];
    let tmp7 = data[0] - data[7];
    // ... more additions

    // Rotations (the only multiplications)
    let z1 = (tmp12 + tmp13) * A1;
    let z5 = (tmp10 - tmp12) * A5;
    // ... just 5 multiplications total

    // Apply post-scaling
    for i in 0..8 {
        data[i] *= S[i];
    }
}
```

**Result**: 5 multiplications instead of 64, a 12x reduction in the most expensive operation.

## Parallel Processing

Modern CPUs have multiple cores. Use them!

### Example: Parallel PNG Filter Selection

Each row of a PNG can be filtered independently (with access to the previous row for context). We parallelize with rayon:

```rust,ignore
// From src/png/filter.rs
#[cfg(feature = "parallel")]
fn apply_filters_parallel(data: &[u8], height: usize, ...) -> Vec<u8> {
    let rows: Vec<Vec<u8>> = (0..height)
        .into_par_iter()  // Parallel iterator!
        .map(|y| {
            // Each row processed independently
            let row = &data[y * row_bytes..(y+1) * row_bytes];
            filter_row(row, prev_row, ...)
        })
        .collect();

    // Combine results
    rows.into_iter().flatten().collect()
}
```

**Key insight**: Compression has inherent parallelism at the row level. The work per row is significant enough to amortize threading overhead.

## Bits, Bytes, and Binary Representation

Understanding binary representation is fundamental to compression optimization.

### Bit Shifting for Division and Multiplication

Bit shifts are much faster than division/multiplication by powers of 2:

```text
x >> n  is equivalent to  x / (2^n)   but faster
x << n  is equivalent to  x * (2^n)   but faster
```

We use this extensively for fixed-point arithmetic:

```rust,ignore
// Division by 256 using bit shift
let y = (77 * r + 150 * g + 29 * b + 128) >> 8;
```

### Bit Masking for Modulo

For powers of 2, bitwise AND is faster than modulo:

```text
x & (n-1)  is equivalent to  x % n  when n is a power of 2
```

We use this for hash table indexing:

```rust,ignore
// From src/compress/lz77.rs
const HASH_SIZE: usize = 1 << 15;  // 32768 (power of 2)

fn hash3(data: &[u8], pos: usize) -> usize {
    // ... compute hash
    (hash >> 17) as usize & (HASH_SIZE - 1)  // Fast modulo
}
```

### Finding Differences with XOR

XOR highlights differences between values:

```text
a ^ b = 0  means a == b
a ^ b ≠ 0  means a ≠ b, and the set bits show where they differ
```

We use this for fast string matching:

```rust,ignore
// Find first differing byte in 8-byte comparison
let xor = a ^ b;
if xor != 0 {
    // trailing_zeros tells us the position of first difference
    let diff_pos = (xor.trailing_zeros() / 8) as usize;
}
```

## Summary: The Optimization Checklist

When optimizing, consider these techniques in order of impact:

1. **Choose the right algorithm**

   - AAN DCT instead of naive DCT
   - Hash tables instead of linear search

2. **Transform data before processing**

   - Reorder for spatial locality (palette sorting)
   - Group similar data together (progressive encoding)

3. **Structure the search space**

   - Model decisions as a graph
   - Use DP/Viterbi instead of brute-force enumeration
   - Find optimal paths, not all paths

4. **Consider multi-pass when output quality matters**

   - Analyze first, then optimize (optimized Huffman)
   - Iterate when there are circular dependencies

5. **Use appropriate data structures**

   - Lookup tables for repeated computations
   - Power-of-2 sizes for fast modulo

6. **Reduce work**

   - Defer expensive operations (batch modulo)
   - Early exit when possible
   - Skip unnecessary work (good match threshold)

7. **Use efficient numeric representations**

   - Integer arithmetic instead of floating-point
   - Smaller types when range permits
   - Fixed-point for fractional values

8. **Batch operations**

   - Process 8 bytes at once with u64
   - Process 16 bytes at once with SIMD
   - Write multiple bits at once

9. **Cache results**

   - Precompute lookup tables
   - Lazy-initialize constants
   - Reuse scratch buffers

10. **Parallelize**
    - Use rayon for row-level parallelism
    - Runtime feature detection for SIMD

## Next Steps

These optimization patterns are applied throughout pixo. To see them in action:

- **Data transformation**: `src/png/mod.rs` (Zeng palette sorting)
- **Graph-based optimization**: `src/jpeg/trellis.rs` (trellis quantization)
- **Optimal parsing**: `src/compress/lz77.rs` (forward DP for LZ77)
- **Multi-pass processing**: `src/compress/deflate.rs` (iterative refinement)
- **Lookup tables**: `src/compress/deflate.rs` (LENGTH_LOOKUP, DISTANCE_LOOKUP_SMALL)
- **Fixed-point arithmetic**: `src/color.rs` (rgb_to_ycbcr)
- **SIMD implementations**: `src/simd/x86_64.rs`
- **Fast DCT algorithm**: `src/jpeg/dct.rs`
- **Batch bit operations**: `src/bits.rs`

Continue to [Compression Evolution](./compression-evolution.md) to explore the history and philosophy behind modern compression techniques.

---

## References

- Agner Fog's optimization manuals: <https://www.agner.org/optimize/>
- Intel Intrinsics Guide: <https://www.intel.com/content/www/us/en/docs/intrinsics-guide/>
- Arai, Agui, Nakajima (1988). "A Fast DCT-SQ Scheme for Images"
- "What Every Programmer Should Know About Memory" by Ulrich Drepper
