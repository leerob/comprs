# Performance Optimization Strategies

This document explains the performance optimization techniques used in comprs. These are general principles that apply to any high-performance code, illustrated with concrete examples from this library.

## The Optimization Mindset

Before diving into techniques, it's worth understanding the hierarchy of optimization impact:

```
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

```
// Slow: modulo per byte
for byte in data {
    s1 = (s1 + byte) % 65521;
    s2 = (s2 + s1) % 65521;
}
```

But modulo is expensive! Our optimization: do the math to find how many bytes we can process before overflow, then only apply modulo at chunk boundaries:

```rust
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

**Result**: We reduced 5552 modulo operations to just 2 per chunk — a 2776x reduction in the most expensive operation.

### Example: Early Exit

When selecting the best PNG filter, we try all 5 filters and pick the one with the lowest "score" (sum of absolute values). But if we find a perfect score of 0, we can stop immediately:

```rust
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

```rust
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

```
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

```rust
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

```
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

```rust
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

```
Y  = 0.299×R + 0.587×G + 0.114×B
Cb = -0.169×R - 0.331×G + 0.5×B + 128
Cr = 0.5×R - 0.419×G - 0.081×B + 128
```

The naive approach uses `f32` with `round()` and `clamp()`. Our optimization: scale coefficients by 256 and use integer math with bit shifts:

```rust
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

```
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

```rust
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

```
// Slow: byte-by-byte
while length < max && data[pos1 + length] == data[pos2 + length] {
    length += 1;
}
```

Our optimization: compare 8 bytes at a time using `u64`:

```rust
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

```rust
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

```
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

```rust
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

```rust
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

```rust
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

```rust
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

```rust
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

## Algorithm Selection: The AAN Fast DCT

Sometimes the biggest wins come from choosing a better algorithm entirely.

### The Problem: DCT is Expensive

The naive 2D DCT requires 64 multiplications per 8-point transform:

```
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

```rust
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

**Result**: 5 multiplications instead of 64 — a 12x reduction in the most expensive operation.

## Parallel Processing

Modern CPUs have multiple cores. Use them!

### Example: Parallel PNG Filter Selection

Each row of a PNG can be filtered independently (with access to the previous row for context). We parallelize with rayon:

```rust
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

```
x >> n  is equivalent to  x / (2^n)   but faster
x << n  is equivalent to  x * (2^n)   but faster
```

We use this extensively for fixed-point arithmetic:

```rust
// Division by 256 using bit shift
let y = (77 * r + 150 * g + 29 * b + 128) >> 8;
```

### Bit Masking for Modulo

For powers of 2, bitwise AND is faster than modulo:

```
x & (n-1)  is equivalent to  x % n  when n is a power of 2
```

We use this for hash table indexing:

```rust
// From src/compress/lz77.rs
const HASH_SIZE: usize = 1 << 15;  // 32768 (power of 2)

fn hash3(data: &[u8], pos: usize) -> usize {
    // ... compute hash
    (hash >> 17) as usize & (HASH_SIZE - 1)  // Fast modulo
}
```

### Finding Differences with XOR

XOR highlights differences between values:

```
a ^ b = 0  means a == b
a ^ b ≠ 0  means a ≠ b, and the set bits show where they differ
```

We use this for fast string matching:

```rust
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

2. **Use appropriate data structures**
   - Lookup tables for repeated computations
   - Power-of-2 sizes for fast modulo

3. **Reduce work**
   - Defer expensive operations (batch modulo)
   - Early exit when possible
   - Skip unnecessary work (good match threshold)

4. **Use efficient numeric representations**
   - Integer arithmetic instead of floating-point
   - Smaller types when range permits
   - Fixed-point for fractional values

5. **Batch operations**
   - Process 8 bytes at once with u64
   - Process 16 bytes at once with SIMD
   - Write multiple bits at once

6. **Cache results**
   - Precompute lookup tables
   - Lazy-initialize constants
   - Reuse scratch buffers

7. **Parallelize**
   - Use rayon for row-level parallelism
   - Runtime feature detection for SIMD

## Next Steps

These optimization patterns are applied throughout comprs. To see them in action:

- **Lookup tables**: `src/compress/deflate.rs` (LENGTH_LOOKUP, DISTANCE_LOOKUP_SMALL)
- **Fixed-point arithmetic**: `src/color.rs` (rgb_to_ycbcr)
- **SIMD implementations**: `src/simd/x86_64.rs`
- **Fast DCT algorithm**: `src/jpeg/dct.rs`
- **Batch bit operations**: `src/bits.rs`

---

## References

- Agner Fog's optimization manuals: https://www.agner.org/optimize/
- Intel Intrinsics Guide: https://www.intel.com/content/www/us/en/docs/intrinsics-guide/
- Arai, Agui, Nakajima (1988). "A Fast DCT-SQ Scheme for Images"
- "What Every Programmer Should Know About Memory" by Ulrich Drepper
