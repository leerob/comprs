# Introduction to Rust

This document introduces the Rust programming language through the lens of pixo. Rather than covering Rust in the abstract, we'll explore how its unique features enable us to write safe, fast, and expressive compression code.

If you're coming from C, C++, Python, or JavaScript, Rust will feel both familiar and surprisingly different. The initial learning curve is real—but the payoff is code that's fast like C, safe like a garbage-collected language, and expressive like a modern functional language.

## Why Rust for Compression?

Compression algorithms have demanding requirements:

| Requirement           | Challenge                           | Rust's Solution                         |
| --------------------- | ----------------------------------- | --------------------------------------- |
| **Performance**       | Bit manipulation, tight loops, SIMD | Zero-cost abstractions, inline assembly |
| **Memory safety**     | Buffer overflows, use-after-free    | Compile-time ownership checking         |
| **Correctness**       | Edge cases, off-by-one errors       | Strong type system, exhaustive matching |
| **Low-level control** | Exact memory layout, no GC pauses   | No runtime, predictable performance     |

Rust gives us C-level performance without C-level bugs. In a compression library, where we're manipulating bits and bytes at high speed, this matters: one buffer overflow can corrupt your output, and one use-after-free can crash your program.

## Ownership: Rust's Core Innovation

The most distinctive feature of Rust is its **ownership system**. Every value has exactly one owner, and when that owner goes out of scope, the value is dropped.

### The Rules

```text
1. Each value has exactly one owner
2. When the owner goes out of scope, the value is dropped
3. Ownership can be transferred (moved) or borrowed
```

### How It Works in Practice

Consider our LZ77 compressor:

```rust,ignore
// From src/compress/lz77.rs
pub fn compress(&mut self, data: &[u8]) -> Vec<Token> {
    let mut tokens = Vec::with_capacity(data.len());
    self.compress_into(data, &mut tokens);
    tokens  // Ownership of tokens is returned to caller
}
```

Let's trace what happens:

1. `Vec::with_capacity` allocates memory and creates a `Vec<Token>`. The `tokens` variable **owns** this memory.
2. We pass `&mut tokens` to `compress_into`. This is a **mutable borrow**—we lend the vector temporarily.
3. We return `tokens`. Ownership **moves** to the caller. No copy, no garbage collection—just a pointer handoff.

If we tried to use `tokens` after returning it, the compiler would reject our code:

```rust,ignore
// This won't compile!
let tokens = compressor.compress(data);
let oops = tokens;     // Ownership moves to 'oops'
println!("{:?}", tokens);  // Error: tokens was moved
```

### Borrowing: Lending Without Giving Up Ownership

Often we want to use a value without taking ownership. That's what **references** are for:

```rust,ignore
// From src/compress/lz77.rs
pub fn compress_into(&mut self, data: &[u8], tokens: &mut Vec<Token>) {
    // data is borrowed immutably (&[u8])
    // tokens is borrowed mutably (&mut Vec<Token>)
    // self is borrowed mutably (&mut self)
}
```

The reference types tell us:

- `&[u8]` — immutable borrow of a slice. We can read but not modify.
- `&mut Vec<Token>` — mutable borrow. We can modify the vector.
- `&mut self` — mutable borrow of the compressor itself.

**The borrowing rules:**

```text
At any given time, you can have EITHER:
  - One mutable reference (&mut T), OR
  - Any number of immutable references (&T)

But never both simultaneously.
```

This prevents data races at compile time. No mutexes are needed for single-threaded code.

## Enums: More Than Just Constants

Rust enums are **algebraic data types**—each variant can hold different data.

### Modeling LZ77 Tokens

An LZ77 token is either a literal byte or a back-reference. In C, you might use a struct with a tag field:

```c
// C approach: union with tag
struct Token {
    enum { LITERAL, MATCH } type;
    union {
        uint8_t literal;
        struct { uint16_t length, distance; } match;
    };
};
```

In Rust, the enum itself carries the data:

```rust,ignore
// From src/compress/lz77.rs
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
```

The `#[derive(...)]` attribute automatically implements common traits—`Debug` for printing, `Clone` and `Copy` for duplication, `PartialEq` and `Eq` for comparison.

### Pattern Matching: Exhaustive and Safe

To use an enum, you **match** on it:

```rust,ignore
// From src/compress/lz77.rs (test)
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
```

The compiler **requires** you to handle every variant. If we added a new `Token::Run` variant, every `match` statement would fail to compile until updated. This catches bugs that would be runtime errors in other languages.

### The Option Type: Null Done Right

Rust has no null pointers. Instead, optional values use the `Option` enum:

```rust,ignore
// Part of the standard library
enum Option<T> {
    Some(T),
    None,
}
```

From our LZ77 implementation:

```rust,ignore
// From src/compress/lz77.rs
fn find_best_match(&self, data: &[u8], pos: usize, chain_limit: usize)
    -> Option<(usize, usize)>  // Returns Some((length, distance)) or None
{
    // ...
    if best_length >= MIN_MATCH_LENGTH {
        Some((best_length, best_distance))
    } else {
        None
    }
}
```

To use the result, you must handle both cases:

```rust,ignore
if let Some((length, distance)) = best_match {
    // We have a match—use it
    tokens.push(Token::Match { length, distance });
} else {
    // No match—emit literal
    tokens.push(Token::Literal(data[pos]));
}
```

The compiler won't let you forget to handle `None`. No null pointer exceptions, ever.

## Error Handling: Result Over Exceptions

Rust doesn't have exceptions. Instead, fallible operations return `Result`:

```rust,ignore
// Part of the standard library
enum Result<T, E> {
    Ok(T),   // Success, carrying a value of type T
    Err(E),  // Failure, carrying an error of type E
}
```

### Defining Error Types

We define our own error type:

```rust,ignore
// From src/error.rs
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Invalid image dimensions (zero width or height).
    InvalidDimensions { width: u32, height: u32 },
    /// Pixel data length doesn't match expected size.
    InvalidDataLength { expected: usize, actual: usize },
    /// Invalid quality parameter (must be 1-100 for JPEG).
    InvalidQuality(u8),
    /// Invalid compression level parameter (must be 1-9 for PNG/zlib).
    InvalidCompressionLevel(u8),
    // ... more variants
}
```

Each variant carries exactly the data needed to describe the error. No stringly-typed error messages—structured data we can match on.

### The Type Alias Pattern

We define a type alias for convenience:

```rust,ignore
// From src/error.rs
pub type Result<T> = std::result::Result<T, Error>;
```

Now `Result<Vec<u8>>` means "either a `Vec<u8>` or an `Error`".

### Using Results in Practice

```rust,ignore
// From src/png/mod.rs
pub fn encode(data: &[u8], width: u32, height: u32, color_type: ColorType)
    -> Result<Vec<u8>>
{
    // Validate dimensions
    if width == 0 || height == 0 {
        return Err(Error::InvalidDimensions { width, height });
    }

    // Validate data length
    let expected_len = width as usize * height as usize * bpp;
    if data.len() != expected_len {
        return Err(Error::InvalidDataLength {
            expected: expected_len,
            actual: data.len(),
        });
    }

    // ... encoding logic ...
    Ok(output)  // Success!
}
```

The **`?` operator** propagates errors automatically:

```rust,ignore
// This:
let png = encode(&pixels, width, height, color_type)?;

// Is equivalent to:
let png = match encode(&pixels, width, height, color_type) {
    Ok(data) => data,
    Err(e) => return Err(e),
};
```

Errors are explicit in the type signature, propagated with minimal syntax, and impossible to ignore.

## Structs and Methods

Rust structs are plain data containers. Methods are defined in separate `impl` blocks.

### Defining a Struct

```rust,ignore
// From src/compress/lz77.rs
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
```

### Adding Methods

```rust,ignore
impl Lz77Compressor {
    /// Create a new LZ77 compressor.
    pub fn new(level: u8) -> Self {
        let level = level.clamp(1, 9);

        let (max_chain_length, lazy_matching) = match level {
            1 => (4, false),
            2 => (8, false),
            // ...
            9 => (4096, false),
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
        self.compress_into(data, &mut tokens);
        tokens
    }
}
```

Note the naming conventions:

- `new()` — constructor (by convention, not a language keyword)
- `Self` — alias for the type being implemented
- `&mut self` — method takes mutable borrow of the instance

### The Builder Pattern

For complex configuration, we use structs with sensible defaults:

```rust,ignore
// From src/png/mod.rs
#[derive(Debug, Clone)]
pub struct PngOptions {
    pub compression_level: u8,
    pub filter_strategy: FilterStrategy,
    pub optimize_alpha: bool,
    pub reduce_color_type: bool,
    // ...
}

impl Default for PngOptions {
    fn default() -> Self {
        Self {
            compression_level: 2,
            filter_strategy: FilterStrategy::AdaptiveFast,
            optimize_alpha: false,
            reduce_color_type: false,
            // ...
        }
    }
}
```

Users can create options with defaults and override what they need:

```rust,ignore
let opts = PngOptions {
    compression_level: 9,
    ..PngOptions::default()  // Use defaults for everything else
};
```

## Traits: Shared Behavior

Traits define shared behavior across types—similar to interfaces in other languages.

### Implementing Standard Traits

The `Display` trait controls how a type is formatted with `{}`:

```rust,ignore
// From src/error.rs
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidDimensions { width, height } => {
                write!(f, "Invalid image dimensions: {width}x{height}")
            }
            Error::InvalidDataLength { expected, actual } => {
                write!(f, "Invalid pixel data length: expected {expected} bytes, got {actual}")
            }
            // ... handle all variants
        }
    }
}

impl std::error::Error for Error {}
```

Now our error type works with Rust's standard error handling ecosystem.

### The Default Trait

`Default` provides a sensible default value:

```rust,ignore
// From src/compress/lz77.rs
impl Default for Lz77Compressor {
    fn default() -> Self {
        Self::new(6)  // Level 6 is a good balance
    }
}
```

This enables:

```rust,ignore
let compressor = Lz77Compressor::default();
// or in struct initialization:
let opts = PngOptions::default();
```

## Iterators: Functional Data Processing

Rust's iterators are **zero-cost abstractions**—they compile to the same machine code as hand-written loops, but are more expressive and composable.

### Iterator Methods in Action

From our co-occurrence matrix construction:

```rust,ignore
// From src/png/mod.rs
fn weighted_edges(matrix: &[Vec<u32>]) -> Vec<(usize, usize)> {
    let mut edges: Vec<((usize, usize), u32)> = Vec::new();
    for (i, row) in matrix.iter().enumerate() {
        for (j, &weight) in row.iter().enumerate().take(i) {
            if weight > 0 {
                edges.push(((j, i), weight));
            }
        }
    }
    edges.sort_by(|(_, w1), (_, w2)| w2.cmp(w1));
    edges.into_iter().map(|(e, _)| e).collect()
}
```

Key iterator methods:

- `.iter()` — create an iterator over references
- `.enumerate()` — add indices: `(0, first), (1, second), ...`
- `.take(n)` — limit to first n elements
- `.map(f)` — transform each element
- `.collect()` — gather into a collection

### Chaining Iterators

Iterators compose without overhead:

```rust,ignore
// From src/png/mod.rs
let new_indexed: Vec<u8> = indexed
    .iter()
    .map(|&b| byte_map[b as usize])
    .collect();
```

This reads as: "for each byte in `indexed`, look up its new value in `byte_map`, and collect the results into a new vector."

## Modules and Visibility

Rust code is organized into modules. By default, everything is private.

### Module Structure

```rust,ignore
// From src/lib.rs
pub mod bits;        // Public module
pub mod color;
pub mod compress;
pub mod error;
pub mod jpeg;
pub mod png;

#[cfg(feature = "simd")]
pub mod simd;        // Only compiled when "simd" feature is enabled

pub use color::ColorType;         // Re-export for convenience
pub use error::{Error, Result};
```

### Visibility Modifiers

```text
pub           — visible everywhere
pub(crate)    — visible within this crate only
pub(super)    — visible to parent module
(nothing)     — private to current module
```

Example:

```rust,ignore
// From src/color.rs
impl ColorType {
    // Public: anyone can call this
    #[inline]
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            ColorType::Gray => 1,
            ColorType::GrayAlpha => 2,
            ColorType::Rgb => 3,
            ColorType::Rgba => 4,
        }
    }

    // Crate-internal: only our code can use this
    #[inline]
    pub(crate) const fn png_color_type(self) -> u8 {
        // ...
    }
}
```

## Conditional Compilation

Rust supports conditional compilation with the `#[cfg]` attribute.

### Feature Flags

```rust,ignore
// From src/lib.rs
#[cfg(feature = "simd")]
pub mod simd;

#[cfg(feature = "wasm")]
pub mod wasm;
```

### Platform-Specific Code

SIMD implementations differ by architecture:

```rust,ignore
// From src/simd/mod.rs
#[inline]
pub fn adler32(data: &[u8]) -> u32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { x86_64::adler32_avx2(data) };
        }
        if is_x86_feature_detected!("ssse3") {
            return unsafe { x86_64::adler32_ssse3(data) };
        }
        fallback::adler32(data)
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on aarch64
        unsafe { aarch64::adler32_neon(data) }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    fallback::adler32(data)
}
```

The compiler only includes code for the target platform. No runtime dispatch overhead for unsupported architectures.

### Runtime Feature Detection

For x86, we detect CPU features at runtime:

```rust,ignore
if is_x86_feature_detected!("avx2") {
    // Use AVX2 implementation
}
```

This macro expands to efficient CPU feature detection, cached after first check.

## Unsafe Code: Controlled Escape Hatch

Rust's safety guarantees come from the compiler. Sometimes we need to do things the compiler can't verify—that's what `unsafe` is for.

### When We Use Unsafe

1. **SIMD intrinsics** — calling CPU-specific instructions
2. **FFI** — interfacing with C code
3. **Raw pointer manipulation** — rarely, for performance

```rust,ignore
// From src/simd/x86_64.rs
#[target_feature(enable = "sse2")]
pub unsafe fn score_filter_sse2(filtered: &[u8]) -> u64 {
    let mut sum = _mm_setzero_si128();

    while remaining >= 16 {
        let chunk = _mm_loadu_si128(ptr as *const __m128i);
        let sad = _mm_sad_epu8(chunk, _mm_setzero_si128());
        sum = _mm_add_epi64(sum, sad);
        // ...
    }
    // ...
}
```

The `unsafe` keyword marks a **trust boundary**:

- Inside `unsafe`, we promise to uphold invariants the compiler can't check
- Outside `unsafe`, the compiler guarantees safety
- Most code never needs `unsafe`

### Library-Level Safety Policy

We declare our safety policy at the crate level:

```rust,ignore
// From src/lib.rs
// Only allow unsafe when SIMD or WASM feature is enabled
#![cfg_attr(not(any(feature = "simd", feature = "wasm")), forbid(unsafe_code))]
```

This means the core compression algorithms contain zero unsafe code. Only the optional SIMD acceleration uses it, and only in carefully audited functions.

## Inline Functions and Performance Hints

Rust provides attributes to guide optimization:

```rust,ignore
// From src/compress/lz77.rs
#[inline(always)]
fn hash4(data: &[u8], pos: usize) -> usize {
    // Hot path—always inline for performance
}

#[inline]
pub fn literal_cost(&self, byte: u8) -> f32 {
    // Suggest inlining, but let compiler decide
}
```

The `#[inline]` attribute is a hint. `#[inline(always)]` is stronger—use sparingly on genuinely hot code.

## Constants and Compile-Time Computation

Rust can evaluate code at compile time:

```rust,ignore
// From src/jpeg/dct.rs
/// Precomputed cosine values for IDCT.
const COS_TABLE: [[f32; 8]; 8] = precompute_cos_table();

/// Precompute the cosine table at compile time.
const fn precompute_cos_table() -> [[f32; 8]; 8] {
    let mut table = [[0.0f32; 8]; 8];
    let mut i = 0;
    while i < 8 {
        let mut j = 0;
        while j < 8 {
            let angle = ((2 * i + 1) * j) as f32 * PI / 16.0;
            table[i][j] = cos_approx(angle);
            j += 1;
        }
        i += 1;
    }
    table
}
```

The `const fn` keyword means this function can run at compile time. The resulting table is baked into the binary—zero runtime cost.

## Slices: Flexible Views into Data

Slices (`&[T]`) are views into contiguous sequences. They're the primary way to work with arrays and vectors.

```rust,ignore
// From src/bits.rs
pub fn write_bytes(&mut self, bytes: &[u8]) {
    for &byte in bytes {
        self.write_byte(byte);
    }
}
```

Slices carry their length, preventing buffer overflows:

```rust,ignore
let data: Vec<u8> = vec![1, 2, 3, 4, 5];
let slice: &[u8] = &data[1..4];  // [2, 3, 4]
println!("{}", slice.len());     // 3
```

The `chunks` and `chunks_exact` methods are particularly useful for image processing:

```rust,ignore
// Process 4 bytes at a time (RGBA pixels)
for chunk in data.chunks_exact(4) {
    let r = chunk[0];
    let g = chunk[1];
    let b = chunk[2];
    let a = chunk[3];
    // ...
}
```

## Lifetimes: When Borrowing Gets Complex

Most of the time, Rust infers lifetimes. Occasionally, we need to be explicit:

```rust,ignore
// From src/png/mod.rs
fn maybe_optimize_alpha<'a>(
    data: &'a [u8],
    color_type: ColorType,
    optimize_alpha: bool,
) -> std::borrow::Cow<'a, [u8]>
```

The `'a` is a **lifetime parameter**. It says: "the returned `Cow` borrows from `data` and can't outlive it."

`Cow` (Copy on Write) is an enum that's either:

- `Borrowed(&[u8])` — no allocation, just a reference
- `Owned(Vec<u8>)` — we made a copy and own it

This pattern lets us avoid allocation when no modification is needed.

## Putting It All Together

Here's a complete example showing many Rust features working together:

```rust,ignore
// From src/color.rs
/// Convert RGB to YCbCr color space (used by JPEG).
#[inline]
pub fn rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    // Type conversion: u8 -> i32 for arithmetic
    let r = r as i32;
    let g = g as i32;
    let b = b as i32;

    // Fixed-point arithmetic (scaled by 256)
    // +128 for rounding before right shift
    let y = (77 * r + 150 * g + 29 * b + 128) >> 8;
    let cb = ((-43 * r - 85 * g + 128 * b + 128) >> 8) + 128;
    let cr = ((128 * r - 107 * g - 21 * b + 128) >> 8) + 128;

    // Method chaining: clamp then cast
    (
        y.clamp(0, 255) as u8,
        cb.clamp(0, 255) as u8,
        cr.clamp(0, 255) as u8,
    )
}
```

This function demonstrates:

- **Type annotations** where needed (`i32`, `u8`)
- **Explicit type casts** (`as i32`, `as u8`)
- **Method chaining** (`.clamp(0, 255) as u8`)
- **Tuple return types** (`(u8, u8, u8)`)
- **Inline hint** for performance-critical code
- **Doc comments** (`///`) that become API documentation

## Common Patterns in This Codebase

### 1. The `into` Pattern for Buffer Reuse

We provide both allocating and non-allocating versions of functions:

```rust,ignore
// Allocates a new Vec
pub fn compress(&mut self, data: &[u8]) -> Vec<Token> {
    let mut tokens = Vec::with_capacity(data.len());
    self.compress_into(data, &mut tokens);
    tokens
}

// Reuses caller-provided buffer
pub fn compress_into(&mut self, data: &[u8], tokens: &mut Vec<Token>) {
    tokens.clear();
    // ... populate tokens
}
```

This lets callers avoid repeated allocations in hot loops.

### 2. Presets with Override

```rust,ignore
impl PngOptions {
    pub fn fast() -> Self { /* low compression, high speed */ }
    pub fn balanced() -> Self { /* good tradeoff */ }
    pub fn max() -> Self { /* maximum compression */ }

    pub fn from_preset(preset: u8) -> Self {
        match preset {
            0 => Self::fast(),
            2 => Self::max(),
            _ => Self::balanced(),
        }
    }
}
```

Users get sensible defaults with easy customization.

### 3. Exhaustive Matching with Early Return

```rust,ignore
// Handle error cases first
if width == 0 || height == 0 {
    return Err(Error::InvalidDimensions { width, height });
}
if data.len() != expected_len {
    return Err(Error::InvalidDataLength { expected, actual });
}

// Happy path continues here
```

This pattern keeps the main logic unindented and easy to follow.

## Key Takeaways

1. **Ownership is about memory management**—Rust tracks who's responsible for freeing memory, eliminating leaks and use-after-free bugs.

2. **Borrowing enables sharing**—References let multiple parts of code use data without transferring ownership.

3. **Enums are powerful**—They carry data and the compiler ensures exhaustive handling.

4. **Result over exceptions**—Errors are explicit in types, propagated with `?`, impossible to ignore.

5. **Traits define interfaces**—Common behavior is shared through traits like `Default`, `Display`, and `Error`.

6. **Zero-cost abstractions**—Iterators, generics, and other high-level features compile to efficient machine code.

7. **Unsafe is contained**—When we need to bypass the compiler's checks, `unsafe` clearly marks the boundary.

## Next Steps

Continue to [Huffman Coding](./huffman-coding.md) to learn the foundational entropy coding technique used in both PNG and JPEG compression.

This introduction covers the Rust features most relevant to understanding pixo. To dive deeper into Rust itself:

- **The Rust Book**: <https://doc.rust-lang.org/book/>
- **Rust by Example**: <https://doc.rust-lang.org/rust-by-example/>
- **The Rustonomicon** (for unsafe): <https://doc.rust-lang.org/nomicon/>

Within this codebase, explore:

- `src/error.rs` — Simple error handling example
- `src/color.rs` — Clean, focused module
- `src/compress/lz77.rs` — Complex algorithm with many Rust idioms
- `src/simd/mod.rs` — Conditional compilation and unsafe code

---

## References

- [The Rust Programming Language Book](https://doc.rust-lang.org/book/)
- [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- [Rust Performance Book](https://nnethercote.github.io/perf-book/)
