# Huffman Coding

Huffman coding is one of the most elegant algorithms in computer science. Invented by David Huffman in 1952 as a term paper at MIT, it provides an optimal way to assign variable-length codes to symbols based on their frequency of occurrence.

## The Core Insight

Consider Morse code: the most common letter in English, 'E', is encoded as a single dot (·), while the rare letter 'Q' is encoded as dash-dash-dot-dash (– – · –). This is efficient! Common symbols get short codes, while rare symbols get long codes.

Huffman coding takes this principle to its mathematical optimum.

## The Problem

Given a set of **symbols** (the things you want to encode, like bytes) and their **frequencies** (how often they appear), find a binary encoding that minimizes the total number of bits needed.

### Example: A Simple Message

Suppose we want to encode a message containing only four symbols:

| Symbol | Frequency | Probability |
| ------ | --------- | ----------- |
| A      | 50        | 0.50        |
| B      | 25        | 0.25        |
| C      | 15        | 0.15        |
| D      | 10        | 0.10        |

**Naive approach**: Use fixed 2-bit codes (since we have 4 symbols):

- A = 00, B = 01, C = 10, D = 11
- Total bits: 100 symbols × 2 bits = **200 bits**

**Optimal approach**: Use variable-length codes:

- A = 0 (1 bit)
- B = 10 (2 bits)
- C = 110 (3 bits)
- D = 111 (3 bits)
- Total bits: 50×1 + 25×2 + 15×3 + 10×3 = 50 + 50 + 45 + 30 = **175 bits**

That's a **12.5% savings** just by using smarter codes!

## The Prefix-Free Property

There's a critical constraint: the codes must be **prefix-free** (no code is a prefix of another code). This guarantees that decoding is unambiguous.

Why? Consider what happens without this property:

```text
Suppose: A = 0, B = 01, C = 1

Encoded message: 0 01 1 → 0011

Now try to decode "0011":
- Is it "0 011" = A B C?  (if B=011 existed)
- Is it "00 11" = A A C C? (if A=00 existed)
- Is it "0 0 1 1" = A A C C?
```

The message becomes ambiguous! Prefix-free codes guarantee unique decodability.

### The Huffman Tree

Prefix-free codes can be visualized as a binary tree:

- Each **leaf** is a symbol
- Each **path** from root to leaf defines the code (left=0, right=1)
- No symbol can be an ancestor of another (hence prefix-free)

```text
        (root)
        /    \
       0      1
      /        \
    [A]        ( )
              /   \
             0     1
            /       \
          [B]       ( )
                   /   \
                  0     1
                 /       \
               [C]       [D]

Codes:
  A = 0
  B = 10
  C = 110
  D = 111
```

## The Huffman Algorithm

Building the optimal tree is surprisingly simple:

### Step-by-Step Process

1. **Create a leaf node for each symbol** with its frequency
2. **Insert all nodes into a priority queue** (min-heap by frequency)
3. **While more than one node remains**:
   - Remove the two nodes with lowest frequency
   - Create a new internal node with these as children
   - The new node's frequency = sum of children's frequencies
   - Insert the new node back into the queue
4. **The remaining node is the root** of the Huffman tree

### Worked Example

Starting symbols: A(50), B(25), C(15), D(10)

```text
Step 1: Initial queue
  [D:10] [C:15] [B:25] [A:50]

Step 2: Combine D(10) and C(15)
  Create node N1(25) with children D and C

  Queue: [B:25] [N1:25] [A:50]

         N1(25)
         /    \
      [D:10] [C:15]

Step 3: Combine B(25) and N1(25)
  Create node N2(50) with children B and N1

  Queue: [A:50] [N2:50]

         N2(50)
         /    \
      [B:25]  N1(25)
              /    \
           [D:10] [C:15]

Step 4: Combine A(50) and N2(50)
  Create root node N3(100)

           N3(100)
           /     \
       [A:50]    N2(50)
                 /    \
              [B:25]  N1(25)
                      /    \
                   [D:10] [C:15]
```

Reading the codes by traversing left (0) and right (1):

- A: left → **0**
- B: right, left → **10**
- D: right, right, left → **110**
- C: right, right, right → **111**

## Canonical Huffman Codes

In practice, we don't store the tree structure. Instead, we use **canonical Huffman codes**, which can be reconstructed from just the code lengths.

### The Canonical Property

Canonical codes follow two rules:

1. Shorter codes come before (are numerically less than) longer codes
2. Codes of the same length are assigned in symbol order

### Reconstruction Algorithm

Given just the code lengths, we can regenerate the codes:

```text
Lengths: A=1, B=2, C=3, D=3

Step 1: Count codes of each length
  Length 1: 1 code (A)
  Length 2: 1 code (B)
  Length 3: 2 codes (C, D)

Step 2: Calculate first code of each length
  Length 1: first code = 0
  Length 2: first code = (0 + 1) << 1 = 2 (binary: 10)
  Length 3: first code = (2 + 1) << 1 = 6 (binary: 110)

Step 3: Assign codes in order
  A (length 1): 0     (binary: 0)
  B (length 2): 2     (binary: 10)
  C (length 3): 6     (binary: 110)
  D (length 3): 7     (binary: 111)
```

This is exactly what our implementation does in `generate_canonical_codes`:

```rust,ignore
// From src/compress/huffman.rs
pub fn generate_canonical_codes(lengths: &[u8]) -> Vec<HuffmanCode> {
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
```

## Why Huffman is Optimal

Huffman coding is **optimal** for symbol-by-symbol encoding. This means no other prefix-free code can achieve a shorter expected length.

### Proof Sketch

1. **Sibling Property**: In an optimal tree, the two symbols with lowest frequency must be siblings at the deepest level. (If not, swapping them with deeper nodes would reduce total length.)

2. **Huffman constructs exactly this**: By always combining the two lowest-frequency nodes, Huffman guarantees the sibling property at every level.

3. **Mathematical bound**: The expected code length L satisfies:
   ```text
   H(X) ≤ L < H(X) + 1
   ```
   Where H(X) is the entropy (theoretical minimum).

## DEFLATE's Fixed Huffman Codes

For efficiency, DEFLATE (used in PNG) defines pre-computed "fixed" Huffman codes that don't require transmitting the tree:

```text
Literal/Length codes (0-287):
  0-143:   8-bit codes (00110000 to 10111111)
  144-255: 9-bit codes (110010000 to 111111111)
  256-279: 7-bit codes (0000000 to 0010111)
  280-287: 8-bit codes (11000000 to 11000111)

Distance codes (0-29):
  All:     5-bit codes (00000 to 11101)
```

Our implementation of fixed codes:

```rust,ignore
// From src/compress/huffman.rs
pub fn fixed_literal_codes() -> Vec<HuffmanCode> {
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
}
```

## JPEG's Huffman Tables

JPEG uses different Huffman tables for:

- **DC coefficients** (the average brightness of each 8×8 block)
- **AC coefficients** (the details within each block)
- **Luminance** (brightness) and **Chrominance** (color)

This specialization allows each table to be optimized for its specific data distribution.

## Code Length Limiting

Sometimes the natural Huffman tree produces codes that are too long. DEFLATE limits codes to 15 bits maximum.

The **Kraft inequality** tells us this is always possible:

```text
For a valid prefix-free code: Σ 2^(-lᵢ) ≤ 1
```

If the sum equals exactly 1, the code is "complete" (no wasted code space).

Our implementation handles length limiting:

```rust,ignore
// From src/compress/huffman.rs - simplified
fn limit_code_lengths(lengths: &mut [u8], max_length: usize) {
    let max_length = max_length as u8;

    // Truncate long codes
    for length in lengths.iter_mut() {
        if *length > max_length {
            *length = max_length;
        }
    }

    // Redistribute using Kraft inequality to maintain validity
    // ...
}
```

## Bit Order Considerations

An important implementation detail: DEFLATE and JPEG use different bit ordering!

- **DEFLATE**: Bits are packed **LSB first** (least significant bit first)
- **JPEG**: Bits are packed **MSB first** (most significant bit first)

Our library handles both:

```rust,ignore
// For DEFLATE (PNG)
pub struct BitWriter { /* LSB first */ }

// For JPEG
pub struct BitWriterMsb { /* MSB first */ }
```

## Practical Performance

Huffman coding typically achieves compression ratios of:

- **Text**: 40-60% of original (1.7x - 2.5x compression)
- **Random data**: ~100% (no compression possible)
- **Highly redundant data**: Can approach theoretical entropy

When combined with LZ77 (as in DEFLATE), compression improves dramatically because LZ77 creates highly skewed frequency distributions that Huffman codes efficiently.

## Common Pitfalls

1. **Assuming fixed codes**: Different data needs different codes for optimal compression

2. **Forgetting the tree overhead**: For small files, transmitting the Huffman table may cost more than it saves

3. **Ignoring bit ordering**: Mixing up MSB/LSB will produce garbage

4. **Not handling single-symbol case**: A single symbol still needs at least 1 bit

## Summary

Huffman coding provides:

- **Optimal** symbol-by-symbol compression
- **Prefix-free** codes for unambiguous decoding
- **Canonical form** for compact table representation
- Foundation for modern compression (DEFLATE, JPEG, etc.)

## Next Steps

Continue to [LZ77 Compression](./lz77-compression.md) to learn how dictionary compression finds and eliminates repeated patterns.

---

## References

- Huffman, D.A. (1952). "A Method for the Construction of Minimum-Redundancy Codes"
- [RFC 1951 Section 3.2.2 - Use of Huffman coding](https://www.rfc-editor.org/rfc/rfc1951#section-3.2.2)
- See implementation: `src/compress/huffman.rs`
