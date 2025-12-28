# Discrete Cosine Transform (DCT)

The Discrete Cosine Transform is the mathematical heart of JPEG compression. It converts image data from the **spatial domain** (pixel values) to the **frequency domain** (patterns of variation), enabling efficient compression by concentrating image energy into fewer coefficients.

## The Key Insight

Natural images are dominated by **smooth variations** (gradual changes in brightness and color) rather than **rapid oscillations** (checkerboard patterns). The DCT separates these:

```text
Smooth gradient (low frequency):     Checkerboard (high frequency):
████████████████████████████        █ █ █ █ █ █ █ █
███████████████████████████░        █ █ █ █ █ █ █ █
██████████████████████████░░        █ █ █ █ █ █ █ █
█████████████████████████░░░        █ █ █ █ █ █ █ █
████████████████████████░░░░        █ █ █ █ █ █ █ █
███████████████████████░░░░░        █ █ █ █ █ █ █ █
██████████████████████░░░░░░        █ █ █ █ █ █ █ █
█████████████████████░░░░░░░        █ █ █ █ █ █ █ █
```

Photographs have lots of the left, very little of the right. The DCT exploits this!

## Understanding Frequency

In image processing, "frequency" refers to how rapidly pixel values change:

| Frequency | Visual Pattern            | Example                  |
| --------- | ------------------------- | ------------------------ |
| Zero (DC) | Constant value everywhere | Solid color fill         |
| Low       | Slow, smooth changes      | Sky gradient             |
| Medium    | Moderate detail           | Fabric texture           |
| High      | Rapid, sharp changes      | Text edges, hair strands |

## The 1D DCT

Before tackling 2D images, let's understand the 1D case. Given N samples, the DCT produces N frequency coefficients.

### The Formula

For N samples x\[0\] to x\[N-1\], the DCT coefficient X\[k\] is:

```text
X[k] = α(k) × Σ(n=0 to N-1) x[n] × cos(π(2n+1)k / 2N)

where:
  α(0) = 1/√N
  α(k) = √(2/N) for k > 0
```

This looks complex, but the intuition is simple: **we're measuring how much each frequency pattern appears in the data**.

### Worked Example

Consider 8 pixel values: `[100, 110, 120, 130, 140, 130, 120, 110]`

This is roughly a smooth curve peaking in the middle. Let's compute the first few DCT coefficients:

**DC coefficient (k=0)**: The average, scaled.

```text
X[0] = (1/√8) × Σ x[n] × cos(0)
     = (1/√8) × (100+110+120+130+140+130+120+110) × 1
     = (1/2.83) × 960 = 339.4
```

**First AC coefficient (k=1)**: Measures one half-cycle oscillation.

```text
X[1] = √(2/8) × Σ x[n] × cos(π(2n+1)/16)
     ≈ small value (the data doesn't oscillate once)
```

**Higher coefficients**: Measure higher frequencies — mostly near zero for this smooth data.

### Basis Functions

Each DCT coefficient corresponds to a **basis function** — a specific cosine wave:

```text
k=0:  ████████████████  (constant - DC)
k=1:  ██████░░░░░░░░░░  (half cycle)
k=2:  ████░░░░████░░░░  (full cycle)
k=3:  ██░░░░██░░░░██░░  (1.5 cycles)
k=4:  ██░░██░░██░░██░░  (2 cycles)
...
```

The DCT tells us: "Your signal is made of 339.4 units of DC, plus a little of k=1, etc."

## The 2D DCT

Images are 2D, so JPEG uses a 2D DCT on 8×8 blocks.

### Separability

A key property: the 2D DCT can be computed as two 1D DCTs:

1. Apply 1D DCT to each row
2. Apply 1D DCT to each column of the result

This reduces computation from O(N⁴) to O(N³).

```rust,ignore
// From src/jpeg/dct.rs
pub fn dct_2d(block: &[f32; 64]) -> [f32; 64] {
    let mut temp = [0.0f32; 64];
    let mut result = [0.0f32; 64];

    // 1D DCT on rows
    for row in 0..8 {
        let row_start = row * 8;
        dct_1d(
            &block[row_start..row_start + 8],
            &mut temp[row_start..row_start + 8],
        );
    }

    // 1D DCT on columns
    for col in 0..8 {
        let mut col_in = [0.0f32; 8];
        let mut col_out = [0.0f32; 8];

        for row in 0..8 {
            col_in[row] = temp[row * 8 + col];
        }

        dct_1d(&col_in, &mut col_out);

        for row in 0..8 {
            result[row * 8 + col] = col_out[row];
        }
    }

    result
}
```

### The 2D Basis Functions

The 64 basis functions for an 8×8 block form a complete set of patterns:

```text
Basis functions (simplified visualization):

(0,0) DC      (0,1) Horiz   (0,2) Horiz   ...  (0,7) Horiz
   ░░░░░░░░      ██░░░░░░      ██  ██        ██ ██ ██ ██
   ░░░░░░░░      ██░░░░░░      ██  ██        ██ ██ ██ ██
   ░░░░░░░░      ██░░░░░░      ██  ██        ██ ██ ██ ██
   ░░░░░░░░      ██░░░░░░      ██  ██        ██ ██ ██ ██
   constant    1 horiz     2 horiz        8 horiz

(1,0) Vert   (1,1) Diag    (2,2)          (7,7) Checker
   ████████      ██░░░░░░      ██░░██░░       █ █ █ █
   ████████      ██░░░░░░      ░░██░░██       █ █ █ █
   ░░░░░░░░      ░░░░██░░      ██░░██░░       █ █ █ █
   ░░░░░░░░      ░░░░░░██      ░░██░░██       █ █ █ █
  1 vert      diagonal     medium freq    highest freq
```

The coefficient at position (u,v) tells us how much of that basis pattern is in the image block.

## Energy Compaction

The magic of DCT: for typical images, **most energy concentrates in the top-left corner** (low frequencies):

```text
DCT of a typical photo block:

     Low freq ───────────────────▶ High freq
   ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
   │ 312 │ -42 │  18 │  -7 │   3 │   0 │   0 │   0 │  Low
   │ -23 │  14 │  -8 │   4 │  -1 │   0 │   0 │   0 │   │
   │  11 │  -6 │   3 │  -1 │   0 │   0 │   0 │   0 │   │
   │  -5 │   3 │  -1 │   0 │   0 │   0 │   0 │   0 │   │
   │   2 │  -1 │   0 │   0 │   0 │   0 │   0 │   0 │   │
   │  -1 │   0 │   0 │   0 │   0 │   0 │   0 │   0 │   │
   │   0 │   0 │   0 │   0 │   0 │   0 │   0 │   0 │   │
   │   0 │   0 │   0 │   0 │   0 │   0 │   0 │   0 │   ▼
   └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘ High

The vast majority of coefficients are near zero!
```

This is why JPEG works: we can store mainly the top-left values and discard the rest.

## Precomputed Cosine Table

Computing cosines repeatedly is expensive. We precompute them:

```rust,ignore
// From src/jpeg/dct.rs
const COS_TABLE: [[f32; 8]; 8] = precompute_cos_table();

const fn precompute_cos_table() -> [[f32; 8]; 8] {
    let mut table = [[0.0f32; 8]; 8];
    let mut i = 0;
    while i < 8 {
        let mut j = 0;
        while j < 8 {
            // cos((2*i + 1) * j * PI / 16)
            let angle = ((2 * i + 1) * j) as f32 * PI / 16.0;
            table[i][j] = cos_approx(angle);
            j += 1;
        }
        i += 1;
    }
    table
}
```

The table entry `COS_TABLE[n][k]` = cos((2n+1)kπ/16), which appears in the DCT formula.

## The 1D DCT Implementation

```rust,ignore
// From src/jpeg/dct.rs
fn dct_1d(input: &[f32], output: &mut [f32]) {
    for k in 0..8 {
        let mut sum = 0.0f32;
        for n in 0..8 {
            sum += input[n] * COS_TABLE[n][k];
        }
        output[k] = 0.5 * ALPHA[k] * sum;
    }
}
```

Where ALPHA provides the normalization:

```rust,ignore
const ALPHA: [f32; 8] = [
    0.7071067811865476,  // 1/√2 for k=0
    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  // 1 for k>0
];
```

## The Inverse DCT

Decoding requires the **inverse DCT (IDCT)**. Fortunately, DCT is nearly self-inverse:

```rust,ignore
// From src/jpeg/dct.rs
fn idct_1d(input: &[f32], output: &mut [f32]) {
    for n in 0..8 {
        let mut sum = 0.0f32;
        for k in 0..8 {
            sum += ALPHA[k] * input[k] * COS_TABLE[n][k];
        }
        output[n] = 0.5 * sum;
    }
}
```

The IDCT reconstructs spatial values from frequency coefficients.

## Level Shifting

Before DCT, JPEG subtracts 128 from pixel values (converting 0-255 to -128 to 127). This centers the data around zero, which produces a smaller DC coefficient and better numerical behavior.

```rust,ignore
// From src/jpeg/mod.rs
fn extract_block(...) {
    // ...
    y_block[idx] = pixel as f32 - 128.0;  // Level shift
    // ...
}
```

## Why DCT Instead of DFT?

The DCT is related to the DFT (Discrete Fourier Transform), but better for image compression:

| Property          | DCT                | DFT                 |
| ----------------- | ------------------ | ------------------- |
| Output            | Real numbers       | Complex numbers     |
| Energy compaction | Excellent          | Good                |
| Boundary behavior | Smooth (symmetric) | Wraparound          |
| Computation       | ~N²                | ~N log N (with FFT) |

The DCT's smooth boundary handling is crucial; it implicitly mirrors the data, avoiding discontinuities at block edges.

## Fast DCT Algorithms

The naive O(N²) DCT can be accelerated using algorithms like:

- **Loeffler's algorithm**: 29 multiplications for 8-point DCT
- **AAN algorithm**: Fast integer-based DCT
- **SIMD implementations**: Process multiple values in parallel

Our implementation uses the straightforward approach for clarity, but production JPEG codecs use these optimizations.

## DCT and Compression

After DCT:

1. **DC coefficient** (position 0,0): Average value of the block
2. **Low-frequency AC**: Smooth variations (important, keep)
3. **High-frequency AC**: Fine detail (less important, can discard)

Quantization (the next step) divides by larger values for high frequencies, forcing them to zero.

## Visualizing the DCT

Consider what happens to different image patterns:

**Solid block** (all pixels = 100):

```text
Input: All 100s
DCT: Only DC is non-zero (≈283)
     All AC = 0
```

**Vertical edge**:

```text
Input: Left half = 0, Right half = 200
DCT: Large DC (average ≈ 100)
     Large horizontal AC components
     Vertical AC ≈ 0
```

**Checkerboard**:

```text
Input: Alternating 0 and 255
DCT: DC = 127 (average)
     Position (7,7) is very large
     Other positions ≈ 0
```

## The DC Coefficient's Importance

The DC coefficient at position (0,0) represents the **average brightness** of the block:

```text
DC = (1/8) × Σ all pixels

For a block with pixels averaging 100:
  DC ≈ 100 × scaling_factor
```

The DC coefficient typically contains **80-90% of the block's total energy**. JPEG encodes it with special care (differential encoding between blocks).

## Summary

The Discrete Cosine Transform:

- Converts spatial data to frequency components
- Concentrates energy in low frequencies
- Enables efficient lossy compression
- Is reversible (lossless) before quantization
- Uses separability for efficient 2D computation

The DCT is why JPEG can achieve 10-20x compression while maintaining acceptable visual quality — it reveals that most image information is in smooth variations, not fine detail.

## Next Steps

See [JPEG Quantization](./quantization.md) to learn how quantization discards high-frequency DCT coefficients.

---

## References

- Ahmed, N., Natarajan, T., and Rao, K.R. (1974). "Discrete Cosine Transform"
- [ITU-T T.81 - JPEG Standard, Section A.3.3](https://www.w3.org/Graphics/JPEG/itu-t81.pdf)
- Pennebaker, W.B. and Mitchell, J.L. (1993). "JPEG: Still Image Data Compression Standard"
- See implementation: `src/jpeg/dct.rs`
