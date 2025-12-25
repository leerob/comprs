# JPEG Quantization

Quantization is **the step where JPEG becomes lossy**. It permanently discards information that humans are unlikely to perceive, achieving dramatic file size reductions while maintaining acceptable visual quality.

## The Core Concept

Quantization divides each DCT coefficient by a quantization value and rounds to the nearest integer:

```
Quantized[i] = round(DCT[i] / Q[i])
```

This operation is **irreversible** — the original DCT values cannot be perfectly recovered.

### Simple Example

```
DCT coefficient: 47
Quantization value: 10
Quantized result: round(47/10) = round(4.7) = 5

During decoding:
Reconstructed: 5 × 10 = 50
Error: |47 - 50| = 3
```

The error is the "lossy" part. We've traded precision for a smaller number (5 fits in fewer bits than 47).

## Why Does This Work?

Two key insights enable quantization:

### 1. Human Visual Sensitivity

Our eyes aren't equally sensitive to all frequencies:

- **Low frequencies**: Very sensitive (smooth gradients are important)
- **High frequencies**: Less sensitive (fine detail is often imperceptible)

JPEG quantizes high frequencies more aggressively because we won't notice!

### 2. DCT Energy Distribution

As we saw in [DCT documentation](./dct.md), most image energy concentrates in low-frequency coefficients. High-frequency coefficients are often already small, so aggressive quantization forces them to **zero**.

```
Before quantization:     After quantization:
   47   23   11    5       5    2    1    0
   31   15    7    3       3    1    1    0
   17    8    4    2       2    1    0    0
    9    4    2    1       1    0    0    0

Many zeros = excellent compression!
```

## Standard Quantization Tables

JPEG defines standard quantization tables derived from human visual research:

### Luminance Table

Used for the Y (brightness) channel — humans are very sensitive to brightness variations:

```rust
// From src/jpeg/quantize.rs
const STD_LUMINANCE_TABLE: [u8; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68,109,103, 77,
    24, 35, 55, 64, 81,104,113, 92,
    49, 64, 78, 87,103,121,120,101,
    72, 92, 95, 98,112,100,103, 99,
];
```

Visualized:

```
Position in 8×8 block → quantization value

         Low freq ─────────────────▶ High freq
       ┌────┬────┬────┬────┬────┬────┬────┬────┐
       │ 16 │ 11 │ 10 │ 16 │ 24 │ 40 │ 51 │ 61 │  Low
       │ 12 │ 12 │ 14 │ 19 │ 26 │ 58 │ 60 │ 55 │   ↓
       │ 14 │ 13 │ 16 │ 24 │ 40 │ 57 │ 69 │ 56 │   ↓
       │ 14 │ 17 │ 22 │ 29 │ 51 │ 87 │ 80 │ 62 │   ↓
       │ 18 │ 22 │ 37 │ 56 │ 68 │109 │103 │ 77 │   ↓
       │ 24 │ 35 │ 55 │ 64 │ 81 │104 │113 │ 92 │   ↓
       │ 49 │ 64 │ 78 │ 87 │103 │121 │120 │101 │   ↓
       │ 72 │ 92 │ 95 │ 98 │112 │100 │103 │ 99 │  High
       └────┴────┴────┴────┴────┴────┴────┴────┘

Small values (top-left) → preserve detail
Large values (bottom-right) → aggressive rounding → zeros
```

### Chrominance Table

Used for Cb and Cr (color) channels — humans are less sensitive to color detail:

```rust
const STD_CHROMINANCE_TABLE: [u8; 64] = [
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
];
```

Notice: Large portions are just "99" — we aggressively discard high-frequency color information because humans can't perceive it well.

## Quality Scaling

The "quality" parameter (1-100) scales these tables:

```rust
// From src/jpeg/quantize.rs
let scale = if quality < 50 {
    5000 / quality as u32
} else {
    200 - 2 * quality as u32
};

// Scale and clamp each value
let scaled_value = ((std_table[i] as u32 * scale + 50) / 100)
    .clamp(1, 255);
```

### Scale Factor Examples

| Quality | Scale | Effect                                         |
| ------- | ----- | ---------------------------------------------- |
| 1       | 5000  | Values × 50: everything quantized to near-zero |
| 25      | 200   | Values × 2: heavy quantization                 |
| 50      | 100   | Standard tables unchanged                      |
| 75      | 50    | Values ÷ 2: moderate quantization              |
| 100     | 0     | Values → 1: minimal quantization               |

### Visual Impact

```
Quality 95:               Quality 50:              Quality 10:
┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐
│ Subtle differences │   │ Some artifacts     │   │ Severe blocking    │
│ from original      │   │ around edges       │   │ and color banding  │
│                    │   │                    │   │                    │
│ File: 250 KB       │   │ File: 50 KB        │   │ File: 10 KB        │
└────────────────────┘   └────────────────────┘   └────────────────────┘
```

## The Quantization Process

```rust
// From src/jpeg/quantize.rs
pub fn quantize_block(dct: &[f32; 64], quant_table: &[f32; 64]) -> [i16; 64] {
    let mut result = [0i16; 64];
    for i in 0..64 {
        result[i] = (dct[i] / quant_table[i]).round() as i16;
    }
    result
}
```

### Step-by-Step Example

DCT block (after transformation):

```
  235.7   12.3   -8.5    2.1
  -18.4    7.2   -3.1    0.8
   11.6   -4.8    1.9   -0.3
   -5.2    2.3   -0.7    0.1
```

Quantization table (Q50):

```
   16     11     10     16
   12     12     14     19
   14     13     16     24
   14     17     22     29
```

Quantized result:

```
round(235.7/16) = 15    round(12.3/11) = 1    round(-8.5/10) = -1   round(2.1/16) = 0
round(-18.4/12) = -2    round(7.2/12) = 1     round(-3.1/14) = 0    round(0.8/19) = 0
round(11.6/14) = 1      round(-4.8/13) = 0    round(1.9/16) = 0     round(-0.3/24) = 0
round(-5.2/14) = 0      round(2.3/17) = 0     round(-0.7/22) = 0    round(0.1/29) = 0

Result:
   15    1   -1    0
   -2    1    0    0
    1    0    0    0
    0    0    0    0
```

From 16 non-zero values down to **7 non-zero values**! This is typical.

## The Zero Coefficient Effect

After quantization, many coefficients become zero:

- High-frequency coefficients are usually small to begin with
- Large quantization divisors push them to zero
- These zeros compress extremely well (run-length encoding)

**Typical quantized block (Q75)**:

```
DC  AC  AC  AC  AC  AC  AC  AC
┌────┬────┬────┬────┬────┬────┬────┬────┐
│ 42 │ -3 │  2 │ -1 │  0 │  0 │  0 │  0 │
├────┼────┼────┼────┼────┼────┼────┼────┤
│ -2 │  1 │  0 │  0 │  0 │  0 │  0 │  0 │
├────┼────┼────┼────┼────┼────┼────┼────┤
│  1 │  0 │  0 │  0 │  0 │  0 │  0 │  0 │
├────┼────┼────┼────┼────┼────┼────┼────┤
│  0 │  0 │  0 │  0 │  0 │  0 │  0 │  0 │
├────┼────┼────┼────┼────┼────┼────┼────┤
│  0 │  0 │  0 │  0 │  0 │  0 │  0 │  0 │
├────┼────┼────┼────┼────┼────┼────┼────┤
│  0 │  0 │  0 │  0 │  0 │  0 │  0 │  0 │
├────┼────┼────┼────┼────┼────┼────┼────┤
│  0 │  0 │  0 │  0 │  0 │  0 │  0 │  0 │
├────┼────┼────┼────┼────┼────┼────┼────┤
│  0 │  0 │  0 │  0 │  0 │  0 │  0 │  0 │
└────┴────┴────┴────┴────┴────┴────┴────┘

57 zeros out of 64 coefficients = excellent compression!
```

## Zigzag Ordering

After quantization, coefficients are read in zigzag order:

```rust
// From src/jpeg/quantize.rs
pub const ZIGZAG: [usize; 64] = [
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
];
```

This groups low frequencies first (non-zero values) and high frequencies last (zeros), enabling efficient run-length encoding of trailing zeros.

## Dequantization (Decoding)

During JPEG decoding, we multiply back:

```
Reconstructed[i] = Quantized[i] × Q[i]
```

**The error is permanent:**

```
Original DCT: 47
Quantized: 5 (after dividing by 10 and rounding)
Reconstructed: 50 (after multiplying by 10)
Permanent error: 3

This is what makes JPEG lossy.
```

## Quality vs. File Size

| Quality | Approx. Size | Visual Quality                  |
| ------- | ------------ | ------------------------------- |
| 95-100  | 80-100%      | Indistinguishable from original |
| 80-95   | 20-50%       | Excellent, minor artifacts      |
| 60-80   | 10-20%       | Very good, some edge artifacts  |
| 30-60   | 5-10%        | Acceptable, visible artifacts   |
| 10-30   | 2-5%         | Poor, heavy blocking            |
| 1-10    | 1-2%         | Very poor, extreme artifacts    |

## JPEG Artifacts Explained

### Blocking

When quality is too low, the 8×8 block boundaries become visible because adjacent blocks are quantized independently.

```
Original smooth gradient:
████████████████████████████████████████

After aggressive quantization:
████████│███████░│░░░░░░░░│░░░░░░░░
         ↑ Block boundaries become visible
```

### Mosquito Noise

Halos around sharp edges occur because removing high frequencies affects the entire block, not just the edge:

```
Original:                    After quantization:
████░░░░                    ████▒░░░
████░░░░                    ████▒░░░
████░░░░     ───────────▶   ████▒░░░
████░░░░                    ████▒░░░
                              │
                              └── "Mosquito noise" halo
```

### Color Bleeding

Aggressive chrominance quantization causes color to blur across edges:

```
Original (red | blue):      After quantization:
████░░░░                    ███▓▒░░░
████░░░░                    ███▓▒░░░
████░░░░     ───────────▶   ███▓▒░░░
████░░░░                    ███▓▒░░░
                              │
                              └── Colors bleed into each other
```

## Psychovisual Optimization

The standard quantization tables encode decades of human perception research:

1. **Contrast sensitivity function (CSF)**: Humans are most sensitive to spatial frequencies of about 2-5 cycles/degree of visual angle

2. **Masking**: Texture and noise hide artifacts; smooth areas reveal them

3. **Color vs. luminance**: The human visual system allocates more neural resources to brightness than color

Modern JPEG encoders may use **adaptive quantization** — different tables for different parts of the image based on local content.

## Custom Quantization Tables

JPEG allows custom quantization tables to be embedded in the file. Some applications use:

- **Flatter tables**: More uniform quality across frequencies
- **Content-specific tables**: Optimized for faces, landscapes, etc.
- **Perceptually optimized tables**: Research-derived improvements over the standard

## Summary

Quantization is where JPEG's dramatic compression comes from:

1. **Divide** DCT coefficients by quantization values
2. **Round** to integers (information lost permanently)
3. **High frequencies** get larger divisors (more rounding, more zeros)
4. **Zeros compress well** via run-length encoding

The quality parameter controls how aggressive this quantization is:

- Higher quality = smaller divisors = more precision = larger file
- Lower quality = larger divisors = more zeros = smaller file

Understanding quantization helps you choose appropriate quality settings:

- **Archival**: Quality 95-100
- **Web display**: Quality 80-85
- **Thumbnails**: Quality 60-70

*(Advanced: the `max` JPEG preset in this library enables trellis quantization to pick coefficient levels that minimize bit cost at the chosen quality.)*

---

## References

- [ITU-T T.81 - JPEG Standard, Annex K](https://www.w3.org/Graphics/JPEG/itu-t81.pdf)
- Watson, A.B. (1994). "Visibility of DCT Quantization Noise"
- See implementation: `src/jpeg/quantize.rs`
