# PNG Encoding

Consider a screenshot of a text editor. Most pixels are the same background color. Adjacent pixels in the text are nearly identical shades. Yet if you stored every pixel as raw RGB values, a 1920×1080 screenshot would be:

```text
1920 × 1080 × 3 = 6,220,800 bytes ≈ 6 MB
```

With PNG compression, that same screenshot might be just 200-400 KB, a **15-30x reduction**. How does PNG achieve this? By exploiting a key insight: **adjacent pixels are usually similar**.

## The Core Insight

PNG uses **predictive filtering** before compression. Instead of storing raw pixel values, it stores the _difference_ between each pixel and a predicted value based on its neighbors.

Why does this help? Consider these two sequences:

```text
Raw values:      128, 130, 132, 134, 136, 138, 140, 142
Differences:       0,   2,   2,   2,   2,   2,   2,   2
                   ↑
                   (first pixel has no left neighbor)
```

The raw values are all different (8 unique values). The differences are mostly the same (just one unique value: 2). Repeated values compress extremely well with DEFLATE!

## The PNG Pipeline

```text
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Raw Pixels  │───▶│  Filtering  │───▶│   DEFLATE   │───▶│  PNG Chunks │
│  (RGB/RGBA) │    │ (predict &  │    │  (LZ77 +    │    │  (file      │
│             │    │  subtract)  │    │   Huffman)  │    │   format)   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                         │
                         └── This is where the magic happens!
```

The filtering step is what makes PNG so effective. Let's understand each filter type.

## The Five PNG Filters

PNG defines five filter types, each predicting the current pixel from its neighbors:

```text
Neighbor positions for pixel X:

    C | B
   ---+---
    A | X  ← current pixel

A = Left pixel (same row)
B = Above pixel (previous row)
C = Upper-left pixel (diagonal)
```

### Filter 0: None

No prediction. Just use the raw value.

```text
Prediction: 0
Output: X - 0 = X (unchanged)
```

**Best for**: Random or noisy data where neighbors don't help.

### Filter 1: Sub

Predict from the **left** pixel.

```text
Prediction: A (left pixel)
Output: X - A
```

**Best for**: Horizontal gradients (colors changing left-to-right).

### Filter 2: Up

Predict from the **above** pixel.

```text
Prediction: B (above pixel)
Output: X - B
```

**Best for**: Vertical gradients (colors changing top-to-bottom).

### Filter 3: Average

Predict from the **average** of left and above.

```text
Prediction: floor((A + B) / 2)
Output: X - floor((A + B) / 2)
```

**Best for**: Diagonal gradients, smooth transitions.

### Filter 4: Paeth

The cleverest filter! Named after Alan Paeth, it predicts using the neighbor **closest to a linear interpolation**.

```text
p = A + B - C   (linear combination)

Choose whichever of A, B, C is closest to p:
  pa = |p - A|
  pb = |p - B|
  pc = |p - C|

Prediction = A if pa ≤ pb and pa ≤ pc
           = B if pb ≤ pc
           = C otherwise
```

**Best for**: General-purpose; often produces the best results across varied content.

## Worked Example: Filtering a Row

Let's filter a simple 8-pixel grayscale row. Assume the previous row was all zeros (or this is the first row).

```text
Raw pixel values:  100, 102, 104, 106, 108, 110, 112, 114
Previous row:        0,   0,   0,   0,   0,   0,   0,   0
```

### With Filter 0 (None):

```text
Output: 100, 102, 104, 106, 108, 110, 112, 114  (unchanged)
Sum of absolute values: 860
```

### With Filter 1 (Sub):

```text
Position 0: 100 - 0   = 100  (no left pixel, use 0)
Position 1: 102 - 100 = 2
Position 2: 104 - 102 = 2
Position 3: 106 - 104 = 2
Position 4: 108 - 106 = 2
Position 5: 110 - 108 = 2
Position 6: 112 - 110 = 2
Position 7: 114 - 112 = 2

Output: 100, 2, 2, 2, 2, 2, 2, 2
Sum of absolute values: 114  ← Much better!
```

### With Filter 2 (Up):

```text
Position 0: 100 - 0 = 100
Position 1: 102 - 0 = 102
...all positions subtract 0 from previous row...

Output: 100, 102, 104, 106, 108, 110, 112, 114
Sum of absolute values: 860  (same as None for first row)
```

The Sub filter wins here because this is a **horizontal gradient** — values increase left-to-right with constant step.

## Why Filtering Works

The filtered output has **lower entropy** than the raw data:

| Metric          | Raw Values | Sub-Filtered     |
| --------------- | ---------- | ---------------- |
| Unique values   | 8          | 2                |
| Range           | 100-114    | 2-100            |
| Repeated values | 0          | 7 identical "2"s |

DEFLATE's LZ77 stage finds the repeated "2" values and encodes them as back-references. Huffman coding then assigns short codes to the common value "2".

The mathematical principle: if adjacent pixels differ by similar amounts, those differences cluster around a few values, reducing entropy.

## Choosing the Best Filter Per Row

Different rows benefit from different filters. A horizontal stripe needs Sub; a vertical stripe needs Up.

**Adaptive filtering** tries multiple filters and picks the best one for each row independently:

```text
Row 1: Horizontal gradient → Sub wins   → encode with filter 1
Row 2: Vertical continuation → Up wins  → encode with filter 2
Row 3: Noisy texture → None wins        → encode with filter 0
Row 4: Smooth diagonal → Paeth wins     → encode with filter 4
```

Each row starts with a **filter type byte** (0-4), followed by the filtered data.

### Scoring Heuristic

How do we know which filter "wins"? The most common heuristic is **sum of absolute values**:

```rust,ignore
fn score(filtered: &[u8]) -> u64 {
    filtered.iter()
        .map(|&b| (b as i8).unsigned_abs() as u64)
        .sum()
}
```

Lower scores typically compress better because:

- Values near zero dominate
- Fewer unique values appear
- Run-length encoding becomes more effective

## Filter Strategies

Our library provides several strategies:

| Strategy       | Description                      | Speed   | Compression |
| -------------- | -------------------------------- | ------- | ----------- |
| `None`         | Always use filter 0              | Fastest | Poor        |
| `Sub`          | Always use filter 1              | Fast    | Moderate    |
| `Up`           | Always use filter 2              | Fast    | Moderate    |
| `Average`      | Always use filter 3              | Fast    | Moderate    |
| `Paeth`        | Always use filter 4              | Fast    | Good        |
| `MinSum`       | Try all 5, pick lowest score     | Medium  | Very Good   |
| `AdaptiveFast` | Try Sub/Up/Paeth with early exit | Medium  | Good        |
| `Adaptive`     | Try all 5 per row                | Slow    | Best        |

```rust
use pixo::png::{PngOptions, FilterStrategy};

let options = PngOptions {
    filter_strategy: FilterStrategy::MinSum,
    compression_level: 6,
    ..Default::default()
};
```

## Presets

We bundle filter strategy, compression level, and optimizations into convenient presets:

### Fast (Preset 0)

```rust,ignore
PngOptions::fast()
```

- **Compression level**: 2
- **Filter strategy**: AdaptiveFast
- **Best for**: Development, real-time processing

### Balanced (Preset 1)

```rust,ignore
PngOptions::balanced()
```

- **Compression level**: 6
- **Filter strategy**: Adaptive
- **Optimizations**: Palette reduction, color type reduction, alpha optimization, metadata stripping
- **Best for**: General use, web assets

### Max (Preset 2)

```rust,ignore
PngOptions::max()
```

- **Compression level**: 9
- **Filter strategy**: MinSum with iterative refinement
- **DEFLATE**: Optimal parsing + dynamic tables (Zopfli-style)
- **Best for**: Final distribution, every byte counts

## Lossless Optimizations

Beyond filtering, PNG encoders can apply lossless transformations. (When lossy output is acceptable, the library can also quantize to a palette via median-cut with optional Floyd–Steinberg dithering.)

### Palette Reduction

Images with ≤256 unique colors become **indexed color** with a PLTE chunk. A **palette** is a table of colors; each pixel stores a small index into that table.

```text
Before: 24 bits/pixel (RGB)
After:  8 bits/pixel (index) + palette overhead

Savings: up to 67% for simple graphics!
```

### Color Type Reduction

```text
RGBA with all pixels opaque → RGB     (saves 25%)
RGB where R=G=B always     → Gray    (saves 67%)
RGBA where R=G=B + alpha   → GrayA   (saves 50%)
```

### Alpha Optimization

For fully transparent pixels (alpha=0), the RGB values are invisible. Setting them to a constant (e.g., 0,0,0) creates more repetition:

```text
Before: rgba(147, 82, 203, 0), rgba(91, 44, 187, 0)  ← random invisible colors
After:  rgba(0, 0, 0, 0), rgba(0, 0, 0, 0)           ← identical, compresses well
```

## File Structure

A PNG file is a sequence of **chunks**:

```text
┌──────────────────────────────────────────────────────────────────┐
│ PNG Signature (8 bytes): 89 50 4E 47 0D 0A 1A 0A                 │
├──────────────────────────────────────────────────────────────────┤
│ IHDR Chunk: Width, Height, Bit depth, Color type, Compression,  │
│             Filter method, Interlace method                      │
├──────────────────────────────────────────────────────────────────┤
│ PLTE Chunk (optional): Palette for indexed color images         │
├──────────────────────────────────────────────────────────────────┤
│ IDAT Chunk(s): Compressed image data (filtered + DEFLATE)       │
├──────────────────────────────────────────────────────────────────┤
│ IEND Chunk: End marker (empty, just signals EOF)                │
└──────────────────────────────────────────────────────────────────┘
```

Each chunk structure:

```text
┌─────────┬─────────┬─────────────────┬─────────┐
│ Length  │  Type   │      Data       │  CRC-32 │
│ 4 bytes │ 4 bytes │ (Length bytes)  │ 4 bytes │
└─────────┴─────────┴─────────────────┴─────────┘
```

## The Paeth Predictor: A Closer Look

The Paeth predictor deserves special attention because it's non-obvious but effective:

```text
Given: A=left, B=above, C=upper-left

Step 1: Compute the "ideal" prediction as if the pixel were on a plane:
        p = A + B - C

Step 2: Find which actual neighbor is closest to this ideal:
        pa = |p - A|
        pb = |p - B|
        pc = |p - C|

Step 3: Return the closest one (with tie-breaking: A > B > C priority)
```

### Why It Works

Consider a smooth gradient going diagonally:

```text
C=100 | B=110
------+------
A=110 | X=120

p = 110 + 110 - 100 = 120  ← exactly right!
```

The Paeth predictor essentially fits a plane through the three neighbors and extrapolates to X. For smooth regions, this predicts perfectly. For edges, it intelligently falls back to the neighbor on the correct side of the edge.

## Common Pitfalls

### 1. Using PNG for Photographs

PNG is **lossless**, which means it preserves every pixel perfectly — including noise. Photographs have natural noise that PNG cannot compress efficiently.

```text
Photo as JPEG (Q85): 150 KB
Photo as PNG:        2.5 MB  ← 16x larger!
```

**Rule of thumb**: Use JPEG for photos, PNG for screenshots/graphics.

### 2. Wrong Filter Strategy for Content Type

Using a fixed filter on varied content wastes compression:

| Content            | Best Filter | Wrong Choice Penalty      |
| ------------------ | ----------- | ------------------------- |
| Horizontal stripes | Sub         | Up gives 0% compression   |
| Vertical stripes   | Up          | Sub gives 0% compression  |
| Photos/textures    | Adaptive    | Fixed filter loses 10-30% |

### 3. Over-Optimizing Small Images

Adaptive filtering has CPU overhead. For tiny images (<64×64), a simple fixed filter (Sub or Paeth) is often faster with minimal compression loss.

### 4. Ignoring Alpha Channel Waste

RGBA images with no transparency waste 25% of bytes. Our balanced preset auto-detects this and reduces to RGB.

### 5. Not Using Palette Mode for Simple Graphics

Icons, logos, and diagrams often have <256 colors. Palette mode dramatically reduces size:

```text
Simple icon as RGBA: 12 KB
Same icon as indexed: 2 KB  ← 6x smaller!
```

## Summary

PNG compression works through two key mechanisms:

1. **Predictive filtering** — Exploit spatial correlation between adjacent pixels
2. **DEFLATE compression** — LZ77 + Huffman on the filtered data

The key insight of PNG is that filtering **transforms the data** to be more compressible without losing any information. Every byte you save in the filtered output translates directly to a smaller file.

## Next Steps

Continue to [JPEG Encoding](./jpeg-encoding.md) to learn about lossy compression for photographs.

---

## References

- [RFC 2083 - PNG Specification](https://www.w3.org/TR/PNG/)
- [PNG Filter Algorithms (W3C)](https://www.w3.org/TR/PNG-Filters.html)
- Paeth, A.W. (1991). "Image File Compression Made Easy"
- See implementation: `src/png/mod.rs`, `src/png/filter.rs`
