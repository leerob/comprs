# JPEG Encoding

A raw 12-megapixel photo from your phone is about **36 MB** of pixel data. Saved as JPEG at quality 85, it's around **3-4 MB** — a 10x reduction with barely perceptible quality loss. How does JPEG achieve this?

The answer lies in a clever insight: **humans don't see all image details equally**. We're highly sensitive to brightness changes but less sensitive to color variations. We notice smooth gradients but miss fine textures. JPEG exploits these perceptual blind spots to discard information we won't miss.

JPEG (Joint Photographic Experts Group) is the most widely used image format for photographs. Unlike PNG, JPEG uses **lossy compression** — it permanently discards some image data to achieve dramatically smaller file sizes.

## When to Use JPEG

**JPEG excels at:**

- Photographs (natural scenes with smooth gradients)
- Any image where small imperfections are acceptable
- Web images where bandwidth matters

**Avoid JPEG for:**

- Text and screenshots (artifacts around sharp edges)
- Graphics with solid colors (better as PNG)
- Images needing transparency (JPEG has no alpha channel)
- Medical/scientific imaging (artifacts could be problematic)

## The JPEG Pipeline

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Raw Pixels  │───▶│  Color      │───▶│    DCT      │───▶│  Quantize   │
│   (RGB)     │    │  Convert    │    │ (frequency) │    │ (lossy!)    │
└─────────────┘    │  (YCbCr)    │    └─────────────┘    └─────────────┘
                   └─────────────┘                              │
                                                                ▼
                   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
                   │  JPEG File  │◀───│  Huffman    │◀───│  Entropy    │
                   │             │    │  Encode     │    │  Prep       │
                   └─────────────┘    └─────────────┘    │(zigzag+RLE) │
                                                         └─────────────┘
```

Each stage has a specific purpose (baseline 4:4:4 by default in this library; options can enable optimized Huffman tables, progressive scans, 4:2:0 subsampling, and trellis quantization for smaller files; see `JpegOptions` presets fast/balanced/max in the crate docs):

| Stage          | Purpose                        | Lossy?  |
| -------------- | ------------------------------ | ------- |
| Color Convert  | Separate brightness from color | No      |
| DCT            | Transform to frequency domain  | No      |
| Quantize       | Discard high-frequency detail  | **Yes** |
| Entropy Prep   | Prepare for efficient encoding | No      |
| Huffman Encode | Compress the result            | No      |

## Stage 1: Color Space Conversion

JPEG converts RGB to **YCbCr**:

- **Y**: Luminance (brightness)
- **Cb**: Blue chrominance (blue - luminance)
- **Cr**: Red chrominance (red - luminance)

Why? Two reasons:

1. **Human vision prioritizes brightness over color**. We can compress Cb and Cr more aggressively.

2. **Decorrelation**: RGB channels are highly correlated (bright pixels have high R, G, and B). YCbCr separates these into independent signals.

```rust
// From src/color.rs
pub fn rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let r = r as f32;
    let g = g as f32;
    let b = b as f32;

    // ITU-R BT.601 conversion
    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    let cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0;
    let cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0;

    (
        y.round().clamp(0.0, 255.0) as u8,
        cb.round().clamp(0.0, 255.0) as u8,
        cr.round().clamp(0.0, 255.0) as u8,
    )
}
```

Notice the weights: green contributes 58.7% to brightness because human eyes have more green-sensitive cells!

## Stage 2: Block Processing

JPEG processes the image in **8×8 blocks**. Each block is transformed independently.

```
Image divided into 8×8 blocks:
┌───┬───┬───┬───┐
│ 1 │ 2 │ 3 │ 4 │
├───┼───┼───┼───┤
│ 5 │ 6 │ 7 │ 8 │
├───┼───┼───┼───┤
│ 9 │10 │11 │12 │
└───┴───┴───┴───┘
```

**Why 8×8?** This size is a sweet spot:
- Small enough that pixels within a block are correlated (similar colors)
- Large enough that the DCT produces useful frequency separation
- Matches CPU cache lines for efficient processing
- 64 coefficients fit nicely in hardware implementations

If the image dimensions aren't multiples of 8, we pad by replicating edge pixels.

```rust
// From src/jpeg/mod.rs
fn extract_block(
    data: &[u8],
    width: usize,
    height: usize,
    block_x: usize,
    block_y: usize,
    color_type: ColorType,
) -> ([f32; 64], [f32; 64], [f32; 64]) {
    let mut y_block = [0.0f32; 64];
    // ...

    for dy in 0..8 {
        for dx in 0..8 {
            // Clamp to image bounds (padding)
            let x = (block_x + dx).min(width - 1);
            let y = (block_y + dy).min(height - 1);
            // ...
        }
    }
}
```

## Stage 3: Discrete Cosine Transform (DCT)

The DCT converts spatial data to **frequency components**. See [DCT documentation](./dct.md) for the mathematical details.

Key insight: After DCT, most of the image energy concentrates in the **low-frequency components** (top-left of the 8×8 block). High-frequency components (bottom-right) are often small.

```
DCT Output (typical photo block):
┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
│ 952 │ -27 │  14 │   3 │   0 │   1 │   0 │   0 │
│ -29 │  11 │   5 │   2 │   1 │   0 │   0 │   0 │
│  13 │   7 │   4 │   2 │   0 │   0 │   0 │   0 │
│   4 │   3 │   2 │   1 │   0 │   0 │   0 │   0 │
│   1 │   1 │   0 │   0 │   0 │   0 │   0 │   0 │
│   0 │   0 │   0 │   0 │   0 │   0 │   0 │   0 │
│   0 │   0 │   0 │   0 │   0 │   0 │   0 │   0 │
│   0 │   0 │   0 │   0 │   0 │   0 │   0 │   0 │
└─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
  DC    ───────────────────────────────▶
        Low frequency        High frequency
```

The top-left value is the **DC coefficient** (average brightness). All others are **AC coefficients** (variations from the average).

## Stage 4: Quantization (The Lossy Step!)

This is where JPEG discards information. Each DCT coefficient is divided by a quantization value and rounded:

```
Quantized = round(DCT_coefficient / Quantization_value)
```

The quantization tables have larger values for high frequencies (aggressive rounding) and smaller values for low frequencies (preserve detail):

```rust
// From src/jpeg/quantize.rs
const STD_LUMINANCE_TABLE: [u8; 64] = [
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
];
```

See [Quantization documentation](./quantization.md) for details on how quality affects these tables.

After quantization, many coefficients become **zero**, especially in the high-frequency region:

```
Before quantization:     After quantization (Q=75):
952  -27   14    3       60  -2    1    0
-29   11    5    2       -2   1    0    0
 13    7    4    2        1   0    0    0
  4    3    2    1        0   0    0    0
```

## Stage 5: Zigzag Scan

We read the quantized coefficients in **zigzag order**, grouping low frequencies first:

```
Read order:
┌───┬───┬───┬───┬───┬───┬───┬───┐
│ 0 │ 1 │ 5 │ 6 │14 │15 │27 │28 │
│ 2 │ 4 │ 7 │13 │16 │26 │29 │42 │
│ 3 │ 8 │12 │17 │25 │30 │41 │43 │
│ 9 │11 │18 │24 │31 │40 │44 │53 │
│10 │19 │23 │32 │39 │45 │52 │54 │
│20 │22 │33 │38 │46 │51 │55 │60 │
│21 │34 │37 │47 │50 │56 │59 │61 │
│35 │36 │48 │49 │57 │58 │62 │63 │
└───┴───┴───┴───┴───┴───┴───┴───┘
```

**Why zigzag?** After quantization, most non-zero values cluster in the top-left (low frequencies), while the bottom-right (high frequencies) is mostly zeros. Zigzag ordering:
- Reads non-zero values first
- Groups zeros together at the end
- Enables efficient run-length encoding ("15 zeros, then -2, then EOB")

```rust
// From src/jpeg/quantize.rs
pub const ZIGZAG: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];
```

## Stage 6: DC Coefficient Encoding (DPCM)

DC coefficients (the average brightness of each block) change slowly between adjacent blocks. We encode the **difference** from the previous block (Differential Pulse Code Modulation):

```
Block DCs:   512,  515,  513,  516,  514
Differences:  512,    3,   -2,    3,   -2

Differences are small numbers → fewer bits needed!
```

**Why DPCM?** Adjacent 8×8 blocks in a photograph usually have similar average brightness. A blue sky might have DC values like 180, 181, 180, 182... The differences (0, 1, -1, 2) require far fewer bits than the absolute values.

```rust
// From src/jpeg/huffman.rs
pub fn encode_block(..., prev_dc: i16, ...) -> i16 {
    // ...
    let dc = zigzag[0];
    let dc_diff = dc - prev_dc;
    let dc_cat = category(dc_diff);

    // Encode category then value
    // ...

    dc  // Return for next block's difference
}
```

## Stage 7: AC Coefficient Encoding (Run-Length)

AC coefficients are encoded as (run, value) pairs:

- **Run**: Number of zeros before this value
- **Value**: The non-zero coefficient

```
Zigzag sequence: 60, -2, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0, ...EOB

Encoded as:
  DC: 60
  (0, -2)  ← zero run of 0, then -2
  (0, 1)   ← zero run of 0, then 1
  (3, -1)  ← zero run of 3, then -1
  EOB      ← end of block (all remaining are 0)
```

**Why run-length encoding?** After quantization, a typical block might be 60% zeros. Instead of encoding each zero individually, we say "skip 3 zeros, then -1". The EOB (End of Block) symbol is especially powerful — it says "everything else is zero" in just a few bits.

For long runs of zeros (16+), a special ZRL (zero run length) code is used:

```rust
// From src/jpeg/huffman.rs
while zero_run >= 16 {
    let zrl_code = tables.get_ac_code(0xF0, is_luminance);  // ZRL = 16 zeros
    writer.write_bits(zrl_code.code as u32, zrl_code.length);
    zero_run -= 16;
}
```

## Stage 8: Huffman Encoding

Finally, the run/value pairs are Huffman encoded using Huffman tables. By default we use the standard JPEG tables; with the new `optimize_huffman` option, we build per-image tables from coefficient frequencies (mozjpeg-style `optimize_coding`) and fall back to the standard tables if code lengths would exceed 16 bits.

- **DC tables**: Encode the category (number of bits needed for the difference)
- **AC tables**: Encode the (run, size) byte

JPEG uses separate tables for luminance (Y) and chrominance (Cb, Cr) to optimize for their different statistics.

We can push compression further by building custom Huffman tables tuned to each image's actual symbol frequencies, rather than using the standard tables. For details on this and other advanced optimizations, see [Performance Optimization](./performance-optimization.md).

## JPEG File Structure

A JPEG file consists of **markers** and **segments**:

```
┌──────────────┐
│ SOI (FFD8)   │  Start of Image
├──────────────┤
│ APP0 (FFE0)  │  JFIF marker (metadata)
├──────────────┤
│ DQT (FFDB)   │  Define Quantization Tables
├──────────────┤
│ SOF0 (FFC0)  │  Start of Frame (dimensions, components)
├──────────────┤
│ DHT (FFC4)   │  Define Huffman Tables
├──────────────┤
│ SOS (FFDA)   │  Start of Scan (encoded image data follows)
├──────────────┤
│ (image data) │  Entropy-coded blocks
├──────────────┤
│ EOI (FFD9)   │  End of Image
└──────────────┘
```

```rust
// From src/jpeg/mod.rs
const SOI: u16 = 0xFFD8;  // Start of Image
const EOI: u16 = 0xFFD9;  // End of Image
const APP0: u16 = 0xFFE0; // JFIF marker
const DQT: u16 = 0xFFDB;  // Define Quantization Table
const SOF0: u16 = 0xFFC0; // Start of Frame (baseline DCT)
const DHT: u16 = 0xFFC4;  // Define Huffman Table
const SOS: u16 = 0xFFDA;  // Start of Scan
```

## Byte Stuffing

Since 0xFF marks the start of JPEG markers, if 0xFF appears in the compressed data, we must **stuff** a 0x00 after it:

```
Data byte:     0xFF
In file:       0xFF 0x00  (stuffed)

Marker:        0xFF 0xD8
In file:       0xFF 0xD8  (not stuffed - it's a real marker)
```

```rust
// From src/bits.rs (BitWriterMsb)
if self.current_byte == 0xFF {
    self.buffer.push(0x00);  // Byte stuffing
}
```

## Quality Setting

The quality parameter (1-100) scales the quantization tables:

- **Quality 100**: Quantization values near 1 (minimal loss)
- **Quality 50**: Standard quantization tables
- **Quality 1**: Very high quantization values (maximum loss)

```rust
// From src/jpeg/quantize.rs
let scale = if quality < 50 {
    5000 / quality as u32
} else {
    200 - 2 * quality as u32
};
```

| Quality | Scale | Compression | Visual Quality |
| ------- | ----- | ----------- | -------------- |
| 100     | 1     | ~2-3x       | Excellent      |
| 85      | 30    | ~10-15x     | Very good      |
| 50      | 100   | ~20-30x     | Good           |
| 25      | 200   | ~40-60x     | Poor           |

## Complete Encoding Flow

```rust
// Encode a simple image
let pixels = vec![255, 0, 0];  // 1x1 red pixel
let jpeg = jpeg::encode(&pixels, 1, 1, 85)?;
```

What happens:

1. Validate input
2. Create quantization tables for quality 85
3. Create Huffman tables
4. Write SOI, APP0, DQT, SOF0, DHT, SOS markers
5. For each 8×8 block:
   - Convert RGB to YCbCr
   - Apply 2D DCT
   - Quantize with quality-scaled tables
   - Encode DC differentially
   - Encode AC with run-length + Huffman
6. Write EOI marker

## JPEG Artifacts: Cause and Effect

Understanding JPEG's characteristic artifacts reveals how the algorithm works:

### Blocking (Grid Pattern)

```
Original smooth gradient:        After aggressive JPEG:
████████████████████████        ████████│███████░│░░░░░░░░
████████████████████████  →     ████████│███████░│░░░░░░░░
████████████████████████        ████████│███████░│░░░░░░░░
                                        ↑        ↑
                                      Block boundaries visible
```

**Cause**: Each 8×8 block is quantized independently. At low quality, adjacent blocks may quantize to noticeably different average values.

**Why it happens**: The DC coefficient (block average) gets rounded differently in neighboring blocks, creating visible discontinuities at boundaries.

### Mosquito Noise (Edge Halos)

```
Original sharp edge:            After JPEG:
████████░░░░░░░░               ████████▒░░░░░░░
████████░░░░░░░░  →            ████████▒░░░░░░░
████████░░░░░░░░               ████████▒░░░░░░░
                                       ↑
                                    Halo artifact
```

**Cause**: Sharp edges contain high-frequency DCT components. When these are quantized away, the edge "rings" — the Gibbs phenomenon from signal processing.

**Why it happens**: The DCT represents sharp transitions as a sum of many frequencies. Removing high frequencies leaves behind oscillations near the edge.

### Color Bleeding

```
Original (red|blue):            After JPEG with 4:2:0:
████████░░░░░░░░               ████████▓▒░░░░░░
████████░░░░░░░░  →            ████████▓▒░░░░░░
                                       ↑↑
                                    Color smears across edge
```

**Cause**: Chroma subsampling (4:2:0) averages color over 2×2 pixel blocks. Combined with DCT quantization in the color channels, color detail is lost.

**Why it happens**: The Cb and Cr channels are encoded at half resolution, then upscaled on decode. Fine color detail cannot survive this process.

### Quality vs. Artifact Severity

| Quality | Blocking | Mosquito | Color Bleed | File Size |
|---------|----------|----------|-------------|-----------|
| 95-100  | None     | None     | Minimal     | Large     |
| 80-90   | Minimal  | Minimal  | Slight      | Medium    |
| 50-70   | Visible  | Moderate | Noticeable  | Small     |
| 10-40   | Severe   | Severe   | Severe      | Tiny      |

## Common Pitfalls

### 1. Using JPEG for Screenshots or Text

JPEG's DCT-based compression creates artifacts around sharp edges. Text becomes blurry with visible halos:

```
Original text:  The quick brown fox
After JPEG Q50: T̲h̲e̲ q̲u̲i̲c̲k̲ b̲r̲o̲w̲n̲ f̲o̲x̲  ← fuzzy edges, ringing
```

**Rule**: Use PNG for screenshots, text, diagrams, and UI elements.

### 2. Re-Compressing JPEG Files

Each JPEG save introduces more quantization error. Editing and re-saving repeatedly degrades quality:

```
Original    → Save Q85 → Edit → Save Q85 → Edit → Save Q85
Quality:       Good        OK        Meh        Bad
```

**Rule**: Keep original files; export to JPEG only as a final step.

### 3. Quality 100 ≠ Lossless

Even at quality 100, JPEG still quantizes (with small divisors). For truly lossless storage, use PNG or keep the raw source.

### 4. Wrong Quality for the Use Case

| Use Case | Recommended Quality | Why |
|----------|---------------------|-----|
| Archival | 92-95 | Preserve detail, still save space |
| Web display | 80-85 | Good balance of quality/size |
| Thumbnails | 60-75 | Small size matters more |
| Preview/draft | 40-60 | Speed and size over quality |

### 5. Ignoring Chroma Subsampling

Default 4:4:4 subsampling preserves color detail but increases file size. For photos (where color detail is less critical), 4:2:0 can reduce size by 25-35% with minimal visible difference.

```
4:4:4: Full color resolution (larger file)
4:2:0: Half color resolution (smaller file, usually fine for photos)
```

## Summary

JPEG achieves excellent photo compression through:

- **Color space conversion** (YCbCr) for decorrelation
- **8×8 block DCT** to concentrate energy
- **Quantization** to discard imperceptible detail
- **Zigzag scan** to group zeros
- **DPCM** for DC coefficients
- **Run-length encoding** for AC coefficients
- **Huffman coding** for final compression

The result: 10-20x compression with minimal visible quality loss.

## Next Steps

For deeper understanding of the mathematical foundation, see [Discrete Cosine Transform (DCT)](./dct.md) and [JPEG Quantization](./quantization.md).

---

## References

- [ITU-T T.81 - JPEG Standard](https://www.w3.org/Graphics/JPEG/itu-t81.pdf)
- Wallace, G.K. (1991). "The JPEG Still Picture Compression Standard"
- See implementation: `src/jpeg/mod.rs`, `src/jpeg/huffman.rs`
