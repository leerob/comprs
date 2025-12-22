# JPEG Encoding

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

Each stage has a specific purpose:

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

This places zeros together at the end, enabling efficient run-length encoding.

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

DC coefficients change slowly between adjacent blocks. We encode the **difference** from the previous block (Differential Pulse Code Modulation):

```
Block DCs:   512,  515,  513,  516,  514
Differences:  512,    3,   -2,    3,   -2

Differences are small numbers → fewer bits needed!
```

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

Finally, the run/value pairs are Huffman encoded using standard tables:

- **DC tables**: Encode the category (number of bits needed for the difference)
- **AC tables**: Encode the (run, size) byte

JPEG uses separate tables for luminance (Y) and chrominance (Cb, Cr) to optimize for their different statistics.

```rust
// From src/jpeg/huffman.rs
impl HuffmanTables {
    fn get_dc_code(&self, category: u8, is_luminance: bool) -> HuffCode {
        if is_luminance {
            self.dc_lum_codes[category as usize]
        } else {
            self.dc_chrom_codes[category as usize]
        }
    }
}
```

### Optional: Optimized Huffman tables

You can trade a small amount of extra work for smaller files by enabling a two-pass optimization that collects per-image symbol frequencies and builds custom Huffman tables:

```rust
use comprs::jpeg::{self, JpegOptions, Subsampling};

let opts = JpegOptions {
    quality: 85,
    subsampling: Subsampling::S444,
    restart_interval: None,
    optimize_huffman: true, // enable optimization
};

let jpeg = jpeg::encode_with_options(&pixels, width, height, 85, ColorType::Rgb, &opts)?;
```

When enabled, the encoder first scans the image to gather DC/AC frequencies, builds canonical JPEG Huffman tables from those frequencies (respecting the 16-bit length limit), then encodes the image with the tailored tables. This typically reduces JPEG size without affecting quality, at the cost of a second pass over the image data.

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

## JPEG Artifacts

Understanding JPEG's artifacts helps explain the algorithm:

**Blocking**: Visible 8×8 grid at low quality — blocks are processed independently

**Mosquito noise**: Halos around sharp edges — high frequencies are quantized away

**Color bleeding**: Chroma blur in detailed areas — Cb/Cr may be subsampled

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
