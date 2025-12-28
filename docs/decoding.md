# Decoding: The Other Half of a Codec

When you open a JPEG in your photo app, the encoder's work happened long ago, perhaps on a camera, a web server, or your friend's phone. What matters now is the **decoder**: the algorithm that reconstructs viewable pixels from a compact bitstream.

This document explores decoding for both PNG and JPEG, explaining how each encoding step has a corresponding decoding step.

## What Is a Codec?

The word **codec** combines "coder" and "decoder" — the two halves of any compression system:

```text
Original     ┌──────────┐    Compressed     ┌──────────┐     Reconstructed
  Data  ────▶│  Encoder │────  Stream  ────▶│  Decoder │────▶   Data
             └──────────┘                   └──────────┘
```

For **lossless** codecs (like PNG), the reconstructed data is identical to the original. For **lossy** codecs (like JPEG), it's an approximation — close enough that humans don't notice the difference, but not bit-for-bit identical.

### Symmetry of Operations

Every encoding operation has a corresponding decoding operation:

| Encoding Step                       | Decoding Step                        |
| ----------------------------------- | ------------------------------------ |
| Predict pixel, store difference     | Add prediction to difference         |
| Build Huffman codes, encode symbols | Rebuild Huffman tree, decode symbols |
| Quantize (divide and round)         | Dequantize (multiply)                |
| Forward DCT                         | Inverse DCT                          |
| Convert RGB → YCbCr                 | Convert YCbCr → RGB                  |

The decoder reverses each step, but in opposite order: what was encoded last is decoded first.

## Asymmetric Complexity

An important insight: **decoders are often simpler than encoders**.

Consider JPEG:

- The **encoder** must decide how aggressively to quantize, which Huffman tables to use, whether progressive mode helps, and which filter strategy minimizes output size
- The **decoder** just follows instructions: read the tables from the file, apply them mechanically

```text
Encoding decision:  "Should I use Q=75 or Q=80? Let me try both..."
Decoding action:    "The file says Q=75. Multiply by these values."
```

This asymmetry is intentional. Encoders run once (when creating the file), but decoders run millions of times (every time someone views the image). A complex encoder that produces smaller files is worth the effort; the decoder should be fast and simple.

### Implications

1. **Decoders must be robust**: They'll encounter files from buggy encoders, partially corrupted data, and edge cases the encoder never tested
2. **Decoders can't optimize**: They must faithfully reproduce what the encoder intended, even if a better reconstruction is theoretically possible
3. **Standards specify decoding**: JPEG and PNG standards precisely define decoder behavior; encoder behavior is more flexible

## PNG Decoding Pipeline

PNG decoding reverses the encoding pipeline:

```text
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  PNG File   │───▶│   INFLATE   │───▶│  Unfilter   │───▶│ Raw Pixels  │
│  (chunks)   │    │ (decompress)│    │ (add back   │    │  (RGB/RGBA) │
└─────────────┘    └─────────────┘    │ predictions)│    └─────────────┘
      │                  │            └─────────────┘
      │                  │
      ├── Parse IHDR     ├── Rebuild Huffman trees
      ├── Find IDATs     ├── Decode literals & lengths
      └── Check CRCs     └── Copy from sliding window
```

### Step 1: Parse Chunks

A PNG file is a sequence of chunks. The decoder reads:

```text
PNG Signature: 89 50 4E 47 0D 0A 1A 0A  (validates this is a PNG)

IHDR chunk: width, height, bit depth, color type
PLTE chunk: palette (if indexed color)
IDAT chunk(s): compressed pixel data (may be split across multiple chunks)
IEND chunk: end marker
```

Each chunk includes a CRC-32 checksum. A decoder should verify these to detect corruption.

### Step 2: INFLATE (Decompress DEFLATE)

The IDAT chunks contain zlib-wrapped DEFLATE data. Decompression involves:

1. **Parse zlib header** — verify compression method is DEFLATE
2. **Process DEFLATE blocks** — each block is stored, fixed Huffman, or dynamic Huffman
3. **Decode Huffman symbols** — rebuild trees from transmitted code lengths
4. **Reconstruct data** — emit literal bytes or copy from sliding window
5. **Verify Adler-32 checksum** — ensure decompressed data is correct

#### Huffman Decoding

The encoder transmitted Huffman code lengths, not the codes themselves. The decoder rebuilds the codes using the canonical Huffman construction:

```text
Code lengths received: A=2, B=1, C=3, D=3

Rebuild codes (shorter codes get smaller values):
  B: 0       (length 1)
  A: 10      (length 2)
  C: 110     (length 3)
  D: 111     (length 3)
```

When decoding, read bits until you match a code:

```text
Input bits: 1 1 0 1 0 0 ...
            ─────
              C    A  B

Decoded: C, A, B, ...
```

#### LZ77 Back-References

When the decoder sees a length/distance pair instead of a literal byte, it copies from already-decoded data:

```text
Output so far: "the quick brown "
Decode: (length=3, distance=16)  ← "copy 3 bytes from 16 positions back"
Result: "the quick brown the"
                        ^^^
                        copied from position 0
```

The "sliding window" is just the recent output — typically 32KB for DEFLATE.

```rust,ignore
// From src/decode/inflate.rs
let start = output.len() - distance;
for i in 0..length {
    let byte = output[start + (i % distance)];
    output.push(byte);
}
```

### Step 3: Unfilter (Reverse Predictions)

After INFLATE, we have filtered scanlines. Each row starts with a filter type byte (0-4), followed by filtered pixel data.

To unfilter, we **add back** the prediction:

| Filter      | Encoding (stored value) | Decoding (reconstruction) |
| ----------- | ----------------------- | ------------------------- |
| None (0)    | X                       | X                         |
| Sub (1)     | X - A                   | stored + A                |
| Up (2)      | X - B                   | stored + B                |
| Average (3) | X - floor((A+B)/2)      | stored + floor((A+B)/2)   |
| Paeth (4)   | X - Paeth(A,B,C)        | stored + Paeth(A,B,C)     |

Where A = left pixel, B = above pixel, C = upper-left pixel.

**Worked example** (Sub filter):

```text
Stored values:    [1, 100, 2, 2, 2, 2, 2, 2]
                   ↑ filter type = Sub
                       ↑ first pixel (no left neighbor, A=0)

Decode:
  Pixel 0: 100 + 0   = 100
  Pixel 1: 2   + 100 = 102
  Pixel 2: 2   + 102 = 104
  Pixel 3: 2   + 104 = 106
  ...

Reconstructed: 100, 102, 104, 106, 108, 110, 112, 114
```

## JPEG Decoding Pipeline

JPEG decoding is more complex due to its lossy, block-based nature:

```text
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  JPEG File  │───▶│  Huffman    │───▶│ Dequantize  │───▶│ Inverse DCT │
│  (markers)  │    │   Decode    │    │ (multiply)  │    │ (IDCT)      │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
      │                                                         │
      │                                                         ▼
      ├── Parse SOI, APP0           ┌─────────────┐    ┌─────────────┐
      ├── Read DQT tables           │ Raw Pixels  │◀───│  YCbCr →    │
      ├── Read DHT tables           │   (RGB)     │    │    RGB      │
      └── Parse SOS scan            └─────────────┘    └─────────────┘
```

### Step 1: Parse Markers

JPEG uses marker segments starting with 0xFF:

```text
FFD8        SOI (Start of Image)
FFE0 xxxx   APP0 (JFIF metadata)
FFDB xxxx   DQT (Quantization tables)
FFC0 xxxx   SOF0 (Frame header: dimensions, components)
FFC4 xxxx   DHT (Huffman tables)
FFDA xxxx   SOS (Start of Scan, followed by entropy-coded data)
FFD9        EOI (End of Image)
```

The decoder extracts quantization tables, Huffman tables, and image dimensions before decoding pixel data.

### Step 2: Entropy Decoding

The scan data is Huffman-encoded. The decoder:

1. **Reads DC coefficient** — decoded differentially from previous block's DC
2. **Reads AC coefficients** — (run, value) pairs until end-of-block (EOB)
3. **Handles byte stuffing** — 0xFF 0x00 in data becomes just 0xFF

```text
Bitstream: [Huffman codes...]

DC decoding:
  1. Decode category (number of bits for the difference)
  2. Read that many bits for the actual difference
  3. Add to previous DC value

AC decoding:
  1. Decode (run, size) symbol
  2. If EOB (0,0): remaining coefficients are zero
  3. If ZRL (15,0): skip 16 zeros, continue
  4. Otherwise: skip 'run' zeros, read 'size' bits for next coefficient
```

The result is 64 quantized DCT coefficients in zigzag order.

### Step 3: Dequantization

The encoder divided DCT coefficients by quantization values and rounded. The decoder multiplies to recover approximate coefficients:

```text
Quantized:        [60, -2, 1, 0, 0, ...]
Quantization table: [16, 11, 10, ...]

Dequantized:
  60 × 16 = 960
  -2 × 11 = -22
   1 × 10 = 10
   0 × 16 = 0
   ...
```

**Note**: Information lost during quantization cannot be recovered. The decoder gets an approximation, not the original DCT coefficients.

### Step 4: Inverse DCT

The Inverse DCT (IDCT) transforms frequency coefficients back to spatial pixels:

```text
DCT coefficients (8×8):          Spatial pixels (8×8):
┌─────────────────────┐          ┌─────────────────────┐
│ 960 -22  10   0 ... │   IDCT   │ 128 130 132 134 ... │
│ -15   8   0   0 ... │ ───────▶ │ 129 131 133 135 ... │
│  ...                │          │  ...                │
└─────────────────────┘          └─────────────────────┘
```

The 2D IDCT is separable — apply 1D IDCT to rows, then to columns:

```text
IDCT formula: x[n] = Σ X[k] × cos(π(2n+1)k / 16) × c[k]
              k=0..7

where c[0] = 1/√2, c[k] = 1 for k > 0
```

### Step 5: Color Conversion

JPEG typically stores images in YCbCr color space. The decoder converts back to RGB:

```text
Y  = brightness (luma)
Cb = blue chrominance
Cr = red chrominance

R = Y + 1.402 × (Cr - 128)
G = Y - 0.344136 × (Cb - 128) - 0.714136 × (Cr - 128)
B = Y + 1.772 × (Cb - 128)
```

If chroma subsampling (4:2:0) was used, the decoder upsamples Cb and Cr to full resolution first.

### Step 6: Block Reassembly

The decoder processes the image in 8×8 blocks. The final step is assembling these blocks into the complete image, handling edge blocks that may extend beyond image boundaries.

## Bit-Level Reading

Both PNG and JPEG require reading individual bits from a byte stream, but with different conventions:

| Format      | Bit Order | Byte Stuffing    |
| ----------- | --------- | ---------------- |
| PNG/DEFLATE | LSB-first | None             |
| JPEG        | MSB-first | 0xFF 0x00 → 0xFF |

```rust,ignore
// From src/decode/bit_reader.rs

// DEFLATE: LSB-first, new bytes fill upper bits
self.bit_buf |= (self.data[self.pos] as u64) << self.bits_in_buf;

// JPEG: MSB-first, new bytes shift left
self.bit_buf = (self.bit_buf << 8) | (byte as u32);
```

## Common Pitfalls

### 1. Ignoring Checksums

Both formats include checksums (CRC-32 for PNG chunks, Adler-32 for zlib). Skipping verification seems faster, but means corrupt data produces garbage instead of errors.

**Recommendation**: Always verify checksums and report meaningful errors.

### 2. Assuming Valid Input

A robust decoder must handle:

- Truncated files (unexpected end-of-stream)
- Invalid Huffman codes (no matching symbol)
- Out-of-bounds back-references (distance > output length)
- Malformed chunk lengths

```rust,ignore
// From src/decode/inflate.rs
if distance > output.len() {
    return Err(Error::InvalidDecode("distance too far back".into()));
}
```

### 3. Forgetting Filter State

PNG filtering is row-by-row, but each row depends on the previous row (for Up, Average, Paeth filters). A decoder must maintain the previous row's unfiltered data.

### 4. Mishandling JPEG Restart Markers

JPEG can include restart markers (0xFFD0-0xFFD7) within scan data to enable error recovery and parallel decoding. Decoders must reset DC prediction at each restart.

### 5. Integer Overflow in Dequantization

Dequantized DCT coefficients can exceed the range of a single byte. Use appropriate integer types during IDCT, then clamp to 0-255 at the end.

## Implementation in pixo

The library includes decoding infrastructure in `src/decode/`:

- **`bit_reader.rs`** — `BitReader` for DEFLATE (LSB-first) and `MsbBitReader` for JPEG (MSB-first with byte stuffing)
- **`inflate.rs`** — Complete DEFLATE decompression with fixed and dynamic Huffman support

```rust,ignore
// From src/decode/inflate.rs
pub fn inflate_zlib(data: &[u8]) -> Result<Vec<u8>> {
    // Parse zlib header
    // Inflate DEFLATE blocks
    // Verify Adler-32 checksum
}
```

You can access decoding support through the CLI feature.

## Summary

Decoding completes the codec by reversing every encoding step:

| PNG Encoding                  | PNG Decoding                     |
| ----------------------------- | -------------------------------- |
| Filter (subtract predictions) | Unfilter (add predictions)       |
| DEFLATE (LZ77 + Huffman)      | INFLATE (reverse LZ77 + Huffman) |
| Write chunks with CRC         | Parse chunks, verify CRC         |

| JPEG Encoding     | JPEG Decoding         |
| ----------------- | --------------------- |
| RGB → YCbCr       | YCbCr → RGB           |
| Forward DCT       | Inverse DCT           |
| Quantize (divide) | Dequantize (multiply) |
| Huffman encode    | Huffman decode        |

Understanding both halves of a codec gives you:

1. **Deeper algorithm understanding** — Seeing the reverse illuminates the forward
2. **Debugging ability** — You can trace exactly where corruption or errors occur
3. **Transferable skills** — These same patterns appear in audio, video, and network protocols

## Next Steps

Return to [PNG Encoding](./png-encoding.md) or [JPEG Encoding](./jpeg-encoding.md) with fresh eyes, or explore [Performance Optimization](./performance-optimization.md) to see how both encoding and decoding can be accelerated.

---

## References

- [RFC 1951 - DEFLATE](https://www.rfc-editor.org/rfc/rfc1951) — Section 3.2.6 covers decompression
- [RFC 2083 - PNG](https://www.w3.org/TR/PNG/) — Chapter 9: Filtering, Chapter 10: Compression
- [ITU-T T.81 - JPEG](https://www.w3.org/Graphics/JPEG/itu-t81.pdf) — Annex F: Sequential DCT-based decoding
- See implementation: `src/decode/bit_reader.rs`, `src/decode/inflate.rs`
