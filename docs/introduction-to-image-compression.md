# Introduction to Image Compression

## Why Compress Images?

Imagine you have a 4K photograph (3840 × 2160 pixels). Each pixel needs 3 bytes (one each for red, green, and blue). That's:

```
3840 × 2160 × 3 = 24,883,200 bytes ≈ 24 MB
```

A single uncompressed photo would be 24 megabytes! At that size:

- A 256GB phone could only store about 10,000 photos
- Loading a webpage with 10 images would require downloading 240MB
- Streaming video would be completely impractical

Image compression solves this problem. The same 4K photo compressed as JPEG might be just 2-4 MB, or as a high-quality PNG around 8-12 MB. That's a 6-12x reduction!

## The Two Fundamental Approaches

All image compression falls into two categories:

### Lossless Compression

**Definition**: The original image can be perfectly reconstructed from the compressed data.

**Example**: PNG format

**How it works**: Finds and eliminates redundancy without discarding any information.

**Use cases**:

- Screenshots (sharp text and UI elements)
- Medical imaging (every detail matters)
- Graphics and logos (precise edges)
- Archival storage (preserve original)

### Lossy Compression

**Definition**: Some information is permanently discarded to achieve smaller sizes.

**Example**: JPEG format

**How it works**: Removes information that humans are unlikely to notice.

**Use cases**:

- Photographs (natural images have noise anyway)
- Web images (bandwidth matters more than perfection)
- Social media (good enough is good enough)
- Video frames (temporal persistence hides artifacts)

## Understanding Redundancy

Compression works by exploiting **redundancy** — the observation that most data isn't random and contains predictable patterns. Let's explore the types of redundancy in images:

### 1. Statistical Redundancy

Not all colors appear equally often. In a photo of a blue sky, blue pixels vastly outnumber others.

```
If we assign shorter codes to common colors and longer codes to rare colors,
we can reduce the average bits per pixel.

Example:
  - Blue sky pixel: occurs 60% of time → assign 2-bit code
  - Cloud white:    occurs 30% of time → assign 3-bit code
  - Bird black:     occurs 10% of time → assign 4-bit code

Average: 0.6×2 + 0.3×3 + 0.1×4 = 2.5 bits vs 8 bits per pixel = 3.2x savings!
```

**Algorithm**: Huffman coding exploits this.

### 2. Spatial Redundancy

Adjacent pixels in an image are usually similar. A photo of a wall doesn't change color dramatically from pixel to pixel.

```
Original row:     128, 130, 129, 131, 128, 132, 130, 129
Differences:           +2,  -1,  +2,  -3,  +4,  -2,  -1

The differences are smaller numbers that compress better!
```

**Algorithm**: LZ77 and PNG filters exploit this.

### 3. Frequency Redundancy

Natural images are dominated by low-frequency components (smooth gradients) with relatively few high-frequency components (sharp edges).

```
Consider a gradient from dark to light:

Low frequency (slow change):  ████████████████
                              Dark → Light gradually

High frequency (fast change): ██  ██  ██  ██
                              Alternating pattern

Most of a photograph is low-frequency (sky, skin, walls).
We can use fewer bits for this dominant information.
```

**Algorithm**: DCT (Discrete Cosine Transform) in JPEG exploits this.

### 4. Perceptual Redundancy

Human vision has specific limitations:

- We're more sensitive to brightness changes than color changes
- We can't perceive very small differences
- We're less sensitive to details in rapidly changing regions

JPEG exploits all of these!

## The Information Theory Foundation

In 1948, Claude Shannon published "A Mathematical Theory of Communication" which laid the theoretical foundation for all data compression.

### Entropy: The Limit of Compression

**Entropy** measures the average amount of information per symbol. It sets the theoretical minimum bits needed to represent data.

For a source with symbols appearing with probabilities p₁, p₂, ..., pₙ:

```
H = -Σ pᵢ × log₂(pᵢ)
```

**Example**: A coin flip

- Fair coin (50/50): H = -0.5×log₂(0.5) - 0.5×log₂(0.5) = 1 bit
- Biased coin (90/10): H = -0.9×log₂(0.9) - 0.1×log₂(0.1) ≈ 0.47 bits

A biased coin has lower entropy — it's more predictable, so it can be compressed more!

### Shannon's Source Coding Theorem

> _The average number of bits needed to represent symbols from a source cannot be less than the entropy of that source._

This means:

- **Lossless compression can never beat the entropy limit**
- If we want to go smaller, we must use lossy compression
- Algorithms that approach the entropy limit are "optimal"

Huffman coding achieves optimality (to the nearest bit) for symbol-by-symbol encoding.

## The comprs Pipeline

This library implements two complete encoding pipelines:

### PNG Pipeline (Lossless)

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Raw Pixels  │───▶│  Filtering  │───▶│  DEFLATE    │───▶│  PNG File   │
│             │    │  (predict)  │    │ (compress)  │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                         │                   │
                         │                   ├── LZ77 (find repeats)
                         │                   └── Huffman (optimal codes)
                         │
                         ├── Sub (predict from left)
                         ├── Up (predict from above)
                         ├── Average (predict from avg)
                         └── Paeth (smart predictor)
```

### JPEG Pipeline (Lossy)

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Raw Pixels  │───▶│  Color      │───▶│    DCT      │───▶│ Quantize    │
│   (RGB)     │    │  Convert    │    │ (frequency) │    │ (lose info) │
└─────────────┘    │  (YCbCr)    │    └─────────────┘    └─────────────┘
                   └─────────────┘                              │
                                                                ▼
                   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
                   │  JPEG File  │◀───│  Huffman    │◀───│  Zigzag +   │
                   │             │    │  Encode     │    │  RLE        │
                   └─────────────┘    └─────────────┘    └─────────────┘
```

## Historical Context

Understanding the history helps appreciate why these algorithms exist:

| Year | Milestone                                                             |
| ---- | --------------------------------------------------------------------- |
| 1948 | Shannon publishes information theory — establishes theoretical limits |
| 1952 | Huffman develops optimal prefix codes (as a term paper!)              |
| 1977 | Lempel & Ziv publish LZ77 — dictionary compression                    |
| 1987 | DEFLATE combines LZ77 + Huffman — used in gzip, PNG, ZIP              |
| 1992 | JPEG standard published — lossy compression for photos                |
| 1996 | PNG standard published — lossless alternative to GIF                  |

The algorithms in this library represent 70+ years of computer science research, refined into elegant, practical implementations.

## Key Takeaways

1. **Compression exploits patterns**: Random data cannot be compressed; real data has structure.

2. **There's no free lunch**: Lossless compression has theoretical limits (entropy). To go smaller, you must lose information.

3. **Different images need different approaches**:

   - Photos → JPEG (lossy OK, natural noise hides artifacts)
   - Screenshots → PNG (sharp edges need lossless)

4. **Compression is layered**: Modern formats combine multiple techniques (filtering + dictionary + entropy coding).

5. **Perceptual tricks work**: JPEG's genius is knowing what humans can't see.

## Next Steps

Continue to [Introduction to Rust](./introduction-to-rust.md) to learn the language features that make this library possible, or jump directly to [Huffman Coding](./huffman-coding.md) if you're already familiar with Rust.

---

## References

- Shannon, C.E. (1948). "A Mathematical Theory of Communication"
- [RFC 1951 - DEFLATE](https://www.rfc-editor.org/rfc/rfc1951)
- [RFC 2083 - PNG](https://www.w3.org/TR/PNG/)
- [ITU-T T.81 - JPEG](https://www.w3.org/Graphics/JPEG/itu-t81.pdf)
