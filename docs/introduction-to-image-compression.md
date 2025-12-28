# Introduction to Image Compression

## Why Compress Images?

Imagine you have a 4K photograph (3840 × 2160 pixels). Each pixel needs 3 bytes (one each for red, green, and blue). That's:

```text
3840 × 2160 × 3 = 24,883,200 bytes ≈ 24 MB
```

A single uncompressed photo would be 24 megabytes! At that size:

- A 256GB phone could only store about 10,000 photos
- Loading a webpage with 10 images would require downloading 240MB
- Streaming video would be completely impractical

Image compression solves this problem. The same 4K photo compressed as JPEG might be just 2-4 MB, or as a high-quality PNG around 8-12 MB. That's a 6-12x reduction!

## The Origins of Compression

Back in the 1840s, it was expensive to use a telegraph, so a fine gentleman named Samuel Morse came up with, you guessed it, Morse code. If you gave shorter codes to the most common letters, you could save some money! "E" is the most frequent in English, so it gets a single dot. "Q" is dash-dash-dot-dash.

This was clever engineering, but nobody knew how good compression could theoretically get. It wasn't until 1948 that Claude Shannon answered both questions in a paper called "A Mathematical Theory of Communication."

### Shannon's Big Idea: Entropy

Shannon asked a deceptively simple question: _How much information is actually in a message?_

His answer was **entropy** — a number that tells you the theoretical minimum bits needed to encode something. For English text, it's about 1.0-1.5 bits per character (not 8!). For a fair coin flip, it's exactly 1 bit. For a biased coin that lands heads 99% of the time? Just 0.08 bits — because it's so predictable.

**You cannot compress below entropy without losing information.** Shannon proved this mathematically. It's a law of the universe, like the speed of light. No algorithm, no matter how clever, can beat it.

### From Theory to Practice

After Shannon, getting to today took decades:

**1952: Huffman Coding** — David Huffman, as a _grad student_, invented optimal prefix codes. His algorithm assigns shorter codes to common symbols and longer codes to rare ones, approaching Shannon's limit. JPEG and DEFLATE still use this today.

**1977: LZ77** — Lempel and Ziv realized that repetition is everywhere. Instead of encoding "the cat sat on the mat" character by character, why not say "copy 4 bytes from 20 positions back"? This dictionary-based approach handles patterns that entropy coding alone can't see.

**1987: DEFLATE** — Phil Katz combined LZ77 with Huffman coding for his PKZIP format. This hybrid approach is still the backbone of PNG, ZIP, and gzip almost 40 years later.

### The Leap to Images

Text compression is one-dimensional — just a stream of characters. Images are two-dimensional grids of pixels, and that changes everything.

A 1920×1080 photo has over 2 million pixels. Uncompressed at 24 bits per pixel, that's 6 MB. But images have structure that text doesn't:

1. **Spatial redundancy** — neighboring pixels are usually similar. The sky doesn't change color every pixel.

2. **Perceptual redundancy** — your eyes can't see everything. High-frequency detail? Subtle color differences? Your brain glosses right over them.

PNG exploits the first type. Before compression, it runs "filters" that transform each row based on its neighbors. A smooth gradient might become a row of nearly identical values — perfect for DEFLATE.

JPEG exploits both. It converts pixels to frequency space (DCT), then _throws away_ the frequencies you can't see anyway. This is lossy compression — you're trading information for size. But the information you're losing? You'd never notice it was there.

### The Surprising Connection

Here's what's remarkable: Shannon's entropy formula from 1948 is still the benchmark we measure against today. When we say "DEFLATE achieves 2.5 bits per byte on this text," we're comparing to Shannon's theoretical limit.

And when JPEG decides which frequencies to discard, it's making a calculated bet about human perception — trading bits of _mathematical_ information for bits of _perceptual_ information. Shannon gave us the math. Decades of research gave us the insight into what humans actually need to see.

Every image you've ever loaded on the web is built on this foundation: Morse's intuition about frequent letters, Shannon's proof of fundamental limits, and 70+ years of researchers figuring out how to get as close to those limits as possible.

## Compression Is Everywhere

The techniques in this library power far more than just PNG and JPEG:

| Domain            | Technologies                         | Techniques Used                             |
| ----------------- | ------------------------------------ | ------------------------------------------- |
| Web               | gzip, Brotli, zstd                   | LZ77 + Huffman (same as PNG!)               |
| Network protocols | HTTP/2 HPACK, Protocol Buffers, gRPC | Dictionary coding, variable-length integers |
| Audio             | MP3, AAC, Opus                       | Transform coding + quantization (like JPEG) |
| Video             | H.264, H.265, AV1                    | Block transforms + motion prediction        |
| Databases         | Column stores, Parquet               | Dictionary + run-length encoding            |
| File systems      | ZFS, NTFS, Btrfs                     | Transparent LZ4/zstd compression            |

Notice the pattern: **the same core ideas appear everywhere**. LZ77 dictionary compression from 1977 lives inside your web browser (gzip), your file system, and PNG images. Transform coding from JPEG's DCT powers every video call you've ever made.

### Codecs: The Complete Picture

A **codec** (coder-decoder) is the complete system for encoding and decoding data. Every encoding algorithm needs a corresponding decoder:

```text
┌─────────────┐         compressed         ┌─────────────┐
│   Encoder   │ ──────────────────────────▶│   Decoder   │
│  (compress) │          data              │(decompress) │
└─────────────┘                            └─────────────┘
      ▲                                           │
      │         original data                     │
      └───────────────────────────────────────────┘
                    (lossless: identical)
                    (lossy: approximation)
```

See [Decoding](./decoding.md) for the decoder perspective.

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

```text
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

```text
Original row:     128, 130, 129, 131, 128, 132, 130, 129
Differences:           +2,  -1,  +2,  -3,  +4,  -2,  -1

The differences are smaller numbers that compress better!
```

**Algorithm**: LZ77 and PNG filters exploit this.

### 3. Frequency Redundancy

Natural images are dominated by low-frequency components (smooth gradients) with relatively few high-frequency components (sharp edges).

```text
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

## Shannon's Source Coding Theorem

As we saw earlier, Shannon proved that entropy sets the fundamental limit. His theorem states it precisely:

> _The average number of bits needed to represent symbols from a source cannot be less than the entropy of that source._

This has profound implications for image compression:

- **Lossless compression can never beat the entropy limit** — PNG can only get so small
- **To go smaller, you must lose information** — this is what JPEG does
- **Algorithms that approach the entropy limit are "optimal"** — Huffman coding achieves this for symbol-by-symbol encoding

## The pixo Pipeline

This library implements two complete encoding pipelines:

### PNG Pipeline (Lossless)

```text
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

```text
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
