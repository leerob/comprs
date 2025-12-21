# comprs

A minimal-dependency, high-performance image compression library written in Rust.

## Features

- **Zero runtime dependencies by default** - All compression algorithms implemented from scratch
- **PNG encoding** with all 5 filter types and hand-implemented DEFLATE compression
- **JPEG encoding** with DCT, quantization, and Huffman coding
- **High performance** - Optimized for speed with minimal allocations
- **Simple API** - Easy to use with sensible defaults

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
comprs = "0.1"
```

## Usage

### PNG Encoding

```rust
use comprs::{png, ColorType};

// Encode RGB pixels as PNG
let pixels: Vec<u8> = vec![255, 0, 0, 0, 255, 0, 0, 0, 255]; // 3 RGB pixels
let png_data = png::encode(&pixels, 3, 1, ColorType::Rgb).unwrap();

// With custom options
use comprs::png::{PngOptions, FilterStrategy};

let options = PngOptions {
    compression_level: 9,  // 1-9, higher = better compression
    filter_strategy: FilterStrategy::Adaptive,
};
let png_data = png::encode_with_options(&pixels, 3, 1, ColorType::Rgb, &options).unwrap();
```

### JPEG Encoding

```rust
use comprs::jpeg;

// Encode RGB pixels as JPEG
let pixels: Vec<u8> = vec![255, 128, 64]; // 1 RGB pixel
let jpeg_data = jpeg::encode(&pixels, 1, 1, 85).unwrap(); // quality: 1-100

// With subsampling options (4:4:4 default, 4:2:0 available)
use comprs::jpeg::{JpegOptions, Subsampling};

let options = JpegOptions {
    quality: 85,
    subsampling: Subsampling::S420, // downsample chroma for smaller files
};
let jpeg_data = jpeg::encode_with_options(&pixels, 1, 1, 85, ColorType::Rgb, &options).unwrap();
```

### Supported Color Types

- `ColorType::Gray` - Grayscale (1 byte/pixel)
- `ColorType::GrayAlpha` - Grayscale + Alpha (2 bytes/pixel)
- `ColorType::Rgb` - RGB (3 bytes/pixel)
- `ColorType::Rgba` - RGBA (4 bytes/pixel)

Note: JPEG only supports `Gray` and `Rgb` color types.

### Performance toggles

- `parallel` feature enables rayon-powered parallel PNG adaptive filtering.
- `simd` (planned) is reserved for future SIMD acceleration paths.

## Architecture

The library is organized into modular components:

```
comprs/
├── src/
│   ├── lib.rs          # Public API
│   ├── color.rs        # Color types and conversions (RGB → YCbCr)
│   ├── bits.rs         # Bit-level I/O utilities
│   ├── error.rs        # Error types
│   ├── png/
│   │   ├── mod.rs      # PNG encoder
│   │   ├── filter.rs   # PNG filtering (None, Sub, Up, Average, Paeth)
│   │   └── chunk.rs    # PNG chunk formatting
│   ├── jpeg/
│   │   ├── mod.rs      # JPEG encoder
│   │   ├── dct.rs      # Discrete Cosine Transform
│   │   ├── quantize.rs # Quantization tables
│   │   └── huffman.rs  # JPEG Huffman encoding
│   └── compress/
│       ├── deflate.rs  # DEFLATE algorithm
│       ├── lz77.rs     # LZ77 compression
│       ├── huffman.rs  # Huffman coding
│       └── crc32.rs    # CRC32 checksum
```

## Algorithms Implemented

### PNG (Lossless)

1. **PNG Filtering** - All 5 filter types for optimal compression
   - None, Sub, Up, Average, Paeth predictor
   - Adaptive selection per scanline

2. **DEFLATE Compression** (RFC 1951)
   - LZ77 with 32KB sliding window and hash chains
   - Huffman coding (fixed codes)
   - Configurable compression levels

3. **CRC32 Checksums** - Table-based for speed

### JPEG (Lossy)

1. **Color Space Conversion** - RGB to YCbCr (ITU-R BT.601)

2. **8×8 Block DCT** - Separable 2D Discrete Cosine Transform

3. **Quantization** - Standard JPEG tables with quality scaling

4. **Entropy Coding**
   - Zigzag scan ordering
   - Run-length encoding of AC coefficients
   - DPCM for DC coefficients
   - Standard Huffman tables

## Benchmarks

Run benchmarks comparing against the `image` crate:

```bash
cargo bench
```

## Testing

Run the test suite:

```bash
cargo test
```

The library includes:
- Unit tests for all algorithms
- PNG conformance tests
- JPEG conformance tests

## Optional Features

- `simd` - Enable SIMD optimizations (requires nightly for some platforms)
- `parallel` - Enable parallel processing with rayon

## Performance Notes

- PNG compression uses adaptive filter selection for best compression
- JPEG encoder processes images in 8×8 blocks for cache efficiency
- All algorithms minimize allocations during encoding

## License

MIT

## References

- [PNG Specification (RFC 2083)](https://www.w3.org/TR/PNG/)
- [DEFLATE Specification (RFC 1951)](https://www.rfc-editor.org/rfc/rfc1951)
- [JPEG Standard (ITU-T T.81)](https://www.w3.org/Graphics/JPEG/itu-t81.pdf)
