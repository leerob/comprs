# comprs

A minimal-dependency, high-performance image compression library written in Rust.

## Features

- **Zero runtime dependencies by default** - All compression algorithms implemented from scratch
- **PNG encoding** with all 5 filter types and hand-implemented DEFLATE compression
- **JPEG encoding** with DCT, quantization, and Huffman coding
- **High performance** - Optimized for speed with minimal allocations
- **Simple API** - Easy to use with sensible defaults

## Toolchain

The project builds and tests on **stable Rust 1.82+**. Dev-dependencies are pinned to avoid `edition2024` transitive pulls, so no nightly toolchain is required.

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
    compression_level: 9,  // validated: 1-9, higher = better compression
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
    restart_interval: None,         // Some(n) inserts DRI markers every n MCUs
};
let jpeg_data = jpeg::encode_with_options(&pixels, 1, 1, 85, ColorType::Rgb, &options).unwrap();
```

### Buffer reuse (PNG & JPEG)

Both encoders support writing into a caller-provided buffer to avoid repeated allocations when encoding multiple images in a loop:

```rust
// PNG
let mut png_buf = Vec::new();
png::encode_into(
    &mut png_buf,
    &pixels,
    3,
    1,
    ColorType::Rgb,
    &PngOptions::default(),
).unwrap();

// JPEG
let mut jpg_buf = Vec::new();
jpeg::encode_with_options_into(
    &mut jpg_buf,
    &pixels,
    3,
    1,
    85,
    ColorType::Rgb,
    &jpeg::JpegOptions {
        quality: 85,
        subsampling: jpeg::Subsampling::S444,
        restart_interval: None,
    },
).unwrap();
```

### Command-Line Interface

The library includes a CLI tool for quick image compression from the terminal. It only supports PNG and JPEG images in and out.

The CLI uses lightweight decoder crates (`png`, `jpeg-decoder`) instead of the full `image` crate.

#### Installation

```bash
# Install from source
cargo install --path . --features cli

# Or build locally
cargo build --release --features cli
```

#### Usage

```bash
# Basic usage - compress to JPEG (default)
comprs input.png -o output.jpg

# Compress to PNG with maximum compression
comprs input.jpg -o output.png -c 9

# JPEG with custom quality (1-100)
comprs photo.png -o photo.jpg -q 90

# JPEG with 4:2:0 chroma subsampling (smaller files)
comprs photo.png -o photo.jpg --subsampling s420

# PNG with specific filter strategy
comprs input.jpg -o output.png --filter paeth

# Adaptive fast (reduced trials) or sampled (every Nth row) strategies
comprs input.jpg -o output.png --filter adaptive-fast
comprs input.jpg -o output.png --filter adaptive-sampled --adaptive-sample-interval 8

# Convert to grayscale
comprs color.png -o gray.jpg --grayscale

# Verbose output with timing and size info
comprs input.png -o output.jpg -v
```

#### CLI Options

| Option              | Description                                                      | Default                    |
| ------------------- | ---------------------------------------------------------------- | -------------------------- |
| `-o, --output`      | Output file path                                                 | `<input>.compressed.<ext>` |
| `-f, --format`      | Output format (`png`, `jpeg`, `jpg`)                             | Detected from extension    |
| `-q, --quality`     | JPEG quality (1-100)                                             | 85                         |
| `-c, --compression` | PNG compression level (1-9)                                      | 6                          |
| `--subsampling`     | JPEG chroma subsampling (`s444`, `s420`)                         | s444                       |
| `--filter`          | PNG filter (`none`, `sub`, `up`, `average`, `paeth`, `adaptive`, `adaptive-fast`, `adaptive-sampled`) | adaptive                   |
| `--adaptive-sample-interval` | Rows between full adaptive evaluations when using `adaptive-sampled` | 4 |
| `--grayscale`       | Convert to grayscale                                             | false                      |
| `-v, --verbose`     | Show detailed output                                             | false                      |

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
   - Huffman coding (auto-selects fixed vs dynamic)
   - Stored-block fallback for incompressible data
   - Wrapped as zlib (RFC 1950) for PNG IDAT

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

- **Encoding benches**: compare PNG/JPEG against the `image` crate (`benches/encode_benchmark.rs`) for size and throughput.
- **Deflate microbench**: compare `deflate_zlib` against `flate2` on compressible and random 1 MiB payloads (`benches/deflate_micro.rs`), reporting throughput in bytes.

Run all benches:

```bash
cargo bench
```

Run a specific bench:

```bash
cargo bench --bench encode_benchmark
cargo bench --bench deflate_micro
```

## Testing

Run the full suite:

```bash
cargo test
```

Coverage highlights:

- Unit tests for bits/LZ77/Huffman/CRC/Adler and codec internals.
- Property-based tests for PNG/JPEG decode/roundtrip robustness.
- Structural checks (CRC/length for PNG chunks; marker ordering/DRI for JPEG).
- Conformance harnesses: PngSuite and libjpeg-turbo corpus (skip gracefully offline).

Note: tests and benches are validated on nightly toolchain; ensure `rustup override set nightly` in this repo to align with CI/tooling.

## Optional Features

- `cli` - Build the command-line interface (adds `clap`, `png`, `jpeg-decoder`)
- `simd` *(default)* - Enable SIMD optimizations with runtime feature detection (falls back to scalar paths when unavailable)
- `parallel` *(default)* - Enable parallel processing with rayon

### Performance defaults (PNG)
- Default compression level: **2** (favor speed)
- Default filter strategy: **AdaptiveFast**, with height-aware sampling on tall images
- SIMD + parallel enabled by default
- Small inputs: prefer fixed Huffman (token thresholds tuned to avoid double encoding)

Presets:
- `PngOptions::fast()` — level 2, AdaptiveFast (default)
- `PngOptions::balanced()` — level 6, Adaptive
- `PngOptions::max_compression()` — level 9, AdaptiveSampled(interval=2)
## Performance Notes

- PNG compression uses adaptive filter selection for best compression
- JPEG encoder processes images in 8×8 blocks for cache efficiency
- All algorithms minimize allocations during encoding

## Documentation

We provide comprehensive documentation explaining the algorithms and compression strategies used in this library. These guides are designed to be accessible to developers who may not be familiar with low-level compression details.

### Getting Started

- **[Documentation Index](./docs/README.md)** — Start here for an overview and reading guide
- **[Introduction to Image Compression](./docs/introduction-to-image-compression.md)** — Why and how we compress images

### Core Compression Algorithms

- **[Huffman Coding](./docs/huffman-coding.md)** — Optimal variable-length codes based on symbol frequency
- **[LZ77 Compression](./docs/lz77-compression.md)** — Dictionary-based compression with sliding windows
- **[DEFLATE Algorithm](./docs/deflate.md)** — How LZ77 and Huffman combine for powerful compression

### Image Format Documentation

- **[PNG Encoding](./docs/png-encoding.md)** — Lossless compression with predictive filtering
- **[JPEG Encoding](./docs/jpeg-encoding.md)** — Lossy compression pipeline overview
- **[Discrete Cosine Transform (DCT)](./docs/dct.md)** — The mathematical heart of JPEG
- **[JPEG Quantization](./docs/quantization.md)** — How JPEG achieves dramatic compression ratios

## License

MIT

## References

- [PNG Specification (RFC 2083)](https://www.w3.org/TR/PNG/)
- [DEFLATE Specification (RFC 1951)](https://www.rfc-editor.org/rfc/rfc1951)
- [JPEG Standard (ITU-T T.81)](https://www.w3.org/Graphics/JPEG/itu-t81.pdf)
