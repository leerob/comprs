# Benchmarks

This directory contains performance benchmarks for the comprs library.

**For comprehensive benchmark results, library comparisons, and recommendations, see [BENCHMARKS.md](BENCHMARKS.md).**

## Quick Start

```bash
# Run comprehensive comparison (prints summary table)
cargo bench --bench comparison

# Quick summary without full benchmarks
cargo bench --bench comparison -- --summary-only
```

## Benchmark Suites

### comparison (recommended)

Comprehensive benchmark comparing comprs against all major alternatives:

- **All three presets**: Fast, Balanced, Max for PNG and JPEG
- **Lossy PNG**: Quantization comparison (comprs vs imagequant vs pngquant)
- **External tools**: oxipng, mozjpeg, pngquant (if installed)
- **Rust alternatives**: image crate, flate2, imagequant
- **Summary tables**: WASM binary sizes, output sizes, timing

```bash
cargo bench --bench comparison
```

### encode_benchmark

Focused PNG/JPEG encoding benchmarks:

- Tests gradient images at 64x64, 128x128, 256x256, 512x512
- Compares comprs vs image crate
- Tests JPEG quality levels and subsampling options

```bash
cargo bench --bench encode_benchmark
```

### jpeg_mozjpeg

JPEG-specific comparison with mozjpeg:

- All three comprs presets vs mozjpeg
- Size and speed comparison
- Requires mozjpeg installed (`brew install mozjpeg`)

```bash
cargo bench --bench jpeg_mozjpeg
```

### deflate_micro

DEFLATE compression micro-benchmarks:

- Compares comprs vs flate2 on 1 MiB payloads
- Tests compressible and random data
- Reports throughput in bytes/second

```bash
cargo bench --bench deflate_micro
```

### components

Component-level benchmarks for internal algorithms:

- DCT (Discrete Cosine Transform)
- Huffman encoding/decoding
- LZ77 compression
- CRC32/Adler32 checksums

```bash
cargo bench --bench components
```

## Installing External Tools

For complete benchmarks, install the reference tools:

```bash
# macOS (Homebrew)
brew install oxipng mozjpeg pngquant

# Verify installation
oxipng --version
cjpeg -version
pngquant --version
```

## Lossy PNG Compression

The comparison benchmark includes lossy PNG compression using palette quantization:

### comprs Lossy Mode

comprs supports lossy PNG compression via `QuantizationMode`:

```rust
use comprs::png::{PngOptions, QuantizationMode, QuantizationOptions};

let mut opts = PngOptions::balanced();
opts.quantization = QuantizationOptions {
    mode: QuantizationMode::Auto,  // Auto-detect when beneficial
    max_colors: 256,               // Maximum palette size
    dithering: false,              // Floyd-Steinberg dithering
};

let png = png::encode_with_options(&pixels, width, height, ColorType::Rgb, &opts)?;
```

**Quantization Modes:**

- `Off`: Lossless PNG (default)
- `Auto`: Apply quantization when beneficial (moderate color count)
- `Force`: Always quantize RGB/RGBA images

### Comparison Tools

| Tool | Type | Description |
|------|------|-------------|
| comprs | Rust | Built-in median-cut quantization |
| imagequant | Rust crate | libimagequant bindings (high quality) |
| pngquant | CLI | Reference tool, excellent quality |

### Expected Results

Lossy PNG compression typically achieves 50-80% size reduction compared to lossless:

- **comprs**: Fast, good quality, ~70% smaller
- **imagequant**: Slower, excellent quality, ~75% smaller
- **pngquant**: Reference quality, ~75% smaller

## Benchmark Results

Results are saved to `target/criterion/` with HTML reports:

```bash
# Save baseline for comparison
cargo bench --bench comparison -- --save-baseline my-machine

# Compare against baseline
cargo bench --bench comparison -- --baseline my-machine
```

## Performance Tips

### For Benchmarking

1. **Use release mode**: `cargo bench` automatically uses release optimizations
2. **Close other applications**: Consistent hardware state improves accuracy
3. **Multiple runs**: Let Criterion collect enough samples for statistical significance

### For Production

1. **Reuse buffers**: Use `encode_into` variants to avoid allocations
2. **Choose appropriate quality**: JPEG 75-85 is usually sufficient
3. **Consider subsampling**: 4:2:0 reduces JPEG size by ~30-40%
4. **Match filter to content**: PNG `None` for noise, `Adaptive` for photos
5. **Use presets**: `fast()`, `balanced()`, `max()` are tuned for common use cases
