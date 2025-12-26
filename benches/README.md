# Benchmarks

This directory contains performance benchmarks for the pixo library.

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

Comprehensive benchmark comparing pixo against all major alternatives:

- **All three presets**: Fast, Balanced, Max for PNG and JPEG
- **Lossy PNG**: Quantization comparison (pixo vs imagequant vs pngquant)
- **External tools**: oxipng, mozjpeg, pngquant (if installed)
- **Rust alternatives**: image crate, flate2, imagequant
- **Kodak images**: Real photographic images for realistic benchmarks
- **Summary tables**: WASM binary sizes, output sizes, timing

```bash
cargo bench --bench comparison
```

### components

Component-level micro-benchmarks for internal optimization work:

- LZ77 compression
- Huffman encoding (fixed and dynamic)
- PNG filters (Adaptive, AdaptiveFast)
- CRC32/Adler32 checksums
- DEFLATE variants (standard vs packed tokens)

Use this when optimizing specific compression components:

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

### pixo Lossy Mode

pixo supports lossy PNG compression via `QuantizationMode`:

```rust
use pixo::png::{PngOptions, QuantizationMode, QuantizationOptions};

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
