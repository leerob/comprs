# Comprehensive Benchmark Report

Last updated: December 2025

This document provides a comprehensive comparison of comprs against other image compression tools, helping you make informed decisions based on your specific requirements.

## Environment

- **Host**: Apple Silicon (M-series) / x86_64
- **Toolchain**: rustc 1.88.0 (release) for comprs
- **External binaries** (for reference comparisons):
  - oxipng: Homebrew install, `-o4 --strip safe`
  - mozjpeg cjpeg: Homebrew install, `-quality 85 -optimize -progressive`
  - pngquant: Homebrew install, `--quality=65-80 --speed=4` (lossy PNG)
- **Benchmark command**: `cargo bench --bench comparison`

---

## Table of Contents

1. [PNG Lossless Compression](#1-png-lossless-compression)
2. [PNG Lossy Compression (Quantization)](#2-png-lossy-compression-quantization)
3. [JPEG Compression Comparison](#3-jpeg-compression-comparison)
4. [WASM Binary Size Comparison](#4-wasm-binary-size-comparison)
5. [Rust Library Ecosystem](#5-rust-library-ecosystem)
6. [JavaScript/Node.js Library Ecosystem](#6-javascriptnodejs-library-ecosystem)
7. [Recommendations: When to Use Which Tool](#7-recommendations-when-to-use-which-tool)

---

## 1. PNG Lossless Compression

Comparing comprs presets against oxipng and the image crate. All columns show **size / time**.

| Image                       | Dimensions | comprs Fast      | comprs Balanced  | comprs Max      | oxipng           | image crate   | Delta vs oxipng |
| --------------------------- | ---------- | ---------------- | ---------------- | --------------- | ---------------- | ------------- | --------------- |
| playground.png              | 1920×1080  | 1,475,576 / 0.4s | 1,340,919 / 0.2s | 1,332,458 / 77s | 1,134,213 / 2.1s | ~1.4MB / 0.3s | +17.5%          |
| squoosh_example.png         | 1920×1280  | 2,366,900 / 0.2s | 1,928,383 / 0.4s | 1,859,691 / 41s | 1,633,408 / 1.8s | ~2.0MB / 0.4s | +13.9%          |
| squoosh_example_palette.png | 800×600    | 268,636 / 48ms   | 147,626 / 45ms   | 144,855 / 2.8s  | 104,206 / 0.9s   | ~180KB / 50ms | +39.0%          |
| rocket.png                  | 800×600    | 1,716,340 / 0.1s | 1,390,853 / 0.2s | 1,379,515 / 15s | 1,280,518 / 1.2s | ~1.5MB / 0.2s | +7.7%           |

### PNG Preset Summary

| Preset       | Avg Size vs oxipng | Speed Characteristic       | Best For                   |
| ------------ | ------------------ | -------------------------- | -------------------------- |
| **Fast**     | +30-160% larger    | 5-20× faster than oxipng   | Development, previews      |
| **Balanced** | +8-42% larger      | 4-10× faster than oxipng   | Production, general use    |
| **Max**      | +8-39% larger      | Much slower (optimal LZ77) | Maximum compression needed |

### PNG Settings Footnotes

| Tool                | Settings                                                                         |
| ------------------- | -------------------------------------------------------------------------------- |
| comprs Fast [1]     | level=2, AdaptiveFast filter, no optimizations                                   |
| comprs Balanced [2] | level=6, Adaptive filter, alpha_opt, reduce_color, reduce_palette, strip_meta    |
| comprs Max [3]      | level=9, MinSum filter, optimal LZ77 parsing, iterative Huffman, block splitting |
| oxipng [4]          | `-o4 --strip safe` (Homebrew install)                                            |
| image crate [5]     | Default PngEncoder settings                                                      |

---

## 2. PNG Lossy Compression (Quantization)

Lossy PNG compression reduces file size by limiting the color palette to 256 colors (8-bit indexed PNG). This provides **significant size reductions (50-80%)** for photographic or complex images while maintaining PNG's lossless transparency support.

### Real Image Comparison

Testing on actual images from the test fixtures:

| Image            | Dimensions | comprs Lossy | pngquant | Delta | Winner |
| ---------------- | ---------- | ------------ | -------- | ----- | ------ |
| avatar-color.png | 740×740    | 135 KB       | 113 KB   | +19%  | pngquant |
| rocket.png       | 1376×768   | 281 KB       | 393 KB   | **-28%** | **comprs** |

**Key findings:**
- On images with solid colors/flat areas (rocket.png), **comprs wins by 28%**
- On complex photographic images, pngquant's libimagequant produces smaller files
- Both achieve **50-80% reduction** compared to lossless PNG
- comprs has zero external dependencies (143 KB WASM vs pngquant's native binary)

### Synthetic Benchmark (512×512 gradient)

Gradient images are a **worst-case scenario** for quantization because they contain many unique colors that require dithering, making compression less effective.

| Encoder          | Size     | Time     | Notes                                    |
| ---------------- | -------- | -------- | ---------------------------------------- |
| comprs Lossless  | 7.6 KB   | 13.0 ms  | Baseline (no quantization)               |
| comprs Lossy     | 6.3 KB   | 95.2 ms  | 256 colors, no dithering (-17%)          |
| imagequant       | 64.2 KB  | 47.9 ms  | libimagequant (dithered, larger)         |
| pngquant         | 61.6 KB  | 237.2 ms | --quality=65-80 (dithered, larger)       |

> **Note**: On gradient images, the dithering applied by imagequant/pngquant creates noise patterns that are harder to compress with DEFLATE. comprs's simpler median-cut without dithering produces better results for this edge case.

### When to Use Lossy PNG

| Scenario                          | Recommendation                                              |
| --------------------------------- | ----------------------------------------------------------- |
| **Photographic images**           | Use lossy - 50-80% smaller than lossless                    |
| **Images with flat colors/UI**    | comprs Lossy often beats pngquant                           |
| **Complex photos, max compression** | pngquant produces smaller files                           |
| **Icons and logos (<256 colors)** | Use lossless - already optimized                            |
| **WASM bundle size matters**      | comprs Lossy (no external deps, 143 KB WASM)                |

### Lossy PNG Settings

| Tool             | Settings                                                                |
| ---------------- | ----------------------------------------------------------------------- |
| comprs Lossy     | median-cut quantization, 256 colors, optional Floyd-Steinberg dithering |
| pngquant         | `--quality=65-80 --speed=4` (libimagequant internally)                  |
| imagequant       | Rust bindings to libimagequant library                                  |

---

## 3. JPEG Compression Comparison

Comparing comprs presets against mozjpeg and the image crate. All columns show **size / time**.

| Image           | Dimensions | comprs Fast     | comprs Balanced | comprs Max      | mozjpeg          | image crate     | Delta vs mozjpeg |
| --------------- | ---------- | --------------- | --------------- | --------------- | ---------------- | --------------- | ---------------- |
| multi-agent.jpg | 2300×1342  | 435.9KB / 94ms  | 435.9KB / 181ms | 368.0KB / 251ms | 352.3KB / ~200ms | ~480KB / ~100ms | **+4.4%**        |
| browser.jpg     | 2300×1342  | 383.4KB / 94ms  | 383.4KB / 179ms | 309.7KB / 253ms | 297.2KB / ~200ms | ~420KB / ~100ms | **+4.2%**        |
| review.jpg      | 2300×1342  | 405.9KB / 94ms  | 405.9KB / 181ms | 334.3KB / 251ms | 317.9KB / ~200ms | ~450KB / ~100ms | **+5.2%**        |
| web.jpg         | 3220×1812  | 664.3KB / 177ms | 664.3KB / 339ms | 547.1KB / 474ms | 518.5KB / ~350ms | ~730KB / ~180ms | **+5.5%**        |

### JPEG Preset Summary

| Preset       | Avg Size vs mozjpeg | Speed Characteristic          | Best For                 |
| ------------ | ------------------- | ----------------------------- | ------------------------ |
| **Fast**     | +24-29% larger      | 2× faster, no optimization    | Development, real-time   |
| **Balanced** | +24-29% larger      | ~1× similar, Huffman opt      | General use              |
| **Max**      | **+4.2% to +5.5%**  | Similar speed, full opt stack | Production, best quality |

### JPEG Settings Footnotes

| Tool                | Settings                                                               |
| ------------------- | ---------------------------------------------------------------------- |
| comprs Fast [1]     | quality 85, 4:4:4 subsampling, baseline DCT, no optimization           |
| comprs Balanced [2] | quality 85, 4:4:4 subsampling, Huffman optimization                    |
| comprs Max [3]      | quality 85, 4:2:0 subsampling, progressive, trellis quant, Huffman opt |
| mozjpeg [4]         | `cjpeg -quality 85 -optimize -progressive` (Homebrew install)          |
| image crate [5]     | quality 85, default settings                                           |

---

## 4. WASM Binary Size Comparison

Critical for web applications where bundle size impacts load time.

| Library         | WASM Size  | Notes                    |
| --------------- | ---------- | ------------------------ |
| **comprs**      | **143 KB** | Zero deps, pure Rust [1] |
| wasm-mozjpeg    | ~208 KB   | Emscripten compiled      |
| squoosh oxipng  | ~625 KB   | Google's Squoosh codec   |
| squoosh mozjpeg | ~803 KB   | Google's Squoosh codec   |
| image crate     | ~6-10 MB  | Many codecs included     |

### Binary Size Footnotes

[1] comprs build configuration:

```toml
[profile.release]
lto = true           # Link-time optimization
opt-level = "z"      # Optimize for size
codegen-units = 1    # Single codegen unit
panic = "abort"      # Remove unwinding code
strip = true         # Strip symbols
```

---

## 5. Rust Library Ecosystem

Comparison of Rust image compression libraries.

| Library           | WASM-friendly   | Binary Size  | Throughput | Notes                                    |
| ----------------- | --------------- | ------------ | ---------- | ---------------------------------------- |
| **comprs**        | Yes             | 143 KB       | Good       | Zero deps, pure Rust, simple WASM target |
| `image`           | Yes             | ~2-4 MB      | Good       | Pure Rust, many codecs included          |
| `photon-rs`       | Yes             | ~200-400 KB  | Excellent  | Pure Rust, designed for WASM [2]         |
| `zune-image`      | Yes             | ~500 KB-1 MB | Excellent  | Pure Rust, SIMD optimized [3]            |
| `mozjpeg`         | Emscripten only | ~30-50 KB    | Excellent  | C library, complex build setup           |
| `libpng` bindings | Emscripten only | N/A          | Excellent  | C library, requires Emscripten           |

### Rust Library Footnotes

- [2] photon-rs: Gaussian blur 180ms vs 2400ms in pure JS (13× faster)
- [3] zune-png: 1.8× speedup over libpng on x86 (Phoronix benchmarks)

---

## 6. JavaScript/Node.js Library Ecosystem

Comparison of JavaScript and Node.js image compression options.

| Library                     | Environment    | Bundle Size     | Throughput        | Notes                                          |
| --------------------------- | -------------- | --------------- | ----------------- | ---------------------------------------------- |
| `sharp`                     | Node.js only   | 7-12 MB native  | Excellent         | libvips bindings, 4-5× faster than ImageMagick |
| `jimp`                      | Browser + Node | ~4 MB unpacked  | Slow (~0.7 img/s) | Pure JS, no native deps                        |
| `pngjs`                     | Browser + Node | ~180 KB gzipped | Slow              | PNG only, pure JS                              |
| `jpeg-js`                   | Browser + Node | ~35 KB gzipped  | Slow              | JPEG only, pure JS                             |
| `browser-image-compression` | Browser        | ~50 KB gzipped  | Varies            | Uses Canvas API internally                     |
| `squoosh-lib`               | Browser + Node | 30-100 KB/codec | Good              | Google's WASM codecs                           |

### Browser Native APIs

| API                  | Binary Size | Quality Control | Format Support  |
| -------------------- | ----------- | --------------- | --------------- |
| Canvas `toBlob()`    | 0           | Limited         | JPEG, PNG, WebP |
| Canvas `toDataURL()` | 0           | Limited         | JPEG, PNG, WebP |
| `OffscreenCanvas`    | 0           | Limited         | JPEG, PNG, WebP |

**Limitations of browser Canvas APIs:**

- JPEG quality is browser-dependent and not standardized
- No control over PNG compression level or filter selection
- No access to advanced features (progressive JPEG, interlacing)
- Output varies between browsers

---

## 7. Recommendations: When to Use Which Tool

### Decision Matrix by Primary Constraint

| If you need...             | PNG                   | JPEG                  | Why                                  |
| -------------------------- | --------------------- | --------------------- | ------------------------------------ |
| Smallest WASM binary       | comprs (143 KB)       | comprs (143 KB)       | 2-10× smaller than alternatives      |
| Best lossless compression  | oxipng                | N/A                   | Gold standard, but larger binaries   |
| Best lossy PNG compression | comprs Lossy/pngquant | N/A                   | 50-80% smaller than lossless         |
| Fastest encoding           | comprs Fast or image  | comprs Fast           | Minimal overhead                     |
| Best speed/size tradeoff   | comprs Balanced       | comprs Balanced       | Good compression, fast enough        |
| Browser + Node support     | comprs, pngjs, jimp   | comprs, jpeg-js, jimp | Pure JS/WASM, no native deps         |
| Node.js only, max perf     | sharp                 | sharp                 | Native libvips, fastest              |
| Zero dependencies          | comprs                | comprs                | Pure Rust, no C toolchain            |

### Quick Decision Guide

| Scenario                                     | Recommendation                                          |
| -------------------------------------------- | ------------------------------------------------------- |
| **Building a web app with WASM?**            | Use comprs (143 KB binary, good compression)            |
| **Need smallest PNG file size?**             | Use comprs Lossy (50-80% smaller than lossless)         |
| **CLI tool, size doesn't matter?**           | Use oxipng/mozjpeg/pngquant (best compression ratios)   |
| **Node.js server, need speed?**              | Use sharp (native bindings, excellent performance)      |
| **Pure browser, no WASM?**                   | Use Canvas API (0 KB) or pngjs/jpeg-js (slow but works) |
| **Need predictable output across browsers?** | Use comprs (identical output everywhere)                |
| **Optimizing existing images in CI/CD?**     | Use oxipng/mozjpeg/pngquant CLI tools                   |

### comprs Preset Selection Guide

| Use Case                   | PNG Preset         | JPEG Preset |
| -------------------------- | ------------------ | ----------- |
| Development/previews       | Fast               | Fast        |
| Production web assets      | Balanced or Lossy  | Max         |
| Bandwidth-critical apps    | Lossy              | Max         |
| Real-time processing       | Fast               | Fast        |
| Static site generation     | Balanced or Max    | Max         |
| Photographic web images    | **Lossy** (50-80% smaller) | Max   |
| Icons/logos/UI assets      | Lossless (Balanced)| N/A         |

---

## Running Benchmarks

Generate fresh benchmark data on your hardware:

```bash
# Run comprehensive comparison (prints summary table)
cargo bench --bench comparison

# Run PNG/JPEG encoding benchmarks
cargo bench --bench encode_benchmark

# Run JPEG-only comparison with mozjpeg
cargo bench --bench jpeg_mozjpeg

# Run component-level micro-benchmarks
cargo bench --bench components

# Quick summary without full benchmarks
cargo bench --bench comparison -- --summary-only
```

Results are saved to `target/criterion/` with HTML reports.

---

## Data Sources

- **npm packages**: npmjs.com package pages (unpacked sizes)
- **Bundlephobia**: bundlephobia.com for minified+gzipped bundle sizes
- **sharp**: sharp.pixelplumbing.com documentation
- **Squoosh**: squoosh-browser-sense package, wasm-mozjpeg project
- **zune-png**: Phoronix benchmarks (1.8× faster than libpng)
- **photon-rs**: Project documentation (13× speedup for Gaussian blur vs JS)
- **jimp**: skypack.dev benchmarks (~0.7 img/s vs sharp's ~11 img/s)

_Sizes are approximate and vary by version, build configuration, and optimization settings._
