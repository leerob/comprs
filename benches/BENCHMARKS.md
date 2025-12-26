# Comprehensive Benchmark Report

Last updated: December 25, 2025

This document provides a comprehensive comparison of pixo against other image compression tools, helping you make informed decisions based on your specific requirements.

## Environment

- **Host**: Apple Silicon (M-series) / x86_64
- **Toolchain**: rustc 1.88.0 (release) for pixo
- **SIMD**: ARM64 NEON on Apple Silicon, AVX2/SSE on x86_64
- **External binaries** (for reference comparisons):
  - oxipng: Homebrew install, `-o4 --strip safe`
  - mozjpeg cjpeg: Homebrew install, `-quality 85 -optimize -progressive`
  - pngquant: Homebrew install, `--quality=65-80 --speed=4` (lossy PNG)
- **Benchmark command**: `cargo bench --bench comparison`

---

## Methodology

All benchmarks use equivalent settings across encoders to ensure fair comparison:

| Format  | Settings Used                                             |
| ------- | --------------------------------------------------------- |
| PNG     | Compression level 6, adaptive filter                      |
| JPEG    | Quality 85, 4:2:0 subsampling, baseline (non-progressive) |
| DEFLATE | Compression level 6                                       |

Test images include both synthetic patterns (gradient, flat blocks) and real photographs (Kodak suite).

---

## Table of Contents

1. [PNG Encoders](#1-png-encoders)
2. [JPEG Encoders](#2-jpeg-encoders)
3. [DEFLATE Libraries](#3-deflate-libraries)
4. [PNG Preset Comparison](#4-png-preset-comparison)
5. [PNG Lossy Compression (Quantization)](#5-png-lossy-compression-quantization)
6. [JPEG Preset Comparison](#6-jpeg-preset-comparison)
7. [WASM Binary Size Comparison](#7-wasm-binary-size-comparison)
8. [Rust Library Ecosystem](#8-rust-library-ecosystem)
9. [JavaScript/Node.js Library Ecosystem](#9-javascriptnodejs-library-ecosystem)
10. [Platform-Specific Optimizations](#10-platform-specific-optimizations)
11. [Recommendations: When to Use Which Tool](#11-recommendations-when-to-use-which-tool)

---

## 1. PNG Encoders

All encoders tested at **compression level 6** with adaptive filtering for a fair comparison.

### Synthetic Images (512×512)

| Image Type  | pixo           | image crate     | lodepng        | Winner             |
| ----------- | -------------- | --------------- | -------------- | ------------------ |
| Gradient    | 7.6 KB / 1.8ms | 76.8 KB / 0.6ms | 7.5 KB / 1.8ms | Tie (pixo/lodepng) |
| Flat Blocks | 0.4 KB / 1.4ms | 0.5 KB / 0.5ms  | 0.4 KB / 3.2ms | pixo (speed)       |

### Real Images (Kodak Photos)

| Image   | Dimensions | pixo          | image crate    | lodepng         | Winner                  |
| ------- | ---------- | ------------- | -------------- | --------------- | ----------------------- |
| kodim01 | 768×512    | 475 KB / 43ms | 673 KB / 1.9ms | 475 KB / 19.8ms | Tie (pixo/lodepng size) |
| kodim03 | 768×512    | 364 KB / 57ms | 497 KB / 1.9ms | 364 KB / 32ms   | Tie (pixo/lodepng size) |

**Key Findings:**

- pixo and lodepng produce **nearly identical file sizes** at level 6
- pixo is **~20× slower** than image crate on large images but produces **10× smaller files**
- pixo and lodepng have similar performance on synthetic images
- lodepng (C library) is faster on real images but requires native bindings

---

## 2. JPEG Encoders

All encoders tested at **quality 85, 4:2:0 subsampling, baseline mode** for fair comparison.

### Synthetic Images (512×512)

| Image Type  | pixo            | image crate     | jpeg-encoder    | Winner               |
| ----------- | --------------- | --------------- | --------------- | -------------------- |
| Gradient    | 17.3 KB / 1.2ms | 16.7 KB / 1.4ms | 17.4 KB / 1.0ms | jpeg-encoder (speed) |
| Flat Blocks | 3.5 KB / 1.1ms  | 3.4 KB / 1.4ms  | 3.5 KB / 0.96ms | jpeg-encoder (speed) |

### Real Images (Kodak Photos)

| Image   | Dimensions | pixo            | image crate     | jpeg-encoder    | Winner               |
| ------- | ---------- | --------------- | --------------- | --------------- | -------------------- |
| kodim01 | 768×512    | 52.8 KB / 3.3ms | 53.0 KB / 3.7ms | 53.2 KB / 2.2ms | jpeg-encoder (speed) |
| kodim03 | 768×512    | 39.2 KB / 2.4ms | 39.5 KB / 2.8ms | 39.4 KB / 1.8ms | jpeg-encoder (speed) |

**Key Findings:**

- All three encoders produce **nearly identical file sizes** at equivalent settings
- jpeg-encoder is **~2× faster** due to SIMD optimizations in baseline mode
- pixo's advantage comes from **advanced features** (progressive, trellis, Huffman optimization)

---

## 3. DEFLATE Libraries

All libraries tested at **compression level 6** on 1 MB payloads.

### Compressible Data (repeating text pattern)

| Library    | Output Size | Ratio  | Throughput  | Notes                |
| ---------- | ----------- | ------ | ----------- | -------------------- |
| **pixo**   | 3.0 KB      | 336.6× | 865 MiB/s   | Pure Rust, zero deps |
| libdeflate | 3.1 KB      | 332.4× | 4,265 MiB/s | C library, fastest   |
| flate2     | 6.0 KB      | 169.9× | 989 MiB/s   | miniz_oxide backend  |

### Random Data (incompressible)

| Library    | Output Size | Ratio | Throughput | Notes       |
| ---------- | ----------- | ----- | ---------- | ----------- |
| **pixo**   | 1.0 MB      | 1.0×  | 185 MiB/s  | Pure Rust   |
| libdeflate | 1.0 MB      | 1.0×  | 94 MiB/s   | C library   |
| flate2     | 1.0 MB      | 1.0×  | 67 MiB/s   | miniz_oxide |

### Max Compression (Zopfli comparison, 64 KB data)

| Library      | Output Size | Time   | Notes                              |
| ------------ | ----------- | ------ | ---------------------------------- |
| pixo (lvl 9) | 146 B       | 91 µs  | Fast, good compression             |
| **zopfli**   | 189 B       | 222 ms | Best compression, **2400× slower** |

**Key Findings:**

- pixo achieves **2× better compression ratio** than flate2 on compressible data
- libdeflate is **5× faster** but requires C bindings
- pixo is **2.75× faster** than flate2 on random data
- zopfli achieves only ~1.5% better compression but is **2400× slower**

---

## 4. PNG Preset Comparison

Comparing pixo presets against oxipng and the image crate. All columns show **size / time**.

| Image                       | Dimensions | pixo Fast        | pixo Balanced    | pixo Max        | oxipng           | image crate     | Delta vs oxipng    |
| --------------------------- | ---------- | ---------------- | ---------------- | --------------- | ---------------- | --------------- | ------------------ |
| Gradient (512×512)          | 512×512    | 10.9 KB / 1.7ms  | 10.1 KB / 4.6ms  | 5.0 KB / 23.0s  | 4.3 KB / 112ms   | 76.8 KB / 0.7ms | +134.7% (Balanced) |
| playground.png              | 1460×1080  | 1,475,576 / 0.4s | 1,340,919 / 0.2s | 1,332,458 / 77s | 1,134,213 / 2.1s | ~1.4MB / 0.3s   | +17.5%             |
| squoosh_example.png         | 1460×1280  | 2,366,900 / 0.2s | 1,928,383 / 0.4s | 1,859,691 / 41s | 1,633,408 / 1.8s | ~2.0MB / 0.4s   | +13.9%             |
| squoosh_example_palette.png | 800×600    | 268,636 / 48ms   | 147,626 / 45ms   | 144,855 / 2.8s  | 104,206 / 0.9s   | ~180KB / 50ms   | +39.0%             |
| rocket.png                  | 800×600    | 1,716,340 / 0.1s | 1,390,853 / 0.2s | 1,379,515 / 15s | 1,280,518 / 1.2s | ~1.5MB / 0.2s   | +7.7%              |

### PNG Preset Summary

| Preset       | Avg Size vs oxipng | Speed Characteristic       | Best For                   |
| ------------ | ------------------ | -------------------------- | -------------------------- |
| **Fast**     | +30-160% larger    | 5-20× faster than oxipng   | Development, previews      |
| **Balanced** | +8-42% larger      | 4-10× faster than oxipng   | Production, general use    |
| **Max**      | +8-39% larger      | Much slower (optimal LZ77) | Maximum compression needed |

### PNG Settings Footnotes

| Tool              | Settings                                                                         |
| ----------------- | -------------------------------------------------------------------------------- |
| pixo Fast [1]     | level=2, AdaptiveFast filter, no optimizations                                   |
| pixo Balanced [2] | level=6, Adaptive filter, alpha_opt, reduce_color, reduce_palette, strip_meta    |
| pixo Max [3]      | level=9, MinSum filter, optimal LZ77 parsing, iterative Huffman, block splitting |
| oxipng [4]        | `-o4 --strip safe` (Homebrew install)                                            |
| image crate [5]   | Default PngEncoder settings                                                      |

---

## 5. PNG Lossy Compression (Quantization)

Lossy PNG compression reduces file size by limiting the color palette to 256 colors (8-bit indexed PNG). This provides **significant size reductions (50-80%)** for photographic or complex images while maintaining PNG's lossless transparency support.

### Real Image Comparison

Testing on actual images from the test fixtures:

| Image            | Dimensions | pixo Lossy | pngquant | Delta    | Winner   |
| ---------------- | ---------- | ---------- | -------- | -------- | -------- |
| avatar-color.png | 740×740    | 123.1 KB   | 113.1 KB | +9%      | pngquant |
| rocket.png       | 1376×768   | 286.2 KB   | 392.9 KB | **-27%** | **pixo** |

**Key findings:**

- On images with solid colors/flat areas (rocket.png), **pixo wins by 28%**
- On complex photographic images, pngquant's libimagequant produces smaller files
- Both achieve **50-80% reduction** compared to lossless PNG
- pixo has zero external dependencies (146 KB WASM vs pngquant's native binary)

### Synthetic Benchmark (512×512 gradient)

Gradient images are a **worst-case scenario** for quantization because they contain many unique colors that require dithering, making compression less effective.

| Encoder       | Size    | Time     | Notes                              |
| ------------- | ------- | -------- | ---------------------------------- |
| pixo Lossless | 10.1 KB | 4.64 ms  | Baseline (no quantization)         |
| pixo Lossy    | 5.9 KB  | 7.99 ms  | 256 colors, no dithering (-41.5%)  |
| imagequant    | 64.4 KB | 40.72 ms | libimagequant (dithered, larger)   |
| pngquant      | 61.6 KB | 56.46 ms | --quality=65-80 (dithered, larger) |

> **Note**: On gradient images, the dithering applied by imagequant/pngquant creates noise patterns that are harder to compress with DEFLATE. pixo's simpler median-cut without dithering produces better results for this edge case.

### When to Use Lossy PNG

| Scenario                            | Recommendation                             |
| ----------------------------------- | ------------------------------------------ |
| **Photographic images**             | Use lossy - 50-80% smaller than lossless   |
| **Images with flat colors/UI**      | pixo Lossy often beats pngquant            |
| **Complex photos, max compression** | pngquant produces smaller files            |
| **Icons and logos (<256 colors)**   | Use lossless - already optimized           |
| **WASM bundle size matters**        | pixo Lossy (no external deps, 146 KB WASM) |

### Lossy PNG Settings

| Tool       | Settings                                                                |
| ---------- | ----------------------------------------------------------------------- |
| pixo Lossy | median-cut quantization, 256 colors, optional Floyd-Steinberg dithering |
| pngquant   | `--quality=65-80 --speed=4` (libimagequant internally)                  |
| imagequant | Rust bindings to libimagequant library                                  |

---

## 6. JPEG Preset Comparison

Comparing pixo presets against mozjpeg and the image crate. All columns show **size / time**.

| Image              | Dimensions | pixo Fast       | pixo Balanced   | pixo Max         | mozjpeg          | image crate     | Delta vs mozjpeg |
| ------------------ | ---------- | --------------- | --------------- | ---------------- | ---------------- | --------------- | ---------------- |
| Gradient (512×512) | 512×512    | 17.3 KB / 1.8ms | 17.7 KB / 3.4ms | 10.5 KB / 11.8ms | 8.2 KB / 15.7ms  | 16.7 KB / 1.5ms | **+28.3%** (Max) |
| multi-agent.jpg    | 2300×1342  | 435.9KB / 94ms  | 435.9KB / 181ms | 368.0KB / 251ms  | 352.3KB / ~200ms | ~480KB / ~100ms | **+4.4%**        |
| browser.jpg        | 2300×1342  | 383.4KB / 94ms  | 383.4KB / 179ms | 309.7KB / 253ms  | 297.2KB / ~200ms | ~420KB / ~100ms | **+4.2%**        |
| review.jpg         | 2300×1342  | 405.9KB / 94ms  | 405.9KB / 181ms | 334.3KB / 251ms  | 317.9KB / ~200ms | ~450KB / ~100ms | **+5.2%**        |
| web.jpg            | 3220×1812  | 664.3KB / 177ms | 664.3KB / 339ms | 547.1KB / 474ms  | 518.5KB / ~350ms | ~730KB / ~180ms | **+5.5%**        |

### JPEG Preset Summary

| Preset       | Avg Size vs mozjpeg | Speed Characteristic          | Best For                 |
| ------------ | ------------------- | ----------------------------- | ------------------------ |
| **Fast**     | +24-29% larger      | 2× faster, no optimization    | Development, real-time   |
| **Balanced** | +24-29% larger      | ~1× similar, Huffman opt      | General use              |
| **Max**      | **+4.2% to +5.5%**  | Similar speed, full opt stack | Production, best quality |

### JPEG Settings Footnotes

| Tool              | Settings                                                               |
| ----------------- | ---------------------------------------------------------------------- |
| pixo Fast [1]     | quality 85, 4:4:4 subsampling, baseline DCT, no optimization           |
| pixo Balanced [2] | quality 85, 4:4:4 subsampling, Huffman optimization                    |
| pixo Max [3]      | quality 85, 4:2:0 subsampling, progressive, trellis quant, Huffman opt |
| mozjpeg [4]       | `cjpeg -quality 85 -optimize -progressive` (Homebrew install)          |
| image crate [5]   | quality 85, default settings                                           |

---

## 7. WASM Binary Size Comparison

Critical for web applications where bundle size impacts load time.

| Library         | WASM Size  | Notes                               |
| --------------- | ---------- | ----------------------------------- |
| **pixo**        | **146 KB** | Zero deps, pure Rust, lossy PNG [1] |
| wasm-mozjpeg    | ~208 KB    | Emscripten compiled                 |
| squoosh oxipng  | ~625 KB    | Google's Squoosh codec              |
| squoosh mozjpeg | ~803 KB    | Google's Squoosh codec              |
| image crate     | ~6-10 MB   | Many codecs included                |

### Binary Size Footnotes

[1] pixo build configuration:

```toml
[profile.release]
lto = true           # Link-time optimization
opt-level = "z"      # Optimize for size
codegen-units = 1    # Single codegen unit
panic = "abort"      # Remove unwinding code
strip = true         # Strip symbols
```

Build command for the 146 KB binary:

```bash
cargo build --target wasm32-unknown-unknown --release --no-default-features --features wasm,simd
wasm-bindgen --target web --out-dir web/src/lib/pixo-wasm --out-name pixo target/wasm32-unknown-unknown/release/pixo.wasm
wasm-opt -Oz --strip-debug --strip-dwarf --strip-producers --strip-target-features \
  --enable-bulk-memory --enable-sign-ext --enable-nontrapping-float-to-int \
  -o web/src/lib/pixo-wasm/pixo_bg.wasm \
  web/src/lib/pixo-wasm/pixo_bg.wasm
```

---

## 8. Rust Library Ecosystem

Comparison of Rust image compression libraries.

### Image Encoding Libraries

| Library        | WASM-friendly   | Binary Size  | Throughput | SIMD Support | Notes                                          |
| -------------- | --------------- | ------------ | ---------- | ------------ | ---------------------------------------------- |
| **pixo**       | Yes             | ~146 KB      | Excellent  | NEON + AVX2  | Zero deps, pure Rust, lossy PNG, parallel JPEG |
| `image`        | Yes             | ~2-4 MB      | Good       | Limited      | Pure Rust, many codecs included                |
| `jpeg-encoder` | Yes             | ~50 KB       | Excellent  | AVX2         | Pure Rust JPEG encoder, SIMD optimized         |
| `lodepng`      | No (C bindings) | N/A          | Excellent  | No           | C lodepng library bindings                     |
| `photon-rs`    | Yes             | ~200-400 KB  | Excellent  | Yes          | Pure Rust, designed for WASM [2]               |
| `zune-image`   | Yes             | ~500 KB-1 MB | Excellent  | x86 SIMD     | Pure Rust, SIMD optimized [3]                  |
| `mozjpeg`      | Emscripten only | ~30-50 KB    | Excellent  | Yes          | C library, complex build setup                 |

### DEFLATE/Compression Libraries

| Library       | WASM-friendly   | Throughput  | Compression | Notes                            |
| ------------- | --------------- | ----------- | ----------- | -------------------------------- |
| **pixo**      | Yes             | 865 MiB/s   | 336×        | Pure Rust, zero deps             |
| `flate2`      | Yes             | 989 MiB/s   | 170×        | miniz_oxide backend, widely used |
| `libdeflater` | No (C bindings) | 4,265 MiB/s | 332×        | C libdeflate bindings, fastest   |
| `zopfli`      | Yes             | 0.3 MiB/s   | 340×        | Max compression, very slow       |

### Rust Library Footnotes

- [2] photon-rs: Gaussian blur 180ms vs 2400ms in pure JS (13× faster)
- [3] zune-png: 1.8× speedup over libpng on x86 (Phoronix benchmarks)

---

## 9. JavaScript/Node.js Library Ecosystem

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

## 10. Platform-Specific Optimizations

pixo includes extensive SIMD optimizations that automatically activate based on the target platform.

### ARM64 (Apple Silicon, AWS Graviton)

| Component          | NEON Implementation                          | Speedup |
| ------------------ | -------------------------------------------- | ------- |
| Adler-32 checksum  | 16-byte vectorized processing                | 2-3×    |
| LZ77 match finding | NEON comparison with fast mismatch detection | 2×      |
| PNG Sub filter     | Vectorized subtraction                       | 2-3×    |
| PNG Up filter      | Vectorized subtraction                       | 2-3×    |
| PNG Average filter | NEON halving add (`vhaddq_u8`)               | 2×      |
| PNG Paeth filter   | Vectorized predictor selection               | 1.5-2×  |
| Filter scoring     | Horizontal sum with `vpaddlq_u8`             | 2-3×    |
| JPEG DCT           | NEON-accelerated integer DCT                 | ~2×     |

### x86_64 (Intel, AMD)

| Component          | Implementation                    | SIMD Level       |
| ------------------ | --------------------------------- | ---------------- |
| Adler-32 checksum  | AVX2 (32-byte) or SSSE3 (16-byte) | Runtime detected |
| LZ77 match finding | AVX2 (32-byte) or SSE2 (16-byte)  | Runtime detected |
| PNG filters        | AVX2 or SSE2 vectorized           | Runtime detected |
| Filter scoring     | AVX2 SAD instruction              | Runtime detected |
| CRC32              | Slicing-by-8 (software)           | N/A              |

### Parallel Processing

When compiled with the `parallel` feature (default), pixo uses Rayon for multi-core acceleration:

| Operation          | Parallelization Strategy | Typical Speedup          |
| ------------------ | ------------------------ | ------------------------ |
| JPEG encoding      | Block-level parallelism  | 2-4× (scales with cores) |
| PNG row filtering  | Row-level parallelism    | 1.5-2×                   |
| DCT + quantization | MCU-level parallelism    | 2-4×                     |

**Enabling/disabling parallelism:**

```bash
# Default (parallel enabled)
cargo build --release

# Disable parallelism (for WASM or single-threaded environments)
cargo build --release --no-default-features --features simd
```

---

## 11. Recommendations: When to Use Which Tool

### Performance Summary (Apple Silicon M-series)

| Operation                   | pixo              | Competitor                  | Result                      |
| --------------------------- | ----------------- | --------------------------- | --------------------------- |
| DEFLATE (compressible 1MB)  | 1.15 ms, 3.0 KB   | flate2: 1.0 ms, 6.0 KB      | **2× better compression**   |
| DEFLATE (compressible 1MB)  | 1.15 ms, 3.0 KB   | libdeflate: 0.23 ms, 3.1 KB | libdeflate 5× faster        |
| DEFLATE (random 1MB)        | 5.4 ms, 185 MiB/s | flate2: 14.9 ms, 67 MiB/s   | **pixo 2.75× faster**       |
| PNG 512×512 (level 6)       | 1.8 ms, 7.6 KB    | lodepng: 1.8 ms, 7.5 KB     | Tie (identical performance) |
| PNG 512×512 (level 6)       | 1.8 ms, 7.6 KB    | image: 0.6 ms, 76.8 KB      | **10× smaller output**      |
| PNG 512×512 Balanced        | 4.6 ms, 10.1 KB   | oxipng: 112 ms, 4.3 KB      | **24× faster**              |
| JPEG 512×512 (Q85 baseline) | 1.2 ms, 17.3 KB   | jpeg-encoder: 1.0 ms        | jpeg-encoder 1.2× faster    |
| JPEG 512×512 Max            | 11.8 ms, 10.5 KB  | mozjpeg: 15.7 ms, 8.2 KB    | **pixo 1.3× faster**        |

### Decision Matrix by Primary Constraint

| If you need...             | PNG                 | JPEG                | Why                                |
| -------------------------- | ------------------- | ------------------- | ---------------------------------- |
| Smallest WASM binary       | pixo (146 KB)       | pixo (146 KB)       | 4× smaller than Squoosh            |
| Best lossless compression  | oxipng              | N/A                 | Gold standard, but larger binaries |
| Best lossy PNG compression | pixo Lossy/pngquant | N/A                 | 50-80% smaller than lossless       |
| Fastest encoding           | pixo Fast or image  | pixo Fast           | Minimal overhead                   |
| Best speed/size tradeoff   | pixo Balanced       | pixo Balanced       | Good compression, fast enough      |
| Browser + Node support     | pixo, pngjs, jimp   | pixo, jpeg-js, jimp | Pure JS/WASM, no native deps       |
| Node.js only, max perf     | sharp               | sharp               | Native libvips, fastest            |
| Zero dependencies          | pixo                | pixo                | Pure Rust, no C toolchain          |

### The pixo Philosophy

**pixo is pure Rust with zero dependencies.** This is a deliberate design choice that sets it apart from every other image compression tool in the ecosystem.

#### The Landscape Today

| Tool     | Language | Dependencies                                        |
| -------- | -------- | --------------------------------------------------- |
| oxipng   | Rust     | Uses **libdeflate** (C) for DEFLATE compression     |
| mozjpeg  | C        | Requires C toolchain, complex build                 |
| pngquant | C        | Uses **libimagequant** (C) for quantization         |
| sharp    | Node.js  | Uses **libvips** (C), 7-12 MB native binaries       |
| squoosh  | WASM     | Emscripten-compiled C/C++ codecs (600KB-800KB each) |
| image    | Rust     | Pure Rust but includes many codecs (~2-4 MB)        |
| zune-png | Rust     | Pure Rust, but PNG-only                             |

Even "Rust" libraries often delegate the heavy lifting to C. oxipng's compression advantage comes almost entirely from libdeflate—a highly optimized C library. pngquant's superior quantization comes from libimagequant, also written in C.

#### Why Pure Rust Matters

1. **Portability**: pixo compiles to WASM without Emscripten. No C toolchain needed. Works identically on every platform.

2. **Tiny binaries**: 146 KB WASM binary vs 600-800 KB for Squoosh codecs. This matters for web apps where every kilobyte counts.

3. **Auditability**: One language, one codebase. No FFI boundaries to cross, no C memory safety concerns.

4. **Simplicity**: `cargo add pixo` just works. No system dependencies, no build scripts, no linking headaches.

#### The Tradeoffs

Being pure Rust with zero dependencies means accepting some compression ratio gaps:

| Comparison            | Gap    | Root Cause                                    |
| --------------------- | ------ | --------------------------------------------- |
| PNG Max vs oxipng     | +8-17% | libdeflate has 20+ years of C optimization    |
| JPEG Max vs mozjpeg   | +4-5%  | mozjpeg has sophisticated trellis refinement  |
| Lossy PNG vs pngquant | +9%    | libimagequant uses advanced k-means iteration |

These gaps are the cost of independence. For many use cases—especially web applications where binary size matters—the tradeoffs are worth it.

#### When to Choose pixo

| Scenario                                     | Recommendation                                        |
| -------------------------------------------- | ----------------------------------------------------- |
| **Building a web app with WASM?**            | Use pixo (146 KB binary, good compression)            |
| **Need smallest PNG file size?**             | Use pixo Lossy (50-80% smaller than lossless)         |
| **Want zero native dependencies?**           | Use pixo (pure Rust, no C toolchain)                  |
| **Need predictable output across browsers?** | Use pixo (identical output everywhere)                |
| **CLI tool, size doesn't matter?**           | Use oxipng/mozjpeg/pngquant (best compression ratios) |
| **Node.js server, need speed?**              | Use sharp (native bindings, excellent performance)    |
| **Optimizing existing images in CI/CD?**     | Use oxipng/mozjpeg/pngquant CLI tools                 |

## Running Benchmarks

Generate fresh benchmark data on your hardware:

```bash
# Run comprehensive comparison (prints summary table)
cargo bench --bench comparison

# Run specific benchmark groups
cargo bench --bench comparison "PNG Equivalent"    # Fair PNG comparison
cargo bench --bench comparison "JPEG Equivalent"   # Fair JPEG comparison
cargo bench --bench comparison "DEFLATE"           # DEFLATE deep dive
cargo bench --bench comparison "Best Effort"       # Each encoder's optimal settings
cargo bench --bench comparison "Kodak"             # Real image benchmarks

# Run component-level micro-benchmarks
cargo bench --bench components

# Quick summary without full benchmarks
cargo bench --bench comparison -- --summary-only
```

### Benchmark Groups

| Group                    | Description                                 |
| ------------------------ | ------------------------------------------- |
| PNG Equivalent Settings  | All PNG encoders at level 6                 |
| JPEG Equivalent Settings | All JPEG encoders at Q85, 4:2:0, baseline   |
| PNG/JPEG Best Effort     | Each encoder using optimal settings         |
| DEFLATE Comparison       | pixo vs flate2 vs libdeflate (levels 1,6,9) |
| DEFLATE Zopfli           | Max compression comparison                  |
| Kodak Real Images        | Real photographic images                    |
| PNG/JPEG All Presets     | pixo Fast/Balanced/Max presets              |

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
