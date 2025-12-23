# Benchmarks

This directory contains performance benchmarks for the comprs library.

## Running Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark suites
cargo bench --bench encode_benchmark  # PNG/JPEG encoding vs image crate
cargo bench --bench deflate_micro     # DEFLATE compression vs flate2
cargo bench --bench components        # Component-level benchmarks (DCT, Huffman, etc.)
cargo bench --bench comparison        # Full comparison with summary table
```

## Benchmark Suites

### encode_benchmark

Compares comprs PNG/JPEG encoding against the `image` crate:

- **PNG Encoding**: Tests gradient images at 64x64, 128x128, 256x256, 512x512
- **JPEG Encoding**: Tests quality 85 with 4:4:4 and 4:2:0 subsampling
- **Compression Ratio**: Compares output sizes for gradient and noisy images

### deflate_micro

Micro-benchmarks for DEFLATE compression:

- Compares `deflate_zlib` against `flate2` on 1 MiB payloads
- Tests both compressible (repeated patterns) and incompressible (random) data
- Reports throughput in bytes/second

### components

Component-level benchmarks for internal algorithms:

- DCT (Discrete Cosine Transform)
- Huffman encoding/decoding
- LZ77 compression
- CRC32/Adler32 checksums

### comparison

Comprehensive library comparison benchmark that produces a detailed summary table:

- **PNG encoding**: comprs vs image crate (gradient and noisy images)
- **JPEG encoding**: comprs vs image crate (q85 with 4:4:4 and 4:2:0 subsampling)
- **DEFLATE compression**: comprs vs flate2 (compressible and random data)
- **Binary sizes**: WASM binary size comparison
- **Output sizes**: Compressed output size comparison

Run this benchmark for a formatted summary table:

```bash
cargo bench --bench comparison
```

The benchmark will print a table like:

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         COMPRS BENCHMARK SUMMARY                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────────────────┐
│ WASM Binary Size Comparison                                                  │
├────────────────────┬─────────────┬────────────────────────────────────────────┤
│ Library            │ WASM Size   │ Notes                                      │
├────────────────────┼─────────────┼────────────────────────────────────────────┤
│ comprs             │      92 KB  │ Zero deps, pure Rust                       │
│ image crate        │    ~2-4 MB  │ Pure Rust, many codecs                     │
│ ...                │             │                                            │
└────────────────────┴─────────────┴────────────────────────────────────────────┘
```

---

## Cross-language targets and capabilities

These are the libraries we aim to benchmark across Rust and JS/Node. Some targets have prerequisites or may be skipped when tooling is unavailable.

### Rust targets
| Library        | Formats         | Environment | Prereqs / notes                                     | Skip rule                                  |
| -------------- | --------------- | ----------- | --------------------------------------------------- | ------------------------------------------ |
| **comprs**     | PNG, JPEG       | Rust/WASM   | Included; zero deps                                 | –                                          |
| `image`        | PNG, JPEG (+)   | Rust        | Dev-dep already present                             | –                                          |
| `photon-rs`    | PNG/JPEG*       | Rust/WASM   | Image pipeline crate; may wrap `image` encoders     | Skip if encode APIs unavailable            |
| `zune-image`   | PNG/JPEG*       | Rust/WASM   | SIMD-focused; encoding support varies by version    | Skip if encode APIs unavailable            |
| `wasm-mozjpeg` | JPEG            | WASM        | Emscripten/wasm toolchain required                  | Skip if toolchain not installed            |
| `libpng` bind  | PNG             | C/WASM      | C toolchain or Emscripten required                  | Skip if toolchain not installed            |

### JS/Node targets
| Library                     | Formats   | Environment          | Prereqs / notes                                        | Skip rule                            |
| --------------------------- | --------- | -------------------- | ------------------------------------------------------ | ------------------------------------ |
| `sharp`                     | PNG, JPEG | Node (libvips)       | Native deps; large install footprint                   | Skip if libvips/native install fails |
| `jimp`                      | PNG, JPEG | Node (pure JS)       | No native deps; slower                                 | –                                    |
| `pngjs`                     | PNG       | Node (pure JS)       | No native deps                                         | –                                    |
| `jpeg-js`                   | JPEG      | Node (pure JS)       | No native deps                                         | –                                    |
| `browser-image-compression` | PNG, JPEG | Browser / node-canvas| Needs Canvas polyfill or headless browser              | Skip if canvas/headless unavailable  |
| `squoosh-lib`               | PNG, JPEG | Node/Browser (WASM)  | Downloads WASM codecs; bundle size includes WASM files | Skip if WASM fetch fails             |

### Shared corpus
- Synthetic RGB buffers: gradient + noisy at 256×256 and 512×512 (see `benches/corpus.rs`).
- JPEG qualities: 75 and 85 by default.
- Real fixtures: reuse `tests/fixtures/{playground.png, rocket.png, multi-agent.jpg}`.

Both Rust and JS runners should rely on these definitions to keep inputs consistent.

### Running cross-language benchmarks
1) **Rust summary JSON** (no Criterion run):
```bash
cargo bench --bench comparison -- --summary-only --export-json /tmp/rust-summary.json
```

2) **JS/Node benchmarks** (from `benchmarks/js`; first run `pnpm install`):
```bash
cd benchmarks/js
pnpm bench:quick -- --output /tmp/js-bench.json   # QUICK=1, small iteration count
# or for longer runs
pnpm bench -- --output /tmp/js-bench.json         # default iterations=6, override with BENCH_ITERATIONS
```
- Results include speed, output sizes, and package sizes for sharp, jimp, pngjs, jpeg-js.
- `browser-image-compression` is skipped (browser-only). `@squoosh/lib` currently errors on Node 22 due to a read-only global `navigator`; captured as `status=error`.

3) **Aggregate + rankings**:
```bash
node benchmarks/aggregate.mjs \
  --rust /tmp/rust-summary.json \
  --js /tmp/js-bench.json \
  --output /tmp/cross-bench.md
```
The aggregator builds Markdown tables ranking speed (PNG/JPEG), output size, and binary/package size across Rust and JS results.

---

## Library Comparison

When choosing an image compression solution, consider these trade-offs:

### Rust Libraries

| Library           | WASM-friendly   | Binary Size | Throughput | Notes                                              |
| ----------------- | --------------- | ----------- | ---------- | -------------------------------------------------- |
| **comprs**        | Yes             | 92 KB       | Good       | Zero deps, pure Rust, simple WASM target           |
| `image`           | Yes             | ~2-4MB      | Good       | Pure Rust, many codecs included                    |
| `photon-rs`       | Yes             | ~200-400KB  | Excellent  | Pure Rust, designed for WASM, 13x faster than JS   |
| `zune-image`      | Yes             | ~500KB-1MB  | Excellent  | Pure Rust, SIMD optimized, 1.8x faster than libpng |
| `mozjpeg`         | Emscripten only | ~30-50KB    | Excellent  | C library, requires complex build setup            |
| `libpng` bindings | Emscripten only | N/A         | Excellent  | C library, requires Emscripten toolchain           |

_Note: photon-rs benchmarks show Gaussian blur at 180ms vs 2400ms in pure JS. zune-png benchmarks show 1.8x speedup over libpng on x86._

### JavaScript/Node.js Libraries

| Library                     | Environment    | Bundle Size (minified+gzip) | Throughput        | Notes                                          |
| --------------------------- | -------------- | --------------------------- | ----------------- | ---------------------------------------------- |
| `sharp`                     | Node.js only   | 7-12MB native binaries      | Excellent         | libvips bindings, 4-5x faster than ImageMagick |
| `jimp`                      | Browser + Node | ~4MB unpacked               | Slow (~0.7 img/s) | Pure JS, no native deps                        |
| `pngjs`                     | Browser + Node | ~180KB gzipped              | Slow              | PNG only, pure JS                              |
| `jpeg-js`                   | Browser + Node | ~35KB gzipped               | Slow              | JPEG only, pure JS                             |
| `browser-image-compression` | Browser        | ~50KB gzipped               | Varies            | Uses Canvas API internally                     |
| `squoosh-lib`               | Browser + Node | 30-100KB/codec gzipped      | Good              | Google's WASM codecs                           |

_Note: Sizes vary by version. sharp includes prebuilt libvips binaries (~534KB JS + native binaries). jimp-compact is ~27x smaller than jimp._

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

## Why comprs for WASM?

comprs is designed from the ground up for easy WASM deployment:

### 1. Zero C Dependencies

Libraries like `mozjpeg`, `libpng`, and `zlib` are C/C++ code that requires the **Emscripten** toolchain to compile to WASM:

```bash
# Complex: C library to WASM
emcc mozjpeg.c -o mozjpeg.wasm -s WASM=1 -O3

# Simple: Pure Rust to WASM
wasm-pack build --target web --features wasm
```

comprs compiles with `wasm32-unknown-unknown` - the standard, simple target.

### 2. Small Binary Size

| What you need | comprs    | image crate (est.) |
| ------------- | --------- | ------------------ |
| PNG + JPEG    | **92 KB** | ~2-4MB             |

The `image` crate includes decoders and encoders for BMP, GIF, ICO, TIFF, WebP, AVIF, and more - even if you only need PNG. You can use feature flags to reduce size, but the core still includes significant code.

comprs achieves its small size through:

- Zero runtime dependencies (hand-implemented DEFLATE, DCT, Huffman)
- Release profile optimizations (LTO, `opt-level = "z"`, `panic = "abort"`, symbol stripping)

### 3. Predictable Output

Browser Canvas APIs produce different output across browsers:

```javascript
// Safari, Chrome, and Firefox produce different JPEG bytes
canvas.toBlob(callback, "image/jpeg", 0.85);
```

comprs produces identical output regardless of environment.

### 4. Full Control

| Feature            | comprs                 | Canvas API        |
| ------------------ | ---------------------- | ----------------- |
| JPEG quality       | 1-100, precise         | Browser-dependent |
| PNG compression    | 1-9 levels             | None              |
| PNG filters        | All 5 types + adaptive | None              |
| Chroma subsampling | 4:4:4 or 4:2:0         | None              |

---

## Binary Size Breakdown

Measured WASM binary sizes for comprs:

| Configuration                    | Size      |
| -------------------------------- | --------- |
| Default release                  | 127 KB    |
| **Optimized** (LTO, opt-level=z) | **92 KB** |

The optimizations in `Cargo.toml` provide a **28% size reduction**.

### Build Configuration

The release profile in `Cargo.toml` enables size optimizations:

```toml
[profile.release]
lto = true           # Link-time optimization
opt-level = "z"      # Optimize for size
codegen-units = 1    # Single codegen unit for better optimization
panic = "abort"      # Remove unwinding code
strip = true         # Strip symbols from binary
```

Additional WASM-specific settings in `.cargo/config.toml`:

```toml
[target.wasm32-unknown-unknown]
rustflags = ["-C", "target-feature=+bulk-memory"]
```

### Building for WASM

```bash
cargo build --release --target wasm32-unknown-unknown --features wasm
```

For additional size reduction, install `wasm-opt` (from binaryen) and run:

```bash
wasm-opt -Oz -o optimized.wasm target/wasm32-unknown-unknown/release/comprs.wasm
```

### Comparison to Alternatives

| Library         | WASM Size (uncompressed) |
| --------------- | ------------------------ |
| **comprs**      | **92 KB**                |
| wasm-mozjpeg    | ~208KB                   |
| squoosh oxipng  | ~625KB                   |
| squoosh mozjpeg | ~803KB                   |
| image crate     | ~6-10MB                  |

_Note: Squoosh codec sizes from squoosh-browser-sense. Sizes depend on build configuration and optimization level._

---

## Performance Tips

### For Benchmarking

1. **Use release mode**: `cargo bench` automatically uses release optimizations
2. **Warm up**: Criterion handles warm-up automatically
3. **Consistent hardware**: Close other applications during benchmarks
4. **Multiple runs**: Let Criterion collect enough samples for statistical significance

### For Production

1. **Reuse buffers**: Use `encode_into` variants to avoid allocations
2. **Choose appropriate quality**: JPEG 75-85 is usually sufficient
3. **Consider subsampling**: 4:2:0 reduces JPEG size by ~30-40%
4. **Match filter to content**: PNG `None` for noise, `Adaptive` for photos

---

## Benchmark Results

Run benchmarks on your hardware to get accurate numbers:

```bash
cargo bench --bench encode_benchmark -- --save-baseline my-machine
```

Results are saved to `target/criterion/` with HTML reports.

---

## Data Sources

Library sizes and performance claims were gathered from:

- **npm packages**: npmjs.com package pages (unpacked sizes)
- **Bundlephobia**: bundlephobia.com for minified+gzipped bundle sizes
- **sharp**: sharp.pixelplumbing.com documentation
- **Squoosh**: squoosh-browser-sense package, wasm-mozjpeg project
- **zune-png**: Phoronix benchmarks (1.8x faster than libpng)
- **photon-rs**: Project documentation (13x speedup for Gaussian blur vs JS)
- **jimp**: skypack.dev benchmarks (~0.7 img/s vs sharp's ~11 img/s)

_Sizes are approximate and vary by version, build configuration, and optimization settings. Last updated: December 2025._
