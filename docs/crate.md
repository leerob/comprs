# Rust Crate

Use comprs as a library in your Rust projects.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
comprs = { version = "0.1", default-features = true }
```

## Toolchain

The project builds and tests on **stable Rust 1.82+**.

## PNG Decoding (opt-in feature)

PNG decoding is feature-gated to keep binaries small and is **not** included in WASM builds.

```toml
[dependencies]
comprs = { version = "0.1", features = ["decode"] }
```

```rust
use comprs::png;

let bytes = std::fs::read("image.png")?;
let decoded = png::decode(&bytes)?;
println!("{}x{} {:?}", decoded.width, decoded.height, decoded.color_type);
```

Notes:
- Supports non-interlaced PNGs, bit depths 1/2/4/8, color types 0/2/3/4/6.
- Interlaced and 16-bit PNGs are rejected.
- Palette + tRNS are expanded to RGB/RGBA; <8-bit grayscale is expanded to 8-bit.

## PNG Encoding

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

## JPEG Encoding

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

## Buffer Reuse

Both encoders support writing into a caller-provided buffer to avoid repeated allocations:

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

## Color Types

- `ColorType::Gray` — Grayscale (1 byte/pixel)
- `ColorType::GrayAlpha` — Grayscale + Alpha (2 bytes/pixel)
- `ColorType::Rgb` — RGB (3 bytes/pixel)
- `ColorType::Rgba` — RGBA (4 bytes/pixel)

Note: JPEG only supports `Gray` and `Rgb` color types.

## Features

- `wasm` — Build WebAssembly bindings (adds `wasm-bindgen`, `js-sys`)
- `cli` — Build the command-line interface (adds `clap`, `png`, `jpeg-decoder`)
- `simd` _(default)_ — Enable SIMD optimizations with runtime feature detection
- `parallel` _(default)_ — Enable parallel processing with rayon

## PNG Presets

- `PngOptions::fast()` — level 2, AdaptiveFast (default)
- `PngOptions::balanced()` — level 6, Adaptive
- `PngOptions::max_compression()` — level 9, AdaptiveSampled(interval=2)

## Testing

```bash
cargo test
```

## Benchmarks

```bash
cargo bench
```

See [benchmarks](../benches/BENCHMARKS.md) for detailed comparisons.
