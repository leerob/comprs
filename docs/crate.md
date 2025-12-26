# Rust Crate

Use pixo as a library in your Rust projects.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
pixo = "0.1"
```

## Toolchain

The project builds and tests on **stable Rust 1.82+**.

## PNG Encoding

```rust
use pixo::{png, ColorType};

// Encode RGB pixels as PNG
let pixels: Vec<u8> = vec![255, 0, 0, 0, 255, 0, 0, 0, 255]; // 3 RGB pixels
let png_data = png::encode(&pixels, 3, 1, ColorType::Rgb).unwrap();

// With custom options
use pixo::png::{PngOptions, FilterStrategy};

let options = PngOptions {
    compression_level: 9,  // validated: 1-9, higher = better compression
    filter_strategy: FilterStrategy::Adaptive,
};
let png_data = png::encode_with_options(&pixels, 3, 1, ColorType::Rgb, &options).unwrap();
```

## JPEG Encoding

```rust
use pixo::jpeg;

// Encode RGB pixels as JPEG
let pixels: Vec<u8> = vec![255, 128, 64]; // 1 RGB pixel
let jpeg_data = jpeg::encode(&pixels, 1, 1, 85).unwrap(); // quality: 1-100

// With subsampling options (4:4:4 default, 4:2:0 available)
use pixo::jpeg::{JpegOptions, Subsampling};

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
