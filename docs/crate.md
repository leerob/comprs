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
use pixo::png::{FilterStrategy, PngOptions};

// Encode RGB pixels as PNG
let pixels: Vec<u8> = vec![255, 0, 0, 0, 255, 0, 0, 0, 255]; // 3 RGB pixels
let options = PngOptions::builder(3, 1).color_type(ColorType::Rgb).build();
let png_data = png::encode(&pixels, &options).unwrap();

// With custom options
let options = PngOptions::builder(3, 1)
    .color_type(ColorType::Rgb)
    .compression_level(9) // validated: 1-9, higher = better compression
    .filter_strategy(FilterStrategy::Adaptive)
    .optimize_alpha(true)
    .strip_metadata(true)
    .reduce_color_type(true)
    .reduce_palette(true)
    .build();
let png_data = png::encode(&pixels, &options).unwrap();
```

## JPEG Encoding

```rust
use pixo::{jpeg, ColorType};
use pixo::jpeg::{JpegOptions, Subsampling};

// Encode RGB pixels as JPEG
let pixels: Vec<u8> = vec![255, 128, 64]; // 1 RGB pixel
let options = JpegOptions::builder(1, 1)
    .color_type(ColorType::Rgb)
    .quality(85)
    .build();
let jpeg_data = jpeg::encode(&pixels, &options).unwrap();

// With subsampling options (4:4:4 default, 4:2:0 available)
let options = JpegOptions::balanced(1, 1, 85);
let jpeg_data = jpeg::encode(&pixels, &options).unwrap();
```

## Buffer Reuse

Both encoders support writing into a caller-provided buffer to avoid repeated allocations:

```rust,ignore
use pixo::{png, jpeg, ColorType};
use pixo::png::PngOptions;
use pixo::jpeg::JpegOptions;

// PNG
let mut png_buf = Vec::new();
let opts = PngOptions::builder(3, 1).color_type(ColorType::Rgb).build();
png::encode_into(&mut png_buf, &pixels, &opts).unwrap();

// JPEG
let mut jpg_buf = Vec::new();
let opts = JpegOptions::balanced(1, 1, 85);
jpeg::encode_into(&mut jpg_buf, &pixels, &opts).unwrap();
```

## Color Types

- `ColorType::Gray` — Grayscale (1 byte/pixel)
- `ColorType::GrayAlpha` — Grayscale + Alpha (2 bytes/pixel)
- `ColorType::Rgb` — RGB (3 bytes/pixel)
- `ColorType::Rgba` — RGBA (4 bytes/pixel)

Note: JPEG only supports `Gray` and `Rgb` color types.

## Features

- `simd` _(default)_ — Enable SIMD optimizations with runtime feature detection
- `parallel` _(default)_ — Enable parallel processing with rayon
- `wasm` — Build WebAssembly bindings (adds `wasm-bindgen`, `talc`)
- `cli` — Build the command-line interface (adds `clap`)

## PNG Presets

- `PngOptions::fast()` — level 2, AdaptiveFast (default)
- `PngOptions::balanced()` — level 6, Adaptive + lossless optimizations
- `PngOptions::max()` — level 9, MinSum + optimal DEFLATE (slowest, smallest)

Lossy palette quantization can be enabled by setting:

```rust,ignore
use pixo::png::{PngOptions, QuantizationMode};

let options = PngOptions::builder()
    .quantization_mode(QuantizationMode::Auto) // enable palette quantization when beneficial
    .build();
```

## Testing

```bash
cargo test
```

## Benchmarks

```bash
cargo bench
```

See [benchmarks](../benches/BENCHMARKS.md) for detailed comparisons.
