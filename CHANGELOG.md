# Changelog

All notable changes to this project will be documented in this file.

## [0.4.0] - 2025-12-27

### Changed

- **BREAKING: PNG/JPEG/Resize APIs now use builder pattern** — Unified, cleaner, more extensible API
  - All encode functions now take `(data, &options)` instead of positional arguments
  - Width, height, and color type are now part of the options struct
  - Use `PngOptions::builder(w, h).color_type(...).build()` to configure PNG
  - Use `JpegOptions::builder(w, h).color_type(...).quality(85).build()` to configure JPEG
  - Use `ResizeOptions::builder(src_w, src_h).dst(dst_w, dst_h).build()` to configure resize

### Migration Guide

**Before (0.3.x):**
```rust
// PNG
png::encode(&pixels, width, height, ColorType::Rgba)?;
png::encode_with_options(&pixels, width, height, ColorType::Rgba, &options)?;

// JPEG
jpeg::encode(&pixels, width, height, quality)?;
jpeg::encode_with_options(&pixels, width, height, ColorType::Rgb, &options)?;
```

**After (0.4.0):**
```rust
// PNG
let opts = PngOptions::builder(width, height).color_type(ColorType::Rgba).build();
png::encode(&pixels, &opts)?;

// JPEG  
let opts = JpegOptions::builder(width, height)
    .color_type(ColorType::Rgb)
    .quality(85)
    .build();
jpeg::encode(&pixels, &opts)?;
```

### Removed

- `png::encode_with_options()` — use `png::encode()` with builder options
- `jpeg::encode_with_options()` — use `jpeg::encode()` with builder options
- `jpeg::encode_with_color()` — use builder `.color_type()` method
- `jpeg::encode_with_options_into()` — use `jpeg::encode_into()`
- `timing` feature flag — unused profiling feature removed to reduce maintenance burden

### Added

- `PngOptions::builder(width, height)` — builder pattern for PNG options
- `JpegOptions::builder(width, height)` — builder pattern for JPEG options
- `PngOptionsBuilder::color_type()` — set color type in builder
- `JpegOptionsBuilder::color_type()` — set color type in builder
- Preset helpers now require dimensions: `PngOptions::fast(w, h)`, `JpegOptions::fast(w, h, quality)`

## [0.3.0] - 2025-12-27

### Added

- **Image Resizing API** — New `resize` module for high-quality image resizing
  - Three algorithms: `Nearest` (fastest), `Bilinear` (balanced), `Lanczos3` (highest quality, default)
  - Support for all color types: Gray, GrayAlpha, RGB, RGBA
  - Separable filtering with precomputed contributions for O(2n) per-pixel performance
  - Parallel processing support via the `parallel` feature flag

- **WASM Resize Support** — `resizeImage()` function in WASM bindings
  - Aspect ratio preservation option
  - Algorithm selection (nearest, bilinear, lanczos3)

## [0.2.1] - 2025-12-27

### Changed

- Documentation link fixes for GitHub and docs.rs compatibility

## [0.2.0] - 2025-12-26

### Added

- PNG encoder with all filter types and DEFLATE compression
- JPEG encoder with baseline and progressive modes
- Lossy PNG via palette quantization
- Trellis quantization for JPEG
- SIMD acceleration (x86_64 AVX2/SSE, aarch64 NEON)
- WASM bindings with 149 KB binary size
- CLI tool for command-line compression
- PNG/JPEG decoding support
- Comprehensive documentation and guides

## [0.1.0] - 2025-12-21

### Added

- Initial release
- Core compression algorithms (Huffman, LZ77, DEFLATE)
- Basic PNG and JPEG encoding
