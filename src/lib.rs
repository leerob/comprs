//! # pixo
//!
//! A minimal-dependency, high-performance image compression library with PNG
//! and JPEG encoders written entirely in Rust.
//!
//! - **PNG**: All filter types, DEFLATE compressor, palette quantization,
//!   buffer reuse helpers, and presets from fast to max compression.
//! - **JPEG**: Baseline & progressive encoding, optimized Huffman tables,
//!   trellis quantization, and buffer reuse helpers.
//! - **Performance**: Optional SIMD and parallel fast paths, tuned defaults,
//!   optimal DEFLATE for final packing, and small WASM output.
//! - **Docs-first**: Conceptual guides (PNG, JPEG, DEFLATE, DCT, quantization,
//!   performance) are embedded directly in rustdoc under [`guides`].
//!
//! ## Quickstart
//!
//! ```rust
//! use pixo::{png, jpeg, ColorType};
//! use pixo::png::PngOptions;
//! use pixo::jpeg::JpegOptions;
//!
//! # fn main() -> pixo::Result<()> {
//! // PNG: 3x1 RGB pixels (red, green, blue)
//! let png_pixels = vec![255, 0, 0, 0, 255, 0, 0, 0, 255];
//! let png_opts = PngOptions::builder(3, 1).color_type(ColorType::Rgb).build();
//! let png_bytes = png::encode(&png_pixels, &png_opts)?;
//! assert!(!png_bytes.is_empty());
//!
//! // JPEG: 1x1 RGB pixel
//! let jpg_pixels = vec![255, 0, 0];
//! let jpg_opts = JpegOptions::builder(1, 1).color_type(ColorType::Rgb).quality(85).build();
//! let jpg_bytes = jpeg::encode(&jpg_pixels, &jpg_opts)?;
//! assert!(!jpg_bytes.is_empty());
//! # Ok(())
//! # }
//! ```
//!
//! ### Custom options (PNG)
//!
//! ```rust
//! use pixo::png::{encode, FilterStrategy, PngOptions};
//! use pixo::ColorType;
//!
//! # fn main() -> pixo::Result<()> {
//! let pixels = vec![255, 0, 0, 0, 255, 0, 0, 0, 255];
//! let options = PngOptions::builder(3, 1)
//!     .color_type(ColorType::Rgb)
//!     .preset(1) // balanced: compression level 6, adaptive filters + lossless opts
//!     .filter_strategy(FilterStrategy::Adaptive)
//!     .optimize_alpha(true)
//!     .build();
//! let png_bytes = encode(&pixels, &options)?;
//! assert!(!png_bytes.is_empty());
//! # Ok(())
//! # }
//! ```
//!
//! ### Custom options (JPEG)
//!
//! ```rust
//! use pixo::jpeg::{encode, JpegOptions};
//! use pixo::ColorType;
//!
//! # fn main() -> pixo::Result<()> {
//! let pixels = vec![255, 0, 0];
//! let options = JpegOptions::max(1, 1, 85); // progressive + trellis + optimized Huffman
//! let jpg_bytes = encode(&pixels, &options)?;
//! assert!(!jpg_bytes.is_empty());
//! # Ok(())
//! # }
//! ```
//!
//! ### Image resizing
//!
//! ```rust
//! use pixo::{resize, ColorType, ResizeAlgorithm, ResizeOptions};
//!
//! # fn main() -> pixo::Result<()> {
//! // 4x4 RGBA image -> 2x2 using Lanczos3 (highest quality)
//! let pixels = vec![128u8; 4 * 4 * 4];
//! let options = ResizeOptions::builder(4, 4)
//!     .dst(2, 2)
//!     .color_type(ColorType::Rgba)
//!     .algorithm(ResizeAlgorithm::Lanczos3)
//!     .build();
//! let resized = resize::resize(&pixels, &options)?;
//! assert_eq!(resized.len(), 2 * 2 * 4);
//! # Ok(())
//! # }
//! ```
//!
//! ### Buffer reuse
//!
//! ```rust
//! use pixo::{jpeg, png, ColorType};
//! use pixo::png::PngOptions;
//! use pixo::jpeg::JpegOptions;
//!
//! # fn main() -> pixo::Result<()> {
//! let pixels = vec![255, 0, 0, 0, 255, 0]; // 2 RGB pixels
//!
//! let mut png_buf = Vec::new();
//! let png_opts = PngOptions::builder(2, 1).color_type(ColorType::Rgb).build();
//! png::encode_into(&mut png_buf, &pixels, &png_opts)?;
//!
//! let mut jpg_buf = Vec::new();
//! let jpg_opts = JpegOptions::balanced(2, 1, 85);
//! jpeg::encode_into(&mut jpg_buf, &pixels, &jpg_opts)?;
//!
//! assert!(!png_buf.is_empty() && !jpg_buf.is_empty());
//! # Ok(())
//! # }
//! ```
//!
//! ## Feature flags
//! - `simd` (default): Enable SIMD-accelerated kernels with runtime detection.
//! - `parallel` (default): Parallel row-level filtering in PNG via rayon.
//! - `wasm`: WebAssembly bindings (see [`guides::wasm`]).
//! - `cli`: Command-line encoder (see [`guides::cli`]).
//!
//! ## Guides inside rustdoc
//! - [`guides::overview`] — Documentation index.
//! - [`guides::introduction_to_image_compression`] — Compression fundamentals.
//! - [`guides::png_encoding`] — PNG pipeline, filters, palette quantization.
//! - [`guides::jpeg_encoding`] — JPEG pipeline, DCT, quantization, trellis.
//! - [`guides::performance_optimization`] — SIMD/parallel/algorithm tuning.
//! - [`guides::huffman_coding`], [`guides::lz77_compression`], [`guides::deflate`] — Core compressors.
//! - [`guides::dct`] and [`guides::quantization`] — JPEG math deep dives.
//! - [`guides::wasm`] and [`guides::cli`] — Platform integration.
//!
//! ## When to use which format
//! - **PNG**: sharp UI, screenshots, graphics; lossless and optionally palette-quantized.
//! - **JPEG**: photographs and gradients; choose quality + subsampling to trade size for fidelity.
//!
//! ## Safety and performance notes
//! - Unsafe is only compiled when `simd` or `wasm` is enabled; otherwise `forbid(unsafe_code)` is applied.
//! - Prefer `encode_into` variants when encoding repeatedly to reuse allocations.
//! - For smallest PNGs, use `PngOptions::max(w, h)` (slow) or enable auto quantization for lossy palette outputs.
//! - For smallest JPEGs, use `JpegOptions::max(w, h, quality)` which enables optimized Huffman, progressive scans, and trellis quantization.

#![cfg_attr(docsrs, feature(doc_cfg))]
// Allow unsafe code only when SIMD or WASM feature is enabled
#![cfg_attr(not(any(feature = "simd", feature = "wasm")), forbid(unsafe_code))]

pub mod bits;
pub mod color;
pub mod compress;
#[cfg(feature = "cli")]
#[cfg_attr(docsrs, doc(cfg(feature = "cli")))]
pub mod decode;
pub mod error;
pub mod jpeg;
pub mod png;
pub mod resize;

#[cfg(feature = "simd")]
#[cfg_attr(docsrs, doc(cfg(feature = "simd")))]
pub mod simd;

#[cfg(feature = "wasm")]
#[cfg_attr(docsrs, doc(cfg(feature = "wasm")))]
pub mod wasm;

pub use color::ColorType;
pub use error::{Error, Result};
pub use resize::{ResizeAlgorithm, ResizeOptions};

/// High-level and conceptual guides rendered inside rustdoc.
///
/// The markdown files in `docs/` use relative links (e.g., `./huffman-coding.md`)
/// which work when browsing on GitHub. At build time, `build.rs` transforms these
/// to rustdoc intra-doc links (e.g., `crate::guides::huffman_coding`) so they work
/// on docs.rs.
#[cfg(doc)]
pub mod guides {
    #![allow(clippy::all)]

    #[doc = include_str!(concat!(env!("OUT_DIR"), "/docs/README.md"))]
    pub mod overview {}

    #[doc = include_str!(concat!(env!("OUT_DIR"), "/docs/crate.md"))]
    pub mod crate_usage {}

    #[doc = include_str!(concat!(env!("OUT_DIR"), "/docs/introduction-to-image-compression.md"))]
    pub mod introduction_to_image_compression {}

    #[doc = include_str!(concat!(env!("OUT_DIR"), "/docs/introduction-to-rust.md"))]
    pub mod introduction_to_rust {}

    #[doc = include_str!(concat!(env!("OUT_DIR"), "/docs/huffman-coding.md"))]
    pub mod huffman_coding {}

    #[doc = include_str!(concat!(env!("OUT_DIR"), "/docs/lz77-compression.md"))]
    pub mod lz77_compression {}

    #[doc = include_str!(concat!(env!("OUT_DIR"), "/docs/deflate.md"))]
    pub mod deflate {}

    #[doc = include_str!(concat!(env!("OUT_DIR"), "/docs/png-encoding.md"))]
    pub mod png_encoding {}

    #[doc = include_str!(concat!(env!("OUT_DIR"), "/docs/jpeg-encoding.md"))]
    pub mod jpeg_encoding {}

    #[doc = include_str!(concat!(env!("OUT_DIR"), "/docs/dct.md"))]
    pub mod dct {}

    #[doc = include_str!(concat!(env!("OUT_DIR"), "/docs/quantization.md"))]
    pub mod quantization {}

    #[doc = include_str!(concat!(env!("OUT_DIR"), "/docs/performance-optimization.md"))]
    pub mod performance_optimization {}

    #[doc = include_str!(concat!(env!("OUT_DIR"), "/docs/compression-evolution.md"))]
    pub mod compression_evolution {}

    #[doc = include_str!(concat!(env!("OUT_DIR"), "/docs/wasm.md"))]
    pub mod wasm {}

    #[doc = include_str!(concat!(env!("OUT_DIR"), "/docs/cli.md"))]
    pub mod cli {}

    #[doc = include_str!(concat!(env!("OUT_DIR"), "/docs/decoding.md"))]
    pub mod decoding {}
}
