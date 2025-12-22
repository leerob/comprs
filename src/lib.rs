//! # comprs
//!
//! A minimal-dependency, high-performance image compression library.
//!
//! This library provides PNG and JPEG encoding with hand-implemented
//! compression algorithms (DEFLATE, DCT, Huffman) for maximum control
//! and performance.
//!
//! ## Features
//!
//! - **Zero runtime dependencies by default**
//! - **PNG encoding** with all filter types and DEFLATE compression
//! - **JPEG encoding** with DCT, quantization, and Huffman coding
//! - Optional SIMD acceleration via `simd` feature
//! - Optional parallel processing via `parallel` feature
//! - Buffer reuse helpers: `png::encode_into` and
//!   `jpeg::encode_with_options_into` let you supply the output buffer to
//!   avoid repeated allocations when encoding multiple images.
//!
//! ## Example
//!
//! ```rust
//! use comprs::{png, jpeg, ColorType};
//!
//! // Encode as PNG
//! let pixels: Vec<u8> = vec![255, 0, 0, 255]; // 1x1 red RGBA pixel
//! let png_data = png::encode(&pixels, 1, 1, ColorType::Rgba).unwrap();
//!
//! // Encode as JPEG
//! let rgb_pixels: Vec<u8> = vec![255, 0, 0]; // 1x1 red RGB pixel
//! let jpeg_data = jpeg::encode(&rgb_pixels, 1, 1, 85).unwrap();
//! ```

// Allow unsafe code only when SIMD feature is enabled (required for intrinsics)
#![cfg_attr(not(feature = "simd"), forbid(unsafe_code))]
#![warn(missing_docs)]

pub mod bits;
pub mod color;
pub mod compress;
pub mod error;
pub mod jpeg;
pub mod png;

#[cfg(feature = "simd")]
pub mod simd;

pub use color::ColorType;
pub use error::{Error, Result};
