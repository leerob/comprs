//! Image decoders for PNG and JPEG formats.
//!
//! This module provides native decoders that remove the dependency on external
//! crates like `png` and `image` for the CLI feature.
//!
//! # Supported Formats
//!
//! ## PNG
//! - All color types (Gray, GrayAlpha, RGB, RGBA, Indexed)
//! - All filter types (None, Sub, Up, Average, Paeth)
//! - 8-bit and 16-bit depth (16-bit downsampled to 8-bit)
//! - Sub-8-bit packed depths (1, 2, 4 bit)
//! - Non-interlaced images only (Adam7 not supported)
//!
//! ## JPEG
//! - Baseline DCT (SOF0)
//! - 8-bit precision
//! - Grayscale and YCbCr color
//! - Common subsampling (4:4:4, 4:2:0, 4:2:2)
//! - Progressive (SOF2) not supported
//!
//! # Example
//!
//! ```ignore
//! use pixo::decode::{decode_png, decode_jpeg};
//!
//! // Decode a PNG
//! let png_bytes = std::fs::read("image.png")?;
//! let png = decode_png(&png_bytes)?;
//! println!("{}x{} {:?}", png.width, png.height, png.color_type);
//!
//! // Decode a JPEG
//! let jpg_bytes = std::fs::read("image.jpg")?;
//! let jpg = decode_jpeg(&jpg_bytes)?;
//! println!("{}x{} {:?}", jpg.width, jpg.height, jpg.color_type);
//! ```

mod bit_reader;
mod idct;
mod inflate;
mod jpeg;
mod png;

pub use jpeg::{decode_jpeg, JpegImage};
pub use png::{decode_png, PngImage};
