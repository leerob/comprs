//! WebAssembly bindings for comprs.
//!
//! This module provides wasm-bindgen exports for encoding PNG and JPEG images
//! from JavaScript/TypeScript. Enable with the `wasm` feature.
//!
//! # Example (JavaScript)
//!
//! ```javascript
//! import init, { encode_png, encode_jpeg } from 'comprs';
//!
//! await init();
//!
//! // Get pixels from canvas
//! const ctx = canvas.getContext('2d');
//! const imageData = ctx.getImageData(0, 0, width, height);
//!
//! // Encode as PNG (RGBA = 3)
//! const pngBytes = encode_png(imageData.data, width, height, 3, 6);
//!
//! // Encode as JPEG (need RGB, so strip alpha first)
//! const rgb = stripAlpha(imageData.data);
//! const jpegBytes = encode_jpeg(rgb, width, height, 85, 2, false);
//! ```

use wasm_bindgen::prelude::*;

use crate::color::ColorType;
use crate::jpeg::{self, JpegOptions, Subsampling};
use crate::png::{self, FilterStrategy, PngOptions};

/// Convert a u8 color type code to ColorType enum.
///
/// - 0 = Gray
/// - 1 = GrayAlpha
/// - 2 = Rgb
/// - 3 = Rgba
fn color_type_from_u8(value: u8) -> Result<ColorType, JsError> {
    match value {
        0 => Ok(ColorType::Gray),
        1 => Ok(ColorType::GrayAlpha),
        2 => Ok(ColorType::Rgb),
        3 => Ok(ColorType::Rgba),
        _ => Err(JsError::new(&format!(
            "Invalid color type: {value}. Expected 0 (Gray), 1 (GrayAlpha), 2 (Rgb), or 3 (Rgba)",
        ))),
    }
}

/// Encode raw pixel data as PNG.
///
/// # Arguments
///
/// * `data` - Raw pixel data as Uint8Array (row-major order)
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `color_type` - Color type: 0=Gray, 1=GrayAlpha, 2=Rgb, 3=Rgba
/// * `compression_level` - Compression level 1-9 (6 recommended)
///
/// # Returns
///
/// PNG file bytes as Uint8Array.
#[wasm_bindgen(js_name = "encodePng")]
pub fn encode_png(
    data: &[u8],
    width: u32,
    height: u32,
    color_type: u8,
    compression_level: u8,
) -> Result<Vec<u8>, JsError> {
    let color = color_type_from_u8(color_type)?;

    let options = PngOptions {
        compression_level,
        filter_strategy: FilterStrategy::Adaptive,
    };

    png::encode_with_options(data, width, height, color, &options)
        .map_err(|e| JsError::new(&e.to_string()))
}

/// Encode raw pixel data as PNG with a specific filter strategy.
///
/// # Arguments
///
/// * `data` - Raw pixel data as Uint8Array (row-major order)
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `color_type` - Color type: 0=Gray, 1=GrayAlpha, 2=Rgb, 3=Rgba
/// * `compression_level` - Compression level 1-9 (6 recommended)
/// * `filter` - Filter strategy: 0=None, 1=Sub, 2=Up, 3=Average, 4=Paeth, 5=Adaptive, 6=AdaptiveFast
///
/// # Returns
///
/// PNG file bytes as Uint8Array.
#[wasm_bindgen(js_name = "encodePngWithFilter")]
pub fn encode_png_with_filter(
    data: &[u8],
    width: u32,
    height: u32,
    color_type: u8,
    compression_level: u8,
    filter: u8,
) -> Result<Vec<u8>, JsError> {
    let color = color_type_from_u8(color_type)?;

    let filter_strategy = match filter {
        0 => FilterStrategy::None,
        1 => FilterStrategy::Sub,
        2 => FilterStrategy::Up,
        3 => FilterStrategy::Average,
        4 => FilterStrategy::Paeth,
        5 => FilterStrategy::Adaptive,
        6 => FilterStrategy::AdaptiveFast,
        _ => {
            return Err(JsError::new(&format!(
                "Invalid filter: {filter}. Expected 0-6",
            )))
        }
    };

    let options = PngOptions {
        compression_level,
        filter_strategy,
    };

    png::encode_with_options(data, width, height, color, &options)
        .map_err(|e| JsError::new(&e.to_string()))
}

/// Encode raw pixel data as JPEG.
///
/// # Arguments
///
/// * `data` - Raw pixel data as Uint8Array (row-major order)
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `quality` - Quality level 1-100 (85 recommended)
/// * `color_type` - Color type: 0=Gray, 2=Rgb (JPEG only supports these)
/// * `subsampling_420` - If true, use 4:2:0 chroma subsampling (smaller files)
///
/// # Returns
///
/// JPEG file bytes as Uint8Array.
#[wasm_bindgen(js_name = "encodeJpeg")]
pub fn encode_jpeg(
    data: &[u8],
    width: u32,
    height: u32,
    quality: u8,
    color_type: u8,
    subsampling_420: bool,
) -> Result<Vec<u8>, JsError> {
    let color = match color_type {
        0 => ColorType::Gray,
        2 => ColorType::Rgb,
        _ => {
            return Err(JsError::new(&format!(
                "Invalid color type for JPEG: {color_type}. Expected 0 (Gray) or 2 (Rgb)",
            )))
        }
    };

    let options = JpegOptions {
        quality,
        subsampling: if subsampling_420 {
            Subsampling::S420
        } else {
            Subsampling::S444
        },
        restart_interval: None,
    };

    jpeg::encode_with_options(data, width, height, quality, color, &options)
        .map_err(|e| JsError::new(&e.to_string()))
}

/// Get the number of bytes per pixel for a color type.
///
/// Useful for validating input data length.
///
/// * 0 (Gray) = 1 byte
/// * 1 (GrayAlpha) = 2 bytes
/// * 2 (Rgb) = 3 bytes
/// * 3 (Rgba) = 4 bytes
#[wasm_bindgen(js_name = "bytesPerPixel")]
pub fn bytes_per_pixel(color_type: u8) -> Result<u8, JsError> {
    let color = color_type_from_u8(color_type)?;
    Ok(color.bytes_per_pixel() as u8)
}

// Tests for the WASM module.
// Note: Tests that involve JsError can only run on wasm32 targets.
// For native testing, we test the underlying encoding functions directly
// in the png and jpeg modules.
#[cfg(all(test, target_arch = "wasm32"))]
mod tests {
    use super::*;

    #[test]
    fn test_encode_png_1x1() {
        let pixels = vec![255, 0, 0, 255]; // 1x1 red RGBA
        let result = encode_png(&pixels, 1, 1, 3, 6);
        assert!(result.is_ok());
        let png = result.unwrap();
        // Check PNG signature
        assert_eq!(
            &png[0..8],
            &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
        );
    }

    #[test]
    fn test_encode_jpeg_1x1() {
        let pixels = vec![255, 0, 0]; // 1x1 red RGB
        let result = encode_jpeg(&pixels, 1, 1, 85, 2, false);
        assert!(result.is_ok());
        let jpeg = result.unwrap();
        // Check JPEG SOI marker
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn test_invalid_color_type() {
        let pixels = vec![255, 0, 0];
        let result = encode_png(&pixels, 1, 1, 99, 6);
        assert!(result.is_err());
    }

    #[test]
    fn test_jpeg_invalid_color_type() {
        let pixels = vec![255, 0, 0, 255];
        // JPEG doesn't support RGBA (color_type 3)
        let result = encode_jpeg(&pixels, 1, 1, 85, 3, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_bytes_per_pixel() {
        assert_eq!(bytes_per_pixel(0).unwrap(), 1);
        assert_eq!(bytes_per_pixel(1).unwrap(), 2);
        assert_eq!(bytes_per_pixel(2).unwrap(), 3);
        assert_eq!(bytes_per_pixel(3).unwrap(), 4);
        assert!(bytes_per_pixel(99).is_err());
    }
}
