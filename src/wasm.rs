//! WebAssembly bindings for comprs.
//!
//! This module provides a minimal WASM API for encoding PNG and JPEG images.
//! Only 3 functions are exported to keep the binary size small (~214 KB).
//!
//! # Building
//!
//! ```bash
//! # Install wasm target and wasm-bindgen
//! rustup target add wasm32-unknown-unknown
//! cargo install wasm-bindgen-cli
//!
//! # Build (if using Homebrew Rust, may need: RUSTC=~/.cargo/bin/rustc)
//! cargo build --target wasm32-unknown-unknown --release --features wasm
//!
//! # Generate JS bindings
//! wasm-bindgen --target web --out-dir web/src/lib/comprs-wasm --out-name comprs \
//!   target/wasm32-unknown-unknown/release/comprs.wasm
//! ```
//!
//! # Example (JavaScript)
//!
//! ```javascript
//! import init, { encodePng, encodeJpeg, bytesPerPixel } from 'comprs';
//!
//! await init();
//!
//! // Get pixels from canvas
//! const ctx = canvas.getContext('2d');
//! const imageData = ctx.getImageData(0, 0, width, height);
//!
//! // Encode as PNG (RGBA=3, preset=1 balanced, lossy=true for smaller files)
//! const pngBytes = encodePng(imageData.data, width, height, 3, 1, true);
//!
//! // Encode as JPEG (RGB=2, quality=85, preset=1 balanced, subsampling=true)
//! const rgb = stripAlpha(imageData.data);
//! const jpegBytes = encodeJpeg(rgb, width, height, 2, 85, 1, true);
//! ```

// Use talc allocator for WASM - smaller binary and proper memory management.
#[cfg(all(target_arch = "wasm32", feature = "wasm"))]
#[global_allocator]
static ALLOC: talc::TalckWasm = unsafe { talc::TalckWasm::new_global() };

use wasm_bindgen::prelude::*;

use crate::color::ColorType;
use crate::jpeg::{self, JpegOptions, Subsampling};
use crate::png::{self, PngOptions};

/// Convert a u8 color type code to ColorType enum.
fn color_type_from_u8(value: u8) -> Result<ColorType, JsError> {
    ColorType::try_from(value).map_err(|v| {
        JsError::new(&format!(
            "Invalid color type: {v}. Expected 0 (Gray), 1 (GrayAlpha), 2 (Rgb), or 3 (Rgba)",
        ))
    })
}

/// Encode raw pixel data as PNG.
///
/// # Arguments
///
/// * `data` - Raw pixel data as Uint8Array (row-major order)
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `color_type` - Color type: 0=Gray, 1=GrayAlpha, 2=Rgb, 3=Rgba
/// * `preset` - Optimization preset: 0=fast, 1=balanced, 2=max
/// * `lossy` - If true, enable quantization for smaller files (reduces colors to 256)
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
    preset: u8,
    lossy: bool,
) -> Result<Vec<u8>, JsError> {
    let color = color_type_from_u8(color_type)?;
    // lossy=true means we want quantization, which is lossless=false internally
    let options = PngOptions::builder().preset(preset).lossy(lossy).build();
    png::encode_with_options(data, width, height, color, &options)
        .map_err(|e| JsError::new(&e.to_string()))
}

/// Encode raw pixel data as JPEG.
///
/// # Arguments
///
/// * `data` - Raw pixel data as Uint8Array (row-major order, RGB only)
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `color_type` - Color type: 0=Gray, 2=Rgb (JPEG only supports these)
/// * `quality` - Quality level 1-100 (85 recommended)
/// * `preset` - Optimization preset: 0=fast, 1=balanced, 2=max
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
    color_type: u8,
    quality: u8,
    preset: u8,
    subsampling_420: bool,
) -> Result<Vec<u8>, JsError> {
    let color = match ColorType::try_from(color_type) {
        Ok(ColorType::Gray) => ColorType::Gray,
        Ok(ColorType::Rgb) => ColorType::Rgb,
        _ => {
            return Err(JsError::new(&format!(
                "Invalid color type for JPEG: {color_type}. Expected 0 (Gray) or 2 (Rgb)",
            )))
        }
    };
    let options = JpegOptions::builder()
        .quality(quality)
        .preset(preset)
        .subsampling(if subsampling_420 {
            Subsampling::S420
        } else {
            Subsampling::S444
        })
        .build();
    jpeg::encode_with_options(data, width, height, color, &options)
        .map_err(|e| JsError::new(&e.to_string()))
}

/// Get the number of bytes per pixel for a color type.
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
                                           // encode_png(data, w, h, color_type, preset, lossy)
        let result = encode_png(&pixels, 1, 1, 3, 1, false);
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
                                      // encode_jpeg(data, w, h, color_type, quality, preset, subsampling_420)
        let result = encode_jpeg(&pixels, 1, 1, 2, 85, 1, false);
        assert!(result.is_ok());
        let jpeg = result.unwrap();
        // Check JPEG SOI marker
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn test_invalid_color_type() {
        let pixels = vec![255, 0, 0];
        let result = encode_png(&pixels, 1, 1, 99, 1, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_jpeg_invalid_color_type() {
        let pixels = vec![255, 0, 0, 255];
        // JPEG doesn't support RGBA (color_type 3)
        let result = encode_jpeg(&pixels, 1, 1, 3, 85, 1, false);
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
