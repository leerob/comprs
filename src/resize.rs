//! Image resizing algorithms.
//!
//! This module provides high-quality image resizing with multiple algorithms:
//! - **Nearest neighbor**: Fastest, pixelated results (good for pixel art)
//! - **Bilinear**: Fast, smooth results (good balance)
//! - **Lanczos3**: Highest quality, slower (best for photographs)
//!
//! # Example
//!
//! ```rust
//! use pixo::resize::{resize, ResizeAlgorithm};
//! use pixo::ColorType;
//!
//! // Resize a 100x100 RGBA image to 50x50 using Lanczos3
//! let pixels = vec![128u8; 100 * 100 * 4];
//! let resized = resize(
//!     &pixels,
//!     100,
//!     100,
//!     50,
//!     50,
//!     ColorType::Rgba,
//!     ResizeAlgorithm::Lanczos3,
//! ).unwrap();
//! assert_eq!(resized.len(), 50 * 50 * 4);
//! ```

use crate::color::ColorType;
use crate::error::{Error, Result};
use std::f32::consts::PI;

/// Maximum supported dimension for resizing.
const MAX_DIMENSION: u32 = 1 << 24; // 16 million pixels

/// Resizing algorithm to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResizeAlgorithm {
    /// Nearest neighbor: fastest, pixelated results.
    /// Best for pixel art or when speed is critical.
    Nearest,
    /// Bilinear interpolation: fast with smooth results.
    /// Good balance between quality and speed.
    #[default]
    Bilinear,
    /// Lanczos3 resampling: highest quality, slowest.
    /// Best for photographs and high-quality downscaling.
    Lanczos3,
}

/// Resize an image to new dimensions.
///
/// # Arguments
///
/// * `data` - Raw pixel data (row-major order)
/// * `src_width` - Source image width in pixels
/// * `src_height` - Source image height in pixels
/// * `dst_width` - Destination image width in pixels
/// * `dst_height` - Destination image height in pixels
/// * `color_type` - Color type of the pixel data
/// * `algorithm` - Resizing algorithm to use
///
/// # Returns
///
/// Resized pixel data with the same color type.
///
/// # Errors
///
/// Returns an error if dimensions are invalid or data length doesn't match.
pub fn resize(
    data: &[u8],
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    color_type: ColorType,
    algorithm: ResizeAlgorithm,
) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    resize_into(
        &mut output,
        data,
        src_width,
        src_height,
        dst_width,
        dst_height,
        color_type,
        algorithm,
    )?;
    Ok(output)
}

/// Resize an image into a caller-provided buffer.
///
/// The `output` buffer will be cleared and reused, allowing callers to avoid
/// repeated allocations across multiple resize operations.
///
/// # Arguments
///
/// * `output` - Buffer to write resized data into (will be cleared)
/// * `data` - Raw pixel data (row-major order)
/// * `src_width` - Source image width in pixels
/// * `src_height` - Source image height in pixels
/// * `dst_width` - Destination image width in pixels
/// * `dst_height` - Destination image height in pixels
/// * `color_type` - Color type of the pixel data
/// * `algorithm` - Resizing algorithm to use
#[allow(clippy::too_many_arguments)]
pub fn resize_into(
    output: &mut Vec<u8>,
    data: &[u8],
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    color_type: ColorType,
    algorithm: ResizeAlgorithm,
) -> Result<()> {
    // Validate source dimensions
    if src_width == 0 || src_height == 0 {
        return Err(Error::InvalidDimensions {
            width: src_width,
            height: src_height,
        });
    }

    // Validate destination dimensions
    if dst_width == 0 || dst_height == 0 {
        return Err(Error::InvalidDimensions {
            width: dst_width,
            height: dst_height,
        });
    }

    // Check maximum dimensions
    if src_width > MAX_DIMENSION
        || src_height > MAX_DIMENSION
        || dst_width > MAX_DIMENSION
        || dst_height > MAX_DIMENSION
    {
        return Err(Error::ImageTooLarge {
            width: src_width.max(dst_width),
            height: src_height.max(dst_height),
            max: MAX_DIMENSION,
        });
    }

    let bytes_per_pixel = color_type.bytes_per_pixel();

    // Validate input data length
    let expected_len = (src_width as usize)
        .checked_mul(src_height as usize)
        .and_then(|v| v.checked_mul(bytes_per_pixel))
        .ok_or(Error::InvalidDataLength {
            expected: usize::MAX,
            actual: data.len(),
        })?;

    if data.len() != expected_len {
        return Err(Error::InvalidDataLength {
            expected: expected_len,
            actual: data.len(),
        });
    }

    // Calculate output size
    let output_len = (dst_width as usize)
        .checked_mul(dst_height as usize)
        .and_then(|v| v.checked_mul(bytes_per_pixel))
        .ok_or(Error::InvalidDataLength {
            expected: usize::MAX,
            actual: 0,
        })?;

    output.clear();
    output.resize(output_len, 0);

    // Dispatch to appropriate algorithm
    match algorithm {
        ResizeAlgorithm::Nearest => resize_nearest(
            output,
            data,
            src_width as usize,
            src_height as usize,
            dst_width as usize,
            dst_height as usize,
            bytes_per_pixel,
        ),
        ResizeAlgorithm::Bilinear => resize_bilinear(
            output,
            data,
            src_width as usize,
            src_height as usize,
            dst_width as usize,
            dst_height as usize,
            bytes_per_pixel,
        ),
        ResizeAlgorithm::Lanczos3 => resize_lanczos3(
            output,
            data,
            src_width as usize,
            src_height as usize,
            dst_width as usize,
            dst_height as usize,
            bytes_per_pixel,
        ),
    }

    Ok(())
}

/// Nearest neighbor resizing - fastest, pixelated results.
fn resize_nearest(
    output: &mut [u8],
    data: &[u8],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    bytes_per_pixel: usize,
) {
    let x_ratio = src_width as f32 / dst_width as f32;
    let y_ratio = src_height as f32 / dst_height as f32;

    for dst_y in 0..dst_height {
        let src_y = ((dst_y as f32 + 0.5) * y_ratio - 0.5)
            .round()
            .max(0.0)
            .min((src_height - 1) as f32) as usize;

        for dst_x in 0..dst_width {
            let src_x = ((dst_x as f32 + 0.5) * x_ratio - 0.5)
                .round()
                .max(0.0)
                .min((src_width - 1) as f32) as usize;

            let src_idx = (src_y * src_width + src_x) * bytes_per_pixel;
            let dst_idx = (dst_y * dst_width + dst_x) * bytes_per_pixel;

            output[dst_idx..dst_idx + bytes_per_pixel]
                .copy_from_slice(&data[src_idx..src_idx + bytes_per_pixel]);
        }
    }
}

/// Bilinear interpolation - good balance of quality and speed.
fn resize_bilinear(
    output: &mut [u8],
    data: &[u8],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    bytes_per_pixel: usize,
) {
    let x_ratio = if dst_width > 1 {
        (src_width - 1) as f32 / (dst_width - 1) as f32
    } else {
        0.0
    };
    let y_ratio = if dst_height > 1 {
        (src_height - 1) as f32 / (dst_height - 1) as f32
    } else {
        0.0
    };

    for dst_y in 0..dst_height {
        let src_y_f = dst_y as f32 * y_ratio;
        let src_y0 = src_y_f.floor() as usize;
        let src_y1 = (src_y0 + 1).min(src_height - 1);
        let y_frac = src_y_f - src_y0 as f32;

        for dst_x in 0..dst_width {
            let src_x_f = dst_x as f32 * x_ratio;
            let src_x0 = src_x_f.floor() as usize;
            let src_x1 = (src_x0 + 1).min(src_width - 1);
            let x_frac = src_x_f - src_x0 as f32;

            // Get the four surrounding pixels
            let idx00 = (src_y0 * src_width + src_x0) * bytes_per_pixel;
            let idx01 = (src_y0 * src_width + src_x1) * bytes_per_pixel;
            let idx10 = (src_y1 * src_width + src_x0) * bytes_per_pixel;
            let idx11 = (src_y1 * src_width + src_x1) * bytes_per_pixel;

            let dst_idx = (dst_y * dst_width + dst_x) * bytes_per_pixel;

            // Interpolate each channel
            for c in 0..bytes_per_pixel {
                let p00 = data[idx00 + c] as f32;
                let p01 = data[idx01 + c] as f32;
                let p10 = data[idx10 + c] as f32;
                let p11 = data[idx11 + c] as f32;

                // Bilinear interpolation
                let top = p00 * (1.0 - x_frac) + p01 * x_frac;
                let bottom = p10 * (1.0 - x_frac) + p11 * x_frac;
                let value = top * (1.0 - y_frac) + bottom * y_frac;

                output[dst_idx + c] = value.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
}

/// Lanczos kernel function.
#[inline]
fn lanczos_kernel(x: f32, a: f32) -> f32 {
    if x.abs() < f32::EPSILON {
        1.0
    } else if x.abs() >= a {
        0.0
    } else {
        let pi_x = PI * x;
        let pi_x_a = PI * x / a;
        (a * pi_x.sin() * pi_x_a.sin()) / (pi_x * pi_x_a)
    }
}

/// Lanczos3 resampling - highest quality, slowest.
fn resize_lanczos3(
    output: &mut [u8],
    data: &[u8],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    bytes_per_pixel: usize,
) {
    const A: f32 = 3.0; // Lanczos3 uses a=3

    let x_scale = src_width as f32 / dst_width as f32;
    let y_scale = src_height as f32 / dst_height as f32;

    // For downscaling, we need to expand the kernel support
    let x_support = if x_scale > 1.0 { A * x_scale } else { A };
    let y_support = if y_scale > 1.0 { A * y_scale } else { A };

    for dst_y in 0..dst_height {
        // Map destination y to source y (center of pixel)
        let src_y_center = (dst_y as f32 + 0.5) * y_scale - 0.5;

        // Calculate the range of source pixels to sample
        let y_start = ((src_y_center - y_support).floor() as isize).max(0) as usize;
        let y_end = ((src_y_center + y_support).ceil() as usize + 1).min(src_height);

        for dst_x in 0..dst_width {
            let src_x_center = (dst_x as f32 + 0.5) * x_scale - 0.5;

            let x_start = ((src_x_center - x_support).floor() as isize).max(0) as usize;
            let x_end = ((src_x_center + x_support).ceil() as usize + 1).min(src_width);

            // Accumulate weighted samples for each channel
            let mut channel_sums = [0.0f32; 4]; // Max 4 channels (RGBA)
            let mut weight_sum = 0.0f32;

            for src_y in y_start..y_end {
                let dy = (src_y as f32 - src_y_center) / y_scale.max(1.0);
                let ky = lanczos_kernel(dy, A);

                for src_x in x_start..x_end {
                    let dx = (src_x as f32 - src_x_center) / x_scale.max(1.0);
                    let kx = lanczos_kernel(dx, A);

                    let weight = kx * ky;
                    weight_sum += weight;

                    let src_idx = (src_y * src_width + src_x) * bytes_per_pixel;
                    for c in 0..bytes_per_pixel {
                        channel_sums[c] += data[src_idx + c] as f32 * weight;
                    }
                }
            }

            // Normalize and write output
            let dst_idx = (dst_y * dst_width + dst_x) * bytes_per_pixel;
            if weight_sum.abs() > f32::EPSILON {
                for c in 0..bytes_per_pixel {
                    let value = channel_sums[c] / weight_sum;
                    output[dst_idx + c] = value.round().clamp(0.0, 255.0) as u8;
                }
            } else {
                // Fallback to nearest neighbor if weights sum to zero
                let src_x = (src_x_center.round() as usize).clamp(0, src_width - 1);
                let src_y = (src_y_center.round() as usize).clamp(0, src_height - 1);
                let src_idx = (src_y * src_width + src_x) * bytes_per_pixel;
                output[dst_idx..dst_idx + bytes_per_pixel]
                    .copy_from_slice(&data[src_idx..src_idx + bytes_per_pixel]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resize_nearest_basic() {
        // 2x2 RGBA image -> 4x4
        let pixels = vec![
            255, 0, 0, 255, // Red
            0, 255, 0, 255, // Green
            0, 0, 255, 255, // Blue
            255, 255, 0, 255, // Yellow
        ];

        let result = resize(
            &pixels,
            2,
            2,
            4,
            4,
            ColorType::Rgba,
            ResizeAlgorithm::Nearest,
        )
        .unwrap();
        assert_eq!(result.len(), 4 * 4 * 4);
    }

    #[test]
    fn test_resize_bilinear_basic() {
        // 2x2 RGB image -> 4x4
        let pixels = vec![
            255, 0, 0, // Red
            0, 255, 0, // Green
            0, 0, 255, // Blue
            255, 255, 0, // Yellow
        ];

        let result = resize(
            &pixels,
            2,
            2,
            4,
            4,
            ColorType::Rgb,
            ResizeAlgorithm::Bilinear,
        )
        .unwrap();
        assert_eq!(result.len(), 4 * 4 * 3);
    }

    #[test]
    fn test_resize_lanczos3_basic() {
        // 4x4 grayscale image -> 2x2
        let pixels = vec![0u8; 4 * 4];

        let result = resize(
            &pixels,
            4,
            4,
            2,
            2,
            ColorType::Gray,
            ResizeAlgorithm::Lanczos3,
        )
        .unwrap();
        assert_eq!(result.len(), 2 * 2);
    }

    #[test]
    fn test_resize_same_size() {
        // Resize to same size should be near-identity
        let pixels = vec![128u8; 8 * 8 * 4];

        let result = resize(
            &pixels,
            8,
            8,
            8,
            8,
            ColorType::Rgba,
            ResizeAlgorithm::Bilinear,
        )
        .unwrap();
        assert_eq!(result.len(), pixels.len());
    }

    #[test]
    fn test_resize_downscale() {
        // 16x16 -> 4x4
        let pixels: Vec<u8> = (0..16 * 16 * 3).map(|i| (i % 256) as u8).collect();

        let result = resize(
            &pixels,
            16,
            16,
            4,
            4,
            ColorType::Rgb,
            ResizeAlgorithm::Lanczos3,
        )
        .unwrap();
        assert_eq!(result.len(), 4 * 4 * 3);
    }

    #[test]
    fn test_resize_upscale() {
        // 4x4 -> 16x16
        let pixels: Vec<u8> = (0..4 * 4 * 3).map(|i| (i % 256) as u8).collect();

        let result = resize(
            &pixels,
            4,
            4,
            16,
            16,
            ColorType::Rgb,
            ResizeAlgorithm::Bilinear,
        )
        .unwrap();
        assert_eq!(result.len(), 16 * 16 * 3);
    }

    #[test]
    fn test_resize_non_square() {
        // 8x4 -> 4x8
        let pixels = vec![200u8; 8 * 4 * 4];

        let result = resize(
            &pixels,
            8,
            4,
            4,
            8,
            ColorType::Rgba,
            ResizeAlgorithm::Nearest,
        )
        .unwrap();
        assert_eq!(result.len(), 4 * 8 * 4);
    }

    #[test]
    fn test_resize_invalid_src_dimensions() {
        let pixels = vec![0u8; 0];
        let result = resize(
            &pixels,
            0,
            10,
            5,
            5,
            ColorType::Rgb,
            ResizeAlgorithm::Nearest,
        );
        assert!(matches!(result, Err(Error::InvalidDimensions { .. })));
    }

    #[test]
    fn test_resize_invalid_dst_dimensions() {
        let pixels = vec![0u8; 10 * 10 * 3];
        let result = resize(
            &pixels,
            10,
            10,
            0,
            5,
            ColorType::Rgb,
            ResizeAlgorithm::Nearest,
        );
        assert!(matches!(result, Err(Error::InvalidDimensions { .. })));
    }

    #[test]
    fn test_resize_invalid_data_length() {
        let pixels = vec![0u8; 10]; // Wrong size for 10x10 RGB
        let result = resize(
            &pixels,
            10,
            10,
            5,
            5,
            ColorType::Rgb,
            ResizeAlgorithm::Nearest,
        );
        assert!(matches!(result, Err(Error::InvalidDataLength { .. })));
    }

    #[test]
    fn test_resize_1x1_to_larger() {
        // 1x1 -> 4x4 (edge case)
        let pixels = vec![255, 128, 64, 255]; // RGBA

        let result = resize(
            &pixels,
            1,
            1,
            4,
            4,
            ColorType::Rgba,
            ResizeAlgorithm::Bilinear,
        )
        .unwrap();
        assert_eq!(result.len(), 4 * 4 * 4);

        // All pixels should be the same as the source (single color)
        for i in 0..16 {
            assert_eq!(result[i * 4], 255);
            assert_eq!(result[i * 4 + 1], 128);
            assert_eq!(result[i * 4 + 2], 64);
            assert_eq!(result[i * 4 + 3], 255);
        }
    }

    #[test]
    fn test_resize_to_1x1() {
        // 4x4 -> 1x1 (edge case)
        let pixels = vec![128u8; 4 * 4 * 3];

        let result = resize(
            &pixels,
            4,
            4,
            1,
            1,
            ColorType::Rgb,
            ResizeAlgorithm::Lanczos3,
        )
        .unwrap();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_resize_gray_alpha() {
        // Test GrayAlpha (2 bytes per pixel)
        let pixels = vec![100, 200, 150, 250]; // 2x1 GrayAlpha

        let result = resize(
            &pixels,
            2,
            1,
            4,
            2,
            ColorType::GrayAlpha,
            ResizeAlgorithm::Bilinear,
        )
        .unwrap();
        assert_eq!(result.len(), 4 * 2 * 2);
    }

    #[test]
    fn test_resize_into_reuses_buffer() {
        let mut output = Vec::with_capacity(1024);
        let pixels = vec![128u8; 8 * 8 * 4];

        resize_into(
            &mut output,
            &pixels,
            8,
            8,
            4,
            4,
            ColorType::Rgba,
            ResizeAlgorithm::Nearest,
        )
        .unwrap();

        let first_cap = output.capacity();
        assert_eq!(output.len(), 4 * 4 * 4);

        // Resize again with same output buffer
        resize_into(
            &mut output,
            &pixels,
            8,
            8,
            4,
            4,
            ColorType::Rgba,
            ResizeAlgorithm::Bilinear,
        )
        .unwrap();

        // Capacity should be preserved (buffer reuse)
        assert!(output.capacity() >= first_cap);
    }

    #[test]
    fn test_resize_algorithm_default() {
        // Default should be Bilinear
        assert_eq!(ResizeAlgorithm::default(), ResizeAlgorithm::Bilinear);
    }

    #[test]
    fn test_lanczos_kernel() {
        // At x=0, kernel should be 1
        assert!((lanczos_kernel(0.0, 3.0) - 1.0).abs() < 0.001);

        // At x >= a, kernel should be 0
        assert!(lanczos_kernel(3.0, 3.0).abs() < 0.001);
        assert!(lanczos_kernel(4.0, 3.0).abs() < f32::EPSILON);

        // Kernel should be symmetric
        assert!((lanczos_kernel(1.5, 3.0) - lanczos_kernel(-1.5, 3.0)).abs() < 0.001);
    }

    #[test]
    fn test_resize_large_dimension_error() {
        let pixels = vec![0u8; 3];
        let result = resize(
            &pixels,
            1,
            1,
            (1 << 25) as u32,
            1,
            ColorType::Rgb,
            ResizeAlgorithm::Nearest,
        );
        assert!(matches!(result, Err(Error::ImageTooLarge { .. })));
    }

    #[test]
    fn test_all_algorithms_produce_valid_output() {
        let pixels: Vec<u8> = (0..32 * 32 * 4).map(|i| (i % 256) as u8).collect();

        for algo in [
            ResizeAlgorithm::Nearest,
            ResizeAlgorithm::Bilinear,
            ResizeAlgorithm::Lanczos3,
        ] {
            let result = resize(&pixels, 32, 32, 16, 16, ColorType::Rgba, algo).unwrap();
            assert_eq!(result.len(), 16 * 16 * 4);

            // All values should be valid u8
            assert!(!result.is_empty());
        }
    }
}
