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
//! use pixo::resize::{resize, ResizeOptions, ResizeAlgorithm};
//! use pixo::ColorType;
//!
//! // Resize a 100x100 RGBA image to 50x50 using Lanczos3
//! let pixels = vec![128u8; 100 * 100 * 4];
//! let options = ResizeOptions::builder(100, 100)
//!     .dst(50, 50)
//!     .color_type(ColorType::Rgba)
//!     .algorithm(ResizeAlgorithm::Lanczos3)
//!     .build();
//! let resized = resize(&pixels, &options).unwrap();
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

/// Options for image resizing operations.
///
/// Use [`ResizeOptions::builder()`] to create options with a fluent API.
///
/// # Example
///
/// ```rust
/// use pixo::resize::{resize, ResizeOptions, ResizeAlgorithm};
/// use pixo::ColorType;
///
/// let pixels = vec![128u8; 100 * 100 * 4];
/// let options = ResizeOptions::builder(100, 100)
///     .dst(50, 50)
///     .color_type(ColorType::Rgba)
///     .algorithm(ResizeAlgorithm::Lanczos3)
///     .build();
/// let resized = resize(&pixels, &options).unwrap();
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResizeOptions {
    /// Source image width in pixels.
    pub src_width: u32,
    /// Source image height in pixels.
    pub src_height: u32,
    /// Destination image width in pixels.
    pub dst_width: u32,
    /// Destination image height in pixels.
    pub dst_height: u32,
    /// Color type of the pixel data.
    pub color_type: ColorType,
    /// Resizing algorithm to use.
    pub algorithm: ResizeAlgorithm,
}

impl ResizeOptions {
    /// Create a builder for [`ResizeOptions`].
    ///
    /// The source dimensions are required; destination defaults to matching source
    /// (no resize), color type defaults to RGBA, and algorithm defaults to Bilinear.
    pub fn builder(src_width: u32, src_height: u32) -> ResizeOptionsBuilder {
        ResizeOptionsBuilder::new(src_width, src_height)
    }
}

/// Builder for [`ResizeOptions`].
#[derive(Debug, Clone)]
pub struct ResizeOptionsBuilder {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    color_type: ColorType,
    algorithm: ResizeAlgorithm,
}

impl ResizeOptionsBuilder {
    /// Create a new builder with source dimensions.
    pub fn new(src_width: u32, src_height: u32) -> Self {
        Self {
            src_width,
            src_height,
            dst_width: src_width,
            dst_height: src_height,
            color_type: ColorType::Rgba,
            algorithm: ResizeAlgorithm::default(),
        }
    }

    /// Set the destination dimensions.
    pub fn dst(mut self, width: u32, height: u32) -> Self {
        self.dst_width = width;
        self.dst_height = height;
        self
    }

    /// Set the color type of the pixel data.
    pub fn color_type(mut self, color_type: ColorType) -> Self {
        self.color_type = color_type;
        self
    }

    /// Set the resizing algorithm.
    pub fn algorithm(mut self, algorithm: ResizeAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Build the [`ResizeOptions`].
    #[must_use]
    pub fn build(self) -> ResizeOptions {
        ResizeOptions {
            src_width: self.src_width,
            src_height: self.src_height,
            dst_width: self.dst_width,
            dst_height: self.dst_height,
            color_type: self.color_type,
            algorithm: self.algorithm,
        }
    }
}

/// Resize an image to new dimensions.
///
/// # Arguments
///
/// * `data` - Raw pixel data (row-major order)
/// * `options` - Resize options specifying dimensions, color type, and algorithm
///
/// # Returns
///
/// Resized pixel data with the same color type.
///
/// # Errors
///
/// Returns an error if dimensions are invalid or data length doesn't match.
#[must_use = "resizing produces pixel data that should be used"]
pub fn resize(data: &[u8], options: &ResizeOptions) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    resize_into(&mut output, data, options)?;
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
/// * `options` - Resize options specifying dimensions, color type, and algorithm
#[must_use = "this `Result` may indicate a resize error"]
pub fn resize_into(output: &mut Vec<u8>, data: &[u8], options: &ResizeOptions) -> Result<()> {
    resize_impl(
        output,
        data,
        options.src_width,
        options.src_height,
        options.dst_width,
        options.dst_height,
        options.color_type,
        options.algorithm,
    )
}

/// Internal resize implementation.
#[allow(clippy::too_many_arguments)]
fn resize_impl(
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

/// Precomputed contribution for a source pixel to destination pixels.
#[derive(Clone)]
struct Contribution {
    /// Starting source pixel index
    start: usize,
    /// Weights for each source pixel (already normalized)
    weights: Vec<f32>,
}

/// Precompute Lanczos3 contributions for a 1D resample.
/// Returns a vec of Contribution, one for each destination pixel.
fn precompute_contributions(src_size: usize, dst_size: usize) -> Vec<Contribution> {
    const A: f32 = 3.0;

    let scale = src_size as f32 / dst_size as f32;
    // For downscaling, expand the kernel support proportionally
    let filter_scale = scale.max(1.0);
    let support = A * filter_scale;

    let mut contributions = Vec::with_capacity(dst_size);

    for dst_idx in 0..dst_size {
        // Map destination pixel center to source coordinates
        let src_center = (dst_idx as f32 + 0.5) * scale - 0.5;

        // Determine the range of source pixels that contribute
        let start = ((src_center - support).floor() as isize).max(0) as usize;
        let end = ((src_center + support).ceil() as usize + 1).min(src_size);

        // Compute weights
        let mut weights = Vec::with_capacity(end - start);
        let mut weight_sum = 0.0f32;

        for src_idx in start..end {
            let x = (src_idx as f32 - src_center) / filter_scale;
            let w = lanczos_kernel(x, A);
            weights.push(w);
            weight_sum += w;
        }

        // Normalize weights
        if weight_sum.abs() > f32::EPSILON {
            for w in &mut weights {
                *w /= weight_sum;
            }
        }

        contributions.push(Contribution { start, weights });
    }

    contributions
}

/// Apply 1D horizontal resampling to a single row.
#[inline]
fn resample_row_horizontal(
    src_row: &[u8],
    dst_row: &mut [u8],
    contributions: &[Contribution],
    bytes_per_pixel: usize,
) {
    for (dst_x, contrib) in contributions.iter().enumerate() {
        let dst_idx = dst_x * bytes_per_pixel;

        // Accumulate weighted samples for each channel
        let mut channel_sums = [0.0f32; 4];

        for (i, &weight) in contrib.weights.iter().enumerate() {
            let src_x = contrib.start + i;
            let src_idx = src_x * bytes_per_pixel;
            for c in 0..bytes_per_pixel {
                channel_sums[c] += src_row[src_idx + c] as f32 * weight;
            }
        }

        for c in 0..bytes_per_pixel {
            dst_row[dst_idx + c] = channel_sums[c].round().clamp(0.0, 255.0) as u8;
        }
    }
}

/// Apply 1D vertical resampling to produce one destination row.
#[inline]
fn resample_column_vertical(
    temp: &[u8],
    dst_row: &mut [u8],
    contrib: &Contribution,
    temp_width: usize,
    bytes_per_pixel: usize,
) {
    let row_stride = temp_width * bytes_per_pixel;

    for dst_x in 0..temp_width {
        let dst_idx = dst_x * bytes_per_pixel;
        let mut channel_sums = [0.0f32; 4];

        for (i, &weight) in contrib.weights.iter().enumerate() {
            let src_y = contrib.start + i;
            let src_idx = src_y * row_stride + dst_x * bytes_per_pixel;
            for c in 0..bytes_per_pixel {
                channel_sums[c] += temp[src_idx + c] as f32 * weight;
            }
        }

        for c in 0..bytes_per_pixel {
            dst_row[dst_idx + c] = channel_sums[c].round().clamp(0.0, 255.0) as u8;
        }
    }
}

/// Lanczos3 resampling using separable filtering (horizontal then vertical).
/// This is O(2n) per pixel instead of O(n²), providing massive speedup.
fn resize_lanczos3(
    output: &mut [u8],
    data: &[u8],
    src_width: usize,
    src_height: usize,
    dst_width: usize,
    dst_height: usize,
    bytes_per_pixel: usize,
) {
    // Precompute contribution weights (computed once, reused for all rows/cols)
    let h_contribs = precompute_contributions(src_width, dst_width);
    let v_contribs = precompute_contributions(src_height, dst_height);

    // Intermediate buffer: src_height rows × dst_width columns
    let temp_size = src_height * dst_width * bytes_per_pixel;
    let mut temp = vec![0u8; temp_size];

    // Pass 1: Horizontal resampling (parallel when available)
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;

        let src_row_stride = src_width * bytes_per_pixel;
        let temp_row_stride = dst_width * bytes_per_pixel;

        temp.par_chunks_mut(temp_row_stride)
            .enumerate()
            .for_each(|(y, temp_row)| {
                let src_start = y * src_row_stride;
                let src_row = &data[src_start..src_start + src_row_stride];
                resample_row_horizontal(src_row, temp_row, &h_contribs, bytes_per_pixel);
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        let src_row_stride = src_width * bytes_per_pixel;
        let temp_row_stride = dst_width * bytes_per_pixel;

        for y in 0..src_height {
            let src_start = y * src_row_stride;
            let src_row = &data[src_start..src_start + src_row_stride];
            let temp_start = y * temp_row_stride;
            let temp_row = &mut temp[temp_start..temp_start + temp_row_stride];
            resample_row_horizontal(src_row, temp_row, &h_contribs, bytes_per_pixel);
        }
    }

    // Pass 2: Vertical resampling (parallel when available)
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;

        let dst_row_stride = dst_width * bytes_per_pixel;

        output
            .par_chunks_mut(dst_row_stride)
            .enumerate()
            .for_each(|(dst_y, dst_row)| {
                resample_column_vertical(
                    &temp,
                    dst_row,
                    &v_contribs[dst_y],
                    dst_width,
                    bytes_per_pixel,
                );
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        let dst_row_stride = dst_width * bytes_per_pixel;

        for dst_y in 0..dst_height {
            let dst_start = dst_y * dst_row_stride;
            let dst_row = &mut output[dst_start..dst_start + dst_row_stride];
            resample_column_vertical(
                &temp,
                dst_row,
                &v_contribs[dst_y],
                dst_width,
                bytes_per_pixel,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper for tests: builds ResizeOptions from positional args
    fn test_resize(
        data: &[u8],
        src_width: u32,
        src_height: u32,
        dst_width: u32,
        dst_height: u32,
        color_type: ColorType,
        algorithm: ResizeAlgorithm,
    ) -> Result<Vec<u8>> {
        let options = ResizeOptions::builder(src_width, src_height)
            .dst(dst_width, dst_height)
            .color_type(color_type)
            .algorithm(algorithm)
            .build();
        resize(data, &options)
    }

    /// Helper for tests: resize_into with positional args
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    fn test_resize_into(
        output: &mut Vec<u8>,
        data: &[u8],
        src_width: u32,
        src_height: u32,
        dst_width: u32,
        dst_height: u32,
        color_type: ColorType,
        algorithm: ResizeAlgorithm,
    ) -> Result<()> {
        let options = ResizeOptions::builder(src_width, src_height)
            .dst(dst_width, dst_height)
            .color_type(color_type)
            .algorithm(algorithm)
            .build();
        resize_into(output, data, &options)
    }

    #[test]
    fn test_resize_nearest_basic() {
        // 2x2 RGBA image -> 4x4
        let pixels = vec![
            255, 0, 0, 255, // Red
            0, 255, 0, 255, // Green
            0, 0, 255, 255, // Blue
            255, 255, 0, 255, // Yellow
        ];

        let result = test_resize(
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

        let result = test_resize(
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

        let result = test_resize(
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

        let result = test_resize(
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

        let result = test_resize(
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

        let result = test_resize(
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

        let result = test_resize(
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
        let result = test_resize(
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
        let result = test_resize(
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
        let result = test_resize(
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

        let result = test_resize(
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

        let result = test_resize(
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

        let result = test_resize(
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
    fn test_resize_buffer_reuse() {
        let mut output = Vec::with_capacity(1024);
        let pixels = vec![128u8; 8 * 8 * 4];

        let options1 = ResizeOptions::builder(8, 8)
            .dst(4, 4)
            .color_type(ColorType::Rgba)
            .algorithm(ResizeAlgorithm::Nearest)
            .build();
        resize_into(&mut output, &pixels, &options1).unwrap();

        let first_cap = output.capacity();
        assert_eq!(output.len(), 4 * 4 * 4);

        // Resize again with same output buffer
        let options2 = ResizeOptions::builder(8, 8)
            .dst(4, 4)
            .color_type(ColorType::Rgba)
            .algorithm(ResizeAlgorithm::Bilinear)
            .build();
        resize_into(&mut output, &pixels, &options2).unwrap();

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
        let result = test_resize(
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
            let result = test_resize(&pixels, 32, 32, 16, 16, ColorType::Rgba, algo).unwrap();
            assert_eq!(result.len(), 16 * 16 * 4);

            // All values should be valid u8
            assert!(!result.is_empty());
        }
    }

    // ============ Tests for separable filtering and precomputed contributions ============

    #[test]
    fn test_precompute_contributions_basic() {
        // Test that contributions are computed correctly for simple cases
        let contribs = precompute_contributions(100, 50); // 2x downscale

        assert_eq!(contribs.len(), 50);

        // Each contribution should have weights that sum to ~1.0
        for contrib in &contribs {
            let sum: f32 = contrib.weights.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "Weights should sum to 1.0, got {sum}"
            );
        }
    }

    #[test]
    fn test_precompute_contributions_upscale() {
        // Test upscaling contributions
        let contribs = precompute_contributions(50, 100); // 2x upscale

        assert_eq!(contribs.len(), 100);

        for contrib in &contribs {
            let sum: f32 = contrib.weights.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "Weights should sum to 1.0, got {sum}"
            );
            // Upscaling should have reasonable kernel support (Lanczos3 has radius 3)
            assert!(
                contrib.weights.len() <= 8,
                "Kernel too large: {}",
                contrib.weights.len()
            );
        }
    }

    #[test]
    fn test_precompute_contributions_same_size() {
        // Same size should produce identity-like contributions
        let contribs = precompute_contributions(100, 100);

        assert_eq!(contribs.len(), 100);

        for (i, contrib) in contribs.iter().enumerate() {
            // The primary weight should be very close to 1.0 at the center
            let max_weight = contrib.weights.iter().cloned().fold(0.0f32, f32::max);
            assert!(
                max_weight > 0.9,
                "Max weight at {i} should be near 1.0, got {max_weight}"
            );
        }
    }

    #[test]
    fn test_lanczos3_large_downscale() {
        // Test significant downscaling (simulates the 5000x6000 -> smaller case)
        let src_w = 500;
        let src_h = 600;
        let dst_w = 100;
        let dst_h = 120;

        let pixels: Vec<u8> = (0..(src_w * src_h * 4))
            .map(|i| ((i * 7) % 256) as u8)
            .collect();

        let result = test_resize(
            &pixels,
            src_w as u32,
            src_h as u32,
            dst_w as u32,
            dst_h as u32,
            ColorType::Rgba,
            ResizeAlgorithm::Lanczos3,
        )
        .unwrap();

        assert_eq!(result.len(), dst_w * dst_h * 4);

        // Verify output is reasonable (not all zeros, not all 255s)
        let sum: u64 = result.iter().map(|&x| x as u64).sum();
        let avg = sum as f64 / result.len() as f64;
        assert!(
            avg > 10.0 && avg < 245.0,
            "Average pixel value {avg} is suspicious"
        );
    }

    #[test]
    fn test_lanczos3_preserves_solid_color() {
        // A solid color image should remain solid after resize
        let color = [100u8, 150, 200, 255];
        let pixels: Vec<u8> = (0..64 * 64).flat_map(|_| color).collect();

        let result = test_resize(
            &pixels,
            64,
            64,
            32,
            32,
            ColorType::Rgba,
            ResizeAlgorithm::Lanczos3,
        )
        .unwrap();

        // All pixels should be very close to the original color
        for i in 0..(32 * 32) {
            let idx = i * 4;
            for c in 0..4 {
                let diff = (result[idx + c] as i32 - color[c] as i32).abs();
                assert!(diff <= 1, "Color drift too large at pixel {i}, channel {c}");
            }
        }
    }

    #[test]
    fn test_lanczos3_upscale_quality() {
        // Test that Lanczos3 produces smooth gradients on upscale
        // Create a simple 2x2 gradient
        let pixels = vec![
            0, 0, 0, 255, // Black
            255, 255, 255, 255, // White
            128, 128, 128, 255, // Gray
            64, 64, 64, 255, // Dark gray
        ];

        let result = test_resize(
            &pixels,
            2,
            2,
            8,
            8,
            ColorType::Rgba,
            ResizeAlgorithm::Lanczos3,
        )
        .unwrap();

        assert_eq!(result.len(), 8 * 8 * 4);

        // Corner pixels should be close to original values
        // Top-left should be near black
        assert!(result[0] < 30, "Top-left should be near black");
        // Top-right (pixel 7) should be near white
        let tr_idx = 7 * 4;
        assert!(result[tr_idx] > 200, "Top-right should be near white");
    }

    #[test]
    fn test_lanczos3_asymmetric_resize() {
        // Test non-uniform scaling (different x and y ratios)
        let pixels: Vec<u8> = (0..100 * 50 * 3).map(|i| (i % 256) as u8).collect();

        let result = test_resize(
            &pixels,
            100,
            50,
            25,
            200, // 4x downscale in X, 4x upscale in Y
            ColorType::Rgb,
            ResizeAlgorithm::Lanczos3,
        )
        .unwrap();

        assert_eq!(result.len(), 25 * 200 * 3);
    }

    #[test]
    fn test_resample_row_horizontal_basic() {
        // Test the horizontal resampling helper directly
        let contribs = precompute_contributions(4, 2);
        let src_row = vec![100u8, 150, 200, 250]; // 4 gray pixels
        let mut dst_row = vec![0u8; 2];

        resample_row_horizontal(&src_row, &mut dst_row, &contribs, 1);

        // Output should be reasonable averages
        assert!(dst_row[0] > 50 && dst_row[0] < 200);
        assert!(dst_row[1] > 100 && dst_row[1] < 255);
    }

    #[test]
    fn test_lanczos3_1x1_edge_case() {
        // 1x1 -> 10x10 with Lanczos3
        let pixels = vec![128, 64, 192, 255]; // Single RGBA pixel

        let result = test_resize(
            &pixels,
            1,
            1,
            10,
            10,
            ColorType::Rgba,
            ResizeAlgorithm::Lanczos3,
        )
        .unwrap();

        assert_eq!(result.len(), 10 * 10 * 4);

        // All pixels should match the source (single color expansion)
        for i in 0..100 {
            let idx = i * 4;
            assert_eq!(result[idx], 128, "R mismatch at {i}");
            assert_eq!(result[idx + 1], 64, "G mismatch at {i}");
            assert_eq!(result[idx + 2], 192, "B mismatch at {i}");
            assert_eq!(result[idx + 3], 255, "A mismatch at {i}");
        }
    }

    #[test]
    fn test_contribution_bounds_are_valid() {
        // Ensure precomputed contributions don't go out of bounds
        for (src, dst) in [(100, 10), (10, 100), (1000, 50), (50, 1000)] {
            let contribs = precompute_contributions(src, dst);

            for (i, contrib) in contribs.iter().enumerate() {
                assert!(
                    contrib.start < src,
                    "Contribution {} start {} >= src {}",
                    i,
                    contrib.start,
                    src
                );
                let end = contrib.start + contrib.weights.len();
                assert!(end <= src, "Contribution {i} end {end} > src {src}");
            }
        }
    }

    #[test]
    fn test_lanczos3_extreme_downscale() {
        // 1000x1000 -> 10x10 (100x downscale)
        let pixels: Vec<u8> = (0..1000 * 1000 * 3).map(|i| (i % 256) as u8).collect();

        let result = test_resize(
            &pixels,
            1000,
            1000,
            10,
            10,
            ColorType::Rgb,
            ResizeAlgorithm::Lanczos3,
        )
        .unwrap();

        assert_eq!(result.len(), 10 * 10 * 3);
    }

    #[test]
    fn test_lanczos3_prime_dimensions() {
        // Test with prime number dimensions (edge case for algorithms)
        let pixels: Vec<u8> = (0..97 * 89 * 4).map(|i| (i % 256) as u8).collect();

        let result = test_resize(
            &pixels,
            97,
            89,
            41,
            37,
            ColorType::Rgba,
            ResizeAlgorithm::Lanczos3,
        )
        .unwrap();

        assert_eq!(result.len(), 41 * 37 * 4);
    }
}
