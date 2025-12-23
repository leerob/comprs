//! PNG encoder implementation.
//!
//! Implements PNG encoding according to the PNG specification (RFC 2083).

mod bit_depth;
pub mod chunk;
pub mod filter;

use crate::color::ColorType;
use crate::compress::deflate::deflate_zlib_packed;
#[cfg(feature = "timing")]
use crate::compress::deflate::{deflate_zlib_packed_with_stats, DeflateStats};
use crate::error::{Error, Result};
// bit_depth module reserved for future bit-depth reductions.

/// PNG file signature (magic bytes).
const PNG_SIGNATURE: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

/// Maximum supported image dimension.
const MAX_DIMENSION: u32 = 1 << 24; // 16 million pixels

/// PNG encoding options.
#[derive(Debug, Clone)]
pub struct PngOptions {
    /// Compression level (1-9, default 6).
    pub compression_level: u8,
    /// Filter selection strategy.
    pub filter_strategy: FilterStrategy,
    /// If true, zero out color channels for fully transparent pixels to improve compressibility.
    /// Only applies to color types with alpha (RGBA, GrayAlpha).
    pub optimize_alpha: bool,
    /// If true, attempt to reduce color type (e.g., RGB→Gray, RGBA→RGB/GrayAlpha) when lossless-safe.
    pub reduce_color_type: bool,
    /// If true, strip non-critical ancillary chunks (tEXt, iTXt, zTXt, time) to reduce size.
    pub strip_metadata: bool,
    /// If true, attempt palette reduction when <=256 colors (writes PLTE/tRNS).
    pub reduce_palette: bool,
}

impl Default for PngOptions {
    fn default() -> Self {
        Self {
            // Prefer speed; level 2 favors throughput over ratio.
            compression_level: 2,
            // AdaptiveFast reduces per-row work with minimal compression impact.
            filter_strategy: FilterStrategy::AdaptiveFast,
            optimize_alpha: false,
            reduce_color_type: false,
            strip_metadata: false,
            reduce_palette: false,
        }
    }
}

impl PngOptions {
    /// Speed-focused preset (matches current default).
    pub fn fast() -> Self {
        Self {
            compression_level: 2,
            filter_strategy: FilterStrategy::AdaptiveFast,
            optimize_alpha: false,
            reduce_color_type: false,
            strip_metadata: false,
            reduce_palette: false,
        }
    }

    /// Balanced preset targeting better compression at moderate speed.
    pub fn balanced() -> Self {
        Self {
            compression_level: 6,
            filter_strategy: FilterStrategy::Adaptive,
            optimize_alpha: false,
            reduce_color_type: false,
            strip_metadata: false,
            reduce_palette: false,
        }
    }

    /// Highest compression preset; slowest.
    pub fn max_compression() -> Self {
        Self {
            compression_level: 9,
            filter_strategy: FilterStrategy::AdaptiveSampled { interval: 2 },
            optimize_alpha: false,
            reduce_color_type: false,
            strip_metadata: false,
            reduce_palette: false,
        }
    }
}

/// PNG filter selection strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterStrategy {
    /// Always use no filter (fastest encoding).
    None,
    /// Always use Sub filter.
    Sub,
    /// Always use Up filter.
    Up,
    /// Always use Average filter.
    Average,
    /// Always use Paeth filter.
    Paeth,
    /// Choose best filter per row using entropy scoring (slower, better compression).
    Entropy,
    /// Choose best filter per row (best compression, slower).
    Adaptive,
    /// Adaptive but with early cut and limited trials (faster).
    AdaptiveFast,
    /// Adaptive on sampled rows, reuse chosen filter on intervening rows.
    /// `interval` must be >= 1. Example: interval=4 runs full adaptive
    /// every 4th row and reuses the last chosen filter for others.
    AdaptiveSampled {
        /// Number of rows between full adaptive evaluations (minimum 1).
        interval: u32,
    },
}

/// Encode raw pixel data as PNG.
///
/// # Arguments
/// * `data` - Raw pixel data (row-major order)
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `color_type` - Color type of the input data
///
/// # Returns
/// Complete PNG file as bytes.
pub fn encode(data: &[u8], width: u32, height: u32, color_type: ColorType) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    encode_into(
        &mut output,
        data,
        width,
        height,
        color_type,
        &PngOptions::default(),
    )?;
    Ok(output)
}

/// Encode raw pixel data as PNG with custom options.
pub fn encode_with_options(
    data: &[u8],
    width: u32,
    height: u32,
    color_type: ColorType,
    options: &PngOptions,
) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    encode_into(&mut output, data, width, height, color_type, options)?;
    Ok(output)
}

/// Encode raw pixel data as PNG into a caller-provided buffer.
///
/// The `output` buffer will be cleared before writing. This API allows callers
/// to reuse an allocation across multiple encodes.
pub fn encode_into(
    output: &mut Vec<u8>,
    data: &[u8],
    width: u32,
    height: u32,
    color_type: ColorType,
    options: &PngOptions,
) -> Result<()> {
    if !(1..=9).contains(&options.compression_level) {
        return Err(Error::InvalidCompressionLevel(options.compression_level));
    }

    // Validate dimensions
    if width == 0 || height == 0 {
        return Err(Error::InvalidDimensions { width, height });
    }

    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(Error::ImageTooLarge {
            width,
            height,
            max: MAX_DIMENSION,
        });
    }

    // Validate data length
    let bytes_per_pixel = color_type.bytes_per_pixel();
    let expected_len = width as usize * height as usize * bytes_per_pixel;
    if data.len() != expected_len {
        return Err(Error::InvalidDataLength {
            expected: expected_len,
            actual: data.len(),
        });
    }

    output.clear();

    // Estimate output size (compressed data + overhead)
    output.reserve(expected_len / 2 + 1024);

    // Write PNG signature
    output.extend_from_slice(&PNG_SIGNATURE);

    // Optionally reduce color type/palette before encoding.
    let reduced =
        maybe_reduce_color_type(data, width as usize, height as usize, color_type, options);
    let bytes_per_pixel = reduced.bytes_per_pixel;

    // Write IHDR chunk
    write_ihdr(
        output,
        width,
        height,
        reduced.bit_depth,
        reduced.color_type_byte,
    );

    // Write palette/tRNS if present
    if let Some(ref palette) = reduced.palette {
        let mut plte = Vec::with_capacity(palette.len() * 3);
        for entry in palette {
            plte.extend_from_slice(&entry[..3]);
        }
        chunk::write_chunk(output, b"PLTE", &plte);

        if palette.iter().any(|p| p[3] != 255) {
            let alphas: Vec<u8> = palette.iter().map(|p| p[3]).collect();
            chunk::write_chunk(output, b"tRNS", &alphas);
        }
    }

    // Apply filtering and compression
    let data = maybe_optimize_alpha(
        &reduced.data,
        reduced.effective_color_type,
        options.optimize_alpha,
    );

    let filtered = filter::apply_filters(&data, width, height, bytes_per_pixel, options);
    let compressed = deflate_zlib_packed(&filtered, options.compression_level);

    // Write IDAT chunk(s)
    write_idat_chunks(output, &compressed);

    // Write IEND chunk
    write_iend(output);

    // Optionally strip metadata chunks to reduce size
    if options.strip_metadata {
        strip_metadata_chunks(output);
    }

    Ok(())
}

/// Encode raw pixel data as PNG into a caller-provided buffer, returning DEFLATE timing stats.
///
/// Only available when the `timing` feature is enabled. This mirrors `encode_into`
/// but surfaces per-stage DEFLATE timings to aid profiling without external tools.
#[cfg(feature = "timing")]
pub fn encode_into_with_stats(
    output: &mut Vec<u8>,
    data: &[u8],
    width: u32,
    height: u32,
    color_type: ColorType,
    options: &PngOptions,
) -> Result<DeflateStats> {
    if !(1..=9).contains(&options.compression_level) {
        return Err(Error::InvalidCompressionLevel(options.compression_level));
    }

    if width == 0 || height == 0 {
        return Err(Error::InvalidDimensions { width, height });
    }

    if width > MAX_DIMENSION || height > MAX_DIMENSION {
        return Err(Error::ImageTooLarge {
            width,
            height,
            max: MAX_DIMENSION,
        });
    }

    let bytes_per_pixel = color_type.bytes_per_pixel();
    let expected_len = width as usize * height as usize * bytes_per_pixel;
    if data.len() != expected_len {
        return Err(Error::InvalidDataLength {
            expected: expected_len,
            actual: data.len(),
        });
    }

    output.clear();
    output.reserve(expected_len / 2 + 1024);
    output.extend_from_slice(&PNG_SIGNATURE);
    // Reduce (palette/color) for stats path as well.
    let reduced =
        maybe_reduce_color_type(data, width as usize, height as usize, color_type, options);
    write_ihdr(
        output,
        width,
        height,
        reduced.bit_depth,
        reduced.color_type_byte,
    );
    if let Some(ref palette) = reduced.palette {
        let mut plte = Vec::with_capacity(palette.len() * 3);
        for entry in palette {
            plte.extend_from_slice(&entry[..3]);
        }
        chunk::write_chunk(output, b"PLTE", &plte);
        if palette.iter().any(|p| p[3] != 255) {
            let alphas: Vec<u8> = palette.iter().map(|p| p[3]).collect();
            chunk::write_chunk(output, b"tRNS", &alphas);
        }
    }

    let filtered = filter::apply_filters(
        &reduced.data,
        width,
        height,
        reduced.bytes_per_pixel,
        options,
    );
    let (compressed, stats) = deflate_zlib_packed_with_stats(&filtered, options.compression_level);
    write_idat_chunks(output, &compressed);
    write_iend(output);

    Ok(stats)
}

/// Write IHDR (image header) chunk.
fn write_ihdr(output: &mut Vec<u8>, width: u32, height: u32, bit_depth: u8, color_type_byte: u8) {
    let mut ihdr_data = Vec::with_capacity(13);

    // Width (4 bytes, big-endian)
    ihdr_data.extend_from_slice(&width.to_be_bytes());

    // Height (4 bytes, big-endian)
    ihdr_data.extend_from_slice(&height.to_be_bytes());

    // Bit depth (1 byte)
    ihdr_data.push(bit_depth);

    // Color type (1 byte)
    ihdr_data.push(color_type_byte);

    // Compression method (1 byte) - always 0 (DEFLATE)
    ihdr_data.push(0);

    // Filter method (1 byte) - always 0 (adaptive)
    ihdr_data.push(0);

    // Interlace method (1 byte) - 0 (no interlace)
    ihdr_data.push(0);

    chunk::write_chunk(output, b"IHDR", &ihdr_data);
}

/// Write IDAT (image data) chunks.
fn write_idat_chunks(output: &mut Vec<u8>, compressed: &[u8]) {
    // Write in larger chunks to reduce per-chunk overhead (CRC/length)
    const CHUNK_SIZE: usize = 256 * 1024;

    for chunk_data in compressed.chunks(CHUNK_SIZE) {
        chunk::write_chunk(output, b"IDAT", chunk_data);
    }
}

/// Write IEND (image end) chunk.
fn write_iend(output: &mut Vec<u8>) {
    chunk::write_chunk(output, b"IEND", &[]);
}

/// If enabled, zero color channels for fully transparent pixels to improve compression.
fn maybe_optimize_alpha<'a>(
    data: &'a [u8],
    color_type: ColorType,
    optimize_alpha: bool,
) -> std::borrow::Cow<'a, [u8]> {
    if !optimize_alpha {
        return std::borrow::Cow::Borrowed(data);
    }

    let bytes_per_pixel = color_type.bytes_per_pixel();
    if !matches!(color_type, ColorType::Rgba | ColorType::GrayAlpha) {
        return std::borrow::Cow::Borrowed(data);
    }

    let mut out = data.to_owned();
    match color_type {
        ColorType::Rgba => {
            for px in out.chunks_exact_mut(bytes_per_pixel) {
                let alpha = px[3];
                if alpha == 0 {
                    px[0] = 0;
                    px[1] = 0;
                    px[2] = 0;
                }
            }
        }
        ColorType::GrayAlpha => {
            for px in out.chunks_exact_mut(bytes_per_pixel) {
                let alpha = px[1];
                if alpha == 0 {
                    px[0] = 0;
                }
            }
        }
        _ => {}
    }

    std::borrow::Cow::Owned(out)
}

struct ReducedImage<'a> {
    data: std::borrow::Cow<'a, [u8]>,
    effective_color_type: ColorType,
    color_type_byte: u8,
    bit_depth: u8,
    bytes_per_pixel: usize,
    palette: Option<Vec<[u8; 4]>>,
}

/// Optionally reduce color type/palette when lossless-safe.
fn maybe_reduce_color_type<'a>(
    data: &'a [u8],
    width: usize,
    height: usize,
    color_type: ColorType,
    options: &PngOptions,
) -> ReducedImage<'a> {
    // Gray: optional bit-depth reduction
    if matches!(color_type, ColorType::Gray) && options.reduce_color_type {
        return ReducedImage {
            data: std::borrow::Cow::Borrowed(data),
            effective_color_type: ColorType::Gray,
            color_type_byte: ColorType::Gray.png_color_type(),
            bit_depth: 8,
            bytes_per_pixel: 1,
            palette: None,
        };
    }

    // Palette reduction takes priority if enabled and possible
    if options.reduce_palette {
        if let Some((indexed, palette)) = build_palette(data, color_type, width, height) {
            return ReducedImage {
                data: std::borrow::Cow::Owned(indexed),
                effective_color_type: ColorType::Rgb, // For optimize_alpha logic (unused for palette)
                color_type_byte: 3,
                bit_depth: 8,
                bytes_per_pixel: 1,
                palette: Some(palette),
            };
        }
    }

    if !options.reduce_color_type {
        return ReducedImage {
            data: std::borrow::Cow::Borrowed(data),
            effective_color_type: color_type,
            color_type_byte: color_type.png_color_type(),
            bit_depth: color_type.png_bit_depth(),
            bytes_per_pixel: color_type.bytes_per_pixel(),
            palette: None,
        };
    }

    match color_type {
        ColorType::Rgb => {
            if all_gray_rgb(data) {
                let mut gray = Vec::with_capacity(width * height);
                for chunk in data.chunks_exact(3) {
                    gray.push(chunk[0]);
                }
                ReducedImage {
                    data: std::borrow::Cow::Owned(gray),
                    effective_color_type: ColorType::Gray,
                    color_type_byte: ColorType::Gray.png_color_type(),
                    bit_depth: 8,
                    bytes_per_pixel: 1,
                    palette: None,
                }
            } else {
                ReducedImage {
                    data: std::borrow::Cow::Borrowed(data),
                    effective_color_type: color_type,
                    color_type_byte: color_type.png_color_type(),
                    bit_depth: 8,
                    bytes_per_pixel: 3,
                    palette: None,
                }
            }
        }
        ColorType::Rgba => {
            let (all_opaque, all_gray) = analyze_rgba(data);
            if all_opaque && all_gray {
                let mut gray = Vec::with_capacity(width * height);
                for chunk in data.chunks_exact(4) {
                    gray.push(chunk[0]);
                }
                ReducedImage {
                    data: std::borrow::Cow::Owned(gray),
                    effective_color_type: ColorType::Gray,
                    color_type_byte: ColorType::Gray.png_color_type(),
                    bit_depth: 8,
                    bytes_per_pixel: 1,
                    palette: None,
                }
            } else if all_opaque {
                let mut rgb = Vec::with_capacity(width * height * 3);
                for chunk in data.chunks_exact(4) {
                    rgb.extend_from_slice(&chunk[..3]);
                }
                ReducedImage {
                    data: std::borrow::Cow::Owned(rgb),
                    effective_color_type: ColorType::Rgb,
                    color_type_byte: ColorType::Rgb.png_color_type(),
                    bit_depth: 8,
                    bytes_per_pixel: 3,
                    palette: None,
                }
            } else if all_gray {
                let mut ga = Vec::with_capacity(width * height * 2);
                for chunk in data.chunks_exact(4) {
                    ga.push(chunk[0]);
                    ga.push(chunk[3]);
                }
                ReducedImage {
                    data: std::borrow::Cow::Owned(ga),
                    effective_color_type: ColorType::GrayAlpha,
                    color_type_byte: ColorType::GrayAlpha.png_color_type(),
                    bit_depth: 8,
                    bytes_per_pixel: 2,
                    palette: None,
                }
            } else {
                ReducedImage {
                    data: std::borrow::Cow::Borrowed(data),
                    effective_color_type: color_type,
                    color_type_byte: color_type.png_color_type(),
                    bit_depth: 8,
                    bytes_per_pixel: 4,
                    palette: None,
                }
            }
        }
        _ => ReducedImage {
            data: std::borrow::Cow::Borrowed(data),
            effective_color_type: color_type,
            color_type_byte: color_type.png_color_type(),
            bit_depth: color_type.png_bit_depth(),
            bytes_per_pixel: color_type.bytes_per_pixel(),
            palette: None,
        },
    }
}

fn build_palette(
    data: &[u8],
    color_type: ColorType,
    width: usize,
    height: usize,
) -> Option<(Vec<u8>, Vec<[u8; 4]>)> {
    match color_type {
        ColorType::Rgb | ColorType::Rgba => {}
        _ => return None,
    }
    use std::collections::HashMap;
    let mut map: HashMap<[u8; 4], u8> = HashMap::with_capacity(256);
    let mut palette: Vec<[u8; 4]> = Vec::new();
    let mut indexed = Vec::with_capacity(width * height);

    let stride = color_type.bytes_per_pixel();
    for chunk in data.chunks_exact(stride) {
        let entry = match color_type {
            ColorType::Rgb => [chunk[0], chunk[1], chunk[2], 255],
            ColorType::Rgba => [chunk[0], chunk[1], chunk[2], chunk[3]],
            _ => unreachable!(),
        };
        if let Some(&idx) = map.get(&entry) {
            indexed.push(idx);
        } else {
            if palette.len() == 256 {
                return None;
            }
            let idx = palette.len() as u8;
            palette.push(entry);
            map.insert(entry, idx);
            indexed.push(idx);
        }
    }

    Some((indexed, palette))
}

fn all_gray_rgb(data: &[u8]) -> bool {
    for chunk in data.chunks_exact(3) {
        if !(chunk[0] == chunk[1] && chunk[1] == chunk[2]) {
            return false;
        }
    }
    true
}

fn analyze_rgba(data: &[u8]) -> (bool, bool) {
    let mut all_opaque = true;
    let mut all_gray = true;
    for chunk in data.chunks_exact(4) {
        let a = chunk[3];
        if a != 255 {
            all_opaque = false;
        }
        if !(chunk[0] == chunk[1] && chunk[1] == chunk[2]) {
            all_gray = false;
        }
        if !all_opaque && !all_gray {
            break;
        }
    }
    (all_opaque, all_gray)
}

/// Strip non-critical ancillary chunks to reduce output size.
/// Currently removes tEXt, zTXt, iTXt, and tIME chunks.
fn strip_metadata_chunks(output: &mut Vec<u8>) {
    // PNG layout: signature (8), then chunks: length(4), type(4), data(n), crc(4)
    let mut cursor = 8;
    let mut stripped = Vec::with_capacity(output.len());
    stripped.extend_from_slice(&output[..8]); // signature

    while cursor + 8 <= output.len() {
        let len_bytes = &output[cursor..cursor + 4];
        let len = u32::from_be_bytes(len_bytes.try_into().unwrap()) as usize;
        let chunk_type = &output[cursor + 4..cursor + 8];
        let chunk_data_start = cursor + 8;
        let chunk_data_end = chunk_data_start + len;
        let chunk_crc_end = chunk_data_end + 4;
        if chunk_crc_end > output.len() {
            break; // malformed; bail out
        }

        let is_ancillary = (chunk_type[0] & 0x20) != 0;
        let should_strip = is_ancillary
            && (chunk_type == b"tEXt"
                || chunk_type == b"zTXt"
                || chunk_type == b"iTXt"
                || chunk_type == b"tIME");

        if !should_strip {
            stripped.extend_from_slice(&output[cursor..chunk_crc_end]);
        }

        cursor = chunk_crc_end;

        if chunk_type == b"IEND" {
            break;
        }
    }

    output.clear();
    output.extend_from_slice(&stripped);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_1x1_rgb() {
        let pixels = vec![255, 0, 0]; // Red pixel
        let png = encode(&pixels, 1, 1, ColorType::Rgb).unwrap();

        // Check PNG signature
        assert_eq!(&png[0..8], &PNG_SIGNATURE);

        // Should have IHDR, IDAT, IEND
        assert!(png.len() > 8 + 12 + 12 + 12); // signature + 3 chunks minimum
    }

    #[test]
    fn test_encode_1x1_rgba() {
        let pixels = vec![255, 0, 0, 255]; // Red opaque pixel
        let png = encode(&pixels, 1, 1, ColorType::Rgba).unwrap();

        assert_eq!(&png[0..8], &PNG_SIGNATURE);
    }

    #[test]
    fn test_encode_invalid_dimensions() {
        let pixels = vec![255, 0, 0];
        let result = encode(&pixels, 0, 1, ColorType::Rgb);
        assert!(matches!(result, Err(Error::InvalidDimensions { .. })));
    }

    #[test]
    fn test_encode_invalid_data_length() {
        let pixels = vec![255, 0]; // Too short for 1x1 RGB
        let result = encode(&pixels, 1, 1, ColorType::Rgb);
        assert!(matches!(result, Err(Error::InvalidDataLength { .. })));
    }

    #[test]
    fn test_encode_into_reuses_buffer() {
        let mut output = Vec::with_capacity(64);
        let pixels1 = vec![0u8, 0, 0]; // black 1x1 RGB
        encode_into(
            &mut output,
            &pixels1,
            1,
            1,
            ColorType::Rgb,
            &PngOptions::default(),
        )
        .unwrap();
        let first = output.clone();
        let first_cap = output.capacity();
        assert!(!first.is_empty());

        let pixels2 = vec![255u8, 0, 0]; // red 1x1 RGB
        encode_into(
            &mut output,
            &pixels2,
            1,
            1,
            ColorType::Rgb,
            &PngOptions::default(),
        )
        .unwrap();

        assert_ne!(
            first, output,
            "buffer should have been reused and rewritten"
        );
        assert!(output.capacity() >= first_cap);
        assert_eq!(&output[0..8], &PNG_SIGNATURE);
    }

    #[test]
    fn test_optimize_alpha_zeroes_color() {
        let pixels = vec![
            10, 20, 30, 0, // fully transparent, should zero RGB
            1, 2, 3, 255, // opaque, should stay
        ];
        let out = maybe_optimize_alpha(&pixels, ColorType::Rgba, true);
        assert_eq!(
            &out[..],
            &[0, 0, 0, 0, 1, 2, 3, 255],
            "color channels should be zeroed when alpha is 0"
        );
    }

    #[test]
    fn test_reduce_color_rgb_to_gray() {
        // All channels equal -> reducible to Gray
        let pixels = vec![
            10, 10, 10, //
            50, 50, 50, //
        ];
        let opts = PngOptions {
            reduce_color_type: true,
            ..Default::default()
        };
        let reduced = maybe_reduce_color_type(&pixels, 2, 1, ColorType::Rgb, &opts);
        assert!(matches!(reduced.effective_color_type, ColorType::Gray));
        assert_eq!(&reduced.data[..], &[10, 50]);
    }

    #[test]
    fn test_reduce_color_rgba_drop_alpha() {
        // Opaque RGBA should drop alpha to RGB
        let pixels = vec![
            1, 2, 3, 255, //
            4, 5, 6, 255, //
        ];
        let opts = PngOptions {
            reduce_color_type: true,
            ..Default::default()
        };
        let reduced = maybe_reduce_color_type(&pixels, 2, 1, ColorType::Rgba, &opts);
        assert!(matches!(reduced.effective_color_type, ColorType::Rgb));
        assert_eq!(&reduced.data[..], &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_reduce_color_rgba_to_grayalpha() {
        // Grayscale RGBA with varying alpha -> GrayAlpha
        let pixels = vec![
            8, 8, 8, 10, //
            9, 9, 9, 0, //
        ];
        let opts = PngOptions {
            reduce_color_type: true,
            ..Default::default()
        };
        let reduced = maybe_reduce_color_type(&pixels, 2, 1, ColorType::Rgba, &opts);
        assert!(matches!(reduced.effective_color_type, ColorType::GrayAlpha));
        assert_eq!(&reduced.data[..], &[8, 10, 9, 0]);
    }

    #[test]
    fn test_strip_metadata_chunks() {
        // Build a minimal PNG with a tEXt chunk and ensure it is removed.
        let mut png_bytes = Vec::new();
        png_bytes.extend_from_slice(&PNG_SIGNATURE);
        // IHDR with 1x1 RGB
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&1u32.to_be_bytes()); // width
        ihdr.extend_from_slice(&1u32.to_be_bytes()); // height
        ihdr.push(8); // bit depth
        ihdr.push(2); // color type (RGB)
        ihdr.push(0); // compression
        ihdr.push(0); // filter
        ihdr.push(0); // interlace
        chunk::write_chunk(&mut png_bytes, b"IHDR", &ihdr);
        // tEXt chunk
        chunk::write_chunk(&mut png_bytes, b"tEXt", b"Comment\0hello");
        // IDAT (empty)
        chunk::write_chunk(&mut png_bytes, b"IDAT", &[]);
        // IEND
        chunk::write_chunk(&mut png_bytes, b"IEND", &[]);

        strip_metadata_chunks(&mut png_bytes);
        // Should no longer contain tEXt
        assert!(!png_bytes.windows(4).any(|w| w == b"tEXt"));
        // Should still contain IHDR/IDAT/IEND
        assert!(png_bytes.windows(4).any(|w| w == b"IHDR"));
        assert!(png_bytes.windows(4).any(|w| w == b"IDAT"));
        assert!(png_bytes.windows(4).any(|w| w == b"IEND"));
    }

    #[test]
    fn test_palette_reduction_writes_plte() {
        // Two-color RGBA image, opaque -> should palette to 2 entries.
        let pixels = vec![
            255, 0, 0, 255, // red
            0, 255, 0, 255, // green
        ];
        let opts = PngOptions {
            reduce_palette: true,
            ..Default::default()
        };
        let png = encode_with_options(&pixels, 2, 1, ColorType::Rgba, &opts).unwrap();
        // Color type byte in IHDR should be 3 (palette)
        assert_eq!(png[25], 3);
        // Bit depth should remain 8 (no bit packing yet)
        assert_eq!(png[24], 8);
        assert!(png.windows(4).any(|w| w == b"PLTE"));
    }

    #[test]
    fn test_encode_2x2_checkerboard() {
        // 2x2 black and white checkerboard
        let pixels = vec![
            0, 0, 0, 255, 255, 255, // Row 1: black, white
            255, 255, 255, 0, 0, 0, // Row 2: white, black
        ];
        let png = encode(&pixels, 2, 2, ColorType::Rgb).unwrap();

        assert_eq!(&png[0..8], &PNG_SIGNATURE);
    }

    #[test]
    fn test_encode_grayscale() {
        let pixels = vec![128, 255, 0, 64]; // 2x2 grayscale
        let png = encode(&pixels, 2, 2, ColorType::Gray).unwrap();

        assert_eq!(&png[0..8], &PNG_SIGNATURE);
    }
}
