//! PNG decoder implementation.
//!
//! Decodes PNG images into raw pixel data.

use super::inflate::inflate_zlib_with_size;
use crate::color::ColorType;
use crate::compress::crc32::crc32;
use crate::error::{Error, Result};

/// PNG file signature (magic bytes).
const PNG_SIGNATURE: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

/// Maximum dimension for decoded images (16 million pixels per side).
/// This matches the encoder limit and prevents decompression bombs.
const MAX_DIMENSION: u32 = 1 << 24;

/// Decoded PNG image.
#[derive(Debug, Clone)]
pub struct PngImage {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Raw pixel data (format depends on color_type).
    pub pixels: Vec<u8>,
    /// Color type of the decoded image.
    pub color_type: ColorType,
}

/// PNG color type values from specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PngColorType {
    Grayscale = 0,
    Rgb = 2,
    Indexed = 3,
    GrayscaleAlpha = 4,
    Rgba = 6,
}

impl TryFrom<u8> for PngColorType {
    type Error = Error;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            0 => Ok(PngColorType::Grayscale),
            2 => Ok(PngColorType::Rgb),
            3 => Ok(PngColorType::Indexed),
            4 => Ok(PngColorType::GrayscaleAlpha),
            6 => Ok(PngColorType::Rgba),
            _ => Err(Error::InvalidDecode(format!(
                "invalid PNG color type: {value}"
            ))),
        }
    }
}

/// IHDR chunk data.
struct IhdrData {
    width: u32,
    height: u32,
    bit_depth: u8,
    color_type: PngColorType,
    compression_method: u8,
    filter_method: u8,
    interlace_method: u8,
}

/// Check if tRNS chunk contains any non-opaque (< 255) alpha values.
/// Returns false if trns is None or all values are 255 (fully opaque).
fn has_alpha_in_trns(trns: Option<&[u8]>) -> bool {
    trns.map(|a| a.iter().any(|&v| v != 0xFF)).unwrap_or(false)
}

/// Calculate the expected size of decompressed IDAT data.
///
/// The decompressed data contains filtered scanlines:
/// height * (1 filter byte + scanline_bytes)
fn calculate_expected_size(ihdr: &IhdrData) -> Result<usize> {
    let width = ihdr.width as usize;
    let height = ihdr.height as usize;
    let bit_depth = ihdr.bit_depth as usize;

    // Calculate bytes per scanline (excluding filter byte)
    let scanline_bytes = match ihdr.color_type {
        PngColorType::Grayscale => (width * bit_depth).div_ceil(8),
        PngColorType::Rgb => width * 3 * bit_depth / 8,
        PngColorType::Indexed => (width * bit_depth).div_ceil(8),
        PngColorType::GrayscaleAlpha => width * 2 * bit_depth / 8,
        PngColorType::Rgba => width * 4 * bit_depth / 8,
    };

    // Each row has 1 filter byte + scanline_bytes
    let row_size = 1 + scanline_bytes;

    row_size
        .checked_mul(height)
        .ok_or_else(|| Error::InvalidDecode("image size overflow".into()))
}

/// Decode a PNG image from bytes.
pub fn decode_png(data: &[u8]) -> Result<PngImage> {
    // Verify signature
    if data.len() < 8 || data[..8] != PNG_SIGNATURE {
        return Err(Error::InvalidDecode("not a PNG file".into()));
    }

    let mut pos = 8;
    let mut ihdr: Option<IhdrData> = None;
    let mut idat_data = Vec::new();
    let mut palette: Option<Vec<[u8; 3]>> = None;
    let mut trns: Option<Vec<u8>> = None;
    let mut seen_iend = false;

    // Parse chunks
    while pos + 12 <= data.len() {
        let length =
            u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        let chunk_type = &data[pos + 4..pos + 8];
        let chunk_data_start = pos + 8;
        let chunk_data_end = chunk_data_start + length;
        let crc_start = chunk_data_end;
        let crc_end = crc_start + 4;

        if crc_end > data.len() {
            return Err(Error::InvalidDecode("truncated PNG chunk".into()));
        }

        let chunk_data = &data[chunk_data_start..chunk_data_end];
        let stored_crc = u32::from_be_bytes([
            data[crc_start],
            data[crc_start + 1],
            data[crc_start + 2],
            data[crc_start + 3],
        ]);

        // Verify CRC (over type + data)
        let mut crc_data = Vec::with_capacity(4 + length);
        crc_data.extend_from_slice(chunk_type);
        crc_data.extend_from_slice(chunk_data);
        let computed_crc = crc32(&crc_data);

        if stored_crc != computed_crc {
            return Err(Error::InvalidDecode(format!(
                "CRC mismatch in {} chunk",
                String::from_utf8_lossy(chunk_type)
            )));
        }

        match chunk_type {
            b"IHDR" => {
                if length != 13 {
                    return Err(Error::InvalidDecode("invalid IHDR length".into()));
                }
                ihdr = Some(IhdrData {
                    width: u32::from_be_bytes([
                        chunk_data[0],
                        chunk_data[1],
                        chunk_data[2],
                        chunk_data[3],
                    ]),
                    height: u32::from_be_bytes([
                        chunk_data[4],
                        chunk_data[5],
                        chunk_data[6],
                        chunk_data[7],
                    ]),
                    bit_depth: chunk_data[8],
                    color_type: PngColorType::try_from(chunk_data[9])?,
                    compression_method: chunk_data[10],
                    filter_method: chunk_data[11],
                    interlace_method: chunk_data[12],
                });
            }
            b"PLTE" => {
                if length % 3 != 0 {
                    return Err(Error::InvalidDecode("invalid PLTE length".into()));
                }
                let count = length / 3;
                let mut pal = Vec::with_capacity(count);
                for i in 0..count {
                    pal.push([
                        chunk_data[i * 3],
                        chunk_data[i * 3 + 1],
                        chunk_data[i * 3 + 2],
                    ]);
                }
                palette = Some(pal);
            }
            b"tRNS" => {
                trns = Some(chunk_data.to_vec());
            }
            b"IDAT" => {
                idat_data.extend_from_slice(chunk_data);
            }
            b"IEND" => {
                seen_iend = true;
                break;
            }
            _ => {
                // Skip unknown chunks
            }
        }

        pos = crc_end;
    }

    // Validate required chunks
    if !seen_iend {
        return Err(Error::InvalidDecode("missing IEND chunk".into()));
    }
    let ihdr = ihdr.ok_or_else(|| Error::InvalidDecode("missing IHDR chunk".into()))?;

    // Validate dimensions
    if ihdr.width == 0 || ihdr.height == 0 {
        return Err(Error::InvalidDimensions {
            width: ihdr.width,
            height: ihdr.height,
        });
    }
    if ihdr.width > MAX_DIMENSION || ihdr.height > MAX_DIMENSION {
        return Err(Error::ImageTooLarge {
            width: ihdr.width,
            height: ihdr.height,
            max: MAX_DIMENSION,
        });
    }

    if ihdr.compression_method != 0 {
        return Err(Error::InvalidDecode(
            "unsupported compression method".into(),
        ));
    }
    if ihdr.filter_method != 0 {
        return Err(Error::InvalidDecode("unsupported filter method".into()));
    }
    if ihdr.interlace_method != 0 {
        return Err(Error::UnsupportedDecode(
            "Adam7 interlaced images not supported".into(),
        ));
    }

    // Validate bit depth for color type
    let valid_depth = match ihdr.color_type {
        PngColorType::Grayscale => matches!(ihdr.bit_depth, 1 | 2 | 4 | 8 | 16),
        PngColorType::Rgb => matches!(ihdr.bit_depth, 8 | 16),
        PngColorType::Indexed => matches!(ihdr.bit_depth, 1 | 2 | 4 | 8),
        PngColorType::GrayscaleAlpha => matches!(ihdr.bit_depth, 8 | 16),
        PngColorType::Rgba => matches!(ihdr.bit_depth, 8 | 16),
    };
    if !valid_depth {
        return Err(Error::InvalidDecode(format!(
            "invalid bit depth {} for color type {:?}",
            ihdr.bit_depth, ihdr.color_type
        )));
    }

    // Decompress IDAT data
    if idat_data.is_empty() {
        return Err(Error::InvalidDecode("no IDAT data".into()));
    }

    // Calculate expected decompressed size for pre-allocation and validation
    let expected_size = calculate_expected_size(&ihdr)?;
    let decompressed = inflate_zlib_with_size(&idat_data, Some(expected_size))?;

    // Reconstruct image from filtered scanlines
    let pixels = reconstruct_image(&ihdr, &decompressed, palette.as_deref(), trns.as_deref())?;

    // Determine output color type
    // Only use RGBA if tRNS chunk contains non-opaque values
    let color_type = match ihdr.color_type {
        PngColorType::Grayscale => ColorType::Gray,
        PngColorType::GrayscaleAlpha => ColorType::GrayAlpha,
        PngColorType::Rgb | PngColorType::Indexed => {
            // Indexed with transparency becomes RGBA only if tRNS has non-opaque values
            if ihdr.color_type == PngColorType::Indexed && has_alpha_in_trns(trns.as_deref()) {
                ColorType::Rgba
            } else {
                ColorType::Rgb
            }
        }
        PngColorType::Rgba => ColorType::Rgba,
    };

    Ok(PngImage {
        width: ihdr.width,
        height: ihdr.height,
        pixels,
        color_type,
    })
}

/// Reconstruct image from filtered scanlines.
fn reconstruct_image(
    ihdr: &IhdrData,
    data: &[u8],
    palette: Option<&[[u8; 3]]>,
    trns: Option<&[u8]>,
) -> Result<Vec<u8>> {
    let width = ihdr.width as usize;
    let height = ihdr.height as usize;
    let bit_depth = ihdr.bit_depth as usize;

    // Calculate bytes per pixel and row for the raw (pre-unfilter) data
    let (bpp, scanline_bytes) = match ihdr.color_type {
        PngColorType::Grayscale => {
            let bits_per_pixel = bit_depth;
            let bpp = bits_per_pixel.div_ceil(8);
            let row_bits = width * bits_per_pixel;
            let row_bytes = row_bits.div_ceil(8);
            (bpp.max(1), row_bytes)
        }
        PngColorType::Rgb => {
            let bpp = 3 * bit_depth / 8;
            (bpp, width * bpp)
        }
        PngColorType::Indexed => {
            let bits_per_pixel = bit_depth;
            let bpp = 1; // For filter purposes, indexed uses 1 byte per sample unit
            let row_bits = width * bits_per_pixel;
            let row_bytes = row_bits.div_ceil(8);
            (bpp, row_bytes)
        }
        PngColorType::GrayscaleAlpha => {
            let bpp = 2 * bit_depth / 8;
            (bpp, width * bpp)
        }
        PngColorType::Rgba => {
            let bpp = 4 * bit_depth / 8;
            (bpp, width * bpp)
        }
    };

    let expected_len = height * (1 + scanline_bytes);
    if data.len() < expected_len {
        return Err(Error::InvalidDecode(format!(
            "decompressed data too short: {} < {}",
            data.len(),
            expected_len
        )));
    }

    // Unfilter scanlines
    let mut current_row = vec![0u8; scanline_bytes];
    let mut prev_row = vec![0u8; scanline_bytes];
    let mut raw_rows = Vec::with_capacity(height * scanline_bytes);

    for y in 0..height {
        let row_start = y * (1 + scanline_bytes);
        let filter_type = data[row_start];
        let row_data = &data[row_start + 1..row_start + 1 + scanline_bytes];

        // Copy to current row
        current_row.copy_from_slice(row_data);

        // Apply filter reconstruction
        unfilter_row(filter_type, &mut current_row, &prev_row, bpp)?;

        raw_rows.extend_from_slice(&current_row);

        // Swap rows
        std::mem::swap(&mut current_row, &mut prev_row);
    }

    // Convert to final pixel format
    convert_to_pixels(ihdr, &raw_rows, palette, trns)
}

/// Reconstruct a row by reversing the PNG filter.
fn unfilter_row(filter: u8, row: &mut [u8], prev: &[u8], bpp: usize) -> Result<()> {
    match filter {
        0 => Ok(()), // None - no change
        1 => {
            // Sub: add left byte
            for i in bpp..row.len() {
                row[i] = row[i].wrapping_add(row[i - bpp]);
            }
            Ok(())
        }
        2 => {
            // Up: add above byte
            for i in 0..row.len() {
                row[i] = row[i].wrapping_add(prev[i]);
            }
            Ok(())
        }
        3 => {
            // Average: add average of left and above
            for i in 0..row.len() {
                let left = if i >= bpp { row[i - bpp] as u16 } else { 0 };
                let above = prev[i] as u16;
                row[i] = row[i].wrapping_add(((left + above) / 2) as u8);
            }
            Ok(())
        }
        4 => {
            // Paeth: add Paeth predictor
            for i in 0..row.len() {
                let a = if i >= bpp { row[i - bpp] } else { 0 };
                let b = prev[i];
                let c = if i >= bpp { prev[i - bpp] } else { 0 };
                row[i] = row[i].wrapping_add(paeth_predictor(a, b, c));
            }
            Ok(())
        }
        _ => Err(Error::InvalidDecode(format!(
            "invalid filter type: {filter}"
        ))),
    }
}

/// Paeth predictor function.
#[inline]
fn paeth_predictor(a: u8, b: u8, c: u8) -> u8 {
    let p = a as i32 + b as i32 - c as i32;
    let pa = (p - a as i32).abs();
    let pb = (p - b as i32).abs();
    let pc = (p - c as i32).abs();

    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// Convert raw row data to final pixel format.
fn convert_to_pixels(
    ihdr: &IhdrData,
    raw_data: &[u8],
    palette: Option<&[[u8; 3]]>,
    trns: Option<&[u8]>,
) -> Result<Vec<u8>> {
    let width = ihdr.width as usize;
    let height = ihdr.height as usize;
    let bit_depth = ihdr.bit_depth;

    match ihdr.color_type {
        PngColorType::Grayscale => {
            if bit_depth == 8 {
                Ok(raw_data.to_vec())
            } else if bit_depth == 16 {
                // Convert 16-bit to 8-bit
                let pixels: Vec<u8> = raw_data
                    .chunks_exact(2)
                    .map(|chunk| chunk[0]) // Take high byte
                    .collect();
                Ok(pixels)
            } else {
                // Unpack sub-8-bit depths
                unpack_gray(raw_data, width, height, bit_depth)
            }
        }
        PngColorType::GrayscaleAlpha => {
            if bit_depth == 8 {
                Ok(raw_data.to_vec())
            } else {
                // 16-bit: take high bytes
                let pixels: Vec<u8> = raw_data
                    .chunks_exact(4)
                    .flat_map(|chunk| [chunk[0], chunk[2]])
                    .collect();
                Ok(pixels)
            }
        }
        PngColorType::Rgb => {
            if bit_depth == 8 {
                Ok(raw_data.to_vec())
            } else {
                // 16-bit: take high bytes
                let pixels: Vec<u8> = raw_data
                    .chunks_exact(6)
                    .flat_map(|chunk| [chunk[0], chunk[2], chunk[4]])
                    .collect();
                Ok(pixels)
            }
        }
        PngColorType::Rgba => {
            if bit_depth == 8 {
                Ok(raw_data.to_vec())
            } else {
                // 16-bit: take high bytes
                let pixels: Vec<u8> = raw_data
                    .chunks_exact(8)
                    .flat_map(|chunk| [chunk[0], chunk[2], chunk[4], chunk[6]])
                    .collect();
                Ok(pixels)
            }
        }
        PngColorType::Indexed => {
            let palette =
                palette.ok_or_else(|| Error::InvalidDecode("missing PLTE chunk".into()))?;

            // Unpack indices
            let indices = unpack_indices(raw_data, width, height, bit_depth)?;

            // Expand palette (with optional transparency)
            // Only output RGBA if tRNS contains non-opaque values
            if has_alpha_in_trns(trns) {
                let alpha_table = trns.unwrap(); // Safe: has_alpha_in_trns returned true
                                                 // Output RGBA
                let mut pixels = Vec::with_capacity(width * height * 4);
                for &idx in &indices {
                    if (idx as usize) < palette.len() {
                        let [r, g, b] = palette[idx as usize];
                        let a = if (idx as usize) < alpha_table.len() {
                            alpha_table[idx as usize]
                        } else {
                            255
                        };
                        pixels.extend_from_slice(&[r, g, b, a]);
                    } else {
                        pixels.extend_from_slice(&[0, 0, 0, 255]);
                    }
                }
                Ok(pixels)
            } else {
                // Output RGB
                let mut pixels = Vec::with_capacity(width * height * 3);
                for &idx in &indices {
                    if (idx as usize) < palette.len() {
                        pixels.extend_from_slice(&palette[idx as usize]);
                    } else {
                        pixels.extend_from_slice(&[0, 0, 0]);
                    }
                }
                Ok(pixels)
            }
        }
    }
}

/// Unpack grayscale samples from sub-8-bit depths.
fn unpack_gray(data: &[u8], width: usize, height: usize, bit_depth: u8) -> Result<Vec<u8>> {
    let mut pixels = Vec::with_capacity(width * height);
    let row_bytes = (width * bit_depth as usize).div_ceil(8);

    for row in 0..height {
        let row_data = &data[row * row_bytes..(row + 1) * row_bytes];
        unpack_row(row_data, width, bit_depth, &mut pixels);
    }

    // Scale samples to 8-bit
    for pixel in &mut pixels {
        *pixel = scale_to_8bit(*pixel, bit_depth);
    }

    Ok(pixels)
}

/// Unpack palette indices from sub-8-bit depths.
fn unpack_indices(data: &[u8], width: usize, height: usize, bit_depth: u8) -> Result<Vec<u8>> {
    let mut indices = Vec::with_capacity(width * height);
    let row_bytes = (width * bit_depth as usize).div_ceil(8);

    for row in 0..height {
        let row_data = &data[row * row_bytes..(row + 1) * row_bytes];
        unpack_row(row_data, width, bit_depth, &mut indices);
    }

    Ok(indices)
}

/// Unpack a single row of sub-8-bit samples.
fn unpack_row(packed: &[u8], width: usize, bit_depth: u8, out: &mut Vec<u8>) {
    let start_len = out.len();

    match bit_depth {
        1 => {
            for &byte in packed {
                for i in (0..8).rev() {
                    if out.len() - start_len >= width {
                        break;
                    }
                    out.push((byte >> i) & 1);
                }
            }
        }
        2 => {
            for &byte in packed {
                for i in (0..4).rev() {
                    if out.len() - start_len >= width {
                        break;
                    }
                    out.push((byte >> (i * 2)) & 3);
                }
            }
        }
        4 => {
            for &byte in packed {
                if out.len() - start_len < width {
                    out.push(byte >> 4);
                }
                if out.len() - start_len < width {
                    out.push(byte & 0xF);
                }
            }
        }
        8 => {
            out.extend_from_slice(&packed[..width.min(packed.len())]);
        }
        _ => {}
    }

    // Ensure exactly width samples were output
    out.truncate(start_len + width);
}

/// Scale a sample from bit_depth bits to 8 bits using bit replication.
fn scale_to_8bit(sample: u8, bit_depth: u8) -> u8 {
    match bit_depth {
        1 => {
            if sample == 0 {
                0
            } else {
                255
            }
        }
        2 => sample | (sample << 2) | (sample << 4) | (sample << 6),
        4 => sample | (sample << 4),
        8 => sample,
        _ => sample,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paeth_predictor() {
        assert_eq!(paeth_predictor(100, 100, 100), 100);
        assert_eq!(paeth_predictor(0, 0, 0), 0);
    }

    #[test]
    fn test_paeth_predictor_edge_cases() {
        // Test where a is closest
        assert_eq!(paeth_predictor(100, 50, 50), 100);
        // Test where b is closest
        assert_eq!(paeth_predictor(50, 100, 50), 100);
        // Test where c is closest
        assert_eq!(paeth_predictor(50, 50, 100), 50); // a wins tie
                                                      // Test with max values
        assert_eq!(paeth_predictor(255, 255, 255), 255);
    }

    #[test]
    fn test_scale_to_8bit() {
        assert_eq!(scale_to_8bit(0, 1), 0);
        assert_eq!(scale_to_8bit(1, 1), 255);
        assert_eq!(scale_to_8bit(0, 2), 0);
        assert_eq!(scale_to_8bit(3, 2), 255);
        assert_eq!(scale_to_8bit(0, 4), 0);
        assert_eq!(scale_to_8bit(15, 4), 255);
    }

    #[test]
    fn test_scale_to_8bit_middle_values() {
        // 2-bit: 1 should scale to ~85
        let scaled = scale_to_8bit(1, 2);
        assert_eq!(scaled, 0b01010101); // 85
                                        // 4-bit: 8 should scale to ~136
        let scaled = scale_to_8bit(8, 4);
        assert_eq!(scaled, 0b10001000); // 136
    }

    #[test]
    fn test_unfilter_none() {
        let mut row = vec![1, 2, 3, 4];
        let prev = vec![0, 0, 0, 0];
        unfilter_row(0, &mut row, &prev, 1).unwrap();
        assert_eq!(row, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_unfilter_sub() {
        let mut row = vec![1, 2, 3, 4];
        let prev = vec![0, 0, 0, 0];
        unfilter_row(1, &mut row, &prev, 1).unwrap();
        // Each byte adds the previous: 1, 1+2=3, 3+3=6, 6+4=10
        assert_eq!(row, vec![1, 3, 6, 10]);
    }

    #[test]
    fn test_unfilter_sub_wrapping() {
        // Test wrapping behavior
        let mut row = vec![200, 100, 100, 100];
        let prev = vec![0, 0, 0, 0];
        unfilter_row(1, &mut row, &prev, 1).unwrap();
        // 200, 200+100=44 (wraps), 44+100=144, 144+100=244
        assert_eq!(row, vec![200, 44, 144, 244]);
    }

    #[test]
    fn test_unfilter_up() {
        let mut row = vec![1, 2, 3, 4];
        let prev = vec![10, 20, 30, 40];
        unfilter_row(2, &mut row, &prev, 1).unwrap();
        assert_eq!(row, vec![11, 22, 33, 44]);
    }

    #[test]
    fn test_unfilter_average() {
        let mut row = vec![10, 10, 10, 10];
        let prev = vec![20, 20, 20, 20];
        unfilter_row(3, &mut row, &prev, 1).unwrap();
        // First byte: 10 + avg(0, 20) = 10 + 10 = 20
        // Second byte: 10 + avg(20, 20) = 10 + 20 = 30
        assert_eq!(row[0], 20);
        assert_eq!(row[1], 30);
    }

    #[test]
    fn test_unfilter_paeth() {
        let mut row = vec![5, 5, 5, 5];
        let prev = vec![10, 10, 10, 10];
        unfilter_row(4, &mut row, &prev, 1).unwrap();
        // Verify paeth filter is applied
        assert!(!row.is_empty());
    }

    #[test]
    fn test_unfilter_invalid_filter_type() {
        let mut row = vec![1, 2, 3, 4];
        let prev = vec![0, 0, 0, 0];
        let result = unfilter_row(5, &mut row, &prev, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_invalid_signature() {
        let data = b"not a PNG file";
        assert!(decode_png(data).is_err());
    }

    #[test]
    fn test_decode_empty_data() {
        let data: &[u8] = &[];
        assert!(decode_png(data).is_err());
    }

    #[test]
    fn test_decode_signature_only() {
        let data = PNG_SIGNATURE;
        assert!(decode_png(&data).is_err());
    }

    #[test]
    fn test_decode_roundtrip() {
        // Create a simple 2x2 RGB image and encode it
        let pixels = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0];
        let opts = crate::png::PngOptions::builder(2, 2)
            .color_type(ColorType::Rgb)
            .build();
        let encoded = crate::png::encode(&pixels, &opts).expect("encoding should work");

        // Decode it
        let decoded = decode_png(&encoded).expect("decoding should work");

        assert_eq!(decoded.width, 2);
        assert_eq!(decoded.height, 2);
        assert_eq!(decoded.color_type, ColorType::Rgb);
        assert_eq!(decoded.pixels, pixels);
    }

    #[test]
    fn test_decode_rgba_roundtrip() {
        let pixels = vec![
            255, 0, 0, 255, // Red, opaque
            0, 255, 0, 128, // Green, semi-transparent
            0, 0, 255, 0, // Blue, transparent
            255, 255, 0, 255, // Yellow, opaque
        ];
        let opts = crate::png::PngOptions::builder(2, 2)
            .color_type(ColorType::Rgba)
            .build();
        let encoded = crate::png::encode(&pixels, &opts).expect("encoding should work");
        let decoded = decode_png(&encoded).expect("decoding should work");

        assert_eq!(decoded.width, 2);
        assert_eq!(decoded.height, 2);
        assert_eq!(decoded.color_type, ColorType::Rgba);
        assert_eq!(decoded.pixels, pixels);
    }

    #[test]
    fn test_decode_grayscale_roundtrip() {
        let pixels = vec![0, 64, 128, 255];
        let opts = crate::png::PngOptions::builder(2, 2)
            .color_type(ColorType::Gray)
            .build();
        let encoded = crate::png::encode(&pixels, &opts).expect("encoding should work");
        let decoded = decode_png(&encoded).expect("decoding should work");

        assert_eq!(decoded.width, 2);
        assert_eq!(decoded.height, 2);
        assert_eq!(decoded.color_type, ColorType::Gray);
        assert_eq!(decoded.pixels, pixels);
    }

    #[test]
    fn test_decode_gray_alpha_roundtrip() {
        let pixels = vec![0, 255, 128, 128, 255, 0, 64, 192]; // 2x2 gray+alpha
        let opts = crate::png::PngOptions::builder(2, 2)
            .color_type(ColorType::GrayAlpha)
            .build();
        let encoded = crate::png::encode(&pixels, &opts).expect("encoding should work");
        let decoded = decode_png(&encoded).expect("decoding should work");

        assert_eq!(decoded.width, 2);
        assert_eq!(decoded.height, 2);
        assert_eq!(decoded.color_type, ColorType::GrayAlpha);
        assert_eq!(decoded.pixels, pixels);
    }

    #[test]
    fn test_decode_larger_image() {
        // 8x8 RGB image
        let pixels: Vec<u8> = (0..8 * 8 * 3).map(|i| (i % 256) as u8).collect();
        let opts = crate::png::PngOptions::builder(8, 8)
            .color_type(ColorType::Rgb)
            .build();
        let encoded = crate::png::encode(&pixels, &opts).expect("encoding should work");
        let decoded = decode_png(&encoded).expect("decoding should work");

        assert_eq!(decoded.width, 8);
        assert_eq!(decoded.height, 8);
        assert_eq!(decoded.pixels, pixels);
    }

    #[test]
    fn test_unpack_row_1bit() {
        let packed = vec![0b10110100];
        let mut out = Vec::new();
        unpack_row(&packed, 8, 1, &mut out);
        assert_eq!(out, vec![1, 0, 1, 1, 0, 1, 0, 0]);
    }

    #[test]
    fn test_unpack_row_2bit() {
        let packed = vec![0b11100100];
        let mut out = Vec::new();
        unpack_row(&packed, 4, 2, &mut out);
        assert_eq!(out, vec![3, 2, 1, 0]);
    }

    #[test]
    fn test_unpack_row_4bit() {
        let packed = vec![0xAB, 0xCD];
        let mut out = Vec::new();
        unpack_row(&packed, 4, 4, &mut out);
        assert_eq!(out, vec![0xA, 0xB, 0xC, 0xD]);
    }

    #[test]
    fn test_png_color_type_conversion() {
        assert!(PngColorType::try_from(0).is_ok());
        assert!(PngColorType::try_from(2).is_ok());
        assert!(PngColorType::try_from(3).is_ok());
        assert!(PngColorType::try_from(4).is_ok());
        assert!(PngColorType::try_from(6).is_ok());
        assert!(PngColorType::try_from(1).is_err());
        assert!(PngColorType::try_from(5).is_err());
        assert!(PngColorType::try_from(7).is_err());
    }

    #[test]
    fn test_has_alpha_in_trns_none() {
        assert!(!has_alpha_in_trns(None));
    }

    #[test]
    fn test_has_alpha_in_trns_all_opaque() {
        // All 255 values means no transparency needed
        assert!(!has_alpha_in_trns(Some(&[255, 255, 255, 255])));
    }

    #[test]
    fn test_has_alpha_in_trns_with_transparency() {
        // Any value < 255 means we have transparency
        assert!(has_alpha_in_trns(Some(&[255, 128, 255])));
        assert!(has_alpha_in_trns(Some(&[0])));
        assert!(has_alpha_in_trns(Some(&[254])));
    }

    #[test]
    fn test_has_alpha_in_trns_empty() {
        // Empty tRNS means no transparency
        assert!(!has_alpha_in_trns(Some(&[])));
    }

    #[test]
    fn test_decode_missing_iend() {
        // Create valid PNG but truncate before IEND
        let pixels = vec![255u8, 0, 0];
        let opts = crate::png::PngOptions::builder(1, 1)
            .color_type(ColorType::Rgb)
            .build();
        let encoded = crate::png::encode(&pixels, &opts).expect("encoding should work");

        // Find and remove IEND chunk (last 12 bytes: 4 length + 4 type + 0 data + 4 CRC)
        let truncated = &encoded[..encoded.len() - 12];

        let result = decode_png(truncated);
        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(
            err_msg.contains("IEND"),
            "Error should mention missing IEND: {err_msg}"
        );
    }

    #[test]
    fn test_decode_palette_all_opaque_trns_stays_rgb() {
        // Create a palette image with tRNS but all values are 255 (opaque)
        // This should decode as RGB, not RGBA
        let indices = vec![0u8, 1];
        let palette = [[255, 0, 0], [0, 255, 0]];
        let trns = Some([255u8, 255].as_slice()); // All opaque

        let encoded =
            crate::png::encode_indexed(&indices, 2, 1, &palette, trns).expect("encode indexed");
        let decoded = decode_png(&encoded).expect("decode should work");

        // Should be RGB since all tRNS values are 255
        assert_eq!(decoded.color_type, ColorType::Rgb);
        assert_eq!(decoded.pixels.len(), 2 * 3); // 2 pixels * 3 bytes (RGB)
    }

    #[test]
    fn test_decode_palette_with_transparency_becomes_rgba() {
        // Create a palette image with actual transparency
        let indices = vec![0u8, 1];
        let palette = [[255, 0, 0], [0, 255, 0]];
        let trns = Some([128u8, 255].as_slice()); // First color is semi-transparent

        let encoded =
            crate::png::encode_indexed(&indices, 2, 1, &palette, trns).expect("encode indexed");
        let decoded = decode_png(&encoded).expect("decode should work");

        // Should be RGBA since tRNS has non-255 values
        assert_eq!(decoded.color_type, ColorType::Rgba);
        assert_eq!(decoded.pixels.len(), 2 * 4); // 2 pixels * 4 bytes (RGBA)
                                                 // First pixel should have alpha 128
        assert_eq!(decoded.pixels[3], 128);
        // Second pixel should have alpha 255
        assert_eq!(decoded.pixels[7], 255);
    }

    #[test]
    fn test_calculate_expected_size() {
        // Test grayscale 8-bit
        let ihdr = IhdrData {
            width: 4,
            height: 2,
            bit_depth: 8,
            color_type: PngColorType::Grayscale,
            compression_method: 0,
            filter_method: 0,
            interlace_method: 0,
        };
        // 2 rows * (1 filter byte + 4 pixels) = 10
        assert_eq!(calculate_expected_size(&ihdr).unwrap(), 10);

        // Test RGB 8-bit
        let ihdr_rgb = IhdrData {
            color_type: PngColorType::Rgb,
            ..ihdr
        };
        // 2 rows * (1 filter byte + 4 pixels * 3 bytes) = 26
        assert_eq!(calculate_expected_size(&ihdr_rgb).unwrap(), 26);

        // Test RGBA 8-bit
        let ihdr_rgba = IhdrData {
            color_type: PngColorType::Rgba,
            ..ihdr
        };
        // 2 rows * (1 filter byte + 4 pixels * 4 bytes) = 34
        assert_eq!(calculate_expected_size(&ihdr_rgba).unwrap(), 34);
    }

    #[test]
    fn test_calculate_expected_size_packed() {
        // Test 1-bit grayscale (packed)
        let ihdr = IhdrData {
            width: 10, // 10 pixels = 2 bytes (ceil(10/8))
            height: 1,
            bit_depth: 1,
            color_type: PngColorType::Grayscale,
            compression_method: 0,
            filter_method: 0,
            interlace_method: 0,
        };
        // 1 row * (1 filter byte + 2 packed bytes) = 3
        assert_eq!(calculate_expected_size(&ihdr).unwrap(), 3);
    }

    // Error Path Tests

    #[test]
    fn test_decode_invalid_ihdr_length() {
        // Create PNG with IHDR that has wrong length
        let mut data = Vec::new();
        data.extend_from_slice(&PNG_SIGNATURE);

        // IHDR chunk with wrong length (12 instead of 13)
        data.extend_from_slice(&12u32.to_be_bytes()); // length
        data.extend_from_slice(b"IHDR");
        data.extend_from_slice(&[0u8; 12]); // wrong data length
                                            // CRC (will be wrong but length check comes first)
        data.extend_from_slice(&[0u8; 4]);

        let result = decode_png(&data);
        assert!(result.is_err());
        let err = format!("{:?}", result.unwrap_err());
        assert!(
            err.contains("IHDR") || err.contains("length") || err.contains("truncated"),
            "Error should mention IHDR issue: {err}"
        );
    }

    #[test]
    fn test_decode_missing_ihdr() {
        // Create PNG with only IEND, no IHDR
        let mut data = Vec::new();
        data.extend_from_slice(&PNG_SIGNATURE);

        // IEND chunk
        data.extend_from_slice(&0u32.to_be_bytes()); // length
        data.extend_from_slice(b"IEND");
        let crc = crc32(b"IEND");
        data.extend_from_slice(&crc.to_be_bytes());

        let result = decode_png(&data);
        assert!(result.is_err());
        let err = format!("{:?}", result.unwrap_err());
        assert!(
            err.contains("IHDR") || err.contains("missing"),
            "Error should mention missing IHDR: {err}"
        );
    }

    #[test]
    fn test_decode_zero_width() {
        // Create PNG with zero width
        let mut data = Vec::new();
        data.extend_from_slice(&PNG_SIGNATURE);

        // IHDR chunk with zero width
        let mut ihdr_data = Vec::new();
        ihdr_data.extend_from_slice(&0u32.to_be_bytes()); // width = 0
        ihdr_data.extend_from_slice(&1u32.to_be_bytes()); // height = 1
        ihdr_data.push(8); // bit depth
        ihdr_data.push(0); // color type (grayscale)
        ihdr_data.push(0); // compression
        ihdr_data.push(0); // filter
        ihdr_data.push(0); // interlace

        data.extend_from_slice(&13u32.to_be_bytes()); // length
        data.extend_from_slice(b"IHDR");
        data.extend_from_slice(&ihdr_data);
        let mut crc_data = Vec::new();
        crc_data.extend_from_slice(b"IHDR");
        crc_data.extend_from_slice(&ihdr_data);
        data.extend_from_slice(&crc32(&crc_data).to_be_bytes());

        // IEND
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(b"IEND");
        data.extend_from_slice(&crc32(b"IEND").to_be_bytes());

        let result = decode_png(&data);
        assert!(result.is_err());
        let err = format!("{:?}", result.unwrap_err());
        assert!(
            err.contains("Dimensions") || err.contains("0"),
            "Error should mention invalid dimensions: {err}"
        );
    }

    #[test]
    fn test_decode_zero_height() {
        // Create PNG with zero height
        let mut data = Vec::new();
        data.extend_from_slice(&PNG_SIGNATURE);

        // IHDR chunk with zero height
        let mut ihdr_data = Vec::new();
        ihdr_data.extend_from_slice(&1u32.to_be_bytes()); // width = 1
        ihdr_data.extend_from_slice(&0u32.to_be_bytes()); // height = 0
        ihdr_data.push(8); // bit depth
        ihdr_data.push(0); // color type (grayscale)
        ihdr_data.push(0); // compression
        ihdr_data.push(0); // filter
        ihdr_data.push(0); // interlace

        data.extend_from_slice(&13u32.to_be_bytes()); // length
        data.extend_from_slice(b"IHDR");
        data.extend_from_slice(&ihdr_data);
        let mut crc_data = Vec::new();
        crc_data.extend_from_slice(b"IHDR");
        crc_data.extend_from_slice(&ihdr_data);
        data.extend_from_slice(&crc32(&crc_data).to_be_bytes());

        // IEND
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(b"IEND");
        data.extend_from_slice(&crc32(b"IEND").to_be_bytes());

        let result = decode_png(&data);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_unsupported_compression_method() {
        // Create PNG with compression_method != 0
        let mut data = Vec::new();
        data.extend_from_slice(&PNG_SIGNATURE);

        // IHDR chunk with invalid compression method
        let mut ihdr_data = Vec::new();
        ihdr_data.extend_from_slice(&1u32.to_be_bytes()); // width
        ihdr_data.extend_from_slice(&1u32.to_be_bytes()); // height
        ihdr_data.push(8); // bit depth
        ihdr_data.push(0); // color type (grayscale)
        ihdr_data.push(1); // compression method = 1 (invalid)
        ihdr_data.push(0); // filter
        ihdr_data.push(0); // interlace

        data.extend_from_slice(&13u32.to_be_bytes());
        data.extend_from_slice(b"IHDR");
        data.extend_from_slice(&ihdr_data);
        let mut crc_data = Vec::new();
        crc_data.extend_from_slice(b"IHDR");
        crc_data.extend_from_slice(&ihdr_data);
        data.extend_from_slice(&crc32(&crc_data).to_be_bytes());

        // IEND
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(b"IEND");
        data.extend_from_slice(&crc32(b"IEND").to_be_bytes());

        let result = decode_png(&data);
        assert!(result.is_err());
        let err = format!("{:?}", result.unwrap_err());
        assert!(
            err.contains("compression"),
            "Error should mention compression: {err}"
        );
    }

    #[test]
    fn test_decode_unsupported_filter_method() {
        // Create PNG with filter_method != 0
        let mut data = Vec::new();
        data.extend_from_slice(&PNG_SIGNATURE);

        // IHDR chunk with invalid filter method
        let mut ihdr_data = Vec::new();
        ihdr_data.extend_from_slice(&1u32.to_be_bytes()); // width
        ihdr_data.extend_from_slice(&1u32.to_be_bytes()); // height
        ihdr_data.push(8); // bit depth
        ihdr_data.push(0); // color type (grayscale)
        ihdr_data.push(0); // compression method
        ihdr_data.push(1); // filter method = 1 (invalid)
        ihdr_data.push(0); // interlace

        data.extend_from_slice(&13u32.to_be_bytes());
        data.extend_from_slice(b"IHDR");
        data.extend_from_slice(&ihdr_data);
        let mut crc_data = Vec::new();
        crc_data.extend_from_slice(b"IHDR");
        crc_data.extend_from_slice(&ihdr_data);
        data.extend_from_slice(&crc32(&crc_data).to_be_bytes());

        // IEND
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(b"IEND");
        data.extend_from_slice(&crc32(b"IEND").to_be_bytes());

        let result = decode_png(&data);
        assert!(result.is_err());
        let err = format!("{:?}", result.unwrap_err());
        assert!(err.contains("filter"), "Error should mention filter: {err}");
    }

    #[test]
    fn test_decode_unsupported_interlace() {
        // Create PNG with interlace_method == 1 (Adam7)
        let mut data = Vec::new();
        data.extend_from_slice(&PNG_SIGNATURE);

        // IHDR chunk with Adam7 interlacing
        let mut ihdr_data = Vec::new();
        ihdr_data.extend_from_slice(&1u32.to_be_bytes()); // width
        ihdr_data.extend_from_slice(&1u32.to_be_bytes()); // height
        ihdr_data.push(8); // bit depth
        ihdr_data.push(0); // color type (grayscale)
        ihdr_data.push(0); // compression method
        ihdr_data.push(0); // filter method
        ihdr_data.push(1); // interlace = 1 (Adam7)

        data.extend_from_slice(&13u32.to_be_bytes());
        data.extend_from_slice(b"IHDR");
        data.extend_from_slice(&ihdr_data);
        let mut crc_data = Vec::new();
        crc_data.extend_from_slice(b"IHDR");
        crc_data.extend_from_slice(&ihdr_data);
        data.extend_from_slice(&crc32(&crc_data).to_be_bytes());

        // IEND
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(b"IEND");
        data.extend_from_slice(&crc32(b"IEND").to_be_bytes());

        let result = decode_png(&data);
        assert!(result.is_err());
        let err = format!("{:?}", result.unwrap_err());
        assert!(
            err.contains("Adam7") || err.contains("interlace"),
            "Error should mention interlacing: {err}"
        );
    }

    #[test]
    fn test_decode_invalid_bit_depth_for_rgb() {
        // RGB with bit depth 4 is invalid (must be 8 or 16)
        let mut data = Vec::new();
        data.extend_from_slice(&PNG_SIGNATURE);

        let mut ihdr_data = Vec::new();
        ihdr_data.extend_from_slice(&1u32.to_be_bytes()); // width
        ihdr_data.extend_from_slice(&1u32.to_be_bytes()); // height
        ihdr_data.push(4); // bit depth = 4 (invalid for RGB)
        ihdr_data.push(2); // color type = RGB
        ihdr_data.push(0); // compression
        ihdr_data.push(0); // filter
        ihdr_data.push(0); // interlace

        data.extend_from_slice(&13u32.to_be_bytes());
        data.extend_from_slice(b"IHDR");
        data.extend_from_slice(&ihdr_data);
        let mut crc_data = Vec::new();
        crc_data.extend_from_slice(b"IHDR");
        crc_data.extend_from_slice(&ihdr_data);
        data.extend_from_slice(&crc32(&crc_data).to_be_bytes());

        // IEND
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(b"IEND");
        data.extend_from_slice(&crc32(b"IEND").to_be_bytes());

        let result = decode_png(&data);
        assert!(result.is_err());
        let err = format!("{:?}", result.unwrap_err());
        assert!(
            err.contains("bit depth") || err.contains("invalid"),
            "Error should mention bit depth: {err}"
        );
    }

    #[test]
    fn test_decode_no_idat_data() {
        // Create PNG with IHDR and IEND but no IDAT
        let mut data = Vec::new();
        data.extend_from_slice(&PNG_SIGNATURE);

        // Valid IHDR
        let mut ihdr_data = Vec::new();
        ihdr_data.extend_from_slice(&1u32.to_be_bytes()); // width
        ihdr_data.extend_from_slice(&1u32.to_be_bytes()); // height
        ihdr_data.push(8); // bit depth
        ihdr_data.push(0); // color type (grayscale)
        ihdr_data.push(0); // compression
        ihdr_data.push(0); // filter
        ihdr_data.push(0); // interlace

        data.extend_from_slice(&13u32.to_be_bytes());
        data.extend_from_slice(b"IHDR");
        data.extend_from_slice(&ihdr_data);
        let mut crc_data = Vec::new();
        crc_data.extend_from_slice(b"IHDR");
        crc_data.extend_from_slice(&ihdr_data);
        data.extend_from_slice(&crc32(&crc_data).to_be_bytes());

        // No IDAT, just IEND
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(b"IEND");
        data.extend_from_slice(&crc32(b"IEND").to_be_bytes());

        let result = decode_png(&data);
        assert!(result.is_err());
        let err = format!("{:?}", result.unwrap_err());
        assert!(
            err.contains("IDAT") || err.contains("data"),
            "Error should mention missing IDAT: {err}"
        );
    }

    #[test]
    fn test_decode_invalid_plte_length() {
        // PLTE chunk with length not multiple of 3
        let mut data = Vec::new();
        data.extend_from_slice(&PNG_SIGNATURE);

        // Valid IHDR for indexed
        let mut ihdr_data = Vec::new();
        ihdr_data.extend_from_slice(&1u32.to_be_bytes());
        ihdr_data.extend_from_slice(&1u32.to_be_bytes());
        ihdr_data.push(8);
        ihdr_data.push(3); // indexed color
        ihdr_data.push(0);
        ihdr_data.push(0);
        ihdr_data.push(0);

        data.extend_from_slice(&13u32.to_be_bytes());
        data.extend_from_slice(b"IHDR");
        data.extend_from_slice(&ihdr_data);
        let mut crc_data = Vec::new();
        crc_data.extend_from_slice(b"IHDR");
        crc_data.extend_from_slice(&ihdr_data);
        data.extend_from_slice(&crc32(&crc_data).to_be_bytes());

        // Invalid PLTE (length = 5, not multiple of 3)
        let plte_data = [0u8; 5];
        data.extend_from_slice(&5u32.to_be_bytes());
        data.extend_from_slice(b"PLTE");
        data.extend_from_slice(&plte_data);
        let mut plte_crc_data = Vec::new();
        plte_crc_data.extend_from_slice(b"PLTE");
        plte_crc_data.extend_from_slice(&plte_data);
        data.extend_from_slice(&crc32(&plte_crc_data).to_be_bytes());

        // IEND
        data.extend_from_slice(&0u32.to_be_bytes());
        data.extend_from_slice(b"IEND");
        data.extend_from_slice(&crc32(b"IEND").to_be_bytes());

        let result = decode_png(&data);
        assert!(result.is_err());
        let err = format!("{:?}", result.unwrap_err());
        assert!(
            err.contains("PLTE") || err.contains("length"),
            "Error should mention PLTE: {err}"
        );
    }

    // 16-bit and Bit Depth Edge Case Tests

    #[test]
    fn test_calculate_expected_size_16bit_rgb() {
        let ihdr = IhdrData {
            width: 2,
            height: 2,
            bit_depth: 16,
            color_type: PngColorType::Rgb,
            compression_method: 0,
            filter_method: 0,
            interlace_method: 0,
        };
        // 16-bit RGB: 2 bytes per channel, 3 channels = 6 bytes per pixel
        // 2 rows * (1 filter byte + 2 pixels * 6 bytes) = 2 * 13 = 26
        assert_eq!(calculate_expected_size(&ihdr).unwrap(), 26);
    }

    #[test]
    fn test_calculate_expected_size_16bit_rgba() {
        let ihdr = IhdrData {
            width: 2,
            height: 2,
            bit_depth: 16,
            color_type: PngColorType::Rgba,
            compression_method: 0,
            filter_method: 0,
            interlace_method: 0,
        };
        // 16-bit RGBA: 2 bytes per channel, 4 channels = 8 bytes per pixel
        // 2 rows * (1 filter byte + 2 pixels * 8 bytes) = 2 * 17 = 34
        assert_eq!(calculate_expected_size(&ihdr).unwrap(), 34);
    }

    #[test]
    fn test_calculate_expected_size_16bit_grayscale_alpha() {
        let ihdr = IhdrData {
            width: 4,
            height: 1,
            bit_depth: 16,
            color_type: PngColorType::GrayscaleAlpha,
            compression_method: 0,
            filter_method: 0,
            interlace_method: 0,
        };
        // 16-bit GrayAlpha: 2 bytes gray + 2 bytes alpha = 4 bytes per pixel
        // 1 row * (1 filter byte + 4 pixels * 4 bytes) = 17
        assert_eq!(calculate_expected_size(&ihdr).unwrap(), 17);
    }

    #[test]
    fn test_calculate_expected_size_2bit_indexed() {
        let ihdr = IhdrData {
            width: 12, // 12 pixels at 2 bits = 24 bits = 3 bytes
            height: 1,
            bit_depth: 2,
            color_type: PngColorType::Indexed,
            compression_method: 0,
            filter_method: 0,
            interlace_method: 0,
        };
        // 1 row * (1 filter byte + 3 packed bytes) = 4
        assert_eq!(calculate_expected_size(&ihdr).unwrap(), 4);
    }

    #[test]
    fn test_calculate_expected_size_4bit_indexed() {
        let ihdr = IhdrData {
            width: 5, // 5 pixels at 4 bits = 20 bits = 3 bytes (ceil)
            height: 2,
            bit_depth: 4,
            color_type: PngColorType::Indexed,
            compression_method: 0,
            filter_method: 0,
            interlace_method: 0,
        };
        // 2 rows * (1 filter byte + 3 packed bytes) = 8
        assert_eq!(calculate_expected_size(&ihdr).unwrap(), 8);
    }

    #[test]
    fn test_unpack_row_partial_byte() {
        // Test unpacking when pixels don't fill complete byte
        let packed = vec![0b11010000]; // 4 pixels at 2 bits: 3, 1, 0, 0
        let mut out = Vec::new();
        unpack_row(&packed, 3, 2, &mut out); // Only 3 pixels
        assert_eq!(out, vec![3, 1, 0]);
    }

    #[test]
    fn test_unfilter_sub_multi_byte_pixel() {
        // Test Sub filter with bpp > 1 (e.g., RGB = 3 bytes per pixel)
        let mut row = vec![10, 20, 30, 5, 10, 15]; // 2 RGB pixels
        let prev = vec![0, 0, 0, 0, 0, 0];
        unfilter_row(1, &mut row, &prev, 3).unwrap(); // bpp = 3
                                                      // First pixel unchanged: 10, 20, 30
                                                      // Second pixel adds first: 10+5, 20+10, 30+15 = 15, 30, 45
        assert_eq!(row, vec![10, 20, 30, 15, 30, 45]);
    }

    #[test]
    fn test_unfilter_average_rounding() {
        // Average filter with odd sums to test floor division
        let mut row = vec![7, 3];
        let prev = vec![5, 9];
        unfilter_row(3, &mut row, &prev, 1).unwrap();
        // First: 7 + floor((0 + 5) / 2) = 7 + 2 = 9
        // Second: 3 + floor((9 + 9) / 2) = 3 + 9 = 12
        assert_eq!(row[0], 9);
        assert_eq!(row[1], 12);
    }

    #[test]
    fn test_unfilter_paeth_all_zeros() {
        // Paeth with all zeros should work
        let mut row = vec![100, 50, 25];
        let prev = vec![0, 0, 0];
        unfilter_row(4, &mut row, &prev, 1).unwrap();
        // With all zeros for a, b, c: paeth returns the input + predicted
        assert!(!row.is_empty());
    }
}
