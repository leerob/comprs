//! PNG decoding (opt-in via `decode` feature).
//!
//! Supported:
//! - Color types: 0 (Gray), 2 (RGB), 3 (Indexed), 4 (Gray+Alpha), 6 (RGBA)
//! - Bit depths: 1, 2, 4, 8 (16-bit rejected)
//! - Non-interlaced only
//! - PLTE/tRNS handled; palette output converted to RGB or RGBA as needed
//! - CRC validation for all chunks and Adler32 for IDAT stream
//!
//! Not supported:
//! - Interlaced PNGs
//! - 16-bit channels
//! - Ancillary chunks beyond PLTE/tRNS are ignored but validated for CRC

use crate::color::ColorType;
use crate::compress::{crc32, inflate_zlib};
use crate::error::{Error, Result};

const PNG_SIGNATURE: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
const MAX_DIMENSION: u32 = 1 << 24; // 16 million pixels (same as encoder)

/// Decoded PNG image.
#[derive(Debug, Clone)]
pub struct DecodedPng {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Color type of output pixels (8-bit channels).
    pub color_type: ColorType,
    /// Pixel data in row-major order.
    ///
    /// - Gray: width * height bytes
    /// - GrayAlpha: width * height * 2 bytes
    /// - RGB: width * height * 3 bytes
    /// - RGBA: width * height * 4 bytes
    pub data: Vec<u8>,
}

/// Decode a PNG from memory.
pub fn decode(png_data: &[u8]) -> Result<DecodedPng> {
    if png_data.len() < PNG_SIGNATURE.len() + 12 {
        return Err(Error::InvalidDecode("PNG too small".into()));
    }
    if png_data[..8] != PNG_SIGNATURE {
        return Err(Error::InvalidDecode("invalid PNG signature".into()));
    }

    let mut cursor = 8;
    let mut ihdr: Option<Ihdr> = None;
    let mut palette: Option<Vec<[u8; 3]>> = None;
    let mut trns: Option<Vec<u8>> = None;
    let mut idat: Vec<u8> = Vec::new();
    let mut seen_iend = false;

    while cursor + 12 <= png_data.len() {
        let length = u32::from_be_bytes(png_data[cursor..cursor + 4].try_into().unwrap()) as usize;
        let chunk_type = &png_data[cursor + 4..cursor + 8];
        let data_start = cursor + 8;
        let data_end = data_start + length;
        let crc_end = data_end + 4;
        if crc_end > png_data.len() {
            return Err(Error::InvalidDecode("chunk overruns buffer".into()));
        }

        let chunk_data = &png_data[data_start..data_end];
        let stored_crc = u32::from_be_bytes(png_data[data_end..crc_end].try_into().unwrap());
        let computed_crc = {
            let mut tmp = Vec::with_capacity(4 + length);
            tmp.extend_from_slice(chunk_type);
            tmp.extend_from_slice(chunk_data);
            crc32(&tmp)
        };
        if stored_crc != computed_crc {
            return Err(Error::InvalidDecode(format!(
                "CRC mismatch for chunk {:?}",
                std::str::from_utf8(chunk_type).unwrap_or("????")
            )));
        }

        match chunk_type {
            b"IHDR" => {
                if ihdr.is_some() {
                    return Err(Error::InvalidDecode("multiple IHDR chunks".into()));
                }
                ihdr = Some(parse_ihdr(chunk_data)?);
            }
            b"PLTE" => {
                if length % 3 != 0 {
                    return Err(Error::InvalidDecode("PLTE length not multiple of 3".into()));
                }
                let mut pal = Vec::with_capacity(length / 3);
                for chunk in chunk_data.chunks_exact(3) {
                    pal.push([chunk[0], chunk[1], chunk[2]]);
                }
                palette = Some(pal);
            }
            b"tRNS" => {
                trns = Some(chunk_data.to_vec());
            }
            b"IDAT" => idat.extend_from_slice(chunk_data),
            b"IEND" => {
                seen_iend = true;
                break;
            }
            _ => {
                // Ancillary chunks are ignored but CRC-validated above.
            }
        }

        cursor = crc_end;
    }

    if !seen_iend {
        return Err(Error::InvalidDecode("missing IEND chunk".into()));
    }
    let ihdr = ihdr.ok_or_else(|| Error::InvalidDecode("missing IHDR chunk".into()))?;

    // Validate constraints
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
    if ihdr.interlace_method != 0 {
        return Err(Error::UnsupportedDecode(
            "interlaced PNG not supported".into(),
        ));
    }
    validate_color(&ihdr, palette.as_ref())?;

    let bits_per_pixel = match ihdr.color_type {
        0 => ihdr.bit_depth as usize,
        2 => 3 * ihdr.bit_depth as usize,
        3 => ihdr.bit_depth as usize,
        4 => 2 * ihdr.bit_depth as usize,
        6 => 4 * ihdr.bit_depth as usize,
        _ => unreachable!(),
    };
    let filtered_row_bytes = (ihdr.width as usize * bits_per_pixel).div_ceil(8) + 1;
    let expected_bytes = filtered_row_bytes
        .checked_mul(ihdr.height as usize)
        .ok_or_else(|| Error::InvalidDecode("image size overflow".into()))?;

    let decompressed = inflate_zlib(&idat, Some(expected_bytes))?;
    if decompressed.len() != expected_bytes {
        return Err(Error::InvalidDecode(
            "IDAT decompressed size mismatch".into(),
        ));
    }

    let bpp_for_filter = bytes_per_pixel_for_filter(&ihdr);
    let row_len = filtered_row_bytes - 1;
    let mut raw = unfilter_scanlines(&decompressed, row_len, ihdr.height as usize, bpp_for_filter)?;

    // Expand packed bits to 8-bit channels
    let expanded = match ihdr.color_type {
        0 => expand_gray(
            &raw,
            ihdr.width as usize,
            ihdr.height as usize,
            ihdr.bit_depth,
        )?,
        4 => expand_graya(&raw),
        2 => {
            if ihdr.bit_depth != 8 {
                return Err(Error::UnsupportedDecode("RGB bit depth must be 8".into()));
            }
            std::mem::take(&mut raw)
        }
        6 => {
            if ihdr.bit_depth != 8 {
                return Err(Error::UnsupportedDecode("RGBA bit depth must be 8".into()));
            }
            std::mem::take(&mut raw)
        }
        3 => {
            let pal = palette
                .as_ref()
                .ok_or_else(|| Error::InvalidDecode("palette image missing PLTE chunk".into()))?;
            expand_palette(
                &raw,
                ihdr.width as usize,
                ihdr.height as usize,
                ihdr.bit_depth,
                pal,
                trns.as_deref(),
            )?
        }
        _ => unreachable!(),
    };

    let color_type = match ihdr.color_type {
        0 => ColorType::Gray,
        2 => ColorType::Rgb,
        3 => {
            if has_alpha_in_trns(trns.as_deref()) {
                ColorType::Rgba
            } else {
                ColorType::Rgb
            }
        }
        4 => ColorType::GrayAlpha,
        6 => ColorType::Rgba,
        _ => unreachable!(),
    };

    Ok(DecodedPng {
        width: ihdr.width,
        height: ihdr.height,
        color_type,
        data: expanded,
    })
}

fn has_alpha_in_trns(trns: Option<&[u8]>) -> bool {
    trns.map(|a| a.iter().any(|&v| v != 0xFF)).unwrap_or(false)
}

struct Ihdr {
    width: u32,
    height: u32,
    bit_depth: u8,
    color_type: u8,
    interlace_method: u8,
}

fn parse_ihdr(data: &[u8]) -> Result<Ihdr> {
    if data.len() != 13 {
        return Err(Error::InvalidDecode("IHDR length must be 13".into()));
    }
    let width = u32::from_be_bytes(data[0..4].try_into().unwrap());
    let height = u32::from_be_bytes(data[4..8].try_into().unwrap());
    let bit_depth = data[8];
    let color_type = data[9];
    let compression = data[10];
    let filter = data[11];
    let interlace_method = data[12];

    if compression != 0 {
        return Err(Error::UnsupportedDecode(
            "unsupported compression method".into(),
        ));
    }
    if filter != 0 {
        return Err(Error::UnsupportedDecode("unsupported filter method".into()));
    }
    Ok(Ihdr {
        width,
        height,
        bit_depth,
        color_type,
        interlace_method,
    })
}

fn validate_color(ihdr: &Ihdr, palette: Option<&Vec<[u8; 3]>>) -> Result<()> {
    match ihdr.color_type {
        0 => match ihdr.bit_depth {
            1 | 2 | 4 | 8 => {}
            _ => {
                return Err(Error::UnsupportedDecode(
                    "unsupported bit depth for grayscale".into(),
                ))
            }
        },
        2 => {
            if ihdr.bit_depth != 8 {
                return Err(Error::UnsupportedDecode(
                    "RGB only supports 8-bit depth".into(),
                ));
            }
        }
        3 => {
            if !matches!(ihdr.bit_depth, 1 | 2 | 4 | 8) {
                return Err(Error::UnsupportedDecode(
                    "palette images must use 1,2,4,8 bit depth".into(),
                ));
            }
            if palette.is_none() {
                return Err(Error::InvalidDecode(
                    "palette image missing PLTE chunk".into(),
                ));
            }
        }
        4 => {
            if ihdr.bit_depth != 8 {
                return Err(Error::UnsupportedDecode(
                    "GrayAlpha only supports 8-bit depth".into(),
                ));
            }
        }
        6 => {
            if ihdr.bit_depth != 8 {
                return Err(Error::UnsupportedDecode(
                    "RGBA only supports 8-bit depth".into(),
                ));
            }
        }
        _ => {
            return Err(Error::UnsupportedDecode(
                "unsupported PNG color type".into(),
            ))
        }
    }
    Ok(())
}

fn bytes_per_pixel_for_filter(ihdr: &Ihdr) -> usize {
    match ihdr.color_type {
        0 | 3 => 1, // packed samples filter on bytes
        2 => (3 * ihdr.bit_depth as usize).div_ceil(8),
        4 => (2 * ihdr.bit_depth as usize).div_ceil(8),
        6 => (4 * ihdr.bit_depth as usize).div_ceil(8),
        _ => 1,
    }
}

fn unfilter_scanlines(data: &[u8], row_len: usize, height: usize, bpp: usize) -> Result<Vec<u8>> {
    if bpp == 0 {
        return Err(Error::InvalidDecode(
            "bytes per pixel cannot be zero".into(),
        ));
    }
    let stride = row_len + 1;
    if stride
        .checked_mul(height)
        .map(|v| v > data.len())
        .unwrap_or(true)
    {
        return Err(Error::InvalidDecode("scanline data length mismatch".into()));
    }
    let mut out = Vec::with_capacity(row_len * height);
    let mut prev_row = vec![0u8; row_len];
    for y in 0..height {
        let start = y * stride;
        let filter = data[start];
        let raw = &data[start + 1..start + stride];
        if raw.len() != row_len {
            return Err(Error::InvalidDecode("row length mismatch".into()));
        }
        let mut row = vec![0u8; row_len];
        match filter {
            0 => row.copy_from_slice(raw),
            1 => unfilter_sub(raw, bpp, &mut row),
            2 => unfilter_up(raw, &prev_row, &mut row),
            3 => unfilter_avg(raw, &prev_row, bpp, &mut row),
            4 => unfilter_paeth(raw, &prev_row, bpp, &mut row),
            _ => return Err(Error::InvalidDecode("invalid filter type".into())),
        }
        out.extend_from_slice(&row);
        prev_row = row;
    }
    Ok(out)
}

fn unfilter_sub(raw: &[u8], bpp: usize, out: &mut [u8]) {
    for i in 0..raw.len() {
        let left = if i >= bpp { out[i - bpp] } else { 0 };
        out[i] = raw[i].wrapping_add(left);
    }
}

fn unfilter_up(raw: &[u8], prev: &[u8], out: &mut [u8]) {
    for i in 0..raw.len() {
        out[i] = raw[i].wrapping_add(prev[i]);
    }
}

fn unfilter_avg(raw: &[u8], prev: &[u8], bpp: usize, out: &mut [u8]) {
    for i in 0..raw.len() {
        let left = if i >= bpp { out[i - bpp] } else { 0 };
        let up = prev[i];
        out[i] = raw[i].wrapping_add(((left as u16 + up as u16) / 2) as u8);
    }
}

fn paeth_predictor(a: u8, b: u8, c: u8) -> u8 {
    let a = a as i32;
    let b = b as i32;
    let c = c as i32;
    let p = a + b - c;
    let pa = (p - a).abs();
    let pb = (p - b).abs();
    let pc = (p - c).abs();
    if pa <= pb && pa <= pc {
        a as u8
    } else if pb <= pc {
        b as u8
    } else {
        c as u8
    }
}

fn unfilter_paeth(raw: &[u8], prev: &[u8], bpp: usize, out: &mut [u8]) {
    for i in 0..raw.len() {
        let left = if i >= bpp { out[i - bpp] } else { 0 };
        let up = prev[i];
        let up_left = if i >= bpp { prev[i - bpp] } else { 0 };
        out[i] = raw[i].wrapping_add(paeth_predictor(left, up, up_left));
    }
}

fn expand_gray(data: &[u8], width: usize, height: usize, bit_depth: u8) -> Result<Vec<u8>> {
    if bit_depth == 8 {
        return Ok(data.to_vec());
    }
    let bits = bit_depth as usize;
    let row_len = (width * bits).div_ceil(8);
    let mut out = Vec::with_capacity(width * height);
    for row_idx in 0..height {
        let row_start = row_idx * row_len;
        let row = &data[row_start..row_start + row_len];
        let mut idx = 0;
        let mut bit_pos = 0;
        for _ in 0..width {
            if idx >= row.len() {
                break;
            }
            let byte = row[idx];
            let remaining = 8 - bit_pos;
            let take = bits.min(remaining);
            let mask = (((1u16 << take) - 1) << (remaining - take)) as u8;
            let sample = (byte & mask) >> (remaining - take);
            let value = scale_sample(sample, bits as u8);
            out.push(value);
            bit_pos += take;
            if bit_pos >= 8 {
                bit_pos = 0;
                idx += 1;
            }
        }
    }
    if out.len() != width * height {
        return Err(Error::InvalidDecode(
            "not enough grayscale samples after expansion".into(),
        ));
    }
    Ok(out)
}

fn expand_graya(data: &[u8]) -> Vec<u8> {
    data.to_vec()
}

fn expand_palette(
    data: &[u8],
    width: usize,
    height: usize,
    bit_depth: u8,
    palette: &[[u8; 3]],
    trns: Option<&[u8]>,
) -> Result<Vec<u8>> {
    let bits = bit_depth as usize;
    let row_len = (width * bits).div_ceil(8);
    let mut indices = Vec::with_capacity(width * height);
    for row_idx in 0..height {
        let row_start = row_idx * row_len;
        let row = &data[row_start..row_start + row_len];
        let mut idx = 0;
        let mut bit_pos = 0;
        for _ in 0..width {
            if idx >= row.len() {
                break;
            }
            let byte = row[idx];
            let remaining = 8 - bit_pos;
            let take = bits.min(remaining);
            let mask = (((1u16 << take) - 1) << (remaining - take)) as u8;
            let sample = (byte & mask) >> (remaining - take);
            indices.push(sample as usize);
            bit_pos += take;
            if bit_pos >= 8 {
                bit_pos = 0;
                idx += 1;
            }
        }
    }
    if indices.len() != width * height {
        return Err(Error::InvalidDecode(
            "not enough palette indices after expansion".into(),
        ));
    }

    let has_alpha = has_alpha_in_trns(trns);
    if has_alpha {
        let mut out = Vec::with_capacity(width * 4);
        for &i in &indices {
            let rgb = *palette
                .get(i)
                .ok_or_else(|| Error::InvalidDecode("palette index out of range".into()))?;
            let alpha = trns.and_then(|t| t.get(i)).copied().unwrap_or(0xFF);
            out.extend_from_slice(&[rgb[0], rgb[1], rgb[2], alpha]);
        }
        Ok(out)
    } else {
        let mut out = Vec::with_capacity(width * 3);
        for &i in &indices {
            let rgb = *palette
                .get(i)
                .ok_or_else(|| Error::InvalidDecode("palette index out of range".into()))?;
            out.extend_from_slice(&rgb);
        }
        Ok(out)
    }
}

fn scale_sample(sample: u8, bits: u8) -> u8 {
    match bits {
        1 => {
            if sample == 0 {
                0
            } else {
                255
            }
        }
        2 => (sample as u16 * 255 / 3) as u8, // 0,1,2,3 -> 0..255
        4 => (sample as u16 * 17) as u8,      // 0..15 -> 0..255
        8 => sample,
        _ => sample,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::png;

    fn recompute_ihdr_crc(png_bytes: &mut [u8]) {
        // Assumes IHDR is first chunk immediately after signature.
        let ihdr_type_start = 8 + 4; // length + type start
        let ihdr_data_start = ihdr_type_start + 4;
        let ihdr_data_end = ihdr_data_start + 13;
        let crc_start = ihdr_data_end;
        let mut payload = Vec::with_capacity(4 + 13);
        payload.extend_from_slice(&png_bytes[ihdr_type_start..ihdr_type_start + 4]); // "IHDR"
        payload.extend_from_slice(&png_bytes[ihdr_data_start..ihdr_data_end]);
        let crc = crate::compress::crc32::crc32(&payload).to_be_bytes();
        png_bytes[crc_start..crc_start + 4].copy_from_slice(&crc);
    }

    fn encode_then_decode(data: &[u8], width: u32, height: u32, ct: ColorType) -> DecodedPng {
        let encoded = png::encode(data, width, height, ct).expect("encode");
        png::decode(&encoded).expect("decode")
    }

    #[test]
    fn roundtrip_rgb() {
        let pixels = vec![
            255, 0, 0, // red
            0, 255, 0, // green
            0, 0, 255, // blue
            255, 255, 0, // yellow
        ];
        let decoded = encode_then_decode(&pixels, 2, 2, ColorType::Rgb);
        assert_eq!(decoded.color_type, ColorType::Rgb);
        assert_eq!(decoded.data, pixels);
    }

    #[test]
    fn roundtrip_rgba() {
        let pixels = vec![
            255, 0, 0, 128, //
            0, 255, 0, 255, //
        ];
        let decoded = encode_then_decode(&pixels, 1, 2, ColorType::Rgba);
        assert_eq!(decoded.color_type, ColorType::Rgba);
        assert_eq!(decoded.data, pixels);
    }

    #[test]
    fn roundtrip_gray_1bit() {
        // 4 pixels: 0, 1, 0, 1 pattern
        let pixels = vec![0, 255, 0, 255];
        let opts = png::PngOptions {
            reduce_color_type: true, // allow bit-depth reduction
            ..Default::default()
        };
        let encoded = png::encode_with_options(&pixels, 4, 1, ColorType::Gray, &opts).unwrap();
        let decoded = png::decode(&encoded).unwrap();
        assert_eq!(decoded.color_type, ColorType::Gray);
        assert_eq!(decoded.data, pixels);
    }

    #[test]
    fn roundtrip_palette_trns() {
        // 2-color RGBA, force palette via reduce_palette
        let pixels = vec![
            255, 0, 0, 0, // transparent red
            0, 255, 0, 255, // opaque green
        ];
        let opts = png::PngOptions {
            reduce_palette: true,
            ..Default::default()
        };
        let encoded = png::encode_with_options(&pixels, 2, 1, ColorType::Rgba, &opts).unwrap();
        let decoded = png::decode(&encoded).unwrap();
        assert_eq!(decoded.color_type, ColorType::Rgba);
        assert_eq!(decoded.data, pixels);
    }

    #[test]
    fn rejects_interlaced() {
        // Build minimal interlaced PNG (manually tweak IHDR bit)
        let pixels = vec![255u8, 0, 0];
        let mut png_bytes = png::encode(&pixels, 1, 1, ColorType::Rgb).unwrap();
        // IHDR interlace byte is the last byte of IHDR data (offset 28)
        png_bytes[28] = 1;
        recompute_ihdr_crc(&mut png_bytes);
        let err = png::decode(&png_bytes).unwrap_err();
        assert!(matches!(err, Error::UnsupportedDecode(_)));
    }

    #[test]
    fn rejects_bad_crc() {
        let pixels = vec![0u8, 0, 0];
        let mut png_bytes = png::encode(&pixels, 1, 1, ColorType::Rgb).unwrap();
        // Corrupt IHDR CRC (last byte) so CRC check fails
        let ihdr_crc_pos = 8 + 4 + 4 + 13; // sig + len + type + data
        if ihdr_crc_pos + 4 <= png_bytes.len() {
            png_bytes[ihdr_crc_pos + 3] ^= 0xFF;
        }
        let err = png::decode(&png_bytes).unwrap_err();
        assert!(matches!(err, Error::InvalidDecode(_)));
    }

    #[test]
    fn rejects_palette_index_out_of_range() {
        // Build palette image with 1 palette entry but data index=1
        let encoded = png::encode_indexed(&[1u8], 1, 1, &[[0, 0, 0]], None).unwrap();
        let err = png::decode(&encoded).unwrap_err();
        assert!(matches!(err, Error::InvalidDecode(_)));
    }

    #[test]
    fn rejects_16bit_bit_depth() {
        let pixels = vec![255u8, 0, 0];
        let mut png_bytes = png::encode(&pixels, 1, 1, ColorType::Rgb).unwrap();
        // Set bit depth to 16 (offset 24) and fix CRC
        png_bytes[24] = 16;
        recompute_ihdr_crc(&mut png_bytes);
        let err = png::decode(&png_bytes).unwrap_err();
        assert!(matches!(err, Error::UnsupportedDecode(_)));
    }
}
