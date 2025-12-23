//! PNG encoder implementation.
//!
//! Implements PNG encoding according to the PNG specification (RFC 2083).

pub mod chunk;
pub mod filter;

use crate::color::ColorType;
use crate::compress::deflate::deflate_zlib_packed;
#[cfg(feature = "timing")]
use crate::compress::deflate::deflate_zlib_packed_with_stats;
use crate::compress::deflate::DeflateStats;
use crate::error::{Error, Result};

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
    /// Palette quantization options (off by default).
    pub quantization: QuantizationOptions,
}

impl Default for PngOptions {
    fn default() -> Self {
        Self {
            // Prefer speed; level 2 favors throughput over ratio.
            compression_level: 2,
            // AdaptiveFast reduces per-row work with minimal compression impact.
            filter_strategy: FilterStrategy::AdaptiveFast,
            // Quantization off by default to preserve lossless output.
            quantization: QuantizationOptions::default(),
        }
    }
}

impl PngOptions {
    /// Speed-focused preset (matches current default).
    pub fn fast() -> Self {
        Self {
            compression_level: 2,
            filter_strategy: FilterStrategy::AdaptiveFast,
            quantization: QuantizationOptions::default(),
        }
    }

    /// Balanced preset targeting better compression at moderate speed.
    pub fn balanced() -> Self {
        Self {
            compression_level: 6,
            filter_strategy: FilterStrategy::Adaptive,
            quantization: QuantizationOptions::default(),
        }
    }

    /// Highest compression preset; slowest.
    pub fn max_compression() -> Self {
        Self {
            compression_level: 9,
            filter_strategy: FilterStrategy::AdaptiveSampled { interval: 2 },
            quantization: QuantizationOptions::default(),
        }
    }
}

/// Quantization mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationMode {
    /// Disable palette quantization (lossless path).
    Off,
    /// Apply quantization when possible/beneficial for PNG (auto).
    Auto,
    /// Force quantization regardless of heuristics.
    Force,
}

/// Options controlling palette quantization.
#[derive(Debug, Clone)]
pub struct QuantizationOptions {
    /// Strategy: Off/Auto/Force.
    pub mode: QuantizationMode,
    /// Maximum palette size (1-256).
    pub max_colors: u16,
    /// Enable Floyd–Steinberg dithering (on RGB channels only).
    pub dithering: bool,
}

impl Default for QuantizationOptions {
    fn default() -> Self {
        Self {
            mode: QuantizationMode::Off,
            max_colors: 256,
            dithering: false,
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

/// Encode indexed (palette) pixel data as PNG.
///
/// The `data` slice contains palette indices (0..palette.len()). The palette
/// must contain between 1 and 256 entries. Optional `transparency` supplies
/// per-entry alpha values (tRNS); its length must not exceed the palette
/// length. Bit depth is fixed to 8 for indexed output.
pub fn encode_indexed(
    data: &[u8],
    width: u32,
    height: u32,
    palette: &[[u8; 3]],
    transparency: Option<&[u8]>,
) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    encode_indexed_into(
        &mut output,
        data,
        width,
        height,
        palette,
        transparency,
        &PngOptions::default(),
    )?;
    Ok(output)
}

/// Encode indexed (palette) pixel data as PNG with custom options.
pub fn encode_indexed_with_options(
    data: &[u8],
    width: u32,
    height: u32,
    palette: &[[u8; 3]],
    transparency: Option<&[u8]>,
    options: &PngOptions,
) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    encode_indexed_into(
        &mut output,
        data,
        width,
        height,
        palette,
        transparency,
        options,
    )?;
    Ok(output)
}

/// Encode indexed (palette) pixel data into a caller-provided buffer.
pub fn encode_indexed_into(
    output: &mut Vec<u8>,
    data: &[u8],
    width: u32,
    height: u32,
    palette: &[[u8; 3]],
    transparency: Option<&[u8]>,
    options: &PngOptions,
) -> Result<()> {
    validate_common(width, height, options)?;

    let palette_len = palette.len();
    if palette_len == 0 || palette_len > 256 {
        return Err(Error::InvalidPaletteLength { len: palette_len });
    }
    if let Some(alpha) = transparency {
        if alpha.len() > palette_len {
            return Err(Error::InvalidTransparencyLength {
                palette_len,
                alpha_len: alpha.len(),
            });
        }
    }

    let expected_len = width as usize * height as usize; // 1 byte per index
    if data.len() != expected_len {
        return Err(Error::InvalidDataLength {
            expected: expected_len,
            actual: data.len(),
        });
    }

    output.clear();
    output.reserve(expected_len / 2 + 2048);

    // Write signature and IHDR (color type 3, bit depth 8).
    output.extend_from_slice(&PNG_SIGNATURE);
    write_ihdr_indexed(output, width, height);

    write_plte_chunk(output, palette);
    if let Some(alpha) = transparency {
        write_trns_chunk(output, alpha);
    }

    // Palette-aware filtering: avoid adaptive overhead; prefer None/Sub.
    let mut palette_options = options.clone();
    palette_options.filter_strategy = match options.filter_strategy {
        FilterStrategy::Adaptive
        | FilterStrategy::AdaptiveFast
        | FilterStrategy::AdaptiveSampled { .. } => FilterStrategy::None,
        other => other,
    };

    let filtered = filter::apply_filters(data, width, height, 1, &palette_options);
    let compressed = deflate_zlib_packed(&filtered, palette_options.compression_level);

    write_idat_chunks(output, &compressed);
    write_iend(output);
    Ok(())
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
    encode_internal(output, data, width, height, color_type, options, false)
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
    let mut stats = DeflateStats::default();
    encode_internal_with_stats(
        output, data, width, height, color_type, options, true, &mut stats,
    )?;
    Ok(stats)
}

/// Write IHDR (image header) chunk.
fn write_ihdr(output: &mut Vec<u8>, width: u32, height: u32, color_type: ColorType) {
    let mut ihdr_data = Vec::with_capacity(13);

    // Width (4 bytes, big-endian)
    ihdr_data.extend_from_slice(&width.to_be_bytes());

    // Height (4 bytes, big-endian)
    ihdr_data.extend_from_slice(&height.to_be_bytes());

    // Bit depth (1 byte)
    ihdr_data.push(color_type.png_bit_depth());

    // Color type (1 byte)
    ihdr_data.push(color_type.png_color_type());

    // Compression method (1 byte) - always 0 (DEFLATE)
    ihdr_data.push(0);

    // Filter method (1 byte) - always 0 (adaptive)
    ihdr_data.push(0);

    // Interlace method (1 byte) - 0 (no interlace)
    ihdr_data.push(0);

    chunk::write_chunk(output, b"IHDR", &ihdr_data);
}

/// Write IHDR for indexed color (bit depth 8, color type 3).
fn write_ihdr_indexed(output: &mut Vec<u8>, width: u32, height: u32) {
    let mut ihdr_data = Vec::with_capacity(13);
    ihdr_data.extend_from_slice(&width.to_be_bytes());
    ihdr_data.extend_from_slice(&height.to_be_bytes());
    ihdr_data.push(8); // bit depth
    ihdr_data.push(3); // color type: indexed
    ihdr_data.push(0); // compression
    ihdr_data.push(0); // filter
    ihdr_data.push(0); // interlace
    chunk::write_chunk(output, b"IHDR", &ihdr_data);
}

/// Write PLTE chunk.
fn write_plte_chunk(output: &mut Vec<u8>, palette: &[[u8; 3]]) {
    let mut data = Vec::with_capacity(palette.len() * 3);
    for entry in palette {
        data.extend_from_slice(entry);
    }
    chunk::write_chunk(output, b"PLTE", &data);
}

/// Write optional tRNS chunk for palette transparency.
fn write_trns_chunk(output: &mut Vec<u8>, alpha: &[u8]) {
    chunk::write_chunk(output, b"tRNS", alpha);
}

fn validate_common(width: u32, height: u32, options: &PngOptions) -> Result<()> {
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
    Ok(())
}

fn maybe_trim_transparency(alpha: &[u8]) -> Option<Vec<u8>> {
    if alpha.is_empty() {
        return None;
    }
    if alpha.iter().all(|&a| a == 255) {
        return None;
    }
    let mut last = 0usize;
    for (i, &a) in alpha.iter().enumerate() {
        if a != 255 {
            last = i;
        }
    }
    Some(alpha[..=last].to_vec())
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

fn encode_internal(
    output: &mut Vec<u8>,
    data: &[u8],
    width: u32,
    height: u32,
    color_type: ColorType,
    options: &PngOptions,
    _collect_stats: bool,
) -> Result<()> {
    let mut dummy_stats = DeflateStats::default();
    encode_internal_with_stats(
        output,
        data,
        width,
        height,
        color_type,
        options,
        false,
        &mut dummy_stats,
    )
}

fn encode_internal_with_stats(
    output: &mut Vec<u8>,
    data: &[u8],
    width: u32,
    height: u32,
    color_type: ColorType,
    options: &PngOptions,
    collect_stats: bool,
    stats_out: &mut DeflateStats,
) -> Result<()> {
    validate_common(width, height, options)?;
    let bpp = color_type.bytes_per_pixel();
    let expected_len = width as usize * height as usize * bpp;
    if data.len() != expected_len {
        return Err(Error::InvalidDataLength {
            expected: expected_len,
            actual: data.len(),
        });
    }

    // Quantization path for RGB/RGBA when requested.
    let should_quantize = match options.quantization.mode {
        QuantizationMode::Off => false,
        QuantizationMode::Force => matches!(color_type, ColorType::Rgb | ColorType::Rgba),
        QuantizationMode::Auto => {
            matches!(color_type, ColorType::Rgb | ColorType::Rgba)
                && should_quantize_auto(
                    data,
                    bpp,
                    options.quantization.max_colors.min(256) as usize,
                )
        }
    };

    if should_quantize {
        let (palette_rgba, indices) = quantize_image(
            data,
            width,
            height,
            color_type,
            options.quantization.max_colors.min(256) as usize,
            options.quantization.dithering,
        )?;

        // Build PLTE and optional tRNS
        let mut plte: Vec<[u8; 3]> = Vec::with_capacity(palette_rgba.len());
        let mut alpha: Vec<u8> = Vec::with_capacity(palette_rgba.len());
        for [r, g, b, a] in &palette_rgba {
            plte.push([*r, *g, *b]);
            alpha.push(*a);
        }
        let alpha = maybe_trim_transparency(&alpha);

        return encode_indexed_into(
            output,
            &indices,
            width,
            height,
            &plte,
            alpha.as_deref(),
            options,
        );
    }

    // Lossless original path.
    output.clear();
    output.reserve(expected_len / 2 + 1024);
    output.extend_from_slice(&PNG_SIGNATURE);
    write_ihdr(output, width, height, color_type);

    let filtered = filter::apply_filters(data, width, height, bpp, options);
    if collect_stats {
        #[cfg(feature = "timing")]
        {
            let (compressed, stats) =
                deflate_zlib_packed_with_stats(&filtered, options.compression_level);
            write_idat_chunks(output, &compressed);
            write_iend(output);
            *stats_out = stats;
            return Ok(());
        }
        #[cfg(not(feature = "timing"))]
        {
            let compressed = deflate_zlib_packed(&filtered, options.compression_level);
            write_idat_chunks(output, &compressed);
            write_iend(output);
            *stats_out = DeflateStats::default();
            return Ok(());
        }
    }

    let compressed = deflate_zlib_packed(&filtered, options.compression_level);
    write_idat_chunks(output, &compressed);
    write_iend(output);
    Ok(())
}
#[derive(Clone)]
struct ColorCount {
    rgba: [u8; 4],
    count: u32,
}

#[derive(Clone)]
struct ColorBox {
    colors: Vec<ColorCount>,
    r_min: u8,
    r_max: u8,
    g_min: u8,
    g_max: u8,
    b_min: u8,
    b_max: u8,
    a_min: u8,
    a_max: u8,
}

impl ColorBox {
    fn from_colors(colors: Vec<ColorCount>) -> Self {
        let mut r_min = 255;
        let mut r_max = 0;
        let mut g_min = 255;
        let mut g_max = 0;
        let mut b_min = 255;
        let mut b_max = 0;
        let mut a_min = 255;
        let mut a_max = 0;
        for c in &colors {
            let [r, g, b, a] = c.rgba;
            r_min = r_min.min(r);
            r_max = r_max.max(r);
            g_min = g_min.min(g);
            g_max = g_max.max(g);
            b_min = b_min.min(b);
            b_max = b_max.max(b);
            a_min = a_min.min(a);
            a_max = a_max.max(a);
        }
        Self {
            colors,
            r_min,
            r_max,
            g_min,
            g_max,
            b_min,
            b_max,
            a_min,
            a_max,
        }
    }

    fn range(&self) -> (u8, u8) {
        let r_range = self.r_max - self.r_min;
        let g_range = self.g_max - self.g_min;
        let b_range = self.b_max - self.b_min;
        let a_range = self.a_max - self.a_min;
        let mut max_range = r_range;
        let mut channel = 0u8;
        if g_range > max_range {
            max_range = g_range;
            channel = 1;
        }
        if b_range > max_range {
            max_range = b_range;
            channel = 2;
        }
        if a_range > max_range {
            max_range = a_range;
            channel = 3;
        }
        (channel, max_range)
    }

    fn can_split(&self) -> bool {
        self.colors.len() > 1
    }

    fn split(self) -> (ColorBox, ColorBox) {
        let (channel, _) = self.range();
        let mut colors = self.colors;
        colors.sort_by_key(|c| match channel {
            0 => c.rgba[0],
            1 => c.rgba[1],
            2 => c.rgba[2],
            _ => c.rgba[3],
        });

        let total: u32 = colors.iter().map(|c| c.count).sum();
        let mut acc = 0;
        let mut split_idx = 0;
        for (i, c) in colors.iter().enumerate() {
            acc += c.count;
            if acc >= total / 2 {
                split_idx = i;
                break;
            }
        }
        let left = colors[..=split_idx].to_vec();
        let right = colors[split_idx + 1..].to_vec();
        (ColorBox::from_colors(left), ColorBox::from_colors(right))
    }

    fn make_palette_entry(&self) -> [u8; 4] {
        let mut r_sum: u64 = 0;
        let mut g_sum: u64 = 0;
        let mut b_sum: u64 = 0;
        let mut a_sum: u64 = 0;
        let mut total: u64 = 0;
        for c in &self.colors {
            let cnt = c.count as u64;
            let [r, g, b, a] = c.rgba;
            r_sum += r as u64 * cnt;
            g_sum += g as u64 * cnt;
            b_sum += b as u64 * cnt;
            a_sum += a as u64 * cnt;
            total += cnt;
        }
        if total == 0 {
            return [0, 0, 0, 255];
        }
        [
            (r_sum / total) as u8,
            (g_sum / total) as u8,
            (b_sum / total) as u8,
            (a_sum / total) as u8,
        ]
    }
}

fn median_cut_palette(colors: Vec<ColorCount>, max_colors: usize) -> Vec<[u8; 4]> {
    if colors.is_empty() {
        return vec![[0, 0, 0, 255]];
    }
    let mut boxes = vec![ColorBox::from_colors(colors)];
    while boxes.len() < max_colors {
        // pick box with largest range
        let (idx, _) = boxes
            .iter()
            .enumerate()
            .max_by_key(|(_, b)| {
                let (_, r) = b.range();
                r
            })
            .unwrap();
        if !boxes[idx].can_split() {
            break;
        }
        let b = boxes.remove(idx);
        let (l, r) = b.split();
        if !l.colors.is_empty() {
            boxes.push(l);
        }
        if !r.colors.is_empty() {
            boxes.push(r);
        }
    }

    boxes.into_iter().map(|b| b.make_palette_entry()).collect()
}

fn nearest_palette_index(color: [u8; 4], palette: &[[u8; 4]]) -> u8 {
    let mut best_idx = 0u8;
    let mut best_dist = u32::MAX;
    for (i, p) in palette.iter().enumerate() {
        let dr = color[0] as i32 - p[0] as i32;
        let dg = color[1] as i32 - p[1] as i32;
        let db = color[2] as i32 - p[2] as i32;
        let da = color[3] as i32 - p[3] as i32;
        let dist = (dr * dr + dg * dg + db * db + da * da) as u32;
        if dist < best_dist {
            best_dist = dist;
            best_idx = i as u8;
        }
    }
    best_idx
}

fn quantize_image(
    data: &[u8],
    width: u32,
    height: u32,
    color_type: ColorType,
    max_colors: usize,
    dithering: bool,
) -> Result<(Vec<[u8; 4]>, Vec<u8>)> {
    let bpp = color_type.bytes_per_pixel();
    if bpp != 3 && bpp != 4 {
        return Err(Error::UnsupportedColorType);
    }
    let mut hist = std::collections::HashMap::<u32, u32>::new();
    for chunk in data.chunks_exact(bpp) {
        let (r, g, b, a) = match (bpp, chunk) {
            (3, [r, g, b]) => (*r, *g, *b, 255u8),
            (4, [r, g, b, a]) => (*r, *g, *b, *a),
            _ => unreachable!(),
        };
        let key = ((r as u32) << 24) | ((g as u32) << 16) | ((b as u32) << 8) | a as u32;
        *hist.entry(key).or_insert(0) += 1;
    }

    let colors: Vec<ColorCount> = hist
        .into_iter()
        .map(|(k, count)| {
            let r = (k >> 24) as u8;
            let g = (k >> 16) as u8;
            let b = (k >> 8) as u8;
            let a = k as u8;
            ColorCount {
                rgba: [r, g, b, a],
                count,
            }
        })
        .collect();

    // Early out: already within palette size.
    if colors.len() as usize <= max_colors {
        let palette: Vec<[u8; 4]> = colors.iter().map(|c| c.rgba).collect();
        // Map color to index using direct mapping
        let mut map = std::collections::HashMap::new();
        for (i, c) in palette.iter().enumerate() {
            let key =
                ((c[0] as u32) << 24) | ((c[1] as u32) << 16) | ((c[2] as u32) << 8) | c[3] as u32;
            map.insert(key, i as u8);
        }
        let mut indices = Vec::with_capacity(data.len() / bpp);
        for chunk in data.chunks_exact(bpp) {
            let (r, g, b, a) = match (bpp, chunk) {
                (3, [r, g, b]) => (*r, *g, *b, 255u8),
                (4, [r, g, b, a]) => (*r, *g, *b, *a),
                _ => unreachable!(),
            };
            let key = ((r as u32) << 24) | ((g as u32) << 16) | ((b as u32) << 8) | a as u32;
            let idx = *map.get(&key).unwrap();
            indices.push(idx);
        }
        return Ok((palette, indices));
    }

    let palette = median_cut_palette(colors.clone(), max_colors);

    // Map histogram colors to nearest palette entry.
    let mut color_to_idx = std::collections::HashMap::new();
    for c in colors {
        let idx = nearest_palette_index(c.rgba, &palette);
        let key = ((c.rgba[0] as u32) << 24)
            | ((c.rgba[1] as u32) << 16)
            | ((c.rgba[2] as u32) << 8)
            | c.rgba[3] as u32;
        color_to_idx.insert(key, idx);
    }

    if !dithering {
        let mut indices = Vec::with_capacity(data.len() / bpp);
        for chunk in data.chunks_exact(bpp) {
            let (r, g, b, a) = match (bpp, chunk) {
                (3, [r, g, b]) => (*r, *g, *b, 255u8),
                (4, [r, g, b, a]) => (*r, *g, *b, *a),
                _ => unreachable!(),
            };
            let key = ((r as u32) << 24) | ((g as u32) << 16) | ((b as u32) << 8) | a as u32;
            let idx = *color_to_idx.get(&key).unwrap_or(&0);
            indices.push(idx);
        }
        return Ok((palette, indices));
    }

    // Floyd–Steinberg dithering on RGB (alpha preserved).
    let width_usize = width as usize;
    let mut indices = Vec::with_capacity(data.len() / bpp);
    let mut err_r = vec![0f32; width_usize + 2];
    let mut err_g = vec![0f32; width_usize + 2];
    let mut err_b = vec![0f32; width_usize + 2];
    let mut next_err_r = vec![0f32; width_usize + 2];
    let mut next_err_g = vec![0f32; width_usize + 2];
    let mut next_err_b = vec![0f32; width_usize + 2];

    let mut pos = 0;
    for _y in 0..height as usize {
        for x in 0..width_usize {
            let (r, g, b, a) = if bpp == 3 {
                let r = data[pos];
                let g = data[pos + 1];
                let b = data[pos + 2];
                (r, g, b, 255u8)
            } else {
                let r = data[pos];
                let g = data[pos + 1];
                let b = data[pos + 2];
                let a = data[pos + 3];
                (r, g, b, a)
            };
            pos += bpp;

            let adj_r = (r as f32 + err_r[x + 1]).clamp(0.0, 255.0) as u8;
            let adj_g = (g as f32 + err_g[x + 1]).clamp(0.0, 255.0) as u8;
            let adj_b = (b as f32 + err_b[x + 1]).clamp(0.0, 255.0) as u8;

            let idx = nearest_palette_index([adj_r, adj_g, adj_b, a], &palette);
            indices.push(idx);
            let p = palette[idx as usize];
            let er = adj_r as f32 - p[0] as f32;
            let eg = adj_g as f32 - p[1] as f32;
            let eb = adj_b as f32 - p[2] as f32;

            // Distribute error
            //       * 7
            // 3 5 1
            err_r[x + 2] += er * 7.0 / 16.0;
            err_g[x + 2] += eg * 7.0 / 16.0;
            err_b[x + 2] += eb * 7.0 / 16.0;

            next_err_r[x] += er * 3.0 / 16.0;
            next_err_g[x] += eg * 3.0 / 16.0;
            next_err_b[x] += eb * 3.0 / 16.0;

            next_err_r[x + 1] += er * 5.0 / 16.0;
            next_err_g[x + 1] += eg * 5.0 / 16.0;
            next_err_b[x + 1] += eb * 5.0 / 16.0;

            next_err_r[x + 2] += er * 1.0 / 16.0;
            next_err_g[x + 2] += eg * 1.0 / 16.0;
            next_err_b[x + 2] += eb * 1.0 / 16.0;
        }
        err_r.fill(0.0);
        err_g.fill(0.0);
        err_b.fill(0.0);
        std::mem::swap(&mut err_r, &mut next_err_r);
        std::mem::swap(&mut err_g, &mut next_err_g);
        std::mem::swap(&mut err_b, &mut next_err_b);
    }

    Ok((palette, indices))
}

fn should_quantize_auto(data: &[u8], bpp: usize, max_colors: usize) -> bool {
    // Sample up to ~20k pixels to estimate color diversity.
    let total_pixels = data.len() / bpp;
    if total_pixels == 0 {
        return false;
    }
    let sample_cap = 20_000usize;
    let stride = (total_pixels / sample_cap).max(1);
    let mut set = std::collections::HashSet::with_capacity(sample_cap.min(total_pixels));
    let mut idx = 0usize;
    while idx < data.len() {
        let key = match bpp {
            3 => {
                let r = data[idx];
                let g = data[idx + 1];
                let b = data[idx + 2];
                ((r as u32) << 16) | ((g as u32) << 8) | b as u32
            }
            4 => {
                let r = data[idx];
                let g = data[idx + 1];
                let b = data[idx + 2];
                let a = data[idx + 3];
                ((r as u32) << 24) | ((g as u32) << 16) | ((b as u32) << 8) | a as u32
            }
            _ => return false,
        };
        set.insert(key);
        if set.len() > max_colors.saturating_mul(32) {
            // Too many distinct colors; likely a photo—skip auto quantization.
            return false;
        }
        idx = idx.saturating_add(stride * bpp);
    }

    // Quantize when observed diversity is within the palette-friendly bound.
    true
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

    #[test]
    fn test_encode_indexed_basic() {
        // 2x2 image with four palette entries
        let pixels = vec![0u8, 1, 2, 3];
        let palette = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [0, 0, 0]];
        let png = encode_indexed(&pixels, 2, 2, &palette, None).unwrap();

        // Signature + IHDR + PLTE + IDAT + IEND (minimal structure)
        assert_eq!(&png[0..8], &PNG_SIGNATURE);
        // IHDR color type byte at offset 25 should be 3 (indexed)
        assert_eq!(png[25], 3);
        // Ensure PLTE exists
        assert!(png.windows(4).any(|w| w == b"PLTE"), "PLTE chunk missing");
    }

    #[test]
    fn test_quantize_and_encode_indexed() {
        // Three distinct colors; force quantization to 2 palette entries.
        let pixels = vec![
            255, 0, 0, 255, // red
            0, 255, 0, 255, // green
            0, 0, 255, 255, // blue
        ];
        let mut opts = PngOptions::default();
        opts.quantization.mode = QuantizationMode::Force;
        opts.quantization.max_colors = 2;
        let png = encode_with_options(&pixels, 3, 1, ColorType::Rgba, &opts).unwrap();
        // Should be indexed
        assert_eq!(png[25], 3);
        // PLTE present
        assert!(png.windows(4).any(|w| w == b"PLTE"));
    }
}
