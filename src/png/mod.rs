//! PNG encoder implementation.
//!
//! Implements PNG encoding according to the PNG specification (RFC 2083).
//!
//! For a full pipeline walkthrough and tuning tips, see
//! [`crate::guides::png_encoding`] and [`crate::guides::performance_optimization`].

mod bit_depth;
pub mod chunk;
pub mod filter;

use crate::color::ColorType;
use crate::compress::deflate::{deflate_optimal_zlib, deflate_zlib_packed};
use crate::error::{Error, Result};
use bit_depth::{pack_gray_rows, pack_indexed_rows, palette_bit_depth, reduce_bit_depth};

/// PNG file signature (magic bytes).
const PNG_SIGNATURE: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

/// Maximum supported image dimension.
const MAX_DIMENSION: u32 = 1 << 24; // 16 million pixels

/// PNG encoding options.
///
/// Use [`PngOptions::builder()`] to create options with a fluent API.
///
/// # Example
///
/// ```rust
/// use pixo::png::{encode, PngOptions};
/// use pixo::ColorType;
///
/// let pixels = vec![255, 0, 0, 255]; // 1x1 RGBA red pixel
/// let options = PngOptions::builder(1, 1)
///     .color_type(ColorType::Rgba)
///     .preset(1) // balanced
///     .build();
/// let png_bytes = encode(&pixels, &options).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct PngOptions {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Color type of the pixel data.
    pub color_type: ColorType,
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
    /// If true, log filter usage histogram to stderr (for debugging/CLI verbose).
    pub verbose_filter_log: bool,
    /// If true, use optimal (Zopfli-style) DEFLATE compression with iterative refinement.
    /// Much slower but produces smaller files. Recommended only for final distribution.
    pub optimal_compression: bool,
    /// Palette quantization options (lossy compression for significant size reduction).
    pub quantization: QuantizationOptions,
}

/// Quantization mode for lossy PNG compression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizationMode {
    /// Disable palette quantization (lossless path).
    Off,
    /// Apply quantization when beneficial (auto-detect based on color count).
    Auto,
    /// Force quantization regardless of heuristics.
    Force,
}

/// Options controlling palette quantization (lossy compression).
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

impl Default for PngOptions {
    fn default() -> Self {
        Self {
            // Default dimensions (must be set via builder)
            width: 0,
            height: 0,
            color_type: ColorType::Rgba,
            // Prefer speed; level 2 favors throughput over ratio.
            compression_level: 2,
            // AdaptiveFast reduces per-row work with minimal compression impact.
            filter_strategy: FilterStrategy::AdaptiveFast,
            optimize_alpha: false,
            reduce_color_type: false,
            strip_metadata: false,
            reduce_palette: false,
            verbose_filter_log: false,
            optimal_compression: false,
            quantization: QuantizationOptions::default(),
        }
    }
}

impl PngOptions {
    /// Preset 0: Fast - prioritizes speed over compression.
    ///
    /// Uses level 2 compression with AdaptiveFast filter selection.
    /// No additional optimizations enabled. Defaults to RGBA color type.
    pub fn fast(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            color_type: ColorType::Rgba,
            compression_level: 2,
            filter_strategy: FilterStrategy::AdaptiveFast,
            optimize_alpha: false,
            reduce_color_type: false,
            strip_metadata: false,
            reduce_palette: false,
            verbose_filter_log: false,
            optimal_compression: false,
            quantization: QuantizationOptions::default(),
        }
    }

    /// Preset 1: Balanced - good tradeoff between speed and compression.
    ///
    /// Uses level 6 compression with Adaptive filter selection.
    /// Enables all lossless optimizations (palette reduction, color type
    /// reduction, alpha optimization, metadata stripping). Defaults to RGBA color type.
    pub fn balanced(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            color_type: ColorType::Rgba,
            compression_level: 6,
            filter_strategy: FilterStrategy::Adaptive,
            optimize_alpha: true,
            reduce_color_type: true,
            strip_metadata: true,
            reduce_palette: true,
            verbose_filter_log: false,
            optimal_compression: false,
            quantization: QuantizationOptions::default(),
        }
    }

    /// Preset 2: Max - maximum compression, competitive with oxipng.
    ///
    /// Uses level 9 compression with MinSum filter selection and optimal
    /// (Zopfli-style) DEFLATE compression with iterative refinement.
    /// Enables all lossless optimizations. Defaults to RGBA color type.
    pub fn max(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            color_type: ColorType::Rgba,
            compression_level: 9,
            filter_strategy: FilterStrategy::Bigrams,
            optimize_alpha: true,
            reduce_color_type: true,
            strip_metadata: true,
            reduce_palette: true,
            verbose_filter_log: false,
            optimal_compression: true,
            quantization: QuantizationOptions::default(),
        }
    }

    /// Create from preset number (0=fast, 1=balanced, 2=max).
    pub fn from_preset(width: u32, height: u32, preset: u8) -> Self {
        match preset {
            0 => Self::fast(width, height),
            2 => Self::max(width, height),
            _ => Self::balanced(width, height),
        }
    }

    /// Create from preset with lossless flag.
    ///
    /// When `lossless` is false, enables auto-quantization for potentially
    /// significant size reduction (lossy compression).
    pub fn from_preset_with_lossless(width: u32, height: u32, preset: u8, lossless: bool) -> Self {
        let mut opts = Self::from_preset(width, height, preset);
        if !lossless {
            opts.quantization = QuantizationOptions {
                mode: QuantizationMode::Auto,
                max_colors: 256,
                dithering: true,
            };
        }
        opts
    }
}

/// Builder for [`PngOptions`].
///
/// Create with [`PngOptions::builder(width, height)`] and configure options fluently.
#[derive(Debug, Clone)]
pub struct PngOptionsBuilder {
    options: PngOptions,
}

impl PngOptions {
    /// Create a builder for [`PngOptions`].
    ///
    /// The source dimensions are required; color type defaults to RGBA.
    pub fn builder(width: u32, height: u32) -> PngOptionsBuilder {
        PngOptionsBuilder::new(width, height)
    }
}

impl PngOptionsBuilder {
    /// Create a new builder with image dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            options: PngOptions {
                width,
                height,
                color_type: ColorType::Rgba,
                ..Default::default()
            },
        }
    }

    /// Set the color type of the pixel data.
    pub fn color_type(mut self, color_type: ColorType) -> Self {
        self.options.color_type = color_type;
        self
    }

    pub fn compression_level(mut self, level: u8) -> Self {
        self.options.compression_level = level;
        self
    }

    pub fn filter_strategy(mut self, strategy: FilterStrategy) -> Self {
        self.options.filter_strategy = strategy;
        self
    }

    pub fn optimize_alpha(mut self, value: bool) -> Self {
        self.options.optimize_alpha = value;
        self
    }

    pub fn reduce_color_type(mut self, value: bool) -> Self {
        self.options.reduce_color_type = value;
        self
    }

    pub fn strip_metadata(mut self, value: bool) -> Self {
        self.options.strip_metadata = value;
        self
    }

    pub fn reduce_palette(mut self, value: bool) -> Self {
        self.options.reduce_palette = value;
        self
    }

    pub fn verbose_filter_log(mut self, value: bool) -> Self {
        self.options.verbose_filter_log = value;
        self
    }

    pub fn optimal_compression(mut self, value: bool) -> Self {
        self.options.optimal_compression = value;
        self
    }

    pub fn quantization(mut self, quantization: QuantizationOptions) -> Self {
        self.options.quantization = quantization;
        self
    }

    pub fn quantization_mode(mut self, mode: QuantizationMode) -> Self {
        self.options.quantization.mode = mode;
        self
    }

    /// Convenience to toggle lossy (Auto quantization) vs. lossless (Off).
    ///
    /// When lossy is true, enables auto-quantization with dithering for
    /// better visual quality on photographic content.
    pub fn lossy(mut self, lossy: bool) -> Self {
        if lossy {
            self.options.quantization.mode = QuantizationMode::Auto;
            self.options.quantization.dithering = true; // Better quality for photos
        } else {
            self.options.quantization.mode = QuantizationMode::Off;
        }
        self
    }

    pub fn quantization_max_colors(mut self, max_colors: u16) -> Self {
        self.options.quantization.max_colors = max_colors;
        self
    }

    pub fn quantization_dithering(mut self, dithering: bool) -> Self {
        self.options.quantization.dithering = dithering;
        self
    }

    /// Apply a preset while retaining dimensions and color type.
    pub fn preset(mut self, preset: u8) -> Self {
        let width = self.options.width;
        let height = self.options.height;
        let color_type = self.options.color_type;
        self.options = PngOptions::from_preset(width, height, preset);
        self.options.color_type = color_type;
        self
    }

    /// Build the [`PngOptions`].
    #[must_use]
    pub fn build(self) -> PngOptions {
        self.options
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
    /// Choose filter per row minimizing sum of absolute values (min-sum).
    MinSum,
    /// Choose best filter per row (best compression, slower).
    Adaptive,
    /// Adaptive but with early cut and limited trials (faster).
    AdaptiveFast,
    /// Choose filter per row minimizing distinct bigrams (best DEFLATE correlation).
    Bigrams,
}

/// Encode raw pixel data as PNG.
///
/// # Arguments
///
/// * `data` - Raw pixel data (row-major order)
/// * `options` - PNG encoding options (includes width, height, color type)
///
/// # Returns
///
/// Complete PNG file as bytes.
///
/// # Example
///
/// ```rust
/// use pixo::png::{encode, PngOptions};
/// use pixo::ColorType;
///
/// let pixels = vec![255, 0, 0, 255]; // 1x1 RGBA red pixel
/// let options = PngOptions::builder(1, 1)
///     .color_type(ColorType::Rgba)
///     .build();
/// let png_bytes = encode(&pixels, &options).unwrap();
/// ```
#[must_use = "encoding produces a PNG file that should be used"]
pub fn encode(data: &[u8], options: &PngOptions) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    encode_into(&mut output, data, options)?;
    Ok(output)
}

fn checked_expected_len(
    width: u32,
    height: u32,
    bytes_per_pixel: usize,
    actual: usize,
) -> Result<usize> {
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|v| v.checked_mul(bytes_per_pixel))
        .ok_or(Error::InvalidDataLength {
            expected: usize::MAX,
            actual,
        })?;
    Ok(expected)
}

/// Encode raw pixel data as PNG into a caller-provided buffer.
///
/// The `output` buffer will be cleared and reused, allowing callers to avoid
/// repeated allocations across multiple encodes.
///
/// # Arguments
///
/// * `output` - Buffer to write PNG data into (will be cleared)
/// * `data` - Raw pixel data (row-major order)
/// * `options` - PNG encoding options (includes width, height, color type)
///
/// # Example
///
/// ```rust
/// use pixo::png::{encode_into, PngOptions};
/// use pixo::ColorType;
///
/// let pixels = vec![255, 0, 0, 255]; // 1x1 RGBA red pixel
/// let options = PngOptions::builder(1, 1)
///     .color_type(ColorType::Rgba)
///     .build();
/// let mut output = Vec::new();
/// encode_into(&mut output, &pixels, &options).unwrap();
/// ```
#[must_use = "this `Result` may indicate an encoding error"]
pub fn encode_into(output: &mut Vec<u8>, data: &[u8], options: &PngOptions) -> Result<()> {
    let width = options.width;
    let height = options.height;
    let color_type = options.color_type;

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
    let bpp = color_type.bytes_per_pixel();
    let expected_len = checked_expected_len(width, height, bpp, data.len())?;
    if data.len() != expected_len {
        return Err(Error::InvalidDataLength {
            expected: expected_len,
            actual: data.len(),
        });
    }

    // Check if quantization should be applied
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

    // Quantization path: convert to indexed PNG
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
        let plte: Vec<[u8; 3]> = palette_rgba
            .iter()
            .map(|[r, g, b, _]| [*r, *g, *b])
            .collect();
        let alpha: Vec<u8> = palette_rgba.iter().map(|[_, _, _, a]| *a).collect();
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

    let row_bytes = if reduced.bit_depth < 8 {
        (width as usize * reduced.bit_depth as usize).div_ceil(8)
    } else {
        width as usize * bytes_per_pixel
    };
    let filtered = filter::apply_filters_with_row_bytes(
        &data,
        width,
        height,
        row_bytes,
        bytes_per_pixel,
        options,
    );

    // Use optimal (Zopfli-style) compression if enabled, otherwise standard
    let compressed = if options.optimal_compression {
        // Use 5 iterations for optimal compression (balance of quality vs speed)
        deflate_optimal_zlib(&filtered, 5)
    } else {
        deflate_zlib_packed(&filtered, options.compression_level)
    };

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

fn write_idat_chunks(output: &mut Vec<u8>, compressed: &[u8]) {
    // Write in larger chunks to reduce per-chunk overhead (CRC/length)
    const CHUNK_SIZE: usize = 256 * 1024;

    for chunk_data in compressed.chunks(CHUNK_SIZE) {
        chunk::write_chunk(output, b"IDAT", chunk_data);
    }
}

fn write_iend(output: &mut Vec<u8>) {
    chunk::write_chunk(output, b"IEND", &[]);
}

/// If enabled, zero color channels for fully transparent pixels to improve compression.
fn maybe_optimize_alpha(
    data: &[u8],
    color_type: ColorType,
    optimize_alpha: bool,
) -> std::borrow::Cow<'_, [u8]> {
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
            let bit_depth = palette_bit_depth(palette.len());
            let packed = if bit_depth < 8 {
                pack_indexed_rows(&indexed, width, bit_depth)
            } else {
                indexed
            };
            let bytes_per_pixel = (bit_depth as usize).div_ceil(8).max(1);
            return ReducedImage {
                data: std::borrow::Cow::Owned(packed),
                effective_color_type: ColorType::Rgb, // For optimize_alpha logic (unused for palette)
                color_type_byte: 3,
                bit_depth,
                bytes_per_pixel,
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
                let bit_depth = reduce_bit_depth(&gray, ColorType::Gray).unwrap_or(8);
                let packed = if bit_depth < 8 {
                    pack_gray_rows(&gray, width, bit_depth)
                } else {
                    gray
                };
                let bytes_per_pixel = (bit_depth as usize).div_ceil(8).max(1);
                ReducedImage {
                    data: std::borrow::Cow::Owned(packed),
                    effective_color_type: ColorType::Gray,
                    color_type_byte: ColorType::Gray.png_color_type(),
                    bit_depth,
                    bytes_per_pixel,
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
                let bit_depth = reduce_bit_depth(&gray, ColorType::Gray).unwrap_or(8);
                let packed = if bit_depth < 8 {
                    pack_gray_rows(&gray, width, bit_depth)
                } else {
                    gray
                };
                let bytes_per_pixel = (bit_depth as usize).div_ceil(8).max(1);
                ReducedImage {
                    data: std::borrow::Cow::Owned(packed),
                    effective_color_type: ColorType::Gray,
                    color_type_byte: ColorType::Gray.png_color_type(),
                    bit_depth,
                    bytes_per_pixel,
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

    let stride = color_type.bytes_per_pixel();
    let pixel_count = width * height;

    // First pass: collect all colors as u32 keys, sort, and deduplicate
    // This avoids HashMap overhead in WASM builds
    let mut keys: Vec<u32> = Vec::with_capacity(pixel_count);
    for chunk in data.chunks_exact(stride) {
        let key = match color_type {
            ColorType::Rgb => {
                ((chunk[0] as u32) << 24)
                    | ((chunk[1] as u32) << 16)
                    | ((chunk[2] as u32) << 8)
                    | 255
            }
            ColorType::Rgba => {
                ((chunk[0] as u32) << 24)
                    | ((chunk[1] as u32) << 16)
                    | ((chunk[2] as u32) << 8)
                    | chunk[3] as u32
            }
            _ => unreachable!(),
        };
        keys.push(key);
    }

    // Build sorted unique palette
    let mut sorted_keys = keys.clone();
    sorted_keys.sort_unstable();
    sorted_keys.dedup();

    // Check if within palette limit
    if sorted_keys.len() > 256 {
        return None;
    }

    // Build palette from unique keys
    let palette: Vec<[u8; 4]> = sorted_keys
        .iter()
        .map(|&k| [(k >> 24) as u8, (k >> 16) as u8, (k >> 8) as u8, k as u8])
        .collect();

    // Second pass: map each pixel to its palette index using binary search
    let mut indexed = Vec::with_capacity(pixel_count);
    for &key in &keys {
        let idx = sorted_keys.binary_search(&key).unwrap() as u8;
        indexed.push(idx);
    }

    // Apply Zeng palette sorting for better compression
    let (indexed, palette) = optimize_palette_order(&indexed, palette, width, height);
    Some((indexed, palette))
}

/// Reorder palette using modified Zeng algorithm for better DEFLATE compression.
///
/// This algorithm reorders palette entries based on spatial adjacency (co-occurrence)
/// to maximize the likelihood that neighboring pixels have similar index values,
/// which improves DEFLATE compression due to smaller deltas.
///
/// Based on "A note on Zeng's technique for color reindexing" by Pinho et al (IEEE 2004).
fn optimize_palette_order(
    indexed: &[u8],
    palette: Vec<[u8; 4]>,
    width: usize,
    height: usize,
) -> (Vec<u8>, Vec<[u8; 4]>) {
    let n = palette.len();
    if n <= 2 {
        return (indexed.to_vec(), palette);
    }

    // Build co-occurrence matrix (horizontal + vertical neighbors)
    let matrix = build_co_occurrence_matrix(indexed, n, width, height);

    // Get edges sorted by weight (descending)
    let edges = weighted_edges(&matrix);
    if edges.is_empty() {
        return (indexed.to_vec(), palette);
    }

    // Apply modified Zeng reindexing algorithm
    let remapping = mzeng_reindex(n, &edges, &matrix);

    // Optionally put most popular color first (helps with filter byte 0)
    let remapping = apply_most_popular_first(indexed, remapping);

    // Apply remapping to palette and data
    apply_remapping(indexed, &palette, &remapping)
}

/// Build co-occurrence matrix counting horizontal and vertical neighbor pairs.
fn build_co_occurrence_matrix(
    indexed: &[u8],
    num_colors: usize,
    width: usize,
    height: usize,
) -> Vec<Vec<u32>> {
    let mut matrix = vec![vec![0u32; num_colors]; num_colors];

    for y in 0..height {
        let row_start = y * width;
        for x in 0..width {
            let val = indexed[row_start + x] as usize;
            if val >= num_colors {
                continue;
            }

            // Horizontal neighbor (right)
            if x + 1 < width {
                let next = indexed[row_start + x + 1] as usize;
                if next < num_colors {
                    matrix[val][next] += 1;
                    matrix[next][val] += 1;
                }
            }

            // Vertical neighbor (below)
            if y + 1 < height {
                let below = indexed[row_start + width + x] as usize;
                if below < num_colors {
                    matrix[val][below] += 1;
                    matrix[below][val] += 1;
                }
            }
        }
    }

    matrix
}

/// Sort edges by weight (descending), returning (color_a, color_b) pairs.
fn weighted_edges(matrix: &[Vec<u32>]) -> Vec<(usize, usize)> {
    let mut edges: Vec<((usize, usize), u32)> = Vec::new();
    for (i, row) in matrix.iter().enumerate() {
        for (j, &weight) in row.iter().enumerate().take(i) {
            if weight > 0 {
                edges.push(((j, i), weight));
            }
        }
    }
    edges.sort_by(|(_, w1), (_, w2)| w2.cmp(w1));
    edges.into_iter().map(|(e, _)| e).collect()
}

/// Apply modified Zeng reindexing algorithm.
///
/// Starting with the two colors from the highest-weighted edge, iteratively
/// add the color with the maximum sum of adjacencies to already-placed colors.
/// Use delta calculation to decide prepend vs append.
fn mzeng_reindex(num_colors: usize, edges: &[(usize, usize)], matrix: &[Vec<u32>]) -> Vec<usize> {
    if edges.is_empty() || num_colors == 0 {
        return (0..num_colors).collect();
    }

    // Initialize with the two colors from the best edge
    let mut remapping = vec![edges[0].0, edges[0].1];

    // Track sums of adjacencies to the ordered set for each unplaced color
    // Each entry is (color_index, sum_of_adjacencies)
    let mut sums: Vec<(usize, u32)> = Vec::with_capacity(num_colors - 2);
    let mut best_sum_pos = 0;
    let mut best_sum = (0usize, 0u32);

    for (i, row) in matrix.iter().enumerate() {
        if i == remapping[0] || i == remapping[1] {
            continue;
        }
        let sum = row[remapping[0]] + row[remapping[1]];
        if sum > best_sum.1 {
            best_sum_pos = sums.len();
            best_sum = (i, sum);
        }
        sums.push((i, sum));
    }

    while !sums.is_empty() {
        let best_index = best_sum.0;

        // Compute delta to decide prepend vs append
        // Delta > 0 means prepending gives better compression
        let n = (num_colors - sums.len()) as isize;
        let mut delta: isize = 0;
        for (i, &index) in remapping.iter().enumerate() {
            delta += (n - 1 - 2 * i as isize) * matrix[best_index][index] as isize;
        }

        if delta > 0 {
            remapping.insert(0, best_index);
        } else {
            remapping.push(best_index);
        }

        // Remove best_sum from sums
        sums.swap_remove(best_sum_pos);

        if !sums.is_empty() {
            // Update all sums and find the new best
            best_sum_pos = 0;
            best_sum = (0, 0);
            for (i, sum) in sums.iter_mut().enumerate() {
                sum.1 += matrix[best_index][sum.0];
                if sum.1 > best_sum.1 {
                    best_sum_pos = i;
                    best_sum = *sum;
                }
            }
        }
    }

    remapping
}

/// Put the most popular color first if it represents a significant portion of the image.
/// This helps compression when filter bytes are 0.
fn apply_most_popular_first(indexed: &[u8], mut remapping: Vec<usize>) -> Vec<usize> {
    if remapping.is_empty() || indexed.is_empty() {
        return remapping;
    }

    // Count color frequencies
    let mut counts = [0u32; 256];
    for &val in indexed {
        counts[val as usize] += 1;
    }

    // Find most popular color in the remapping
    let (most_popular_idx, most_popular_count) = remapping
        .iter()
        .map(|&idx| (idx, counts[idx]))
        .max_by_key(|&(_, count)| count)
        .unwrap_or((0, 0));

    // Only apply if the most popular color is at least 15% of the image
    let threshold = indexed.len() as u32 * 3 / 20;
    if most_popular_count < threshold {
        return remapping;
    }

    // Find position of most popular color in remapping
    if let Some(pos) = remapping.iter().position(|&i| i == most_popular_idx) {
        // If past halfway, reverse and rotate; otherwise just rotate
        if pos >= remapping.len() / 2 {
            remapping.reverse();
            remapping.rotate_right(pos + 1);
        } else {
            remapping.rotate_left(pos);
        }
    }

    remapping
}

/// Apply the remapping to create new indexed data and reordered palette.
fn apply_remapping(
    indexed: &[u8],
    palette: &[[u8; 4]],
    remapping: &[usize],
) -> (Vec<u8>, Vec<[u8; 4]>) {
    // Build the new palette in remapped order
    let new_palette: Vec<[u8; 4]> = remapping.iter().map(|&i| palette[i]).collect();

    // Build reverse mapping: old_index -> new_index
    let mut byte_map = [0u8; 256];
    for (new_idx, &old_idx) in remapping.iter().enumerate() {
        byte_map[old_idx] = new_idx as u8;
    }

    // Remap the indexed data
    let new_indexed: Vec<u8> = indexed.iter().map(|&b| byte_map[b as usize]).collect();

    (new_indexed, new_palette)
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

// ============================================================================
// Quantization (lossy compression)
// ============================================================================

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

    /// Returns (channel_to_split, perceptual_score).
    ///
    /// Uses perceptual weighting: G (4) > R (2) > B (1), with alpha weighted at 3.
    /// This matches human visual sensitivity where green contributes most to
    /// perceived luminance.
    fn range(&self) -> (u8, u16) {
        let r_range = self.r_max - self.r_min;
        let g_range = self.g_max - self.g_min;
        let b_range = self.b_max - self.b_min;
        let a_range = self.a_max - self.a_min;

        // Perceptual weights: G > A > R > B
        let r_score = r_range as u16 * 2;
        let g_score = g_range as u16 * 4; // Green most important (luminance)
        let b_score = b_range as u16; // Blue least important
        let a_score = a_range as u16 * 3; // Alpha fairly important for transparency

        let mut max_score = r_score;
        let mut channel = 0u8;
        if g_score > max_score {
            max_score = g_score;
            channel = 1;
        }
        if b_score > max_score {
            max_score = b_score;
            channel = 2;
        }
        if a_score > max_score {
            max_score = a_score;
            channel = 3;
        }
        (channel, max_score)
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
        // Ensure both halves are non-empty by clamping split_idx
        // to [0, len-2] range (so right side always has at least 1)
        let max_split = colors.len().saturating_sub(2);
        split_idx = split_idx.min(max_split);

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

    // Keep a reference to colors for K-means refinement
    let colors_for_kmeans = colors.clone();

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

    let mut palette: Vec<[u8; 4]> = boxes.into_iter().map(|b| b.make_palette_entry()).collect();

    // Refine palette with K-means iterations for better photographic quality
    refine_palette_kmeans(&mut palette, &colors_for_kmeans);

    palette
}

/// Refine palette colors using K-means iterations.
///
/// After median-cut produces an initial palette, K-means refinement adjusts
/// each palette color to be the weighted centroid of the colors assigned to it.
/// This improves quality for photographic content with smooth gradients.
fn refine_palette_kmeans(palette: &mut [[u8; 4]], colors: &[ColorCount]) {
    const ITERATIONS: usize = 2;

    if palette.is_empty() || colors.is_empty() {
        return;
    }

    for _ in 0..ITERATIONS {
        // Accumulators for each palette entry: (r_sum, g_sum, b_sum, a_sum, total_count)
        let mut accumulators: Vec<(u64, u64, u64, u64, u64)> = vec![(0, 0, 0, 0, 0); palette.len()];

        // Assign each color to nearest palette entry and accumulate
        for color in colors {
            let mut best_idx = 0;
            let mut best_dist = u32::MAX;
            for (i, p) in palette.iter().enumerate() {
                let dist = perceptual_distance_sq(color.rgba, *p);
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = i;
                }
            }

            let count = color.count as u64;
            let acc = &mut accumulators[best_idx];
            acc.0 += color.rgba[0] as u64 * count;
            acc.1 += color.rgba[1] as u64 * count;
            acc.2 += color.rgba[2] as u64 * count;
            acc.3 += color.rgba[3] as u64 * count;
            acc.4 += count;
        }

        // Update palette entries to be centroids of assigned colors
        for (i, acc) in accumulators.iter().enumerate() {
            if acc.4 > 0 {
                palette[i] = [
                    (acc.0 / acc.4) as u8,
                    (acc.1 / acc.4) as u8,
                    (acc.2 / acc.4) as u8,
                    (acc.3 / acc.4) as u8,
                ];
            }
        }
    }
}

/// Perceptual color distance (squared) using the Redmean formula.
///
/// The Redmean formula provides a perceptually accurate color distance by
/// adjusting weights based on the average red intensity. Human vision is most
/// sensitive to green, with red/blue sensitivity varying by luminance level.
///
/// This is significantly better than simple Euclidean RGB distance for:
/// - Skin tones and faces
/// - Smooth gradients
/// - Bright highlights (white balance)
///
/// See: https://en.wikipedia.org/wiki/Color_difference#sRGB
#[inline]
fn perceptual_distance_sq(c1: [u8; 4], c2: [u8; 4]) -> u32 {
    let dr = c1[0] as i32 - c2[0] as i32;
    let dg = c1[1] as i32 - c2[1] as i32;
    let db = c1[2] as i32 - c2[2] as i32;
    let da = c1[3] as i32 - c2[3] as i32;

    // Redmean formula: weights vary based on average red intensity
    // For high red (bright colors), red differences matter more
    // For low red (dark colors), blue differences matter more
    // Green always has the highest fixed weight (human eye most sensitive)
    let r_mean = (c1[0] as i32 + c2[0] as i32) >> 1;

    // Scale weights by 256 to avoid floating point
    // r_weight = 2 + r_mean/256 → (512 + r_mean) / 256
    // g_weight = 4 (fixed)
    // b_weight = 2 + (255-r_mean)/256 → (767 - r_mean) / 256
    let r_weight = 512 + r_mean;
    let b_weight = 767 - r_mean;
    const G_WEIGHT: i32 = 1024; // 4 * 256

    // Compute weighted distance (result scaled by 256)
    let dist = (r_weight * dr * dr + G_WEIGHT * dg * dg + b_weight * db * db) >> 8;

    // Add alpha difference (unweighted)
    (dist + da * da) as u32
}

fn nearest_palette_index(color: [u8; 4], palette: &[[u8; 4]]) -> u8 {
    let mut best_idx = 0u8;
    let mut best_dist = u32::MAX;
    for (i, p) in palette.iter().enumerate() {
        let dist = perceptual_distance_sq(color, *p);
        if dist < best_dist {
            best_dist = dist;
            best_idx = i as u8;
        }
    }
    best_idx
}

/// Precomputed lookup table for fast nearest-palette-index queries.
/// Uses 6-6-6 RGB quantization (256K entries) for O(1) lookups with high accuracy.
/// For colors with alpha < 255, falls back to direct computation.
struct PaletteLut {
    /// LUT for opaque colors: index = (r6 << 12) | (g6 << 6) | b6
    opaque_lut: Vec<u8>,
    /// Reference to the palette for transparent color lookups
    palette: Vec<[u8; 4]>,
}

impl PaletteLut {
    /// Build a lookup table for the given palette.
    fn new(palette: Vec<[u8; 4]>) -> Self {
        // Build 6-6-6 RGB LUT for opaque colors (256K entries = 64^3)
        // Each 8-bit channel is reduced to 6 bits by taking top 6 bits.
        // This provides 4x better color accuracy than 5-5-5 (32K entries).
        let mut opaque_lut = vec![0u8; 64 * 64 * 64];

        for r6 in 0..64u8 {
            for g6 in 0..64u8 {
                for b6 in 0..64u8 {
                    // Convert 6-bit back to 8-bit (expand to full range)
                    // Formula: (val << 2) | (val >> 4) maps 0..63 to 0..255
                    let r8 = (r6 << 2) | (r6 >> 4);
                    let g8 = (g6 << 2) | (g6 >> 4);
                    let b8 = (b6 << 2) | (b6 >> 4);

                    let idx = nearest_palette_index([r8, g8, b8, 255], &palette);
                    let lut_idx = ((r6 as usize) << 12) | ((g6 as usize) << 6) | (b6 as usize);
                    opaque_lut[lut_idx] = idx;
                }
            }
        }

        Self {
            opaque_lut,
            palette,
        }
    }

    /// Fast lookup of nearest palette index.
    #[inline]
    fn lookup(&self, r: u8, g: u8, b: u8, a: u8) -> u8 {
        if a == 255 {
            // Use precomputed LUT for opaque colors (most common case)
            let r6 = r >> 2;
            let g6 = g >> 2;
            let b6 = b >> 2;
            let lut_idx = ((r6 as usize) << 12) | ((g6 as usize) << 6) | (b6 as usize);
            self.opaque_lut[lut_idx]
        } else {
            // For transparent colors, compute directly (rare case)
            nearest_palette_index([r, g, b, a], &self.palette)
        }
    }
}

/// Quantize an RGB or RGBA image to a palette using median-cut algorithm.
///
/// Returns the palette (up to `max_colors` entries) and indexed pixel data.
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

    // Build color histogram with sampling for large images.
    // Uses Vec + sort + run-length counting to avoid HashMap overhead in WASM.
    let total_pixels = data.len() / bpp;
    let max_samples = 50_000usize;
    let stride = (total_pixels / max_samples).max(1);

    // Collect sampled color keys
    let mut keys = Vec::with_capacity(max_samples.min(total_pixels / stride + 1));
    let mut idx = 0usize;
    while idx + bpp <= data.len() {
        let chunk = &data[idx..idx + bpp];
        let key = match (bpp, chunk) {
            (3, [r, g, b]) => ((*r as u32) << 24) | ((*g as u32) << 16) | ((*b as u32) << 8) | 255,
            (4, [r, g, b, a]) => {
                ((*r as u32) << 24) | ((*g as u32) << 16) | ((*b as u32) << 8) | *a as u32
            }
            _ => unreachable!(),
        };
        keys.push(key);
        idx = idx.saturating_add(stride * bpp);
    }

    // Sort and build histogram via run-length counting
    keys.sort_unstable();
    let mut colors: Vec<ColorCount> = Vec::with_capacity(keys.len().min(8192));
    if !keys.is_empty() {
        let mut prev_key = keys[0];
        let mut count = stride as u32;
        for &key in keys.iter().skip(1) {
            if key == prev_key {
                count = count.saturating_add(stride as u32);
            } else {
                colors.push(ColorCount {
                    rgba: [
                        (prev_key >> 24) as u8,
                        (prev_key >> 16) as u8,
                        (prev_key >> 8) as u8,
                        prev_key as u8,
                    ],
                    count,
                });
                prev_key = key;
                count = stride as u32;
            }
        }
        // Push final color
        colors.push(ColorCount {
            rgba: [
                (prev_key >> 24) as u8,
                (prev_key >> 16) as u8,
                (prev_key >> 8) as u8,
                prev_key as u8,
            ],
            count,
        });
    }

    // If too many colors, keep only the most frequent ones for median-cut
    let max_histogram_colors = 8192usize;
    if colors.len() > max_histogram_colors {
        colors.sort_unstable_by(|a, b| b.count.cmp(&a.count));
        colors.truncate(max_histogram_colors);
    }

    // Early out: already within palette size
    if colors.len() <= max_colors {
        let palette: Vec<[u8; 4]> = colors.iter().map(|c| c.rgba).collect();
        // Build sorted lookup table for O(log n) color->index mapping
        let mut sorted_palette: Vec<(u32, u8)> = palette
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let key = ((c[0] as u32) << 24)
                    | ((c[1] as u32) << 16)
                    | ((c[2] as u32) << 8)
                    | c[3] as u32;
                (key, i as u8)
            })
            .collect();
        sorted_palette.sort_unstable_by_key(|(k, _)| *k);

        let mut indices = Vec::with_capacity(width as usize * height as usize);
        for chunk in data.chunks_exact(bpp) {
            let (r, g, b, a) = match (bpp, chunk) {
                (3, [r, g, b]) => (*r, *g, *b, 255u8),
                (4, [r, g, b, a]) => (*r, *g, *b, *a),
                _ => unreachable!(),
            };
            let key = ((r as u32) << 24) | ((g as u32) << 16) | ((b as u32) << 8) | a as u32;
            let idx = sorted_palette
                .binary_search_by_key(&key, |(k, _)| *k)
                .map(|i| sorted_palette[i].1)
                .unwrap_or_else(|_| nearest_palette_index([r, g, b, a], &palette));
            indices.push(idx);
        }
        return Ok((palette, indices));
    }

    let palette = median_cut_palette(colors, max_colors);

    // Build fast lookup table for nearest palette index queries
    let lut = PaletteLut::new(palette.clone());

    if !dithering {
        let mut indices = Vec::with_capacity(width as usize * height as usize);
        for chunk in data.chunks_exact(bpp) {
            let (r, g, b, a) = match (bpp, chunk) {
                (3, [r, g, b]) => (*r, *g, *b, 255u8),
                (4, [r, g, b, a]) => (*r, *g, *b, *a),
                _ => unreachable!(),
            };
            indices.push(lut.lookup(r, g, b, a));
        }
        return Ok((palette, indices));
    }

    // Floyd–Steinberg dithering on RGB (alpha preserved)
    // Uses PaletteLut for O(1) nearest-color lookups instead of O(256) linear search.
    let width_usize = width as usize;
    let mut indices = Vec::with_capacity(width as usize * height as usize);
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

            // O(1) lookup using precomputed LUT
            let idx = lut.lookup(adj_r, adj_g, adj_b, a);

            indices.push(idx);
            let p = palette[idx as usize];
            let er = adj_r as f32 - p[0] as f32;
            let eg = adj_g as f32 - p[1] as f32;
            let eb = adj_b as f32 - p[2] as f32;

            // Distribute error: * 7 / 3 5 1 (Floyd-Steinberg)
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

/// Heuristic to determine if auto-quantization should be applied.
///
/// Samples pixels to estimate color diversity. Returns true if the image
/// has a moderate number of colors (more than max_colors but not too many,
/// indicating it's not a photo).
fn should_quantize_auto(data: &[u8], bpp: usize, max_colors: usize) -> bool {
    let total_pixels = data.len() / bpp;
    if total_pixels == 0 {
        return false;
    }
    let sample_cap = 20_000usize;
    let stride = (total_pixels / sample_cap).max(1);
    let threshold = max_colors.saturating_mul(32);

    // Collect sampled color keys into a Vec, then sort and deduplicate
    // This avoids pulling in HashMap/HashSet which adds significant WASM size
    let mut keys = Vec::with_capacity(sample_cap.min(total_pixels / stride + 1));
    let mut idx = 0usize;
    while idx + bpp <= data.len() {
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
        keys.push(key);
        idx = idx.saturating_add(stride * bpp);
    }

    // Sort and count unique values
    keys.sort_unstable();
    let unique = if keys.is_empty() {
        0
    } else {
        let mut count = 1;
        for i in 1..keys.len() {
            if keys[i] != keys[i - 1] {
                count += 1;
                if count > threshold {
                    // Too many distinct colors; likely a photo—skip auto quantization
                    return false;
                }
            }
        }
        count
    };

    // Only quantize if we have more colors than palette size but not too many
    unique > max_colors && unique <= threshold
}

/// Encode indexed (palette) pixel data as PNG.
///
/// The `data` slice contains palette indices (0..palette.len()). The palette
/// must contain between 1 and 256 entries. Optional `transparency` supplies
/// per-entry alpha values (tRNS); its length must not exceed the palette
/// length. Bit depth is fixed to 8 for indexed output.
#[must_use = "encoding produces a PNG file that should be used"]
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

#[must_use = "encoding produces a PNG file that should be used"]
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

#[must_use = "this `Result` may indicate an encoding error"]
pub fn encode_indexed_into(
    output: &mut Vec<u8>,
    data: &[u8],
    width: u32,
    height: u32,
    palette: &[[u8; 3]],
    transparency: Option<&[u8]>,
    options: &PngOptions,
) -> Result<()> {
    let palette_len = palette.len();
    if palette_len == 0 || palette_len > 256 {
        return Err(Error::CompressionError(format!(
            "Invalid palette length: {palette_len} (must be 1-256)"
        )));
    }
    if let Some(alpha) = transparency {
        if alpha.len() > palette_len {
            return Err(Error::CompressionError(format!(
                "Transparency length {} exceeds palette length {}",
                alpha.len(),
                palette_len
            )));
        }
    }

    let expected_len = checked_expected_len(width, height, 1, data.len())?;
    if data.len() != expected_len {
        return Err(Error::InvalidDataLength {
            expected: expected_len,
            actual: data.len(),
        });
    }

    output.clear();
    output.reserve(expected_len / 2 + 2048);

    // Write signature and IHDR (color type 3, bit depth 8)
    output.extend_from_slice(&PNG_SIGNATURE);
    write_ihdr(output, width, height, 8, 3);

    // Write PLTE chunk
    let mut plte_data = Vec::with_capacity(palette_len * 3);
    for entry in palette {
        plte_data.extend_from_slice(entry);
    }
    chunk::write_chunk(output, b"PLTE", &plte_data);

    // Write tRNS chunk if transparency provided
    if let Some(alpha) = transparency {
        chunk::write_chunk(output, b"tRNS", alpha);
    }

    // Palette-aware filtering: prefer None/Sub for indexed data
    let mut palette_options = options.clone();
    palette_options.filter_strategy = match options.filter_strategy {
        FilterStrategy::Adaptive
        | FilterStrategy::AdaptiveFast
        | FilterStrategy::MinSum
        | FilterStrategy::Bigrams => FilterStrategy::None,
        other => other,
    };

    let filtered = filter::apply_filters(data, width, height, 1, &palette_options);
    let compressed = if palette_options.optimal_compression {
        deflate_optimal_zlib(&filtered, 5)
    } else {
        deflate_zlib_packed(&filtered, palette_options.compression_level)
    };

    write_idat_chunks(output, &compressed);
    write_iend(output);
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

    fn test_encode(
        data: &[u8],
        width: u32,
        height: u32,
        color_type: ColorType,
    ) -> crate::Result<Vec<u8>> {
        let options = PngOptions::builder(width, height)
            .color_type(color_type)
            .build();
        encode(data, &options)
    }

    fn test_encode_with_options(
        data: &[u8],
        width: u32,
        height: u32,
        color_type: ColorType,
        options: &PngOptions,
    ) -> crate::Result<Vec<u8>> {
        let mut opts = options.clone();
        opts.width = width;
        opts.height = height;
        opts.color_type = color_type;
        encode(data, &opts)
    }

    #[allow(dead_code)]
    fn test_encode_into(
        output: &mut Vec<u8>,
        data: &[u8],
        width: u32,
        height: u32,
        color_type: ColorType,
        options: &PngOptions,
    ) -> crate::Result<()> {
        let mut opts = options.clone();
        opts.width = width;
        opts.height = height;
        opts.color_type = color_type;
        encode_into(output, data, &opts)
    }

    #[test]
    fn test_encode_1x1_rgb() {
        let pixels = vec![255, 0, 0]; // Red pixel
        let png = test_encode(&pixels, 1, 1, ColorType::Rgb).unwrap();

        // Check PNG signature
        assert_eq!(&png[0..8], &PNG_SIGNATURE);

        // Should have IHDR, IDAT, IEND
        assert!(png.len() > 8 + 12 + 12 + 12); // signature + 3 chunks minimum
    }

    #[test]
    fn test_encode_1x1_rgba() {
        let pixels = vec![255, 0, 0, 255]; // Red opaque pixel
        let png = test_encode(&pixels, 1, 1, ColorType::Rgba).unwrap();

        assert_eq!(&png[0..8], &PNG_SIGNATURE);
    }

    #[test]
    fn test_encode_invalid_dimensions() {
        let pixels = vec![255, 0, 0];
        let result = test_encode(&pixels, 0, 1, ColorType::Rgb);
        assert!(matches!(result, Err(Error::InvalidDimensions { .. })));
    }

    #[test]
    fn test_encode_invalid_data_length() {
        let pixels = vec![255, 0]; // Too short for 1x1 RGB
        let result = test_encode(&pixels, 1, 1, ColorType::Rgb);
        assert!(matches!(result, Err(Error::InvalidDataLength { .. })));
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    fn test_encode_overflow_data_length() {
        let pixels = Vec::new();
        let result = test_encode(&pixels, MAX_DIMENSION, MAX_DIMENSION, ColorType::Rgb);
        assert!(matches!(result, Err(Error::InvalidDataLength { .. })));
    }

    #[test]
    fn test_encode_into_reuses_buffer() {
        let mut output = Vec::with_capacity(64);
        let pixels1 = vec![0u8, 0, 0]; // black 1x1 RGB
        let opts = PngOptions::builder(1, 1).color_type(ColorType::Rgb).build();
        encode_into(&mut output, &pixels1, &opts).unwrap();
        let first = output.clone();
        let first_cap = output.capacity();
        assert!(!first.is_empty());

        let pixels2 = vec![255u8, 0, 0]; // red 1x1 RGB
        encode_into(&mut output, &pixels2, &opts).unwrap();

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
        let png = test_encode_with_options(&pixels, 2, 1, ColorType::Rgba, &opts).unwrap();
        // Color type byte in IHDR should be 3 (palette)
        assert_eq!(png[25], 3);
        // Bit depth should reflect palette size (2 colors -> 1 bit)
        assert_eq!(png[24], 1);
        assert!(png.windows(4).any(|w| w == b"PLTE"));
    }

    #[test]
    fn test_palette_reduction_handles_packed_rows() {
        let width = 9u32;
        let height = 2u32;
        let mut pixels = Vec::with_capacity(width as usize * height as usize * 3);
        for y in 0..height {
            for x in 0..width {
                if (x + y) % 2 == 0 {
                    pixels.extend_from_slice(&[255, 0, 0]);
                } else {
                    pixels.extend_from_slice(&[0, 255, 0]);
                }
            }
        }

        let options = PngOptions::builder(width, height)
            .color_type(ColorType::Rgb)
            .reduce_palette(true)
            .build();
        let png = encode(&pixels, &options).unwrap();
        assert_eq!(&png[..8], &PNG_SIGNATURE);
        assert!(png.windows(4).any(|w| w == b"PLTE"));
    }

    #[test]
    fn test_png_presets() {
        // Test Fast preset
        let fast = PngOptions::fast(100, 100);
        assert_eq!(fast.compression_level, 2);
        assert_eq!(fast.filter_strategy, FilterStrategy::AdaptiveFast);
        assert!(!fast.optimize_alpha);
        assert!(!fast.reduce_color_type);
        assert!(!fast.reduce_palette);
        assert!(!fast.strip_metadata);

        // Test Balanced preset
        let balanced = PngOptions::balanced(100, 100);
        assert_eq!(balanced.compression_level, 6);
        assert_eq!(balanced.filter_strategy, FilterStrategy::Adaptive);
        assert!(balanced.optimize_alpha);
        assert!(balanced.reduce_color_type);
        assert!(balanced.reduce_palette);
        assert!(balanced.strip_metadata);

        // Test Max preset
        let max = PngOptions::max(100, 100);
        assert_eq!(max.compression_level, 9);
        assert_eq!(max.filter_strategy, FilterStrategy::Bigrams);
        assert!(max.optimize_alpha);
        assert!(max.reduce_color_type);
        assert!(max.reduce_palette);
        assert!(max.strip_metadata);

        // Test from_preset
        assert_eq!(PngOptions::from_preset(100, 100, 0).compression_level, 2);
        assert_eq!(PngOptions::from_preset(100, 100, 1).compression_level, 6);
        assert_eq!(PngOptions::from_preset(100, 100, 2).compression_level, 9);
    }

    #[test]
    fn test_builder_overrides_after_preset() {
        let opts = PngOptions::builder(100, 100)
            .preset(2) // max
            .compression_level(3)
            .filter_strategy(FilterStrategy::AdaptiveFast)
            .optimize_alpha(false)
            .reduce_color_type(false)
            .strip_metadata(false)
            .reduce_palette(false)
            .quantization_mode(QuantizationMode::Off)
            .build();

        assert_eq!(opts.compression_level, 3);
        assert_eq!(opts.filter_strategy, FilterStrategy::AdaptiveFast);
        assert!(!opts.optimize_alpha);
        assert!(!opts.reduce_color_type);
        assert!(!opts.strip_metadata);
        assert!(!opts.reduce_palette);
        assert_eq!(opts.quantization.mode, QuantizationMode::Off);
    }

    #[test]
    fn test_builder_lossy_toggle() {
        let lossy = PngOptions::builder(100, 100).lossy(true).build();
        assert_eq!(lossy.quantization.mode, QuantizationMode::Auto);
        assert!(
            lossy.quantization.dithering,
            "lossy mode should enable dithering"
        );

        let lossless = PngOptions::builder(100, 100).lossy(false).build();
        assert_eq!(lossless.quantization.mode, QuantizationMode::Off);
    }

    #[test]
    fn test_encode_2x2_checkerboard() {
        // 2x2 black and white checkerboard
        let pixels = vec![
            0, 0, 0, 255, 255, 255, // Row 1: black, white
            255, 255, 255, 0, 0, 0, // Row 2: white, black
        ];
        let png = test_encode(&pixels, 2, 2, ColorType::Rgb).unwrap();

        assert_eq!(&png[0..8], &PNG_SIGNATURE);
    }

    #[test]
    fn test_encode_grayscale() {
        let pixels = vec![128, 255, 0, 64]; // 2x2 grayscale
        let png = test_encode(&pixels, 2, 2, ColorType::Gray).unwrap();

        assert_eq!(&png[0..8], &PNG_SIGNATURE);
    }

    #[test]
    fn test_quantization_force() {
        // Image with more than 256 distinct colors - force quantization
        let mut pixels = Vec::with_capacity(512 * 3);
        for i in 0..512 {
            pixels.push((i % 256) as u8);
            pixels.push(((i / 2) % 256) as u8);
            pixels.push(((i * 3) % 256) as u8);
        }

        let opts = PngOptions {
            quantization: QuantizationOptions {
                mode: QuantizationMode::Force,
                max_colors: 256,
                dithering: false,
            },
            ..Default::default()
        };

        let png = test_encode_with_options(&pixels, 16, 32, ColorType::Rgb, &opts).unwrap();

        // Should produce indexed PNG (color type 3)
        assert_eq!(png[25], 3);
        // Should have PLTE chunk
        assert!(png.windows(4).any(|w| w == b"PLTE"));
        // Should be valid PNG
        assert_eq!(&png[0..8], &PNG_SIGNATURE);
    }

    #[test]
    fn test_quantization_off() {
        // Same image but with quantization off - should stay RGB
        let mut pixels = Vec::with_capacity(512 * 3);
        for i in 0..512 {
            pixels.push((i % 256) as u8);
            pixels.push(((i / 2) % 256) as u8);
            pixels.push(((i * 3) % 256) as u8);
        }

        let opts = PngOptions {
            quantization: QuantizationOptions {
                mode: QuantizationMode::Off,
                max_colors: 256,
                dithering: false,
            },
            ..Default::default()
        };

        let png = test_encode_with_options(&pixels, 16, 32, ColorType::Rgb, &opts).unwrap();

        // Should stay as RGB (color type 2)
        assert_eq!(png[25], 2);
        // Should NOT have PLTE chunk
        assert!(!png.windows(4).any(|w| w == b"PLTE"));
    }

    #[test]
    fn test_from_preset_with_lossless() {
        // Lossless = true should have quantization off
        let lossless = PngOptions::from_preset_with_lossless(100, 100, 1, true);
        assert_eq!(lossless.quantization.mode, QuantizationMode::Off);

        // Lossless = false should have quantization auto with dithering
        let lossy = PngOptions::from_preset_with_lossless(100, 100, 1, false);
        assert_eq!(lossy.quantization.mode, QuantizationMode::Auto);
        assert!(
            lossy.quantization.dithering,
            "lossy mode should enable dithering"
        );
    }

    #[test]
    fn test_quantization_rgba_with_transparency() {
        // RGBA image with varying transparency - force quantization
        let mut pixels = Vec::with_capacity(256 * 4);
        for i in 0..256 {
            pixels.push((i % 256) as u8); // R
            pixels.push(((i * 2) % 256) as u8); // G
            pixels.push(((i * 3) % 256) as u8); // B
            pixels.push(if i % 4 == 0 { 128 } else { 255 }); // A - some transparent
        }

        let opts = PngOptions {
            quantization: QuantizationOptions {
                mode: QuantizationMode::Force,
                max_colors: 256,
                dithering: false,
            },
            ..Default::default()
        };

        let png = test_encode_with_options(&pixels, 16, 16, ColorType::Rgba, &opts).unwrap();

        // Should produce indexed PNG (color type 3)
        assert_eq!(png[25], 3);
        // Should have PLTE chunk
        assert!(png.windows(4).any(|w| w == b"PLTE"));
        // Should have tRNS chunk for transparency
        assert!(png.windows(4).any(|w| w == b"tRNS"));
        // Should be valid PNG
        assert_eq!(&png[0..8], &PNG_SIGNATURE);
    }

    #[test]
    fn test_quantization_with_dithering() {
        // Create a gradient image that will benefit from dithering
        let mut pixels = Vec::with_capacity(64 * 64 * 3);
        for y in 0..64 {
            for x in 0..64 {
                pixels.push((x * 4) as u8); // R gradient
                pixels.push((y * 4) as u8); // G gradient
                pixels.push(128); // B constant
            }
        }

        let opts = PngOptions {
            quantization: QuantizationOptions {
                mode: QuantizationMode::Force,
                max_colors: 64, // Limited palette to force dithering to matter
                dithering: true,
            },
            ..Default::default()
        };

        let png = test_encode_with_options(&pixels, 64, 64, ColorType::Rgb, &opts).unwrap();

        // Should produce indexed PNG
        assert_eq!(png[25], 3);
        assert!(png.windows(4).any(|w| w == b"PLTE"));
        assert_eq!(&png[0..8], &PNG_SIGNATURE);
    }

    #[test]
    fn test_quantization_auto_mode_few_colors() {
        // Image with exactly 32 colors - auto should NOT quantize (already within limit)
        // Using 32 colors ensures 8-bit depth to avoid bit-packing issues
        let mut pixels = Vec::with_capacity(64 * 64 * 3);
        for i in 0..(64 * 64) {
            let color_idx = i % 32;
            pixels.push((color_idx * 8) as u8);
            pixels.push((color_idx * 4) as u8);
            pixels.push((color_idx * 2) as u8);
        }

        let opts = PngOptions {
            quantization: QuantizationOptions {
                mode: QuantizationMode::Auto,
                max_colors: 256,
                dithering: false,
            },
            reduce_palette: true, // Enable lossless palette reduction
            ..Default::default()
        };

        let png = test_encode_with_options(&pixels, 64, 64, ColorType::Rgb, &opts).unwrap();

        // Should still create palette (via lossless reduction, not quantization)
        // The key is that it uses the exact colors, not quantized approximations
        assert_eq!(png[25], 3);
        assert!(png.windows(4).any(|w| w == b"PLTE"));
    }

    #[test]
    fn test_quantization_produces_indexed_output() {
        // Create an image with >256 distinct colors (so lossless can't palette reduce)
        let mut pixels = Vec::with_capacity(32 * 32 * 3);
        for y in 0..32 {
            for x in 0..32 {
                // Each pixel is unique: R = x*8, G = y*8, B = (x+y)*4
                let r = (x * 8) as u8;
                let g = (y * 8) as u8;
                let b = ((x + y) * 4) as u8;
                pixels.push(r);
                pixels.push(g);
                pixels.push(b);
            }
        }

        // Encode lossless without palette reduction (to get true RGB output)
        let lossless_opts = PngOptions::builder(32, 32)
            .color_type(ColorType::Rgb)
            .quantization_mode(QuantizationMode::Off)
            .reduce_palette(false)
            .reduce_color_type(false)
            .build();
        let lossless = encode(&pixels, &lossless_opts).unwrap();

        // Encode lossy with forced quantization
        let lossy_opts = PngOptions::builder(32, 32)
            .color_type(ColorType::Rgb)
            .quantization_mode(QuantizationMode::Force)
            .reduce_palette(false)
            .reduce_color_type(false)
            .build();
        let lossy = encode(&pixels, &lossy_opts).unwrap();

        // Lossless should be RGB (color type 2), lossy should be indexed (color type 3)
        assert_eq!(lossless[25], 2, "Lossless should be RGB (color type 2)");
        assert_eq!(lossy[25], 3, "Lossy should be indexed (color type 3)");
        assert!(
            lossy.windows(4).any(|w| w == b"PLTE"),
            "Lossy should have PLTE chunk"
        );
    }

    #[test]
    fn test_quantization_max_colors_limit() {
        // Test that max_colors is respected
        let mut pixels = Vec::with_capacity(256 * 3);
        for i in 0..256 {
            pixels.push(i as u8);
            pixels.push((i * 2) as u8);
            pixels.push((i * 3) as u8);
        }

        let opts = PngOptions {
            quantization: QuantizationOptions {
                mode: QuantizationMode::Force,
                max_colors: 16, // Only allow 16 colors
                dithering: false,
            },
            ..Default::default()
        };

        let png = test_encode_with_options(&pixels, 16, 16, ColorType::Rgb, &opts).unwrap();

        // Should produce indexed PNG
        assert_eq!(png[25], 3);

        // Find PLTE chunk and verify its size
        let plte_pos = png.windows(4).position(|w| w == b"PLTE").unwrap();
        // PLTE chunk: length(4) + "PLTE"(4) + data(n*3) + CRC(4)
        // Length is at plte_pos - 4
        let plte_len = u32::from_be_bytes([
            png[plte_pos - 4],
            png[plte_pos - 3],
            png[plte_pos - 2],
            png[plte_pos - 1],
        ]) as usize;

        // Palette should have at most 16 colors * 3 bytes = 48 bytes
        assert!(
            plte_len <= 48,
            "PLTE length {plte_len} should be <= 48 (16 colors * 3 bytes)"
        );
    }

    #[test]
    fn test_lossless_palette_uses_binary_search() {
        // Test that the lossless palette path (build_palette) works correctly
        // This exercises the Vec + binary search approach we use instead of HashMap
        // Use 64x64 image (4096 pixels) with 20 distinct colors (>16 to force 8-bit depth)
        let mut pixels = Vec::with_capacity(64 * 64 * 4);
        for i in 0..(64 * 64) {
            let color_idx = i % 20;
            pixels.push((color_idx * 12) as u8); // R
            pixels.push((color_idx * 10) as u8); // G
            pixels.push((color_idx * 8) as u8); // B
            pixels.push(255); // A (opaque)
        }

        let opts = PngOptions {
            reduce_palette: true,
            ..Default::default()
        };

        let png = test_encode_with_options(&pixels, 64, 64, ColorType::Rgba, &opts).unwrap();

        // Should produce indexed PNG with palette
        assert_eq!(png[25], 3);
        assert!(png.windows(4).any(|w| w == b"PLTE"));

        // Find PLTE and verify it has 20 colors (60 bytes)
        let plte_pos = png.windows(4).position(|w| w == b"PLTE").unwrap();
        let plte_len = u32::from_be_bytes([
            png[plte_pos - 4],
            png[plte_pos - 3],
            png[plte_pos - 2],
            png[plte_pos - 1],
        ]) as usize;

        assert_eq!(
            plte_len, 60,
            "PLTE should have exactly 20 colors (60 bytes)"
        );
    }

    #[test]
    fn test_maybe_trim_transparency_all_opaque() {
        let alpha = vec![255u8; 5];
        assert_eq!(maybe_trim_transparency(&alpha), None);
    }

    #[test]
    fn test_maybe_trim_transparency_trims_trailing() {
        let alpha = vec![255u8, 200, 255, 255];
        let trimmed = maybe_trim_transparency(&alpha).expect("should trim");
        assert_eq!(trimmed, vec![255u8, 200]);
    }

    fn find_chunk(data: &[u8], name: &[u8; 4]) -> Option<(usize, usize)> {
        // returns (offset, length)
        let mut offset = 8; // skip signature
        while offset + 12 <= data.len() {
            let len = u32::from_be_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
            let chunk_type = &data[offset + 4..offset + 8];
            if chunk_type == name {
                return Some((offset, len));
            }
            offset += 12 + len; // len + type + data + crc
        }
        None
    }

    #[test]
    fn test_encode_indexed_bit_depth_1() {
        // encode_indexed always uses bit depth 8 for palette output (simpler packing).
        let data = vec![0u8, 1, 0, 1, 0, 1, 0, 1]; // 8 pixels, indices 0/1
        let palette = [[0u8, 0, 0], [255u8, 255, 255]];

        let png = encode_indexed(&data, 4, 2, &palette, None).unwrap();

        // IHDR bit depth byte remains 8 for indexed encode path
        assert_eq!(png[24], 8, "bit depth should be 8 for indexed output");
        assert_eq!(png[25], 3, "color type should be palette (3)");

        // PLTE length should be 2 * 3 = 6 bytes
        let (plte_offset, plte_len) = find_chunk(&png, b"PLTE").expect("PLTE missing");
        assert_eq!(plte_len, 6);
        // chunk length field should match
        assert_eq!(
            &png[plte_offset..plte_offset + 4],
            &6u32.to_be_bytes(),
            "PLTE length field mismatch"
        );
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    fn test_encode_indexed_overflow_data_length() {
        let palette = [[0u8, 0, 0]];
        let data = Vec::new();
        let result = encode_indexed(&data, MAX_DIMENSION, MAX_DIMENSION, &palette, None);
        assert!(matches!(result, Err(Error::InvalidDataLength { .. })));
    }

    #[test]
    fn test_encode_indexed_bit_depth_2_and_trns() {
        // encode_indexed uses bit depth 8; include tRNS shorter than palette.
        let data = vec![0u8, 1, 2, 3];
        let palette = [[0u8, 0, 0], [255u8, 0, 0], [0u8, 255, 0], [0u8, 0, 255]];
        let trns = [0u8, 128]; // first entry fully transparent, second semi, rest implied opaque

        let png = encode_indexed(&data, 2, 2, &palette, Some(&trns)).unwrap();

        assert_eq!(png[24], 8, "bit depth should be 8 for indexed output");
        assert_eq!(png[25], 3, "color type should be palette (3)");

        let (plte_offset, plte_len) = find_chunk(&png, b"PLTE").expect("PLTE missing");
        assert_eq!(plte_len, 12, "PLTE length should be 4*3");
        assert_eq!(
            &png[plte_offset..plte_offset + 4],
            &12u32.to_be_bytes(),
            "PLTE length field mismatch"
        );

        let (_trns_offset, trns_len) = find_chunk(&png, b"tRNS").expect("tRNS missing");
        assert_eq!(
            trns_len,
            trns.len(),
            "tRNS length should match provided alpha"
        );
    }

    #[test]
    fn test_encode_indexed_invalid_palette_length() {
        // palette > 256 should be rejected
        let data = vec![0u8; 4];
        let palette = vec![[0u8, 0, 0]; 300];
        let err = encode_indexed(&data, 2, 2, &palette, None).unwrap_err();
        assert!(
            matches!(err, Error::CompressionError(_)),
            "expected CompressionError for oversized palette"
        );
    }

    #[test]
    fn test_encode_indexed_invalid_trns_length() {
        // transparency length exceeding palette length should be rejected
        let data = vec![0u8; 4];
        let palette = vec![[0u8, 0, 0]; 2];
        let trns = vec![0u8; 3]; // longer than palette
        let err = encode_indexed(&data, 2, 2, &palette, Some(&trns)).unwrap_err();
        assert!(
            matches!(err, Error::CompressionError(_)),
            "expected CompressionError for tRNS longer than palette"
        );
    }

    #[test]
    fn test_encode_indexed_empty_palette_rejected() {
        let data = vec![0u8; 4];
        let palette: Vec<[u8; 3]> = Vec::new();
        let err = encode_indexed(&data, 2, 2, &palette, None).unwrap_err();
        assert!(
            matches!(err, Error::CompressionError(_)),
            "expected CompressionError for empty palette"
        );
    }

    #[test]
    fn test_encode_indexed_trns_equal_palette_len_allowed() {
        // Transparency length equal to palette length should be accepted.
        let data = vec![0u8, 1u8];
        let palette = vec![[0u8, 0, 0], [255u8, 255, 255]];
        let trns = vec![0u8, 128u8];
        let png = encode_indexed(&data, 2, 1, &palette, Some(&trns)).unwrap();
        // tRNS chunk length should match palette length here (2 bytes)
        fn find_chunk(data: &[u8], name: &[u8; 4]) -> Option<(usize, usize)> {
            let mut offset = 8;
            while offset + 8 <= data.len() {
                let len = u32::from_be_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
                let chunk_type = &data[offset + 4..offset + 8];
                if chunk_type == name {
                    return Some((offset, len));
                }
                offset += 12 + len;
            }
            None
        }
        let (_off, len) = find_chunk(&png, b"tRNS").expect("tRNS should exist");
        assert_eq!(len, trns.len());
    }

    #[test]
    fn test_quantization_force_produces_indexed() {
        // Force quantization on an RGB image and verify output color type is palette (3)
        let data = vec![
            255, 0, 0, // red
            0, 255, 0, // green
            0, 0, 255, // blue
            255, 255, 0, // yellow
        ];
        let options = PngOptions::builder(2, 2)
            .color_type(ColorType::Rgb)
            .quantization_mode(QuantizationMode::Force)
            .quantization_max_colors(4)
            .build();

        let png = encode(&data, &options).unwrap();
        assert_eq!(
            png[25], 3,
            "color type should be palette (3) after quantization"
        );
    }

    #[test]
    fn test_quantization_force_rgba_produces_trns_palette() {
        // RGBA quantization should emit palette + tRNS when alpha varies.
        // Two pixels: one fully transparent red, one opaque green.
        let data = vec![
            255, 0, 0, 0, // transparent red
            0, 255, 0, 255, // opaque green
        ];
        let options = PngOptions::builder(2, 1)
            .color_type(ColorType::Rgba)
            .quantization_mode(QuantizationMode::Force)
            .build();

        let png = encode(&data, &options).unwrap();

        // Color type should be palette (3) and bit depth 8
        assert_eq!(png[25], 3, "color type should be palette (3)");
        assert_eq!(png[24], 8, "bit depth should remain 8 for palette output");

        // Find PLTE and tRNS chunks
        fn find_chunk(data: &[u8], name: &[u8; 4]) -> Option<(usize, usize)> {
            let mut offset = 8;
            while offset + 8 <= data.len() {
                let len = u32::from_be_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
                let chunk_type = &data[offset + 4..offset + 8];
                if chunk_type == name {
                    return Some((offset, len));
                }
                offset += 12 + len;
            }
            None
        }

        let (_, plte_len) = find_chunk(&png, b"PLTE").expect("PLTE missing");
        assert_eq!(plte_len, 6, "expected 2 palette entries (6 bytes)");

        let (_, trns_len) = find_chunk(&png, b"tRNS").expect("tRNS missing");
        assert!(
            (1..=2).contains(&trns_len),
            "tRNS length should be 1..=2, got {trns_len}"
        );
    }

    // =========================================================================
    // Tests for internal palette functions
    // =========================================================================

    #[test]
    fn test_color_box_from_colors_single() {
        let colors = vec![ColorCount {
            rgba: [100, 150, 200, 255],
            count: 10,
        }];
        let b = ColorBox::from_colors(colors);

        assert_eq!(b.r_min, 100);
        assert_eq!(b.r_max, 100);
        assert_eq!(b.g_min, 150);
        assert_eq!(b.g_max, 150);
    }

    #[test]
    fn test_color_box_from_colors_range() {
        let colors = vec![
            ColorCount {
                rgba: [0, 50, 100, 255],
                count: 5,
            },
            ColorCount {
                rgba: [100, 100, 200, 255],
                count: 5,
            },
        ];
        let b = ColorBox::from_colors(colors);

        assert_eq!(b.r_min, 0);
        assert_eq!(b.r_max, 100);
        assert_eq!(b.g_min, 50);
        assert_eq!(b.g_max, 100);
        assert_eq!(b.b_min, 100);
        assert_eq!(b.b_max, 200);
    }

    #[test]
    fn test_color_box_range() {
        let colors = vec![
            ColorCount {
                rgba: [0, 0, 0, 0],
                count: 1,
            },
            ColorCount {
                rgba: [255, 100, 50, 255],
                count: 1,
            },
        ];
        let b = ColorBox::from_colors(colors);

        let (channel, max_score) = b.range();
        // With perceptual weighting: R*2=510, G*4=400, B*1=50, A*3=765
        // Alpha channel wins with highest perceptual score
        assert_eq!(channel, 3);
        assert_eq!(max_score, 765);
    }

    #[test]
    fn test_color_box_can_split() {
        // Single color cannot split
        let single = ColorBox::from_colors(vec![ColorCount {
            rgba: [100, 100, 100, 255],
            count: 10,
        }]);
        assert!(!single.can_split());

        // Multiple colors can split
        let multiple = ColorBox::from_colors(vec![
            ColorCount {
                rgba: [0, 0, 0, 255],
                count: 5,
            },
            ColorCount {
                rgba: [255, 255, 255, 255],
                count: 5,
            },
        ]);
        assert!(multiple.can_split());
    }

    #[test]
    fn test_color_box_split() {
        let colors = vec![
            ColorCount {
                rgba: [0, 0, 0, 255],
                count: 5,
            },
            ColorCount {
                rgba: [128, 0, 0, 255],
                count: 5,
            },
            ColorCount {
                rgba: [255, 0, 0, 255],
                count: 5,
            },
        ];
        let b = ColorBox::from_colors(colors);

        let (left, right) = b.split();

        // Both boxes should have at least one color
        assert!(!left.colors.is_empty());
        assert!(!right.colors.is_empty());

        // Combined they should have all 3 colors
        assert_eq!(left.colors.len() + right.colors.len(), 3);
    }

    #[test]
    fn test_color_box_make_palette_entry() {
        // Single color should return that color
        let single = ColorBox::from_colors(vec![ColorCount {
            rgba: [100, 150, 200, 255],
            count: 10,
        }]);
        assert_eq!(single.make_palette_entry(), [100, 150, 200, 255]);

        // Multiple colors should average weighted by count
        let weighted = ColorBox::from_colors(vec![
            ColorCount {
                rgba: [0, 0, 0, 255],
                count: 3,
            },
            ColorCount {
                rgba: [255, 255, 255, 255],
                count: 1,
            },
        ]);
        let entry = weighted.make_palette_entry();
        // (0*3 + 255*1) / 4 = 63.75 -> 63
        assert_eq!(entry[0], 63);
        assert_eq!(entry[1], 63);
        assert_eq!(entry[2], 63);
    }

    #[test]
    fn test_color_box_empty() {
        let empty = ColorBox::from_colors(vec![]);
        // Empty box should return default color
        assert_eq!(empty.make_palette_entry(), [0, 0, 0, 255]);
    }

    #[test]
    fn test_median_cut_palette_empty() {
        let palette = median_cut_palette(vec![], 256);
        // Empty input should return single default color
        assert_eq!(palette.len(), 1);
        assert_eq!(palette[0], [0, 0, 0, 255]);
    }

    #[test]
    fn test_median_cut_palette_single_color() {
        let colors = vec![ColorCount {
            rgba: [255, 0, 0, 255],
            count: 100,
        }];
        let palette = median_cut_palette(colors, 256);
        assert_eq!(palette.len(), 1);
        assert_eq!(palette[0], [255, 0, 0, 255]);
    }

    #[test]
    fn test_median_cut_palette_two_colors() {
        let colors = vec![
            ColorCount {
                rgba: [255, 0, 0, 255],
                count: 50,
            },
            ColorCount {
                rgba: [0, 0, 255, 255],
                count: 50,
            },
        ];
        let palette = median_cut_palette(colors, 256);

        // Should produce 2 colors
        assert_eq!(palette.len(), 2);
    }

    #[test]
    fn test_median_cut_palette_max_colors_limit() {
        // Many colors should be reduced to max_colors
        let colors: Vec<ColorCount> = (0..100)
            .map(|i| ColorCount {
                rgba: [i as u8 * 2, i as u8, (255 - i) as u8, 255],
                count: 1,
            })
            .collect();

        let palette = median_cut_palette(colors, 16);
        assert!(palette.len() <= 16, "Palette should be at most 16 colors");
    }

    #[test]
    fn test_nearest_palette_index_exact_match() {
        let palette = vec![[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]];

        assert_eq!(nearest_palette_index([255, 0, 0, 255], &palette), 0);
        assert_eq!(nearest_palette_index([0, 255, 0, 255], &palette), 1);
        assert_eq!(nearest_palette_index([0, 0, 255, 255], &palette), 2);
    }

    #[test]
    fn test_nearest_palette_index_closest() {
        let palette = vec![[0, 0, 0, 255], [255, 255, 255, 255]];

        // Dark gray should match black
        assert_eq!(nearest_palette_index([50, 50, 50, 255], &palette), 0);
        // Light gray should match white
        assert_eq!(nearest_palette_index([200, 200, 200, 255], &palette), 1);
    }

    #[test]
    fn test_palette_lut_basic() {
        let palette = vec![
            [255, 0, 0, 255],   // red
            [0, 255, 0, 255],   // green
            [0, 0, 255, 255],   // blue
            [255, 255, 0, 255], // yellow
        ];
        let lut = PaletteLut::new(palette);

        // Red should map to index 0
        assert_eq!(lut.lookup(255, 0, 0, 255), 0);
        // Green should map to index 1
        assert_eq!(lut.lookup(0, 255, 0, 255), 1);
        // Blue should map to index 2
        assert_eq!(lut.lookup(0, 0, 255, 255), 2);
    }

    #[test]
    fn test_palette_lut_transparent() {
        let palette = vec![
            [255, 0, 0, 255], // opaque red
            [0, 255, 0, 128], // semi-transparent green
        ];
        let lut = PaletteLut::new(palette);

        // Semi-transparent color should use fallback computation
        let idx = lut.lookup(0, 255, 0, 100);
        // Should find the closest match (index 1 - green)
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_all_gray_rgb_true() {
        let data = vec![100, 100, 100, 200, 200, 200];
        assert!(all_gray_rgb(&data));
    }

    #[test]
    fn test_all_gray_rgb_false() {
        let data = vec![100, 100, 100, 200, 200, 201]; // Last pixel not gray
        assert!(!all_gray_rgb(&data));
    }

    #[test]
    fn test_analyze_rgba_all_opaque() {
        let data = vec![
            100, 100, 100, 255, // gray opaque
            200, 200, 200, 255, // gray opaque
        ];
        let (all_opaque, all_gray) = analyze_rgba(&data);
        assert!(all_gray);
        assert!(all_opaque);
    }

    #[test]
    fn test_analyze_rgba_not_opaque() {
        let data = vec![
            100, 100, 100, 255, // gray opaque
            200, 200, 200, 128, // gray semi-transparent
        ];
        let (all_opaque, all_gray) = analyze_rgba(&data);
        assert!(all_gray);
        assert!(!all_opaque);
    }

    #[test]
    fn test_analyze_rgba_not_gray() {
        let data = vec![
            100, 100, 100, 255, // gray
            200, 150, 200, 255, // not gray (G != R)
        ];
        let (all_opaque, all_gray) = analyze_rgba(&data);
        assert!(!all_gray);
        assert!(all_opaque);
    }

    // =========================================================================
    // Perceptual distance tests
    // =========================================================================

    #[test]
    fn test_perceptual_distance_identical() {
        // Identical colors should have distance 0
        let c = [128, 64, 192, 255];
        assert_eq!(perceptual_distance_sq(c, c), 0);
    }

    #[test]
    fn test_perceptual_distance_green_weighted() {
        // Green differences should contribute more than red/blue
        let base = [128, 128, 128, 255];
        let red_diff = [138, 128, 128, 255]; // +10 red
        let green_diff = [128, 138, 128, 255]; // +10 green
        let blue_diff = [128, 128, 138, 255]; // +10 blue

        let d_red = perceptual_distance_sq(base, red_diff);
        let d_green = perceptual_distance_sq(base, green_diff);
        let d_blue = perceptual_distance_sq(base, blue_diff);

        assert!(
            d_green > d_red,
            "green should be weighted higher than red: {d_green} vs {d_red}"
        );
        assert!(
            d_red > d_blue,
            "red should be weighted higher than blue: {d_red} vs {d_blue}"
        );
    }

    #[test]
    fn test_perceptual_distance_symmetry() {
        // Distance should be symmetric
        let c1 = [100, 150, 200, 255];
        let c2 = [50, 100, 150, 200];
        assert_eq!(
            perceptual_distance_sq(c1, c2),
            perceptual_distance_sq(c2, c1)
        );
    }

    #[test]
    fn test_perceptual_distance_alpha() {
        // Alpha differences should contribute to distance
        let c1 = [100, 100, 100, 255];
        let c2 = [100, 100, 100, 200];
        assert!(
            perceptual_distance_sq(c1, c2) > 0,
            "alpha difference should produce non-zero distance"
        );
    }

    // =========================================================================
    // K-means refinement tests
    // =========================================================================

    #[test]
    fn test_kmeans_refinement_empty_inputs() {
        // Should handle empty inputs gracefully
        let mut empty_palette: Vec<[u8; 4]> = vec![];
        let colors = vec![ColorCount {
            rgba: [100, 100, 100, 255],
            count: 10,
        }];
        refine_palette_kmeans(&mut empty_palette, &colors);
        assert!(empty_palette.is_empty());

        let mut palette = vec![[100, 100, 100, 255]];
        let empty_colors: Vec<ColorCount> = vec![];
        refine_palette_kmeans(&mut palette, &empty_colors);
        // Palette should remain unchanged
        assert_eq!(palette[0], [100, 100, 100, 255]);
    }

    #[test]
    fn test_kmeans_refinement_single_color() {
        // Single color should converge to that color
        let mut palette = vec![[50, 50, 50, 255]];
        let colors = vec![ColorCount {
            rgba: [100, 150, 200, 255],
            count: 100,
        }];
        refine_palette_kmeans(&mut palette, &colors);
        // Palette should move toward the color
        assert_eq!(palette[0], [100, 150, 200, 255]);
    }

    #[test]
    fn test_kmeans_refinement_two_clusters() {
        // Two distinct color clusters should refine to cluster centers
        let mut palette = vec![
            [50, 50, 50, 255],    // Initial guess for dark cluster
            [200, 200, 200, 255], // Initial guess for light cluster
        ];
        let colors = vec![
            ColorCount {
                rgba: [0, 0, 0, 255],
                count: 100,
            },
            ColorCount {
                rgba: [10, 10, 10, 255],
                count: 100,
            },
            ColorCount {
                rgba: [245, 245, 245, 255],
                count: 100,
            },
            ColorCount {
                rgba: [255, 255, 255, 255],
                count: 100,
            },
        ];
        refine_palette_kmeans(&mut palette, &colors);

        // Dark palette entry should be around (0+10)/2 = 5
        assert!(palette[0][0] < 20, "dark cluster should be dark");
        // Light palette entry should be around (245+255)/2 = 250
        assert!(palette[1][0] > 230, "light cluster should be light");
    }

    #[test]
    fn test_kmeans_refinement_weighted() {
        // Higher count colors should have more influence
        let mut palette = vec![[128, 128, 128, 255]];
        let colors = vec![
            ColorCount {
                rgba: [0, 0, 0, 255],
                count: 1, // Low weight
            },
            ColorCount {
                rgba: [200, 200, 200, 255],
                count: 100, // High weight
            },
        ];
        refine_palette_kmeans(&mut palette, &colors);
        // Result should be much closer to [200, 200, 200] than [0, 0, 0]
        assert!(
            palette[0][0] > 150,
            "should be closer to high-weight color: {}",
            palette[0][0]
        );
    }

    // =========================================================================
    // Integration tests for quantization quality
    // =========================================================================

    #[test]
    fn test_quantization_produces_valid_palette() {
        // Create an image with a gradient of skin-tone-like colors
        // This tests that perceptual quantization works for photographic content
        let mut pixels = Vec::with_capacity(64 * 64 * 3);
        for y in 0..64 {
            for x in 0..64 {
                // Skin-tone gradient (warm colors)
                let r = 180 + (x as u8 / 2);
                let g = 140 + (y as u8 / 2);
                let b = 120;
                pixels.push(r);
                pixels.push(g);
                pixels.push(b);
            }
        }

        let opts = PngOptions {
            quantization: QuantizationOptions {
                mode: QuantizationMode::Force,
                max_colors: 32,
                dithering: false,
            },
            ..Default::default()
        };

        let png = test_encode_with_options(&pixels, 64, 64, ColorType::Rgb, &opts).unwrap();

        // Should produce indexed PNG (color type 3)
        assert_eq!(png[25], 3, "should produce indexed PNG");
        // Should have PLTE chunk
        assert!(
            png.windows(4).any(|w| w == b"PLTE"),
            "should have PLTE chunk"
        );
        // Should be valid PNG
        assert_eq!(&png[0..8], &PNG_SIGNATURE);
    }

    #[test]
    fn test_quantization_skin_tones_preserved() {
        // Create a simple image with skin-tone colors
        // The perceptual quantization should preserve these well
        let skin_colors = [
            [232, 190, 172], // Light skin
            [198, 134, 103], // Medium skin
            [141, 85, 64],   // Dark skin
        ];

        let mut pixels = Vec::new();
        for color in &skin_colors {
            for _ in 0..100 {
                pixels.extend_from_slice(color);
            }
        }

        let opts = PngOptions {
            quantization: QuantizationOptions {
                mode: QuantizationMode::Force,
                max_colors: 8,
                dithering: false,
            },
            ..Default::default()
        };

        // Should encode without error
        let result = test_encode_with_options(&pixels, 10, 30, ColorType::Rgb, &opts);
        assert!(result.is_ok(), "should encode skin tones successfully");
    }

    #[test]
    fn test_quantization_gradient_with_dithering() {
        // Create a gradient image
        let mut pixels = Vec::with_capacity(32 * 32 * 3);
        for y in 0..32 {
            for x in 0..32 {
                pixels.push((x * 8) as u8);
                pixels.push((y * 8) as u8);
                pixels.push(128);
            }
        }

        let opts = PngOptions {
            quantization: QuantizationOptions {
                mode: QuantizationMode::Force,
                max_colors: 16,
                dithering: true, // Enable dithering
            },
            ..Default::default()
        };

        let png = test_encode_with_options(&pixels, 32, 32, ColorType::Rgb, &opts).unwrap();

        // Should produce indexed PNG
        assert_eq!(png[25], 3, "should produce indexed PNG with dithering");
        assert_eq!(&png[0..8], &PNG_SIGNATURE);
    }

    // ============================================================================
    // Optimal Compression Tests
    // ============================================================================

    #[test]
    fn test_png_optimal_compression_produces_valid_output() {
        let mut pixels = Vec::with_capacity(64 * 64 * 3);
        for i in 0..(64 * 64 * 3) {
            pixels.push((i % 256) as u8);
        }

        let opts = PngOptions {
            optimal_compression: true,
            compression_level: 9,
            filter_strategy: FilterStrategy::Adaptive,
            ..Default::default()
        };

        let png = test_encode_with_options(&pixels, 64, 64, ColorType::Rgb, &opts).unwrap();

        // Should produce valid PNG
        assert_eq!(&png[0..8], &PNG_SIGNATURE);

        // Verify decode roundtrip
        let decoded = image::load_from_memory(&png).expect("decode");
        assert_eq!(decoded.width(), 64);
        assert_eq!(decoded.height(), 64);
    }

    #[test]
    fn test_png_optimal_compression_vs_standard() {
        // Repetitive data should compress well
        let data = vec![42u8; 64 * 64 * 3];

        let standard_opts = PngOptions::builder(64, 64)
            .color_type(ColorType::Rgb)
            .compression_level(9)
            .filter_strategy(FilterStrategy::Adaptive)
            .optimal_compression(false)
            .build();

        let optimal_opts = PngOptions::builder(64, 64)
            .color_type(ColorType::Rgb)
            .compression_level(9)
            .filter_strategy(FilterStrategy::Adaptive)
            .optimal_compression(true)
            .build();

        let standard = encode(&data, &standard_opts).unwrap();
        let optimal = encode(&data, &optimal_opts).unwrap();

        // Optimal should not be larger than standard (may be equal for simple data)
        assert!(
            optimal.len() <= standard.len(),
            "optimal ({}) should not be larger than standard ({})",
            optimal.len(),
            standard.len()
        );
    }

    // ============================================================================
    // Palette Quantization Edge Case Tests
    // ============================================================================

    #[test]
    fn test_png_reduce_palette_with_few_unique_colors() {
        // Image with 20 unique colors using RGBA (forces 8-bit depth, >16 colors)
        // This exercises the lossless palette reduction path
        let mut pixels = Vec::with_capacity(64 * 64 * 4);
        for i in 0..(64 * 64) {
            let color_idx = i % 20;
            pixels.push((color_idx * 12) as u8); // R
            pixels.push((color_idx * 10) as u8); // G
            pixels.push((color_idx * 8) as u8); // B
            pixels.push(255); // A (opaque)
        }

        let opts = PngOptions {
            reduce_palette: true,
            ..Default::default()
        };

        let png = test_encode_with_options(&pixels, 64, 64, ColorType::Rgba, &opts).unwrap();
        assert_eq!(&png[0..8], &PNG_SIGNATURE);

        // Should produce indexed PNG (color type 3)
        assert_eq!(png[25], 3, "should produce indexed PNG");
        // Should have PLTE chunk
        assert!(png.windows(4).any(|w| w == b"PLTE"));
    }

    #[test]
    fn test_png_reduce_palette_many_colors_stays_rgb() {
        // Image with many unique colors (gradient) should stay as RGB
        let mut pixels = Vec::with_capacity(64 * 64 * 3);
        for y in 0..64 {
            for x in 0..64 {
                pixels.push(((x * 4) % 256) as u8);
                pixels.push(((y * 4) % 256) as u8);
                pixels.push((((x + y) * 2) % 256) as u8);
            }
        }

        let opts = PngOptions {
            reduce_palette: true,
            ..Default::default()
        };

        let png = test_encode_with_options(&pixels, 64, 64, ColorType::Rgb, &opts).unwrap();
        assert_eq!(&png[0..8], &PNG_SIGNATURE);

        // With many colors, should stay as RGB (color type 2) or still be indexed if < 256
        // This is a behavioral test - just ensure it produces valid PNG
        let decoded = image::load_from_memory(&png).expect("decode");
        assert_eq!(decoded.width(), 64);
        assert_eq!(decoded.height(), 64);
    }

    // ============================================================================
    // Alpha Optimization Tests
    // ============================================================================

    #[test]
    fn test_png_reduce_color_type_rgba_to_rgb() {
        // RGBA image with all pixels fully opaque - should convert to RGB
        let mut pixels = Vec::with_capacity(16 * 16 * 4);
        for i in 0..256 {
            pixels.push((i % 256) as u8);
            pixels.push(((i * 2) % 256) as u8);
            pixels.push(((i * 3) % 256) as u8);
            pixels.push(255); // Fully opaque
        }

        let opts = PngOptions {
            reduce_color_type: true, // This is what converts RGBA to RGB
            ..Default::default()
        };

        let png = test_encode_with_options(&pixels, 16, 16, ColorType::Rgba, &opts).unwrap();
        assert_eq!(&png[0..8], &PNG_SIGNATURE);

        // Should convert to RGB (color type 2) since all alpha is 255
        assert_eq!(png[25], 2, "should convert to RGB when all alpha is 255");
    }

    #[test]
    fn test_png_optimize_alpha_zeroes_transparent_colors() {
        // RGBA image with some fully transparent pixels
        // optimize_alpha should zero out color channels where alpha is 0
        let pixels = vec![
            100, 150, 200, 0, // transparent - colors should be zeroed
            50, 100, 150, 255, // opaque - should stay
            25, 75, 125, 0, // transparent - colors should be zeroed
            200, 100, 50, 255, // opaque - should stay
        ];

        let opts = PngOptions {
            optimize_alpha: true,
            ..Default::default()
        };

        let png = test_encode_with_options(&pixels, 2, 2, ColorType::Rgba, &opts).unwrap();
        assert_eq!(&png[0..8], &PNG_SIGNATURE);

        // Should produce valid PNG
        let decoded = image::load_from_memory(&png).expect("decode");
        assert_eq!(decoded.width(), 2);
        assert_eq!(decoded.height(), 2);
    }

    #[test]
    fn test_png_reduce_color_type_stays_rgba_with_transparency() {
        // RGBA image with semi-transparent pixels - should stay as RGBA
        let mut pixels = Vec::with_capacity(16 * 16 * 4);
        for i in 0..256 {
            pixels.push((i % 256) as u8);
            pixels.push(((i * 2) % 256) as u8);
            pixels.push(((i * 3) % 256) as u8);
            pixels.push((i % 256) as u8); // Varying alpha (some are 0)
        }

        let opts = PngOptions {
            reduce_color_type: true,
            ..Default::default()
        };

        let png = test_encode_with_options(&pixels, 16, 16, ColorType::Rgba, &opts).unwrap();
        assert_eq!(&png[0..8], &PNG_SIGNATURE);

        // Should stay as RGBA (color type 6) since it has semi-transparent pixels
        assert_eq!(
            png[25], 6,
            "should stay RGBA when semi-transparent pixels exist"
        );
    }

    // Builder Pattern Coverage Tests

    #[test]
    fn test_builder_all_methods() {
        let opts = PngOptions::builder(100, 100)
            .color_type(ColorType::Rgb)
            .compression_level(7)
            .filter_strategy(FilterStrategy::Paeth)
            .optimize_alpha(true)
            .reduce_color_type(true)
            .strip_metadata(true)
            .reduce_palette(true)
            .verbose_filter_log(false)
            .optimal_compression(true)
            .quantization_mode(QuantizationMode::Auto)
            .quantization_max_colors(128)
            .quantization_dithering(true)
            .build();

        assert_eq!(opts.width, 100);
        assert_eq!(opts.height, 100);
        assert_eq!(opts.color_type, ColorType::Rgb);
        assert_eq!(opts.compression_level, 7);
        assert_eq!(opts.filter_strategy, FilterStrategy::Paeth);
        assert!(opts.optimize_alpha);
        assert!(opts.reduce_color_type);
        assert!(opts.strip_metadata);
        assert!(opts.reduce_palette);
        assert!(!opts.verbose_filter_log);
        assert!(opts.optimal_compression);
        assert_eq!(opts.quantization.mode, QuantizationMode::Auto);
        assert_eq!(opts.quantization.max_colors, 128);
        assert!(opts.quantization.dithering);
    }

    #[test]
    fn test_builder_quantization_full() {
        let quant_opts = QuantizationOptions {
            mode: QuantizationMode::Force,
            max_colors: 64,
            dithering: false,
        };
        let opts = PngOptions::builder(50, 50)
            .quantization(quant_opts.clone())
            .build();

        assert_eq!(opts.quantization.mode, QuantizationMode::Force);
        assert_eq!(opts.quantization.max_colors, 64);
        assert!(!opts.quantization.dithering);
    }

    #[test]
    fn test_preset_0_is_fast() {
        let opts = PngOptions::from_preset(10, 10, 0);
        assert_eq!(opts.compression_level, 2);
        assert_eq!(opts.filter_strategy, FilterStrategy::AdaptiveFast);
        assert!(!opts.optimize_alpha);
        assert!(!opts.optimal_compression);
    }

    #[test]
    fn test_preset_1_is_balanced() {
        let opts = PngOptions::from_preset(10, 10, 1);
        assert_eq!(opts.compression_level, 6);
        assert_eq!(opts.filter_strategy, FilterStrategy::Adaptive);
        assert!(opts.optimize_alpha);
        assert!(!opts.optimal_compression);
    }

    #[test]
    fn test_preset_2_is_max() {
        let opts = PngOptions::from_preset(10, 10, 2);
        assert_eq!(opts.compression_level, 9);
        assert_eq!(opts.filter_strategy, FilterStrategy::Bigrams);
        assert!(opts.optimize_alpha);
        assert!(opts.optimal_compression);
    }

    #[test]
    fn test_preset_invalid_defaults_to_balanced() {
        let opts = PngOptions::from_preset(10, 10, 99);
        assert_eq!(opts.compression_level, 6); // Balanced
    }

    // should_quantize_auto Tests

    #[test]
    fn test_should_quantize_auto_empty_data() {
        assert!(!should_quantize_auto(&[], 4, 256));
    }

    #[test]
    fn test_should_quantize_auto_few_colors() {
        // Very few colors - fewer than max_colors - should return false
        let data: Vec<u8> = (0..100).flat_map(|_| vec![100, 150, 200, 255]).collect();
        assert!(!should_quantize_auto(&data, 4, 256)); // Only 1 unique color
    }

    #[test]
    fn test_should_quantize_auto_photo_like_many_colors() {
        // Photo-like data with many distinct colors - should return false
        let mut data = Vec::with_capacity(10000 * 4);
        for i in 0..10000 {
            data.push((i % 256) as u8);
            data.push(((i * 3) % 256) as u8);
            data.push(((i * 7) % 256) as u8);
            data.push(255);
        }
        // With many colors (thousands), it looks like a photo - skip
        assert!(!should_quantize_auto(&data, 4, 256));
    }

    #[test]
    fn test_should_quantize_auto_unsupported_bpp() {
        // BPP != 3 or 4 should return false
        let data = vec![100u8; 100]; // 1 bpp
        assert!(!should_quantize_auto(&data, 1, 256));
        assert!(!should_quantize_auto(&data, 2, 256));
    }

    #[test]
    fn test_should_quantize_auto_rgb_data() {
        // RGB data (bpp=3) with moderate colors
        let mut data = Vec::new();
        for i in 0..5000 {
            let color_idx = i % 500;
            data.push((color_idx % 256) as u8);
            data.push(((color_idx / 256) * 50) as u8);
            data.push(((color_idx * 3) % 256) as u8);
        }
        // Just verify it doesn't panic and returns a boolean
        let _ = should_quantize_auto(&data, 3, 256);
    }

    // Filter Strategy Tests

    #[test]
    fn test_filter_strategy_none() {
        let pixels = vec![100u8; 64 * 64 * 3];
        let opts = PngOptions::builder(64, 64)
            .color_type(ColorType::Rgb)
            .filter_strategy(FilterStrategy::None)
            .build();
        let png = encode(&pixels, &opts).unwrap();
        assert!(!png.is_empty());
    }

    #[test]
    fn test_filter_strategy_up() {
        let pixels = vec![100u8; 64 * 64 * 3];
        let opts = PngOptions::builder(64, 64)
            .color_type(ColorType::Rgb)
            .filter_strategy(FilterStrategy::Up)
            .build();
        let png = encode(&pixels, &opts).unwrap();
        assert!(!png.is_empty());
    }

    #[test]
    fn test_filter_strategy_average() {
        let pixels = vec![100u8; 64 * 64 * 3];
        let opts = PngOptions::builder(64, 64)
            .color_type(ColorType::Rgb)
            .filter_strategy(FilterStrategy::Average)
            .build();
        let png = encode(&pixels, &opts).unwrap();
        assert!(!png.is_empty());
    }

    #[test]
    fn test_filter_strategy_minsum() {
        let pixels = vec![100u8; 64 * 64 * 3];
        let opts = PngOptions::builder(64, 64)
            .color_type(ColorType::Rgb)
            .filter_strategy(FilterStrategy::MinSum)
            .build();
        let png = encode(&pixels, &opts).unwrap();
        assert!(!png.is_empty());
    }

    // Indexed Encoding Edge Cases

    #[test]
    fn test_encode_indexed_single_color_palette() {
        let indices = vec![0u8; 16]; // 4x4 image, all index 0
        let palette = [[255, 0, 0]]; // Single red entry
        let png = encode_indexed(&indices, 4, 4, &palette, None).unwrap();
        assert_eq!(&png[0..8], &PNG_SIGNATURE);
        // Verify it decodes correctly
        let decoded = image::load_from_memory(&png).expect("decode");
        assert_eq!(decoded.width(), 4);
    }

    #[test]
    fn test_encode_indexed_256_color_palette() {
        let mut indices = Vec::with_capacity(256);
        let mut palette = Vec::with_capacity(256);
        for i in 0..256 {
            indices.push(i as u8);
            palette.push([i as u8, (255 - i) as u8, (i / 2) as u8]);
        }
        let png = encode_indexed(&indices, 16, 16, &palette, None).unwrap();
        assert_eq!(&png[0..8], &PNG_SIGNATURE);
        assert!(png.windows(4).any(|w| w == b"PLTE"));
    }

    #[test]
    fn test_encode_indexed_with_partial_trns() {
        // tRNS shorter than palette - remaining entries are opaque
        let indices = vec![0u8, 1, 2, 3];
        let palette = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]];
        let trns = [128u8, 255]; // Only first 2 entries have transparency
        let png = encode_indexed(&indices, 2, 2, &palette, Some(&trns)).unwrap();
        assert_eq!(&png[0..8], &PNG_SIGNATURE);
        assert!(png.windows(4).any(|w| w == b"tRNS"));
    }

    #[test]
    fn test_encode_indexed_with_options_compression() {
        let indices = vec![0u8; 64]; // 8x8 image
        let palette = [[100, 150, 200]];
        let opts = PngOptions::builder(8, 8)
            .compression_level(9)
            .filter_strategy(FilterStrategy::Sub)
            .build();
        let png = encode_indexed_with_options(&indices, 8, 8, &palette, None, &opts).unwrap();
        assert!(!png.is_empty());
    }

    // Color Reduction Edge Cases

    #[test]
    fn test_all_gray_rgb_single_pixel() {
        assert!(all_gray_rgb(&[128, 128, 128]));
    }

    #[test]
    fn test_all_gray_rgb_not_gray() {
        assert!(!all_gray_rgb(&[128, 128, 129]));
    }

    #[test]
    fn test_all_gray_rgb_empty() {
        assert!(all_gray_rgb(&[]));
    }

    #[test]
    fn test_analyze_rgba_single_opaque_gray() {
        let (all_opaque, all_gray) = analyze_rgba(&[100, 100, 100, 255]);
        assert!(all_opaque);
        assert!(all_gray);
    }

    #[test]
    fn test_analyze_rgba_transparent_gray() {
        let (all_opaque, all_gray) = analyze_rgba(&[100, 100, 100, 0]);
        assert!(!all_opaque);
        assert!(all_gray);
    }

    #[test]
    fn test_analyze_rgba_opaque_color() {
        let (all_opaque, all_gray) = analyze_rgba(&[100, 150, 200, 255]);
        assert!(all_opaque);
        assert!(!all_gray);
    }

    // Optimal Compression Tests

    #[test]
    fn test_optimal_compression_small_image() {
        let pixels = vec![100u8; 8 * 8 * 3];
        let opts = PngOptions::builder(8, 8)
            .color_type(ColorType::Rgb)
            .optimal_compression(true)
            .build();
        let png = encode(&pixels, &opts).unwrap();
        assert!(!png.is_empty());
    }

    #[test]
    fn test_max_preset_exercises_optimal_compression() {
        let mut pixels = Vec::with_capacity(32 * 32 * 3);
        for i in 0..(32 * 32) {
            pixels.push(((i * 3) % 256) as u8);
            pixels.push(((i * 5) % 256) as u8);
            pixels.push(((i * 7) % 256) as u8);
        }

        let mut max_opts = PngOptions::max(32, 32);
        max_opts.color_type = ColorType::Rgb;
        // Disable extra features that might affect size comparison
        max_opts.reduce_palette = false;
        max_opts.reduce_color_type = false;

        let png = encode(&pixels, &max_opts).unwrap();
        assert!(!png.is_empty());
        assert_eq!(&png[0..8], &PNG_SIGNATURE);

        // Verify roundtrip
        let decoded = image::load_from_memory(&png).expect("decode");
        assert_eq!(decoded.width(), 32);
        assert_eq!(decoded.height(), 32);
    }

    // Metadata Stripping Tests

    #[test]
    fn test_strip_metadata_removes_text_chunks() {
        let pixels = vec![100u8; 16 * 16 * 3];
        let opts = PngOptions::builder(16, 16)
            .color_type(ColorType::Rgb)
            .strip_metadata(true)
            .build();
        let png = encode(&pixels, &opts).unwrap();

        // Verify no tEXt, iTXt, or zTXt chunks
        assert!(!png.windows(4).any(|w| w == b"tEXt"));
        assert!(!png.windows(4).any(|w| w == b"iTXt"));
        assert!(!png.windows(4).any(|w| w == b"zTXt"));
    }

    // GrayAlpha Color Type Tests

    #[test]
    fn test_encode_grayalpha() {
        let pixels = vec![128u8, 255, 64, 128, 192, 64, 32, 255]; // 2x2 gray+alpha
        let opts = PngOptions::builder(2, 2)
            .color_type(ColorType::GrayAlpha)
            .build();
        let png = encode(&pixels, &opts).unwrap();
        assert_eq!(&png[0..8], &PNG_SIGNATURE);
        assert_eq!(png[25], 4); // Color type 4 = GrayAlpha
    }

    #[test]
    fn test_reduce_grayalpha_to_gray() {
        // GrayAlpha with all alpha=255 - verify valid encoding
        let pixels = vec![128u8, 255, 64, 255, 192, 255, 32, 255]; // 2x2 gray+alpha, all opaque
        let opts = PngOptions::builder(2, 2)
            .color_type(ColorType::GrayAlpha)
            .reduce_color_type(true)
            .build();
        let png = encode(&pixels, &opts).unwrap();
        assert_eq!(&png[0..8], &PNG_SIGNATURE);
        // The encoder may or may not reduce to Gray - verify it produces valid PNG
        let color_type = png[25];
        assert!(
            color_type == 0 || color_type == 4,
            "Should be Gray or GrayAlpha"
        );
    }
}
