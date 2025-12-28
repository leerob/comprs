//! JPEG encoder implementation.
//!
//! Implements baseline and progressive JPEG encoding (DCT-based lossy compression).
//! Supports:
//! - Baseline sequential DCT (SOF0)
//! - Progressive DCT (SOF2) with spectral selection and successive approximation
//! - Integer and floating-point DCT
//! - Optimized Huffman tables
//! - Trellis quantization for better R-D optimization
//!
//! For a conceptual overview and tuning guidance, see
//! [`crate::guides::jpeg_encoding`], [`crate::guides::dct`],
//! and [`crate::guides::quantization`].

pub mod dct;
pub mod huffman;
pub mod progressive;
pub mod quantize;
pub mod trellis;

use crate::bits::BitWriterMsb;
use crate::color::{rgb_to_ycbcr, ColorType};
use crate::error::{Error, Result};

use dct::dct_2d;
use huffman::{encode_block, HuffmanTables};
use progressive::{
    encode_ac_first, encode_dc_refine, get_dc_code, simple_progressive_script, ScanParams, ScanSpec,
};
use quantize::{quantize_block, zigzag_reorder, QuantizationTables};

/// Maximum supported image dimension for JPEG.
const MAX_DIMENSION: u32 = 65535;

/// JPEG markers.
const SOI: u16 = 0xFFD8; // Start of Image
const EOI: u16 = 0xFFD9; // End of Image
const APP0: u16 = 0xFFE0; // JFIF marker
const DQT: u16 = 0xFFDB; // Define Quantization Table
const SOF0: u16 = 0xFFC0; // Start of Frame (baseline DCT)
const SOF2: u16 = 0xFFC2; // Start of Frame (progressive DCT)
const DHT: u16 = 0xFFC4; // Define Huffman Table
const SOS: u16 = 0xFFDA; // Start of Scan

/// Internal context for JPEG encoding operations.
struct EncodeContext<'a> {
    data: &'a [u8],
    width: usize,
    height: usize,
    color_type: ColorType,
    subsampling: Subsampling,
    quant_tables: &'a QuantizationTables,
    huff_tables: &'a HuffmanTables,
    use_trellis: bool,
}

/// A vector of quantized 8x8 DCT coefficient blocks.
type DctCoefficients = Vec<[i16; 64]>;

/// YCbCr coefficient triplet (Y, Cb, Cr channels).
type YCbCrCoefficients = (DctCoefficients, DctCoefficients, DctCoefficients);

/// Encode raw pixel data as JPEG.
///
/// # Arguments
///
/// * `data` - Raw pixel data (row-major order)
/// * `options` - JPEG encoding options (includes width, height, color type, quality)
///
/// # Returns
///
/// Complete JPEG file as bytes.
///
/// # Example
///
/// ```rust
/// use pixo::jpeg::{encode, JpegOptions};
/// use pixo::ColorType;
///
/// let pixels = vec![255, 0, 0]; // 1x1 RGB red pixel
/// let options = JpegOptions::builder(1, 1)
///     .color_type(ColorType::Rgb)
///     .quality(85)
///     .build();
/// let jpeg_bytes = encode(&pixels, &options).unwrap();
/// ```
#[must_use = "encoding produces a JPEG file that should be used"]
pub fn encode(data: &[u8], options: &JpegOptions) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    encode_into(&mut output, data, options)?;
    Ok(output)
}

/// Chroma subsampling options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Subsampling {
    /// 4:4:4, no subsampling.
    S444,
    /// 4:2:0, 2x2 chroma downsample.
    S420,
}

/// JPEG encoding options.
///
/// Use [`JpegOptions::builder(1, 1)`] to create options with a fluent API.
///
/// # Example
///
/// ```rust
/// use pixo::jpeg::{encode, JpegOptions};
/// use pixo::ColorType;
///
/// let pixels = vec![255, 0, 0]; // 1x1 RGB red pixel
/// let options = JpegOptions::builder(1, 1)
///     .color_type(ColorType::Rgb)
///     .quality(85)
///     .build();
/// let jpeg_bytes = encode(&pixels, &options).unwrap();
/// ```
#[derive(Debug, Clone, Copy)]
pub struct JpegOptions {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Color type of the pixel data (Rgb or Gray).
    pub color_type: ColorType,
    /// Quality level 1-100.
    pub quality: u8,
    /// Subsampling scheme.
    pub subsampling: Subsampling,
    /// Restart interval in MCUs (None = disabled).
    pub restart_interval: Option<u16>,
    /// If true, build image-optimized Huffman tables (like mozjpeg optimize_coding).
    pub optimize_huffman: bool,
    /// If true, use progressive encoding (multiple scans).
    pub progressive: bool,
    /// If true, use trellis quantization for better R-D optimization.
    pub trellis_quant: bool,
}

impl Default for JpegOptions {
    fn default() -> Self {
        Self {
            // Default dimensions (must be set via builder)
            width: 0,
            height: 0,
            color_type: ColorType::Rgb,
            quality: 75,
            subsampling: Subsampling::S444,
            restart_interval: None,
            optimize_huffman: false,
            progressive: false,
            trellis_quant: false,
        }
    }
}

impl JpegOptions {
    /// Preset 0: Fast - standard Huffman, 4:4:4, baseline (fastest encoding).
    /// Defaults to RGB color type.
    pub fn fast(width: u32, height: u32, quality: u8) -> Self {
        Self {
            width,
            height,
            color_type: ColorType::Rgb,
            quality,
            subsampling: Subsampling::S444,
            restart_interval: None,
            optimize_huffman: false,
            progressive: false,
            trellis_quant: false,
        }
    }

    /// Preset 1: Balanced - optimized Huffman, 4:4:4, baseline (good balance).
    /// Defaults to RGB color type.
    pub fn balanced(width: u32, height: u32, quality: u8) -> Self {
        Self {
            width,
            height,
            color_type: ColorType::Rgb,
            quality,
            subsampling: Subsampling::S444,
            restart_interval: None,
            optimize_huffman: true,
            progressive: false,
            trellis_quant: false,
        }
    }

    /// Preset 2: Max - all optimizations enabled (maximum compression).
    /// Uses 4:2:0 subsampling, optimized Huffman, progressive encoding, and trellis quantization.
    /// Defaults to RGB color type.
    pub fn max(width: u32, height: u32, quality: u8) -> Self {
        Self {
            width,
            height,
            color_type: ColorType::Rgb,
            quality,
            subsampling: Subsampling::S420,
            restart_interval: None,
            optimize_huffman: true,
            progressive: true,
            trellis_quant: true,
        }
    }

    /// Create from preset (0=fast, 1=balanced, 2=max).
    pub fn from_preset(width: u32, height: u32, quality: u8, preset: u8) -> Self {
        match preset {
            0 => Self::fast(width, height, quality),
            2 => Self::max(width, height, quality),
            _ => Self::balanced(width, height, quality),
        }
    }

    /// Create a builder for [`JpegOptions`].
    ///
    /// The source dimensions are required; color type defaults to RGB, quality to 75.
    pub fn builder(width: u32, height: u32) -> JpegOptionsBuilder {
        JpegOptionsBuilder::new(width, height)
    }
}

/// Builder for [`JpegOptions`].
///
/// Create with [`JpegOptions::builder(width, height)`] and configure options fluently.
#[derive(Debug, Clone)]
pub struct JpegOptionsBuilder {
    options: JpegOptions,
}

impl JpegOptionsBuilder {
    /// Create a new builder with image dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            options: JpegOptions {
                width,
                height,
                color_type: ColorType::Rgb,
                ..Default::default()
            },
        }
    }

    /// Set the color type of the pixel data.
    pub fn color_type(mut self, color_type: ColorType) -> Self {
        self.options.color_type = color_type;
        self
    }

    pub fn quality(mut self, quality: u8) -> Self {
        self.options.quality = quality;
        self
    }

    pub fn subsampling(mut self, subsampling: Subsampling) -> Self {
        self.options.subsampling = subsampling;
        self
    }

    /// None disables restarts.
    pub fn restart_interval(mut self, interval: Option<u16>) -> Self {
        self.options.restart_interval = interval;
        self
    }

    pub fn optimize_huffman(mut self, value: bool) -> Self {
        self.options.optimize_huffman = value;
        self
    }

    pub fn progressive(mut self, value: bool) -> Self {
        self.options.progressive = value;
        self
    }

    pub fn trellis_quant(mut self, value: bool) -> Self {
        self.options.trellis_quant = value;
        self
    }

    /// Apply a preset while retaining dimensions, color type, and quality.
    pub fn preset(mut self, preset: u8) -> Self {
        let width = self.options.width;
        let height = self.options.height;
        let color_type = self.options.color_type;
        let quality = self.options.quality;
        self.options = JpegOptions::from_preset(width, height, quality, preset);
        self.options.color_type = color_type;
        self
    }

    /// Build the [`JpegOptions`].
    #[must_use]
    pub fn build(self) -> JpegOptions {
        self.options
    }
}

/// Encode raw pixel data as JPEG into a caller-provided buffer.
///
/// The `output` buffer will be cleared and reused, allowing callers to avoid
/// repeated allocations across multiple encodes.
///
/// # Arguments
///
/// * `output` - Buffer to write JPEG data into (will be cleared)
/// * `data` - Raw pixel data (row-major order)
/// * `options` - JPEG encoding options (includes width, height, color type, quality)
///
/// # Example
///
/// ```rust
/// use pixo::jpeg::{encode_into, JpegOptions};
/// use pixo::ColorType;
///
/// let pixels = vec![255, 0, 0]; // 1x1 RGB red pixel
/// let options = JpegOptions::builder(1, 1)
///     .color_type(ColorType::Rgb)
///     .quality(85)
///     .build();
/// let mut output = Vec::new();
/// encode_into(&mut output, &pixels, &options).unwrap();
/// ```
#[must_use = "this `Result` may indicate an encoding error"]
pub fn encode_into(output: &mut Vec<u8>, data: &[u8], options: &JpegOptions) -> Result<()> {
    let width = options.width;
    let height = options.height;
    let color_type = options.color_type;
    // Validate quality
    if options.quality == 0 || options.quality > 100 {
        return Err(Error::InvalidQuality(options.quality));
    }
    if matches!(options.restart_interval, Some(0)) {
        return Err(Error::InvalidRestartInterval(0));
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

    // Validate color type (JPEG only supports RGB and Gray)
    let bytes_per_pixel = match color_type {
        ColorType::Rgb => 3,
        ColorType::Gray => 1,
        _ => return Err(Error::UnsupportedColorType),
    };

    // Validate data length
    let expected_len = (width as usize)
        .checked_mul(height as usize)
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

    output.clear();
    output.reserve(expected_len / 4);

    // Create quantization and Huffman tables
    let quant_tables = QuantizationTables::with_quality(options.quality);
    let huff_tables = if options.optimize_huffman {
        build_optimized_huffman_tables(
            data,
            width,
            height,
            color_type,
            options.subsampling,
            options.restart_interval,
            &quant_tables,
        )
        .unwrap_or_default()
    } else {
        HuffmanTables::default()
    };

    // Write JPEG headers
    write_soi(output);
    write_app0(output);
    write_dqt(output, &quant_tables);

    if options.progressive {
        // Progressive JPEG encoding
        write_sof2(output, width, height, color_type, options.subsampling);
        write_dht(output, &huff_tables);
        if let Some(interval) = options.restart_interval {
            write_dri(output, interval);
        }

        // Encode with progressive scans
        let ctx = EncodeContext {
            data,
            width: width as usize,
            height: height as usize,
            color_type,
            subsampling: options.subsampling,
            quant_tables: &quant_tables,
            huff_tables: &huff_tables,
            use_trellis: options.trellis_quant,
        };
        encode_progressive(output, &ctx);
    } else {
        // Baseline JPEG encoding
        write_sof0(output, width, height, color_type, options.subsampling);
        write_dht(output, &huff_tables);
        if let Some(interval) = options.restart_interval {
            write_dri(output, interval);
        }

        // Write scan data
        write_sos(output, color_type);
        let ctx = EncodeContext {
            data,
            width: width as usize,
            height: height as usize,
            color_type,
            subsampling: options.subsampling,
            quant_tables: &quant_tables,
            huff_tables: &huff_tables,
            use_trellis: options.trellis_quant,
        };
        encode_scan(output, &ctx, options.restart_interval);
    }

    // Write end marker
    write_eoi(output);

    Ok(())
}

fn write_soi(output: &mut Vec<u8>) {
    output.extend_from_slice(&SOI.to_be_bytes());
}

fn write_eoi(output: &mut Vec<u8>) {
    output.extend_from_slice(&EOI.to_be_bytes());
}

fn write_app0(output: &mut Vec<u8>) {
    output.extend_from_slice(&APP0.to_be_bytes());

    // Length (16 bytes including length field)
    output.extend_from_slice(&16u16.to_be_bytes());

    // JFIF identifier
    output.extend_from_slice(b"JFIF\0");

    // Version 1.01
    output.push(1);
    output.push(1);

    // Units: 0 = no units (aspect ratio only)
    output.push(0);

    // X density
    output.extend_from_slice(&1u16.to_be_bytes());

    // Y density
    output.extend_from_slice(&1u16.to_be_bytes());

    // Thumbnail dimensions (0x0 = no thumbnail)
    output.push(0);
    output.push(0);
}

fn write_dqt(output: &mut Vec<u8>, tables: &QuantizationTables) {
    // Luminance table
    output.extend_from_slice(&DQT.to_be_bytes());
    output.extend_from_slice(&67u16.to_be_bytes()); // Length: 2 + 1 + 64
    output.push(0); // Table 0, 8-bit precision
    output.extend_from_slice(&tables.luminance);

    // Chrominance table
    output.extend_from_slice(&DQT.to_be_bytes());
    output.extend_from_slice(&67u16.to_be_bytes());
    output.push(1); // Table 1, 8-bit precision
    output.extend_from_slice(&tables.chrominance);
}

fn write_sof0(
    output: &mut Vec<u8>,
    width: u32,
    height: u32,
    color_type: ColorType,
    subsampling: Subsampling,
) {
    write_sof_marker(output, SOF0, width, height, color_type, subsampling);
}

fn write_sof2(
    output: &mut Vec<u8>,
    width: u32,
    height: u32,
    color_type: ColorType,
    subsampling: Subsampling,
) {
    write_sof_marker(output, SOF2, width, height, color_type, subsampling);
}

fn write_sof_marker(
    output: &mut Vec<u8>,
    marker: u16,
    width: u32,
    height: u32,
    color_type: ColorType,
    subsampling: Subsampling,
) {
    output.extend_from_slice(&marker.to_be_bytes());

    let num_components = match color_type {
        ColorType::Gray => 1,
        _ => 3,
    };

    // Length: 8 + 3*num_components
    let length = 8 + 3 * num_components;
    output.extend_from_slice(&(length as u16).to_be_bytes());

    // Precision: 8 bits
    output.push(8);

    // Height and width
    output.extend_from_slice(&(height as u16).to_be_bytes());
    output.extend_from_slice(&(width as u16).to_be_bytes());

    // Number of components
    output.push(num_components);

    if num_components == 1 {
        // Grayscale: 1 component
        output.push(1); // Component ID
        output.push(0x11); // Sampling factor (1x1)
        output.push(0); // Quantization table 0
    } else {
        // YCbCr: 3 components
        // Y component
        output.push(1); // Component ID
        let y_sampling = match subsampling {
            Subsampling::S444 => 0x11, // H=1, V=1
            Subsampling::S420 => 0x22, // H=2, V=2
        };
        output.push(y_sampling);
        output.push(0); // Quantization table 0 (luminance)

        // Cb component
        output.push(2);
        output.push(0x11);
        output.push(1); // Quantization table 1 (chrominance)

        // Cr component
        output.push(3);
        output.push(0x11);
        output.push(1);
    }
}

fn write_dht(output: &mut Vec<u8>, tables: &HuffmanTables) {
    // DC luminance
    write_huffman_table(output, 0x00, &tables.dc_lum_bits, &tables.dc_lum_vals);

    // DC chrominance
    write_huffman_table(output, 0x01, &tables.dc_chrom_bits, &tables.dc_chrom_vals);

    // AC luminance
    write_huffman_table(output, 0x10, &tables.ac_lum_bits, &tables.ac_lum_vals);

    // AC chrominance
    write_huffman_table(output, 0x11, &tables.ac_chrom_bits, &tables.ac_chrom_vals);
}

fn write_dri(output: &mut Vec<u8>, interval: u16) {
    output.extend_from_slice(&0xFFDDu16.to_be_bytes());
    output.extend_from_slice(&4u16.to_be_bytes()); // length = 4
    output.extend_from_slice(&interval.to_be_bytes());
}

fn write_huffman_table(output: &mut Vec<u8>, table_id: u8, bits: &[u8; 16], vals: &[u8]) {
    output.extend_from_slice(&DHT.to_be_bytes());

    // Length: 2 + 1 + 16 + num_values
    let length = 2 + 1 + 16 + vals.len();
    output.extend_from_slice(&(length as u16).to_be_bytes());

    // Table class and ID
    output.push(table_id);

    // Number of codes of each length
    output.extend_from_slice(bits);

    // Values
    output.extend_from_slice(vals);
}

fn write_sos(output: &mut Vec<u8>, color_type: ColorType) {
    output.extend_from_slice(&SOS.to_be_bytes());

    let num_components = match color_type {
        ColorType::Gray => 1,
        _ => 3,
    };

    // Length: 6 + 2*num_components
    let length = 6 + 2 * num_components;
    output.extend_from_slice(&(length as u16).to_be_bytes());

    // Number of components
    output.push(num_components);

    if num_components == 1 {
        output.push(1); // Component ID
        output.push(0x00); // DC/AC table selectors
    } else {
        // Y component: DC table 0, AC table 0
        output.push(1);
        output.push(0x00);

        // Cb component: DC table 1, AC table 1
        output.push(2);
        output.push(0x11);

        // Cr component: DC table 1, AC table 1
        output.push(3);
        output.push(0x11);
    }

    // Spectral selection and successive approximation
    output.push(0); // Start of spectral selection
    output.push(63); // End of spectral selection
    output.push(0); // Successive approximation
}

fn write_sos_progressive(output: &mut Vec<u8>, scan: &ScanSpec, color_type: ColorType) {
    output.extend_from_slice(&SOS.to_be_bytes());

    let num_components = scan.components.len() as u8;

    // Length: 6 + 2*num_components
    let length = 6 + 2 * num_components as u16;
    output.extend_from_slice(&length.to_be_bytes());

    // Number of components
    output.push(num_components);

    // Component specifications
    for &comp_id in &scan.components {
        // Component IDs are 1-based in JPEG, our indices are 0-based
        let jpeg_comp_id = comp_id + 1;
        output.push(jpeg_comp_id);

        // Table selectors: luminance (Y) uses tables 0, chroma uses tables 1
        let is_luminance = comp_id == 0;
        let table_sel = if is_luminance { 0x00 } else { 0x11 };
        output.push(table_sel);
    }

    // Spectral selection
    output.push(scan.ss);
    output.push(scan.se);

    // Successive approximation: high nibble = ah, low nibble = al
    output.push((scan.ah << 4) | scan.al);

    let _ = color_type; // Used for validation in future
}

fn build_optimized_huffman_tables(
    data: &[u8],
    width: u32,
    height: u32,
    color_type: ColorType,
    subsampling: Subsampling,
    restart_interval: Option<u16>,
    quant_tables: &QuantizationTables,
) -> Option<HuffmanTables> {
    let width = width as usize;
    let height = height as usize;

    let mut dc_lum = [0u64; 12];
    let mut dc_chrom = [0u64; 12];
    let mut ac_lum = [0u64; 256];
    let mut ac_chrom = [0u64; 256];

    match (color_type, subsampling) {
        (ColorType::Gray, _) => {
            let padded_width = (width + 7) & !7;
            let padded_height = (height + 7) & !7;
            let total_mcus = ((padded_width / 8) * (padded_height / 8)) as u32;
            let mut prev_dc_y = 0i16;
            let mut mcu_count: u32 = 0;

            for block_y in (0..padded_height).step_by(8) {
                for block_x in (0..padded_width).step_by(8) {
                    let (y_block, _, _) =
                        extract_block(data, width, height, block_x, block_y, color_type);
                    let y_dct = dct_2d(&y_block);
                    let y_quant = quantize_block(&y_dct, &quant_tables.luminance_table);
                    prev_dc_y = count_block(&y_quant, prev_dc_y, true, &mut dc_lum, &mut ac_lum);

                    mcu_count += 1;
                    // Reset DC prediction at restart boundaries (same logic as encode_scan)
                    if let Some(interval) = restart_interval {
                        if interval > 0
                            && mcu_count.rem_euclid(interval as u32) == 0
                            && mcu_count < total_mcus
                        {
                            prev_dc_y = 0;
                        }
                    }
                }
            }
            HuffmanTables::optimized_from_counts(&dc_lum, None, &ac_lum, None)
        }
        (_, Subsampling::S444) => {
            let padded_width = (width + 7) & !7;
            let padded_height = (height + 7) & !7;
            let total_mcus = ((padded_width / 8) * (padded_height / 8)) as u32;
            let mut prev_dc_y = 0i16;
            let mut prev_dc_cb = 0i16;
            let mut prev_dc_cr = 0i16;
            let mut mcu_count: u32 = 0;

            for block_y in (0..padded_height).step_by(8) {
                for block_x in (0..padded_width).step_by(8) {
                    let (y_block, cb_block, cr_block) =
                        extract_block(data, width, height, block_x, block_y, color_type);

                    let y_quant = quantize_block(&dct_2d(&y_block), &quant_tables.luminance_table);
                    prev_dc_y = count_block(&y_quant, prev_dc_y, true, &mut dc_lum, &mut ac_lum);

                    let cb_quant =
                        quantize_block(&dct_2d(&cb_block), &quant_tables.chrominance_table);
                    prev_dc_cb =
                        count_block(&cb_quant, prev_dc_cb, false, &mut dc_chrom, &mut ac_chrom);

                    let cr_quant =
                        quantize_block(&dct_2d(&cr_block), &quant_tables.chrominance_table);
                    prev_dc_cr =
                        count_block(&cr_quant, prev_dc_cr, false, &mut dc_chrom, &mut ac_chrom);

                    mcu_count += 1;
                    // Reset DC prediction at restart boundaries (same logic as encode_scan)
                    if let Some(interval) = restart_interval {
                        if interval > 0
                            && mcu_count.rem_euclid(interval as u32) == 0
                            && mcu_count < total_mcus
                        {
                            prev_dc_y = 0;
                            prev_dc_cb = 0;
                            prev_dc_cr = 0;
                        }
                    }
                }
            }

            HuffmanTables::optimized_from_counts(&dc_lum, Some(&dc_chrom), &ac_lum, Some(&ac_chrom))
        }
        (_, Subsampling::S420) => {
            let padded_width_420 = (width + 15) & !15;
            let padded_height_420 = (height + 15) & !15;
            let total_mcus = ((padded_width_420 / 16) * (padded_height_420 / 16)) as u32;
            let mut prev_dc_y = 0i16;
            let mut prev_dc_cb = 0i16;
            let mut prev_dc_cr = 0i16;
            let mut mcu_count: u32 = 0;

            for mcu_y in (0..padded_height_420).step_by(16) {
                for mcu_x in (0..padded_width_420).step_by(16) {
                    let (y_blocks, cb_block, cr_block) =
                        extract_mcu_420(data, width, height, mcu_x, mcu_y);

                    for y_block in &y_blocks {
                        let y_quant =
                            quantize_block(&dct_2d(y_block), &quant_tables.luminance_table);
                        prev_dc_y =
                            count_block(&y_quant, prev_dc_y, true, &mut dc_lum, &mut ac_lum);
                    }

                    let cb_quant =
                        quantize_block(&dct_2d(&cb_block), &quant_tables.chrominance_table);
                    prev_dc_cb =
                        count_block(&cb_quant, prev_dc_cb, false, &mut dc_chrom, &mut ac_chrom);

                    let cr_quant =
                        quantize_block(&dct_2d(&cr_block), &quant_tables.chrominance_table);
                    prev_dc_cr =
                        count_block(&cr_quant, prev_dc_cr, false, &mut dc_chrom, &mut ac_chrom);

                    mcu_count += 1;
                    // Reset DC prediction at restart boundaries (same logic as encode_scan)
                    if let Some(interval) = restart_interval {
                        if interval > 0
                            && mcu_count.rem_euclid(interval as u32) == 0
                            && mcu_count < total_mcus
                        {
                            prev_dc_y = 0;
                            prev_dc_cb = 0;
                            prev_dc_cr = 0;
                        }
                    }
                }
            }

            HuffmanTables::optimized_from_counts(&dc_lum, Some(&dc_chrom), &ac_lum, Some(&ac_chrom))
        }
    }
}

fn count_block(
    block: &[i16; 64],
    prev_dc: i16,
    _is_luminance: bool,
    dc_counts: &mut [u64; 12],
    ac_counts: &mut [u64; 256],
) -> i16 {
    let zz = zigzag_reorder(block);
    let dc = zz[0];
    let dc_diff = dc - prev_dc;
    let dc_cat = category_i16(dc_diff);
    dc_counts[dc_cat as usize] += 1;

    let mut zero_run = 0usize;
    for &ac in zz.iter().skip(1) {
        if ac == 0 {
            zero_run += 1;
        } else {
            while zero_run >= 16 {
                ac_counts[0xF0] += 1;
                zero_run -= 16;
            }
            let ac_cat = category_i16(ac);
            let rs = ((zero_run as u8) << 4) | ac_cat;
            ac_counts[rs as usize] += 1;
            zero_run = 0;
        }
    }
    if zero_run > 0 {
        ac_counts[0] += 1; // EOB
    }

    // Return current DC for next differential block
    dc
}

#[inline]
fn category_i16(value: i16) -> u8 {
    let abs_val = value.unsigned_abs();
    if abs_val == 0 {
        0
    } else {
        16 - abs_val.leading_zeros() as u8
    }
}

fn encode_progressive(output: &mut Vec<u8>, ctx: &EncodeContext) {
    // Step 1: Compute all DCT coefficients and store them
    let (y_coeffs, cb_coeffs, cr_coeffs) = compute_all_coefficients(
        ctx.data,
        ctx.width,
        ctx.height,
        ctx.color_type,
        ctx.subsampling,
        ctx.quant_tables,
        ctx.use_trellis,
    );

    // Step 2: Get progressive scan script
    let script = simple_progressive_script();

    // Step 3: Encode each scan
    for scan in &script {
        write_sos_progressive(output, scan, ctx.color_type);

        let mut writer = BitWriterMsb::new();

        if scan.is_dc_scan() {
            encode_dc_scan(
                &mut writer,
                scan,
                &y_coeffs,
                &cb_coeffs,
                &cr_coeffs,
                ctx.subsampling,
                ctx.huff_tables,
            );
        } else if scan.is_first_scan() {
            encode_ac_first_scan(
                &mut writer,
                scan,
                &y_coeffs,
                &cb_coeffs,
                &cr_coeffs,
                ctx.subsampling,
                ctx.huff_tables,
            );
        } else {
            encode_ac_refine_scan(
                &mut writer,
                scan,
                &y_coeffs,
                &cb_coeffs,
                &cr_coeffs,
                ctx.subsampling,
                ctx.huff_tables,
            );
        }

        output.extend_from_slice(&writer.finish());
    }
}

/// Compute all DCT coefficients for the image.
/// Uses parallel processing with Rayon when the `parallel` feature is enabled.
#[allow(clippy::type_complexity)]
fn compute_all_coefficients(
    data: &[u8],
    width: usize,
    height: usize,
    color_type: ColorType,
    subsampling: Subsampling,
    quant_tables: &QuantizationTables,
    use_trellis: bool,
) -> YCbCrCoefficients {
    #[cfg(feature = "parallel")]
    {
        compute_all_coefficients_parallel(
            data,
            width,
            height,
            color_type,
            subsampling,
            quant_tables,
            use_trellis,
        )
    }

    #[cfg(not(feature = "parallel"))]
    {
        compute_all_coefficients_sequential(
            data,
            width,
            height,
            color_type,
            subsampling,
            quant_tables,
            use_trellis,
        )
    }
}

/// Quantize a DCT block with optional trellis optimization.
#[inline]
fn quantize_dct(dct: &[f32; 64], table: &[f32; 64], use_trellis: bool) -> [i16; 64] {
    if use_trellis {
        trellis::trellis_quantize(dct, table, None)
    } else {
        quantize_block(dct, table)
    }
}

/// Process a single 8x8 block and return quantized YCbCr coefficients.
#[inline]
#[allow(clippy::too_many_arguments)]
fn process_block_444(
    data: &[u8],
    width: usize,
    height: usize,
    block_x: usize,
    block_y: usize,
    color_type: ColorType,
    quant_tables: &QuantizationTables,
    use_trellis: bool,
) -> ([i16; 64], [i16; 64], [i16; 64]) {
    let (y_block, cb_block, cr_block) =
        extract_block(data, width, height, block_x, block_y, color_type);

    let y_quant = quantize_dct(
        &dct_2d(&y_block),
        &quant_tables.luminance_table,
        use_trellis,
    );
    let cb_quant = quantize_dct(
        &dct_2d(&cb_block),
        &quant_tables.chrominance_table,
        use_trellis,
    );
    let cr_quant = quantize_dct(
        &dct_2d(&cr_block),
        &quant_tables.chrominance_table,
        use_trellis,
    );

    (y_quant, cb_quant, cr_quant)
}

/// Process a 4:2:0 MCU and return quantized coefficients.
#[inline]
fn process_mcu_420(
    data: &[u8],
    width: usize,
    height: usize,
    mcu_x: usize,
    mcu_y: usize,
    quant_tables: &QuantizationTables,
    use_trellis: bool,
) -> ([[i16; 64]; 4], [i16; 64], [i16; 64]) {
    let (y_blocks, cb_block, cr_block) = extract_mcu_420(data, width, height, mcu_x, mcu_y);

    let mut y_quants = [[0i16; 64]; 4];
    for (i, y_block) in y_blocks.iter().enumerate() {
        y_quants[i] = quantize_dct(&dct_2d(y_block), &quant_tables.luminance_table, use_trellis);
    }

    let cb_quant = quantize_dct(
        &dct_2d(&cb_block),
        &quant_tables.chrominance_table,
        use_trellis,
    );
    let cr_quant = quantize_dct(
        &dct_2d(&cr_block),
        &quant_tables.chrominance_table,
        use_trellis,
    );

    (y_quants, cb_quant, cr_quant)
}

/// Sequential implementation of coefficient computation.
#[cfg_attr(feature = "parallel", allow(dead_code))]
#[allow(clippy::type_complexity)]
fn compute_all_coefficients_sequential(
    data: &[u8],
    width: usize,
    height: usize,
    color_type: ColorType,
    subsampling: Subsampling,
    quant_tables: &QuantizationTables,
    use_trellis: bool,
) -> YCbCrCoefficients {
    match (color_type, subsampling) {
        (ColorType::Gray, _) => {
            let padded_width = (width + 7) & !7;
            let padded_height = (height + 7) & !7;
            let block_count = (padded_width / 8) * (padded_height / 8);
            let mut y_coeffs = Vec::with_capacity(block_count);

            for block_y in (0..padded_height).step_by(8) {
                for block_x in (0..padded_width).step_by(8) {
                    let (y_block, _, _) =
                        extract_block(data, width, height, block_x, block_y, color_type);
                    let y_quant = quantize_dct(
                        &dct_2d(&y_block),
                        &quant_tables.luminance_table,
                        use_trellis,
                    );
                    y_coeffs.push(y_quant);
                }
            }
            (y_coeffs, Vec::new(), Vec::new())
        }
        (_, Subsampling::S444) => {
            let padded_width = (width + 7) & !7;
            let padded_height = (height + 7) & !7;
            let block_count = (padded_width / 8) * (padded_height / 8);
            let mut y_coeffs = Vec::with_capacity(block_count);
            let mut cb_coeffs = Vec::with_capacity(block_count);
            let mut cr_coeffs = Vec::with_capacity(block_count);

            for block_y in (0..padded_height).step_by(8) {
                for block_x in (0..padded_width).step_by(8) {
                    let (y, cb, cr) = process_block_444(
                        data,
                        width,
                        height,
                        block_x,
                        block_y,
                        color_type,
                        quant_tables,
                        use_trellis,
                    );
                    y_coeffs.push(y);
                    cb_coeffs.push(cb);
                    cr_coeffs.push(cr);
                }
            }
            (y_coeffs, cb_coeffs, cr_coeffs)
        }
        (_, Subsampling::S420) => {
            let padded_width_420 = (width + 15) & !15;
            let padded_height_420 = (height + 15) & !15;
            let mcu_count = (padded_width_420 / 16) * (padded_height_420 / 16);
            let mut y_coeffs = Vec::with_capacity(mcu_count * 4);
            let mut cb_coeffs = Vec::with_capacity(mcu_count);
            let mut cr_coeffs = Vec::with_capacity(mcu_count);

            for mcu_y in (0..padded_height_420).step_by(16) {
                for mcu_x in (0..padded_width_420).step_by(16) {
                    let (y_quants, cb, cr) = process_mcu_420(
                        data,
                        width,
                        height,
                        mcu_x,
                        mcu_y,
                        quant_tables,
                        use_trellis,
                    );
                    y_coeffs.extend_from_slice(&y_quants);
                    cb_coeffs.push(cb);
                    cr_coeffs.push(cr);
                }
            }
            (y_coeffs, cb_coeffs, cr_coeffs)
        }
    }
}

/// Parallel implementation of coefficient computation using Rayon.
#[cfg(feature = "parallel")]
#[allow(clippy::type_complexity)]
fn compute_all_coefficients_parallel(
    data: &[u8],
    width: usize,
    height: usize,
    color_type: ColorType,
    subsampling: Subsampling,
    quant_tables: &QuantizationTables,
    use_trellis: bool,
) -> YCbCrCoefficients {
    use rayon::prelude::*;

    match (color_type, subsampling) {
        (ColorType::Gray, _) => {
            let padded_width = (width + 7) & !7;
            let padded_height = (height + 7) & !7;

            let coords: Vec<(usize, usize)> = (0..padded_height)
                .step_by(8)
                .flat_map(|y| (0..padded_width).step_by(8).map(move |x| (x, y)))
                .collect();

            let y_coeffs: DctCoefficients = coords
                .par_iter()
                .map(|&(block_x, block_y)| {
                    let (y_block, _, _) =
                        extract_block(data, width, height, block_x, block_y, color_type);
                    quantize_dct(
                        &dct_2d(&y_block),
                        &quant_tables.luminance_table,
                        use_trellis,
                    )
                })
                .collect();

            (y_coeffs, Vec::new(), Vec::new())
        }
        (_, Subsampling::S444) => {
            let padded_width = (width + 7) & !7;
            let padded_height = (height + 7) & !7;

            let coords: Vec<(usize, usize)> = (0..padded_height)
                .step_by(8)
                .flat_map(|y| (0..padded_width).step_by(8).map(move |x| (x, y)))
                .collect();

            let results: Vec<_> = coords
                .par_iter()
                .map(|&(block_x, block_y)| {
                    process_block_444(
                        data,
                        width,
                        height,
                        block_x,
                        block_y,
                        color_type,
                        quant_tables,
                        use_trellis,
                    )
                })
                .collect();

            let (y_coeffs, cb_coeffs, cr_coeffs) = unzip3(results);
            (y_coeffs, cb_coeffs, cr_coeffs)
        }
        (_, Subsampling::S420) => {
            let padded_width_420 = (width + 15) & !15;
            let padded_height_420 = (height + 15) & !15;

            let coords: Vec<(usize, usize)> = (0..padded_height_420)
                .step_by(16)
                .flat_map(|y| (0..padded_width_420).step_by(16).map(move |x| (x, y)))
                .collect();

            let results: Vec<_> = coords
                .par_iter()
                .map(|&(mcu_x, mcu_y)| {
                    process_mcu_420(data, width, height, mcu_x, mcu_y, quant_tables, use_trellis)
                })
                .collect();

            let mut y_coeffs = Vec::with_capacity(results.len() * 4);
            let mut cb_coeffs = Vec::with_capacity(results.len());
            let mut cr_coeffs = Vec::with_capacity(results.len());

            for (y_quants, cb, cr) in results {
                y_coeffs.extend_from_slice(&y_quants);
                cb_coeffs.push(cb);
                cr_coeffs.push(cr);
            }

            (y_coeffs, cb_coeffs, cr_coeffs)
        }
    }
}

/// Unzip a vector of 3-tuples into three vectors.
#[cfg(feature = "parallel")]
#[inline]
fn unzip3<A, B, C>(iter: Vec<(A, B, C)>) -> (Vec<A>, Vec<B>, Vec<C>) {
    let len = iter.len();
    let mut a = Vec::with_capacity(len);
    let mut b = Vec::with_capacity(len);
    let mut c = Vec::with_capacity(len);
    for (x, y, z) in iter {
        a.push(x);
        b.push(y);
        c.push(z);
    }
    (a, b, c)
}

fn encode_dc_scan(
    writer: &mut BitWriterMsb,
    scan: &ScanSpec,
    y_coeffs: &[[i16; 64]],
    cb_coeffs: &[[i16; 64]],
    cr_coeffs: &[[i16; 64]],
    subsampling: Subsampling,
    huff_tables: &HuffmanTables,
) {
    let al = scan.al;

    for &comp_id in &scan.components {
        let coeffs = match comp_id {
            0 => y_coeffs,
            1 => cb_coeffs,
            2 => cr_coeffs,
            _ => continue,
        };

        if coeffs.is_empty() {
            continue;
        }

        let is_luminance = comp_id == 0;
        let mut prev_dc = 0i16;

        // For 4:2:0, Y has 4 blocks per MCU, chroma has 1
        let blocks_per_mcu = if comp_id == 0 {
            match subsampling {
                Subsampling::S420 => 4,
                Subsampling::S444 => 1,
            }
        } else {
            1
        };

        let _ = blocks_per_mcu; // Used for proper MCU ordering

        for block in coeffs {
            let dc = block[0];
            let dc_diff = dc - prev_dc;

            if scan.is_refinement_scan() {
                // Refinement: output single bit
                encode_dc_refine(writer, dc, al);
            } else {
                // First scan: encode normally but shifted
                let shifted_dc = dc_diff >> al;
                let dc_cat = category_i16(shifted_dc);
                let dc_code = get_dc_code(huff_tables, dc_cat, is_luminance);
                writer.write_bits(dc_code.0 as u32, dc_code.1);

                if dc_cat > 0 {
                    let (val_bits, val_len) = encode_dc_value(shifted_dc);
                    writer.write_bits(val_bits as u32, val_len);
                }
            }

            prev_dc = dc;
        }
    }
}

fn encode_dc_value(value: i16) -> (u16, u8) {
    let cat = category_i16(value);
    if cat == 0 {
        return (0, 0);
    }

    let bits = if value < 0 {
        (value - 1) as u16
    } else {
        value as u16
    };

    (bits & ((1 << cat) - 1), cat)
}

fn encode_ac_first_scan(
    writer: &mut BitWriterMsb,
    scan: &ScanSpec,
    y_coeffs: &[[i16; 64]],
    cb_coeffs: &[[i16; 64]],
    cr_coeffs: &[[i16; 64]],
    _subsampling: Subsampling,
    huff_tables: &HuffmanTables,
) {
    for &comp_id in &scan.components {
        let coeffs = match comp_id {
            0 => y_coeffs,
            1 => cb_coeffs,
            2 => cr_coeffs,
            _ => continue,
        };

        if coeffs.is_empty() {
            continue;
        }

        let is_luminance = comp_id == 0;
        let mut eob_run = 0u16;

        for block in coeffs {
            let scan_params = ScanParams {
                ss: scan.ss,
                se: scan.se,
                al: scan.al,
                is_luminance,
            };
            encode_ac_first(writer, block, &scan_params, &mut eob_run, huff_tables);
        }

        // Flush any remaining EOB run
        if eob_run > 0 {
            progressive::flush_eob_run_public(writer, &mut eob_run, huff_tables, is_luminance);
        }
    }
}

fn encode_ac_refine_scan(
    writer: &mut BitWriterMsb,
    scan: &ScanSpec,
    y_coeffs: &[[i16; 64]],
    cb_coeffs: &[[i16; 64]],
    cr_coeffs: &[[i16; 64]],
    _subsampling: Subsampling,
    huff_tables: &HuffmanTables,
) {
    for &comp_id in &scan.components {
        let coeffs = match comp_id {
            0 => y_coeffs,
            1 => cb_coeffs,
            2 => cr_coeffs,
            _ => continue,
        };

        if coeffs.is_empty() {
            continue;
        }

        let is_luminance = comp_id == 0;
        let mut eob_run = 0u16;

        for block in coeffs {
            let scan_params = ScanParams {
                ss: scan.ss,
                se: scan.se,
                al: scan.al,
                is_luminance,
            };
            progressive::encode_ac_refine(writer, block, &scan_params, &mut eob_run, huff_tables);
        }

        // Flush any remaining EOB run
        if eob_run > 0 {
            progressive::flush_eob_run_public(writer, &mut eob_run, huff_tables, is_luminance);
        }
    }
}

fn encode_scan(output: &mut Vec<u8>, ctx: &EncodeContext, restart_interval: Option<u16>) {
    let mut writer = BitWriterMsb::new();

    // Calculate padded dimensions
    let (padded_width, padded_height) = match ctx.subsampling {
        Subsampling::S444 | Subsampling::S420 => ((ctx.width + 7) & !7, (ctx.height + 7) & !7),
    };

    // Previous DC values for differential encoding
    let mut prev_dc_y = 0i16;
    let mut prev_dc_cb = 0i16;
    let mut prev_dc_cr = 0i16;
    let mut rst_idx = 0u8;
    let mut mcu_count: u32 = 0;

    let handle_restart = |writer: &mut BitWriterMsb,
                          prev_dc_y: &mut i16,
                          prev_dc_cb: &mut i16,
                          prev_dc_cr: &mut i16,
                          mcu_count: u32,
                          total_mcus: u32,
                          rst_idx: &mut u8| {
        if let Some(interval) = restart_interval {
            // Only write restart marker if there are more MCUs to follow.
            // Skip the marker after the final MCU to avoid redundant bytes.
            let is_restart = interval > 0
                && mcu_count.rem_euclid(interval as u32) == 0
                && mcu_count < total_mcus;
            if is_restart {
                writer.flush();
                writer.write_bytes(&[0xFF, 0xD0 + (*rst_idx & 0x07)]);
                *rst_idx = (*rst_idx + 1) & 0x07;
                *prev_dc_y = 0;
                *prev_dc_cb = 0;
                *prev_dc_cr = 0;
            }
        }
    };

    // Process blocks
    match (ctx.color_type, ctx.subsampling) {
        (ColorType::Gray, _) => {
            let total_mcus = ((padded_width / 8) * (padded_height / 8)) as u32;
            for block_y in (0..padded_height).step_by(8) {
                for block_x in (0..padded_width).step_by(8) {
                    let (y_block, _, _) = extract_block(
                        ctx.data,
                        ctx.width,
                        ctx.height,
                        block_x,
                        block_y,
                        ctx.color_type,
                    );
                    let y_dct = dct_2d(&y_block);
                    let y_quant = quantize_block(&y_dct, &ctx.quant_tables.luminance_table);
                    prev_dc_y =
                        encode_block(&mut writer, &y_quant, prev_dc_y, true, ctx.huff_tables);
                    mcu_count += 1;
                    handle_restart(
                        &mut writer,
                        &mut prev_dc_y,
                        &mut prev_dc_cb,
                        &mut prev_dc_cr,
                        mcu_count,
                        total_mcus,
                        &mut rst_idx,
                    );
                }
            }
        }
        (_, Subsampling::S444) => {
            let total_mcus = ((padded_width / 8) * (padded_height / 8)) as u32;
            for block_y in (0..padded_height).step_by(8) {
                for block_x in (0..padded_width).step_by(8) {
                    let (y_block, cb_block, cr_block) = extract_block(
                        ctx.data,
                        ctx.width,
                        ctx.height,
                        block_x,
                        block_y,
                        ctx.color_type,
                    );

                    let y_dct = dct_2d(&y_block);
                    let y_quant = quantize_block(&y_dct, &ctx.quant_tables.luminance_table);
                    prev_dc_y =
                        encode_block(&mut writer, &y_quant, prev_dc_y, true, ctx.huff_tables);

                    let cb_dct = dct_2d(&cb_block);
                    let cb_quant = quantize_block(&cb_dct, &ctx.quant_tables.chrominance_table);
                    prev_dc_cb =
                        encode_block(&mut writer, &cb_quant, prev_dc_cb, false, ctx.huff_tables);

                    let cr_dct = dct_2d(&cr_block);
                    let cr_quant = quantize_block(&cr_dct, &ctx.quant_tables.chrominance_table);
                    prev_dc_cr =
                        encode_block(&mut writer, &cr_quant, prev_dc_cr, false, ctx.huff_tables);

                    mcu_count += 1;
                    handle_restart(
                        &mut writer,
                        &mut prev_dc_y,
                        &mut prev_dc_cb,
                        &mut prev_dc_cr,
                        mcu_count,
                        total_mcus,
                        &mut rst_idx,
                    );
                }
            }
        }
        (_, Subsampling::S420) => {
            let padded_width_420 = (ctx.width + 15) & !15;
            let padded_height_420 = (ctx.height + 15) & !15;
            let total_mcus = ((padded_width_420 / 16) * (padded_height_420 / 16)) as u32;

            for mcu_y in (0..padded_height_420).step_by(16) {
                for mcu_x in (0..padded_width_420).step_by(16) {
                    let (y_blocks, cb_block, cr_block) =
                        extract_mcu_420(ctx.data, ctx.width, ctx.height, mcu_x, mcu_y);

                    for y_block in &y_blocks {
                        let y_dct = dct_2d(y_block);
                        let y_quant = quantize_block(&y_dct, &ctx.quant_tables.luminance_table);
                        prev_dc_y =
                            encode_block(&mut writer, &y_quant, prev_dc_y, true, ctx.huff_tables);
                    }

                    let cb_dct = dct_2d(&cb_block);
                    let cb_quant = quantize_block(&cb_dct, &ctx.quant_tables.chrominance_table);
                    prev_dc_cb =
                        encode_block(&mut writer, &cb_quant, prev_dc_cb, false, ctx.huff_tables);

                    let cr_dct = dct_2d(&cr_block);
                    let cr_quant = quantize_block(&cr_dct, &ctx.quant_tables.chrominance_table);
                    prev_dc_cr =
                        encode_block(&mut writer, &cr_quant, prev_dc_cr, false, ctx.huff_tables);

                    mcu_count += 1;
                    handle_restart(
                        &mut writer,
                        &mut prev_dc_y,
                        &mut prev_dc_cb,
                        &mut prev_dc_cr,
                        mcu_count,
                        total_mcus,
                        &mut rst_idx,
                    );
                }
            }
        }
    }

    // Flush the bit writer and append to output
    output.extend_from_slice(&writer.finish());
}

fn extract_block(
    data: &[u8],
    width: usize,
    height: usize,
    block_x: usize,
    block_y: usize,
    color_type: ColorType,
) -> ([f32; 64], [f32; 64], [f32; 64]) {
    let mut y_block = [0.0f32; 64];
    let mut cb_block = [0.0f32; 64];
    let mut cr_block = [0.0f32; 64];

    for dy in 0..8 {
        for dx in 0..8 {
            let x = (block_x + dx).min(width - 1);
            let y = (block_y + dy).min(height - 1);
            let idx = dy * 8 + dx;

            match color_type {
                ColorType::Gray => {
                    let gray = data[y * width + x];
                    y_block[idx] = gray as f32 - 128.0;
                    cb_block[idx] = 0.0;
                    cr_block[idx] = 0.0;
                }
                ColorType::Rgb => {
                    let pixel_idx = (y * width + x) * 3;
                    let r = data[pixel_idx];
                    let g = data[pixel_idx + 1];
                    let b = data[pixel_idx + 2];
                    let (yc, cb, cr) = rgb_to_ycbcr(r, g, b);
                    y_block[idx] = yc as f32 - 128.0;
                    cb_block[idx] = cb as f32 - 128.0;
                    cr_block[idx] = cr as f32 - 128.0;
                }
                _ => unreachable!(),
            }
        }
    }

    (y_block, cb_block, cr_block)
}

fn extract_mcu_420(
    data: &[u8],
    width: usize,
    height: usize,
    mcu_x: usize,
    mcu_y: usize,
) -> ([[f32; 64]; 4], [f32; 64], [f32; 64]) {
    let mut y_blocks = [[0.0f32; 64]; 4];
    let mut cb_block = [0.0f32; 64];
    let mut cr_block = [0.0f32; 64];

    // Populate Y blocks and accumulate chroma
    for by in 0..2 {
        for bx in 0..2 {
            let block_idx = by * 2 + bx;
            for dy in 0..8 {
                for dx in 0..8 {
                    let x = (mcu_x + bx * 8 + dx).min(width - 1);
                    let y = (mcu_y + by * 8 + dy).min(height - 1);
                    let pixel_idx = (y * width + x) * 3;
                    let r = data[pixel_idx];
                    let g = data[pixel_idx + 1];
                    let b = data[pixel_idx + 2];
                    let (yc, cb, cr) = rgb_to_ycbcr(r, g, b);
                    let idx = dy * 8 + dx;
                    y_blocks[block_idx][idx] = yc as f32 - 128.0;

                    // Calculate global position within the 16x16 MCU
                    let global_x = bx * 8 + dx;
                    let global_y = by * 8 + dy;
                    // Chroma is subsampled 2:1 in each dimension
                    let cx = global_x / 2;
                    let cy = global_y / 2;
                    let cidx = cy * 8 + cx;
                    cb_block[cidx] += cb as f32;
                    cr_block[cidx] += cr as f32;
                }
            }
        }
    }

    // Average chroma over 2x2
    for c in 0..64 {
        cb_block[c] = cb_block[c] * 0.25 - 128.0;
        cr_block[c] = cr_block[c] * 0.25 - 128.0;
    }

    (y_blocks, cb_block, cr_block)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_encode(data: &[u8], width: u32, height: u32, quality: u8) -> crate::Result<Vec<u8>> {
        let options = JpegOptions::builder(width, height)
            .color_type(ColorType::Rgb)
            .quality(quality)
            .build();
        encode(data, &options)
    }

    fn test_encode_with_color(
        data: &[u8],
        width: u32,
        height: u32,
        quality: u8,
        color_type: ColorType,
    ) -> crate::Result<Vec<u8>> {
        let options = JpegOptions::builder(width, height)
            .color_type(color_type)
            .quality(quality)
            .build();
        encode(data, &options)
    }

    #[allow(dead_code)]
    fn test_encode_with_options(
        data: &[u8],
        width: u32,
        height: u32,
        color_type: ColorType,
        options: &JpegOptions,
    ) -> crate::Result<Vec<u8>> {
        let mut opts = *options;
        opts.width = width;
        opts.height = height;
        opts.color_type = color_type;
        encode(data, &opts)
    }

    fn test_encode_into(
        output: &mut Vec<u8>,
        data: &[u8],
        width: u32,
        height: u32,
        color_type: ColorType,
        options: &JpegOptions,
    ) -> crate::Result<()> {
        let mut opts = *options;
        opts.width = width;
        opts.height = height;
        opts.color_type = color_type;
        encode_into(output, data, &opts)
    }

    #[test]
    fn test_encode_1x1_rgb() {
        let pixels = vec![255, 0, 0]; // Red pixel
        let jpeg = test_encode(&pixels, 1, 1, 85).unwrap();

        // Check JPEG markers
        assert_eq!(&jpeg[0..2], &SOI.to_be_bytes());
        assert_eq!(&jpeg[jpeg.len() - 2..], &EOI.to_be_bytes());
    }

    #[test]
    fn test_encode_8x8_rgb() {
        // 8x8 gradient
        let mut pixels = Vec::with_capacity(8 * 8 * 3);
        for y in 0..8 {
            for x in 0..8 {
                let val = ((x + y) * 16) as u8;
                pixels.extend_from_slice(&[val, val, val]);
            }
        }

        let jpeg = test_encode(&pixels, 8, 8, 85).unwrap();
        assert_eq!(&jpeg[0..2], &SOI.to_be_bytes());
    }

    #[test]
    fn test_encode_invalid_quality() {
        let pixels = vec![255, 0, 0];
        assert!(matches!(
            test_encode(&pixels, 1, 1, 0),
            Err(Error::InvalidQuality(0))
        ));
        assert!(matches!(
            test_encode(&pixels, 1, 1, 101),
            Err(Error::InvalidQuality(101))
        ));
    }

    #[test]
    fn test_encode_invalid_restart_interval() {
        let pixels = vec![255, 0, 0];
        let mut options = JpegOptions::fast(1, 1, 85);
        options.restart_interval = Some(0);
        let mut output = Vec::new();
        assert!(matches!(
            test_encode_into(&mut output, &pixels, 1, 1, ColorType::Rgb, &options),
            Err(Error::InvalidRestartInterval(0))
        ));
    }

    #[test]
    fn test_encode_invalid_dimensions() {
        let pixels = vec![255, 0, 0];
        assert!(matches!(
            test_encode(&pixels, 0, 1, 85),
            Err(Error::InvalidDimensions { .. })
        ));
    }

    #[test]
    fn test_encode_grayscale() {
        let pixels = vec![128; 64]; // 8x8 gray
        let jpeg = test_encode_with_color(&pixels, 8, 8, 85, ColorType::Gray).unwrap();
        assert_eq!(&jpeg[0..2], &SOI.to_be_bytes());
    }

    #[test]
    fn test_encode_with_options_into_reuses_buffer() {
        let mut output = Vec::with_capacity(256);
        let pixels1 = vec![0u8; 3]; // 1x1 black
        let opts = JpegOptions::fast(1, 1, 85);

        test_encode_into(&mut output, &pixels1, 1, 1, ColorType::Rgb, &opts).unwrap();
        let first = output.clone();
        let first_cap = output.capacity();
        assert!(!first.is_empty());

        let pixels2 = vec![255u8, 0, 0]; // 1x1 red
        test_encode_into(&mut output, &pixels2, 1, 1, ColorType::Rgb, &opts).unwrap();

        assert_ne!(first, output);
        assert!(output.capacity() >= first_cap);
        assert_eq!(&output[0..2], &SOI.to_be_bytes());
    }

    #[test]
    fn test_jpeg_options_builder_with_preset_override() {
        let opts = JpegOptions::builder(1, 1)
            .preset(2) // max
            .quality(90)
            .subsampling(Subsampling::S444)
            .optimize_huffman(false)
            .progressive(false)
            .trellis_quant(false)
            .build();

        assert_eq!(opts.quality, 90);
        assert_eq!(opts.subsampling, Subsampling::S444);
        assert!(!opts.optimize_huffman);
        assert!(!opts.progressive);
        assert!(!opts.trellis_quant);
    }

    // ==========================================================================
    // Extended JPEG Tests for Coverage
    // ==========================================================================

    #[test]
    fn test_encode_progressive() {
        let pixels = vec![128u8; 16 * 16 * 3];
        let opts = JpegOptions::builder(16, 16)
            .color_type(ColorType::Rgb)
            .quality(85)
            .subsampling(Subsampling::S420)
            .progressive(true)
            .build();
        let mut output = Vec::new();
        encode_into(&mut output, &pixels, &opts).unwrap();

        // Should have valid JPEG structure
        assert_eq!(&output[0..2], &SOI.to_be_bytes());
        assert_eq!(&output[output.len() - 2..], &EOI.to_be_bytes());
    }

    #[test]
    fn test_encode_progressive_with_trellis() {
        let pixels: Vec<u8> = (0..16 * 16 * 3).map(|i| (i * 7 % 256) as u8).collect();
        let opts = JpegOptions::max(1, 1, 85);
        let mut output = Vec::new();
        test_encode_into(&mut output, &pixels, 16, 16, ColorType::Rgb, &opts).unwrap();

        assert_eq!(&output[0..2], &SOI.to_be_bytes());
    }

    #[test]
    fn test_encode_with_restart_interval() {
        let pixels = vec![100u8; 32 * 32 * 3];
        let opts = JpegOptions::builder(32, 32)
            .color_type(ColorType::Rgb)
            .quality(80)
            .subsampling(Subsampling::S444)
            .restart_interval(Some(10))
            .build();
        let mut output = Vec::new();
        encode_into(&mut output, &pixels, &opts).unwrap();

        assert_eq!(&output[0..2], &SOI.to_be_bytes());
    }

    #[test]
    fn test_encode_various_subsampling() {
        let pixels = vec![128u8; 16 * 16 * 3];

        for subsampling in [Subsampling::S444, Subsampling::S420] {
            let opts = JpegOptions::builder(16, 16)
                .color_type(ColorType::Rgb)
                .quality(85)
                .subsampling(subsampling)
                .build();
            let mut output = Vec::new();
            encode_into(&mut output, &pixels, &opts).unwrap();
            assert_eq!(&output[0..2], &SOI.to_be_bytes());
        }
    }

    #[test]
    fn test_encode_with_optimized_huffman() {
        let pixels: Vec<u8> = (0..16 * 16 * 3).map(|i| (i * 13 % 256) as u8).collect();
        let opts = JpegOptions::balanced(1, 1, 85);
        let mut output = Vec::new();
        test_encode_into(&mut output, &pixels, 16, 16, ColorType::Rgb, &opts).unwrap();

        assert_eq!(&output[0..2], &SOI.to_be_bytes());
    }

    #[test]
    fn test_encode_rgba_unsupported() {
        let pixels = vec![128u8; 8 * 8 * 4]; // RGBA
        let result = test_encode_with_color(&pixels, 8, 8, 85, ColorType::Rgba);
        assert!(matches!(result, Err(Error::UnsupportedColorType)));
    }

    #[test]
    fn test_encode_gray_alpha_unsupported() {
        let pixels = vec![128u8; 8 * 8 * 2]; // GrayAlpha
        let result = test_encode_with_color(&pixels, 8, 8, 85, ColorType::GrayAlpha);
        assert!(matches!(result, Err(Error::UnsupportedColorType)));
    }

    #[test]
    fn test_encode_various_quality_levels() {
        let pixels = vec![128u8; 8 * 8 * 3];

        for quality in [1, 25, 50, 75, 100] {
            let jpeg = test_encode(&pixels, 8, 8, quality).unwrap();
            assert_eq!(&jpeg[0..2], &SOI.to_be_bytes());
            assert_eq!(&jpeg[jpeg.len() - 2..], &EOI.to_be_bytes());
        }
    }

    #[test]
    fn test_encode_non_multiple_of_8_dimensions() {
        // Dimensions that require padding
        let widths = [5, 7, 9, 15, 17];
        let heights = [3, 6, 11, 13, 19];

        for &w in &widths {
            for &h in &heights {
                let pixels = vec![100u8; (w * h * 3) as usize];
                let jpeg = test_encode(&pixels, w, h, 85).unwrap();
                assert_eq!(&jpeg[0..2], &SOI.to_be_bytes());
            }
        }
    }

    #[test]
    fn test_jpeg_options_fast() {
        let opts = JpegOptions::fast(100, 100, 75);
        assert_eq!(opts.quality, 75);
        assert_eq!(opts.subsampling, Subsampling::S444);
        assert!(!opts.optimize_huffman);
        assert!(!opts.progressive);
        assert!(!opts.trellis_quant);
    }

    #[test]
    fn test_jpeg_options_balanced() {
        let opts = JpegOptions::balanced(100, 100, 80);
        assert_eq!(opts.quality, 80);
        assert_eq!(opts.subsampling, Subsampling::S444);
        assert!(opts.optimize_huffman);
        assert!(!opts.progressive);
        assert!(!opts.trellis_quant);
    }

    #[test]
    fn test_jpeg_options_max() {
        let opts = JpegOptions::max(100, 100, 90);
        assert_eq!(opts.quality, 90);
        assert_eq!(opts.subsampling, Subsampling::S420);
        assert!(opts.optimize_huffman);
        assert!(opts.progressive);
        assert!(opts.trellis_quant);
    }

    #[test]
    fn test_jpeg_options_from_preset() {
        // from_preset(width, height, quality, preset)
        let fast = JpegOptions::from_preset(100, 100, 70, 0);
        assert_eq!(fast.quality, 70);
        assert!(!fast.progressive);

        let balanced = JpegOptions::from_preset(100, 100, 80, 1);
        assert_eq!(balanced.quality, 80);
        assert!(balanced.optimize_huffman);

        let max = JpegOptions::from_preset(100, 100, 90, 2);
        assert_eq!(max.quality, 90);
        assert!(max.progressive);
    }

    #[test]
    fn test_jpeg_options_builder_all_options() {
        let opts = JpegOptions::builder(1, 1)
            .quality(95)
            .subsampling(Subsampling::S444)
            .optimize_huffman(true)
            .progressive(true)
            .trellis_quant(true)
            .restart_interval(Some(8))
            .build();

        assert_eq!(opts.quality, 95);
        assert_eq!(opts.subsampling, Subsampling::S444);
        assert!(opts.optimize_huffman);
        assert!(opts.progressive);
        assert!(opts.trellis_quant);
        assert_eq!(opts.restart_interval, Some(8));
    }

    #[test]
    fn test_encode_image_too_large() {
        // Test that very large dimensions are rejected
        let width = (1 << 24) + 1;
        let height = 1;
        let err = test_encode(&[], width, height, 85).unwrap_err();
        assert!(matches!(err, Error::ImageTooLarge { .. }));
    }

    #[test]
    fn test_encode_invalid_data_length() {
        // Wrong number of bytes for dimensions
        let pixels = vec![0u8; 10]; // Not enough for 4x4 RGB
        let err = test_encode(&pixels, 4, 4, 85).unwrap_err();
        assert!(matches!(err, Error::InvalidDataLength { .. }));
    }

    #[cfg(target_pointer_width = "32")]
    #[test]
    fn test_encode_overflow_data_length() {
        let pixels = Vec::new();
        let err = test_encode(&pixels, MAX_DIMENSION, MAX_DIMENSION, 85).unwrap_err();
        assert!(matches!(err, Error::InvalidDataLength { .. }));
    }

    // ============================================================================
    // Grayscale Encoding Tests
    // ============================================================================

    #[test]
    fn test_encode_grayscale_baseline_large() {
        // Grayscale image (1 byte per pixel) with larger size
        let pixels: Vec<u8> = (0..64 * 64).map(|i| (i % 256) as u8).collect();

        let opts = JpegOptions::builder(64, 64)
            .color_type(ColorType::Gray)
            .quality(85)
            .build();

        let mut output = Vec::new();
        encode_into(&mut output, &pixels, &opts).unwrap();

        // Should produce valid JPEG
        assert_eq!(&output[0..2], &[0xFF, 0xD8]); // SOI
        assert_eq!(&output[output.len() - 2..], &[0xFF, 0xD9]); // EOI
    }

    #[test]
    fn test_encode_grayscale_progressive() {
        // Grayscale image with progressive encoding
        let pixels: Vec<u8> = (0..64 * 64).map(|i| (i % 256) as u8).collect();

        let opts = JpegOptions::builder(64, 64)
            .color_type(ColorType::Gray)
            .quality(85)
            .progressive(true)
            .build();

        let mut output = Vec::new();
        encode_into(&mut output, &pixels, &opts).unwrap();

        // Should produce valid JPEG
        assert_eq!(&output[0..2], &[0xFF, 0xD8]); // SOI
        assert_eq!(&output[output.len() - 2..], &[0xFF, 0xD9]); // EOI
    }

    #[test]
    fn test_encode_grayscale_optimized_huffman() {
        // Grayscale image with optimized Huffman tables
        let pixels: Vec<u8> = (0..64 * 64).map(|i| (i % 256) as u8).collect();

        let opts = JpegOptions::builder(64, 64)
            .color_type(ColorType::Gray)
            .quality(85)
            .optimize_huffman(true)
            .build();

        let mut output = Vec::new();
        encode_into(&mut output, &pixels, &opts).unwrap();

        // Should produce valid JPEG
        assert_eq!(&output[0..2], &[0xFF, 0xD8]); // SOI
        assert_eq!(&output[output.len() - 2..], &[0xFF, 0xD9]); // EOI
    }

    // ============================================================================
    // 4:2:0 Progressive with Trellis Tests
    // ============================================================================

    #[test]
    fn test_encode_420_progressive_trellis() {
        // RGB image with 4:2:0 subsampling, progressive, and trellis quantization
        let pixels: Vec<u8> = (0..64 * 64 * 3).map(|i| (i * 7 % 256) as u8).collect();

        let opts = JpegOptions::builder(64, 64)
            .color_type(ColorType::Rgb)
            .quality(85)
            .subsampling(Subsampling::S420)
            .progressive(true)
            .trellis_quant(true)
            .build();

        let mut output = Vec::new();
        encode_into(&mut output, &pixels, &opts).unwrap();

        // Should produce valid JPEG
        assert_eq!(&output[0..2], &[0xFF, 0xD8]); // SOI
        assert_eq!(&output[output.len() - 2..], &[0xFF, 0xD9]); // EOI
    }

    // ============================================================================
    // Optimized Huffman with Restart Intervals Tests
    // ============================================================================

    #[test]
    fn test_encode_optimized_huffman_with_restart() {
        // RGB image with optimized Huffman and restart intervals
        let pixels: Vec<u8> = (0..64 * 64 * 3).map(|i| (i * 11 % 256) as u8).collect();

        let opts = JpegOptions::builder(64, 64)
            .color_type(ColorType::Rgb)
            .quality(85)
            .optimize_huffman(true)
            .restart_interval(Some(8))
            .build();

        let mut output = Vec::new();
        encode_into(&mut output, &pixels, &opts).unwrap();

        // Should produce valid JPEG
        assert_eq!(&output[0..2], &[0xFF, 0xD8]); // SOI
        assert_eq!(&output[output.len() - 2..], &[0xFF, 0xD9]); // EOI

        // Should contain DRI marker (0xFF, 0xDD) for restart interval
        assert!(
            output.windows(2).any(|w| w == [0xFF, 0xDD]),
            "should contain DRI marker for restart interval"
        );
    }

    #[test]
    fn test_encode_optimized_huffman_progressive_with_restart() {
        // Progressive with optimized Huffman and restart intervals
        let pixels: Vec<u8> = (0..64 * 64 * 3).map(|i| (i * 13 % 256) as u8).collect();

        let opts = JpegOptions::builder(64, 64)
            .color_type(ColorType::Rgb)
            .quality(85)
            .optimize_huffman(true)
            .restart_interval(Some(4))
            .progressive(true)
            .build();

        let mut output = Vec::new();
        encode_into(&mut output, &pixels, &opts).unwrap();

        // Should produce valid JPEG
        assert_eq!(&output[0..2], &[0xFF, 0xD8]); // SOI
        assert_eq!(&output[output.len() - 2..], &[0xFF, 0xD9]); // EOI
    }
}
