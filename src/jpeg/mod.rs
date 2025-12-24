//! JPEG encoder implementation.
//!
//! Implements baseline and progressive JPEG encoding (DCT-based lossy compression).
//! Supports:
//! - Baseline sequential DCT (SOF0)
//! - Progressive DCT (SOF2) with spectral selection and successive approximation
//! - Integer and floating-point DCT
//! - Optimized Huffman tables
//! - Trellis quantization for better R-D optimization

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
    encode_ac_first, encode_dc_refine, get_dc_code, simple_progressive_script, ScanSpec,
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

/// Encode raw pixel data as JPEG.
///
/// # Arguments
/// * `data` - Raw pixel data (RGB, row-major order)
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `quality` - Quality level 1-100 (higher = better quality, larger file)
///
/// # Returns
/// Complete JPEG file as bytes.
pub fn encode(data: &[u8], width: u32, height: u32, quality: u8) -> Result<Vec<u8>> {
    let options = JpegOptions::fast(quality);
    let mut output = Vec::new();
    encode_with_options_into(&mut output, data, width, height, ColorType::Rgb, &options)?;
    Ok(output)
}

/// Encode raw pixel data as JPEG with specified color type.
pub fn encode_with_color(
    data: &[u8],
    width: u32,
    height: u32,
    quality: u8,
    color_type: ColorType,
) -> Result<Vec<u8>> {
    let options = JpegOptions::fast(quality);
    let mut output = Vec::new();
    encode_with_options_into(&mut output, data, width, height, color_type, &options)?;
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
#[derive(Debug, Clone, Copy)]
pub struct JpegOptions {
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
    pub fn fast(quality: u8) -> Self {
        Self {
            quality,
            subsampling: Subsampling::S444,
            restart_interval: None,
            optimize_huffman: false,
            progressive: false,
            trellis_quant: false,
        }
    }

    /// Preset 1: Balanced - optimized Huffman, 4:4:4, baseline (good balance).
    pub fn balanced(quality: u8) -> Self {
        Self {
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
    pub fn max(quality: u8) -> Self {
        Self {
            quality,
            subsampling: Subsampling::S420,
            restart_interval: None,
            optimize_huffman: true,
            progressive: true,
            trellis_quant: true,
        }
    }

    /// Create from preset (0=fast, 1=balanced, 2=max).
    pub fn from_preset(quality: u8, preset: u8) -> Self {
        match preset {
            0 => Self::fast(quality),
            2 => Self::max(quality),
            _ => Self::balanced(quality),
        }
    }
}

/// Encode raw pixel data as JPEG with options.
///
/// # Arguments
/// * `data` - Raw pixel data (RGB or Gray, row-major order)
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `color_type` - Color type (Rgb or Gray)
/// * `options` - JPEG encoding options (includes quality)
///
/// # Returns
/// Complete JPEG file as bytes.
pub fn encode_with_options(
    data: &[u8],
    width: u32,
    height: u32,
    color_type: ColorType,
    options: &JpegOptions,
) -> Result<Vec<u8>> {
    let mut output = Vec::new();
    encode_with_options_into(&mut output, data, width, height, color_type, options)?;
    Ok(output)
}

/// Encode raw pixel data as JPEG with options into a caller-provided buffer.
///
/// The `output` buffer will be cleared and reused, allowing callers to avoid
/// repeated allocations across multiple encodes.
///
/// # Arguments
/// * `output` - Buffer to write JPEG data into (will be cleared)
/// * `data` - Raw pixel data (RGB or Gray, row-major order)
/// * `width` - Image width in pixels
/// * `height` - Image height in pixels
/// * `color_type` - Color type (Rgb or Gray)
/// * `options` - JPEG encoding options (includes quality)
pub fn encode_with_options_into(
    output: &mut Vec<u8>,
    data: &[u8],
    width: u32,
    height: u32,
    color_type: ColorType,
    options: &JpegOptions,
) -> Result<()> {
    // Validate quality
    if options.quality == 0 || options.quality > 100 {
        return Err(Error::InvalidQuality(options.quality));
    }
    if matches!(options.restart_interval, Some(0)) {
        return Err(Error::InvalidQuality(0)); // reuse quality error type for invalid param
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
    let expected_len = width as usize * height as usize * bytes_per_pixel;
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
        encode_progressive(
            output,
            data,
            width,
            height,
            color_type,
            options.subsampling,
            &quant_tables,
            &huff_tables,
            options.trellis_quant,
        );
    } else {
        // Baseline JPEG encoding
        write_sof0(output, width, height, color_type, options.subsampling);
        write_dht(output, &huff_tables);
        if let Some(interval) = options.restart_interval {
            write_dri(output, interval);
        }

        // Write scan data
        write_sos(output, color_type);
        encode_scan(
            output,
            data,
            width,
            height,
            color_type,
            options.restart_interval,
            options.subsampling,
            &quant_tables,
            &huff_tables,
        );
    }

    // Write end marker
    write_eoi(output);

    Ok(())
}

/// Write SOI (Start of Image) marker.
fn write_soi(output: &mut Vec<u8>) {
    output.extend_from_slice(&SOI.to_be_bytes());
}

/// Write EOI (End of Image) marker.
fn write_eoi(output: &mut Vec<u8>) {
    output.extend_from_slice(&EOI.to_be_bytes());
}

/// Write APP0 (JFIF) marker.
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

/// Write DQT (Define Quantization Table) marker.
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

/// Write SOF0 (Start of Frame - baseline) marker.
fn write_sof0(
    output: &mut Vec<u8>,
    width: u32,
    height: u32,
    color_type: ColorType,
    subsampling: Subsampling,
) {
    write_sof_marker(output, SOF0, width, height, color_type, subsampling);
}

/// Write SOF2 (Start of Frame - progressive) marker.
fn write_sof2(
    output: &mut Vec<u8>,
    width: u32,
    height: u32,
    color_type: ColorType,
    subsampling: Subsampling,
) {
    write_sof_marker(output, SOF2, width, height, color_type, subsampling);
}

/// Write SOF marker (shared implementation for baseline and progressive).
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

/// Write DHT (Define Huffman Table) marker.
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

/// Write DRI (Define Restart Interval) marker.
fn write_dri(output: &mut Vec<u8>, interval: u16) {
    output.extend_from_slice(&0xFFDDu16.to_be_bytes());
    output.extend_from_slice(&4u16.to_be_bytes()); // length = 4
    output.extend_from_slice(&interval.to_be_bytes());
}

/// Write a single Huffman table.
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

/// Write SOS (Start of Scan) marker for baseline JPEG.
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

/// Write SOS marker for progressive JPEG scan.
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

/// Build optimized Huffman tables by analyzing the image's quantized coefficients.
fn build_optimized_huffman_tables(
    data: &[u8],
    width: u32,
    height: u32,
    color_type: ColorType,
    subsampling: Subsampling,
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
            let mut prev_dc_y = 0i16;
            for block_y in (0..padded_height).step_by(8) {
                for block_x in (0..padded_width).step_by(8) {
                    let (y_block, _, _) =
                        extract_block(data, width, height, block_x, block_y, color_type);
                    let y_dct = dct_2d(&y_block);
                    let y_quant = quantize_block(&y_dct, &quant_tables.luminance_table);
                    prev_dc_y = count_block(&y_quant, prev_dc_y, true, &mut dc_lum, &mut ac_lum);
                }
            }
            HuffmanTables::optimized_from_counts(&dc_lum, None, &ac_lum, None)
        }
        (_, Subsampling::S444) => {
            let padded_width = (width + 7) & !7;
            let padded_height = (height + 7) & !7;
            let mut prev_dc_y = 0i16;
            let mut prev_dc_cb = 0i16;
            let mut prev_dc_cr = 0i16;

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
                }
            }

            HuffmanTables::optimized_from_counts(&dc_lum, Some(&dc_chrom), &ac_lum, Some(&ac_chrom))
        }
        (_, Subsampling::S420) => {
            let padded_width_420 = (width + 15) & !15;
            let padded_height_420 = (height + 15) & !15;
            let mut prev_dc_y = 0i16;
            let mut prev_dc_cb = 0i16;
            let mut prev_dc_cr = 0i16;

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

/// Encode image using progressive JPEG (multiple scans).
#[allow(clippy::too_many_arguments)]
fn encode_progressive(
    output: &mut Vec<u8>,
    data: &[u8],
    width: u32,
    height: u32,
    color_type: ColorType,
    subsampling: Subsampling,
    quant_tables: &QuantizationTables,
    huff_tables: &HuffmanTables,
    use_trellis: bool,
) {
    let width = width as usize;
    let height = height as usize;

    // Step 1: Compute all DCT coefficients and store them
    let (y_coeffs, cb_coeffs, cr_coeffs) = compute_all_coefficients(
        data,
        width,
        height,
        color_type,
        subsampling,
        quant_tables,
        use_trellis,
    );

    // Step 2: Get progressive scan script
    let script = simple_progressive_script();

    // Step 3: Encode each scan
    for scan in &script {
        write_sos_progressive(output, scan, color_type);

        let mut writer = BitWriterMsb::new();

        if scan.is_dc_scan() {
            encode_dc_scan(
                &mut writer,
                scan,
                &y_coeffs,
                &cb_coeffs,
                &cr_coeffs,
                subsampling,
                huff_tables,
            );
        } else if scan.is_first_scan() {
            encode_ac_first_scan(
                &mut writer,
                scan,
                &y_coeffs,
                &cb_coeffs,
                &cr_coeffs,
                subsampling,
                huff_tables,
            );
        } else {
            encode_ac_refine_scan(
                &mut writer,
                scan,
                &y_coeffs,
                &cb_coeffs,
                &cr_coeffs,
                subsampling,
                huff_tables,
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
) -> (Vec<[i16; 64]>, Vec<[i16; 64]>, Vec<[i16; 64]>) {
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
) -> (Vec<[i16; 64]>, Vec<[i16; 64]>, Vec<[i16; 64]>) {
    let mut y_coeffs = Vec::new();
    let mut cb_coeffs = Vec::new();
    let mut cr_coeffs = Vec::new();

    // Helper to quantize a block with optional trellis
    let quantize_with_trellis = |dct: &[f32; 64], table: &[f32; 64]| -> [i16; 64] {
        if use_trellis {
            trellis::trellis_quantize(dct, table, None)
        } else {
            quantize_block(dct, table)
        }
    };

    match (color_type, subsampling) {
        (ColorType::Gray, _) => {
            let padded_width = (width + 7) & !7;
            let padded_height = (height + 7) & !7;

            for block_y in (0..padded_height).step_by(8) {
                for block_x in (0..padded_width).step_by(8) {
                    let (y_block, _, _) =
                        extract_block(data, width, height, block_x, block_y, color_type);
                    let y_dct = dct_2d(&y_block);
                    let y_quant = quantize_with_trellis(&y_dct, &quant_tables.luminance_table);
                    y_coeffs.push(y_quant);
                }
            }
        }
        (_, Subsampling::S444) => {
            let padded_width = (width + 7) & !7;
            let padded_height = (height + 7) & !7;

            for block_y in (0..padded_height).step_by(8) {
                for block_x in (0..padded_width).step_by(8) {
                    let (y_block, cb_block, cr_block) =
                        extract_block(data, width, height, block_x, block_y, color_type);

                    let y_quant =
                        quantize_with_trellis(&dct_2d(&y_block), &quant_tables.luminance_table);
                    y_coeffs.push(y_quant);

                    let cb_quant =
                        quantize_with_trellis(&dct_2d(&cb_block), &quant_tables.chrominance_table);
                    cb_coeffs.push(cb_quant);

                    let cr_quant =
                        quantize_with_trellis(&dct_2d(&cr_block), &quant_tables.chrominance_table);
                    cr_coeffs.push(cr_quant);
                }
            }
        }
        (_, Subsampling::S420) => {
            let padded_width_420 = (width + 15) & !15;
            let padded_height_420 = (height + 15) & !15;

            for mcu_y in (0..padded_height_420).step_by(16) {
                for mcu_x in (0..padded_width_420).step_by(16) {
                    let (y_blocks, cb_block, cr_block) =
                        extract_mcu_420(data, width, height, mcu_x, mcu_y);

                    for y_block in &y_blocks {
                        let y_quant =
                            quantize_with_trellis(&dct_2d(y_block), &quant_tables.luminance_table);
                        y_coeffs.push(y_quant);
                    }

                    let cb_quant =
                        quantize_with_trellis(&dct_2d(&cb_block), &quant_tables.chrominance_table);
                    cb_coeffs.push(cb_quant);

                    let cr_quant =
                        quantize_with_trellis(&dct_2d(&cr_block), &quant_tables.chrominance_table);
                    cr_coeffs.push(cr_quant);
                }
            }
        }
    }

    (y_coeffs, cb_coeffs, cr_coeffs)
}

/// Parallel implementation of coefficient computation using Rayon.
/// Processes blocks in parallel for significant speedup on multi-core systems.
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
) -> (Vec<[i16; 64]>, Vec<[i16; 64]>, Vec<[i16; 64]>) {
    use rayon::prelude::*;

    /// Helper struct to hold block coordinates
    struct BlockCoord {
        x: usize,
        y: usize,
    }

    // Helper to quantize a block with optional trellis
    let quantize_with_trellis = |dct: &[f32; 64], table: &[f32; 64]| -> [i16; 64] {
        if use_trellis {
            trellis::trellis_quantize(dct, table, None)
        } else {
            quantize_block(dct, table)
        }
    };

    match (color_type, subsampling) {
        (ColorType::Gray, _) => {
            let padded_width = (width + 7) & !7;
            let padded_height = (height + 7) & !7;

            // Collect block coordinates
            let coords: Vec<BlockCoord> = (0..padded_height)
                .step_by(8)
                .flat_map(|y| {
                    (0..padded_width)
                        .step_by(8)
                        .map(move |x| BlockCoord { x, y })
                })
                .collect();

            // Process blocks in parallel
            let y_coeffs: Vec<[i16; 64]> = coords
                .par_iter()
                .map(|coord| {
                    let (y_block, _, _) =
                        extract_block(data, width, height, coord.x, coord.y, color_type);
                    let y_dct = dct_2d(&y_block);
                    quantize_with_trellis(&y_dct, &quant_tables.luminance_table)
                })
                .collect();

            (y_coeffs, Vec::new(), Vec::new())
        }
        (_, Subsampling::S444) => {
            let padded_width = (width + 7) & !7;
            let padded_height = (height + 7) & !7;

            // Collect block coordinates
            let coords: Vec<BlockCoord> = (0..padded_height)
                .step_by(8)
                .flat_map(|y| {
                    (0..padded_width)
                        .step_by(8)
                        .map(move |x| BlockCoord { x, y })
                })
                .collect();

            // Process blocks in parallel, collecting all three channels
            let results: Vec<([i16; 64], [i16; 64], [i16; 64])> = coords
                .par_iter()
                .map(|coord| {
                    let (y_block, cb_block, cr_block) =
                        extract_block(data, width, height, coord.x, coord.y, color_type);

                    let y_quant =
                        quantize_with_trellis(&dct_2d(&y_block), &quant_tables.luminance_table);
                    let cb_quant =
                        quantize_with_trellis(&dct_2d(&cb_block), &quant_tables.chrominance_table);
                    let cr_quant =
                        quantize_with_trellis(&dct_2d(&cr_block), &quant_tables.chrominance_table);

                    (y_quant, cb_quant, cr_quant)
                })
                .collect();

            // Unzip results
            let mut y_coeffs = Vec::with_capacity(results.len());
            let mut cb_coeffs = Vec::with_capacity(results.len());
            let mut cr_coeffs = Vec::with_capacity(results.len());

            for (y, cb, cr) in results {
                y_coeffs.push(y);
                cb_coeffs.push(cb);
                cr_coeffs.push(cr);
            }

            (y_coeffs, cb_coeffs, cr_coeffs)
        }
        (_, Subsampling::S420) => {
            let padded_width_420 = (width + 15) & !15;
            let padded_height_420 = (height + 15) & !15;

            // Collect MCU coordinates
            let coords: Vec<BlockCoord> = (0..padded_height_420)
                .step_by(16)
                .flat_map(|y| {
                    (0..padded_width_420)
                        .step_by(16)
                        .map(move |x| BlockCoord { x, y })
                })
                .collect();

            // Process MCUs in parallel
            // Each MCU produces 4 Y blocks + 1 Cb + 1 Cr
            let results: Vec<([[i16; 64]; 4], [i16; 64], [i16; 64])> = coords
                .par_iter()
                .map(|coord| {
                    let (y_blocks, cb_block, cr_block) =
                        extract_mcu_420(data, width, height, coord.x, coord.y);

                    let mut y_quants = [[0i16; 64]; 4];
                    for (i, y_block) in y_blocks.iter().enumerate() {
                        y_quants[i] =
                            quantize_with_trellis(&dct_2d(y_block), &quant_tables.luminance_table);
                    }

                    let cb_quant =
                        quantize_with_trellis(&dct_2d(&cb_block), &quant_tables.chrominance_table);
                    let cr_quant =
                        quantize_with_trellis(&dct_2d(&cr_block), &quant_tables.chrominance_table);

                    (y_quants, cb_quant, cr_quant)
                })
                .collect();

            // Unzip and flatten results
            let mut y_coeffs = Vec::with_capacity(results.len() * 4);
            let mut cb_coeffs = Vec::with_capacity(results.len());
            let mut cr_coeffs = Vec::with_capacity(results.len());

            for (y_quants, cb, cr) in results {
                for y in y_quants {
                    y_coeffs.push(y);
                }
                cb_coeffs.push(cb);
                cr_coeffs.push(cr);
            }

            (y_coeffs, cb_coeffs, cr_coeffs)
        }
    }
}

/// Encode DC scan for progressive JPEG.
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

/// Encode value bits for DC coefficient.
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

/// Encode first AC scan for progressive JPEG.
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
            encode_ac_first(
                writer,
                block,
                scan.ss,
                scan.se,
                scan.al,
                &mut eob_run,
                huff_tables,
                is_luminance,
            );
        }

        // Flush any remaining EOB run
        if eob_run > 0 {
            progressive::flush_eob_run_public(writer, &mut eob_run, huff_tables, is_luminance);
        }
    }
}

/// Encode AC refinement scan for progressive JPEG.
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
            progressive::encode_ac_refine(
                writer,
                block,
                scan.ss,
                scan.se,
                scan.al,
                &mut eob_run,
                huff_tables,
                is_luminance,
            );
        }

        // Flush any remaining EOB run
        if eob_run > 0 {
            progressive::flush_eob_run_public(writer, &mut eob_run, huff_tables, is_luminance);
        }
    }
}

/// Encode the image scan data.
#[allow(clippy::too_many_arguments)]
fn encode_scan(
    output: &mut Vec<u8>,
    data: &[u8],
    width: u32,
    height: u32,
    color_type: ColorType,
    restart_interval: Option<u16>,
    subsampling: Subsampling,
    quant_tables: &QuantizationTables,
    huff_tables: &HuffmanTables,
) {
    let mut writer = BitWriterMsb::new();

    // Convert to YCbCr and process in 8x8 blocks
    let width = width as usize;
    let height = height as usize;

    // Calculate padded dimensions
    let (padded_width, padded_height) = match subsampling {
        Subsampling::S444 | Subsampling::S420 => ((width + 7) & !7, (height + 7) & !7),
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
            if interval > 0 && mcu_count.is_multiple_of(interval as u32) && mcu_count < total_mcus {
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
    match (color_type, subsampling) {
        (ColorType::Gray, _) => {
            let total_mcus = ((padded_width / 8) * (padded_height / 8)) as u32;
            for block_y in (0..padded_height).step_by(8) {
                for block_x in (0..padded_width).step_by(8) {
                    let (y_block, _, _) =
                        extract_block(data, width, height, block_x, block_y, color_type);
                    let y_dct = dct_2d(&y_block);
                    let y_quant = quantize_block(&y_dct, &quant_tables.luminance_table);
                    prev_dc_y = encode_block(&mut writer, &y_quant, prev_dc_y, true, huff_tables);
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
                    let (y_block, cb_block, cr_block) =
                        extract_block(data, width, height, block_x, block_y, color_type);

                    let y_dct = dct_2d(&y_block);
                    let y_quant = quantize_block(&y_dct, &quant_tables.luminance_table);
                    prev_dc_y = encode_block(&mut writer, &y_quant, prev_dc_y, true, huff_tables);

                    let cb_dct = dct_2d(&cb_block);
                    let cb_quant = quantize_block(&cb_dct, &quant_tables.chrominance_table);
                    prev_dc_cb =
                        encode_block(&mut writer, &cb_quant, prev_dc_cb, false, huff_tables);

                    let cr_dct = dct_2d(&cr_block);
                    let cr_quant = quantize_block(&cr_dct, &quant_tables.chrominance_table);
                    prev_dc_cr =
                        encode_block(&mut writer, &cr_quant, prev_dc_cr, false, huff_tables);

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
            let padded_width_420 = (width + 15) & !15;
            let padded_height_420 = (height + 15) & !15;
            let total_mcus = ((padded_width_420 / 16) * (padded_height_420 / 16)) as u32;

            for mcu_y in (0..padded_height_420).step_by(16) {
                for mcu_x in (0..padded_width_420).step_by(16) {
                    let (y_blocks, cb_block, cr_block) =
                        extract_mcu_420(data, width, height, mcu_x, mcu_y);

                    for y_block in &y_blocks {
                        let y_dct = dct_2d(y_block);
                        let y_quant = quantize_block(&y_dct, &quant_tables.luminance_table);
                        prev_dc_y =
                            encode_block(&mut writer, &y_quant, prev_dc_y, true, huff_tables);
                    }

                    let cb_dct = dct_2d(&cb_block);
                    let cb_quant = quantize_block(&cb_dct, &quant_tables.chrominance_table);
                    prev_dc_cb =
                        encode_block(&mut writer, &cb_quant, prev_dc_cb, false, huff_tables);

                    let cr_dct = dct_2d(&cr_block);
                    let cr_quant = quantize_block(&cr_dct, &quant_tables.chrominance_table);
                    prev_dc_cr =
                        encode_block(&mut writer, &cr_quant, prev_dc_cr, false, huff_tables);

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

/// Extract an 8x8 block from the image and convert to YCbCr.
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

/// Extract a 4:2:0 MCU (16x16 luma -> 4 blocks, 8x8 chroma) starting at (mcu_x, mcu_y).
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

    #[test]
    fn test_encode_1x1_rgb() {
        let pixels = vec![255, 0, 0]; // Red pixel
        let jpeg = encode(&pixels, 1, 1, 85).unwrap();

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

        let jpeg = encode(&pixels, 8, 8, 85).unwrap();
        assert_eq!(&jpeg[0..2], &SOI.to_be_bytes());
    }

    #[test]
    fn test_encode_invalid_quality() {
        let pixels = vec![255, 0, 0];
        assert!(matches!(
            encode(&pixels, 1, 1, 0),
            Err(Error::InvalidQuality(0))
        ));
        assert!(matches!(
            encode(&pixels, 1, 1, 101),
            Err(Error::InvalidQuality(101))
        ));
    }

    #[test]
    fn test_encode_invalid_dimensions() {
        let pixels = vec![255, 0, 0];
        assert!(matches!(
            encode(&pixels, 0, 1, 85),
            Err(Error::InvalidDimensions { .. })
        ));
    }

    #[test]
    fn test_encode_grayscale() {
        let pixels = vec![128; 64]; // 8x8 gray
        let jpeg = encode_with_color(&pixels, 8, 8, 85, ColorType::Gray).unwrap();
        assert_eq!(&jpeg[0..2], &SOI.to_be_bytes());
    }

    #[test]
    fn test_encode_with_options_into_reuses_buffer() {
        let mut output = Vec::with_capacity(256);
        let pixels1 = vec![0u8; 3]; // 1x1 black
        let opts = JpegOptions::fast(85);

        encode_with_options_into(&mut output, &pixels1, 1, 1, ColorType::Rgb, &opts).unwrap();
        let first = output.clone();
        let first_cap = output.capacity();
        assert!(!first.is_empty());

        let pixels2 = vec![255u8, 0, 0]; // 1x1 red
        encode_with_options_into(&mut output, &pixels2, 1, 1, ColorType::Rgb, &opts).unwrap();

        assert_ne!(first, output);
        assert!(output.capacity() >= first_cap);
        assert_eq!(&output[0..2], &SOI.to_be_bytes());
    }
}
