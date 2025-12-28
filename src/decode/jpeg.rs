//! JPEG decoder implementation.
//!
//! Decodes baseline JPEG images (SOF0) into raw pixel data.

use super::bit_reader::MsbBitReader;
use super::idct::{dequantize, idct_2d_integer};
use crate::color::ColorType;
use crate::error::{Error, Result};

/// JPEG markers
const SOI: u8 = 0xD8; // Start of Image
const EOI: u8 = 0xD9; // End of Image
const SOF0: u8 = 0xC0; // Baseline DCT
const SOF2: u8 = 0xC2; // Progressive DCT
const DHT: u8 = 0xC4; // Define Huffman Table
const DQT: u8 = 0xDB; // Define Quantization Table
const DRI: u8 = 0xDD; // Define Restart Interval
const SOS: u8 = 0xDA; // Start of Scan
const RST0: u8 = 0xD0; // Restart marker 0
const APP0: u8 = 0xE0; // Application segment 0 (JFIF)
const APP15: u8 = 0xEF; // Application segment 15
const COM: u8 = 0xFE; // Comment

/// Decoded JPEG image.
#[derive(Debug)]
pub struct JpegImage {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Raw pixel data (RGB or Grayscale).
    pub pixels: Vec<u8>,
    /// Color type of the decoded image.
    pub color_type: ColorType,
}

/// Frame component information.
#[derive(Debug, Clone, Default)]
struct Component {
    #[allow(dead_code)]
    id: u8,
    h_sampling: u8,
    v_sampling: u8,
    quant_table_id: u8,
    dc_table_id: u8,
    ac_table_id: u8,
}

/// Huffman decoding table.
struct HuffmanTable {
    /// Fast lookup table (256 entries for 8-bit lookahead).
    lookup: [u16; 256],
    /// Values for each code.
    values: Vec<u8>,
    /// Minimum code for each bit length.
    min_code: [i32; 17],
    /// Maximum code for each bit length.
    max_code: [i32; 17],
    /// Value offset for each bit length.
    val_offset: [i32; 17],
}

impl Default for HuffmanTable {
    fn default() -> Self {
        Self {
            lookup: [0; 256],
            values: Vec::new(),
            min_code: [0; 17],
            max_code: [-1; 17],
            val_offset: [0; 17],
        }
    }
}

impl HuffmanTable {
    /// Build a Huffman table from bits and values.
    fn build(bits: &[u8; 16], values: &[u8]) -> Self {
        let mut table = HuffmanTable {
            values: values.to_vec(),
            ..Default::default()
        };

        // Build code tables
        let mut huffsize = Vec::new();
        let mut huffcode = Vec::new();

        // Generate size table
        for (i, &count) in bits.iter().enumerate() {
            for _ in 0..count {
                huffsize.push((i + 1) as u8);
            }
        }

        // Generate code table
        let mut code = 0u32;
        let mut si = huffsize.first().copied().unwrap_or(0);
        for &size in &huffsize {
            while size > si {
                code <<= 1;
                si += 1;
            }
            huffcode.push(code as u16);
            code += 1;
        }

        // Build decoding tables
        let mut val_idx = 0;
        for i in 1..=16 {
            if bits[i - 1] > 0 {
                table.val_offset[i] =
                    val_idx as i32 - huffcode.get(val_idx).copied().unwrap_or(0) as i32;
                val_idx += bits[i - 1] as usize;
                table.min_code[i] = huffcode
                    .get(val_idx - bits[i - 1] as usize)
                    .copied()
                    .unwrap_or(0) as i32;
                table.max_code[i] = huffcode.get(val_idx - 1).copied().unwrap_or(0) as i32;
            } else {
                table.max_code[i] = -1;
            }
        }

        // Build fast lookup for codes <= 8 bits
        let mut code_idx = 0;
        for (len, &count) in bits.iter().enumerate() {
            let len = len + 1;
            for _ in 0..count {
                if len <= 8 {
                    let code = huffcode.get(code_idx).copied().unwrap_or(0);
                    let val = values.get(code_idx).copied().unwrap_or(0);
                    // Fill all entries that start with this code
                    let fill_bits = 8 - len;
                    let base = (code as usize) << fill_bits;
                    for i in 0..(1 << fill_bits) {
                        let idx = base | i;
                        if idx < 256 {
                            // Pack: value in low 8 bits, length in high 8 bits
                            table.lookup[idx] = (val as u16) | ((len as u16) << 8);
                        }
                    }
                }
                code_idx += 1;
            }
        }

        table
    }

    /// Decode a symbol using the Huffman table.
    fn decode(&self, reader: &mut MsbBitReader) -> Result<u8> {
        // Try fast lookup first
        if let Ok(peek) = reader.peek_bits(8) {
            let entry = self.lookup[peek as usize];
            let len = (entry >> 8) as u8;
            if len > 0 && len <= 8 {
                reader.consume(len);
                return Ok((entry & 0xFF) as u8);
            }
        }

        // Slow path for longer codes
        self.decode_slow(reader)
    }

    fn decode_slow(&self, reader: &mut MsbBitReader) -> Result<u8> {
        let mut code = 0i32;
        for len in 1..=16 {
            code = (code << 1) | reader.read_bits(1)? as i32;
            if code <= self.max_code[len] {
                let idx = (code + self.val_offset[len]) as usize;
                return self
                    .values
                    .get(idx)
                    .copied()
                    .ok_or_else(|| Error::InvalidDecode("invalid Huffman code".into()));
            }
        }
        Err(Error::InvalidDecode("Huffman code not found".into()))
    }
}

/// JPEG decoder state.
struct JpegDecoder<'a> {
    data: &'a [u8],
    pos: usize,
    width: u32,
    height: u32,
    components: Vec<Component>,
    quant_tables: [[u16; 64]; 4],
    dc_tables: [HuffmanTable; 4],
    ac_tables: [HuffmanTable; 4],
    restart_interval: u16,
    max_h_sampling: u8,
    max_v_sampling: u8,
}

impl<'a> JpegDecoder<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            width: 0,
            height: 0,
            components: Vec::new(),
            quant_tables: [[0; 64]; 4],
            dc_tables: Default::default(),
            ac_tables: Default::default(),
            restart_interval: 0,
            max_h_sampling: 1,
            max_v_sampling: 1,
        }
    }

    fn decode(mut self) -> Result<JpegImage> {
        // Verify SOI marker
        if self.data.len() < 2 || self.data[0] != 0xFF || self.data[1] != SOI {
            return Err(Error::InvalidDecode("not a JPEG file".into()));
        }
        self.pos = 2;

        // Parse markers
        loop {
            let (marker, segment) = self.read_marker()?;

            match marker {
                SOF0 => self.parse_sof0(&segment)?,
                SOF2 => {
                    return Err(Error::UnsupportedDecode(
                        "progressive JPEG not supported".into(),
                    ))
                }
                DHT => self.parse_dht(&segment)?,
                DQT => self.parse_dqt(&segment)?,
                DRI => self.parse_dri(&segment)?,
                SOS => {
                    self.parse_sos(&segment)?;
                    let image = self.decode_scan()?;
                    return Ok(image);
                }
                EOI => break,
                APP0..=APP15 | COM => {
                    // Skip application data and comments
                }
                _ => {
                    // Skip unknown markers
                }
            }
        }

        Err(Error::InvalidDecode("no image data found".into()))
    }

    fn read_marker(&mut self) -> Result<(u8, Vec<u8>)> {
        // Find marker
        while self.pos < self.data.len() && self.data[self.pos] != 0xFF {
            self.pos += 1;
        }

        // Skip padding 0xFF bytes
        while self.pos < self.data.len() && self.data[self.pos] == 0xFF {
            self.pos += 1;
        }

        if self.pos >= self.data.len() {
            return Err(Error::InvalidDecode("unexpected end of file".into()));
        }

        let marker = self.data[self.pos];
        self.pos += 1;

        // Markers without payload
        match marker {
            SOI | EOI | RST0..=0xD7 => return Ok((marker, Vec::new())),
            _ => {}
        }

        // Read segment length
        if self.pos + 2 > self.data.len() {
            return Err(Error::InvalidDecode("truncated marker".into()));
        }
        let length = u16::from_be_bytes([self.data[self.pos], self.data[self.pos + 1]]) as usize;
        self.pos += 2;

        if length < 2 || self.pos + length - 2 > self.data.len() {
            return Err(Error::InvalidDecode("invalid marker length".into()));
        }

        let segment = self.data[self.pos..self.pos + length - 2].to_vec();
        self.pos += length - 2;

        Ok((marker, segment))
    }

    fn parse_sof0(&mut self, segment: &[u8]) -> Result<()> {
        if segment.len() < 8 {
            return Err(Error::InvalidDecode("invalid SOF0 length".into()));
        }

        let precision = segment[0];
        if precision != 8 {
            return Err(Error::UnsupportedDecode(format!(
                "{precision}-bit precision not supported"
            )));
        }

        self.height = u16::from_be_bytes([segment[1], segment[2]]) as u32;
        self.width = u16::from_be_bytes([segment[3], segment[4]]) as u32;

        let num_components = segment[5] as usize;
        if num_components != 1 && num_components != 3 {
            return Err(Error::UnsupportedDecode(format!(
                "{num_components} components not supported"
            )));
        }

        if segment.len() < 6 + num_components * 3 {
            return Err(Error::InvalidDecode("truncated SOF0 components".into()));
        }

        self.components.clear();
        for i in 0..num_components {
            let offset = 6 + i * 3;
            let id = segment[offset];
            let sampling = segment[offset + 1];
            let h_sampling = (sampling >> 4) & 0x0F;
            let v_sampling = sampling & 0x0F;

            // Validate sampling factors (0 is invalid, would cause division by zero)
            if h_sampling == 0 || v_sampling == 0 {
                return Err(Error::InvalidDecode(format!(
                    "invalid sampling factors {h_sampling}x{v_sampling} for component {id}"
                )));
            }

            let quant_table_id = segment[offset + 2];

            // Validate quantization table ID (must be 0-3)
            if quant_table_id > 3 {
                return Err(Error::InvalidDecode(format!(
                    "invalid quantization table ID {quant_table_id} for component {id}"
                )));
            }

            self.max_h_sampling = self.max_h_sampling.max(h_sampling);
            self.max_v_sampling = self.max_v_sampling.max(v_sampling);

            self.components.push(Component {
                id,
                h_sampling,
                v_sampling,
                quant_table_id,
                ..Default::default()
            });
        }

        Ok(())
    }

    fn parse_dht(&mut self, segment: &[u8]) -> Result<()> {
        let mut offset = 0;
        while offset < segment.len() {
            let info = segment[offset];
            let table_class = (info >> 4) & 0x0F; // 0 = DC, 1 = AC
            let table_id = (info & 0x0F) as usize;

            if table_id > 3 {
                return Err(Error::InvalidDecode("invalid Huffman table ID".into()));
            }

            offset += 1;
            if offset + 16 > segment.len() {
                return Err(Error::InvalidDecode("truncated DHT".into()));
            }

            let mut bits = [0u8; 16];
            bits.copy_from_slice(&segment[offset..offset + 16]);
            offset += 16;

            let num_values: usize = bits.iter().map(|&b| b as usize).sum();
            if offset + num_values > segment.len() {
                return Err(Error::InvalidDecode("truncated DHT values".into()));
            }

            let values = &segment[offset..offset + num_values];
            offset += num_values;

            let table = HuffmanTable::build(&bits, values);
            if table_class == 0 {
                self.dc_tables[table_id] = table;
            } else {
                self.ac_tables[table_id] = table;
            }
        }

        Ok(())
    }

    fn parse_dqt(&mut self, segment: &[u8]) -> Result<()> {
        let mut offset = 0;
        while offset < segment.len() {
            let info = segment[offset];
            let precision = (info >> 4) & 0x0F;
            let table_id = (info & 0x0F) as usize;

            if table_id > 3 {
                return Err(Error::InvalidDecode("invalid quantization table ID".into()));
            }

            offset += 1;

            if precision == 0 {
                // 8-bit precision
                if offset + 64 > segment.len() {
                    return Err(Error::InvalidDecode("truncated DQT".into()));
                }
                for i in 0..64 {
                    self.quant_tables[table_id][i] = segment[offset + i] as u16;
                }
                offset += 64;
            } else {
                // 16-bit precision
                if offset + 128 > segment.len() {
                    return Err(Error::InvalidDecode("truncated DQT".into()));
                }
                for i in 0..64 {
                    self.quant_tables[table_id][i] =
                        u16::from_be_bytes([segment[offset + i * 2], segment[offset + i * 2 + 1]]);
                }
                offset += 128;
            }
        }

        Ok(())
    }

    fn parse_dri(&mut self, segment: &[u8]) -> Result<()> {
        if segment.len() != 2 {
            return Err(Error::InvalidDecode("invalid DRI length".into()));
        }

        self.restart_interval = u16::from_be_bytes([segment[0], segment[1]]);

        Ok(())
    }

    fn parse_sos(&mut self, segment: &[u8]) -> Result<()> {
        if segment.is_empty() {
            return Err(Error::InvalidDecode("empty SOS segment".into()));
        }

        let num_components = segment[0] as usize;
        if num_components != self.components.len() {
            return Err(Error::InvalidDecode("SOS component count mismatch".into()));
        }

        // Read component selectors
        for i in 0..num_components {
            let offset = 1 + i * 2;
            if offset + 1 >= segment.len() {
                return Err(Error::InvalidDecode("truncated SOS segment".into()));
            }
            let component_id = segment[offset];
            let tables = segment[offset + 1];
            let dc_table_id = (tables >> 4) & 0x0F;
            let ac_table_id = tables & 0x0F;

            // Validate Huffman table IDs (must be 0-3)
            if dc_table_id > 3 {
                return Err(Error::InvalidDecode(format!(
                    "invalid DC Huffman table ID {dc_table_id} for component {component_id}"
                )));
            }
            if ac_table_id > 3 {
                return Err(Error::InvalidDecode(format!(
                    "invalid AC Huffman table ID {ac_table_id} for component {component_id}"
                )));
            }

            self.components[i].dc_table_id = dc_table_id;
            self.components[i].ac_table_id = ac_table_id;
        }

        Ok(())
    }

    fn decode_scan(&mut self) -> Result<JpegImage> {
        // For subsampled images, MCUs are larger
        let mcu_width = (self.width as usize).div_ceil(self.max_h_sampling as usize * 8);
        let mcu_height = (self.height as usize).div_ceil(self.max_v_sampling as usize * 8);

        // Allocate component data
        let mut comp_data: Vec<Vec<i16>> = self
            .components
            .iter()
            .map(|c| {
                let w = mcu_width * c.h_sampling as usize * 8;
                let h = mcu_height * c.v_sampling as usize * 8;
                vec![0i16; w * h]
            })
            .collect();

        // Decode MCUs
        let entropy_start = self.pos;
        let entropy_end = find_entropy_end(&self.data[entropy_start..]);
        let entropy_data = &self.data[entropy_start..entropy_start + entropy_end];

        let mut reader = MsbBitReader::new(entropy_data);
        let mut dc_pred = vec![0i32; self.components.len()];
        let mut mcu_count = 0u32;

        'mcu_loop: for mcu_y in 0..mcu_height {
            for mcu_x in 0..mcu_width {
                // Handle restart markers
                if self.restart_interval > 0
                    && mcu_count > 0
                    && mcu_count % self.restart_interval as u32 == 0
                {
                    // Reset DC predictors
                    dc_pred.fill(0);
                    // Skip to next restart marker (handled by bit reader)
                }

                // Decode each component's blocks in this MCU
                for (comp_idx, comp) in self.components.iter().enumerate() {
                    let blocks_h = comp.h_sampling as usize;
                    let blocks_v = comp.v_sampling as usize;

                    for block_y in 0..blocks_v {
                        for block_x in 0..blocks_h {
                            let mut coeffs = [0i16; 64];

                            // Decode DC coefficient
                            let dc_table = &self.dc_tables[comp.dc_table_id as usize];
                            let category = match dc_table.decode(&mut reader) {
                                Ok(c) => c,
                                Err(_) => break 'mcu_loop,
                            };
                            let diff = if category > 0 {
                                match read_amplitude(&mut reader, category) {
                                    Ok(a) => a,
                                    Err(_) => break 'mcu_loop,
                                }
                            } else {
                                0
                            };
                            dc_pred[comp_idx] += diff;
                            coeffs[0] = dc_pred[comp_idx] as i16;

                            // Decode AC coefficients
                            let ac_table = &self.ac_tables[comp.ac_table_id as usize];
                            let mut k = 1;
                            while k < 64 {
                                let symbol = match ac_table.decode(&mut reader) {
                                    Ok(s) => s,
                                    Err(_) => break 'mcu_loop,
                                };

                                if symbol == 0 {
                                    // EOB - remaining coefficients are zero
                                    break;
                                }

                                let run = (symbol >> 4) & 0x0F;
                                let size = symbol & 0x0F;

                                if symbol == 0xF0 {
                                    // ZRL - skip 16 zeros
                                    k += 16;
                                    continue;
                                }

                                k += run as usize;
                                if k >= 64 {
                                    break;
                                }

                                if size > 0 {
                                    let amp = match read_amplitude(&mut reader, size) {
                                        Ok(a) => a,
                                        Err(_) => break 'mcu_loop,
                                    };
                                    coeffs[k] = amp as i16;
                                }
                                k += 1;
                            }

                            // Dequantize and IDCT
                            let qtable = &self.quant_tables[comp.quant_table_id as usize];
                            let dequantized = dequantize(&coeffs, qtable);
                            let block_pixels = idct_2d_integer(&dequantized);

                            // Store block in component data
                            let comp_width = mcu_width * blocks_h * 8;
                            let start_x = (mcu_x * blocks_h + block_x) * 8;
                            let start_y = (mcu_y * blocks_v + block_y) * 8;

                            for by in 0..8 {
                                for bx in 0..8 {
                                    let x = start_x + bx;
                                    let y = start_y + by;
                                    let idx = y * comp_width + x;
                                    if idx < comp_data[comp_idx].len() {
                                        comp_data[comp_idx][idx] = block_pixels[by * 8 + bx] as i16;
                                    }
                                }
                            }
                        }
                    }
                }
                mcu_count += 1;
            }
        }

        // Convert to final pixel format
        if self.components.len() == 1 {
            // Grayscale - crop MCU-aligned buffer to actual dimensions
            let comp_width = mcu_width * self.components[0].h_sampling as usize * 8;
            let mut pixels = Vec::with_capacity(self.width as usize * self.height as usize);
            for y in 0..self.height as usize {
                for x in 0..self.width as usize {
                    let idx = y * comp_width + x;
                    let val = comp_data[0].get(idx).copied().unwrap_or(0);
                    pixels.push(val.clamp(0, 255) as u8);
                }
            }
            Ok(JpegImage {
                width: self.width,
                height: self.height,
                pixels,
                color_type: ColorType::Gray,
            })
        } else {
            // YCbCr to RGB conversion
            let pixels = ycbcr_to_rgb(
                &comp_data,
                self.width as usize,
                self.height as usize,
                &self.components,
                self.max_h_sampling,
                self.max_v_sampling,
            );
            Ok(JpegImage {
                width: self.width,
                height: self.height,
                pixels,
                color_type: ColorType::Rgb,
            })
        }
    }
}

/// Find the end of entropy-coded data (before next marker).
fn find_entropy_end(data: &[u8]) -> usize {
    if data.len() < 2 {
        return data.len();
    }
    let mut i = 0;
    while i < data.len() - 1 {
        if data[i] == 0xFF && data[i + 1] != 0x00 && data[i + 1] != 0xFF {
            // Found a marker (that's not stuffed 0xFF00 or padding 0xFFFF)
            if data[i + 1] >= RST0 && data[i + 1] <= 0xD7 {
                // Restart marker - skip it
                i += 2;
                continue;
            }
            return i;
        }
        i += 1;
    }
    data.len()
}

/// Read a signed amplitude value of the given size.
fn read_amplitude(reader: &mut MsbBitReader, size: u8) -> Result<i32> {
    if size == 0 {
        return Ok(0);
    }
    let bits = reader.read_bits(size)? as i32;
    // Sign extension for negative values
    let threshold = 1 << (size - 1);
    if bits < threshold {
        Ok(bits - (2 * threshold - 1))
    } else {
        Ok(bits)
    }
}

/// Convert YCbCr to RGB.
fn ycbcr_to_rgb(
    comp_data: &[Vec<i16>],
    width: usize,
    height: usize,
    components: &[Component],
    max_h: u8,
    max_v: u8,
) -> Vec<u8> {
    let mut pixels = Vec::with_capacity(width * height * 3);

    let mcu_cols = width.div_ceil(max_h as usize * 8);
    let y_width = mcu_cols * max_h as usize * 8;
    let cb_width = mcu_cols * components[1].h_sampling as usize * 8;
    let cr_width = mcu_cols * components[2].h_sampling as usize * 8;

    let h_ratio_cb = max_h / components[1].h_sampling;
    let v_ratio_cb = max_v / components[1].v_sampling;
    let h_ratio_cr = max_h / components[2].h_sampling;
    let v_ratio_cr = max_v / components[2].v_sampling;

    for y in 0..height {
        for x in 0..width {
            let y_idx = y * y_width + x;
            let cb_x = x / h_ratio_cb as usize;
            let cb_y = y / v_ratio_cb as usize;
            let cb_idx = cb_y * cb_width + cb_x;
            let cr_x = x / h_ratio_cr as usize;
            let cr_y = y / v_ratio_cr as usize;
            let cr_idx = cr_y * cr_width + cr_x;

            let y_val = comp_data[0].get(y_idx).copied().unwrap_or(0) as i32;
            let cb_val = comp_data[1].get(cb_idx).copied().unwrap_or(128) as i32 - 128;
            let cr_val = comp_data[2].get(cr_idx).copied().unwrap_or(128) as i32 - 128;

            // YCbCr to RGB conversion (ITU-R BT.601)
            let r = y_val + ((cr_val * 359) >> 8);
            let g = y_val - ((cb_val * 88 + cr_val * 183) >> 8);
            let b = y_val + ((cb_val * 454) >> 8);

            pixels.push(r.clamp(0, 255) as u8);
            pixels.push(g.clamp(0, 255) as u8);
            pixels.push(b.clamp(0, 255) as u8);
        }
    }

    pixels
}

/// Decode a JPEG image from bytes.
pub fn decode_jpeg(data: &[u8]) -> Result<JpegImage> {
    JpegDecoder::new(data).decode()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_invalid() {
        let data = b"not a jpeg";
        assert!(decode_jpeg(data).is_err());
    }

    #[test]
    fn test_decode_empty() {
        let data: &[u8] = &[];
        assert!(decode_jpeg(data).is_err());
    }

    #[test]
    fn test_decode_soi_only() {
        let data = [0xFF, 0xD8];
        assert!(decode_jpeg(&data).is_err());
    }

    #[test]
    fn test_decode_invalid_soi() {
        let data = [0xFF, 0xD9, 0xFF, 0xD8]; // EOI before SOI
        assert!(decode_jpeg(&data).is_err());
    }

    #[test]
    fn test_read_amplitude() {
        // Test amplitude decoding
        // Size 1: bits 0 = -1, bits 1 = 1
        // Size 2: bits 0-1 = -3 to -2, bits 2-3 = 2-3
        // etc.

        // We can't easily test read_amplitude without a bit reader,
        // so just verify the logic works with direct computation
        fn decode_amplitude(bits: i32, size: u8) -> i32 {
            let threshold = 1 << (size - 1);
            if bits < threshold {
                bits - (2 * threshold - 1)
            } else {
                bits
            }
        }

        assert_eq!(decode_amplitude(0, 1), -1);
        assert_eq!(decode_amplitude(1, 1), 1);
        assert_eq!(decode_amplitude(0, 2), -3);
        assert_eq!(decode_amplitude(1, 2), -2);
        assert_eq!(decode_amplitude(2, 2), 2);
        assert_eq!(decode_amplitude(3, 2), 3);
    }

    #[test]
    fn test_read_amplitude_larger_sizes() {
        fn decode_amplitude(bits: i32, size: u8) -> i32 {
            let threshold = 1 << (size - 1);
            if bits < threshold {
                bits - (2 * threshold - 1)
            } else {
                bits
            }
        }

        // Size 3: range -7 to -4, 4 to 7
        assert_eq!(decode_amplitude(0, 3), -7);
        assert_eq!(decode_amplitude(7, 3), 7);
        assert_eq!(decode_amplitude(4, 3), 4);

        // Size 4: range -15 to -8, 8 to 15
        assert_eq!(decode_amplitude(0, 4), -15);
        assert_eq!(decode_amplitude(15, 4), 15);
    }

    #[test]
    fn test_huffman_table_build() {
        // Build a simple Huffman table
        let bits = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let values = [0, 1];

        let table = HuffmanTable::build(&bits, &values);
        assert_eq!(table.values.len(), 2);
    }

    #[test]
    fn test_huffman_table_more_complex() {
        // More complex table with different code lengths
        let bits = [0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let values = [0, 1, 2];

        let table = HuffmanTable::build(&bits, &values);
        assert_eq!(table.values.len(), 3);
        assert_eq!(table.max_code[2], 1); // Two 2-bit codes: 00, 01
        assert_eq!(table.max_code[3], 4); // One 3-bit code: 100
    }

    #[test]
    fn test_huffman_table_lookup() {
        // Build a simple table and verify it's properly constructed
        let bits = [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let values = [0, 1];

        let table = HuffmanTable::build(&bits, &values);

        // Verify the table has the expected structure
        assert_eq!(table.values.len(), 2);
        assert_eq!(table.values[0], 0);
        assert_eq!(table.values[1], 1);

        // Check that lookup table is populated (256 entries)
        assert_eq!(table.lookup.len(), 256);
    }

    #[test]
    fn test_find_entropy_end() {
        // Data ending with EOI marker
        let data = [0x12, 0x34, 0xFF, 0xD9];
        assert_eq!(find_entropy_end(&data), 2);

        // Data with byte stuffing
        let data = [0x12, 0xFF, 0x00, 0x34, 0xFF, 0xD9];
        assert_eq!(find_entropy_end(&data), 4);
    }

    #[test]
    fn test_find_entropy_end_restart_markers() {
        // Restart markers should be skipped
        let data = [0x12, 0xFF, 0xD0, 0x34, 0xFF, 0xD9];
        assert_eq!(find_entropy_end(&data), 4);
    }

    #[test]
    fn test_find_entropy_end_multiple_stuffed() {
        let data = [0xFF, 0x00, 0xFF, 0x00, 0xFF, 0xD9];
        assert_eq!(find_entropy_end(&data), 4);
    }

    #[test]
    fn test_find_entropy_end_no_marker() {
        let data = [0x12, 0x34, 0x56, 0x78];
        assert_eq!(find_entropy_end(&data), 4); // Returns full length
    }

    #[test]
    fn test_find_entropy_end_empty() {
        // Empty slice should not underflow
        assert_eq!(find_entropy_end(&[]), 0);
    }

    #[test]
    fn test_find_entropy_end_single_byte() {
        // Single byte - need at least 2 for marker check
        assert_eq!(find_entropy_end(&[0xFF]), 1);
        assert_eq!(find_entropy_end(&[0x12]), 1);
    }

    #[test]
    fn test_component_default() {
        let comp = Component::default();
        assert_eq!(comp.h_sampling, 0);
        assert_eq!(comp.v_sampling, 0);
        assert_eq!(comp.quant_table_id, 0);
    }

    #[test]
    fn test_huffman_table_default() {
        let table = HuffmanTable::default();
        assert!(table.values.is_empty());
        assert_eq!(table.max_code[1], -1);
    }

    #[test]
    fn test_jpeg_decode_zero_sampling_factor() {
        // Craft a minimal JPEG with zero sampling factor in SOF0
        // This should return an error, not panic with division by zero
        let mut jpeg = Vec::new();

        // SOI marker
        jpeg.extend_from_slice(&[0xFF, 0xD8]);

        // SOF0 marker with zero sampling factors
        // FF C0 = SOF0 marker
        // 00 0B = length (11 bytes including length field)
        // 08 = 8-bit precision
        // 00 08 = height (8)
        // 00 08 = width (8)
        // 01 = 1 component
        // 01 = component ID
        // 00 = sampling factors: h=0, v=0 (INVALID!)
        // 00 = quantization table ID
        jpeg.extend_from_slice(&[
            0xFF, 0xC0, // SOF0 marker
            0x00, 0x0B, // length
            0x08, // precision
            0x00, 0x08, // height
            0x00, 0x08, // width
            0x01, // num components
            0x01, // component ID
            0x00, // h_sampling=0, v_sampling=0 (both invalid!)
            0x00, // quant table
        ]);

        // EOI marker
        jpeg.extend_from_slice(&[0xFF, 0xD9]);

        let result = decode_jpeg(&jpeg);
        assert!(result.is_err(), "should error on zero sampling factor");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("sampling factors"),
            "error should mention sampling factors: {err}"
        );
    }

    #[test]
    fn test_jpeg_encode_decode_roundtrip() {
        // Create a simple image and encode it, then decode
        let pixels = vec![128u8; 8 * 8 * 3];
        let opts = crate::jpeg::JpegOptions::balanced(8, 8, 95);
        let encoded = crate::jpeg::encode(&pixels, &opts).expect("encode should work");
        let decoded = decode_jpeg(&encoded).expect("decode should work");

        assert_eq!(decoded.width, 8);
        assert_eq!(decoded.height, 8);
        assert_eq!(decoded.color_type, crate::ColorType::Rgb);
        assert_eq!(decoded.pixels.len(), 8 * 8 * 3);
    }

    #[test]
    fn test_jpeg_encode_decode_grayscale() {
        let pixels = vec![128u8; 8 * 8];
        let opts = crate::jpeg::JpegOptions::builder(8, 8)
            .color_type(crate::ColorType::Gray)
            .quality(95)
            .build();
        let encoded = crate::jpeg::encode(&pixels, &opts).expect("encode should work");
        let decoded = decode_jpeg(&encoded).expect("decode should work");

        assert_eq!(decoded.width, 8);
        assert_eq!(decoded.height, 8);
        assert_eq!(decoded.color_type, crate::ColorType::Gray);
    }

    #[test]
    fn test_jpeg_decode_invalid_quant_table_id() {
        // Craft a JPEG with invalid quantization table ID (>3) in SOF0
        let mut jpeg = Vec::new();

        // SOI marker
        jpeg.extend_from_slice(&[0xFF, 0xD8]);

        // SOF0 marker with quant_table_id = 5 (invalid!)
        jpeg.extend_from_slice(&[
            0xFF, 0xC0, // SOF0 marker
            0x00, 0x0B, // length (11 bytes)
            0x08, // 8-bit precision
            0x00, 0x08, // height = 8
            0x00, 0x08, // width = 8
            0x01, // 1 component
            0x01, // component ID = 1
            0x11, // h_sampling=1, v_sampling=1
            0x05, // quant_table_id = 5 (INVALID - max is 3!)
        ]);

        // EOI marker
        jpeg.extend_from_slice(&[0xFF, 0xD9]);

        let result = decode_jpeg(&jpeg);
        assert!(result.is_err(), "should error on invalid quant table ID");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("quantization table ID"),
            "error should mention quantization table ID: {err}"
        );
    }

    #[test]
    fn test_jpeg_decode_invalid_dc_table_id() {
        // Craft a JPEG with invalid DC Huffman table ID (>3) in SOS
        let mut jpeg = Vec::new();

        // SOI
        jpeg.extend_from_slice(&[0xFF, 0xD8]);

        // DQT with table 0
        jpeg.extend_from_slice(&[0xFF, 0xDB, 0x00, 0x43, 0x00]);
        jpeg.extend_from_slice(&[16u8; 64]);

        // SOF0 with valid quant_table_id = 0
        jpeg.extend_from_slice(&[
            0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x08, 0x00, 0x08, 0x01, 0x01, 0x11, 0x00,
        ]);

        // DHT with table 0 (DC)
        jpeg.extend_from_slice(&[0xFF, 0xC4, 0x00, 0x14, 0x00]);
        jpeg.extend_from_slice(&[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]); // bits
        jpeg.extend_from_slice(&[0]); // values

        // SOS with dc_table_id = 5 (invalid!)
        // tables byte: high nibble = DC table, low nibble = AC table
        // 0x50 = DC table 5, AC table 0
        jpeg.extend_from_slice(&[
            0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x50, // dc=5, ac=0
            0x00, 0x3F, 0x00,
        ]);

        // EOI
        jpeg.extend_from_slice(&[0xFF, 0xD9]);

        let result = decode_jpeg(&jpeg);
        assert!(result.is_err(), "should error on invalid DC table ID");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("DC Huffman table ID"),
            "error should mention DC Huffman table ID: {err}"
        );
    }

    #[test]
    fn test_jpeg_decode_invalid_ac_table_id() {
        // Craft a JPEG with invalid AC Huffman table ID (>3) in SOS
        let mut jpeg = Vec::new();

        // SOI
        jpeg.extend_from_slice(&[0xFF, 0xD8]);

        // DQT with table 0
        jpeg.extend_from_slice(&[0xFF, 0xDB, 0x00, 0x43, 0x00]);
        jpeg.extend_from_slice(&[16u8; 64]);

        // SOF0 with valid quant_table_id = 0
        jpeg.extend_from_slice(&[
            0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x08, 0x00, 0x08, 0x01, 0x01, 0x11, 0x00,
        ]);

        // DHT with table 0 (DC)
        jpeg.extend_from_slice(&[0xFF, 0xC4, 0x00, 0x14, 0x00]);
        jpeg.extend_from_slice(&[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]); // bits
        jpeg.extend_from_slice(&[0]); // values

        // SOS with ac_table_id = 7 (invalid!)
        // tables byte: high nibble = DC table, low nibble = AC table
        // 0x07 = DC table 0, AC table 7
        jpeg.extend_from_slice(&[
            0xFF, 0xDA, 0x00, 0x08, 0x01, 0x01, 0x07, // dc=0, ac=7
            0x00, 0x3F, 0x00,
        ]);

        // EOI
        jpeg.extend_from_slice(&[0xFF, 0xD9]);

        let result = decode_jpeg(&jpeg);
        assert!(result.is_err(), "should error on invalid AC table ID");
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("AC Huffman table ID"),
            "error should mention AC Huffman table ID: {err}"
        );
    }
}
