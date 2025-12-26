//! Bit-depth reduction utilities for PNG.
//!
//! Supports lossless reduction from 8-bit to 1/2/4-bit for grayscale and palette images
//! when all samples fit in the smaller range.

use crate::color::ColorType;

/// Determine the minimal bit depth that can represent the samples.
///
/// Returns Some(bit_depth) where bit_depth ∈ {1,2,4,8} if reducible, else None.
pub fn reduce_bit_depth(data: &[u8], color_type: ColorType) -> Option<u8> {
    match color_type {
        ColorType::Gray => reduce_gray_bit_depth(data),
        _ => None,
    }
}

pub fn palette_bit_depth(len: usize) -> u8 {
    if len == 0 {
        8
    } else if len <= 2 {
        1
    } else if len <= 4 {
        2
    } else if len <= 16 {
        4
    } else {
        8
    }
}

fn reduce_gray_bit_depth(data: &[u8]) -> Option<u8> {
    if data.is_empty() {
        return None;
    }
    let max = data.iter().copied().max().unwrap();
    if max <= 1 {
        Some(1)
    } else if max <= 3 {
        Some(2)
    } else if max <= 15 {
        Some(4)
    } else {
        Some(8)
    }
}

#[cfg(test)]
pub fn pack_gray(data: &[u8], bit_depth: u8) -> Vec<u8> {
    match bit_depth {
        1 => pack_bits(data, 1),
        2 => pack_bits(data, 2),
        4 => pack_bits(data, 4),
        8 => data.to_vec(),
        _ => data.to_vec(),
    }
}

#[cfg(test)]
pub fn pack_indexed(data: &[u8], bit_depth: u8) -> Vec<u8> {
    match bit_depth {
        1 => pack_bits(data, 1),
        2 => pack_bits(data, 2),
        4 => pack_bits(data, 4),
        8 => data.to_vec(),
        _ => data.to_vec(),
    }
}

pub fn pack_gray_rows(data: &[u8], width: usize, bit_depth: u8) -> Vec<u8> {
    pack_bits_rows(data, width, bit_depth)
}

pub fn pack_indexed_rows(data: &[u8], width: usize, bit_depth: u8) -> Vec<u8> {
    pack_bits_rows(data, width, bit_depth)
}

#[cfg(test)]
pub fn pack_bits(data: &[u8], bits: u8) -> Vec<u8> {
    debug_assert!(
        matches!(bits, 1 | 2 | 4 | 8),
        "pack_bits expected bit depth 1, 2, 4, or 8"
    );
    let mut out = Vec::with_capacity((data.len() * bits as usize).div_ceil(8));
    let mut acc: u8 = 0;
    let mut acc_bits = 0;
    let mask = (1u8 << bits) - 1;
    for &v in data {
        let clipped = v & mask;
        acc = (acc << bits) | clipped;
        acc_bits += bits as usize;
        if acc_bits == 8 {
            out.push(acc);
            acc = 0;
            acc_bits = 0;
        }
    }
    if acc_bits > 0 {
        acc <<= 8 - acc_bits;
        out.push(acc);
    }
    out
}

fn pack_bits_rows(data: &[u8], width: usize, bits: u8) -> Vec<u8> {
    debug_assert!(
        matches!(bits, 1 | 2 | 4 | 8),
        "pack_bits_rows expected bit depth 1, 2, 4, or 8"
    );
    if bits == 8 {
        return data.to_vec();
    }
    if width == 0 || data.is_empty() {
        return Vec::new();
    }

    let row_bytes = (width * bits as usize).div_ceil(8);
    let height = data.len() / width;
    let mut out = Vec::with_capacity(row_bytes * height);
    let mask = (1u8 << bits) - 1;

    for row in data.chunks_exact(width) {
        let mut acc: u8 = 0;
        let mut acc_bits = 0usize;
        for &v in row {
            let clipped = v & mask;
            acc = (acc << bits) | clipped;
            acc_bits += bits as usize;
            if acc_bits == 8 {
                out.push(acc);
                acc = 0;
                acc_bits = 0;
            }
        }
        if acc_bits > 0 {
            acc <<= 8 - acc_bits;
            out.push(acc);
        }
    }

    debug_assert!(
        data.len() % width == 0,
        "pack_bits_rows expects full rows (len {}, width {})",
        data.len(),
        width
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduce_gray_bit_depth() {
        assert_eq!(reduce_gray_bit_depth(&[0, 1]), Some(1));
        assert_eq!(reduce_gray_bit_depth(&[0, 2, 3]), Some(2));
        assert_eq!(reduce_gray_bit_depth(&[0, 15]), Some(4));
        assert_eq!(reduce_gray_bit_depth(&[0, 16]), Some(8));
    }

    #[test]
    fn test_reduce_gray_bit_depth_empty() {
        assert_eq!(reduce_gray_bit_depth(&[]), None);
    }

    #[test]
    fn test_pack_bits() {
        // 1-bit packing: [1,0,1,0,1,0,1,0] -> 0b10101010
        let packed = pack_bits(&[1, 0, 1, 0, 1, 0, 1, 0], 1);
        assert_eq!(packed, vec![0b10101010]);

        // 2-bit packing: [0,1,2,3] -> 00 01 10 11 = 0x1B
        let packed = pack_bits(&[0, 1, 2, 3], 2);
        assert_eq!(packed, vec![0b00011011]);

        // 4-bit packing: [0xA, 0xB] -> 0xAB
        let packed = pack_bits(&[0xA, 0xB], 4);
        assert_eq!(packed, vec![0xAB]);
    }

    #[test]
    fn test_reduce_bit_depth_gray() {
        // Test through the public interface
        assert_eq!(reduce_bit_depth(&[0, 1], ColorType::Gray), Some(1));
        assert_eq!(reduce_bit_depth(&[0, 3], ColorType::Gray), Some(2));
        assert_eq!(reduce_bit_depth(&[0, 15], ColorType::Gray), Some(4));
        assert_eq!(reduce_bit_depth(&[0, 255], ColorType::Gray), Some(8));
    }

    #[test]
    fn test_reduce_bit_depth_non_gray() {
        // Non-grayscale color types should return None
        assert_eq!(reduce_bit_depth(&[0, 1], ColorType::Rgb), None);
        assert_eq!(reduce_bit_depth(&[0, 1], ColorType::Rgba), None);
        assert_eq!(reduce_bit_depth(&[0, 1], ColorType::GrayAlpha), None);
    }

    #[test]
    fn test_palette_bit_depth() {
        assert_eq!(palette_bit_depth(0), 8);
        assert_eq!(palette_bit_depth(1), 1);
        assert_eq!(palette_bit_depth(2), 1);
        assert_eq!(palette_bit_depth(3), 2);
        assert_eq!(palette_bit_depth(4), 2);
        assert_eq!(palette_bit_depth(5), 4);
        assert_eq!(palette_bit_depth(16), 4);
        assert_eq!(palette_bit_depth(17), 8);
        assert_eq!(palette_bit_depth(256), 8);
    }

    #[test]
    fn test_pack_gray() {
        // Test all bit depths
        let data = &[0, 1, 0, 1];
        assert_eq!(pack_gray(data, 1).len(), 1);
        assert_eq!(pack_gray(data, 2).len(), 1);
        assert_eq!(pack_gray(data, 4).len(), 2);
        assert_eq!(pack_gray(data, 8), data.to_vec());
        // Test invalid bit depth returns unchanged
        assert_eq!(pack_gray(data, 16), data.to_vec());
    }

    #[test]
    fn test_pack_indexed() {
        // Test all bit depths
        let data = &[0, 1, 2, 3];
        assert_eq!(pack_indexed(data, 1).len(), 1);
        assert_eq!(pack_indexed(data, 2).len(), 1);
        assert_eq!(pack_indexed(data, 4).len(), 2);
        assert_eq!(pack_indexed(data, 8), data.to_vec());
        // Test invalid bit depth returns unchanged
        assert_eq!(pack_indexed(data, 16), data.to_vec());
    }

    #[test]
    fn test_pack_bits_partial_byte() {
        // Test when data doesn't fill a complete byte
        let packed = pack_bits(&[1, 0, 1], 1); // 3 bits -> 101xxxxx
        assert_eq!(packed, vec![0b10100000]);

        let packed = pack_bits(&[1, 2], 2); // 4 bits -> 0110xxxx
        assert_eq!(packed, vec![0b01100000]);

        let packed = pack_bits(&[0xA], 4); // 4 bits -> 1010xxxx
        assert_eq!(packed, vec![0b10100000]);
    }

    #[test]
    fn test_pack_gray_8bit() {
        // 8-bit packing via pack_gray should be identity
        let data = &[0x12, 0x34, 0x56];
        assert_eq!(pack_gray(data, 8), data.to_vec());
    }
}
