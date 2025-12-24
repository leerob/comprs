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

/// Determine bit depth for a palette of given length.
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

/// Pack 8-bit grayscale samples into smaller bit depths.
pub fn pack_gray(data: &[u8], bit_depth: u8) -> Vec<u8> {
    match bit_depth {
        1 => pack_bits(data, 1),
        2 => pack_bits(data, 2),
        4 => pack_bits(data, 4),
        8 => data.to_vec(),
        _ => data.to_vec(),
    }
}

/// Pack indexed pixels (8-bit indices) into smaller bit depths.
pub fn pack_indexed(data: &[u8], bit_depth: u8) -> Vec<u8> {
    match bit_depth {
        1 => pack_bits(data, 1),
        2 => pack_bits(data, 2),
        4 => pack_bits(data, 4),
        8 => data.to_vec(),
        _ => data.to_vec(),
    }
}

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
}
