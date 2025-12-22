//! Paeth predictor in a feature-agnostic module.
//!
//! This scalar implementation is used by PNG filtering and remains available
//! regardless of whether SIMD acceleration is enabled. SIMD paths delegate to
//! this predictor to ensure consistent behavior across feature combinations.

/// Scalar Paeth predictor (PNG spec).
#[inline]
pub fn paeth_predictor(a: u8, b: u8, c: u8) -> u8 {
    let a = a as i16;
    let b = b as i16;
    let c = c as i16;

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
