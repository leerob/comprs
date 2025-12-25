//! Color type definitions and conversions.

/// Supported color types for image encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ColorType {
    /// Grayscale, 1 byte per pixel.
    Gray,
    /// Grayscale with alpha, 2 bytes per pixel.
    GrayAlpha,
    /// RGB, 3 bytes per pixel.
    Rgb,
    /// RGBA, 4 bytes per pixel.
    Rgba,
}

impl ColorType {
    /// Returns the number of bytes per pixel for this color type.
    #[inline]
    pub const fn bytes_per_pixel(self) -> usize {
        match self {
            ColorType::Gray => 1,
            ColorType::GrayAlpha => 2,
            ColorType::Rgb => 3,
            ColorType::Rgba => 4,
        }
    }

    /// Returns the PNG color type value.
    #[inline]
    pub(crate) const fn png_color_type(self) -> u8 {
        match self {
            ColorType::Gray => 0,
            ColorType::GrayAlpha => 4,
            ColorType::Rgb => 2,
            ColorType::Rgba => 6,
        }
    }

    /// Returns the bit depth for PNG encoding.
    #[inline]
    pub(crate) const fn png_bit_depth(self) -> u8 {
        8 // We only support 8-bit depth
    }
}

/// Convert RGB to YCbCr color space (used by JPEG).
///
/// Returns (Y, Cb, Cr) where each component is in range 0-255.
/// Uses fixed-point integer arithmetic for performance (no floating point).
///
/// ITU-R BT.601 conversion coefficients scaled by 256:
/// - Y  = 0.299*R + 0.587*G + 0.114*B  -> (77*R + 150*G + 29*B + 128) >> 8
/// - Cb = -0.169*R - 0.331*G + 0.5*B + 128
/// - Cr = 0.5*R - 0.419*G - 0.081*B + 128
#[inline]
pub fn rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let r = r as i32;
    let g = g as i32;
    let b = b as i32;

    // Fixed-point coefficients (scaled by 256)
    // +128 for rounding before right shift
    let y = (77 * r + 150 * g + 29 * b + 128) >> 8;
    let cb = ((-43 * r - 85 * g + 128 * b + 128) >> 8) + 128;
    let cr = ((128 * r - 107 * g - 21 * b + 128) >> 8) + 128;

    // Clamp to valid range (the math should keep values in range, but be safe)
    (
        y.clamp(0, 255) as u8,
        cb.clamp(0, 255) as u8,
        cr.clamp(0, 255) as u8,
    )
}

/// Convert RGBA to YCbCr, ignoring alpha channel.
#[inline]
pub fn rgba_to_ycbcr(r: u8, g: u8, b: u8, _a: u8) -> (u8, u8, u8) {
    rgb_to_ycbcr(r, g, b)
}

impl TryFrom<u8> for ColorType {
    type Error = u8;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(ColorType::Gray),
            1 => Ok(ColorType::GrayAlpha),
            2 => Ok(ColorType::Rgb),
            3 => Ok(ColorType::Rgba),
            other => Err(other),
        }
    }
}

impl From<ColorType> for u8 {
    fn from(color: ColorType) -> Self {
        color as u8
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytes_per_pixel() {
        assert_eq!(ColorType::Gray.bytes_per_pixel(), 1);
        assert_eq!(ColorType::GrayAlpha.bytes_per_pixel(), 2);
        assert_eq!(ColorType::Rgb.bytes_per_pixel(), 3);
        assert_eq!(ColorType::Rgba.bytes_per_pixel(), 4);
    }

    #[test]
    fn test_rgb_to_ycbcr_black() {
        let (y, cb, cr) = rgb_to_ycbcr(0, 0, 0);
        assert_eq!(y, 0);
        assert_eq!(cb, 128);
        assert_eq!(cr, 128);
    }

    #[test]
    fn test_rgb_to_ycbcr_white() {
        let (y, cb, cr) = rgb_to_ycbcr(255, 255, 255);
        assert_eq!(y, 255);
        assert_eq!(cb, 128);
        assert_eq!(cr, 128);
    }

    #[test]
    fn test_rgb_to_ycbcr_red() {
        let (y, cb, cr) = rgb_to_ycbcr(255, 0, 0);
        // Red should have high Y, low Cb, high Cr
        assert!(y > 50 && y < 100);
        assert!(cb < 128);
        assert!(cr > 200);
    }

    #[test]
    fn test_color_type_try_from() {
        assert!(matches!(ColorType::try_from(0), Ok(ColorType::Gray)));
        assert!(matches!(ColorType::try_from(1), Ok(ColorType::GrayAlpha)));
        assert!(matches!(ColorType::try_from(2), Ok(ColorType::Rgb)));
        assert!(matches!(ColorType::try_from(3), Ok(ColorType::Rgba)));
        assert!(ColorType::try_from(99).is_err());
    }

    #[test]
    fn test_color_type_roundtrip_u8() {
        for (val, ct) in [
            (0u8, ColorType::Gray),
            (1u8, ColorType::GrayAlpha),
            (2u8, ColorType::Rgb),
            (3u8, ColorType::Rgba),
        ] {
            assert_eq!(u8::from(ct), val);
            assert_eq!(ColorType::try_from(val).unwrap(), ct);
        }
    }
}
