//! Error types for the pixo library.

use std::fmt;

/// Result type alias for pixo operations.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors that can occur during image encoding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Invalid image dimensions (zero width or height).
    InvalidDimensions {
        /// Image width.
        width: u32,
        /// Image height.
        height: u32,
    },
    /// Pixel data length doesn't match expected size.
    InvalidDataLength {
        /// Expected number of bytes.
        expected: usize,
        /// Actual number of bytes provided.
        actual: usize,
    },
    /// Invalid quality parameter (must be 1-100 for JPEG).
    InvalidQuality(u8),
    /// Invalid compression level parameter (must be 1-9 for PNG/zlib).
    InvalidCompressionLevel(u8),
    /// Image dimensions exceed maximum supported size.
    ImageTooLarge {
        /// Image width.
        width: u32,
        /// Image height.
        height: u32,
        /// Maximum supported dimension.
        max: u32,
    },
    /// Unsupported color type for the format.
    UnsupportedColorType,
    /// Internal compression error.
    CompressionError(String),
    /// Invalid restart interval parameter (must be 1-65535).
    InvalidRestartInterval(u16),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::InvalidDimensions { width, height } => {
                write!(f, "Invalid image dimensions: {width}x{height}")
            }
            Error::InvalidDataLength { expected, actual } => {
                write!(
                    f,
                    "Invalid pixel data length: expected {expected} bytes, got {actual}",
                )
            }
            Error::InvalidQuality(q) => {
                write!(f, "Invalid quality {q}: must be 1-100")
            }
            Error::InvalidCompressionLevel(level) => {
                write!(f, "Invalid compression level {level}: must be 1-9")
            }
            Error::ImageTooLarge { width, height, max } => {
                write!(f, "Image {width}x{height} exceeds maximum dimension {max}",)
            }
            Error::UnsupportedColorType => {
                write!(f, "Unsupported color type for this format")
            }
            Error::CompressionError(msg) => {
                write!(f, "Compression error: {msg}")
            }
            Error::InvalidRestartInterval(interval) => {
                write!(
                    f,
                    "Invalid restart interval {interval}: must be 1-65535 (or None to disable)"
                )
            }
        }
    }
}

impl std::error::Error for Error {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display_invalid_dimensions() {
        let err = Error::InvalidDimensions {
            width: 0,
            height: 100,
        };
        let msg = format!("{err}");
        assert!(msg.contains("0x100"));
    }

    #[test]
    fn test_error_display_invalid_data_length() {
        let err = Error::InvalidDataLength {
            expected: 100,
            actual: 50,
        };
        let msg = format!("{err}");
        assert!(msg.contains("100"));
        assert!(msg.contains("50"));
    }

    #[test]
    fn test_error_display_invalid_quality() {
        let err = Error::InvalidQuality(0);
        let msg = format!("{err}");
        assert!(msg.contains("0"));
        assert!(msg.contains("1-100"));
    }

    #[test]
    fn test_error_display_invalid_compression_level() {
        let err = Error::InvalidCompressionLevel(10);
        let msg = format!("{err}");
        assert!(msg.contains("10"));
        assert!(msg.contains("1-9"));
    }

    #[test]
    fn test_error_display_image_too_large() {
        let err = Error::ImageTooLarge {
            width: 100000,
            height: 100000,
            max: 65535,
        };
        let msg = format!("{err}");
        assert!(msg.contains("100000"));
        assert!(msg.contains("65535"));
    }

    #[test]
    fn test_error_display_unsupported_color_type() {
        let err = Error::UnsupportedColorType;
        let msg = format!("{err}");
        assert!(msg.contains("Unsupported color type"));
    }

    #[test]
    fn test_error_display_compression_error() {
        let err = Error::CompressionError("test error".to_string());
        let msg = format!("{err}");
        assert!(msg.contains("test error"));
    }

    #[test]
    fn test_error_display_invalid_restart_interval() {
        let err = Error::InvalidRestartInterval(0);
        let msg = format!("{err}");
        assert!(msg.contains("0"));
        assert!(msg.contains("1-65535"));
    }

    #[test]
    fn test_error_is_error_trait() {
        let err: Box<dyn std::error::Error> = Box::new(Error::InvalidQuality(0));
        assert!(err.to_string().contains("Invalid quality"));
    }

    #[test]
    fn test_error_debug() {
        let err = Error::InvalidQuality(50);
        let debug = format!("{err:?}");
        assert!(debug.contains("InvalidQuality"));
    }

    #[test]
    fn test_error_clone_and_eq() {
        let err1 = Error::InvalidQuality(50);
        let err2 = err1.clone();
        assert_eq!(err1, err2);
    }
}
