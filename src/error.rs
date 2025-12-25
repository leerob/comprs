//! Error types for the comprs library.

use std::fmt;

/// Result type alias for comprs operations.
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
