use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

/// Minimal PNG IHDR metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PngHeader {
    pub width: u32,
    pub height: u32,
    pub bit_depth: u8,
    pub color_type: u8,
}

/// Read the PNG IHDR chunk and return its metadata.
///
/// This is a tiny, dependency-free parser sufficient for tests that only need
/// the header fields (width, height, bit depth, color type).
pub fn read_png_header(path: &Path) -> io::Result<PngHeader> {
    let mut file = File::open(path)?;
    let mut sig = [0u8; 8];
    file.read_exact(&mut sig)?;
    const PNG_SIG: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    if sig != PNG_SIG {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a PNG file",
        ));
    }

    // Read length (4) + type (4) for IHDR
    let mut len_buf = [0u8; 4];
    file.read_exact(&mut len_buf)?;
    let ihdr_len = u32::from_be_bytes(len_buf);
    if ihdr_len != 13 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "IHDR length must be 13",
        ));
    }

    let mut ctype = [0u8; 4];
    file.read_exact(&mut ctype)?;
    if &ctype != b"IHDR" {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "missing IHDR chunk",
        ));
    }

    let mut ihdr_data = [0u8; 13];
    file.read_exact(&mut ihdr_data)?;
    // Skip CRC
    let mut _crc = [0u8; 4];
    file.read_exact(&mut _crc)?;

    let width = u32::from_be_bytes([ihdr_data[0], ihdr_data[1], ihdr_data[2], ihdr_data[3]]);
    let height = u32::from_be_bytes([ihdr_data[4], ihdr_data[5], ihdr_data[6], ihdr_data[7]]);
    let bit_depth = ihdr_data[8];
    let color_type = ihdr_data[9];

    Ok(PngHeader {
        width,
        height,
        bit_depth,
        color_type,
    })
}
