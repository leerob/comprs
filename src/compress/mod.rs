//! Compression algorithms.
//!
//! For background on how these pieces fit together, see
//! [`crate::guides::deflate`], [`crate::guides::huffman_coding`],
//! and [`crate::guides::lz77_compression`].

pub mod adler32;
pub mod crc32;
pub mod deflate;
pub mod huffman;
pub mod lz77;

pub use adler32::adler32;
pub use crc32::crc32;
pub use deflate::{deflate, deflate_packed, deflate_zlib, deflate_zlib_packed, Deflater};
