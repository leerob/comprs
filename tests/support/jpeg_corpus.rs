//! JPEG Test Corpus.
//!
//! A curated collection of JPEG test images covering various encoding
//! characteristics: baseline, progressive, different subsampling modes,
//! grayscale, and unusual dimensions.
//!
//! Primary source: libjpeg-turbo test images
//! License: libjpeg-turbo is BSD-3-Clause

#![allow(dead_code)]

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use reqwest::blocking::Client;
use sha2::{Digest, Sha256};

/// Baseline JPEG test images from libjpeg-turbo.
/// Format: (filename, url, sha256)
pub const JPEG_BASELINE: &[(&str, &str, &str)] = &[
    (
        "testorig.jpg",
        "https://raw.githubusercontent.com/libjpeg-turbo/libjpeg-turbo/main/testimages/testorig.jpg",
        "1dffbddf2e835d2ca2a3f8b80b9dcd51cb07cdbcdaad34b0bc9b0f0e821c9df5",
    ),
    (
        "testimg.jpg",
        "https://raw.githubusercontent.com/libjpeg-turbo/libjpeg-turbo/main/testimages/testimg.jpg",
        "5e7bba21657fb11e2b3b2bf9d4ac266bf9b1c6820707c91129f7bf3a9d0a1147",
    ),
];

/// JPEG images with arithmetic coding (less common).
pub const JPEG_ARITHMETIC: &[(&str, &str, &str)] = &[(
    "testimgari.jpg",
    "https://raw.githubusercontent.com/libjpeg-turbo/libjpeg-turbo/main/testimages/testimgari.jpg",
    "4672c7f08864cd0a8c73a4fa4b66ca32b635d38464551c1ecf06564ae8c89b38",
)];

/// Progressive JPEG test images (interlaced display).
pub const JPEG_PROGRESSIVE: &[(&str, &str, &str)] = &[(
    "testimgint.jpg",
    "https://raw.githubusercontent.com/libjpeg-turbo/libjpeg-turbo/main/testimages/testimgint.jpg",
    "491679b8057739b3c8e5bacd1e918efb1691d271cbbd69820ff8d480dcb90963",
)];

/// 12-bit JPEG test images.
pub const JPEG_12BIT: &[(&str, &str, &str)] = &[(
    "testorig12.jpg",
    "https://raw.githubusercontent.com/libjpeg-turbo/libjpeg-turbo/main/testimages/testorig12.jpg",
    "34790770f76db8e60a7765b52ca4edf5f16bc21bcb8c6045ca2efef39a8a013e",
)];

/// Special color handling test images.
pub const JPEG_COLOR: &[(&str, &str, &str)] = &[(
    "cram_bgr24.jpg",
    "https://raw.githubusercontent.com/libjpeg-turbo/libjpeg-turbo/main/testimages/cram_bgr24.jpg",
    "a4bd6d7e704901166a6ed422dfc95168d6b243326f6b5e9d626fae0f82b1bfc9",
)];

/// Fetch JPEG corpus images to the specified directory with SHA256 verification.
fn fetch_jpeg_category(
    fixtures_dir: &Path,
    images: &[(&str, &str, &str)],
    client: &Client,
) -> Result<(), String> {
    fs::create_dir_all(fixtures_dir).map_err(|e| e.to_string())?;

    for &(name, url, expected_sha) in images {
        let dest = fixtures_dir.join(name);
        if dest.exists() {
            // Verify existing file
            let existing = fs::read(&dest).map_err(|e| e.to_string())?;
            let mut hasher = Sha256::new();
            hasher.update(&existing);
            let digest = format!("{:x}", hasher.finalize());
            if digest == expected_sha {
                continue;
            }
            // Re-download if hash mismatch
        }

        let resp = client.get(url).send().map_err(|e| e.to_string())?;
        let resp = resp.error_for_status().map_err(|e| e.to_string())?;
        let bytes = resp.bytes().map_err(|e| e.to_string())?.to_vec();

        // Integrity check
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let digest = format!("{:x}", hasher.finalize());
        if digest != expected_sha {
            return Err(format!(
                "SHA mismatch for {name}: expected {expected_sha}, got {digest}"
            ));
        }

        fs::write(&dest, &bytes).map_err(|e| e.to_string())?;
    }

    Ok(())
}

/// Fetch all JPEG corpus images.
pub fn fetch_jpeg_corpus(fixtures_dir: &Path) -> Result<(), String> {
    let client = Client::builder()
        .user_agent("pixo-test/0.1")
        .build()
        .map_err(|e| e.to_string())?;

    fetch_jpeg_category(fixtures_dir, JPEG_BASELINE, &client)?;
    fetch_jpeg_category(fixtures_dir, JPEG_PROGRESSIVE, &client)?;
    fetch_jpeg_category(fixtures_dir, JPEG_COLOR, &client)?;
    // Note: arithmetic and 12-bit JPEGs may not be decodable by all libraries
    // fetch_jpeg_category(fixtures_dir, JPEG_ARITHMETIC, &client)?;
    // fetch_jpeg_category(fixtures_dir, JPEG_12BIT, &client)?;

    Ok(())
}

/// Fetch the full JPEG corpus including edge cases.
pub fn fetch_jpeg_corpus_full(fixtures_dir: &Path) -> Result<(), String> {
    let client = Client::builder()
        .user_agent("pixo-test/0.1")
        .build()
        .map_err(|e| e.to_string())?;

    fetch_jpeg_category(fixtures_dir, JPEG_BASELINE, &client)?;
    fetch_jpeg_category(fixtures_dir, JPEG_PROGRESSIVE, &client)?;
    fetch_jpeg_category(fixtures_dir, JPEG_COLOR, &client)?;
    fetch_jpeg_category(fixtures_dir, JPEG_ARITHMETIC, &client)?;
    fetch_jpeg_category(fixtures_dir, JPEG_12BIT, &client)?;

    Ok(())
}

/// Read standard JPEG corpus images.
pub fn read_jpeg_corpus() -> Result<Vec<(PathBuf, Vec<u8>)>, String> {
    let fixtures_dir = Path::new("tests/fixtures/jpeg_corpus");
    fetch_jpeg_corpus(fixtures_dir)?;

    let mut cases = Vec::new();
    for entry in fs::read_dir(fixtures_dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("jpg") {
            let mut data = Vec::new();
            fs::File::open(&path)
                .map_err(|e| e.to_string())?
                .read_to_end(&mut data)
                .map_err(|e| e.to_string())?;
            cases.push((path, data));
        }
    }

    // Sort by filename for consistent ordering
    cases.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(cases)
}

/// Read baseline JPEG corpus only.
pub fn read_jpeg_baseline() -> Result<Vec<(PathBuf, Vec<u8>)>, String> {
    let fixtures_dir = Path::new("tests/fixtures/jpeg_corpus");
    let client = Client::builder()
        .user_agent("pixo-test/0.1")
        .build()
        .map_err(|e| e.to_string())?;

    fetch_jpeg_category(fixtures_dir, JPEG_BASELINE, &client)?;

    let mut cases = Vec::new();
    for &(name, _, _) in JPEG_BASELINE {
        let path = fixtures_dir.join(name);
        let mut data = Vec::new();
        fs::File::open(&path)
            .map_err(|e| e.to_string())?
            .read_to_end(&mut data)
            .map_err(|e| e.to_string())?;
        cases.push((path, data));
    }
    Ok(cases)
}

/// Decode JPEG corpus images to raw RGB pixels.
/// Returns (filename, width, height, RGB pixels) for each image.
#[allow(clippy::type_complexity)]
pub fn read_jpeg_corpus_decoded() -> Result<Vec<(String, u32, u32, Vec<u8>)>, String> {
    let raw = read_jpeg_corpus()?;
    let mut decoded = Vec::new();

    for (path, bytes) in raw {
        let img = image::load_from_memory(&bytes).map_err(|e| e.to_string())?;
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        decoded.push((name, w, h, rgb.into_raw()));
    }

    Ok(decoded)
}

/// Get count of standard JPEG corpus images.
pub fn jpeg_corpus_count() -> usize {
    JPEG_BASELINE.len() + JPEG_PROGRESSIVE.len() + JPEG_COLOR.len()
}

/// Get count of full JPEG corpus including edge cases.
pub fn jpeg_corpus_full_count() -> usize {
    JPEG_BASELINE.len()
        + JPEG_PROGRESSIVE.len()
        + JPEG_COLOR.len()
        + JPEG_ARITHMETIC.len()
        + JPEG_12BIT.len()
}
