//! Kodak Lossless True Color Image Suite.
//!
//! The Kodak suite is an industry-standard set of 24 photographic images
//! commonly used for benchmarking image compression algorithms.
//!
//! Source: https://r0k.us/graphics/kodak/
//! License: Public domain / unrestricted use

#![allow(dead_code)]

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use reqwest::blocking::Client;
use sha2::{Digest, Sha256};

/// Kodak image metadata: (filename, sha256)
/// All images are 768x512 or 512x768, 24-bit RGB PNG.
pub const KODAK_IMAGES: &[(&str, &str)] = &[
    (
        "kodim01.png",
        "a56e27cbf5f843c048b6af1d6e090760e9c92fadba88b7dee0205918a37523bd",
    ),
    (
        "kodim02.png",
        "4f4b74a79237e311d72cad958237b5f7088d8bce1c82305ebefe1a70e3022dfd",
    ),
    (
        "kodim03.png",
        "e25ca1ff2f0c0cb5fdfd5f9b0a0bb21ac4c3de3c84a67f35b09a85d3306249db",
    ),
    (
        "kodim04.png",
        "e3b946107c5d3441c022f678d0c3caf1e224d81b1604ba840a4f88e562de61aa",
    ),
    (
        "kodim05.png",
        "10349e963c5c813d327852f82c1795fa4148d69fedffc4c589bee458e3ac3d53",
    ),
    (
        "kodim06.png",
        "363510303b715d4cbc384e1ce227e466b613a09e1b71ae985882bf8e7fbd9b18",
    ),
    (
        "kodim07.png",
        "b77d3f006f42414bb242222e0482e750c0fb9e5ee8d4bed2f6f11c5605fe54a4",
    ),
    (
        "kodim08.png",
        "ba23983c76b4832ee0e8af0592664756841a16779acd69f792e268fb6d13d6e7",
    ),
    (
        "kodim09.png",
        "6a4361c2fc194feb4edaa9f9a4a0620fb9943e460ac7fdf037fb0f6dd6607a7d",
    ),
    (
        "kodim10.png",
        "9dfb70f5867c29ff9ed6313683f19b3d867849e40fbc0c4c54a4a89df341cf23",
    ),
    (
        "kodim11.png",
        "7936814b58b5387fce2e4e2488b4ec830dadd95fa9520f358ddb30990b50f2b6",
    ),
    (
        "kodim12.png",
        "d78c37c2f04f23761ed2367dd77e2db584ddd4c3950833fecf89f199a8126980",
    ),
    (
        "kodim13.png",
        "bc34a3ce58dea09dce1704c997171602de90cb34d0c8503a988b77f473d39b08",
    ),
    (
        "kodim14.png",
        "55a94550ff18f3246c4074fd32b77b0c74447c26b6ad274d564d999c0450ba6e",
    ),
    (
        "kodim15.png",
        "7538cbb80cb9103606c48b806eae57d56c885c7f90b9b3be70a41160f9cbb683",
    ),
    (
        "kodim16.png",
        "a89c7268ccd4718ba424a99fc4643c572cf692ca6eae887185ceb4e9f11d2e54",
    ),
    (
        "kodim17.png",
        "37afcc89fbdcb76d9518e04b2fc011027e2f4cd14b3b2f83cefd721641a47c5b",
    ),
    (
        "kodim18.png",
        "1a9258c365988961d87a0598725b609139c303ad48a5aad6c503c3b1a87849aa",
    ),
    (
        "kodim19.png",
        "b7450b264b1b0a411390d8931b112c27905a992520fc90569dc4b920aa32bbdc",
    ),
    (
        "kodim20.png",
        "3b46c71e3b92a563820ba32936be8330c586c41f938efd94be938386aae4328a",
    ),
    (
        "kodim21.png",
        "ac958597c82073f6bb65129c68f72b651db5b9efd82e11547d07350214bc268b",
    ),
    (
        "kodim22.png",
        "1cee58eb1f2d9c7ebb254d208a03c783ce6cf2c4d8c2cf45e235dd23b4ce1b29",
    ),
    (
        "kodim23.png",
        "e3111a2fd4da24af15d6459ef9eacfe54106b38e27b4a21821b75c3f5d2d5baf",
    ),
    (
        "kodim24.png",
        "1071c68372cc5a01435c2c225a5cf7d4bb803846ec08bb6b3d6721b156d7cb96",
    ),
];

const KODAK_BASE_URL: &str = "https://r0k.us/graphics/kodak/kodak";

/// Fetch all Kodak images to the specified directory with SHA256 verification.
pub fn fetch_kodak(fixtures_dir: &Path) -> Result<(), String> {
    let client = Client::builder()
        .user_agent("pixo-test/0.1")
        .build()
        .map_err(|e| e.to_string())?;

    fs::create_dir_all(fixtures_dir).map_err(|e| e.to_string())?;

    for &(name, expected_sha) in KODAK_IMAGES {
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

        let url = format!("{KODAK_BASE_URL}/{name}");
        let resp = client.get(&url).send().map_err(|e| e.to_string())?;
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

/// Read all Kodak images from the fixtures directory.
/// Returns (path, raw PNG bytes) for each image.
pub fn read_kodak() -> Result<Vec<(PathBuf, Vec<u8>)>, String> {
    let fixtures_dir = Path::new("tests/fixtures/kodak");
    fetch_kodak(fixtures_dir)?;

    let mut cases = Vec::new();
    for entry in fs::read_dir(fixtures_dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) == Some("png") {
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

/// Read a subset of Kodak images for faster benchmarks.
/// Returns first N images.
pub fn read_kodak_subset(count: usize) -> Result<Vec<(PathBuf, Vec<u8>)>, String> {
    let all = read_kodak()?;
    Ok(all.into_iter().take(count).collect())
}

/// Decode Kodak images to raw RGB pixels.
/// Returns (filename, width, height, RGB pixels) for each image.
#[allow(clippy::type_complexity)]
pub fn read_kodak_decoded() -> Result<Vec<(String, u32, u32, Vec<u8>)>, String> {
    let raw = read_kodak()?;
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

/// Decode a subset of Kodak images to raw RGB pixels.
#[allow(clippy::type_complexity)]
pub fn read_kodak_decoded_subset(count: usize) -> Result<Vec<(String, u32, u32, Vec<u8>)>, String> {
    let all = read_kodak_decoded()?;
    Ok(all.into_iter().take(count).collect())
}
