#![allow(dead_code)]

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

#[allow(dead_code)]
use reqwest::blocking::Client;
use sha2::{Digest, Sha256};

// Minimal SHA map (url -> sha256) for integrity. Extend as needed.
#[allow(dead_code)]
const PNGSUITE_SHA256: &[(&str, &str)] = &[
    (
        "basn0g01.png",
        "dcf043b2d8f7d37f564f59c56b2fc4d128be2ee417bf249d0edb991690e54cfb",
    ),
    (
        "basn0g02.png",
        "1dca5ea66c60e1a3c63e99c2242e42ec6e6a3acc3604dc1306469ea48475e19d",
    ),
    (
        "basn0g04.png",
        "4160de3d5b0276e9d7837e6e4757e0f2c76420c0e7740f36f6ae22463fa4fbd3",
    ),
    (
        "basn0g08.png",
        "b0a8784ffc0b2c5be14701d0402516e57c993916e8929e6a1cdc78cda3a95c01",
    ),
    (
        "basn0g16.png",
        "f67f28304de53f6587d30d6a8b97ee09b75c7df0aed8cebd7cb6c9345028051b",
    ),
    (
        "basn2c08.png",
        "22d8ff56a44a72476f0ed1da615386f9c15cfb8d462103d68a26aef2ec0737c5",
    ),
    (
        "basn2c16.png",
        "3f2b36da44cdb54788ce6a35415bfe9a1e19232a8695616241c1bfbc5662e1b9",
    ),
    (
        "basn3p01.png",
        "9fbb8e0bc1ea5b07ad6a98f19fe3c3e298c34a1f0c34434b5a9b9ae2133d5ceb",
    ),
    (
        "basn3p02.png",
        "e06e1c0f1c10b5b63a5078fbddf6b2f389f9b822b800812a4bd8b9f016ac10c6",
    ),
    (
        "basn3p04.png",
        "48c2ce49963a869c6eb4c59298c54d7b3846e7bba61f6c86dc16c6c52ffecf5e",
    ),
    (
        "basn3p08.png",
        "e890c3f55a8da2b7c285fbcf699d00e52176bfd0d97779a565a065e1ec8be6da",
    ),
    (
        "basn4a08.png",
        "4c8a6c2c2321dfe904827db56cc169968ab12572980732d0f79c027c703abadd",
    ),
    (
        "basn4a16.png",
        "a22e42fec9a8cc85a2a7e6658ebf0f1d8c569b1440befcedb2cf7aaba2698dfb",
    ),
    (
        "basn6a08.png",
        "7907c412cb6ef0ada9e2b8949d74a030a7701c46f5bdbddae8a5f1af0c2c16d6",
    ),
    (
        "basn6a16.png",
        "bda76b5f268a9bbd3e26343da9db7348e445488265867beb1412ba3832b885c9",
    ),
];

pub fn fetch_pngsuite(fixtures_dir: &Path) -> Result<(), String> {
    let client = Client::builder()
        .user_agent("comprs-test/0.1")
        .build()
        .map_err(|e| e.to_string())?;

    fs::create_dir_all(fixtures_dir).map_err(|e| e.to_string())?;

    for &(name, url_sha) in PNGSUITE_SHA256 {
        let url = format!("https://raw.githubusercontent.com/glennrp/pngsuite/master/{name}");
        let dest = fixtures_dir.join(name);
        if dest.exists() {
            continue;
        }

        let resp = client.get(&url).send().map_err(|e| e.to_string())?;
        let resp = resp.error_for_status().map_err(|e| e.to_string())?;
        let bytes = resp.bytes().map_err(|e| e.to_string())?.to_vec();

        // Integrity check
        let mut hasher = Sha256::new();
        hasher.update(&bytes);
        let digest = format!("{:x}", hasher.finalize());
        if digest != url_sha {
            return Err(format!(
                "SHA mismatch for {name}: expected {url_sha}, got {digest}"
            ));
        }

        fs::write(&dest, &bytes).map_err(|e| e.to_string())?;
    }

    Ok(())
}

pub fn read_pngsuite() -> Result<Vec<(PathBuf, Vec<u8>)>, String> {
    let fixtures_dir = Path::new("tests/fixtures/pngsuite");
    if let Err(e) = fetch_pngsuite(fixtures_dir) {
        return Err(format!("fetch_pngsuite failed: {e}"));
    }

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
    Ok(cases)
}
