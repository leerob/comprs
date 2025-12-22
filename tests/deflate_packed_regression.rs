//! Regression checks for the packed DEFLATE path to ensure zlib streams are valid.

use comprs::compress::deflate::deflate_zlib_packed;
use comprs::png::{filter, PngOptions};
use comprs::ColorType;
use flate2::read::ZlibDecoder;
use image::GenericImageView;
use std::fs;
use std::io::Read;

fn decompress_zlib(data: &[u8]) -> Vec<u8> {
    let mut decoder = ZlibDecoder::new(data);
    let mut out = Vec::new();
    decoder.read_to_end(&mut out).expect("decompress");
    out
}

fn filtered_pixels(path: &str, color_type: ColorType) -> Option<(Vec<u8>, usize)> {
    let bytes = fs::read(path).ok()?;
    let img = image::load_from_memory(&bytes).ok()?;
    let (w, h) = img.dimensions();
    let bpp = color_type.bytes_per_pixel();
    let pixels = match color_type {
        ColorType::Rgb => img.to_rgb8().into_raw(),
        ColorType::Rgba => img.to_rgba8().into_raw(),
        _ => return None,
    };
    let filtered = filter::apply_filters(&pixels, w, h, bpp, &PngOptions::default());
    Some((filtered, bpp))
}

#[test]
fn deflate_packed_roundtrip_real_fixtures() {
    let fixtures = [
        ("tests/fixtures/multi-agent.jpg", ColorType::Rgb),
        ("tests/fixtures/playground.png", ColorType::Rgba),
    ];

    for (path, ct) in fixtures {
        let Some((filtered, _bpp)) = filtered_pixels(path, ct) else {
            eprintln!("Skipping deflate packed regression for {path}: missing or unreadable");
            continue;
        };

        let compressed = deflate_zlib_packed(&filtered, 6);
        let decompressed = decompress_zlib(&compressed);
        assert_eq!(
            decompressed, filtered,
            "deflate_zlib_packed corrupted stream for {path}"
        );
    }
}
