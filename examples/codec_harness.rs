//! Benchmark harness to compare `comprs` against external codecs (mozjpeg/oxipng).
//!
//! Usage:
//! ```
//! # PNG path (expects oxipng binary if available)
//! cargo run --example codec_harness
//! # Specify fixture dir or binaries
//! FIXTURES=tests/fixtures \
//! OXIPNG_BIN=vendor/oxipng/target/release/oxipng \
//! MOZJPEG_CJPEG=vendor/mozjpeg/cjpeg \
//! cargo run --example codec_harness
//! ```
//!
//! The harness:
//! - Walks the fixture directory (recursively) for images.
//! - For each image, encodes PNG and JPEG with comprs.
//! - If external binaries exist, it runs oxipng for PNG and cjpeg (mozjpeg) for JPEG.
//! - Reports size and elapsed time per encoder.

use std::env;
use std::ffi::OsStr;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use comprs::{jpeg, png, ColorType};
use image::{DynamicImage, GenericImageView, ImageEncoder};

const DEFAULT_FIXTURES: &str = "tests/fixtures";
const DEFAULT_OXIPNG: &str = "vendor/oxipng/target/release/oxipng";
const DEFAULT_CJPEG: &str = "vendor/mozjpeg/cjpeg";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let fixtures_dir = env::var("FIXTURES").unwrap_or_else(|_| DEFAULT_FIXTURES.to_string());
    let oxipng_bin = env::var("OXIPNG_BIN").unwrap_or_else(|_| DEFAULT_OXIPNG.to_string());
    let cjpeg_bin = env::var("MOZJPEG_CJPEG").unwrap_or_else(|_| DEFAULT_CJPEG.to_string());

    let fixtures = collect_fixtures(Path::new(&fixtures_dir))?;
    if fixtures.is_empty() {
        println!("No fixtures found under {fixtures_dir}");
        return Ok(());
    }

    let tmp_dir = PathBuf::from("target/codec-harness");
    fs::create_dir_all(&tmp_dir)?;

    println!("Fixtures: {}", fixtures_dir);
    println!("oxipng: {} ({})", oxipng_bin, availability(&oxipng_bin));
    println!("cjpeg: {} ({})", cjpeg_bin, availability(&cjpeg_bin));
    println!();

    for fixture in fixtures {
        let img = image::open(&fixture)?;
        let (w, h) = img.dimensions();
        println!("=== {} ({}x{}) ===", fixture.display(), w, h);

        run_png_section(&img, &fixture, &tmp_dir, &oxipng_bin)?;
        run_jpeg_section(&img, &fixture, &tmp_dir, &cjpeg_bin)?;

        println!();
    }

    Ok(())
}

fn collect_fixtures(root: &Path) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let mut files = Vec::new();
    recurse_dir(root, &mut files)?;
    // Only keep common image extensions
    files.retain(|p| {
        matches!(
            p.extension().and_then(OsStr::to_str).map(|s| s.to_ascii_lowercase()),
            Some(ref ext) if ["png", "jpg", "jpeg", "webp"].contains(&ext.as_str())
        )
    });
    files.sort();
    Ok(files)
}

fn recurse_dir(dir: &Path, acc: &mut Vec<PathBuf>) -> Result<(), Box<dyn std::error::Error>> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            recurse_dir(&path, acc)?;
        } else {
            acc.push(path);
        }
    }
    Ok(())
}

fn run_png_section(
    img: &DynamicImage,
    source_path: &Path,
    tmp_dir: &Path,
    oxipng_bin: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let rgba = img.to_rgba8();
    let (w, h) = (rgba.width(), rgba.height());

    // comprs encode
    let mut comprs_buf = Vec::new();
    let t0 = Instant::now();
    png::encode_into(
        &mut comprs_buf,
        &rgba,
        w,
        h,
        ColorType::Rgba,
        &png::PngOptions::default(),
    )?;
    let comprs_dur = t0.elapsed();
    println!(
        "PNG comprs (default): {:>8} bytes, {:>6.2} ms",
        comprs_buf.len(),
        to_millis(comprs_dur)
    );

    // Prepare a PNG file for external tools (reuse source if already PNG)
    let png_input = if source_path
        .extension()
        .and_then(OsStr::to_str)
        .map(|s| s.eq_ignore_ascii_case("png"))
        .unwrap_or(false)
    {
        source_path.to_path_buf()
    } else {
        let tmp_png = tmp_dir.join(format!(
            "{}.as_png.png",
            source_path
                .file_stem()
                .and_then(OsStr::to_str)
                .unwrap_or("tmp")
        ));
        let mut file = fs::File::create(&tmp_png)?;
        let encoder = image::codecs::png::PngEncoder::new(&mut file);
        encoder.write_image(rgba.as_raw(), w, h, image::ColorType::Rgba8)?;
        tmp_png
    };

    // oxipng comparison (if available)
    if Path::new(oxipng_bin).exists() {
        let oxi_out = tmp_dir.join(format!(
            "{}.oxipng.png",
            source_path
                .file_stem()
                .and_then(OsStr::to_str)
                .unwrap_or("out")
        ));
        let t1 = Instant::now();
        let status = Command::new(oxipng_bin)
            .args([
                "-o",
                "4",
                "--strip",
                "safe",
                "--out",
                oxi_out.to_str().unwrap(),
                png_input.to_str().unwrap(),
            ])
            .status()?;
        if status.success() {
            let oxi_dur = t1.elapsed();
            let size = fs::metadata(&oxi_out)?.len();
            println!(
                "PNG oxipng (-o4 --strip safe): {:>8} bytes, {:>6.2} ms",
                size,
                to_millis(oxi_dur)
            );
        } else {
            println!("PNG oxipng: failed with status {status:?}");
        }
    } else {
        println!("PNG oxipng: skipped (missing binary at {})", oxipng_bin);
    }

    Ok(())
}

fn run_jpeg_section(
    img: &DynamicImage,
    source_path: &Path,
    tmp_dir: &Path,
    cjpeg_bin: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());

    // comprs JPEG
    let mut comprs_buf = Vec::new();
    let t0 = Instant::now();
    jpeg::encode_with_options_into(
        &mut comprs_buf,
        rgb.as_raw(),
        w,
        h,
        85,
        ColorType::Rgb,
        &jpeg::JpegOptions {
            quality: 85,
            subsampling: jpeg::Subsampling::S444,
            restart_interval: None,
        },
    )?;
    let comprs_dur = t0.elapsed();
    println!(
        "JPG comprs (q85 4:4:4): {:>8} bytes, {:>6.2} ms",
        comprs_buf.len(),
        to_millis(comprs_dur)
    );

    if Path::new(cjpeg_bin).exists() {
        let ppm_path = tmp_dir.join(format!(
            "{}.ppm",
            source_path
                .file_stem()
                .and_then(OsStr::to_str)
                .unwrap_or("tmp")
        ));
        write_ppm(&ppm_path, &rgb)?;

        let mjpeg_out = tmp_dir.join(format!(
            "{}.mozjpeg.jpg",
            source_path
                .file_stem()
                .and_then(OsStr::to_str)
                .unwrap_or("out")
        ));
        let t1 = Instant::now();
        let status = Command::new(cjpeg_bin)
            .args([
                "-quality",
                "85",
                "-optimize",
                "-progressive",
                "-outfile",
                mjpeg_out.to_str().unwrap(),
                ppm_path.to_str().unwrap(),
            ])
            .status()?;
        if status.success() {
            let mjpeg_dur = t1.elapsed();
            let size = fs::metadata(&mjpeg_out)?.len();
            println!(
                "JPG mozjpeg (cjpeg -quality 85 -optimize -progressive): {:>8} bytes, {:>6.2} ms",
                size,
                to_millis(mjpeg_dur)
            );
        } else {
            println!("JPG mozjpeg: failed with status {status:?}");
        }
    } else {
        println!("JPG mozjpeg: skipped (missing binary at {})", cjpeg_bin);
    }

    Ok(())
}

fn write_ppm(path: &Path, rgb: &image::RgbImage) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = fs::File::create(path)?;
    writeln!(file, "P6\n{} {}\n255", rgb.width(), rgb.height())?;
    file.write_all(rgb.as_raw())?;
    Ok(())
}

fn to_millis(dur: std::time::Duration) -> f64 {
    dur.as_secs_f64() * 1000.0
}

fn availability(path: &str) -> &'static str {
    if Path::new(path).exists() {
        "found"
    } else {
        "missing"
    }
}
