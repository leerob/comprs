//! Comprehensive library comparison benchmark.
//!
//! Compares pixo against popular image compression libraries and external tools,
//! including oxipng for PNG and mozjpeg for JPEG.
//!
//! Run with: cargo bench --bench comparison
//!
//! For a quick summary without full benchmarks:
//!   cargo bench --bench comparison -- --summary-only

use std::fs;
use std::io::Write as IoWrite;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use criterion::{black_box, criterion_group, BenchmarkId, Criterion, Throughput};
use flate2::{write::ZlibEncoder, Compression};
use image::ImageEncoder;

use jpeg_encoder::{ColorType as JpegColorType, Encoder as JpegEncoderCrate, SamplingFactor};
use libdeflater::{CompressionLvl, Compressor as LibdeflateCompressor};

use pixo::compress::deflate::deflate_zlib;
use pixo::png::{QuantizationMode, QuantizationOptions};
use pixo::{jpeg, png, ColorType};

// ============================================================================
// Test Data Generation
// ============================================================================

fn generate_gradient_image(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    for y in 0..height {
        for x in 0..width {
            let r = ((x * 255) / width) as u8;
            let g = ((y * 255) / height) as u8;
            let b = (((x + y) * 127) / (width + height)) as u8;
            pixels.extend_from_slice(&[r, g, b]);
        }
    }
    pixels
}

#[allow(dead_code)]
fn generate_noisy_image(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    let mut seed = 12345u32;
    for _ in 0..(width * height) {
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let r = (seed >> 16) as u8;
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let g = (seed >> 16) as u8;
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let b = (seed >> 16) as u8;
        pixels.extend_from_slice(&[r, g, b]);
    }
    pixels
}

fn make_compressible(len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    let pattern = b"The quick brown fox jumps over the lazy dog. ";
    while out.len() < len {
        out.extend_from_slice(pattern);
    }
    out.truncate(len);
    out
}

fn make_random(len: usize, mut seed: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        out.push((seed >> 16) as u8);
    }
    out.truncate(len);
    out
}

/// Generate flat color blocks image (tests RLE/palette efficiency)
fn generate_flat_blocks_image(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    let colors: [[u8; 3]; 4] = [
        [255, 0, 0],   // Red
        [0, 255, 0],   // Green
        [0, 0, 255],   // Blue
        [255, 255, 0], // Yellow
    ];
    let block_w = width / 2;
    let block_h = height / 2;
    for y in 0..height {
        for x in 0..width {
            let block_x = if x < block_w { 0 } else { 1 };
            let block_y = if y < block_h { 0 } else { 1 };
            let color_idx = block_y * 2 + block_x;
            pixels.extend_from_slice(&colors[color_idx]);
        }
    }
    pixels
}

/// Generate text-like pattern (high contrast edges, typical of screenshots)
#[allow(dead_code)]
fn generate_text_pattern_image(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    let mut seed = 42u32;
    for y in 0..height {
        for x in 0..width {
            // Create horizontal "text lines" with noise
            let line_y = y % 16;
            let is_text_line = (4..=12).contains(&line_y);
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let noise = ((seed >> 20) % 3) as u8;

            let val = if is_text_line && (x % 8 < 6) {
                // Dark text on light background with slight variation
                20 + noise * 5
            } else {
                // Light background
                240 - noise * 3
            };
            pixels.extend_from_slice(&[val, val, val]);
        }
    }
    pixels
}

// ============================================================================
// Test Image Infrastructure
// ============================================================================

/// Image category for benchmarking different content types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum ImageCategory {
    Photo,        // Natural photographs (Kodak, review.jpg, web.jpg)
    GraphicsIcon, // Icons and graphics (rocket.png, avatar-color.png)
    Screenshot,   // UI elements, text (browser.jpg)
    Synthetic,    // Generated test patterns
}

/// Test image descriptor
struct TestImage {
    name: &'static str,
    category: ImageCategory,
    path: &'static str,
}

/// Curated list of test images covering different content types
const TEST_IMAGES: &[TestImage] = &[
    // Photos (complex, natural images)
    TestImage {
        name: "kodim01",
        category: ImageCategory::Photo,
        path: "tests/fixtures/kodak/kodim01.png",
    },
    TestImage {
        name: "kodim03",
        category: ImageCategory::Photo,
        path: "tests/fixtures/kodak/kodim03.png",
    },
    TestImage {
        name: "kodim23",
        category: ImageCategory::Photo,
        path: "tests/fixtures/kodak/kodim23.png",
    },
    // Graphics/Icons (flat colors, transparency)
    TestImage {
        name: "rocket",
        category: ImageCategory::GraphicsIcon,
        path: "tests/fixtures/rocket.png",
    },
    TestImage {
        name: "avatar",
        category: ImageCategory::GraphicsIcon,
        path: "tests/fixtures/avatar-color.png",
    },
];

/// Loaded test image with pixel data
struct LoadedImage {
    name: String,
    category: ImageCategory,
    width: u32,
    height: u32,
    pixels: Vec<u8>,
}

/// Load all available test images
fn load_test_images() -> Vec<LoadedImage> {
    let mut images = Vec::new();

    for test_img in TEST_IMAGES {
        let path = std::path::Path::new(test_img.path);
        if !path.exists() {
            continue;
        }

        if let Ok(data) = fs::read(path) {
            if let Ok(img) = image::load_from_memory(&data) {
                let rgb = img.to_rgb8();
                let (w, h) = (rgb.width(), rgb.height());
                images.push(LoadedImage {
                    name: test_img.name.to_string(),
                    category: test_img.category,
                    width: w,
                    height: h,
                    pixels: rgb.into_raw(),
                });
            }
        }
    }

    images
}

/// Load test images filtered by category
#[allow(dead_code)]
fn load_images_by_category(category: ImageCategory) -> Vec<LoadedImage> {
    load_test_images()
        .into_iter()
        .filter(|img| img.category == category)
        .collect()
}

// ============================================================================
// External Tool Detection (merged from codec_harness.rs)
// ============================================================================

fn find_oxipng() -> Option<PathBuf> {
    // Check common installation paths
    let paths = [
        "vendor/oxipng/target/release/oxipng",
        "/usr/local/bin/oxipng",
        "/opt/homebrew/bin/oxipng",
        "/usr/bin/oxipng",
    ];
    if let Some(path) = paths.iter().find(|p| Path::new(p).exists()) {
        return Some(PathBuf::from(path));
    }

    // Check ~/.cargo/bin (where cargo install puts binaries)
    if let Ok(home) = std::env::var("HOME") {
        let cargo_bin = PathBuf::from(home).join(".cargo/bin/oxipng");
        if cargo_bin.exists() {
            return Some(cargo_bin);
        }
    }

    // Try PATH lookup via which
    if let Ok(output) = std::process::Command::new("which").arg("oxipng").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Some(PathBuf::from(path));
            }
        }
    }

    None
}

fn find_cjpeg() -> Option<PathBuf> {
    // Check common installation paths
    let paths = [
        "vendor/mozjpeg/build/cjpeg",
        "vendor/mozjpeg/cjpeg",
        "/usr/local/bin/cjpeg",
        "/opt/homebrew/bin/cjpeg",
        "/usr/bin/cjpeg",
    ];
    if let Some(path) = paths.iter().find(|p| Path::new(p).exists()) {
        return Some(PathBuf::from(path));
    }

    // Try PATH lookup via which
    if let Ok(output) = std::process::Command::new("which").arg("cjpeg").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Some(PathBuf::from(path));
            }
        }
    }

    None
}

fn find_pngquant() -> Option<PathBuf> {
    // Check common installation paths
    let paths = [
        "/usr/local/bin/pngquant",
        "/opt/homebrew/bin/pngquant",
        "/usr/bin/pngquant",
    ];
    if let Some(path) = paths.iter().find(|p| Path::new(p).exists()) {
        return Some(PathBuf::from(path));
    }

    // Check ~/.cargo/bin (where cargo install puts binaries)
    if let Ok(home) = std::env::var("HOME") {
        let cargo_bin = PathBuf::from(home).join(".cargo/bin/pngquant");
        if cargo_bin.exists() {
            return Some(cargo_bin);
        }
    }

    // Try PATH lookup via which
    if let Ok(output) = std::process::Command::new("which").arg("pngquant").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return Some(PathBuf::from(path));
            }
        }
    }

    None
}

// ============================================================================
// External Tool Comparison (merged from codec_harness.rs)
// ============================================================================

/// Encode PNG with oxipng and return (size, duration)
fn encode_with_oxipng(
    pixels: &[u8],
    width: u32,
    height: u32,
    tmp_dir: &Path,
) -> Option<(usize, Duration)> {
    let oxipng_bin = find_oxipng()?;

    // Write input PNG using image crate
    let input_path = tmp_dir.join("oxipng_input.png");
    let output_path = tmp_dir.join("oxipng_output.png");

    {
        let mut file = fs::File::create(&input_path).ok()?;
        let encoder = image::codecs::png::PngEncoder::new(&mut file);
        encoder
            .write_image(pixels, width, height, image::ColorType::Rgb8)
            .ok()?;
    }

    let start = Instant::now();
    let status = Command::new(&oxipng_bin)
        .args([
            "-o",
            "4",
            "--strip",
            "safe",
            "--out",
            output_path.to_str()?,
            input_path.to_str()?,
        ])
        .output()
        .ok()?;
    let duration = start.elapsed();

    if status.status.success() {
        let size = fs::metadata(&output_path).ok()?.len() as usize;
        Some((size, duration))
    } else {
        None
    }
}

/// Write PPM file for mozjpeg input
fn write_ppm(path: &Path, pixels: &[u8], width: u32, height: u32) -> std::io::Result<()> {
    let mut file = fs::File::create(path)?;
    writeln!(file, "P6\n{width} {height}\n255")?;
    file.write_all(pixels)?;
    Ok(())
}

/// Encode JPEG with mozjpeg and return (size, duration)
fn encode_with_mozjpeg(
    pixels: &[u8],
    width: u32,
    height: u32,
    tmp_dir: &Path,
) -> Option<(usize, Duration)> {
    let cjpeg_bin = find_cjpeg()?;

    let ppm_path = tmp_dir.join("mozjpeg_input.ppm");
    let jpg_path = tmp_dir.join("mozjpeg_output.jpg");

    write_ppm(&ppm_path, pixels, width, height).ok()?;

    let start = Instant::now();
    let status = Command::new(&cjpeg_bin)
        .args([
            "-quality",
            "85",
            "-optimize",
            "-progressive",
            "-outfile",
            jpg_path.to_str()?,
            ppm_path.to_str()?,
        ])
        .output()
        .ok()?;
    let duration = start.elapsed();

    if status.status.success() {
        let size = fs::metadata(&jpg_path).ok()?.len() as usize;
        Some((size, duration))
    } else {
        None
    }
}

/// Encode PNG with pngquant (lossy) and return (size, duration)
fn encode_with_pngquant(
    pixels: &[u8],
    width: u32,
    height: u32,
    tmp_dir: &Path,
) -> Option<(usize, Duration)> {
    let pngquant_bin = find_pngquant()?;

    // Write input PNG using image crate
    let input_path = tmp_dir.join("pngquant_input.png");
    let output_path = tmp_dir.join("pngquant_output.png");

    {
        let mut file = fs::File::create(&input_path).ok()?;
        let encoder = image::codecs::png::PngEncoder::new(&mut file);
        encoder
            .write_image(pixels, width, height, image::ColorType::Rgb8)
            .ok()?;
    }

    let start = Instant::now();
    let status = Command::new(&pngquant_bin)
        .args([
            "--quality=65-80",
            "--speed=4",
            "--force",
            "--output",
            output_path.to_str()?,
            input_path.to_str()?,
        ])
        .output()
        .ok()?;
    let duration = start.elapsed();

    if status.status.success() {
        let size = fs::metadata(&output_path).ok()?.len() as usize;
        Some((size, duration))
    } else {
        None
    }
}

/// Quantize with imagequant crate and encode as PNG
fn encode_with_imagequant(pixels: &[u8], width: u32, height: u32) -> Option<(usize, Duration)> {
    use imagequant::RGBA;

    let start = Instant::now();

    // Create RGBA buffer (imagequant requires Vec<RGBA>)
    let rgba: Vec<RGBA> = pixels
        .chunks_exact(3)
        .map(|chunk| RGBA::new(chunk[0], chunk[1], chunk[2], 255))
        .collect();

    let mut liq = imagequant::new();
    liq.set_quality(0, 80).ok()?;

    let mut img = liq
        .new_image(rgba, width as usize, height as usize, 0.0)
        .ok()?;

    let mut res = liq.quantize(&mut img).ok()?;
    res.set_dithering_level(1.0).ok()?;

    let (palette, indices) = res.remapped(&mut img).ok()?;

    // Convert palette to RGB format for PNG encoding
    let rgb_palette: Vec<[u8; 3]> = palette.iter().map(|c| [c.r, c.g, c.b]).collect();
    let alpha: Vec<u8> = palette.iter().map(|c| c.a).collect();

    // Use pixo to encode the indexed PNG
    let mut opts = png::PngOptions::balanced();
    opts.quantization.mode = QuantizationMode::Off; // Already quantized

    let png_data = png::encode_indexed_with_options(
        &indices,
        width,
        height,
        &rgb_palette,
        Some(&alpha),
        &opts,
    )
    .ok()?;

    let duration = start.elapsed();
    Some((png_data.len(), duration))
}

// ============================================================================
// PNG Encoding Comparison (all presets)
// ============================================================================

#[allow(clippy::single_element_loop)]
fn bench_png_all_presets(c: &mut Criterion) {
    let mut group = c.benchmark_group("PNG All Presets");

    for size in [512].iter() {
        let gradient = generate_gradient_image(*size, *size);
        let pixel_bytes = (*size as u64) * (*size as u64) * 3;

        group.throughput(Throughput::Bytes(pixel_bytes));

        let mut png_buf = Vec::new();

        // pixo Fast preset
        group.bench_with_input(
            BenchmarkId::new("pixo_fast", format!("{size}x{size}")),
            &gradient,
            |b, pixels| {
                b.iter(|| {
                    png::encode_into(
                        &mut png_buf,
                        black_box(pixels),
                        *size,
                        *size,
                        ColorType::Rgb,
                        &png::PngOptions::fast(),
                    )
                    .unwrap()
                });
            },
        );

        // pixo Balanced preset
        group.bench_with_input(
            BenchmarkId::new("pixo_balanced", format!("{size}x{size}")),
            &gradient,
            |b, pixels| {
                b.iter(|| {
                    png::encode_into(
                        &mut png_buf,
                        black_box(pixels),
                        *size,
                        *size,
                        ColorType::Rgb,
                        &png::PngOptions::balanced(),
                    )
                    .unwrap()
                });
            },
        );

        // pixo Max preset (skip in normal benchmarks - too slow)
        // group.bench_with_input(
        //     BenchmarkId::new("pixo_max", format!("{size}x{size}")),
        //     &gradient,
        //     |b, pixels| {
        //         b.iter(|| {
        //             png::encode_into(
        //                 &mut png_buf,
        //                 black_box(pixels),
        //                 *size,
        //                 *size,
        //                 ColorType::Rgb,
        //                 &png::PngOptions::max(),
        //             )
        //             .unwrap()
        //         });
        //     },
        // );

        // image crate
        group.bench_with_input(
            BenchmarkId::new("image_crate", format!("{size}x{size}")),
            &gradient,
            |b, pixels| {
                b.iter(|| {
                    let mut output = Vec::new();
                    let encoder = image::codecs::png::PngEncoder::new(&mut output);
                    encoder
                        .write_image(black_box(pixels), *size, *size, image::ColorType::Rgb8)
                        .unwrap();
                    output
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// PNG Lossy Comparison (quantization)
// ============================================================================

#[allow(clippy::single_element_loop)]
fn bench_png_lossy_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("PNG Lossy Comparison");

    for size in [512].iter() {
        let gradient = generate_gradient_image(*size, *size);
        let pixel_bytes = (*size as u64) * (*size as u64) * 3;

        group.throughput(Throughput::Bytes(pixel_bytes));

        let mut png_buf = Vec::new();

        // pixo lossless (baseline)
        group.bench_with_input(
            BenchmarkId::new("pixo_lossless", format!("{size}x{size}")),
            &gradient,
            |b, pixels| {
                b.iter(|| {
                    png::encode_into(
                        &mut png_buf,
                        black_box(pixels),
                        *size,
                        *size,
                        ColorType::Rgb,
                        &png::PngOptions::balanced(),
                    )
                    .unwrap()
                });
            },
        );

        // pixo lossy with auto quantization
        let mut lossy_opts = png::PngOptions::balanced();
        lossy_opts.quantization = QuantizationOptions {
            mode: QuantizationMode::Auto,
            max_colors: 256,
            dithering: false,
        };
        group.bench_with_input(
            BenchmarkId::new("pixo_lossy_auto", format!("{size}x{size}")),
            &gradient,
            |b, pixels| {
                b.iter(|| {
                    png::encode_into(
                        &mut png_buf,
                        black_box(pixels),
                        *size,
                        *size,
                        ColorType::Rgb,
                        &lossy_opts,
                    )
                    .unwrap()
                });
            },
        );

        // pixo lossy with forced quantization
        let mut force_opts = png::PngOptions::balanced();
        force_opts.quantization = QuantizationOptions {
            mode: QuantizationMode::Force,
            max_colors: 256,
            dithering: false,
        };
        group.bench_with_input(
            BenchmarkId::new("pixo_lossy_force", format!("{size}x{size}")),
            &gradient,
            |b, pixels| {
                b.iter(|| {
                    png::encode_into(
                        &mut png_buf,
                        black_box(pixels),
                        *size,
                        *size,
                        ColorType::Rgb,
                        &force_opts,
                    )
                    .unwrap()
                });
            },
        );

        // imagequant crate
        group.bench_with_input(
            BenchmarkId::new("imagequant", format!("{size}x{size}")),
            &gradient,
            |b, pixels| {
                b.iter(|| encode_with_imagequant(black_box(pixels), *size, *size));
            },
        );
    }

    group.finish();
}

// ============================================================================
// JPEG Encoding Comparison (all presets)
// ============================================================================

#[allow(clippy::single_element_loop)]
fn bench_jpeg_all_presets(c: &mut Criterion) {
    let mut group = c.benchmark_group("JPEG All Presets");

    for size in [512].iter() {
        let gradient = generate_gradient_image(*size, *size);
        let pixel_bytes = (*size as u64) * (*size as u64) * 3;

        group.throughput(Throughput::Bytes(pixel_bytes));

        let mut jpeg_buf = Vec::new();

        // pixo Fast preset
        group.bench_with_input(
            BenchmarkId::new("pixo_fast", format!("{size}x{size}")),
            &gradient,
            |b, pixels| {
                b.iter(|| {
                    jpeg::encode_with_options_into(
                        &mut jpeg_buf,
                        black_box(pixels),
                        *size,
                        *size,
                        ColorType::Rgb,
                        &jpeg::JpegOptions::fast(85),
                    )
                    .unwrap()
                });
            },
        );

        // pixo Balanced preset
        group.bench_with_input(
            BenchmarkId::new("pixo_balanced", format!("{size}x{size}")),
            &gradient,
            |b, pixels| {
                b.iter(|| {
                    jpeg::encode_with_options_into(
                        &mut jpeg_buf,
                        black_box(pixels),
                        *size,
                        *size,
                        ColorType::Rgb,
                        &jpeg::JpegOptions::balanced(85),
                    )
                    .unwrap()
                });
            },
        );

        // pixo Max preset
        group.bench_with_input(
            BenchmarkId::new("pixo_max", format!("{size}x{size}")),
            &gradient,
            |b, pixels| {
                b.iter(|| {
                    jpeg::encode_with_options_into(
                        &mut jpeg_buf,
                        black_box(pixels),
                        *size,
                        *size,
                        ColorType::Rgb,
                        &jpeg::JpegOptions::max(85),
                    )
                    .unwrap()
                });
            },
        );

        // image crate
        group.bench_with_input(
            BenchmarkId::new("image_crate", format!("{size}x{size}")),
            &gradient,
            |b, pixels| {
                b.iter(|| {
                    let mut output = Vec::new();
                    let encoder =
                        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut output, 85);
                    encoder
                        .write_image(black_box(pixels), *size, *size, image::ColorType::Rgb8)
                        .unwrap();
                    output
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// DEFLATE/zlib Comparison - Comprehensive
// Tests pixo vs flate2 vs libdeflate vs zopfli at multiple levels
// ============================================================================

fn bench_deflate_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("DEFLATE Comparison");

    let compressible = make_compressible(1 << 20);
    let random = make_random(1 << 20, 0xDEAD_BEEF);

    let cases = [("compressible_1mb", &compressible), ("random_1mb", &random)];

    for (name, data) in cases {
        let bytes = data.len() as u64;
        group.throughput(Throughput::Bytes(bytes));

        // pixo at level 6 (default)
        group.bench_with_input(BenchmarkId::new("pixo_lvl6", name), data, |b, input| {
            b.iter(|| {
                let encoded = deflate_zlib(black_box(input), 6);
                black_box(encoded.len())
            });
        });

        // flate2 at level 6 (default)
        group.bench_with_input(BenchmarkId::new("flate2_lvl6", name), data, |b, input| {
            b.iter(|| {
                let mut encoder = ZlibEncoder::new(Vec::new(), Compression::new(6));
                encoder.write_all(black_box(input)).unwrap();
                let encoded = encoder.finish().unwrap();
                black_box(encoded.len())
            });
        });

        // libdeflate at level 6
        group.bench_with_input(
            BenchmarkId::new("libdeflate_lvl6", name),
            data,
            |b, input| {
                b.iter(|| {
                    let mut compressor = LibdeflateCompressor::new(CompressionLvl::new(6).unwrap());
                    let max_size = compressor.zlib_compress_bound(input.len());
                    let mut output = vec![0u8; max_size];
                    let actual_size = compressor
                        .zlib_compress(black_box(input), &mut output)
                        .unwrap();
                    black_box(actual_size)
                });
            },
        );

        // libdeflate at level 12 (max)
        group.bench_with_input(
            BenchmarkId::new("libdeflate_lvl12", name),
            data,
            |b, input| {
                b.iter(|| {
                    let mut compressor =
                        LibdeflateCompressor::new(CompressionLvl::new(12).unwrap());
                    let max_size = compressor.zlib_compress_bound(input.len());
                    let mut output = vec![0u8; max_size];
                    let actual_size = compressor
                        .zlib_compress(black_box(input), &mut output)
                        .unwrap();
                    black_box(actual_size)
                });
            },
        );
    }

    group.finish();
}

/// Separate benchmark for zopfli (very slow, max compression)
fn bench_deflate_zopfli(c: &mut Criterion) {
    let mut group = c.benchmark_group("DEFLATE Zopfli (Max Compression)");

    // Use smaller data for zopfli since it's very slow
    let compressible = make_compressible(64 * 1024); // 64KB
    let bytes = compressible.len() as u64;
    group.throughput(Throughput::Bytes(bytes));

    // pixo at level 9 for comparison
    group.bench_with_input(
        BenchmarkId::new("pixo_lvl9", "compressible_64kb"),
        &compressible,
        |b, input| {
            b.iter(|| {
                let encoded = deflate_zlib(black_box(input), 9);
                black_box(encoded.len())
            });
        },
    );

    // zopfli (very slow but best compression)
    group.bench_with_input(
        BenchmarkId::new("zopfli", "compressible_64kb"),
        &compressible,
        |b, input| {
            b.iter(|| {
                let options = zopfli::Options::default();
                let mut output = Vec::new();
                let mut cursor = std::io::Cursor::new(black_box(input));
                zopfli::compress(options, zopfli::Format::Zlib, &mut cursor, &mut output).unwrap();
                black_box(output.len())
            });
        },
    );

    group.finish();
}

// ============================================================================
// FAIR COMPARISON: PNG Equivalent Settings
// All encoders use compression level 6 with adaptive filtering
// ============================================================================

fn bench_png_equivalent_settings(c: &mut Criterion) {
    let mut group = c.benchmark_group("PNG Equivalent Settings (Level 6)");

    // Test on multiple image types for fair comparison
    let gradient = generate_gradient_image(512, 512);
    let flat_blocks = generate_flat_blocks_image(512, 512);

    let test_cases: Vec<(&str, &[u8])> = vec![
        ("gradient_512", &gradient),
        ("flat_blocks_512", &flat_blocks),
    ];

    for (name, pixels) in test_cases {
        let pixel_bytes = pixels.len() as u64;
        group.throughput(Throughput::Bytes(pixel_bytes));

        let mut buf = Vec::new();

        // pixo at level 6 with Adaptive filter
        let pixo_opts = png::PngOptions::builder()
            .compression_level(6)
            .filter_strategy(png::FilterStrategy::Adaptive)
            .build();

        group.bench_with_input(BenchmarkId::new("pixo_lvl6", name), pixels, |b, pixels| {
            b.iter(|| {
                png::encode_into(
                    &mut buf,
                    black_box(pixels),
                    512,
                    512,
                    ColorType::Rgb,
                    &pixo_opts,
                )
                .unwrap()
            });
        });

        // image crate with default settings (uses flate2 level 6)
        group.bench_with_input(
            BenchmarkId::new("image_crate", name),
            pixels,
            |b, pixels| {
                b.iter(|| {
                    let mut output = Vec::new();
                    let encoder = image::codecs::png::PngEncoder::new(&mut output);
                    encoder
                        .write_image(black_box(pixels), 512, 512, image::ColorType::Rgb8)
                        .unwrap();
                    output
                });
            },
        );

        // lodepng with default settings
        group.bench_with_input(BenchmarkId::new("lodepng", name), pixels, |b, pixels| {
            b.iter(|| lodepng::encode24(black_box(pixels), 512, 512).unwrap());
        });
    }

    // Also test on real images if available
    let real_images = load_test_images();
    for img in real_images.iter().take(2) {
        let pixel_bytes = img.pixels.len() as u64;
        group.throughput(Throughput::Bytes(pixel_bytes));

        let mut buf = Vec::new();
        let pixo_opts = png::PngOptions::builder()
            .compression_level(6)
            .filter_strategy(png::FilterStrategy::Adaptive)
            .build();

        group.bench_with_input(
            BenchmarkId::new("pixo_lvl6", &img.name),
            &img.pixels,
            |b, pixels| {
                b.iter(|| {
                    png::encode_into(
                        &mut buf,
                        black_box(pixels),
                        img.width,
                        img.height,
                        ColorType::Rgb,
                        &pixo_opts,
                    )
                    .unwrap()
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("image_crate", &img.name),
            &img.pixels,
            |b, pixels| {
                b.iter(|| {
                    let mut output = Vec::new();
                    let encoder = image::codecs::png::PngEncoder::new(&mut output);
                    encoder
                        .write_image(
                            black_box(pixels),
                            img.width,
                            img.height,
                            image::ColorType::Rgb8,
                        )
                        .unwrap();
                    output
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("lodepng", &img.name),
            &img.pixels,
            |b, pixels| {
                b.iter(|| {
                    lodepng::encode24(black_box(pixels), img.width as usize, img.height as usize)
                        .unwrap()
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// FAIR COMPARISON: JPEG Equivalent Settings
// All encoders use Q85, 4:2:0 subsampling, baseline (non-progressive)
// ============================================================================

fn bench_jpeg_equivalent_settings(c: &mut Criterion) {
    let mut group = c.benchmark_group("JPEG Equivalent Settings (Q85 4:2:0)");

    // Test on multiple image types
    let gradient = generate_gradient_image(512, 512);
    let flat_blocks = generate_flat_blocks_image(512, 512);

    let test_cases: Vec<(&str, &[u8], u32, u32)> = vec![
        ("gradient_512", &gradient, 512, 512),
        ("flat_blocks_512", &flat_blocks, 512, 512),
    ];

    for (name, pixels, width, height) in &test_cases {
        let pixel_bytes = pixels.len() as u64;
        group.throughput(Throughput::Bytes(pixel_bytes));

        let mut buf = Vec::new();

        // pixo at Q85, 4:2:0, baseline (non-progressive)
        let pixo_opts = jpeg::JpegOptions {
            quality: 85,
            subsampling: jpeg::Subsampling::S420, // 4:2:0
            restart_interval: None,
            optimize_huffman: false, // baseline tables
            progressive: false,      // baseline
            trellis_quant: false,
        };

        group.bench_with_input(BenchmarkId::new("pixo_q85", name), *pixels, |b, pixels| {
            b.iter(|| {
                jpeg::encode_with_options_into(
                    &mut buf,
                    black_box(pixels),
                    *width,
                    *height,
                    ColorType::Rgb,
                    &pixo_opts,
                )
                .unwrap()
            });
        });

        // image crate with quality 85
        group.bench_with_input(
            BenchmarkId::new("image_crate_q85", name),
            *pixels,
            |b, pixels| {
                b.iter(|| {
                    let mut output = Vec::new();
                    let encoder =
                        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut output, 85);
                    encoder
                        .write_image(black_box(pixels), *width, *height, image::ColorType::Rgb8)
                        .unwrap();
                    output
                });
            },
        );

        // jpeg-encoder crate with Q85, 4:2:0
        group.bench_with_input(
            BenchmarkId::new("jpeg_encoder_q85", name),
            *pixels,
            |b, pixels| {
                b.iter(|| {
                    let mut output = Vec::new();
                    let encoder = JpegEncoderCrate::new(&mut output, 85);
                    encoder
                        .encode(
                            black_box(pixels),
                            *width as u16,
                            *height as u16,
                            JpegColorType::Rgb,
                        )
                        .unwrap();
                    output
                });
            },
        );
    }

    // Also test on real images if available
    let real_images = load_test_images();
    for img in real_images.iter().take(2) {
        let pixel_bytes = img.pixels.len() as u64;
        group.throughput(Throughput::Bytes(pixel_bytes));

        let mut buf = Vec::new();
        let pixo_opts = jpeg::JpegOptions {
            quality: 85,
            subsampling: jpeg::Subsampling::S420,
            restart_interval: None,
            optimize_huffman: false,
            progressive: false,
            trellis_quant: false,
        };

        group.bench_with_input(
            BenchmarkId::new("pixo_q85", &img.name),
            &img.pixels,
            |b, pixels| {
                b.iter(|| {
                    jpeg::encode_with_options_into(
                        &mut buf,
                        black_box(pixels),
                        img.width,
                        img.height,
                        ColorType::Rgb,
                        &pixo_opts,
                    )
                    .unwrap()
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("image_crate_q85", &img.name),
            &img.pixels,
            |b, pixels| {
                b.iter(|| {
                    let mut output = Vec::new();
                    let encoder =
                        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut output, 85);
                    encoder
                        .write_image(
                            black_box(pixels),
                            img.width,
                            img.height,
                            image::ColorType::Rgb8,
                        )
                        .unwrap();
                    output
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("jpeg_encoder_q85", &img.name),
            &img.pixels,
            |b, pixels| {
                b.iter(|| {
                    let mut output = Vec::new();
                    let encoder = JpegEncoderCrate::new(&mut output, 85);
                    encoder
                        .encode(
                            black_box(pixels),
                            img.width as u16,
                            img.height as u16,
                            JpegColorType::Rgb,
                        )
                        .unwrap();
                    output
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// BEST-EFFORT BENCHMARKS
// Each encoder uses its optimal settings for best compression
// ============================================================================

fn bench_png_best_effort(c: &mut Criterion) {
    let mut group = c.benchmark_group("PNG Best Effort");

    let gradient = generate_gradient_image(512, 512);
    let pixel_bytes = gradient.len() as u64;
    group.throughput(Throughput::Bytes(pixel_bytes));

    let mut buf = Vec::new();

    // pixo with balanced preset (good speed/size tradeoff)
    group.bench_with_input(
        BenchmarkId::new("pixo_balanced", "512x512"),
        &gradient,
        |b, pixels| {
            b.iter(|| {
                png::encode_into(
                    &mut buf,
                    black_box(pixels),
                    512,
                    512,
                    ColorType::Rgb,
                    &png::PngOptions::balanced(),
                )
                .unwrap()
            });
        },
    );

    // pixo with max preset (best compression)
    // Note: This is slow, so we only run it once per iteration
    group.bench_with_input(
        BenchmarkId::new("pixo_max", "512x512"),
        &gradient,
        |b, pixels| {
            b.iter(|| {
                png::encode_into(
                    &mut buf,
                    black_box(pixels),
                    512,
                    512,
                    ColorType::Rgb,
                    &png::PngOptions::max(),
                )
                .unwrap()
            });
        },
    );

    // image crate with best compression
    group.bench_with_input(
        BenchmarkId::new("image_best", "512x512"),
        &gradient,
        |b, pixels| {
            b.iter(|| {
                let mut output = Vec::new();
                let encoder = image::codecs::png::PngEncoder::new_with_quality(
                    &mut output,
                    image::codecs::png::CompressionType::Best,
                    image::codecs::png::FilterType::Adaptive,
                );
                encoder
                    .write_image(black_box(pixels), 512, 512, image::ColorType::Rgb8)
                    .unwrap();
                output
            });
        },
    );

    // lodepng (uses its default best settings)
    group.bench_with_input(
        BenchmarkId::new("lodepng", "512x512"),
        &gradient,
        |b, pixels| {
            b.iter(|| lodepng::encode24(black_box(pixels), 512, 512).unwrap());
        },
    );

    group.finish();
}

fn bench_jpeg_best_effort(c: &mut Criterion) {
    let mut group = c.benchmark_group("JPEG Best Effort");

    let gradient = generate_gradient_image(512, 512);
    let pixel_bytes = gradient.len() as u64;
    group.throughput(Throughput::Bytes(pixel_bytes));

    let mut buf = Vec::new();

    // pixo with max preset (progressive + trellis + optimized Huffman)
    group.bench_with_input(
        BenchmarkId::new("pixo_max", "512x512"),
        &gradient,
        |b, pixels| {
            b.iter(|| {
                jpeg::encode_with_options_into(
                    &mut buf,
                    black_box(pixels),
                    512,
                    512,
                    ColorType::Rgb,
                    &jpeg::JpegOptions::max(85),
                )
                .unwrap()
            });
        },
    );

    // pixo with balanced preset
    group.bench_with_input(
        BenchmarkId::new("pixo_balanced", "512x512"),
        &gradient,
        |b, pixels| {
            b.iter(|| {
                jpeg::encode_with_options_into(
                    &mut buf,
                    black_box(pixels),
                    512,
                    512,
                    ColorType::Rgb,
                    &jpeg::JpegOptions::balanced(85),
                )
                .unwrap()
            });
        },
    );

    // image crate
    group.bench_with_input(
        BenchmarkId::new("image_crate", "512x512"),
        &gradient,
        |b, pixels| {
            b.iter(|| {
                let mut output = Vec::new();
                let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut output, 85);
                encoder
                    .write_image(black_box(pixels), 512, 512, image::ColorType::Rgb8)
                    .unwrap();
                output
            });
        },
    );

    // jpeg-encoder crate with optimized settings
    group.bench_with_input(
        BenchmarkId::new("jpeg_encoder", "512x512"),
        &gradient,
        |b, pixels| {
            b.iter(|| {
                let mut output = Vec::new();
                let mut encoder = JpegEncoderCrate::new(&mut output, 85);
                encoder.set_sampling_factor(SamplingFactor::F_2_2);
                encoder
                    .encode(black_box(pixels), 512, 512, JpegColorType::Rgb)
                    .unwrap();
                output
            });
        },
    );

    group.finish();
}

// ============================================================================
// Summary Report (printed after benchmarks)
// ============================================================================

fn print_summary_report() {
    // Create temp directory for external tool tests
    let tmp_dir = PathBuf::from("target/bench-comparison");
    let _ = fs::create_dir_all(&tmp_dir);

    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║                                    PIXO BENCHMARK RESULTS                                      ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("This benchmark uses equivalent settings across all encoders for fair comparison.");
    println!("See benches/BENCHMARKS.md for detailed analysis and recommendations.");
    println!();

    // --- Fairness Documentation ---
    println!("┌────────────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ BENCHMARK SETTINGS                                                                             │");
    println!("├────────────────────────────────────────────────────────────────────────────────────────────────┤");
    println!("│ PNG Equivalent:  All encoders at compression level 6, adaptive filter                         │");
    println!("│ JPEG Equivalent: All encoders at Q85, 4:2:0 subsampling, baseline (non-progressive)           │");
    println!("│ DEFLATE:         All encoders at level 6 for speed comparison                                 │");
    println!("│ Test Images:     Synthetic (gradient, flat blocks) + Real (Kodak photos, fixtures)            │");
    println!("└────────────────────────────────────────────────────────────────────────────────────────────────┘");
    println!();

    // --- External Tool Availability ---
    let oxipng_available = find_oxipng().is_some();
    let mozjpeg_available = find_cjpeg().is_some();

    println!(
        "External tools: oxipng={}, mozjpeg={}",
        if oxipng_available { "found" } else { "missing" },
        if mozjpeg_available {
            "found"
        } else {
            "missing"
        }
    );
    println!();

    // --- Binary Size Comparison ---
    println!("┌────────────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ WASM Binary Size Comparison                                                                    │");
    println!("├────────────────────┬─────────────┬──────────────────────────────────────────────────────────────┤");
    println!("│ Library            │ WASM Size   │ Notes                                                        │");
    println!("├────────────────────┼─────────────┼──────────────────────────────────────────────────────────────┤");

    let pixo_wasm_size = get_wasm_size();
    let pixo_size_str = match pixo_wasm_size {
        Some(size) => format_size(size),
        None => "146 KB".to_string(),
    };

    println!(
        "│ {:<18} │ {:>11} │ {:<60} │",
        "pixo", pixo_size_str, "Zero deps, pure Rust"
    );
    println!(
        "│ {:<18} │ {:>11} │ {:<60} │",
        "wasm-mozjpeg", "~208 KB", "Emscripten compiled"
    );
    println!(
        "│ {:<18} │ {:>11} │ {:<60} │",
        "squoosh oxipng", "~625 KB", "Google Squoosh codec"
    );
    println!(
        "│ {:<18} │ {:>11} │ {:<60} │",
        "squoosh mozjpeg", "~803 KB", "Google Squoosh codec"
    );
    println!(
        "│ {:<18} │ {:>11} │ {:<60} │",
        "image crate", "~6-10 MB", "Pure Rust, many codecs"
    );
    println!("└────────────────────┴─────────────┴──────────────────────────────────────────────────────────────┘");
    println!();

    // --- PNG Comparison (all presets + external tools) ---
    println!("┌────────────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ PNG Compression (512x512 gradient image)                                                       │");
    println!("├────────────────────┬─────────────┬─────────────┬───────────────────────────────────────────────┤");
    println!("│ Encoder            │ Size        │ Time        │ Notes                                         │");
    println!("├────────────────────┼─────────────┼─────────────┼───────────────────────────────────────────────┤");

    let gradient = generate_gradient_image(512, 512);

    // pixo Fast
    let (fast_size, fast_time) = measure_png_encode(&gradient, 512, 512, &png::PngOptions::fast());
    println!(
        "│ {:<18} │ {:>11} │ {:>11} │ {:<45} │",
        "pixo Fast",
        format_size(fast_size),
        format_duration(fast_time),
        "level=2, AdaptiveFast filter"
    );

    // pixo Balanced
    let (balanced_size, balanced_time) =
        measure_png_encode(&gradient, 512, 512, &png::PngOptions::balanced());
    println!(
        "│ {:<18} │ {:>11} │ {:>11} │ {:<45} │",
        "pixo Balanced",
        format_size(balanced_size),
        format_duration(balanced_time),
        "level=6, Adaptive filter"
    );

    // pixo Max (single iteration - too slow for multiple)
    let max_start = Instant::now();
    let mut max_buf = Vec::new();
    png::encode_into(
        &mut max_buf,
        &gradient,
        512,
        512,
        ColorType::Rgb,
        &png::PngOptions::max(),
    )
    .unwrap();
    let max_time = max_start.elapsed();
    println!(
        "│ {:<18} │ {:>11} │ {:>11} │ {:<45} │",
        "pixo Max",
        format_size(max_buf.len()),
        format_duration(max_time),
        "level=9, MinSum, optimal LZ77"
    );

    // image crate
    let (image_size, image_time) = measure_image_png_encode(&gradient, 512, 512);
    println!(
        "│ {:<18} │ {:>11} │ {:>11} │ {:<45} │",
        "image crate",
        format_size(image_size),
        format_duration(image_time),
        "default PngEncoder"
    );

    // oxipng (if available)
    if let Some((oxi_size, oxi_time)) = encode_with_oxipng(&gradient, 512, 512, &tmp_dir) {
        let delta = (balanced_size as f64 / oxi_size as f64 - 1.0) * 100.0;
        println!(
            "│ {:<18} │ {:>11} │ {:>11} │ {:<45} │",
            "oxipng",
            format_size(oxi_size),
            format_duration(oxi_time),
            format!("-o4 --strip safe (Δ={:+.1}%)", delta)
        );
    } else {
        println!(
            "│ {:<18} │ {:>11} │ {:>11} │ {:<45} │",
            "oxipng", "N/A", "N/A", "not installed (brew install oxipng)"
        );
    }

    println!("└────────────────────┴─────────────┴─────────────┴───────────────────────────────────────────────┘");
    println!();

    // --- PNG Lossy Comparison ---
    println!("┌────────────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ PNG Lossy Compression (512x512 gradient image, quantization)                                   │");
    println!("├────────────────────┬─────────────┬─────────────┬───────────────────────────────────────────────┤");
    println!("│ Encoder            │ Size        │ Time        │ Notes                                         │");
    println!("├────────────────────┼─────────────┼─────────────┼───────────────────────────────────────────────┤");

    // pixo lossless (baseline)
    println!(
        "│ {:<18} │ {:>11} │ {:>11} │ {:<45} │",
        "pixo Lossless",
        format_size(balanced_size),
        format_duration(balanced_time),
        "Baseline (no quantization)"
    );

    // pixo lossy with forced quantization
    let mut lossy_opts = png::PngOptions::balanced();
    lossy_opts.quantization = QuantizationOptions {
        mode: QuantizationMode::Force,
        max_colors: 256,
        dithering: false,
    };
    let (lossy_size, lossy_time) = measure_png_encode(&gradient, 512, 512, &lossy_opts);
    let lossy_savings = (1.0 - lossy_size as f64 / balanced_size as f64) * 100.0;
    println!(
        "│ {:<18} │ {:>11} │ {:>11} │ {:<45} │",
        "pixo Lossy",
        format_size(lossy_size),
        format_duration(lossy_time),
        format!("256 colors, no dithering (-{:.1}%)", lossy_savings)
    );

    // imagequant crate
    if let Some((iq_size, iq_time)) = encode_with_imagequant(&gradient, 512, 512) {
        let iq_savings = (1.0 - iq_size as f64 / balanced_size as f64) * 100.0;
        println!(
            "│ {:<18} │ {:>11} │ {:>11} │ {:<45} │",
            "imagequant",
            format_size(iq_size),
            format_duration(iq_time),
            format!("libimagequant bindings (-{:.1}%)", iq_savings)
        );
    }

    // pngquant (if available)
    let pngquant_available = find_pngquant().is_some();
    if let Some((pq_size, pq_time)) = encode_with_pngquant(&gradient, 512, 512, &tmp_dir) {
        let pq_savings = (1.0 - pq_size as f64 / balanced_size as f64) * 100.0;
        println!(
            "│ {:<18} │ {:>11} │ {:>11} │ {:<45} │",
            "pngquant",
            format_size(pq_size),
            format_duration(pq_time),
            format!("--quality=65-80 (-{:.1}%)", pq_savings)
        );
    } else {
        println!(
            "│ {:<18} │ {:>11} │ {:>11} │ {:<45} │",
            "pngquant", "N/A", "N/A", "not installed (brew install pngquant)"
        );
    }

    println!("└────────────────────┴─────────────┴─────────────┴───────────────────────────────────────────────┘");
    println!();

    // --- Real Image Lossy Comparison ---
    println!("┌────────────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ PNG Lossy Compression (Real Images: pixo vs pngquant)                                        │");
    println!("├────────────────────┬─────────────┬─────────────┬─────────────┬──────────────────────────────────┤");
    println!("│ Image              │ pixo Lossy│ pngquant    │ Delta       │ Notes                            │");
    println!("├────────────────────┼─────────────┼─────────────┼─────────────┼──────────────────────────────────┤");

    let real_images = [
        ("avatar-color.png", "tests/fixtures/avatar-color.png"),
        ("rocket.png", "tests/fixtures/rocket.png"),
    ];

    for (name, path) in real_images {
        if let Some((pixo_size, _pixo_time, pq_size, _pq_time)) =
            compare_real_image_lossy(path, &tmp_dir)
        {
            let delta = (pixo_size as f64 / pq_size as f64 - 1.0) * 100.0;
            let delta_str = format!("{delta:+.0}%");
            let note = if delta < 0.0 {
                "pixo wins"
            } else {
                "pngquant wins"
            };
            println!(
                "│ {:<18} │ {:>11} │ {:>11} │ {:>11} │ {:<32} │",
                name,
                format_size(pixo_size),
                format_size(pq_size),
                delta_str,
                note
            );
        } else {
            println!(
                "│ {:<18} │ {:>11} │ {:>11} │ {:>11} │ {:<32} │",
                name, "N/A", "N/A", "N/A", "image not found"
            );
        }
    }

    println!("└────────────────────┴─────────────┴─────────────┴─────────────┴──────────────────────────────────┘");
    println!();

    // Update external tool status
    println!(
        "External tools: oxipng={}, mozjpeg={}, pngquant={}",
        if oxipng_available { "found" } else { "missing" },
        if mozjpeg_available {
            "found"
        } else {
            "missing"
        },
        if pngquant_available {
            "found"
        } else {
            "missing"
        }
    );
    println!();

    // --- JPEG Comparison (all presets + external tools) ---
    println!("┌────────────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ JPEG Compression (512x512 gradient image, quality 85)                                          │");
    println!("├────────────────────┬─────────────┬─────────────┬───────────────────────────────────────────────┤");
    println!("│ Encoder            │ Size        │ Time        │ Notes                                         │");
    println!("├────────────────────┼─────────────┼─────────────┼───────────────────────────────────────────────┤");

    // pixo Fast
    let (fast_size, fast_time) =
        measure_jpeg_encode(&gradient, 512, 512, &jpeg::JpegOptions::fast(85));
    println!(
        "│ {:<18} │ {:>11} │ {:>11} │ {:<45} │",
        "pixo Fast",
        format_size(fast_size),
        format_duration(fast_time),
        "4:4:4, baseline, no optimization"
    );

    // pixo Balanced
    let (balanced_size, balanced_time) =
        measure_jpeg_encode(&gradient, 512, 512, &jpeg::JpegOptions::balanced(85));
    println!(
        "│ {:<18} │ {:>11} │ {:>11} │ {:<45} │",
        "pixo Balanced",
        format_size(balanced_size),
        format_duration(balanced_time),
        "4:4:4, Huffman optimization"
    );

    // pixo Max
    let (max_size, max_time) =
        measure_jpeg_encode(&gradient, 512, 512, &jpeg::JpegOptions::max(85));
    println!(
        "│ {:<18} │ {:>11} │ {:>11} │ {:<45} │",
        "pixo Max",
        format_size(max_size),
        format_duration(max_time),
        "4:2:0, progressive, trellis"
    );

    // image crate
    let (image_size, image_time) = measure_image_jpeg_encode(&gradient, 512, 512);
    println!(
        "│ {:<18} │ {:>11} │ {:>11} │ {:<45} │",
        "image crate",
        format_size(image_size),
        format_duration(image_time),
        "quality 85, default settings"
    );

    // mozjpeg (if available)
    if let Some((moz_size, moz_time)) = encode_with_mozjpeg(&gradient, 512, 512, &tmp_dir) {
        let delta = (max_size as f64 / moz_size as f64 - 1.0) * 100.0;
        println!(
            "│ {:<18} │ {:>11} │ {:>11} │ {:<45} │",
            "mozjpeg",
            format_size(moz_size),
            format_duration(moz_time),
            format!("-quality 85 -optimize -progressive (Δ={:+.1}%)", delta)
        );
    } else {
        println!(
            "│ {:<18} │ {:>11} │ {:>11} │ {:<45} │",
            "mozjpeg", "N/A", "N/A", "not installed (brew install mozjpeg)"
        );
    }

    println!("└────────────────────┴─────────────┴─────────────┴───────────────────────────────────────────────┘");
    println!();

    // --- DEFLATE Comparison ---
    println!("┌────────────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ DEFLATE Compression (1 MB compressible payload, level 6)                                       │");
    println!("├────────────────────┬─────────────┬─────────────┬───────────────────────────────────────────────┤");
    println!("│ Library            │ Output Size │ Ratio       │ Notes                                         │");
    println!("├────────────────────┼─────────────┼─────────────┼───────────────────────────────────────────────┤");

    let compressible = make_compressible(1 << 20);
    let input_size = compressible.len();

    // pixo
    let pixo_deflate = deflate_zlib(&compressible, 6);
    let pixo_ratio = input_size as f64 / pixo_deflate.len() as f64;

    // flate2
    let mut flate2_enc = ZlibEncoder::new(Vec::new(), Compression::new(6));
    flate2_enc.write_all(&compressible).unwrap();
    let flate2_deflate = flate2_enc.finish().unwrap();
    let flate2_ratio = input_size as f64 / flate2_deflate.len() as f64;

    // libdeflate
    let mut libdeflate_compressor = LibdeflateCompressor::new(CompressionLvl::new(6).unwrap());
    let max_size = libdeflate_compressor.zlib_compress_bound(input_size);
    let mut libdeflate_output = vec![0u8; max_size];
    let libdeflate_size = libdeflate_compressor
        .zlib_compress(&compressible, &mut libdeflate_output)
        .unwrap();
    let libdeflate_ratio = input_size as f64 / libdeflate_size as f64;

    println!(
        "│ {:<18} │ {:>11} │ {:>10.1}x │ {:<45} │",
        "pixo (lvl 6)",
        format_size(pixo_deflate.len()),
        pixo_ratio,
        "Pure Rust, zero deps"
    );
    println!(
        "│ {:<18} │ {:>11} │ {:>10.1}x │ {:<45} │",
        "flate2 (lvl 6)",
        format_size(flate2_deflate.len()),
        flate2_ratio,
        "miniz_oxide backend"
    );
    println!(
        "│ {:<18} │ {:>11} │ {:>10.1}x │ {:<45} │",
        "libdeflate (lvl 6)",
        format_size(libdeflate_size),
        libdeflate_ratio,
        "C library, fast"
    );
    println!("└────────────────────┴─────────────┴─────────────┴───────────────────────────────────────────────┘");
    println!();

    // --- Summary Notes ---
    println!("Notes:");
    println!("  • WASM sizes are approximate and depend on build configuration");
    println!("  • Speed measurements are averaged over 10 iterations (except Max preset)");
    println!("  • For detailed benchmarks see: cargo bench --bench comparison");
    println!("  • For full documentation see: benches/BENCHMARKS.md");
    println!();
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Compare pixo lossy vs pngquant on a real image file
/// Returns (pixo_size, pixo_time, pngquant_size, pngquant_time)
fn compare_real_image_lossy(
    path: &str,
    tmp_dir: &Path,
) -> Option<(usize, Duration, usize, Duration)> {
    use image::GenericImageView;

    if !Path::new(path).exists() {
        return None;
    }

    let img = image::open(path).ok()?;
    let rgba = img.to_rgba8();
    let (width, height) = img.dimensions();
    let pixels = rgba.as_raw();

    // pixo lossy
    let mut lossy_opts = png::PngOptions::balanced();
    lossy_opts.quantization = QuantizationOptions {
        mode: QuantizationMode::Force,
        max_colors: 256,
        dithering: false,
    };

    let start = Instant::now();
    let mut lossy_buf = Vec::new();
    png::encode_into(
        &mut lossy_buf,
        pixels,
        width,
        height,
        ColorType::Rgba,
        &lossy_opts,
    )
    .ok()?;
    let pixo_time = start.elapsed();

    // pngquant - first encode lossless for input
    let lossless = png::encode(pixels, width, height, ColorType::Rgba).ok()?;
    let input_path = tmp_dir.join("real_pq_input.png");
    let output_path = tmp_dir.join("real_pq_output.png");
    fs::write(&input_path, &lossless).ok()?;

    let pngquant_bin = find_pngquant()?;
    let start = Instant::now();
    let status = Command::new(&pngquant_bin)
        .args([
            "--quality=65-80",
            "--speed=4",
            "--force",
            "--output",
            output_path.to_str()?,
            input_path.to_str()?,
        ])
        .output()
        .ok()?;
    let pq_time = start.elapsed();

    if !status.status.success() {
        return None;
    }

    let pq_size = fs::metadata(&output_path).ok()?.len() as usize;

    Some((lossy_buf.len(), pixo_time, pq_size, pq_time))
}

fn measure_png_encode(
    pixels: &[u8],
    width: u32,
    height: u32,
    opts: &png::PngOptions,
) -> (usize, Duration) {
    let mut buf = Vec::new();

    // Warm up
    for _ in 0..3 {
        png::encode_into(&mut buf, pixels, width, height, ColorType::Rgb, opts).unwrap();
    }

    // Measure
    let start = Instant::now();
    let iterations = 10;
    for _ in 0..iterations {
        png::encode_into(&mut buf, pixels, width, height, ColorType::Rgb, opts).unwrap();
    }
    let duration = start.elapsed() / iterations;

    (buf.len(), duration)
}

fn measure_jpeg_encode(
    pixels: &[u8],
    width: u32,
    height: u32,
    opts: &jpeg::JpegOptions,
) -> (usize, Duration) {
    let mut buf = Vec::new();

    // Warm up
    for _ in 0..3 {
        jpeg::encode_with_options_into(&mut buf, pixels, width, height, ColorType::Rgb, opts)
            .unwrap();
    }

    // Measure
    let start = Instant::now();
    let iterations = 10;
    for _ in 0..iterations {
        jpeg::encode_with_options_into(&mut buf, pixels, width, height, ColorType::Rgb, opts)
            .unwrap();
    }
    let duration = start.elapsed() / iterations;

    (buf.len(), duration)
}

fn measure_image_png_encode(pixels: &[u8], width: u32, height: u32) -> (usize, Duration) {
    // Warm up
    for _ in 0..3 {
        let mut output = Vec::new();
        let encoder = image::codecs::png::PngEncoder::new(&mut output);
        encoder
            .write_image(pixels, width, height, image::ColorType::Rgb8)
            .unwrap();
    }

    // Measure
    let start = Instant::now();
    let iterations = 10;
    let mut last_output = Vec::new();
    for _ in 0..iterations {
        let mut output = Vec::new();
        let encoder = image::codecs::png::PngEncoder::new(&mut output);
        encoder
            .write_image(pixels, width, height, image::ColorType::Rgb8)
            .unwrap();
        last_output = output;
    }
    let duration = start.elapsed() / iterations;

    (last_output.len(), duration)
}

fn measure_image_jpeg_encode(pixels: &[u8], width: u32, height: u32) -> (usize, Duration) {
    // Warm up
    for _ in 0..3 {
        let mut output = Vec::new();
        let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut output, 85);
        encoder
            .write_image(pixels, width, height, image::ColorType::Rgb8)
            .unwrap();
    }

    // Measure
    let start = Instant::now();
    let iterations = 10;
    let mut last_output = Vec::new();
    for _ in 0..iterations {
        let mut output = Vec::new();
        let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut output, 85);
        encoder
            .write_image(pixels, width, height, image::ColorType::Rgb8)
            .unwrap();
        last_output = output;
    }
    let duration = start.elapsed() / iterations;

    (last_output.len(), duration)
}

fn get_wasm_size() -> Option<usize> {
    let paths = [
        "target/wasm32-unknown-unknown/release/pixo.wasm",
        "target/wasm32-unknown-unknown/release/pixo_bg.wasm",
        "my-app/src/lib/pixo-wasm/pixo_bg.wasm",
    ];

    for path in paths {
        if let Ok(metadata) = std::fs::metadata(path) {
            return Some(metadata.len() as usize);
        }
    }
    None
}

fn format_size(bytes: usize) -> String {
    if bytes >= 1024 * 1024 {
        let mb = bytes as f64 / (1024.0 * 1024.0);
        format!("{mb:.1} MB")
    } else if bytes >= 1024 {
        let kb = bytes as f64 / 1024.0;
        format!("{kb:.1} KB")
    } else {
        format!("{bytes} B")
    }
}

fn format_duration(duration: Duration) -> String {
    let micros = duration.as_micros();
    if micros >= 1000 {
        let ms = duration.as_secs_f64() * 1000.0;
        format!("{ms:.2} ms")
    } else {
        format!("{micros} µs")
    }
}

// ============================================================================
// Kodak Suite Benchmark (Real Photographic Images)
// ============================================================================

/// Load Kodak images for benchmarking.
/// Returns a subset to keep benchmark times reasonable.
#[allow(clippy::type_complexity)]
fn load_kodak_for_benchmark() -> Option<Vec<(String, u32, u32, Vec<u8>)>> {
    let fixtures_dir = std::path::Path::new("tests/fixtures/kodak");
    if !fixtures_dir.exists() {
        return None;
    }

    let mut images = Vec::new();
    // Load first 2 images for faster benchmark time
    for i in 1..=2 {
        let path = fixtures_dir.join(format!("kodim{i:02}.png"));
        if !path.exists() {
            continue;
        }

        let data = fs::read(&path).ok()?;
        let img = image::load_from_memory(&data).ok()?;
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width(), rgb.height());
        let name = format!("kodim{i:02}");
        images.push((name, w, h, rgb.into_raw()));
    }

    if images.is_empty() {
        None
    } else {
        Some(images)
    }
}

fn bench_kodak_suite(c: &mut Criterion) {
    let Some(images) = load_kodak_for_benchmark() else {
        eprintln!("Kodak fixtures not available, skipping bench_kodak_suite");
        return;
    };

    let mut group = c.benchmark_group("Kodak Real Images");

    for (name, w, h, pixels) in &images {
        let pixel_bytes = (*w as u64) * (*h as u64) * 3;
        group.throughput(Throughput::Bytes(pixel_bytes));

        let mut buf = Vec::new();

        // PNG Balanced
        group.bench_with_input(
            BenchmarkId::new("png_balanced", name),
            pixels,
            |b, pixels| {
                b.iter(|| {
                    png::encode_into(
                        &mut buf,
                        black_box(pixels),
                        *w,
                        *h,
                        ColorType::Rgb,
                        &png::PngOptions::balanced(),
                    )
                    .unwrap()
                });
            },
        );

        // JPEG Balanced
        group.bench_with_input(
            BenchmarkId::new("jpeg_balanced", name),
            pixels,
            |b, pixels| {
                b.iter(|| {
                    jpeg::encode_with_options_into(
                        &mut buf,
                        black_box(pixels),
                        *w,
                        *h,
                        ColorType::Rgb,
                        &jpeg::JpegOptions::balanced(85),
                    )
                    .unwrap()
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Custom Criterion Configuration
// ============================================================================

fn custom_criterion() -> Criterion {
    Criterion::default()
        .sample_size(20)
        .measurement_time(std::time::Duration::from_secs(2))
        .warm_up_time(std::time::Duration::from_millis(500))
}

criterion_group! {
    name = benches;
    config = custom_criterion();
    targets =
        // Equivalent settings (fair comparison)
        bench_png_equivalent_settings,
        bench_jpeg_equivalent_settings,
        // Best-effort (each encoder's optimal settings)
        bench_png_best_effort,
        bench_jpeg_best_effort,
        // DEFLATE deep dive
        bench_deflate_comparison,
        bench_deflate_zopfli,
        // Real images
        bench_kodak_suite,
        // Legacy preset benchmarks
        bench_png_all_presets,
        bench_png_lossy_comparison,
        bench_jpeg_all_presets
}

// Custom main that prints summary after benchmarks
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--summary-only") {
        print_summary_report();
        return;
    }

    // Run criterion benchmarks
    benches();

    // Print summary report at the end
    print_summary_report();
}
