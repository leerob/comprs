//! Comprehensive library comparison benchmark.
//!
//! Compares comprs against popular image compression libraries and external tools,
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

use comprs::compress::deflate::deflate_zlib;
use comprs::png::{QuantizationMode, QuantizationOptions};
use comprs::{jpeg, png, ColorType};

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

// ============================================================================
// External Tool Detection (merged from codec_harness.rs)
// ============================================================================

fn find_oxipng() -> Option<PathBuf> {
    let paths = [
        "vendor/oxipng/target/release/oxipng",
        "/usr/local/bin/oxipng",
        "/opt/homebrew/bin/oxipng",
    ];
    paths
        .iter()
        .find(|p| Path::new(p).exists())
        .map(PathBuf::from)
}

fn find_cjpeg() -> Option<PathBuf> {
    let paths = [
        "vendor/mozjpeg/build/cjpeg",
        "vendor/mozjpeg/cjpeg",
        "/usr/local/bin/cjpeg",
        "/opt/homebrew/bin/cjpeg",
    ];
    paths
        .iter()
        .find(|p| Path::new(p).exists())
        .map(PathBuf::from)
}

fn find_pngquant() -> Option<PathBuf> {
    let paths = [
        "/usr/local/bin/pngquant",
        "/opt/homebrew/bin/pngquant",
        "/usr/bin/pngquant",
    ];
    paths
        .iter()
        .find(|p| Path::new(p).exists())
        .map(PathBuf::from)
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

    // Use comprs to encode the indexed PNG
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

fn bench_png_all_presets(c: &mut Criterion) {
    let mut group = c.benchmark_group("PNG All Presets");

    for size in [256, 512].iter() {
        let gradient = generate_gradient_image(*size, *size);
        let pixel_bytes = (*size as u64) * (*size as u64) * 3;

        group.throughput(Throughput::Bytes(pixel_bytes));

        let mut png_buf = Vec::new();

        // comprs Fast preset
        group.bench_with_input(
            BenchmarkId::new("comprs_fast", format!("{size}x{size}")),
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

        // comprs Balanced preset
        group.bench_with_input(
            BenchmarkId::new("comprs_balanced", format!("{size}x{size}")),
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

        // comprs Max preset (skip in normal benchmarks - too slow)
        // group.bench_with_input(
        //     BenchmarkId::new("comprs_max", format!("{size}x{size}")),
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

fn bench_png_lossy_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("PNG Lossy Comparison");

    for size in [256, 512].iter() {
        let gradient = generate_gradient_image(*size, *size);
        let pixel_bytes = (*size as u64) * (*size as u64) * 3;

        group.throughput(Throughput::Bytes(pixel_bytes));

        let mut png_buf = Vec::new();

        // comprs lossless (baseline)
        group.bench_with_input(
            BenchmarkId::new("comprs_lossless", format!("{size}x{size}")),
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

        // comprs lossy with auto quantization
        let mut lossy_opts = png::PngOptions::balanced();
        lossy_opts.quantization = QuantizationOptions {
            mode: QuantizationMode::Auto,
            max_colors: 256,
            dithering: false,
        };
        group.bench_with_input(
            BenchmarkId::new("comprs_lossy_auto", format!("{size}x{size}")),
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

        // comprs lossy with forced quantization
        let mut force_opts = png::PngOptions::balanced();
        force_opts.quantization = QuantizationOptions {
            mode: QuantizationMode::Force,
            max_colors: 256,
            dithering: false,
        };
        group.bench_with_input(
            BenchmarkId::new("comprs_lossy_force", format!("{size}x{size}")),
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

fn bench_jpeg_all_presets(c: &mut Criterion) {
    let mut group = c.benchmark_group("JPEG All Presets");

    for size in [256, 512].iter() {
        let gradient = generate_gradient_image(*size, *size);
        let pixel_bytes = (*size as u64) * (*size as u64) * 3;

        group.throughput(Throughput::Bytes(pixel_bytes));

        let mut jpeg_buf = Vec::new();

        // comprs Fast preset
        group.bench_with_input(
            BenchmarkId::new("comprs_fast", format!("{size}x{size}")),
            &gradient,
            |b, pixels| {
                b.iter(|| {
                    jpeg::encode_with_options_into(
                        &mut jpeg_buf,
                        black_box(pixels),
                        *size,
                        *size,
                        85,
                        ColorType::Rgb,
                        &jpeg::JpegOptions::fast(85),
                    )
                    .unwrap()
                });
            },
        );

        // comprs Balanced preset
        group.bench_with_input(
            BenchmarkId::new("comprs_balanced", format!("{size}x{size}")),
            &gradient,
            |b, pixels| {
                b.iter(|| {
                    jpeg::encode_with_options_into(
                        &mut jpeg_buf,
                        black_box(pixels),
                        *size,
                        *size,
                        85,
                        ColorType::Rgb,
                        &jpeg::JpegOptions::balanced(85),
                    )
                    .unwrap()
                });
            },
        );

        // comprs Max preset
        group.bench_with_input(
            BenchmarkId::new("comprs_max", format!("{size}x{size}")),
            &gradient,
            |b, pixels| {
                b.iter(|| {
                    jpeg::encode_with_options_into(
                        &mut jpeg_buf,
                        black_box(pixels),
                        *size,
                        *size,
                        85,
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
// DEFLATE/zlib Comparison
// ============================================================================

fn bench_deflate_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("DEFLATE Comparison");

    let compressible = make_compressible(1 << 20);
    let random = make_random(1 << 20, 0xDEAD_BEEF);

    let cases = [("compressible_1mb", &compressible), ("random_1mb", &random)];

    for (name, data) in cases {
        let bytes = data.len() as u64;
        group.throughput(Throughput::Bytes(bytes));

        group.bench_with_input(BenchmarkId::new("comprs", name), data, |b, input| {
            b.iter(|| {
                let encoded = deflate_zlib(black_box(input), 6);
                black_box(encoded.len())
            });
        });

        group.bench_with_input(BenchmarkId::new("flate2", name), data, |b, input| {
            b.iter(|| {
                let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
                encoder.write_all(black_box(input)).unwrap();
                let encoded = encoder.finish().unwrap();
                black_box(encoded.len())
            });
        });
    }

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
    println!("║                              COMPRS COMPREHENSIVE BENCHMARK SUMMARY                              ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════════════════════════╝");
    println!();
    println!("See benches/BENCHMARKS.md for detailed analysis and recommendations.");
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

    let comprs_wasm_size = get_wasm_size();
    let comprs_size_str = match comprs_wasm_size {
        Some(size) => format_size(size),
        None => "~92 KB".to_string(),
    };

    println!(
        "│ {:<18} │ {:>11} │ {:<60} │",
        "comprs", comprs_size_str, "Zero deps, pure Rust"
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

    // comprs Fast
    let (fast_size, fast_time) = measure_png_encode(&gradient, 512, 512, &png::PngOptions::fast());
    println!(
        "│ {:<18} │ {:>11} │ {:>11} │ {:<45} │",
        "comprs Fast",
        format_size(fast_size),
        format_duration(fast_time),
        "level=2, AdaptiveFast filter"
    );

    // comprs Balanced
    let (balanced_size, balanced_time) =
        measure_png_encode(&gradient, 512, 512, &png::PngOptions::balanced());
    println!(
        "│ {:<18} │ {:>11} │ {:>11} │ {:<45} │",
        "comprs Balanced",
        format_size(balanced_size),
        format_duration(balanced_time),
        "level=6, Adaptive filter"
    );

    // comprs Max (single iteration - too slow for multiple)
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
        "comprs Max",
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

    // comprs lossless (baseline)
    println!(
        "│ {:<18} │ {:>11} │ {:>11} │ {:<45} │",
        "comprs Lossless",
        format_size(balanced_size),
        format_duration(balanced_time),
        "Baseline (no quantization)"
    );

    // comprs lossy with forced quantization
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
        "comprs Lossy",
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
    println!("│ PNG Lossy Compression (Real Images: comprs vs pngquant)                                        │");
    println!("├────────────────────┬─────────────┬─────────────┬─────────────┬──────────────────────────────────┤");
    println!("│ Image              │ comprs Lossy│ pngquant    │ Delta       │ Notes                            │");
    println!("├────────────────────┼─────────────┼─────────────┼─────────────┼──────────────────────────────────┤");

    let real_images = [
        ("avatar-color.png", "tests/fixtures/avatar-color.png"),
        ("rocket.png", "tests/fixtures/rocket.png"),
    ];

    for (name, path) in real_images {
        if let Some((comprs_size, _comprs_time, pq_size, _pq_time)) =
            compare_real_image_lossy(path, &tmp_dir)
        {
            let delta = (comprs_size as f64 / pq_size as f64 - 1.0) * 100.0;
            let delta_str = format!("{:+.0}%", delta);
            let note = if delta < 0.0 { "comprs wins" } else { "pngquant wins" };
            println!(
                "│ {:<18} │ {:>11} │ {:>11} │ {:>11} │ {:<32} │",
                name,
                format_size(comprs_size),
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
        if mozjpeg_available { "found" } else { "missing" },
        if pngquant_available { "found" } else { "missing" }
    );
    println!();

    // --- JPEG Comparison (all presets + external tools) ---
    println!("┌────────────────────────────────────────────────────────────────────────────────────────────────┐");
    println!("│ JPEG Compression (512x512 gradient image, quality 85)                                          │");
    println!("├────────────────────┬─────────────┬─────────────┬───────────────────────────────────────────────┤");
    println!("│ Encoder            │ Size        │ Time        │ Notes                                         │");
    println!("├────────────────────┼─────────────┼─────────────┼───────────────────────────────────────────────┤");

    // comprs Fast
    let (fast_size, fast_time) =
        measure_jpeg_encode(&gradient, 512, 512, &jpeg::JpegOptions::fast(85));
    println!(
        "│ {:<18} │ {:>11} │ {:>11} │ {:<45} │",
        "comprs Fast",
        format_size(fast_size),
        format_duration(fast_time),
        "4:4:4, baseline, no optimization"
    );

    // comprs Balanced
    let (balanced_size, balanced_time) =
        measure_jpeg_encode(&gradient, 512, 512, &jpeg::JpegOptions::balanced(85));
    println!(
        "│ {:<18} │ {:>11} │ {:>11} │ {:<45} │",
        "comprs Balanced",
        format_size(balanced_size),
        format_duration(balanced_time),
        "4:4:4, Huffman optimization"
    );

    // comprs Max
    let (max_size, max_time) =
        measure_jpeg_encode(&gradient, 512, 512, &jpeg::JpegOptions::max(85));
    println!(
        "│ {:<18} │ {:>11} │ {:>11} │ {:<45} │",
        "comprs Max",
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
    println!("│ DEFLATE Compression (1 MB payload)                                                             │");
    println!("├────────────────────┬─────────────┬─────────────┬───────────────────────────────────────────────┤");
    println!("│ Library            │ Output Size │ Ratio       │ Notes                                         │");
    println!("├────────────────────┼─────────────┼─────────────┼───────────────────────────────────────────────┤");

    let compressible = make_compressible(1 << 20);
    let comprs_deflate = deflate_zlib(&compressible, 6);

    let mut flate2_enc = ZlibEncoder::new(Vec::new(), Compression::default());
    flate2_enc.write_all(&compressible).unwrap();
    let flate2_deflate = flate2_enc.finish().unwrap();

    let comprs_ratio = compressible.len() as f64 / comprs_deflate.len() as f64;
    let flate2_ratio = compressible.len() as f64 / flate2_deflate.len() as f64;

    println!(
        "│ {:<18} │ {:>11} │ {:>10.1}x │ {:<45} │",
        "comprs",
        format_size(comprs_deflate.len()),
        comprs_ratio,
        "Pure Rust, zero deps"
    );
    println!(
        "│ {:<18} │ {:>11} │ {:>10.1}x │ {:<45} │",
        "flate2",
        format_size(flate2_deflate.len()),
        flate2_ratio,
        "miniz_oxide backend"
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

/// Compare comprs lossy vs pngquant on a real image file
/// Returns (comprs_size, comprs_time, pngquant_size, pngquant_time)
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

    // comprs lossy
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
    let comprs_time = start.elapsed();

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

    Some((lossy_buf.len(), comprs_time, pq_size, pq_time))
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
        jpeg::encode_with_options_into(&mut buf, pixels, width, height, 85, ColorType::Rgb, opts)
            .unwrap();
    }

    // Measure
    let start = Instant::now();
    let iterations = 10;
    for _ in 0..iterations {
        jpeg::encode_with_options_into(&mut buf, pixels, width, height, 85, ColorType::Rgb, opts)
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
        "target/wasm32-unknown-unknown/release/comprs.wasm",
        "target/wasm32-unknown-unknown/release/comprs_bg.wasm",
        "my-app/src/lib/comprs-wasm/comprs_bg.wasm",
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
// Custom Criterion Configuration
// ============================================================================

fn custom_criterion() -> Criterion {
    Criterion::default()
        .sample_size(50)
        .measurement_time(std::time::Duration::from_secs(5))
}

criterion_group! {
    name = benches;
    config = custom_criterion();
    targets = bench_png_all_presets, bench_png_lossy_comparison, bench_jpeg_all_presets, bench_deflate_comparison
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
