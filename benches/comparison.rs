//! Comprehensive library comparison benchmark.
//!
//! Compares comprs against popular image compression libraries and produces
//! a detailed summary table similar to the benchmarks/README.md format.
//!
//! Run with: cargo bench --bench comparison
//!
//! For a quick summary without full benchmarks:
//!   cargo bench --bench comparison -- --quick

use std::io::Write as IoWrite;
use std::time::{Duration, Instant};

use criterion::{black_box, criterion_group, BenchmarkId, Criterion, Throughput};
use flate2::{write::ZlibEncoder, Compression};
use image::ImageEncoder;

use comprs::compress::deflate::deflate_zlib;
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
// PNG Encoding Comparison
// ============================================================================

fn bench_png_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("PNG Comparison");

    for size in [256, 512].iter() {
        let gradient = generate_gradient_image(*size, *size);
        let noisy = generate_noisy_image(*size, *size);
        let pixel_bytes = (*size as u64) * (*size as u64) * 3;

        group.throughput(Throughput::Bytes(pixel_bytes));

        // --- Gradient Image ---
        let mut png_buf = Vec::new();
        group.bench_with_input(
            BenchmarkId::new("comprs_gradient", format!("{size}x{size}")),
            &gradient,
            |b, pixels| {
                b.iter(|| {
                    png::encode_into(
                        &mut png_buf,
                        black_box(pixels),
                        *size,
                        *size,
                        ColorType::Rgb,
                        &png::PngOptions::default(),
                    )
                    .unwrap()
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("image_crate_gradient", format!("{size}x{size}")),
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

        // --- Noisy Image ---
        group.bench_with_input(
            BenchmarkId::new("comprs_noisy", format!("{size}x{size}")),
            &noisy,
            |b, pixels| {
                b.iter(|| {
                    png::encode_into(
                        &mut png_buf,
                        black_box(pixels),
                        *size,
                        *size,
                        ColorType::Rgb,
                        &png::PngOptions::default(),
                    )
                    .unwrap()
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("image_crate_noisy", format!("{size}x{size}")),
            &noisy,
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
// JPEG Encoding Comparison
// ============================================================================

fn bench_jpeg_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("JPEG Comparison");

    for size in [256, 512].iter() {
        let gradient = generate_gradient_image(*size, *size);
        let pixel_bytes = (*size as u64) * (*size as u64) * 3;

        group.throughput(Throughput::Bytes(pixel_bytes));

        let mut jpeg_buf = Vec::new();

        // comprs with 4:4:4 subsampling
        group.bench_with_input(
            BenchmarkId::new("comprs_q85_444", format!("{size}x{size}")),
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
                        &jpeg::JpegOptions {
                            quality: 85,
                            subsampling: jpeg::Subsampling::S444,
                            restart_interval: None,
                        },
                    )
                    .unwrap()
                });
            },
        );

        // comprs with 4:2:0 subsampling
        group.bench_with_input(
            BenchmarkId::new("comprs_q85_420", format!("{size}x{size}")),
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
                        &jpeg::JpegOptions {
                            quality: 85,
                            subsampling: jpeg::Subsampling::S420,
                            restart_interval: None,
                        },
                    )
                    .unwrap()
                });
            },
        );

        // image crate
        group.bench_with_input(
            BenchmarkId::new("image_crate_q85", format!("{size}x{size}")),
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
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════════════════════╗");
    println!("║                         COMPRS BENCHMARK SUMMARY                             ║");
    println!("╚══════════════════════════════════════════════════════════════════════════════╝");
    println!();

    // --- Binary Size Comparison ---
    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ WASM Binary Size Comparison                                                  │");
    println!("├────────────────────┬─────────────┬────────────────────────────────────────────┤");
    println!("│ Library            │ WASM Size   │ Notes                                      │");
    println!("├────────────────────┼─────────────┼────────────────────────────────────────────┤");

    // Try to measure actual comprs WASM size
    let comprs_wasm_size = get_wasm_size();
    let comprs_size_str = match comprs_wasm_size {
        Some(size) => format_size(size),
        None => "~92 KB".to_string(),
    };

    println!(
        "│ {:<18} │ {:>11} │ {:<42} │",
        "comprs", comprs_size_str, "Zero deps, pure Rust"
    );
    println!(
        "│ {:<18} │ {:>11} │ {:<42} │",
        "image crate", "~2-4 MB", "Pure Rust, many codecs"
    );
    println!(
        "│ {:<18} │ {:>11} │ {:<42} │",
        "photon-rs", "~200-400 KB", "Pure Rust, WASM optimized"
    );
    println!(
        "│ {:<18} │ {:>11} │ {:<42} │",
        "zune-image", "~500 KB-1 MB", "Pure Rust, SIMD optimized"
    );
    println!(
        "│ {:<18} │ {:>11} │ {:<42} │",
        "wasm-mozjpeg", "~208 KB", "Emscripten compiled"
    );
    println!("└────────────────────┴─────────────┴────────────────────────────────────────────┘");
    println!();

    // --- Output Size Comparison ---
    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Compression Output Size (512x512 gradient image)                             │");
    println!("├────────────────────┬─────────────┬─────────────┬──────────────────────────────┤");
    println!("│ Format             │ comprs      │ image crate │ Ratio                        │");
    println!("├────────────────────┼─────────────┼─────────────┼──────────────────────────────┤");

    // Measure actual output sizes
    let gradient = generate_gradient_image(512, 512);

    // PNG sizes
    let mut comprs_png = Vec::new();
    png::encode_into(
        &mut comprs_png,
        &gradient,
        512,
        512,
        ColorType::Rgb,
        &png::PngOptions::default(),
    )
    .unwrap();

    let mut image_png = Vec::new();
    let encoder = image::codecs::png::PngEncoder::new(&mut image_png);
    encoder
        .write_image(&gradient, 512, 512, image::ColorType::Rgb8)
        .unwrap();

    let png_ratio = comprs_png.len() as f64 / image_png.len() as f64;
    println!(
        "│ {:<18} │ {:>11} │ {:>11} │ {:>28.2}x │",
        "PNG",
        format_size(comprs_png.len()),
        format_size(image_png.len()),
        png_ratio
    );

    // JPEG sizes
    let mut comprs_jpeg = Vec::new();
    jpeg::encode_with_options_into(
        &mut comprs_jpeg,
        &gradient,
        512,
        512,
        85,
        ColorType::Rgb,
        &jpeg::JpegOptions {
            quality: 85,
            subsampling: jpeg::Subsampling::S444,
            restart_interval: None,
        },
    )
    .unwrap();

    let mut image_jpeg = Vec::new();
    let jpeg_encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut image_jpeg, 85);
    jpeg_encoder
        .write_image(&gradient, 512, 512, image::ColorType::Rgb8)
        .unwrap();

    let jpeg_ratio = comprs_jpeg.len() as f64 / image_jpeg.len() as f64;
    println!(
        "│ {:<18} │ {:>11} │ {:>11} │ {:>28.2}x │",
        "JPEG (q85, 4:4:4)",
        format_size(comprs_jpeg.len()),
        format_size(image_jpeg.len()),
        jpeg_ratio
    );

    // JPEG 4:2:0
    let mut comprs_jpeg_420 = Vec::new();
    jpeg::encode_with_options_into(
        &mut comprs_jpeg_420,
        &gradient,
        512,
        512,
        85,
        ColorType::Rgb,
        &jpeg::JpegOptions {
            quality: 85,
            subsampling: jpeg::Subsampling::S420,
            restart_interval: None,
        },
    )
    .unwrap();

    println!(
        "│ {:<18} │ {:>11} │ {:>11} │ {:>28} │",
        "JPEG (q85, 4:2:0)",
        format_size(comprs_jpeg_420.len()),
        "N/A",
        "-"
    );

    println!("└────────────────────┴─────────────┴─────────────┴──────────────────────────────┘");
    println!();

    // --- DEFLATE Comparison ---
    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ DEFLATE Compression (1 MB payload)                                           │");
    println!("├────────────────────┬─────────────┬─────────────┬──────────────────────────────┤");
    println!("│ Library            │ Output Size │ Ratio       │ Notes                        │");
    println!("├────────────────────┼─────────────┼─────────────┼──────────────────────────────┤");

    let compressible = make_compressible(1 << 20);
    let comprs_deflate = deflate_zlib(&compressible, 6);

    let mut flate2_enc = ZlibEncoder::new(Vec::new(), Compression::default());
    flate2_enc.write_all(&compressible).unwrap();
    let flate2_deflate = flate2_enc.finish().unwrap();

    let comprs_ratio = compressible.len() as f64 / comprs_deflate.len() as f64;
    let flate2_ratio = compressible.len() as f64 / flate2_deflate.len() as f64;

    println!(
        "│ {:<18} │ {:>11} │ {:>10.1}x │ {:<28} │",
        "comprs",
        format_size(comprs_deflate.len()),
        comprs_ratio,
        "Pure Rust, zero deps"
    );
    println!(
        "│ {:<18} │ {:>11} │ {:>10.1}x │ {:<28} │",
        "flate2",
        format_size(flate2_deflate.len()),
        flate2_ratio,
        "miniz_oxide backend"
    );
    println!("└────────────────────┴─────────────┴─────────────┴──────────────────────────────┘");
    println!();

    // --- Speed Summary ---
    println!("┌──────────────────────────────────────────────────────────────────────────────┐");
    println!("│ Speed Summary (512x512 image, avg of 10 iterations)                          │");
    println!("├────────────────────┬─────────────┬─────────────┬──────────────────────────────┤");
    println!("│ Operation          │ comprs      │ image crate │ Relative                     │");
    println!("├────────────────────┼─────────────┼─────────────┼──────────────────────────────┤");

    // PNG encoding speed
    let png_comprs_time = measure_time(|| {
        let mut buf = Vec::new();
        png::encode_into(
            &mut buf,
            &gradient,
            512,
            512,
            ColorType::Rgb,
            &png::PngOptions::default(),
        )
        .unwrap();
    });

    let png_image_time = measure_time(|| {
        let mut output = Vec::new();
        let encoder = image::codecs::png::PngEncoder::new(&mut output);
        encoder
            .write_image(&gradient, 512, 512, image::ColorType::Rgb8)
            .unwrap();
    });

    let png_relative = format_relative_speed(png_comprs_time, png_image_time);
    println!(
        "│ {:<18} │ {:>11} │ {:>11} │ {:>28} │",
        "PNG encode",
        format_duration(png_comprs_time),
        format_duration(png_image_time),
        png_relative
    );

    // JPEG encoding speed
    let jpeg_comprs_time = measure_time(|| {
        let mut buf = Vec::new();
        jpeg::encode_with_options_into(
            &mut buf,
            &gradient,
            512,
            512,
            85,
            ColorType::Rgb,
            &jpeg::JpegOptions {
                quality: 85,
                subsampling: jpeg::Subsampling::S444,
                restart_interval: None,
            },
        )
        .unwrap();
    });

    let jpeg_image_time = measure_time(|| {
        let mut output = Vec::new();
        let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut output, 85);
        encoder
            .write_image(&gradient, 512, 512, image::ColorType::Rgb8)
            .unwrap();
    });

    let jpeg_relative = format_relative_speed(jpeg_comprs_time, jpeg_image_time);
    println!(
        "│ {:<18} │ {:>11} │ {:>11} │ {:>28} │",
        "JPEG encode",
        format_duration(jpeg_comprs_time),
        format_duration(jpeg_image_time),
        jpeg_relative
    );

    println!("└────────────────────┴─────────────┴─────────────┴──────────────────────────────┘");
    println!();

    // --- Summary Notes ---
    println!("Notes:");
    println!("  • WASM sizes are approximate and depend on build configuration");
    println!("  • Speed measurements are single-iteration warm estimates (see Criterion for statistical analysis)");
    println!("  • Compression ratios vary significantly based on image content");
    println!("  • Run `cargo bench --bench comparison` for detailed statistical benchmarks");
    println!();
}

fn get_wasm_size() -> Option<usize> {
    // Try to find the WASM binary if it exists
    let paths = [
        "target/wasm32-unknown-unknown/release/comprs.wasm",
        "target/wasm32-unknown-unknown/release/comprs_bg.wasm",
    ];

    for path in paths {
        if let Ok(metadata) = std::fs::metadata(path) {
            return Some(metadata.len() as usize);
        }
    }

    // Try running wasm-pack build to get the size (if available)
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

fn format_relative_speed(comprs_time: Duration, other_time: Duration) -> String {
    let ratio = comprs_time.as_secs_f64() / other_time.as_secs_f64();
    if ratio < 1.0 {
        // comprs is faster
        let faster = 1.0 / ratio;
        format!("comprs {faster:.1}x faster")
    } else if ratio > 1.0 {
        // other is faster
        format!("image {ratio:.1}x faster")
    } else {
        "equal".to_string()
    }
}

fn measure_time<F: FnMut()>(mut f: F) -> Duration {
    // Warm up
    for _ in 0..3 {
        f();
    }

    // Measure average of 10 iterations
    let start = Instant::now();
    let iterations = 10;
    for _ in 0..iterations {
        f();
    }
    start.elapsed() / iterations
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
    targets = bench_png_comparison, bench_jpeg_comparison, bench_deflate_comparison
}

// Custom main that prints summary after benchmarks
fn main() {
    // Check if we should just print summary
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
