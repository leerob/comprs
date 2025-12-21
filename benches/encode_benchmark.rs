//! Benchmarks for comprs image encoding.
//!
//! Compare against the `image` crate for PNG and JPEG encoding.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use image::ImageEncoder;

use comprs::{jpeg, png, ColorType};

/// Generate a test image with gradient pattern.
fn generate_test_image(width: u32, height: u32) -> Vec<u8> {
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

/// Generate a test image with random-ish pattern (harder to compress).
fn generate_noisy_image(width: u32, height: u32) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((width * height * 3) as usize);
    let mut seed = 12345u32;
    for _ in 0..(width * height) {
        // Simple LCG for deterministic "random" values
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

fn png_encoding_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("PNG Encoding");

    for size in [64, 128, 256, 512].iter() {
        let pixels = generate_test_image(*size, *size);
        let pixel_bytes = (*size as u64) * (*size as u64) * 3;

        group.throughput(Throughput::Bytes(pixel_bytes));

        group.bench_with_input(
            BenchmarkId::new("comprs", format!("{}x{}", size, size)),
            &pixels,
            |b, pixels| {
                b.iter(|| {
                    png::encode(black_box(pixels), *size, *size, ColorType::Rgb).unwrap()
                });
            },
        );

        // Compare with image crate
        group.bench_with_input(
            BenchmarkId::new("image_crate", format!("{}x{}", size, size)),
            &pixels,
            |b, pixels| {
                b.iter(|| {
                    let mut output = Vec::new();
                    let encoder = image::codecs::png::PngEncoder::new(&mut output);
                    encoder
                        .write_image(
                            black_box(pixels),
                            *size,
                            *size,
                            image::ExtendedColorType::Rgb8,
                        )
                        .unwrap();
                    output
                });
            },
        );
    }

    group.finish();
}

fn jpeg_encoding_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("JPEG Encoding");

    for size in [64, 128, 256, 512].iter() {
        let pixels = generate_test_image(*size, *size);
        let pixel_bytes = (*size as u64) * (*size as u64) * 3;

        group.throughput(Throughput::Bytes(pixel_bytes));

        let opts_420 = jpeg::JpegOptions {
            quality: 85,
            subsampling: jpeg::Subsampling::S420,
            restart_interval: None,
        };

        group.bench_with_input(
            BenchmarkId::new("comprs_q85_444", format!("{}x{}", size, size)),
            &pixels,
            |b, pixels| {
                b.iter(|| {
                    jpeg::encode(black_box(pixels), *size, *size, 85).unwrap()
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("comprs_q85_420", format!("{}x{}", size, size)),
            &pixels,
            |b, pixels| {
                b.iter(|| {
                    jpeg::encode_with_options(
                        black_box(pixels),
                        *size,
                        *size,
                        85,
                        ColorType::Rgb,
                        &opts_420,
                    )
                    .unwrap()
                });
            },
        );

        // Compare with image crate
        group.bench_with_input(
            BenchmarkId::new("image_crate_q85", format!("{}x{}", size, size)),
            &pixels,
            |b, pixels| {
                b.iter(|| {
                    let mut output = Vec::new();
                    let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut output, 85);
                    encoder
                        .write_image(
                            black_box(pixels),
                            *size,
                            *size,
                            image::ExtendedColorType::Rgb8,
                        )
                        .unwrap();
                    output
                });
            },
        );
    }

    group.finish();
}

fn compression_ratio_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Compression Ratio");

    // Test with different image types
    let sizes = [(256, 256)];

    for (width, height) in sizes.iter() {
        let gradient = generate_test_image(*width, *height);
        let noisy = generate_noisy_image(*width, *height);

        // Gradient image - should compress well
        group.bench_function("PNG gradient (comprs)", |b| {
            b.iter(|| {
                let result = png::encode(black_box(&gradient), *width, *height, ColorType::Rgb).unwrap();
                result.len()
            });
        });

        // Noisy image - harder to compress
        group.bench_function("PNG noisy (comprs)", |b| {
            b.iter(|| {
                let result = png::encode(black_box(&noisy), *width, *height, ColorType::Rgb).unwrap();
                result.len()
            });
        });

        // PNG via image crate for comparison
        group.bench_function("PNG gradient (image crate)", |b| {
            b.iter(|| {
                let mut output = Vec::new();
                let encoder = image::codecs::png::PngEncoder::new(&mut output);
                encoder
                    .write_image(
                        black_box(&gradient),
                        *width,
                        *height,
                        image::ExtendedColorType::Rgb8,
                    )
                    .unwrap();
                output.len()
            });
        });

        group.bench_function("PNG noisy (image crate)", |b| {
            b.iter(|| {
                let mut output = Vec::new();
                let encoder = image::codecs::png::PngEncoder::new(&mut output);
                encoder
                    .write_image(
                        black_box(&noisy),
                        *width,
                        *height,
                        image::ExtendedColorType::Rgb8,
                    )
                    .unwrap();
                output.len()
            });
        });

        // JPEG quality comparison
        for quality in [50, 75, 90].iter() {
            group.bench_function(format!("JPEG q{} (comprs)", quality), |b| {
                b.iter(|| {
                    let result = jpeg::encode(black_box(&gradient), *width, *height, *quality).unwrap();
                    result.len()
                });
            });

            group.bench_function(format!("JPEG q{} (image crate)", quality), |b| {
                b.iter(|| {
                    let mut output = Vec::new();
                    let encoder =
                        image::codecs::jpeg::JpegEncoder::new_with_quality(&mut output, *quality);
                    encoder
                        .write_image(
                            black_box(&gradient),
                            *width,
                            *height,
                            image::ExtendedColorType::Rgb8,
                        )
                        .unwrap();
                    output.len()
                });
            });
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    png_encoding_benchmark,
    jpeg_encoding_benchmark,
    compression_ratio_benchmark,
);
criterion_main!(benches);
