//! Quick size and timing snapshot for PNG/JPEG encoders.
//!
//! Run with:
//!   cargo bench --bench size_snapshot -- --nocapture
//!
//! Outputs mean encode time (ms) and output sizes for a small set of images.

use std::time::{Duration, Instant};

use comprs::{jpeg, png, ColorType};
use image::ImageEncoder;

fn generate_gradient(width: u32, height: u32) -> Vec<u8> {
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

fn generate_noisy(width: u32, height: u32) -> Vec<u8> {
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

fn mean_time_and_size<F: FnMut() -> Vec<u8>>(mut f: F) -> (Duration, usize) {
    // Warmup
    for _ in 0..2 {
        let _ = f();
    }
    let iterations = 5;
    let mut total = Duration::ZERO;
    let mut last_size = 0usize;
    for _ in 0..iterations {
        let start = Instant::now();
        let out = f();
        total += start.elapsed();
        last_size = out.len();
    }
    (total / iterations, last_size)
}

fn main() {
    let cases = [
        ("gradient", generate_gradient as fn(u32, u32) -> Vec<u8>),
        ("noisy", generate_noisy as fn(u32, u32) -> Vec<u8>),
    ];

    let sizes = [(256u32, 256u32)];

    for (name, gen) in cases {
        for (w, h) in sizes {
            let pixels = gen(w, h);
            let label = format!("{} {}x{}", name, w, h);
            println!("==> {}", label);

            // PNG comprs (default)
            let mut png_buf = Vec::new();
            let (t_png_comprs, sz_png_comprs) = mean_time_and_size(|| {
                png::encode_into(
                    &mut png_buf,
                    &pixels,
                    w,
                    h,
                    ColorType::Rgb,
                    &png::PngOptions::default(),
                )
                .unwrap();
                png_buf.clone()
            });
            println!(
                "PNG comprs: {:.3} ms, {} bytes",
                t_png_comprs.as_secs_f64() * 1000.0,
                sz_png_comprs
            );

            // PNG fast preset
            let mut png_buf_fast = Vec::new();
            let (t_png_fast, sz_png_fast) = mean_time_and_size(|| {
                png::encode_into(
                    &mut png_buf_fast,
                    &pixels,
                    w,
                    h,
                    ColorType::Rgb,
                    &png::PngOptions::fast(),
                )
                .unwrap();
                png_buf_fast.clone()
            });
            println!(
                "PNG comprs fast: {:.3} ms, {} bytes",
                t_png_fast.as_secs_f64() * 1000.0,
                sz_png_fast
            );

            // PNG max preset
            let mut png_buf_max = Vec::new();
            let (t_png_max, sz_png_max) = mean_time_and_size(|| {
                png::encode_into(
                    &mut png_buf_max,
                    &pixels,
                    w,
                    h,
                    ColorType::Rgb,
                    &png::PngOptions::max_compression(),
                )
                .unwrap();
                png_buf_max.clone()
            });
            println!(
                "PNG comprs max: {:.3} ms, {} bytes",
                t_png_max.as_secs_f64() * 1000.0,
                sz_png_max
            );

            // PNG image crate
            let (t_png_image, sz_png_image) = mean_time_and_size(|| {
                let mut output = Vec::new();
                let encoder = image::codecs::png::PngEncoder::new(&mut output);
                encoder
                    .write_image(&pixels, w, h, image::ColorType::Rgb8)
                    .unwrap();
                output
            });
            println!(
                "PNG image:  {:.3} ms, {} bytes",
                t_png_image.as_secs_f64() * 1000.0,
                sz_png_image
            );

            // JPEG comprs Q85 4:4:4
            let mut jpeg_buf = Vec::new();
            let (t_jpeg_comprs_444, sz_jpeg_comprs_444) = mean_time_and_size(|| {
                jpeg::encode_with_options_into(
                    &mut jpeg_buf,
                    &pixels,
                    w,
                    h,
                    85,
                    ColorType::Rgb,
                    &jpeg::JpegOptions {
                        quality: 85,
                        subsampling: jpeg::Subsampling::S444,
                        restart_interval: None,
                    },
                )
                .unwrap();
                jpeg_buf.clone()
            });
            println!(
                "JPEG comprs q85 444: {:.3} ms, {} bytes",
                t_jpeg_comprs_444.as_secs_f64() * 1000.0,
                sz_jpeg_comprs_444
            );

            // JPEG preset fast (uses 4:2:0, Q75)
            let mut jpeg_buf_fast = Vec::new();
            let fast_opts = jpeg::JpegOptions::fast();
            let (t_jpeg_fast, sz_jpeg_fast) = mean_time_and_size(|| {
                jpeg::encode_with_options_into(
                    &mut jpeg_buf_fast,
                    &pixels,
                    w,
                    h,
                    fast_opts.quality,
                    ColorType::Rgb,
                    &fast_opts,
                )
                .unwrap();
                jpeg_buf_fast.clone()
            });
            println!(
                "JPEG comprs fast:   {:.3} ms, {} bytes",
                t_jpeg_fast.as_secs_f64() * 1000.0,
                sz_jpeg_fast
            );

            // JPEG comprs Q85 4:2:0
            let mut jpeg_buf_420 = Vec::new();
            let (t_jpeg_comprs_420, sz_jpeg_comprs_420) = mean_time_and_size(|| {
                jpeg::encode_with_options_into(
                    &mut jpeg_buf_420,
                    &pixels,
                    w,
                    h,
                    85,
                    ColorType::Rgb,
                    &jpeg::JpegOptions {
                        quality: 85,
                        subsampling: jpeg::Subsampling::S420,
                        restart_interval: None,
                    },
                )
                .unwrap();
                jpeg_buf_420.clone()
            });
            println!(
                "JPEG comprs q85 420: {:.3} ms, {} bytes",
                t_jpeg_comprs_420.as_secs_f64() * 1000.0,
                sz_jpeg_comprs_420
            );

            // JPEG image crate Q85
            let (t_jpeg_image, sz_jpeg_image) = mean_time_and_size(|| {
                let mut output = Vec::new();
                let encoder =
                    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut output, 85);
                encoder
                    .write_image(&pixels, w, h, image::ColorType::Rgb8)
                    .unwrap();
                output
            });
            println!(
                "JPEG image q85:      {:.3} ms, {} bytes",
                t_jpeg_image.as_secs_f64() * 1000.0,
                sz_jpeg_image
            );

            println!();
        }
    }
}
