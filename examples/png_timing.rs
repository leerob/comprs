//! Simple timing helper for PNG deflate stages.
//!
//! Requires the `timing` feature to surface `DeflateStats`.
//!
//! Usage:
//!   cargo run --release --features timing --example png_timing
//!   cargo run --release --features timing --example png_timing noise 1024 1024

#[cfg(feature = "timing")]
fn main() {
    use comprs::compress::deflate::deflate_zlib_packed_with_stats;
    use comprs::png::{filter, PngOptions};
    use std::env;
    use std::time::Instant;

    let mut args = env::args().skip(1);
    let pattern = args.next().unwrap_or_else(|| "gradient".to_string());
    let width: u32 = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    let height: u32 = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);

    let bytes_per_pixel = 4; // RGBA
    let mut pixels = vec![0u8; width as usize * height as usize * bytes_per_pixel];
    match pattern.as_str() {
        "noise" | "noisy" => fill_noise_rgba(&mut pixels),
        _ => fill_gradient_rgba(&mut pixels, width, height),
    }

    let options = PngOptions::fast();

    let t_filter = Instant::now();
    let filtered = filter::apply_filters(
        &pixels,
        width,
        height,
        bytes_per_pixel as usize,
        &options,
    );
    let filter_time = t_filter.elapsed();

    let t_deflate = Instant::now();
    let (compressed, stats) =
        deflate_zlib_packed_with_stats(&filtered, options.compression_level);
    let deflate_time = t_deflate.elapsed();

    println!("Pattern: {pattern}, size: {width}x{height}, bpp: {bytes_per_pixel}");
    println!("Filtered bytes: {}", filtered.len());
    println!(
        "Filter time: {:.3} ms",
        filter_time.as_secs_f64() * 1000.0
    );
    println!(
        "Deflate total: {:.3} ms (lz77 {:.3} ms, fixed {:.3} ms, dynamic {:.3} ms, choose {:.3} ms)",
        deflate_time.as_secs_f64() * 1000.0,
        stats.lz77_time.as_secs_f64() * 1000.0,
        stats.fixed_huffman_time.as_secs_f64() * 1000.0,
        stats.dynamic_huffman_time.as_secs_f64() * 1000.0,
        stats.choose_time.as_secs_f64() * 1000.0,
    );
    println!(
        "Tokens: {} (literals {}, matches {}), used_dynamic: {}, used_stored: {}, compressed_len: {}",
        stats.token_count,
        stats.literal_count,
        stats.match_count,
        stats.used_dynamic,
        stats.used_stored_block,
        compressed.len()
    );
}

#[cfg(not(feature = "timing"))]
fn main() {
    eprintln!("This example requires the `timing` feature. Re-run with --features timing");
}

#[cfg(feature = "timing")]
fn fill_gradient_rgba(buf: &mut [u8], width: u32, height: u32) {
    for y in 0..height {
        for x in 0..width {
            let idx = ((y * width + x) * 4) as usize;
            let r = ((x * 255) / (width.saturating_sub(1).max(1))) as u8;
            let g = ((y * 255) / (height.saturating_sub(1).max(1))) as u8;
            let b = ((x + y) * 255 / (width + height).saturating_sub(2).max(1)) as u8;
            buf[idx] = r;
            buf[idx + 1] = g;
            buf[idx + 2] = b;
            buf[idx + 3] = 255;
        }
    }
}

#[cfg(feature = "timing")]
fn fill_noise_rgba(buf: &mut [u8]) {
    let mut state: u64 = 0x1234_5678_9ABC_DEF0;
    for chunk in buf.chunks_mut(4) {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let v = state >> 32;
        chunk[0] = v as u8;
        chunk[1] = (v >> 8) as u8;
        chunk[2] = (v >> 16) as u8;
        chunk[3] = (v >> 24) as u8;
    }
}
