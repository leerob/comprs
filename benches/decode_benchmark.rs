#![cfg(feature = "decode")]

//! PNG decode benchmarks.
//!
//! Run with:
//! ```bash
//! cargo bench --bench decode_benchmark --features decode
//! ```

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

use comprs::png;
use comprs::ColorType;

fn gradient(width: u32, height: u32) -> Vec<u8> {
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

fn bench_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("PNG decode");
    for &size in &[128u32, 256, 512] {
        let pixels = gradient(size, size);
        let encoded = png::encode(&pixels, size, size, ColorType::Rgb).unwrap();
        let bytes = encoded.len() as u64;
        group.throughput(Throughput::Bytes(bytes));

        group.bench_with_input(
            BenchmarkId::new("comprs_decode", format!("{size}x{size}")),
            &encoded,
            |b, data| {
                b.iter(|| {
                    let decoded = png::decode(data).unwrap();
                    criterion::black_box(decoded.data.len());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("image_crate_decode", format!("{size}x{size}")),
            &encoded,
            |b, data| {
                b.iter(|| {
                    let img = image::load_from_memory(data).unwrap();
                    criterion::black_box(img.to_rgba8().len());
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_decode);
criterion_main!(benches);
