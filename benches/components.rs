//! Component-level microbenchmarks for pixo.
//! Focuses on LZ77, Huffman encoding, filtering, and checksums.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use pixo::compress::deflate::{
    deflate, deflate_packed, deflate_zlib, deflate_zlib_packed, encode_dynamic_huffman,
    encode_fixed_huffman,
};
use pixo::compress::lz77::Lz77Compressor;
use pixo::compress::{adler32, crc32};
use pixo::png::filter::apply_filters;
use pixo::png::{FilterStrategy, PngOptions};

fn make_pattern(len: usize) -> Vec<u8> {
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

fn gradient_image(width: u32, height: u32) -> Vec<u8> {
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

fn bench_lz77(c: &mut Criterion) {
    let compressible = make_pattern(1 << 20);
    let random = make_random(1 << 20, 0x1234_5678);

    let mut group = c.benchmark_group("lz77_compress");
    group.throughput(Throughput::Bytes(compressible.len() as u64));

    group.bench_with_input(
        BenchmarkId::new("compressible_level6", "1mb"),
        &compressible,
        |b, data| {
            let mut compressor = Lz77Compressor::new(6);
            b.iter(|| {
                black_box(compressor.compress(black_box(data)));
            });
        },
    );

    group.bench_with_input(
        BenchmarkId::new("random_level6", "1mb"),
        &random,
        |b, data| {
            let mut compressor = Lz77Compressor::new(6);
            b.iter(|| {
                black_box(compressor.compress(black_box(data)));
            });
        },
    );

    group.finish();
}

fn bench_huffman(c: &mut Criterion) {
    // Prepare token streams once to isolate Huffman encoding.
    let source = make_pattern(256 * 1024);
    let mut compressor = Lz77Compressor::new(6);
    let tokens = compressor.compress(&source);

    let random = make_random(256 * 1024, 0x9E37_79B9);
    let mut compressor_random = Lz77Compressor::new(6);
    let random_tokens = compressor_random.compress(&random);

    let mut group = c.benchmark_group("huffman_encode");
    group.throughput(Throughput::Elements(tokens.len() as u64));

    group.bench_function("fixed_compressible", |b| {
        b.iter(|| {
            black_box(encode_fixed_huffman(black_box(&tokens)));
        });
    });

    group.bench_function("dynamic_compressible", |b| {
        b.iter(|| {
            black_box(encode_dynamic_huffman(black_box(&tokens)));
        });
    });

    group.bench_function("fixed_random", |b| {
        b.iter(|| {
            black_box(encode_fixed_huffman(black_box(&random_tokens)));
        });
    });

    group.bench_function("dynamic_random", |b| {
        b.iter(|| {
            black_box(encode_dynamic_huffman(black_box(&random_tokens)));
        });
    });

    group.finish();
}

fn bench_filters(c: &mut Criterion) {
    let width = 512;
    let height = 512;
    let pixels = gradient_image(width, height);

    let mut group = c.benchmark_group("png_filters");
    let bytes = pixels.len() as u64;
    group.throughput(Throughput::Bytes(bytes));

    group.bench_function("adaptive_512_rgb", |b| {
        let options = PngOptions {
            filter_strategy: FilterStrategy::Adaptive,
            ..Default::default()
        };
        b.iter(|| {
            black_box(apply_filters(
                black_box(&pixels),
                width,
                height,
                3,
                black_box(&options),
            ));
        });
    });

    group.bench_function("adaptive_fast_512_rgb", |b| {
        let options = PngOptions {
            filter_strategy: FilterStrategy::AdaptiveFast,
            ..Default::default()
        };
        b.iter(|| {
            black_box(apply_filters(
                black_box(&pixels),
                width,
                height,
                3,
                black_box(&options),
            ));
        });
    });

    group.finish();
}

fn bench_checksums(c: &mut Criterion) {
    let data = make_random(1 << 20, 0xDEAD_BEEF);

    let mut group = c.benchmark_group("checksums");
    group.throughput(Throughput::Bytes(data.len() as u64));

    group.bench_function("adler32_1mb", |b| {
        b.iter(|| black_box(adler32::adler32(black_box(&data))));
    });

    group.bench_function("crc32_1mb", |b| {
        b.iter(|| black_box(crc32::crc32(black_box(&data))));
    });

    group.finish();
}

fn bench_deflate_packed(c: &mut Criterion) {
    let compressible = make_pattern(1 << 20);
    let random = make_random(1 << 20, 0xCAFE_BABE);

    let mut group = c.benchmark_group("deflate_vs_packed");
    group.throughput(Throughput::Bytes(compressible.len() as u64));

    group.bench_function("standard_compressible_level6", |b| {
        b.iter(|| {
            black_box(deflate(black_box(&compressible), 6));
        });
    });

    group.bench_function("packed_compressible_level6", |b| {
        b.iter(|| {
            black_box(deflate_packed(black_box(&compressible), 6));
        });
    });

    group.bench_function("standard_random_level6", |b| {
        b.iter(|| {
            black_box(deflate(black_box(&random), 6));
        });
    });

    group.bench_function("packed_random_level6", |b| {
        b.iter(|| {
            black_box(deflate_packed(black_box(&random), 6));
        });
    });

    group.finish();
}

fn bench_deflate_zlib_packed(c: &mut Criterion) {
    let compressible = make_pattern(1 << 20);
    let random = make_random(1 << 20, 0xDEAD_BEEF);

    let mut group = c.benchmark_group("deflate_zlib_vs_packed");
    group.throughput(Throughput::Bytes(compressible.len() as u64));

    group.bench_function("standard_zlib_compressible_level6", |b| {
        b.iter(|| {
            black_box(deflate_zlib(black_box(&compressible), 6));
        });
    });

    group.bench_function("packed_zlib_compressible_level6", |b| {
        b.iter(|| {
            black_box(deflate_zlib_packed(black_box(&compressible), 6));
        });
    });

    group.bench_function("standard_zlib_random_level6", |b| {
        b.iter(|| {
            black_box(deflate_zlib(black_box(&random), 6));
        });
    });

    group.bench_function("packed_zlib_random_level6", |b| {
        b.iter(|| {
            black_box(deflate_zlib_packed(black_box(&random), 6));
        });
    });

    group.finish();
}

fn bench_png_encode(c: &mut Criterion) {
    let width = 512;
    let height = 512;
    let pixels = gradient_image(width, height);
    let bytes = pixels.len() as u64;

    let mut group = c.benchmark_group("png_encode_512_rgb");
    group.throughput(Throughput::Bytes(bytes));

    group.bench_function("pixo_default", |b| {
        b.iter(|| {
            let opts = pixo::png::PngOptions::balanced(width, height);
            black_box(pixo::png::encode(black_box(&pixels), &opts).unwrap())
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_lz77,
    bench_huffman,
    bench_filters,
    bench_checksums,
    bench_deflate_packed,
    bench_deflate_zlib_packed,
    bench_png_encode
);
criterion_main!(benches);
