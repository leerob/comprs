# pixo

[![crates.io](https://img.shields.io/crates/v/pixo.svg)](https://crates.io/crates/pixo)
[![docs.rs](https://docs.rs/pixo/badge.svg)](https://docs.rs/pixo)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

A minimal-dependency, high-performance image compression library written in Rust.

This is a work-in-progress and exploration of coding agents.

<img width="459" height="256" alt="Pixo logo" src="https://github.com/user-attachments/assets/8d2e7acb-8a63-4c82-936c-9ae9cc9ce9f2" />

## Features

- **Zero runtime dependencies** — All encoding algorithms implemented from scratch
- **Entirely in Rust** — No C or C++ codecs required
- **PNG and JPEG support** — Lossless PNG, lossy PNG (quantization), and lossy JPEG
- **Small WASM binary** — 149 KB with competitive compression ([benchmarks](./benches/BENCHMARKS.md))
- **Well-tested** — 82% code coverage, 851 tests ([codebase comparison](./docs/codebase-size-comparison.md))

## Usage

1. **[Playground](https://pixo.leerob.com)** (recommended) — Try it in your browser
2. **[WASM](https://docs.rs/pixo/latest/pixo/guides/wasm/index.html)** — Use in browser or Node.js applications
3. **[CLI](https://docs.rs/pixo/latest/pixo/guides/cli/index.html)** — Compress images from the command line
4. **[Rust Crate](https://docs.rs/pixo/latest/pixo)** — Use as a library in your Rust projects

## Documentation

Comprehensive guides explaining the algorithms and compression strategies:

- [Introduction to Image Compression](https://docs.rs/pixo/latest/pixo/guides/introduction_to_image_compression/index.html) — Why and how we compress images
- [Introduction to Rust](https://docs.rs/pixo/latest/pixo/guides/introduction_to_rust/index.html) — Rust features through the lens of pixo

### Core Algorithms

- [Huffman Coding](https://docs.rs/pixo/latest/pixo/guides/huffman_coding/index.html) — Variable-length codes based on symbol frequency
- [LZ77 Compression](https://docs.rs/pixo/latest/pixo/guides/lz77_compression/index.html) — Dictionary-based compression with sliding windows
- [DEFLATE Algorithm](https://docs.rs/pixo/latest/pixo/guides/deflate/index.html) — How LZ77 and Huffman combine

### Image Formats

- [PNG Encoding](https://docs.rs/pixo/latest/pixo/guides/png_encoding/index.html) — Lossless compression with predictive filtering
- [JPEG Encoding](https://docs.rs/pixo/latest/pixo/guides/jpeg_encoding/index.html) — Lossy compression pipeline
- [Discrete Cosine Transform](https://docs.rs/pixo/latest/pixo/guides/dct/index.html) — The mathematical heart of JPEG
- [Quantization](https://docs.rs/pixo/latest/pixo/guides/quantization/index.html) — How JPEG achieves compression
