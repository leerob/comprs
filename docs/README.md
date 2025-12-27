# pixo Documentation

Welcome to the comprehensive documentation for **pixo**, a minimal-dependency, high-performance image compression library written in Rust.

This documentation is designed to be accessible to developers who may not be familiar with the low-level details of image compression. We use clear explanations, visual examples, and step-by-step breakdowns to help you understand not just *how* these algorithms work, but *why* they work.

**API docs (with embedded guides):** build locally with `cargo doc --all-features --open` or view on docs.rs once published (`docs.rs/pixo`).

## 📚 Documentation Guide

### Getting Started

| Document | Description |
|----------|-------------|
| [`Introduction to Image Compression`](crate::guides::introduction_to_image_compression) | Start here! Understand why we need image compression and the fundamental approaches |
| [`Introduction to Rust`](crate::guides::introduction_to_rust) | Learn Rust through compression code examples from this library |

### Core Compression Algorithms

| Document | Description |
|----------|-------------|
| [`Huffman Coding`](crate::guides::huffman_coding) | Learn how variable-length codes achieve optimal compression |
| [`LZ77 Compression`](crate::guides::lz77_compression) | Understand dictionary-based compression with sliding windows |
| [`DEFLATE Algorithm`](crate::guides::deflate) | See how LZ77 and Huffman combine for powerful compression |

### Image Format Specifics

| Document | Description |
|----------|-------------|
| [`PNG Encoding`](crate::guides::png_encoding) | Lossless image compression with predictive filtering |
| [`JPEG Encoding`](crate::guides::jpeg_encoding) | Lossy compression pipeline overview |
| [`Decoding`](crate::guides::decoding) | The decoder side of the codec: PNG and JPEG pipelines |
| [`Discrete Cosine Transform (DCT)`](crate::guides::dct) | The mathematical heart of JPEG compression |
| [`JPEG Quantization`](crate::guides::quantization) | How JPEG achieves its dramatic compression ratios |

### Performance & Implementation

| Document | Description |
|----------|-------------|
| [`Performance Optimization`](crate::guides::performance_optimization) | Techniques for high-performance compression code |
| [`Compression Evolution`](crate::guides::compression_evolution) | History and philosophy of compression improvements |
| [Benchmarks](../benches/BENCHMARKS.md) | Comprehensive comparison with oxipng, mozjpeg, and other tools |

## 🎯 Learning Path

If you're new to image compression, we recommend reading the documents in this order:

1. **[`Introduction to Image Compression`](crate::guides::introduction_to_image_compression)** - Foundational concepts
2. **[`Introduction to Rust`](crate::guides::introduction_to_rust)** - Understand the implementation language
3. **[`Huffman Coding`](crate::guides::huffman_coding)** - Core entropy coding technique
4. **[`LZ77 Compression`](crate::guides::lz77_compression)** - Dictionary compression basics
5. **[`DEFLATE Algorithm`](crate::guides::deflate)** - Combining the above for PNG
6. **[`PNG Encoding`](crate::guides::png_encoding)** - Complete lossless pipeline
7. **[`Discrete Cosine Transform`](crate::guides::dct)** - Mathematical foundations for JPEG
8. **[`JPEG Quantization`](crate::guides::quantization)** - Controlled information loss
9. **[`JPEG Encoding`](crate::guides::jpeg_encoding)** - Complete lossy pipeline
10. **[`Decoding`](crate::guides::decoding)** - The other half of the codec
11. **[`Performance Optimization`](crate::guides::performance_optimization)** - Making it all fast
12. **[`Compression Evolution`](crate::guides::compression_evolution)** - History and advanced techniques

## 🔧 Implementation Details

Each document includes:

- **Conceptual explanations** with real-world analogies
- **Visual examples** and diagrams (in ASCII art for portability)
- **Worked examples** with actual numbers
- **Common pitfalls** to help you avoid mistakes
- **Code references** to the relevant implementation in this library
- **RFC references** for the definitive specifications

## 📖 Key References

This library implements algorithms defined in these standards:

- **[RFC 1951](https://www.rfc-editor.org/rfc/rfc1951)** - DEFLATE Compressed Data Format
- **[RFC 2083](https://www.w3.org/TR/PNG/)** - PNG (Portable Network Graphics) Specification
- **[ITU-T T.81](https://www.w3.org/Graphics/JPEG/itu-t81.pdf)** - JPEG Standard (baseline DCT)

## 💡 Philosophy

> *"The best way to understand compression is to understand that all data has patterns, and compression is simply the art of describing those patterns more efficiently."*

A complete compression system is a **codec** (coder + decoder). While this library focuses on encoding, understanding both halves deepens your grasp of how compression works. The techniques you learn here — entropy coding, dictionary compression, transform coding — transfer directly to audio, video, network protocols, and beyond.

Every algorithm in this library exploits some form of redundancy:

| Algorithm | Type of Redundancy Exploited |
|-----------|------------------------------|
| Huffman Coding | Statistical redundancy (some symbols occur more often) |
| LZ77 | Spatial redundancy (patterns repeat in data) |
| PNG Filtering | Predictive redundancy (adjacent pixels are similar) |
| DCT | Frequency redundancy (images have few high-frequency components) |
| Quantization | Perceptual redundancy (humans can't see small changes) |

Understanding these principles will help you reason about when and why each algorithm is effective.

Beyond correctness, performance matters. See [Performance Optimization](./performance-optimization.md) for how we make these algorithms fast through techniques like SIMD, lookup tables, and algorithm selection.
