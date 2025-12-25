# comprs Documentation

Welcome to the comprehensive documentation for **comprs**, a minimal-dependency, high-performance image compression library written in Rust.

This documentation is designed to be accessible to developers who may not be familiar with the low-level details of image compression. We use clear explanations, visual examples, and step-by-step breakdowns to help you understand not just *how* these algorithms work, but *why* they work.

## 📚 Documentation Guide

### Getting Started

| Document | Description |
|----------|-------------|
| [Introduction to Image Compression](./introduction-to-image-compression.md) | Start here! Understand why we need image compression and the fundamental approaches |
| [Introduction to Rust](./introduction-to-rust.md) | Learn Rust through compression code examples from this library |

### Core Compression Algorithms

| Document | Description |
|----------|-------------|
| [Huffman Coding](./huffman-coding.md) | Learn how variable-length codes achieve optimal compression |
| [LZ77 Compression](./lz77-compression.md) | Understand dictionary-based compression with sliding windows |
| [DEFLATE Algorithm](./deflate.md) | See how LZ77 and Huffman combine for powerful compression |

### Image Format Specifics

| Document | Description |
|----------|-------------|
| [PNG Encoding](./png-encoding.md) | Lossless image compression with predictive filtering |
| [JPEG Encoding](./jpeg-encoding.md) | Lossy compression pipeline overview |
| [Discrete Cosine Transform (DCT)](./dct.md) | The mathematical heart of JPEG compression |
| [JPEG Quantization](./quantization.md) | How JPEG achieves its dramatic compression ratios |

### Performance & Implementation

| Document | Description |
|----------|-------------|
| [Performance Optimization](./performance-optimization.md) | Techniques for high-performance compression code |
| [Compression Evolution](./compression-evolution.md) | History and philosophy of compression improvements |
| [Benchmarks](../benches/BENCHMARKS.md) | Comprehensive comparison with oxipng, mozjpeg, and other tools |

## 🎯 Learning Path

If you're new to image compression, we recommend reading the documents in this order:

1. **[Introduction to Image Compression](./introduction-to-image-compression.md)** - Foundational concepts
2. **[Introduction to Rust](./introduction-to-rust.md)** - Understand the implementation language
3. **[Huffman Coding](./huffman-coding.md)** - Core entropy coding technique
4. **[LZ77 Compression](./lz77-compression.md)** - Dictionary compression basics
5. **[DEFLATE Algorithm](./deflate.md)** - Combining the above for PNG
6. **[PNG Encoding](./png-encoding.md)** - Complete lossless pipeline
7. **[Discrete Cosine Transform](./dct.md)** - Mathematical foundations for JPEG
8. **[JPEG Quantization](./quantization.md)** - Controlled information loss
9. **[JPEG Encoding](./jpeg-encoding.md)** - Complete lossy pipeline
10. **[Performance Optimization](./performance-optimization.md)** - Making it all fast
11. **[Compression Evolution](./compression-evolution.md)** - History and advanced techniques

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
