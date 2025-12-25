# CLI

The comprs command-line tool for image compression.

## Installation

```bash
# Install from source
cargo install --path . --features cli

# Or build locally
cargo build --release --features cli
```

## Usage

```bash
# Basic usage - compress to JPEG (default)
comprs input.png -o output.jpg

# Compress to PNG with maximum compression
comprs input.jpg -o output.png -c 9

# JPEG with custom quality (1-100)
comprs photo.png -o photo.jpg -q 90

# JPEG with 4:2:0 chroma subsampling (smaller files)
comprs photo.png -o photo.jpg --subsampling s420

# PNG with specific filter strategy
comprs input.jpg -o output.png --filter paeth

# Adaptive fast (reduced trials) or sampled (every Nth row) strategies
comprs input.jpg -o output.png --filter adaptive-fast
comprs input.jpg -o output.png --filter adaptive-sampled --adaptive-sample-interval 8

# Convert to grayscale
comprs color.png -o gray.jpg --grayscale

# Verbose output with timing and size info
comprs input.png -o output.jpg -v
```

## Options

| Option                       | Description                                                                                                                                              | Default                    |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------- |
| `-o, --output`               | Output file path                                                                                                                                         | `<input>.compressed.<ext>` |
| `-f, --format`               | Output format (`png`, `jpeg`, `jpg`)                                                                                                                     | Detected from extension    |
| `-q, --quality`              | JPEG quality (1-100)                                                                                                                                     | 85                         |
| `--jpeg-optimize-huffman`    | Optimize JPEG Huffman tables per image (smaller files, slower)                                                                                           | false                      |
| `--jpeg-restart-interval`    | JPEG restart interval (MCUs, >0 enables DRI)                                                                                                             | 0 (disabled)               |
| `-c, --compression`          | PNG compression level (1-9)                                                                                                                              | 2                          |
| `--subsampling`              | JPEG chroma subsampling (`s444`, `s420`)                                                                                                                 | s444                       |
| `--filter`                   | PNG filter (`none`, `sub`, `up`, `average`, `paeth`, `minsum`, `adaptive`, `adaptivefast`)                                                             | adaptivefast               |
| `--png-preset`               | PNG preset (`fast`, `balanced`, `max`)                                                                                                                   | unset (optional)           |
| `--png-optimize-alpha`       | Zero color channels for fully transparent pixels (PNG)                                                                                                   | false                      |
| `--png-reduce-color`         | Losslessly reduce color type when possible (PNG)                                                                                                         | false                      |
| `--png-strip-metadata`       | Strip ancillary text/time metadata chunks (PNG)                                                                                                          | false                      |
| `--grayscale`                | Convert to grayscale                                                                                                                                     | false                      |
| `-v, --verbose`              | Show detailed output                                                                                                                                     | false                      |
