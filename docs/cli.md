# CLI

The pixo command-line tool for image compression.

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
pixo input.png -o output.jpg

# Compress to PNG with maximum compression
pixo input.jpg -o output.png -c 9

# JPEG with custom quality (1-100)
pixo photo.png -o photo.jpg -q 90

# JPEG with 4:2:0 chroma subsampling (smaller files)
pixo photo.png -o photo.jpg --subsampling s420

# PNG with specific filter strategy
pixo input.jpg -o output.png --filter paeth

# Adaptive fast (reduced trials) filter strategy
pixo input.jpg -o output.png --filter adaptive-fast

# Convert to grayscale
pixo color.png -o gray.jpg --grayscale

# Verbose output with timing and size info
pixo input.png -o output.jpg -v

# Preview operation without writing (dry run)
pixo input.png -o output.jpg -n

# Quiet mode (suppress output)
pixo input.png -o output.jpg --quiet

# JSON output for scripting
pixo input.png -o output.jpg --json

# Read from stdin, write to file
cat photo.png | pixo - -o output.jpg

# Read from file, write to stdout
pixo input.png -o - -f jpeg > output.jpg

# Pipe through compression
curl https://example.com/image.png | pixo - -o - -f jpeg | upload-service
```

## Options

| Option                       | Description                                                      | Default                    |
| ---------------------------- | ---------------------------------------------------------------- | -------------------------- |
| `-o, --output`               | Output file path (use `-` for stdout)                            | `<input>.compressed.<ext>` |
| `-f, --format`               | Output format (`png`, `jpeg`, `jpg`)                             | Detected from extension    |
| `-q, --quality`              | JPEG quality (1-100)                                             | 85                         |
| `--jpeg-optimize-huffman`    | Optimize JPEG Huffman tables per image (smaller files, slower)   | false                      |
| `--jpeg-restart-interval`    | JPEG restart interval (MCUs, >0 enables DRI)                     | 0 (disabled)               |
| `-c, --compression`          | PNG compression level (1-9)                                      | 2                          |
| `--subsampling`              | JPEG chroma subsampling (`s444`, `s420`)                         | s444                       |
| `--filter`                   | PNG filter (`none`, `sub`, `up`, `average`, `paeth`, `minsum`, `adaptive`, `adaptive-fast`) | adaptive-fast |
| `--png-preset`               | PNG preset (`fast`, `balanced`, `max`)                           | unset (optional)           |
| `--png-optimize-alpha`       | Zero color channels for fully transparent pixels (PNG)           | false                      |
| `--png-reduce-color`         | Losslessly reduce color type when possible (PNG)                 | false                      |
| `--png-strip-metadata`       | Strip ancillary text/time metadata chunks (PNG)                  | false                      |
| `--grayscale`                | Convert to grayscale                                             | false                      |
| `-v, --verbose`              | Show detailed output                                             | false                      |
| `--quiet`                    | Suppress all output except errors                                | false                      |
| `--json`                     | Output results as JSON (for scripting)                           | false                      |
| `-n, --dry-run`              | Preview operation without writing files                          | false                      |

## Stdin/Stdout Support

Use `-` as the input path to read from stdin, or `-` as the output path to write to stdout:

```bash
# Read from stdin (output path required)
cat image.png | pixo - -o output.jpg

# Write to stdout
pixo input.png -o - -f jpeg > output.jpg

# Both stdin and stdout
cat image.png | pixo - -o - -f jpeg > output.jpg
```

When writing to stdout, use `-f`/`--format` to specify the output format since it can't be detected from the filename.

## JSON Output

Use `--json` for machine-readable output, useful for scripting:

```bash
$ pixo input.png -o output.jpg --json
{"input":"input.png","output":"output.jpg","input_size":102400,"output_size":51200,"ratio":50.0}
```

With `--dry-run`:

```bash
$ pixo input.png -o output.jpg --json -n
{"dry_run":true,"input":"input.png","output":"output.jpg","input_size":102400,"output_size":51200,"ratio":50.0}
```
