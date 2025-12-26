# Fuzzing

This directory contains fuzz targets for testing pixo with [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz).

## Prerequisites

Install cargo-fuzz (requires nightly Rust):

```bash
cargo install cargo-fuzz
```

## Running Fuzz Targets

### PNG Encoding

```bash
cd tests/fuzz
cargo +nightly fuzz run png_encode
```

### JPEG Encoding

```bash
cd tests/fuzz
cargo +nightly fuzz run jpeg_encode
```

### DEFLATE Compression

```bash
cd tests/fuzz
cargo +nightly fuzz run deflate
```

### LZ77 Compression

```bash
cd tests/fuzz
cargo +nightly fuzz run lz77
```

## Options

### Run with timeout

```bash
cargo +nightly fuzz run png_encode -- -max_total_time=300
```

### Run with specific number of jobs

```bash
cargo +nightly fuzz run png_encode -- -jobs=4
```

### Run with memory limit (in MB)

```bash
cargo +nightly fuzz run png_encode -- -rss_limit_mb=2048
```

## Corpus

Crash inputs and interesting test cases are saved in:

- `tests/fuzz/corpus/<target>/` - Interesting inputs discovered during fuzzing
- `tests/fuzz/artifacts/<target>/` - Crash-inducing inputs

## Reproducing Crashes

To reproduce a crash:

```bash
cargo +nightly fuzz run png_encode tests/fuzz/artifacts/png_encode/crash-<hash>
```

## Coverage

To generate coverage reports:

```bash
cargo +nightly fuzz coverage png_encode
```
