# Fixture Reference & Licensing

This directory aggregates reference assets used for compression benchmarks and regression testing. Please keep it lean and document sources and licenses for every added file.

## Directory Structure

```
tests/fixtures/
├── *.png, *.jpg          # Local fixtures (checked in)
├── reference/            # Reference images from external projects
├── kodak/                # Kodak Lossless True Color Suite (downloaded on-demand)
├── pngsuite/             # PNGSuite test images (downloaded on-demand)
└── jpeg_corpus/          # JPEG test corpus (downloaded on-demand)
```

## Local Fixtures (Checked In)

These files are committed to the repository:

- `avatar-color.png` — Small avatar image for quick tests
- `playground.png` — UI screenshot
- `rocket.png` — Photographic image for compression regression tests
- `browser.jpg` — Browser screenshot
- `multi-agent.jpg` — Multi-agent illustration
- `review.jpg` — Review interface screenshot
- `web.jpg` — Web-related image

## Reference Images

### Squoosh Project

- `reference/squoosh_example.png` — from Squoosh project (`codecs/example.png`)
- `reference/squoosh_example_palette.png` — from Squoosh project (`codecs/example_palette.png`)
- **Source**: https://github.com/GoogleChromeLabs/squoosh
- **License**: Apache 2.0

## Downloaded Test Suites

These image sets are downloaded on-demand during test execution and cached locally. They are **not** committed to the repository (see `.gitignore`).

### Kodak Lossless True Color Image Suite

- **Location**: `tests/fixtures/kodak/`
- **Count**: 24 images
- **Source**: https://r0k.us/graphics/kodak/
- **License**: Public domain / unrestricted use
- **Characteristics**:
  - 768×512 or 512×768 dimensions
  - 24-bit RGB, uncompressed PNG
  - Diverse photographic content: landscapes, portraits, buildings, textures
- **Usage**: Industry-standard benchmark for image compression algorithms

### PNGSuite (Willem van Schaik)

- **Location**: `tests/fixtures/pngsuite/`
- **Count**: ~150 images across categories
- **Source**: http://www.schaik.com/pngsuite/ and https://github.com/lunapaint/pngsuite
- **License**: Public domain
- **Categories**:
  - `basn*` — Basic formats, non-interlaced
  - `basi*` — Basic formats, interlaced (Adam7)
  - `bg*` — Background color tests
  - `c*` — Chunk tests (chromaticity, physical dimensions)
  - `f*` — Filter type tests
  - `g*` — Gamma tests
  - `oi*` — Interlace method tests
  - `s*` — Size tests (1×1 to 40×40)
  - `t*` — Transparency tests
  - `z*` — Zlib compression level tests
  - `x*` — Corrupted files (for error handling tests)
- **Usage**: Comprehensive PNG conformance testing

### JPEG Test Corpus

- **Location**: `tests/fixtures/jpeg_corpus/`
- **Count**: ~6 images
- **Source**: libjpeg-turbo test images
  - https://github.com/libjpeg-turbo/libjpeg-turbo/tree/main/testimages
- **License**: BSD-3-Clause (libjpeg-turbo)
- **Categories**:
  - Baseline JPEG
  - Progressive JPEG
  - Arithmetic coding (edge case)
  - 12-bit JPEG (edge case)
  - Special color handling
- **Usage**: JPEG encoding/decoding conformance testing

## Synthetic Test Images

Generated programmatically in `tests/support/synthetic.rs`. Not stored as files.

- **Solid colors**: Black, white, red, green, blue
- **Gradients**: Horizontal, vertical, diagonal, RGB, radial
- **Patterns**: Checkerboard (various sizes), stripes, high-frequency
- **Edge cases**: 1×1, odd dimensions, extreme aspect ratios
- **Noise**: Deterministic pseudo-random patterns

## Adding New Fixtures

When adding new test images:

1. **Check licensing**: Ensure the image has a permissive license (public domain, CC0, MIT, BSD, Apache 2.0, etc.)
2. **Document source**: Add an entry to this file with:
   - Filename and location
   - Source URL
   - License
   - Brief description
3. **Consider size**: Prefer smaller images for faster tests
4. **Use on-demand download**: For large test suites, use the pattern in `tests/support/*.rs` to download on first use with SHA256 verification
5. **Update `.gitignore`**: Add downloaded directories to `.gitignore`

## License Summary

| Source | License | Commercial Use |
|--------|---------|----------------|
| Kodak Suite | Public domain | Yes |
| PNGSuite | Public domain | Yes |
| libjpeg-turbo | BSD-3-Clause | Yes |
| Squoosh examples | Apache 2.0 | Yes |
