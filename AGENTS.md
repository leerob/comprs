pixo is a **pure Rust, zero-dependency** image compression library implementing PNG and JPEG encoders from scratch. The core philosophy is portability and minimal binary size.

## Commands

```bash
cargo build --release               # Build
cargo test                          # Run all tests
cargo fmt && cargo clippy           # Format and lint (run before committing)
cargo bench --bench comparison      # Compare against alternatives
```

### Building WASM

```bash
node web/scripts/build-wasm.mjs    # Build + optimize WASM
```

This script handles rustup/Homebrew detection, wasm-bindgen, and wasm-opt automatically.

## Architecture

- **`src/png/`** - PNG encoder (filters, palette quantization, bit depth reduction)
- **`src/jpeg/`** - JPEG encoder (baseline/progressive DCT, trellis quantization)
- **`src/compress/`** - DEFLATE implementation (LZ77, Huffman, CRC32/Adler32)
- **`src/simd/`** - SIMD acceleration (x86_64 AVX2/SSE, aarch64 NEON)
- **`src/wasm.rs`** - WebAssembly bindings via wasm-bindgen
- **`web/`** - SvelteKit demo app with WASM integration

### Feature Flags

`simd` (default), `parallel` (default), `cli`, `wasm`

### Presets

`fast()`, `balanced()`, `max()` for both PNG and JPEG encoders.

## Key Patterns

- Use `encode_into` variants to reuse output buffers
- `PngOptions::builder()` / `JpegOptions::builder()` for configuration
- Unsafe code only compiles with `simd` or `wasm` features

## Workflow Rules

- **Changing benchmarks?** Update `benches/BENCHMARKS.md` with new results
- **Adding tests or modifying test coverage?** Update `docs/codebase-size-comparison.md`
