# WebAssembly (WASM)

Use pixo in browser and Node.js applications via WebAssembly.

## API

The WASM build exposes a minimal 3-function API:

```typescript
encodePng(data: Uint8Array, width: number, height: number, colorType: number, preset: number, lossy: boolean): Uint8Array
encodeJpeg(data: Uint8Array, width: number, height: number, colorType: number, quality: number, preset: number, subsampling420: boolean): Uint8Array
bytesPerPixel(colorType: number): number
```

## Building

### Prerequisites

- Rust with `wasm32-unknown-unknown` target
- `wasm-bindgen-cli`

### Build Commands

```bash
# Install the WASM target
rustup target add wasm32-unknown-unknown

# Install wasm-bindgen CLI
cargo install wasm-bindgen-cli

# Build the WASM module
cargo build --target wasm32-unknown-unknown --release --no-default-features --features wasm,simd

# Generate JS bindings (output to web/src/lib/pixo-wasm/)
wasm-bindgen --target web \
  --out-dir web/src/lib/pixo-wasm \
  --out-name pixo \
  target/wasm32-unknown-unknown/release/pixo.wasm

# (Optional but recommended) Optimize with binaryen for minimal size
wasm-opt -Oz --strip-debug --strip-dwarf --strip-producers --strip-target-features \
  --enable-bulk-memory --enable-sign-ext --enable-nontrapping-float-to-int \
  -o web/src/lib/pixo-wasm/pixo_bg.wasm \
  web/src/lib/pixo-wasm/pixo_bg.wasm
```

## Troubleshooting

### Homebrew + rustup Conflict

If you have both Homebrew Rust and rustup installed, cargo may use the wrong compiler:

```bash
# Check which rustc is being used
which rustc  # Should be ~/.cargo/bin/rustc for rustup

# If Homebrew's rustc is used, explicitly set the correct one:
RUSTC=~/.cargo/bin/rustc cargo build --target wasm32-unknown-unknown --release --features wasm
```

## Binary Size

The WASM binary is **236 KB** with full PNG and JPEG support. See [benchmarks](../benches/BENCHMARKS.md#4-wasm-binary-size-comparison) for comparisons with other libraries.
With the size-focused build (`--no-default-features --features wasm,simd`) and
binaryen optimization, the binary shrinks to **~142 KB** without sacrificing PNG/JPEG functionality.
