# pixo web optimizer (SvelteKit + WASM)

Client-side PNG/JPEG compression powered by the `pixo` Rust library compiled to WebAssembly. Select images from disk, tune PNG filters or JPEG quality, and compare originals with a draggable before/after slider—all without leaving the browser.

## Stack

- SvelteKit + TypeScript
- Tailwind CSS
- WebAssembly build of `pixo` (Rust)

## Getting started

```sh
# install deps
npm install

# build the WASM bundle from the Rust crate into src/lib/pixo-wasm
npm run wasm:build

# start the dev server
npm run dev
```

Then open the printed local URL (default `http://localhost:5173` or the next free port). Drop PNG/JPEG files or use the file picker, adjust options, compress, and use the slider to compare.

## Scripts

- `npm run wasm:build` — builds the Rust crate for `wasm32-unknown-unknown` and runs `wasm-bindgen` (uses a cached binary from `wasm-pack` if available).
- `npm run dev` — start the SvelteKit dev server.
- `npm run check` — SvelteKit type/syntax checks.
- `npm run build` — production build.
- `npm run e2e` — headless Playwright smoke test (requires a dev server running on `http://localhost:4173` or set `BASE_URL`). Uploads `tests/fixtures/playground.png`, compresses, and asserts a download is available.

## Shortcuts & accessibility

- Drop zone is keyboard-activatable (Enter/Space) and also responds to Ctrl/Cmd+O to open the file picker (when not typing in an input).
- Inline notices surface unsupported files instead of alerts.
- “Clear completed” removes finished jobs and revokes object URLs to free memory.

## Notes

- Only PNG and JPEG inputs are supported.
- JPEG encodes drop alpha (RGBA → RGB) with optional 4:2:0 subsampling.
- PNG encodes honor the chosen filter strategy and compression level.
- Resizing images is supported.
- All processing happens locally in your browser.
