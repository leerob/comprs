/* tslint:disable */
/* eslint-disable */

/**
 * Get the number of bytes per pixel for a color type.
 *
 * Useful for validating input data length.
 *
 * * 0 (Gray) = 1 byte
 * * 1 (GrayAlpha) = 2 bytes
 * * 2 (Rgb) = 3 bytes
 * * 3 (Rgba) = 4 bytes
 */
export function bytesPerPixel(color_type: number): number;

/**
 * Encode raw pixel data as JPEG.
 *
 * # Arguments
 *
 * * `data` - Raw pixel data as Uint8Array (row-major order)
 * * `width` - Image width in pixels
 * * `height` - Image height in pixels
 * * `quality` - Quality level 1-100 (85 recommended)
 * * `color_type` - Color type: 0=Gray, 2=Rgb (JPEG only supports these)
 * * `subsampling_420` - If true, use 4:2:0 chroma subsampling (smaller files)
 *
 * # Returns
 *
 * JPEG file bytes as Uint8Array.
 */
export function encodeJpeg(data: Uint8Array, width: number, height: number, quality: number, color_type: number, subsampling_420: boolean): Uint8Array;

/**
 * Encode JPEG with preset: 0=fast, 1=balanced, 2=max.
 *
 * # Arguments
 *
 * * `data` - Raw pixel data as Uint8Array (row-major order)
 * * `width` - Image width in pixels
 * * `height` - Image height in pixels
 * * `quality` - Quality level 1-100 (85 recommended)
 * * `color_type` - Color type: 0=Gray, 2=Rgb (JPEG only supports these)
 * * `preset` - Preset: 0=fast, 1=balanced, 2=max
 *
 * # Returns
 *
 * JPEG file bytes as Uint8Array.
 */
export function encodeJpegPreset(data: Uint8Array, width: number, height: number, quality: number, color_type: number, preset: number): Uint8Array;

/**
 * Encode JPEG with an explicit optimize_huffman toggle.
 */
export function encodeJpegWithOptions(data: Uint8Array, width: number, height: number, quality: number, color_type: number, subsampling_420: boolean, optimize_huffman: boolean): Uint8Array;

/**
 * Encode raw pixel data as PNG.
 *
 * # Arguments
 *
 * * `data` - Raw pixel data as Uint8Array (row-major order)
 * * `width` - Image width in pixels
 * * `height` - Image height in pixels
 * * `color_type` - Color type: 0=Gray, 1=GrayAlpha, 2=Rgb, 3=Rgba
 * * `compression_level` - Compression level 1-9 (6 recommended)
 *
 * # Returns
 *
 * PNG file bytes as Uint8Array.
 */
export function encodePng(data: Uint8Array, width: number, height: number, color_type: number, compression_level: number): Uint8Array;

/**
 * Encode PNG with preset: 0=fast, 1=balanced, 2=max.
 *
 * # Arguments
 *
 * * `data` - Raw pixel data as Uint8Array (row-major order)
 * * `width` - Image width in pixels
 * * `height` - Image height in pixels
 * * `color_type` - Color type: 0=Gray, 1=GrayAlpha, 2=Rgb, 3=Rgba
 * * `preset` - Preset: 0=fast, 1=balanced, 2=max
 *
 * # Returns
 *
 * PNG file bytes as Uint8Array.
 */
export function encodePngPreset(data: Uint8Array, width: number, height: number, color_type: number, preset: number): Uint8Array;

/**
 * Encode PNG with preset and lossless flag.
 *
 * When `lossless` is false, enables auto-quantization for potentially
 * significant size reduction (lossy compression). This can reduce file
 * sizes by 50-80% for images with limited color palettes.
 *
 * # Arguments
 *
 * * `data` - Raw pixel data as Uint8Array (row-major order)
 * * `width` - Image width in pixels
 * * `height` - Image height in pixels
 * * `color_type` - Color type: 0=Gray, 1=GrayAlpha, 2=Rgb, 3=Rgba
 * * `preset` - Preset: 0=fast, 1=balanced, 2=max
 * * `lossless` - If true, disable quantization (lossless). If false, enable auto-quantization (lossy).
 *
 * # Returns
 *
 * PNG file bytes as Uint8Array.
 */
export function encodePngPresetLossy(data: Uint8Array, width: number, height: number, color_type: number, preset: number, lossless: boolean): Uint8Array;

/**
 * Encode raw pixel data as PNG with a specific filter strategy.
 *
 * # Arguments
 *
 * * `data` - Raw pixel data as Uint8Array (row-major order)
 * * `width` - Image width in pixels
 * * `height` - Image height in pixels
 * * `color_type` - Color type: 0=Gray, 1=GrayAlpha, 2=Rgb, 3=Rgba
 * * `compression_level` - Compression level 1-9 (6 recommended)
 * * `filter` - Filter strategy: 0=None, 1=Sub, 2=Up, 3=Average, 4=Paeth, 5=Adaptive, 6=AdaptiveFast, 7=MinSum
 *
 * # Returns
 *
 * PNG file bytes as Uint8Array.
 */
export function encodePngWithFilter(data: Uint8Array, width: number, height: number, color_type: number, compression_level: number, filter: number): Uint8Array;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly encodePng: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
  readonly encodePngWithFilter: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
  readonly encodeJpeg: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
  readonly encodeJpegWithOptions: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => void;
  readonly bytesPerPixel: (a: number, b: number) => void;
  readonly encodePngPreset: (a: number, b: number, c: number, d: number, e: number, f: number, g: number) => void;
  readonly encodePngPresetLossy: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
  readonly encodeJpegPreset: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
  readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
  readonly __wbindgen_export: (a: number, b: number) => number;
  readonly __wbindgen_export2: (a: number, b: number, c: number) => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
* Instantiates the given `module`, which can either be bytes or
* a precompiled `WebAssembly.Module`.
*
* @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
*
* @returns {InitOutput}
*/
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
* If `module_or_path` is {RequestInfo} or {URL}, makes a request and
* for everything else, calls `WebAssembly.instantiate` directly.
*
* @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
*
* @returns {Promise<InitOutput>}
*/
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
