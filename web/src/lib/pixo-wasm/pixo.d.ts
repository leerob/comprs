/* tslint:disable */
/* eslint-disable */

/**
 * Get the number of bytes per pixel for a color type.
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
 * * `data` - Raw pixel data as Uint8Array (row-major order, RGB only)
 * * `width` - Image width in pixels
 * * `height` - Image height in pixels
 * * `color_type` - Color type: 0=Gray, 2=Rgb (JPEG only supports these)
 * * `quality` - Quality level 1-100 (85 recommended)
 * * `preset` - Optimization preset: 0=fast, 1=balanced, 2=max
 * * `subsampling_420` - If true, use 4:2:0 chroma subsampling (smaller files)
 *
 * # Returns
 *
 * JPEG file bytes as Uint8Array.
 */
export function encodeJpeg(data: Uint8Array, width: number, height: number, color_type: number, quality: number, preset: number, subsampling_420: boolean): Uint8Array;

/**
 * Encode raw pixel data as PNG.
 *
 * # Arguments
 *
 * * `data` - Raw pixel data as Uint8Array (row-major order)
 * * `width` - Image width in pixels
 * * `height` - Image height in pixels
 * * `color_type` - Color type: 0=Gray, 1=GrayAlpha, 2=Rgb, 3=Rgba
 * * `preset` - Optimization preset: 0=fast, 1=balanced, 2=max
 * * `lossy` - If true, enable quantization for smaller files (reduces colors to 256)
 *
 * # Returns
 *
 * PNG file bytes as Uint8Array.
 */
export function encodePng(data: Uint8Array, width: number, height: number, color_type: number, preset: number, lossy: boolean): Uint8Array;

/**
 * Resize an image to new dimensions.
 *
 * # Arguments
 *
 * * `data` - Raw pixel data as Uint8Array (row-major order)
 * * `src_width` - Source image width in pixels
 * * `src_height` - Source image height in pixels
 * * `dst_width` - Destination image width in pixels
 * * `dst_height` - Destination image height in pixels
 * * `color_type` - Color type: 0=Gray, 1=GrayAlpha, 2=Rgb, 3=Rgba
 * * `algorithm` - Resize algorithm: 0=Nearest, 1=Bilinear, 2=Lanczos3
 *
 * # Returns
 *
 * Resized pixel data as Uint8Array with the same color type.
 */
export function resizeImage(data: Uint8Array, src_width: number, src_height: number, dst_width: number, dst_height: number, color_type: number, algorithm: number): Uint8Array;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
  readonly memory: WebAssembly.Memory;
  readonly bytesPerPixel: (a: number, b: number) => void;
  readonly encodeJpeg: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => void;
  readonly encodePng: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => void;
  readonly resizeImage: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number) => void;
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
