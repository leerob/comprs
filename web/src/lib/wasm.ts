export type EncodeFormat = 'png' | 'jpeg';
export type PngFilter =
	| 'adaptive'
	| 'adaptive-fast'
	| 'none'
	| 'sub'
	| 'up'
	| 'average'
	| 'paeth';

/** Preset levels: 0=fast, 1=balanced, 2=max */
export type PresetLevel = 0 | 1 | 2;

export type CompressOptions = {
	format: EncodeFormat;
	quality?: number; // JPEG quality 1-100
	compressionLevel?: number; // PNG compression level 1-9
	filter?: PngFilter;
	subsampling420?: boolean;
	hasAlpha?: boolean; // If false, strip alpha channel for smaller output
	optimizeHuffman?: boolean; // JPEG: optimize Huffman tables (smaller, slower)
	preset?: PresetLevel; // Preset level: 0=faster, 1=auto, 2=smallest
	lossless?: boolean; // PNG: if true, disable quantization (lossless). Default: false (lossy enabled)
};

export type CompressResult = {
	bytes: Uint8Array;
	blob: Blob;
	elapsedMs: number;
};

// Dynamically imported WASM module (client-side only)
let wasmModule: typeof import('$lib/comprs-wasm/comprs.js') | null = null;
let initialized: Promise<void> | null = null;

export function initWasm() {
	if (!initialized) {
		initialized = import('$lib/comprs-wasm/comprs.js').then(async (mod) => {
			await mod.default();
			wasmModule = mod;
		});
	}
	return initialized;
}

const filterMap: Record<PngFilter, number> = {
	none: 0,
	sub: 1,
	up: 2,
	average: 3,
	paeth: 4,
	adaptive: 5,
	'adaptive-fast': 6
};

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max);

function rgbaToRgb(data: Uint8ClampedArray) {
	const rgb = new Uint8Array((data.length / 4) * 3);
	for (let i = 0, j = 0; i < data.length; i += 4, j += 3) {
		rgb[j] = data[i];
		rgb[j + 1] = data[i + 1];
		rgb[j + 2] = data[i + 2];
	}
	return rgb;
}

export async function compressImage(imageData: ImageData, options: CompressOptions): Promise<CompressResult> {
	await initWasm();
	if (!wasmModule) throw new Error('WASM module not loaded');

	const t0 = performance.now();
	let bytes: Uint8Array;
	let mime: string;

	if (options.format === 'png') {
		// If the image has no meaningful alpha, encode as RGB (color_type 2) for smaller output
		// This avoids the overhead of encoding an all-255 alpha channel
		const useRgb = options.hasAlpha === false;
		const colorType = useRgb ? 2 : 3;
		const pixelData = useRgb ? rgbaToRgb(imageData.data) : new Uint8Array(imageData.data);

		// Default to lossy (lossless = false) for smaller files
		const lossless = options.lossless ?? false;

		// Use preset-based encoding with lossy support if a preset is specified
		if (options.preset !== undefined) {
			const encodePngPresetLossy = (wasmModule as any).encodePngPresetLossy as
				| undefined
				| ((d: Uint8Array, w: number, h: number, ct: number, preset: number, lossless: boolean) => Uint8Array);
			if (encodePngPresetLossy) {
				bytes = encodePngPresetLossy(pixelData, imageData.width, imageData.height, colorType, options.preset, lossless);
			} else {
				// Fallback to non-lossy preset encoding
				const encodePngPreset = (wasmModule as any).encodePngPreset as
					| undefined
					| ((d: Uint8Array, w: number, h: number, ct: number, preset: number) => Uint8Array);
				if (encodePngPreset) {
					bytes = encodePngPreset(pixelData, imageData.width, imageData.height, colorType, options.preset);
				} else {
					// Fallback to filter-based encoding
					const compressionLevel = clamp(options.compressionLevel ?? 6, 1, 9);
					const filterCode = filterMap[options.filter ?? 'adaptive'];
					bytes = wasmModule.encodePngWithFilter(pixelData, imageData.width, imageData.height, colorType, compressionLevel, filterCode);
				}
			}
		} else {
			const compressionLevel = clamp(options.compressionLevel ?? 6, 1, 9);
			const filterCode = filterMap[options.filter ?? 'adaptive'];
			bytes = wasmModule.encodePngWithFilter(pixelData, imageData.width, imageData.height, colorType, compressionLevel, filterCode);
		}
		mime = 'image/png';
	} else {
		const quality = clamp(options.quality ?? 85, 1, 100);
		const rgb = rgbaToRgb(imageData.data);

		// Use preset-based encoding if a preset is specified
		if (options.preset !== undefined) {
			const encodeJpegPreset = (wasmModule as any).encodeJpegPreset as
				| undefined
				| ((d: Uint8Array, w: number, h: number, q: number, ct: number, preset: number) => Uint8Array);
			if (encodeJpegPreset) {
				bytes = encodeJpegPreset(rgb, imageData.width, imageData.height, quality, 2, options.preset);
			} else {
				// Fallback to auto preset behavior (optimize Huffman enabled)
				const encoderWithOpts = (wasmModule as any).encodeJpegWithOptions as
					| undefined
					| ((d: Uint8Array, w: number, h: number, q: number, ct: number, subs420: boolean, opt: boolean) => Uint8Array);
				if (encoderWithOpts) {
					bytes = encoderWithOpts(rgb, imageData.width, imageData.height, quality, 2, options.subsampling420 ?? true, true);
				} else {
					bytes = wasmModule.encodeJpeg(rgb, imageData.width, imageData.height, quality, 2, options.subsampling420 ?? true);
				}
			}
		} else {
			const optimize = options.optimizeHuffman ?? false;
			const encoderWithOpts = (wasmModule as any).encodeJpegWithOptions as
				| undefined
				| ((d: Uint8Array, w: number, h: number, q: number, ct: number, subs420: boolean, opt: boolean) => Uint8Array);
			if (encoderWithOpts) {
				bytes = encoderWithOpts(rgb, imageData.width, imageData.height, quality, 2, options.subsampling420 ?? true, optimize);
			} else {
				bytes = wasmModule.encodeJpeg(rgb, imageData.width, imageData.height, quality, 2, options.subsampling420 ?? true);
			}
		}
		mime = 'image/jpeg';
	}

	const elapsedMs = performance.now() - t0;
	const blob = new Blob([new Uint8Array(bytes)], { type: mime });
	return { bytes, blob, elapsedMs };
}

export async function getBytesPerPixel(color: number) {
	await initWasm();
	if (!wasmModule) throw new Error('WASM module not loaded');
	return wasmModule.bytesPerPixel(color);
}
