import init, { bytesPerPixel, encodeJpeg, encodePngWithFilter } from '$lib/comprs-wasm/comprs.js';
import type { InitOutput } from '$lib/comprs-wasm/comprs.js';

export type EncodeFormat = 'png' | 'jpeg';
export type PngFilter =
	| 'adaptive'
	| 'adaptive-fast'
	| 'none'
	| 'sub'
	| 'up'
	| 'average'
	| 'paeth';

export type CompressOptions = {
	format: EncodeFormat;
	quality?: number; // JPEG quality 1-100
	compressionLevel?: number; // PNG compression level 1-9
	filter?: PngFilter;
	subsampling420?: boolean;
};

export type CompressResult = {
	bytes: Uint8Array;
	blob: Blob;
	elapsedMs: number;
};

let initialized: Promise<InitOutput> | null = null;

export function initWasm() {
	if (!initialized) {
		initialized = init();
	}
	return initialized;
}

async function ensureWasmLoaded() {
	return initWasm();
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
	await ensureWasmLoaded();

	const format = options.format;
	const t0 = performance.now();
	let bytes: Uint8Array;
	let mime = 'image/png';

	if (format === 'png') {
		const compressionLevel = clamp(options.compressionLevel ?? 6, 1, 9);
		const filterCode = filterMap[options.filter ?? 'adaptive'];
		const rgba = new Uint8Array(imageData.data);
		bytes = encodePngWithFilter(rgba, imageData.width, imageData.height, 3, compressionLevel, filterCode);
		mime = 'image/png';
	} else {
		const quality = clamp(options.quality ?? 85, 1, 100);
		const rgb = rgbaToRgb(imageData.data);
		bytes = encodeJpeg(rgb, imageData.width, imageData.height, quality, 2, options.subsampling420 ?? true);
		mime = 'image/jpeg';
	}

	const elapsedMs = performance.now() - t0;
	// Create a fresh buffer to avoid SharedArrayBuffer typing issues
	const view = new Uint8Array(bytes);
	const blob = new Blob([view], { type: mime });
	return { bytes, blob, elapsedMs };
}

export function getBytesPerPixel(color: number) {
	return bytesPerPixel(color);
}
