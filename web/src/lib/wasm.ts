export type EncodeFormat = 'png' | 'jpeg';

/** Preset levels: 0=fast, 1=balanced, 2=max */
export type PresetLevel = 0 | 1 | 2;

export type CompressOptions = {
	format: EncodeFormat;
	quality?: number; // JPEG quality 1-100
	subsampling420?: boolean; // JPEG: use 4:2:0 chroma subsampling
	hasAlpha?: boolean; // If false, strip alpha channel for smaller output
	preset?: PresetLevel; // Preset level: 0=fast, 1=balanced, 2=max
	lossy?: boolean; // PNG: if true, enable quantization for smaller files
};

export type CompressResult = {
	bytes: Uint8Array;
	blob: Blob;
	elapsedMs: number;
};

// Dynamically imported WASM module (client-side only)
let wasmModule: typeof import('$lib/pixo-wasm/pixo.js') | null = null;
let initialized: Promise<void> | null = null;

export function initWasm() {
	if (!initialized) {
		initialized = import('$lib/pixo-wasm/pixo.js').then(async (mod) => {
			await mod.default();
			wasmModule = mod;
		});
	}
	return initialized;
}

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
		const useRgb = options.hasAlpha === false;
		const colorType = useRgb ? 2 : 3;
		const pixelData = useRgb ? rgbaToRgb(imageData.data) : new Uint8Array(imageData.data);

		const preset = options.preset ?? 1; // Default to balanced
		const lossy = options.lossy ?? true; // Default to lossy for smaller files

		// encodePng(data, width, height, colorType, preset, lossy)
		bytes = wasmModule.encodePng(pixelData, imageData.width, imageData.height, colorType, preset, lossy);
		mime = 'image/png';
	} else {
		const quality = clamp(options.quality ?? 85, 1, 100);
		const rgb = rgbaToRgb(imageData.data);

		const preset = options.preset ?? 1; // Default to balanced
		const subsampling420 = options.subsampling420 ?? true;

		// encodeJpeg(data, width, height, colorType, quality, preset, subsampling420)
		bytes = wasmModule.encodeJpeg(rgb, imageData.width, imageData.height, 2, quality, preset, subsampling420);
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
