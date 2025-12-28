export type EncodeFormat = 'png' | 'jpeg';

/** Preset levels: 0=fast, 1=balanced, 2=max */
export type PresetLevel = 0 | 1 | 2;

/** Resize algorithm: 0=Nearest, 1=Bilinear, 2=Lanczos3 */
export type ResizeAlgorithm = 'nearest' | 'bilinear' | 'lanczos3';

export type ResizeOptions = {
	width: number;
	height: number;
	algorithm?: ResizeAlgorithm;
	maintainAspectRatio?: boolean;
};

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

const algorithmToNumber = (algorithm: ResizeAlgorithm): number => {
	switch (algorithm) {
		case 'nearest':
			return 0;
		case 'bilinear':
			return 1;
		case 'lanczos3':
			return 2;
		default:
			return 2; // Default to Lanczos3 for best quality
	}
};

/**
 * Resize an image to new dimensions.
 *
 * @param imageData - Source ImageData
 * @param options - Resize options including target width/height and algorithm
 * @returns Resized ImageData
 */
export async function resizeImage(imageData: ImageData, options: ResizeOptions): Promise<ImageData> {
	await initWasm();
	if (!wasmModule) throw new Error('WASM module not loaded');

	let targetWidth = options.width;
	let targetHeight = options.height;

	// Maintain aspect ratio if requested
	if (options.maintainAspectRatio) {
		const aspectRatio = imageData.width / imageData.height;
		const targetAspectRatio = targetWidth / targetHeight;

		if (aspectRatio > targetAspectRatio) {
			// Source is wider - constrain by width
			targetHeight = Math.round(targetWidth / aspectRatio);
		} else {
			// Source is taller - constrain by height
			targetWidth = Math.round(targetHeight * aspectRatio);
		}
	}

	// Ensure dimensions are at least 1
	targetWidth = Math.max(1, targetWidth);
	targetHeight = Math.max(1, targetHeight);

	const algorithm = algorithmToNumber(options.algorithm ?? 'lanczos3');
	const colorType = 3; // RGBA

	const resizedData = wasmModule.resizeImage(
		new Uint8Array(imageData.data),
		imageData.width,
		imageData.height,
		targetWidth,
		targetHeight,
		colorType,
		algorithm
	);

	return new ImageData(new Uint8ClampedArray(resizedData), targetWidth, targetHeight);
}

/**
 * Calculate new dimensions maintaining aspect ratio.
 *
 * @param srcWidth - Original width
 * @param srcHeight - Original height
 * @param maxWidth - Maximum target width
 * @param maxHeight - Maximum target height
 * @returns New dimensions that fit within max bounds while maintaining aspect ratio
 */
export function calculateResizeDimensions(
	srcWidth: number,
	srcHeight: number,
	maxWidth: number,
	maxHeight: number
): { width: number; height: number } {
	const aspectRatio = srcWidth / srcHeight;

	let width = maxWidth;
	let height = Math.round(maxWidth / aspectRatio);

	if (height > maxHeight) {
		height = maxHeight;
		width = Math.round(maxHeight * aspectRatio);
	}

	return { width: Math.max(1, width), height: Math.max(1, height) };
}
