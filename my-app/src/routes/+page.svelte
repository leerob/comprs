<script lang="ts">
	import { compressImage, initWasm, type PngFilter } from '$lib/wasm';
	import { onDestroy, onMount } from 'svelte';

	type JobStatus = 'idle' | 'compressing' | 'done' | 'error';

	type Job = {
		id: string;
		name: string;
		type: string;
		size: number;
		width: number;
		height: number;
		hasAlpha: boolean;
		originalUrl: string;
		imageData: ImageData;
		status: JobStatus;
		slider: number;
		error?: string;
		options: {
			format: 'png' | 'jpeg';
			quality: number;
			compressionLevel: number;
			filter: PngFilter;
			subsampling420: boolean;
		};
		result?: {
			blob: Blob;
			url: string;
			size: number;
			savings: number;
			elapsedMs: number;
		};
	};

	const acceptMime = ['image/png', 'image/jpeg'];
	let jobs: Job[] = [];
	let dropActive = false;
	let busy = false;
	let notices: { id: string; message: string; tone: 'info' | 'warning' | 'error' }[] = [];
	const fileInputId = 'file-input';
	let wasmReady = false;

	const filterOptions: { label: string; value: PngFilter; hint?: string }[] = [
		{ label: 'Adaptive', value: 'adaptive', hint: 'Best compression' },
		{ label: 'Adaptive (fast)', value: 'adaptive-fast', hint: 'Balanced speed' },
		{ label: 'None', value: 'none' },
		{ label: 'Sub', value: 'sub' },
		{ label: 'Up', value: 'up' },
		{ label: 'Average', value: 'average' },
		{ label: 'Paeth', value: 'paeth' }
	];

	const defaultOptions = {
		format: 'jpeg' as const,
		quality: 85,
		compressionLevel: 6,
		filter: 'adaptive' as PngFilter,
		subsampling420: true
	};

	const jobDefaults = {
		png: {
			format: 'png' as const,
			quality: 85,
			compressionLevel: 6,
			filter: 'adaptive' as PngFilter,
			subsampling420: true
		},
		jpeg: {
			format: 'jpeg' as const,
			quality: 85,
			compressionLevel: 6,
			filter: 'adaptive' as PngFilter,
			subsampling420: true
		}
	};

	function deriveOptionsFromType(mime: string) {
		if (mime === 'image/png') return { ...jobDefaults.png };
		if (mime === 'image/jpeg' || mime === 'image/jpg') return { ...jobDefaults.jpeg };
		return { ...defaultOptions };
	}

	$: completedJobs = jobs.filter((j) => j.result);
	$: totalOriginal = completedJobs.reduce((sum, j) => sum + j.size, 0);
	$: totalCompressed = completedJobs.reduce((sum, j) => sum + (j.result?.size ?? 0), 0);
	$: totalSavingsPct =
		totalOriginal > 0 ? ((totalOriginal - totalCompressed) / totalOriginal) * 100 : 0;

	const canvas = typeof document !== 'undefined' ? document.createElement('canvas') : null;
	const ctx = canvas ? canvas.getContext('2d', { willReadFrequently: true }) : null;

	function formatBytes(bytes: number) {
		if (!bytes) return '0 B';
		const units = ['B', 'KB', 'MB', 'GB'];
		const exponent = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
		const value = bytes / 1024 ** exponent;
		return `${value.toFixed(value >= 10 || value % 1 === 0 ? 0 : 1)} ${units[exponent]}`;
	}

	function formatSavings(delta: number) {
		const sign = delta >= 0 ? '-' : '+';
		return `${sign}${Math.abs(delta).toFixed(1)}%`;
	}

function detectAlpha(data: Uint8ClampedArray) {
	for (let i = 3; i < data.length; i += 4) {
		if (data[i] !== 255) return true;
	}
	return false;
}

	function isSupported(file: File) {
		return acceptMime.includes(file.type);
	}

	function addNotice(message: string, tone: 'info' | 'warning' | 'error' = 'warning') {
		const id = crypto.randomUUID();
		notices = [...notices, { id, message, tone }];
		setTimeout(() => {
			notices = notices.filter((n) => n.id !== id);
		}, 4000);
	}

	function dismissNotice(id: string) {
		notices = notices.filter((n) => n.id !== id);
	}

	function triggerFilePicker() {
		const input = document.getElementById(fileInputId) as HTMLInputElement | null;
		if (input) input.click();
	}

	onMount(() => {
		initWasm()
			.then(() => {
				wasmReady = true;
			})
			.catch((err) => {
				addNotice(err instanceof Error ? err.message : 'Failed to load WASM module', 'error');
			});

		const handler = (e: KeyboardEvent) => {
			const target = e.target as HTMLElement | null;
			const isTypingContext =
				target &&
				(target.tagName === 'INPUT' ||
					target.tagName === 'TEXTAREA' ||
					target.isContentEditable);
			if (isTypingContext) return;

			if (e.key.toLowerCase() === 'o' && (e.metaKey || e.ctrlKey)) {
				e.preventDefault();
				triggerFilePicker();
			}
		};

		window.addEventListener('keydown', handler);
		return () => window.removeEventListener('keydown', handler);
	});

	function resetInput(event: Event) {
		const target = event.target as HTMLInputElement;
		target.value = '';
	}

	async function decodeFile(file: File) {
		if (!canvas || !ctx) throw new Error('Canvas not available in this environment.');
		const bitmap = await createImageBitmap(file);
		canvas.width = bitmap.width;
		canvas.height = bitmap.height;
		ctx.clearRect(0, 0, bitmap.width, bitmap.height);
		ctx.drawImage(bitmap, 0, 0);
		const imageData = ctx.getImageData(0, 0, bitmap.width, bitmap.height);
		bitmap.close();
		return {
			imageData,
			width: bitmap.width,
			height: bitmap.height,
			hasAlpha: detectAlpha(imageData.data)
		};
	}

	async function addFiles(fileList: FileList | File[]) {
		const files = Array.from(fileList);
		const supported = files.filter((f) => isSupported(f));
		const rejected = files.filter((f) => !isSupported(f));

		if (rejected.length) {
			addNotice(`Unsupported files skipped: ${rejected.map((f) => f.name).join(', ')}`, 'warning');
		}
		for (const file of supported) {
			const { imageData, width, height, hasAlpha } = await decodeFile(file);
			const url = URL.createObjectURL(file);
			const id = crypto.randomUUID();
			const initialOptions = deriveOptionsFromType(file.type);
			jobs = [
				...jobs,
				{
					id,
					name: file.name,
					type: file.type,
					size: file.size,
					width,
					height,
					hasAlpha,
					originalUrl: url,
					imageData,
					status: 'idle',
					slider: 50,
					options: { ...initialOptions }
				}
			];
		}
	}

	function removeJob(id: string) {
		const job = jobs.find((j) => j.id === id);
		if (job) {
			URL.revokeObjectURL(job.originalUrl);
			if (job.result?.url) URL.revokeObjectURL(job.result.url);
		}
		jobs = jobs.filter((j) => j.id !== id);
	}

	function revokeAll() {
		for (const job of jobs) {
			URL.revokeObjectURL(job.originalUrl);
			if (job.result?.url) URL.revokeObjectURL(job.result.url);
		}
		jobs = [];
	}

	function clearCompleted() {
		const keep = [];
		for (const job of jobs) {
			if (job.result) {
				URL.revokeObjectURL(job.originalUrl);
				if (job.result.url) URL.revokeObjectURL(job.result.url);
			} else {
				keep.push(job);
			}
		}
		jobs = keep;
	}

	onDestroy(() => {
		revokeAll();
	});

	function updateJob(id: string, updater: (job: Job) => Job) {
		jobs = jobs.map((job) => (job.id === id ? updater(job) : job));
	}

	async function compressJob(id: string) {
		const job = jobs.find((j) => j.id === id);
		if (!job) return;
		if (!wasmReady) {
			try {
				await initWasm();
				wasmReady = true;
			} catch (err) {
				addNotice(err instanceof Error ? err.message : 'Failed to load WASM module', 'error');
				return;
			}
		}
		busy = true;
		updateJob(id, (j) => ({ ...j, status: 'compressing', error: undefined }));
		try {
			const { blob, elapsedMs } = await compressImage(job.imageData, job.options);
			const url = URL.createObjectURL(blob);
			if (job.result?.url) URL.revokeObjectURL(job.result.url);
			const savings = job.size > 0 ? ((job.size - blob.size) / job.size) * 100 : 0;
			updateJob(id, (j) => ({
				...j,
				status: 'done',
				result: {
					blob,
					url,
					size: blob.size,
					savings,
					elapsedMs
				}
			}));
		} catch (err) {
			updateJob(id, (j) => ({
				...j,
				status: 'error',
				error: err instanceof Error ? err.message : 'Compression failed'
			}));
		} finally {
			busy = false;
		}
	}

	function onDrop(event: DragEvent) {
		event.preventDefault();
		dropActive = false;
		if (!event.dataTransfer?.files?.length) return;
		addFiles(event.dataTransfer.files);
	}

	function onDragOver(event: DragEvent) {
		event.preventDefault();
		dropActive = true;
	}

	function onDragLeave() {
		dropActive = false;
	}
</script>

<svelte:head>
	<title>comprs • Web WASM image optimizer</title>
	<meta
		name="description"
		content="Client-side PNG/JPEG compression powered by the comprs Rust WASM encoder."
	/>
</svelte:head>

<div class="mx-auto flex max-w-6xl flex-col gap-8 px-4 py-10">
	{#if notices.length}
		<div class="space-y-2">
			{#each notices as notice (notice.id)}
				<div
					class={`flex items-start justify-between gap-3 rounded-xl border px-4 py-3 text-sm ${
						notice.tone === 'error'
							? 'border-rose-700/60 bg-rose-900/40 text-rose-50'
							: notice.tone === 'warning'
								? 'border-amber-700/60 bg-amber-900/40 text-amber-50'
								: 'border-sky-700/60 bg-sky-900/40 text-sky-50'
					}`}
				>
					<span>{notice.message}</span>
					<button
						class="text-xs text-slate-200 hover:text-white"
						on:click={() => dismissNotice(notice.id)}
						aria-label="Dismiss notification"
					>
						✕
					</button>
				</div>
			{/each}
		</div>
	{/if}

	<header class="space-y-3">
		<p class="text-xs font-semibold uppercase tracking-[0.2em] text-sky-300/80">WASM powered</p>
		<h1 class="text-3xl font-bold text-slate-50 sm:text-4xl">
			Optimize PNG & JPEG in your browser with Rust WASM
		</h1>
		<p class="max-w-3xl text-slate-300">
			Drop images, tune compression (PNG filters or JPEG quality), and compare the before/after with
			an interactive slider. All processing stays local in your browser using the comprs WebAssembly
			encoder.
		</p>
	</header>

	<section class="card p-6">
		<div class="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
			<div>
				<h2 class="text-xl font-semibold text-slate-50">Select your images</h2>
				<p class="text-sm text-slate-400">PNG and JPEG only. Files never leave your device.</p>
			</div>
			<div class="flex gap-2">
				<div
					class={`flex items-center gap-2 rounded-lg border px-3 py-2 text-xs ${
						wasmReady
							? 'border-emerald-700/50 bg-emerald-900/40 text-emerald-100'
							: 'border-amber-700/50 bg-amber-900/40 text-amber-100'
					}`}
				>
					<span
						class="h-2 w-2 rounded-full"
						class:animate-pulse={!wasmReady}
						style={`background:${wasmReady ? '#34d399' : '#fbbf24'}`}
					></span>
					<span>{wasmReady ? 'WASM ready' : 'Loading WASM'}</span>
				</div>
				<button class="btn-ghost" type="button" on:click={revokeAll} disabled={!jobs.length}>
					Clear all
				</button>
				<label
					class="btn-primary cursor-pointer"
					aria-label="Select PNG or JPEG files"
					for={fileInputId}
				>
					Upload files
				</label>
				<input
					id={fileInputId}
					type="file"
					accept="image/png,image/jpeg"
					multiple
					class="hidden"
					on:change={(e) => {
						const fileList = (e.target as HTMLInputElement).files;
						if (fileList) addFiles(fileList);
						resetInput(e);
					}}
				/>
			</div>
		</div>

		<button
			type="button"
			class={`mt-4 grid w-full gap-4 rounded-2xl border-2 border-dashed p-6 text-left transition-colors ${
				dropActive ? 'border-sky-400/80 bg-sky-500/5' : 'border-slate-800 bg-slate-900/50'
			}`}
			aria-label="Image drop zone"
			on:drop|preventDefault={onDrop}
			on:dragover={onDragOver}
			on:dragleave={onDragLeave}
			on:keydown={(e) => {
				if (e.key === 'Enter' || e.key === ' ') {
					e.preventDefault();
					triggerFilePicker();
				}
			}}
			on:click={() => triggerFilePicker()}
		>
			<div class="flex flex-col items-center justify-center gap-2 text-center text-slate-300">
				<p class="text-lg font-semibold text-slate-100">Drag & drop images here</p>
				<p class="text-sm text-slate-400">Only PNG and JPEG are supported.</p>
				<div class="mt-2 flex flex-wrap justify-center gap-3 text-xs text-slate-400">
					<span class="rounded-full border border-slate-700 px-3 py-1">Client-side only</span>
					<span class="rounded-full border border-slate-700 px-3 py-1">WASM (comprs)</span>
					<span class="rounded-full border border-slate-700 px-3 py-1">Preview slider</span>
				</div>
			</div>
		</button>
	</section>

	{#if completedJobs.length > 0}
		<section class="card flex flex-wrap items-center justify-between gap-4 p-5 text-sm text-slate-200">
			<div class="space-y-1">
				<p class="text-xs uppercase tracking-[0.15em] text-slate-500">Summary</p>
				<p class="text-lg font-semibold text-slate-50">
					{completedJobs.length} optimized · {formatSavings(totalSavingsPct)} overall
				</p>
			</div>
			<div class="grid grid-cols-2 gap-3 sm:grid-cols-3">
				<div class="rounded-xl border border-slate-800 bg-slate-900/60 px-3 py-2">
					<p class="text-xs text-slate-500">Original size</p>
					<p class="text-base font-semibold text-slate-50">{formatBytes(totalOriginal)}</p>
				</div>
				<div class="rounded-xl border border-slate-800 bg-slate-900/60 px-3 py-2">
					<p class="text-xs text-slate-500">Compressed</p>
					<p class="text-base font-semibold text-slate-50">{formatBytes(totalCompressed)}</p>
				</div>
				<div class="rounded-xl border border-slate-800 bg-slate-900/60 px-3 py-2">
					<p class="text-xs text-slate-500">Savings</p>
					<p class="text-base font-semibold text-emerald-300">{formatSavings(totalSavingsPct)}</p>
				</div>
			</div>
			<button
				class="btn-ghost text-xs"
				on:click={clearCompleted}
				aria-label="Clear completed jobs"
				disabled={!completedJobs.length}
			>
				Clear completed
			</button>
		</section>
	{/if}

	{#if jobs.length === 0}
		<div class="card p-6 text-slate-300">
			<p class="text-lg font-semibold text-slate-100">No images yet</p>
			<p class="text-sm text-slate-400">
				Add some PNG or JPEG files to start optimizing them right in the browser.
			</p>
		</div>
	{/if}

	<div class="flex flex-col gap-6">
		{#each jobs as job (job.id)}
			<article class="card p-5 space-y-4">
				<div class="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
					<div>
						<p class="text-lg font-semibold text-slate-50">{job.name}</p>
						<p class="text-sm text-slate-400">
							{job.width} × {job.height} · {formatBytes(job.size)} · {job.type}
						</p>
					</div>
					<button class="btn-ghost text-red-200" on:click={() => removeJob(job.id)}>Remove</button>
				</div>

				<div class="grid gap-4 md:grid-cols-4">
					<label class="space-y-2 text-sm text-slate-200">
						<span class="flex items-center justify-between">
							Output format
							<span class="text-xs text-slate-500">PNG / JPEG</span>
						</span>
						<select
							class="w-full rounded-lg border border-slate-700 bg-slate-900/70 px-3 py-2 text-slate-50"
							value={job.options.format}
							on:change={(e) => {
								const value = (e.target as HTMLSelectElement).value as 'png' | 'jpeg';
								updateJob(job.id, (j) => ({
									...j,
									options: { ...j.options, format: value }
								}));
							}}
						>
							<option value="jpeg">JPEG (lossy)</option>
							<option value="png">PNG (lossless)</option>
						</select>
					</label>

					{#if job.options.format === 'jpeg'}
						<label class="space-y-2 text-sm text-slate-200">
							<span class="flex items-center justify-between">
								JPEG quality
								<span class="text-xs text-slate-500">{job.options.quality}</span>
							</span>
							<input
								type="range"
								min="1"
								max="100"
								step="1"
								class="w-full accent-sky-400"
								value={job.options.quality}
								on:input={(e) => {
									const quality = Number((e.target as HTMLInputElement).value);
									updateJob(job.id, (j) => ({
										...j,
										options: { ...j.options, quality }
									}));
								}}
							/>
							<p class="text-xs text-slate-500">85 is a good balance for web.</p>
							{#if job.hasAlpha}
								<p class="text-xs font-semibold text-amber-300">
									Transparency will be flattened when exporting to JPEG.
								</p>
							{/if}
						</label>
					{:else}
						<label class="space-y-2 text-sm text-slate-200">
							<span class="flex items-center justify-between">
								PNG compression
								<span class="text-xs text-slate-500">lvl {job.options.compressionLevel}</span>
							</span>
							<input
								type="range"
								min="1"
								max="9"
								step="1"
								class="w-full accent-sky-400"
								value={job.options.compressionLevel}
								on:input={(e) => {
									const compressionLevel = Number((e.target as HTMLInputElement).value);
									updateJob(job.id, (j) => ({
										...j,
										options: { ...j.options, compressionLevel }
									}));
								}}
							/>
							<p class="text-xs text-slate-500">Higher = smaller file, slower encode.</p>
						</label>
					{/if}

					<label class="space-y-2 text-sm text-slate-200">
						<span class="flex items-center justify-between">
							PNG filter
							<span class="text-xs text-slate-500">{job.options.filter}</span>
						</span>
						<select
							class="w-full rounded-lg border border-slate-700 bg-slate-900/70 px-3 py-2 text-slate-50 disabled:opacity-50"
							disabled={job.options.format !== 'png'}
							value={job.options.filter}
							on:change={(e) => {
								const filter = (e.target as HTMLSelectElement).value as PngFilter;
								updateJob(job.id, (j) => ({
									...j,
									options: { ...j.options, filter }
								}));
							}}
						>
							{#each filterOptions as option}
								<option value={option.value}>{option.label}</option>
							{/each}
						</select>
						<p class="text-xs text-slate-500">Adaptive fast works well for most PNGs.</p>
					</label>

					{#if job.options.format === 'jpeg'}
						<label class="space-y-2 text-sm text-slate-200">
							<span class="flex items-center justify-between">
								Chroma subsampling
								<span class="text-xs text-slate-500">
									{job.options.subsampling420 ? '4:2:0' : '4:4:4'}
								</span>
							</span>
							<div class="flex items-center gap-2 rounded-lg border border-slate-700 px-3 py-2">
								<input
									type="checkbox"
									class="accent-sky-400"
									checked={job.options.subsampling420}
									on:change={(e) => {
										const subsampling420 = (e.target as HTMLInputElement).checked;
										updateJob(job.id, (j) => ({
											...j,
											options: { ...j.options, subsampling420 }
										}));
									}}
								/>
								<span class="text-xs text-slate-400">
									Enable 4:2:0 (smaller files, softer chroma)
								</span>
							</div>
						</label>
					{/if}
				</div>

				<div class="flex flex-wrap items-center gap-3">
					<button
						class="btn-primary"
						on:click={() => compressJob(job.id)}
						disabled={job.status === 'compressing' || busy || !wasmReady}
					>
						{!wasmReady
							? 'Loading WASM…'
							: job.status === 'compressing'
								? 'Compressing…'
								: 'Compress now'}
					</button>
					{#if job.status === 'error'}
						<span class="text-sm font-medium text-red-300">{job.error}</span>
					{/if}
					{#if job.status === 'done' && job.result}
						<span class="rounded-full border border-emerald-500/40 bg-emerald-500/10 px-3 py-1 text-xs font-semibold text-emerald-200">
							{formatSavings(job.result.savings)}
						</span>
						<span class="text-xs text-slate-400">Encoded in {job.result.elapsedMs.toFixed(1)} ms</span>
					{/if}
				</div>

				{#if job.result}
					<div class="grid gap-4 md:grid-cols-2">
						<div class="card p-4 space-y-2 text-sm text-slate-200">
							<div class="flex justify-between">
								<span class="text-slate-400">Original</span>
								<span class="font-semibold">{formatBytes(job.size)}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-slate-400">Compressed</span>
								<span class="font-semibold">{formatBytes(job.result.size)}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-slate-400">Savings</span>
								<span class="font-semibold text-emerald-300">{formatSavings(job.result.savings)}</span>
							</div>
							<div class="flex gap-2 pt-2">
								<a
									class="btn-primary"
									download={`compressed-${job.name}.${job.options.format === 'png' ? 'png' : 'jpg'}`}
									href={job.result.url}
								>
									Download
								</a>
								<button
									class="btn-ghost"
									on:click={() => {
										updateJob(job.id, (j) => ({
											...j,
											result: undefined,
											status: 'idle'
										}));
									}}
								>
									Reset result
								</button>
							</div>
						</div>

						<div class="card relative overflow-hidden">
							<div class="relative aspect-video w-full bg-slate-950/80">
								<img
									src={job.originalUrl}
									alt="Original"
									class="absolute inset-0 h-full w-full object-contain"
									draggable="false"
								/>
								<div
									class="absolute inset-0 overflow-hidden"
									style={`clip-path: inset(0 0 0 0); width: ${job.slider}%;`}
								>
									<img
										src={job.result.url}
										alt="Compressed"
										class="h-full w-full object-contain"
										draggable="false"
									/>
								</div>
								<div
									class="pointer-events-none absolute inset-y-0"
									style={`left: ${job.slider}%;`}
								>
									<div class="h-full w-px bg-white/60 shadow-[0_0_12px_rgba(255,255,255,0.35)]"></div>
								</div>
							</div>
							<div class="flex items-center gap-3 px-4 pb-4 pt-3">
								<label class="text-xs font-semibold text-slate-300" for={`slider-${job.id}`}>
									Drag to compare
								</label>
								<input
									id={`slider-${job.id}`}
									type="range"
									min="0"
									max="100"
									step="1"
									class="w-full accent-sky-400"
									value={job.slider}
									on:input={(e) => {
										const slider = Number((e.target as HTMLInputElement).value);
										updateJob(job.id, (j) => ({ ...j, slider }));
									}}
								/>
							</div>
						</div>
					</div>
				{/if}
			</article>
		{/each}
	</div>
</div>
