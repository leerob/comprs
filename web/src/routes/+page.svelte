<script lang="ts">
	import { compressImage, initWasm, type PngFilter } from '$lib/wasm';
	import { onDestroy, onMount } from 'svelte';
	import JSZip from 'jszip';

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
		thumbnailUrl: string;
		imageData: ImageData;
		status: JobStatus;
		slider: number;
		error?: string;
		result?: {
			blob: Blob;
			url: string;
			size: number;
			savings: number;
			elapsedMs: number;
		};
	};

	type ViewMode = 'drop' | 'single' | 'list';

	const acceptMime = ['image/png', 'image/jpeg'];
	let jobs: Job[] = $state([]);
	let dropActive = $state(false);
	let wasmReady = $state(false);
	let viewMode: ViewMode = $state('drop');
	let selectedJobId: string | null = $state(null);

	// Global compression options
	let globalOptions = $state({
		format: 'png' as 'png' | 'jpeg',
		quality: 85,
		compressionLevel: 6,
		filter: 'adaptive' as PngFilter,
		subsampling420: true
	});

	// Track if user has manually changed the format
	let formatTouched = $state(false);

	const filterOptions: { label: string; value: PngFilter }[] = [
		{ label: 'Adaptive', value: 'adaptive' },
		{ label: 'Adaptive Fast', value: 'adaptive-fast' },
		{ label: 'None', value: 'none' },
		{ label: 'Sub', value: 'sub' },
		{ label: 'Up', value: 'up' },
		{ label: 'Average', value: 'average' },
		{ label: 'Paeth', value: 'paeth' }
	];

	const fileInputId = 'file-input';

	// Derived state
	let completedJobs = $derived(jobs.filter((j) => j.result));
	let totalOriginal = $derived(completedJobs.reduce((sum, j) => sum + j.size, 0));
	let totalCompressed = $derived(completedJobs.reduce((sum, j) => sum + (j.result?.size ?? 0), 0));
	let totalSavingsPct = $derived(
		totalOriginal > 0 ? ((totalOriginal - totalCompressed) / totalOriginal) * 100 : 0
	);
	let selectedJob = $derived(jobs.find((j) => j.id === selectedJobId) ?? null);
	let hasMultipleJobs = $derived(jobs.length > 1);

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
				console.error('Failed to load WASM:', err);
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

			// ESC to close modal (go back to list) or remove if single image
			if (e.key === 'Escape' && viewMode === 'single' && selectedJobId) {
				if (hasMultipleJobs) {
					viewMode = 'list';
					selectedJobId = null;
				} else {
					removeJob(selectedJobId);
				}
			}
		};

		window.addEventListener('keydown', handler);
		window.addEventListener('mousemove', handleMouseMove);
		window.addEventListener('mouseup', handleMouseUp);
		return () => {
			window.removeEventListener('keydown', handler);
			window.removeEventListener('mousemove', handleMouseMove);
			window.removeEventListener('mouseup', handleMouseUp);
		};
	});

	function resetInput(event: Event) {
		const target = event.target as HTMLInputElement;
		target.value = '';
	}

	function createThumbnail(imageData: ImageData, maxSize: number = 80): string {
		const canvas = document.createElement('canvas');
		const ctx = canvas.getContext('2d');
		if (!ctx) return '';

		const scale = Math.min(maxSize / imageData.width, maxSize / imageData.height);
		canvas.width = Math.round(imageData.width * scale);
		canvas.height = Math.round(imageData.height * scale);

		const tempCanvas = document.createElement('canvas');
		const tempCtx = tempCanvas.getContext('2d');
		if (!tempCtx) return '';
		tempCanvas.width = imageData.width;
		tempCanvas.height = imageData.height;
		tempCtx.putImageData(imageData, 0, 0);

		ctx.drawImage(tempCanvas, 0, 0, canvas.width, canvas.height);
		return canvas.toDataURL('image/jpeg', 0.7);
	}

	async function decodeFile(file: File) {
		const canvas = document.createElement('canvas');
		const ctx = canvas.getContext('2d', { willReadFrequently: true });
		if (!ctx) throw new Error('Canvas context not available.');
		const bitmap = await createImageBitmap(file);
		const width = bitmap.width;
		const height = bitmap.height;
		canvas.width = width;
		canvas.height = height;
		ctx.clearRect(0, 0, width, height);
		ctx.drawImage(bitmap, 0, 0);
		const imageData = ctx.getImageData(0, 0, width, height);
		bitmap.close();
		return {
			imageData,
			width,
			height,
			hasAlpha: detectAlpha(imageData.data)
		};
	}

	async function compressJob(job: Job) {
		if (!wasmReady) {
			try {
				await initWasm();
				wasmReady = true;
			} catch (err) {
				console.error('Failed to load WASM:', err);
				return;
			}
		}

		const jobIndex = jobs.findIndex((j) => j.id === job.id);
		if (jobIndex === -1) return;

		jobs[jobIndex] = { ...jobs[jobIndex], status: 'compressing', error: undefined };

		try {
			const options = {
				format: globalOptions.format,
				quality: globalOptions.quality,
				compressionLevel: globalOptions.compressionLevel,
				filter: globalOptions.filter,
				subsampling420: globalOptions.subsampling420
			};
			const { blob, elapsedMs } = await compressImage(job.imageData, options);
			const url = URL.createObjectURL(blob);
			if (job.result?.url) URL.revokeObjectURL(job.result.url);
			const savings = job.size > 0 ? ((job.size - blob.size) / job.size) * 100 : 0;

			const currentIndex = jobs.findIndex((j) => j.id === job.id);
			if (currentIndex !== -1) {
				jobs[currentIndex] = {
					...jobs[currentIndex],
					status: 'done',
					result: { blob, url, size: blob.size, savings, elapsedMs }
				};
			}
		} catch (err) {
			const currentIndex = jobs.findIndex((j) => j.id === job.id);
			if (currentIndex !== -1) {
				jobs[currentIndex] = {
					...jobs[currentIndex],
					status: 'error',
					error: err instanceof Error ? err.message : 'Compression failed'
				};
			}
		}
	}

	async function addFiles(fileList: FileList | File[]) {
		const files = Array.from(fileList).filter((f) => isSupported(f));
		if (files.length === 0) return;

		// Auto-flip to JPEG if user selects a single JPEG and hasn't touched format
		if (!formatTouched && files.length === 1 && files[0].type === 'image/jpeg') {
			globalOptions.format = 'jpeg';
		}

		const newJobs: Job[] = [];

		for (const file of files) {
			let decoded;
			try {
				decoded = await decodeFile(file);
			} catch {
				continue;
			}
			const { imageData, width, height, hasAlpha } = decoded;
			const url = URL.createObjectURL(file);
			const thumbnailUrl = createThumbnail(imageData);
			const id = crypto.randomUUID();

			newJobs.push({
				id,
				name: file.name,
				type: file.type,
				size: file.size,
				width,
				height,
				hasAlpha,
				originalUrl: url,
				thumbnailUrl,
				imageData,
				status: 'idle',
				slider: 50
			});
		}

		if (newJobs.length === 0) return;

		jobs = [...jobs, ...newJobs];

		// Determine view mode
		if (jobs.length === 1) {
			viewMode = 'single';
			selectedJobId = jobs[0].id;
		} else {
			viewMode = 'list';
			selectedJobId = null;
		}

		// Auto-compress all new jobs
		for (const job of newJobs) {
			compressJob(job);
		}
	}

	function removeJob(id: string) {
		const job = jobs.find((j) => j.id === id);
		if (job) {
			URL.revokeObjectURL(job.originalUrl);
			if (job.result?.url) URL.revokeObjectURL(job.result.url);
		}
		jobs = jobs.filter((j) => j.id !== id);

		if (jobs.length === 0) {
			viewMode = 'drop';
			selectedJobId = null;
		} else if (selectedJobId === id) {
			if (jobs.length === 1) {
				viewMode = 'single';
				selectedJobId = jobs[0].id;
			} else {
				viewMode = 'list';
				selectedJobId = null;
			}
		}
	}

	function clearAll() {
		for (const job of jobs) {
			URL.revokeObjectURL(job.originalUrl);
			if (job.result?.url) URL.revokeObjectURL(job.result.url);
		}
		jobs = [];
		viewMode = 'drop';
		selectedJobId = null;
	}

	onDestroy(() => {
		for (const job of jobs) {
			URL.revokeObjectURL(job.originalUrl);
			if (job.result?.url) URL.revokeObjectURL(job.result.url);
		}
	});

	function selectJob(id: string) {
		selectedJobId = id;
		viewMode = 'single';
	}

	function goBackToList() {
		viewMode = 'list';
		selectedJobId = null;
	}

	async function downloadSingle(job: Job) {
		if (!job.result) return;
		const ext = globalOptions.format === 'png' ? 'png' : 'jpg';
		const name = job.name.replace(/\.[^/.]+$/, '') + `-compressed.${ext}`;
		const a = document.createElement('a');
		a.href = job.result.url;
		a.download = name;
		a.click();
	}

	async function downloadAllAsZip() {
		const completed = jobs.filter((j) => j.result);
		if (completed.length === 0) return;

		if (completed.length === 1) {
			downloadSingle(completed[0]);
			return;
		}

		const zip = new JSZip();
		const ext = globalOptions.format === 'png' ? 'png' : 'jpg';

		for (const job of completed) {
			if (!job.result) continue;
			const name = job.name.replace(/\.[^/.]+$/, '') + `-compressed.${ext}`;
			zip.file(name, job.result.blob);
		}

		const blob = await zip.generateAsync({ type: 'blob' });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = 'compressed-images.zip';
		a.click();
		URL.revokeObjectURL(url);
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

	function onDragLeave(event: DragEvent) {
		// Only deactivate if leaving the drop zone entirely
		const rect = (event.currentTarget as HTMLElement).getBoundingClientRect();
		const x = event.clientX;
		const y = event.clientY;
		if (x < rect.left || x > rect.right || y < rect.top || y > rect.bottom) {
			dropActive = false;
		}
	}

	function updateSlider(jobId: string, value: number) {
		const index = jobs.findIndex((j) => j.id === jobId);
		if (index !== -1) {
			jobs[index] = { ...jobs[index], slider: value };
		}
	}

	// Draggable slider state
	let isDragging = $state(false);
	let imageContainerRef: HTMLDivElement | null = $state(null);

	function handleSliderDrag(e: MouseEvent) {
		if (!imageContainerRef || !selectedJob) return;
		const rect = imageContainerRef.getBoundingClientRect();
		const x = e.clientX - rect.left;
		const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
		updateSlider(selectedJob.id, Math.round(percentage));
	}

	function handleMouseDown(e: MouseEvent) {
		if (!selectedJob?.result) return;
		isDragging = true;
		handleSliderDrag(e);
	}

	function handleMouseMove(e: MouseEvent) {
		if (!isDragging) return;
		handleSliderDrag(e);
	}

	function handleMouseUp() {
		isDragging = false;
	}

	// Re-compress all jobs when global options change
	async function recompressAll() {
		for (const job of jobs) {
			if (job.result?.url) {
				URL.revokeObjectURL(job.result.url);
			}
			compressJob(job);
		}
	}
</script>

<svelte:head>
	<title>comprs</title>
	<meta name="description" content="Client-side PNG/JPEG compression powered by Rust WASM." />
</svelte:head>

<div class="flex h-screen flex-col bg-surface-0">
	<!-- Main Content Area -->
	<main
		class="flex-1 overflow-hidden"
		ondrop={onDrop}
		ondragover={onDragOver}
		ondragleave={onDragLeave}
	>
		{#if viewMode === 'drop'}
			<!-- Drop Zone View -->
			<div
				class="flex h-full w-full flex-col items-center justify-center p-4 transition-colors"
				class:bg-surface-1={dropActive}
			>
				<div
					class="flex h-full w-full flex-col items-center justify-center rounded border-2 border-dashed transition-colors"
					class:border-neutral-600={dropActive}
					class:border-neutral-800={!dropActive}
				>
					<div class="flex flex-col items-center gap-6 text-center">
						<div class="text-neutral-500">
							<svg
								class="h-16 w-16"
								fill="none"
								stroke="currentColor"
								viewBox="0 0 24 24"
							>
								<path
									stroke-linecap="round"
									stroke-linejoin="round"
									stroke-width="1"
									d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
								/>
							</svg>
						</div>
						<div class="space-y-2">
							<p class="text-lg text-neutral-300">Drop PNG or JPEG files here</p>
							<p class="text-sm text-neutral-600">or</p>
						</div>
						<button class="btn-primary" onclick={triggerFilePicker}>
							Select Files
						</button>
						<p class="text-xs text-neutral-600">
							<kbd class="inline-flex items-center gap-1 rounded bg-surface-2 px-1.5 py-0.5 text-neutral-400"><span>⌘</span><span>O</span></kbd>
						</p>
					</div>
				</div>
			</div>
		{:else if viewMode === 'single' && selectedJob}
			<!-- Single Image View with Comparison -->
			<div class="relative h-full w-full bg-surface-0">
				{#if hasMultipleJobs}
					<button
						class="absolute left-4 top-4 z-10 btn-ghost flex items-center gap-1"
						onclick={goBackToList}
					>
						<svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
							<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7" />
						</svg>
						Back
					</button>
				{/if}

				<button
					class="absolute right-4 top-4 z-10 btn-ghost text-neutral-500 hover:text-red-400"
					onclick={() => {
						if (hasMultipleJobs) {
							viewMode = 'list';
							selectedJobId = null;
						} else {
							removeJob(selectedJob.id);
						}
					}}
					title={hasMultipleJobs ? "Close" : "Remove"}
				>
					<svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
					</svg>
				</button>

				<!-- Image comparison container -->
				<div class="relative flex h-full w-full items-center justify-center p-8">
					<!-- svelte-ignore a11y_no_static_element_interactions -->
					<div
						class="relative max-h-full max-w-full"
						class:cursor-ew-resize={selectedJob.result}
						bind:this={imageContainerRef}
						onmousedown={handleMouseDown}
					>
						<!-- Original image (background) -->
						<img
							src={selectedJob.originalUrl}
							alt="Original"
							class="max-h-[calc(100vh-180px)] max-w-full object-contain select-none"
							draggable="false"
						/>

						<!-- Compressed image overlay with clip -->
						{#if selectedJob.result}
							<div
								class="absolute inset-0 overflow-hidden"
								style="clip-path: inset(0 {100 - selectedJob.slider}% 0 0);"
							>
								<img
									src={selectedJob.result.url}
									alt="Compressed"
									class="max-h-[calc(100vh-180px)] max-w-full object-contain select-none"
									draggable="false"
								/>
							</div>

							<!-- Slider line -->
							<div
								class="absolute inset-y-0 cursor-ew-resize"
								style="left: {selectedJob.slider}%;"
							>
								<div class="h-full w-px bg-white shadow-[0_0_8px_rgba(255,255,255,0.5)]"></div>
								<div class="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 rounded bg-white px-2 py-0.5 text-[10px] font-medium text-black shadow-md">
									{selectedJob.slider}%
								</div>
							</div>
						{/if}
					</div>
				</div>

				<!-- Slider control -->
				{#if selectedJob.result}
					<div class="absolute bottom-20 left-1/2 w-64 -translate-x-1/2">
						<input
							type="range"
							min="0"
							max="100"
							step="1"
							value={selectedJob.slider}
							oninput={(e) => updateSlider(selectedJob.id, Number((e.target as HTMLInputElement).value))}
							class="w-full"
						/>
					</div>
				{/if}

				<!-- File info overlay -->
				<div class="absolute left-4 bottom-4 text-xs text-neutral-500">
					<p>{selectedJob.name}</p>
					<p>{selectedJob.width} × {selectedJob.height}</p>
				</div>
			</div>
		{:else if viewMode === 'list'}
			<!-- List View -->
			<div class="h-full overflow-auto p-4">
				<div class="mb-4 flex items-center justify-between">
					<h2 class="text-sm text-neutral-400">{jobs.length} files</h2>
					<button class="btn-ghost text-xs text-neutral-500" onclick={clearAll}>
						Clear all
					</button>
				</div>

				<div class="space-y-1">
					{#each jobs as job (job.id)}
						<!-- svelte-ignore a11y_click_events_have_key_events -->
						<!-- svelte-ignore a11y_no_static_element_interactions -->
						<div
							class="group flex w-full cursor-pointer items-center gap-4 rounded bg-surface-1 p-3 text-left transition-colors hover:bg-surface-2"
							onclick={() => selectJob(job.id)}
						>
							<!-- Thumbnail -->
							<div class="h-10 w-10 flex-shrink-0 overflow-hidden rounded bg-surface-2">
								{#if job.thumbnailUrl}
									<img
										src={job.thumbnailUrl}
										alt=""
										class="h-full w-full object-cover"
									/>
								{/if}
							</div>

							<!-- File info -->
							<div class="min-w-0 flex-1">
								<p class="truncate text-sm text-neutral-200">{job.name}</p>
								<p class="text-xs text-neutral-500">
									{job.width} × {job.height}
								</p>
							</div>

							<!-- Size info -->
							<div class="text-right text-xs">
								<p class="text-neutral-400">{formatBytes(job.size)}</p>
								{#if job.result}
									<p class="text-neutral-500">→ {formatBytes(job.result.size)}</p>
								{:else if job.status === 'compressing'}
									<p class="text-neutral-600">...</p>
								{/if}
							</div>

							<!-- Savings -->
							<div class="w-16 text-right">
								{#if job.result}
									<span class="text-sm font-medium" class:text-terminal-green={job.result.savings >= 0} class:text-terminal-red={job.result.savings < 0}>
										{formatSavings(job.result.savings)}
									</span>
								{:else if job.status === 'error'}
									<span class="text-sm text-terminal-red">Error</span>
								{/if}
							</div>

							<!-- Download button -->
							<button
								class="btn-ghost opacity-0 group-hover:opacity-100"
								onclick={(e) => { e.stopPropagation(); downloadSingle(job); }}
								disabled={!job.result}
								aria-label="Download compressed file"
							>
								<svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
									<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
								</svg>
							</button>

							<!-- Remove button -->
							<button
								class="btn-ghost text-neutral-600 opacity-0 hover:text-red-400 group-hover:opacity-100"
								onclick={(e) => { e.stopPropagation(); removeJob(job.id); }}
								aria-label="Remove file"
							>
								<svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
									<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
								</svg>
							</button>
						</div>
					{/each}
				</div>
			</div>
		{/if}
	</main>

	<!-- Hidden file input -->
	<input
		id={fileInputId}
		type="file"
		accept="image/png,image/jpeg"
		multiple
		class="hidden"
		onchange={(e) => {
			const fileList = (e.target as HTMLInputElement).files;
			if (fileList) addFiles(fileList);
			resetInput(e);
		}}
	/>

	<!-- Persistent Bottom Bar -->
	<footer class="flex h-14 items-center justify-between border-t border-border bg-surface-1 px-4">
		<!-- Left: Compression Controls -->
		<div class="flex items-center gap-4">
			<!-- Format -->
			<label class="flex items-center gap-2 text-xs">
				<span class="text-neutral-500">Format</span>
				<select
					class="input w-20"
					bind:value={globalOptions.format}
					onchange={() => { formatTouched = true; recompressAll(); }}
				>
					<option value="png">PNG</option>
					<option value="jpeg">JPEG</option>
				</select>
			</label>

			<!-- Quality / Compression -->
			{#if globalOptions.format === 'jpeg'}
				<label class="flex items-center gap-2 text-xs">
					<span class="text-neutral-500">Quality</span>
					<input
						type="range"
						min="1"
						max="100"
						step="1"
						class="w-20"
						bind:value={globalOptions.quality}
						onchange={recompressAll}
					/>
					<span class="w-8 text-neutral-400">{globalOptions.quality}</span>
				</label>
			{:else}
				<label class="flex items-center gap-2 text-xs">
					<span class="text-neutral-500">Level</span>
					<input
						type="range"
						min="1"
						max="9"
						step="1"
						class="w-20"
						bind:value={globalOptions.compressionLevel}
						onchange={recompressAll}
					/>
					<span class="w-8 text-neutral-400">{globalOptions.compressionLevel}</span>
				</label>

				<label class="flex items-center gap-2 text-xs">
					<span class="text-neutral-500">Filter</span>
					<select
						class="input w-28"
						bind:value={globalOptions.filter}
						onchange={recompressAll}
					>
						{#each filterOptions as opt}
							<option value={opt.value}>{opt.label}</option>
						{/each}
					</select>
				</label>
			{/if}
		</div>

		<!-- Center: Stats -->
		<div class="flex items-center gap-6 text-xs">
			{#if completedJobs.length > 0}
				<div class="flex items-center gap-2">
					<span class="text-neutral-500">Original</span>
					<span class="text-neutral-300">{formatBytes(totalOriginal)}</span>
				</div>
				<span class="text-neutral-600">→</span>
				<div class="flex items-center gap-2">
					<span class="text-neutral-500">Compressed</span>
					<span class="text-neutral-300">{formatBytes(totalCompressed)}</span>
				</div>
				<span class="font-medium" class:text-terminal-green={totalSavingsPct >= 0} class:text-terminal-red={totalSavingsPct < 0}>{formatSavings(totalSavingsPct)}</span>
			{:else if jobs.length > 0}
				<span class="text-neutral-500">Compressing...</span>
			{/if}
		</div>

		<!-- Right: Download Button -->
		<div class="flex items-center gap-2">
			{#if !wasmReady}
				<span class="text-xs text-neutral-500">Loading WASM...</span>
			{:else if completedJobs.length > 0}
				<button
					class="btn-primary"
					onclick={downloadAllAsZip}
				>
					{#if completedJobs.length > 1}
						Download All (.zip)
					{:else}
						Download
					{/if}
				</button>
			{/if}
		</div>
	</footer>
</div>
