<script lang="ts">
  import { compressImage, initWasm, type PresetLevel } from "$lib/wasm";
  import { onDestroy, onMount } from "svelte";
  import JSZip from "jszip";

  type JobStatus = "idle" | "compressing" | "done" | "error";

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

  type ViewMode = "drop" | "single" | "list";

  const acceptMime = ["image/png", "image/jpeg"];
  let jobs: Job[] = $state([]);
  let dropActive = $state(false);
  let wasmReady = $state(false);
  let viewMode: ViewMode = $state("drop");
  let selectedJobId: string | null = $state(null);

  let globalOptions = $state({
    quality: 85,
    subsampling420: true,
    pngPreset: 1 as PresetLevel, // 0=faster, 1=auto, 2=smallest
    pngLossless: false, // Default OFF = lossy enabled for smaller PNGs
  });

  let isCompressing = $derived(jobs.some((j) => j.status === "compressing"));

  let detectedFormat = $derived.by(() => {
    if (jobs.length === 0) return "png" as const;
    const formats = new Set(
      jobs.map((j) => (j.type === "image/jpeg" ? "jpeg" : "png"))
    );
    if (formats.size === 1)
      return formats.values().next().value as "png" | "jpeg";
    return "mixed" as const;
  });

  let formatLabel = $derived(
    detectedFormat === "mixed"
      ? "Mixed"
      : detectedFormat === "jpeg"
        ? "JPEG"
        : "PNG"
  );

  const fileInputId = "file-input";

  let completedJobs = $derived(jobs.filter((j) => j.result));
  let totalOriginal = $derived(
    completedJobs.reduce((sum, j) => sum + j.size, 0)
  );
  let totalCompressed = $derived(
    completedJobs.reduce((sum, j) => sum + (j.result?.size ?? 0), 0)
  );
  let totalSavingsPct = $derived(
    totalOriginal > 0
      ? ((totalOriginal - totalCompressed) / totalOriginal) * 100
      : 0
  );
  let selectedJob = $derived(jobs.find((j) => j.id === selectedJobId) ?? null);
  let hasMultipleJobs = $derived(jobs.length > 1);

  type ZoomLevel = 1 | 2 | 4;
  let zoomLevel: ZoomLevel = $state(1);

  function formatBytes(bytes: number) {
    if (!bytes) return "0 B";
    const units = ["B", "KB", "MB", "GB"];
    const exponent = Math.min(
      Math.floor(Math.log(bytes) / Math.log(1024)),
      units.length - 1
    );
    const value = bytes / 1024 ** exponent;
    return `${value.toFixed(value >= 10 || value % 1 === 0 ? 0 : 1)} ${units[exponent]}`;
  }

  function formatSavings(delta: number) {
    const sign = delta >= 0 ? "-" : "+";
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
    const input = document.getElementById(
      fileInputId
    ) as HTMLInputElement | null;
    if (input) input.click();
  }

  onMount(() => {
    initWasm()
      .then(() => {
        wasmReady = true;
      })
      .catch((err) => {
        console.error("Failed to load WASM:", err);
      });

    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const isTypingContext =
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.isContentEditable);
      if (isTypingContext) return;

      if (e.key.toLowerCase() === "o" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        triggerFilePicker();
      }

      if (e.key === "Escape" && viewMode === "single" && selectedJobId) {
        if (hasMultipleJobs) {
          viewMode = "list";
          selectedJobId = null;
        } else {
          removeJob(selectedJobId);
        }
      }
    };

    window.addEventListener("keydown", handler);
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    window.addEventListener("touchmove", handleTouchMove, { passive: false });
    window.addEventListener("touchend", handleTouchEnd);
    return () => {
      window.removeEventListener("keydown", handler);
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
      window.removeEventListener("touchmove", handleTouchMove);
      window.removeEventListener("touchend", handleTouchEnd);
    };
  });

  function resetInput(event: Event) {
    const target = event.target as HTMLInputElement;
    target.value = "";
  }

  function createThumbnail(imageData: ImageData, maxSize: number = 80): string {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");
    if (!ctx) return "";

    const scale = Math.min(
      maxSize / imageData.width,
      maxSize / imageData.height
    );
    canvas.width = Math.round(imageData.width * scale);
    canvas.height = Math.round(imageData.height * scale);

    const tempCanvas = document.createElement("canvas");
    const tempCtx = tempCanvas.getContext("2d");
    if (!tempCtx) return "";
    tempCanvas.width = imageData.width;
    tempCanvas.height = imageData.height;
    tempCtx.putImageData(imageData, 0, 0);

    ctx.drawImage(tempCanvas, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL("image/jpeg", 0.7);
  }

  async function decodeFile(file: File) {
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new Error("Canvas context not available.");
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
      hasAlpha: detectAlpha(imageData.data),
    };
  }

  async function compressJob(job: Job) {
    const jobIndex = jobs.findIndex((j) => j.id === job.id);
    if (jobIndex === -1) return;

    jobs[jobIndex] = {
      ...jobs[jobIndex],
      status: "compressing",
      error: undefined,
    };

    // Yield to browser so UI can update before blocking WASM call
    await new Promise((resolve) => setTimeout(resolve, 0));

    try {
      // Use the job's original format for compression
      const jobFormat = job.type === "image/jpeg" ? "jpeg" : "png";
      // PNG slider: 0=Smaller(left), 1=Auto, 2=Faster(right)
      // Map to presets: 0->2(max), 1->1(balanced), 2->0(fast)
      const pngPresetValue = (2 - globalOptions.pngPreset) as PresetLevel;
      const { blob, elapsedMs } = await compressImage(job.imageData, {
        ...globalOptions,
        format: jobFormat,
        hasAlpha: job.hasAlpha,
        // PNG uses slider preset, JPEG uses balanced (1)
        preset: jobFormat === "png" ? pngPresetValue : (1 as PresetLevel),
        // Pass lossy option for PNG (inverted from lossless checkbox)
        lossy: jobFormat === "png" ? !globalOptions.pngLossless : undefined,
      });
      const url = URL.createObjectURL(blob);
      if (job.result?.url) URL.revokeObjectURL(job.result.url);
      const savings =
        job.size > 0 ? ((job.size - blob.size) / job.size) * 100 : 0;

      const currentIndex = jobs.findIndex((j) => j.id === job.id);
      if (currentIndex !== -1) {
        jobs[currentIndex] = {
          ...jobs[currentIndex],
          status: "done",
          result: { blob, url, size: blob.size, savings, elapsedMs },
        };
      }
    } catch (err) {
      const currentIndex = jobs.findIndex((j) => j.id === job.id);
      if (currentIndex !== -1) {
        jobs[currentIndex] = {
          ...jobs[currentIndex],
          status: "error",
          error: err instanceof Error ? err.message : "Compression failed",
        };
      }
    }
  }

  async function addFiles(fileList: FileList | File[]) {
    const files = Array.from(fileList).filter((f) => isSupported(f));
    if (files.length === 0) return;

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
        status: "idle",
        slider: 50,
      });
    }

    if (newJobs.length === 0) return;

    jobs = [...jobs, ...newJobs];

    if (jobs.length === 1) {
      viewMode = "single";
      selectedJobId = jobs[0].id;
    } else {
      viewMode = "list";
      selectedJobId = null;
    }

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
      viewMode = "drop";
      selectedJobId = null;
    } else if (selectedJobId === id) {
      if (jobs.length === 1) {
        viewMode = "single";
        selectedJobId = jobs[0].id;
      } else {
        viewMode = "list";
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
    viewMode = "drop";
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
    viewMode = "single";
    zoomLevel = 1;
  }

  function goBackToList() {
    viewMode = "list";
    selectedJobId = null;
  }

  async function downloadSingle(job: Job) {
    if (!job.result) return;
    const ext = job.type === "image/jpeg" ? "jpg" : "png";
    const name = job.name.replace(/\.[^/.]+$/, "") + `-compressed.${ext}`;
    const a = document.createElement("a");
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

    for (const job of completed) {
      if (!job.result) continue;
      const ext = job.type === "image/jpeg" ? "jpg" : "png";
      const name = job.name.replace(/\.[^/.]+$/, "") + `-compressed.${ext}`;
      zip.file(name, job.result.blob);
    }

    const blob = await zip.generateAsync({ type: "blob" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "compressed-images.zip";
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

  let isDragging = $state(false);
  let imageContainerRef: HTMLDivElement | null = $state(null);

  function handleSliderDrag(clientX: number) {
    if (!imageContainerRef || !selectedJob) return;
    const rect = imageContainerRef.getBoundingClientRect();
    const x = clientX - rect.left;
    const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100));
    updateSlider(selectedJob.id, Math.round(percentage));
  }

  function handleMouseDown(e: MouseEvent) {
    if (!selectedJob?.result) return;
    isDragging = true;
    handleSliderDrag(e.clientX);
  }

  function handleMouseMove(e: MouseEvent) {
    if (!isDragging) return;
    handleSliderDrag(e.clientX);
  }

  function handleMouseUp() {
    isDragging = false;
  }

  function handleTouchStart(e: TouchEvent) {
    if (!selectedJob?.result) return;
    isDragging = true;
    if (e.touches.length > 0) {
      handleSliderDrag(e.touches[0].clientX);
    }
  }

  function handleTouchMove(e: TouchEvent) {
    if (!isDragging) return;
    e.preventDefault();
    if (e.touches.length > 0) {
      handleSliderDrag(e.touches[0].clientX);
    }
  }

  function handleTouchEnd() {
    isDragging = false;
  }

  async function recompressAll() {
    for (const job of jobs) {
      if (job.result?.url) {
        URL.revokeObjectURL(job.result.url);
      }
      compressJob(job);
    }
  }
</script>

{#snippet iconLayers()}
  <svg
    class="w-40 h-40 opacity-60 -mb-6"
    viewBox="0 0 200 200"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
  >
    <!-- Bottom layer -->
    <path
      d="M40 125 L100 155 L160 125 L100 95 Z"
      stroke="currentColor"
      stroke-width="1.5"
      fill="none"
      opacity="0.3"
    />

    <!-- Middle layer -->
    <path
      d="M40 100 L100 130 L160 100 L100 70 Z"
      stroke="currentColor"
      stroke-width="1.5"
      fill="none"
      opacity="0.5"
    />

    <!-- Top layer -->
    <path
      d="M40 75 L100 105 L160 75 L100 45 Z"
      stroke="currentColor"
      stroke-width="1.5"
      fill="none"
      opacity="0.9"
    />
  </svg>
{/snippet}

{#snippet iconBack()}
  <svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path
      stroke-linecap="round"
      stroke-linejoin="round"
      stroke-width="2"
      d="M15 19l-7-7 7-7"
    />
  </svg>
{/snippet}

{#snippet iconClose()}
  <svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path
      stroke-linecap="round"
      stroke-linejoin="round"
      stroke-width="2"
      d="M6 18L18 6M6 6l12 12"
    />
  </svg>
{/snippet}

{#snippet iconDownload()}
  <svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path
      stroke-linecap="round"
      stroke-linejoin="round"
      stroke-width="2"
      d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
    />
  </svg>
{/snippet}

{#snippet dropView()}
  <div
    class="flex h-full w-full flex-col items-center justify-center p-4 transition-colors"
    class:bg-surface-1={dropActive}
    data-testid="drop-zone"
  >
    <div
      class="flex h-full w-full flex-col items-center justify-center rounded border-2 border-dashed transition-colors"
      class:border-neutral-600={dropActive}
      class:border-neutral-800={!dropActive}
    >
      <div class="flex flex-col items-center gap-6 text-center">
        <div class="text-neutral-500">
          {@render iconLayers()}
        </div>
        <div class="space-y-2">
          <p class="text-lg text-neutral-300">Drop PNG or JPEG files here</p>
          <p class="text-sm text-neutral-600">or</p>
        </div>
        <button
          class="btn-primary"
          onclick={triggerFilePicker}
          data-testid="select-files-button"
          >Select Files <kbd
            class="ml-1 inline-flex items-center rounded bg-black/10 px-1.5 py-0.5 text-xs font-normal"
            ><span class="text-sm leading-none mr-0.5">⌘</span>O</kbd
          ></button
        >
      </div>
    </div>
  </div>
{/snippet}

{#snippet singleView(job: Job)}
  <div class="relative h-full w-full bg-surface-0" data-testid="single-view">
    {#if hasMultipleJobs}
      <button
        class="absolute left-2 top-2 z-10 btn-ghost flex items-center gap-1 sm:left-4 sm:top-4"
        onclick={goBackToList}
        data-testid="back-button"
      >
        {@render iconBack()}
        <span class="hidden sm:inline">Back</span>
      </button>
    {/if}

    <button
      class="absolute right-2 top-2 z-10 btn-ghost text-neutral-500 hover:text-red-400 sm:right-4 sm:top-4"
      onclick={() => {
        if (hasMultipleJobs) {
          viewMode = "list";
          selectedJobId = null;
        } else {
          removeJob(job.id);
        }
      }}
      title={hasMultipleJobs ? "Close" : "Remove"}
      data-testid="close-button"
    >
      {@render iconClose()}
    </button>

    <div
      class="relative flex h-full w-full items-center justify-center p-4 sm:p-8 {zoomLevel >
      1
        ? 'overflow-auto'
        : 'overflow-hidden'}"
    >
      <!-- svelte-ignore a11y_no_static_element_interactions -->
      <div
        class="relative touch-none {zoomLevel > 1
          ? ''
          : 'max-h-full max-w-full'}"
        class:cursor-ew-resize={job.result}
        bind:this={imageContainerRef}
        onmousedown={handleMouseDown}
        ontouchstart={handleTouchStart}
        style={zoomLevel > 1
          ? `transform: scale(${zoomLevel}); transform-origin: center center;`
          : ""}
        data-testid="image-comparison-container"
      >
        <img
          src={job.originalUrl}
          alt="Original"
          class="object-contain select-none {zoomLevel > 1
            ? ''
            : 'max-h-[calc(100vh-200px)] max-w-[calc(100vw-2rem)] sm:max-h-[calc(100vh-180px)] sm:max-w-full'}"
          draggable="false"
          data-testid="original-image"
        />

        {#if job.result}
          <div
            class="absolute inset-0 overflow-hidden"
            style="clip-path: inset(0 {100 - job.slider}% 0 0);"
            data-testid="compressed-image-overlay"
          >
            <img
              src={job.result.url}
              alt="Compressed"
              class="object-contain select-none {zoomLevel > 1
                ? ''
                : 'max-h-[calc(100vh-200px)] max-w-[calc(100vw-2rem)] sm:max-h-[calc(100vh-180px)] sm:max-w-full'}"
              draggable="false"
              data-testid="compressed-image"
            />
          </div>

          <div
            class="absolute inset-y-0 cursor-ew-resize"
            style="left: {job.slider}%;"
            data-testid="comparison-slider"
          >
            <div
              class="h-full w-px bg-white shadow-[0_0_8px_rgba(255,255,255,0.5)]"
            ></div>
            <div
              class="absolute top-1/2 -translate-x-1/2 -translate-y-1/2 rounded bg-white px-2 py-0.5 text-[10px] font-medium text-black shadow-md"
            >
              {job.slider}%
            </div>
          </div>
        {/if}
      </div>
    </div>

    {#if job.result}
      <div
        class="absolute bottom-16 left-1/2 w-48 -translate-x-1/2 sm:bottom-20 sm:w-64"
        data-testid="slider-container"
      >
        <input
          type="range"
          min="0"
          max="100"
          step="1"
          value={job.slider}
          oninput={(e) =>
            updateSlider(job.id, Number((e.target as HTMLInputElement).value))}
          class="w-full"
          data-testid="comparison-range-slider"
        />
      </div>
    {/if}

    <div
      class="absolute left-2 bottom-2 max-w-[50%] text-xs text-neutral-500 sm:left-4 sm:bottom-4"
      data-testid="image-info"
    >
      <p class="truncate" data-testid="image-name">{job.name}</p>
      <p data-testid="image-dimensions">{job.width} × {job.height}</p>
    </div>

    <div
      class="absolute right-2 bottom-2 flex items-center gap-1 text-xs sm:right-4 sm:bottom-4"
      data-testid="zoom-controls"
    >
      <span class="text-neutral-500 mr-1">Zoom</span>
      {#each [1, 2, 4] as level}
        <button
          class="px-2 py-1 rounded transition-colors {zoomLevel === level
            ? 'bg-neutral-700 text-neutral-200'
            : 'text-neutral-500 hover:text-neutral-300 hover:bg-neutral-800'}"
          onclick={() => (zoomLevel = level as ZoomLevel)}
          data-testid="zoom-{level}x"
        >
          {level}x
        </button>
      {/each}
    </div>
  </div>
{/snippet}

{#snippet jobRow(job: Job)}
  <!-- svelte-ignore a11y_click_events_have_key_events -->
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div
    class="group flex w-full cursor-pointer items-center gap-2 rounded bg-surface-1 p-2 text-left transition-colors hover:bg-surface-2 sm:gap-4 sm:p-3"
    onclick={() => selectJob(job.id)}
    data-testid="job-row"
    data-job-status={job.status}
    data-job-name={job.name}
  >
    <div
      class="h-8 w-8 flex-shrink-0 overflow-hidden rounded bg-surface-2 sm:h-10 sm:w-10"
      data-testid="job-thumbnail"
    >
      {#if job.thumbnailUrl}
        <img src={job.thumbnailUrl} alt="" class="h-full w-full object-cover" />
      {/if}
    </div>

    <div class="min-w-0 flex-1">
      <p
        class="truncate text-xs text-neutral-200 sm:text-sm"
        data-testid="job-name"
      >
        {job.name}
      </p>
      <p
        class="hidden text-xs text-neutral-500 sm:block"
        data-testid="job-dimensions"
      >
        {job.width} × {job.height}
      </p>
    </div>

    <div class="hidden text-right text-xs sm:block">
      <p class="text-neutral-400" data-testid="job-original-size">
        {formatBytes(job.size)}
      </p>
      {#if job.result}
        <p class="text-neutral-500" data-testid="job-compressed-size">
          → {formatBytes(job.result.size)}
        </p>
      {:else if job.status === "compressing"}
        <p class="text-neutral-600" data-testid="job-compressing">...</p>
      {/if}
    </div>

    <div class="w-12 shrink-0 text-right sm:w-16">
      {#if job.result}
        <span
          class="text-xs font-medium sm:text-sm"
          class:text-terminal-green={job.result.savings >= 0}
          class:text-terminal-red={job.result.savings < 0}
          data-testid="job-savings"
        >
          {formatSavings(job.result.savings)}
        </span>
      {:else if job.status === "error"}
        <span
          class="text-xs text-terminal-red sm:text-sm"
          data-testid="job-error">Error</span
        >
      {/if}
    </div>

    <button
      class="btn-ghost hidden opacity-0 group-hover:opacity-100 sm:inline-flex"
      onclick={(e) => {
        e.stopPropagation();
        downloadSingle(job);
      }}
      disabled={!job.result}
      aria-label="Download"
      data-testid="job-download-button"
    >
      {@render iconDownload()}
    </button>

    <button
      class="btn-ghost hidden text-neutral-600 opacity-0 hover:text-red-400 group-hover:opacity-100 sm:inline-flex"
      onclick={(e) => {
        e.stopPropagation();
        removeJob(job.id);
      }}
      aria-label="Remove"
      data-testid="job-remove-button"
    >
      {@render iconClose()}
    </button>
  </div>
{/snippet}

{#snippet listView()}
  <div class="h-full overflow-auto p-2 sm:p-4" data-testid="list-view">
    <div class="mb-3 flex items-center justify-between sm:mb-4">
      <h2 class="text-sm text-neutral-400" data-testid="file-count">
        {jobs.length} files
      </h2>
      <button
        class="btn-ghost text-xs text-neutral-500"
        onclick={clearAll}
        data-testid="clear-all-button">Clear all</button
      >
    </div>
    <div class="space-y-1" data-testid="jobs-list">
      {#each jobs as job (job.id)}
        {@render jobRow(job)}
      {/each}
    </div>
  </div>
{/snippet}

{#snippet controlsFooter()}
  <footer
    class="flex min-h-14 flex-col gap-2 border-t border-border bg-surface-1 px-3 py-2 pb-[max(0.5rem,env(safe-area-inset-bottom))] sm:flex-row sm:flex-wrap sm:items-center sm:justify-between sm:gap-4 sm:px-4 sm:py-0 sm:pb-0"
    data-testid="controls-footer"
    data-wasm-ready={wasmReady}
  >
    <div
      class="flex flex-wrap items-center gap-2 sm:gap-4"
      data-testid="format-controls"
    >
      <div class="flex items-center gap-2 text-xs">
        <span class="text-neutral-500">Format</span>
        <span class="text-neutral-300" data-testid="format-label"
          >{formatLabel}</span
        >
      </div>

      {#if detectedFormat === "jpeg"}
        <label class="flex items-center gap-2 text-xs">
          <span class="text-neutral-500">Quality</span>
          <input
            type="range"
            min="1"
            max="100"
            step="1"
            class="w-16 sm:w-20"
            bind:value={globalOptions.quality}
            onchange={recompressAll}
            data-testid="quality-slider"
          />
          <span class="w-6 sm:w-8 text-neutral-400" data-testid="quality-value"
            >{globalOptions.quality}</span
          >
        </label>
      {:else if detectedFormat === "png"}
        <label class="flex items-center gap-2 text-xs">
          <span class="text-neutral-500 whitespace-nowrap">Smaller</span>
          <input
            type="range"
            min="0"
            max="2"
            step="1"
            class="w-16 sm:w-24"
            bind:value={globalOptions.pngPreset}
            onchange={recompressAll}
            data-testid="png-preset-slider"
          />
          <span class="text-neutral-500 whitespace-nowrap">Faster</span>
        </label>
        <label
          class="flex items-center gap-2 text-xs cursor-pointer"
          title="Enable lossless compression (larger files, preserves all colors)"
        >
          <input
            type="checkbox"
            class="accent-neutral-400"
            bind:checked={globalOptions.pngLossless}
            onchange={recompressAll}
            data-testid="png-lossless-checkbox"
          />
          <span class="text-neutral-400">Lossless</span>
        </label>
      {/if}
      <!-- When mixed, no format-specific controls are shown -->
    </div>

    <div
      class="flex flex-wrap items-center gap-2 text-xs sm:gap-4"
      data-testid="stats-section"
    >
      {#if completedJobs.length > 0}
        <div class="flex items-center gap-1 sm:gap-2">
          <span class="text-neutral-500">Original</span>
          <span class="text-neutral-300" data-testid="total-original-size"
            >{formatBytes(totalOriginal)}</span
          >
        </div>
        <span class="text-neutral-600">→</span>
        <div class="flex items-center gap-1 sm:gap-2">
          <span class="text-neutral-500">Compressed</span>
          <span class="text-neutral-300" data-testid="total-compressed-size"
            >{formatBytes(totalCompressed)}</span
          >
        </div>
        <span
          class="font-medium"
          class:text-terminal-green={totalSavingsPct >= 0}
          class:text-terminal-red={totalSavingsPct < 0}
          data-testid="total-savings"
        >
          {formatSavings(totalSavingsPct)}
        </span>
      {:else if jobs.length > 0}
        <span class="text-neutral-500" data-testid="compressing-status"
          >Compressing...</span
        >
      {/if}
    </div>

    <div
      class="flex shrink-0 items-center gap-2 max-sm:mt-2"
      data-testid="download-section"
    >
      {#if !wasmReady}
        <span class="text-xs text-neutral-500" data-testid="wasm-loading"
          >Loading WASM...</span
        >
      {:else if isCompressing}
        <span class="text-xs text-neutral-400">Compressing...</span>
      {:else if completedJobs.length > 0}
        <button
          class="btn-primary whitespace-nowrap"
          onclick={downloadAllAsZip}
          data-testid="download-button"
        >
          {completedJobs.length > 1 ? "Download All (.zip)" : "Download"}
        </button>
      {/if}
    </div>
  </footer>
{/snippet}

<svelte:head>
  <title>comprs</title>
  <meta
    name="description"
    content="Client-side PNG/JPEG compression powered by Rust WASM."
  />
</svelte:head>

<div
  class="flex h-svh max-w-full flex-col overflow-hidden bg-surface-0"
  data-testid="app-container"
>
  <main
    class="flex-1 overflow-hidden"
    ondrop={onDrop}
    ondragover={onDragOver}
    ondragleave={onDragLeave}
    data-testid="main-content"
    data-view-mode={viewMode}
  >
    {#if viewMode === "drop"}
      {@render dropView()}
    {:else if viewMode === "single" && selectedJob}
      {@render singleView(selectedJob)}
    {:else if viewMode === "list"}
      {@render listView()}
    {/if}
  </main>

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
    data-testid="file-input"
  />

  {@render controlsFooter()}
</div>
