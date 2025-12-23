import fs from "fs";
import path from "path";
import os from "os";
import { performance } from "perf_hooks";
import { createRequire } from "module";

const require = createRequire(import.meta.url);

const DEFAULT_ITERATIONS = process.env.BENCH_ITERATIONS
  ? Number(process.env.BENCH_ITERATIONS)
  : process.env.QUICK
  ? 3
  : 6;

const QUALITY = 85;
const DATASETS = [
  makeDataset("gradient", 512, 512, generateGradient),
  makeDataset("noisy", 512, 512, generateNoisy),
];

const libraryConfigs = [
  {
    name: "sharp",
    pkg: "sharp",
    formats: ["png", "jpeg"],
    factory: createSharpEncoders,
  },
  {
    name: "jimp",
    pkg: "jimp",
    formats: ["png", "jpeg"],
    factory: createJimpEncoders,
  },
  {
    name: "pngjs",
    pkg: "pngjs",
    formats: ["png"],
    factory: createPngjsEncoders,
  },
  {
    name: "jpeg-js",
    pkg: "jpeg-js",
    formats: ["jpeg"],
    factory: createJpegJsEncoders,
  },
  {
    name: "squoosh-lib",
    pkg: "@squoosh/lib",
    formats: ["png", "jpeg"],
    factory: createSquooshEncoders,
    iterationsOverride: 2, // squoosh spins up WASM; keep iterations short
  },
  {
    name: "browser-image-compression",
    pkg: "browser-image-compression",
    skip: "browser-only; requires DOM/canvas",
  },
];

async function main() {
  const { outputPath } = parseArgs(process.argv.slice(2));
  const results = [];
  const binarySizes = [];
  const notes = [
    "QUICK=1 reduces iterations to speed up the run.",
    "Package sizes are measured from installed node_modules when available.",
  ];

  for (const lib of libraryConfigs) {
    if (lib.skip) {
      results.push(
        ...DATASETS.flatMap((dataset) =>
          (lib.formats ?? ["png", "jpeg"]).map((format) => ({
            library: lib.name,
            package: lib.pkg,
            format,
            dataset: dataset.name,
            width: dataset.width,
            height: dataset.height,
            quality: format === "jpeg" ? QUALITY : null,
            output_bytes: null,
            duration_micros: null,
            status: "skipped",
            note: lib.skip,
          }))
        )
      );
      continue;
    }

    const imported = await importOptional(lib.pkg);
    if (!imported.ok) {
      const status =
        imported.error?.code === "ERR_MODULE_NOT_FOUND" ? "missing" : "error";
      const note =
        status === "missing"
          ? "not installed"
          : imported.error?.message ?? "import failed";
      results.push(
        ...DATASETS.flatMap((dataset) =>
          lib.formats.map((format) => ({
            library: lib.name,
            package: lib.pkg,
            format,
            dataset: dataset.name,
            width: dataset.width,
            height: dataset.height,
            quality: format === "jpeg" ? QUALITY : null,
            output_bytes: null,
            duration_micros: null,
            status,
            note,
          }))
        )
      );
      continue;
    }

    const module = imported.module;

    const pkgSize = await computePackageSize(lib.pkg);
    if (pkgSize) {
      binarySizes.push({
        library: lib.name,
        package: lib.pkg,
        bytes: pkgSize,
      });
    }

    const encoders = await lib.factory(module);
    for (const dataset of DATASETS) {
      for (const format of lib.formats) {
        const encoder = encoders[format];
        if (!encoder) {
          results.push({
            library: lib.name,
            package: lib.pkg,
            format,
            dataset: dataset.name,
            width: dataset.width,
            height: dataset.height,
            quality: format === "jpeg" ? QUALITY : null,
            output_bytes: null,
            duration_micros: null,
            status: "skipped",
            note: "encoder unavailable",
          });
          continue;
        }

        const iterations =
            lib.iterationsOverride && lib.iterationsOverride < DEFAULT_ITERATIONS
              ? lib.iterationsOverride
              : DEFAULT_ITERATIONS;

        try {
          const { bytes, durationMicros } = await measureEncoder(
            () => encoder(dataset),
            iterations
          );
          results.push({
            library: lib.name,
            package: lib.pkg,
            format,
            dataset: dataset.name,
            width: dataset.width,
            height: dataset.height,
            quality: format === "jpeg" ? QUALITY : null,
            output_bytes: bytes,
            duration_micros: durationMicros,
            status: "ok",
            note: iterations !== DEFAULT_ITERATIONS ? `iterations=${iterations}` : null,
          });
        } catch (err) {
          results.push({
            library: lib.name,
            package: lib.pkg,
            format,
            dataset: dataset.name,
            width: dataset.width,
            height: dataset.height,
            quality: format === "jpeg" ? QUALITY : null,
            output_bytes: null,
            duration_micros: null,
            status: "error",
            note: err instanceof Error ? err.message : "unknown error",
          });
        }
      }
    }
  }

  const summary = {
    generated_at: new Date().toISOString(),
    iterations: DEFAULT_ITERATIONS,
    quality: QUALITY,
    datasets: DATASETS.map(({ name, width, height }) => ({ name, width, height })),
    results,
    binary_sizes: binarySizes,
    notes,
  };

  if (outputPath) {
    fs.writeFileSync(outputPath, JSON.stringify(summary, null, 2));
    console.log(`Wrote JS benchmark summary to ${outputPath}`);
  } else {
    console.log(JSON.stringify(summary, null, 2));
  }
}

// -----------------------------------------------------------------------------
// Library encoder factories
// -----------------------------------------------------------------------------

async function createSharpEncoders(module) {
  const sharp = module.default ?? module;
  return {
    png: async (dataset) =>
      sharp(dataset.rgb, {
        raw: { width: dataset.width, height: dataset.height, channels: 3 },
      })
        .png({ compressionLevel: 6 })
        .toBuffer(),
    jpeg: async (dataset) =>
      sharp(dataset.rgb, {
        raw: { width: dataset.width, height: dataset.height, channels: 3 },
      })
        .jpeg({ quality: QUALITY })
        .toBuffer(),
  };
}

async function createJimpEncoders(module) {
  const Jimp = module.default ?? module;
  return {
    png: async (dataset) => {
      const img = await new Jimp({
        data: Buffer.from(dataset.rgba ?? toRgba(dataset.rgb)),
        width: dataset.width,
        height: dataset.height,
      });
      return img.getBufferAsync(Jimp.MIME_PNG);
    },
    jpeg: async (dataset) => {
      const img = await new Jimp({
        data: Buffer.from(dataset.rgba ?? toRgba(dataset.rgb)),
        width: dataset.width,
        height: dataset.height,
      });
      img.quality(QUALITY);
      return img.getBufferAsync(Jimp.MIME_JPEG);
    },
  };
}

async function createPngjsEncoders(module) {
  const { PNG } = module;
  return {
    png: async (dataset) =>
      PNG.sync.write({
        width: dataset.width,
        height: dataset.height,
        data: Buffer.from(dataset.rgba ?? toRgba(dataset.rgb)),
      }),
  };
}

async function createJpegJsEncoders(module) {
  const jpeg = module;
  return {
    jpeg: async (dataset) =>
      Buffer.from(
        jpeg.encode(
          {
            data: Buffer.from(dataset.rgba ?? toRgba(dataset.rgb)),
            width: dataset.width,
            height: dataset.height,
          },
          QUALITY
        ).data
      ),
  };
}

async function createSquooshEncoders(module) {
  const { ImagePool } = module;
  const poolSize = Math.max(1, Math.min(os.cpus().length, 2));
  return {
    png: async (dataset) => {
      const imagePool = new ImagePool(poolSize);
      const image = imagePool.ingestImage(dataset.rgb, {
        raw: { width: dataset.width, height: dataset.height, channels: 3 },
      });
      await image.encode({ oxipng: {} });
      await imagePool.close();
      return Buffer.from(image.encodedWith.oxipng.binary);
    },
    jpeg: async (dataset) => {
      const imagePool = new ImagePool(poolSize);
      const image = imagePool.ingestImage(dataset.rgb, {
        raw: { width: dataset.width, height: dataset.height, channels: 3 },
      });
      await image.encode({ mozjpeg: { quality: QUALITY } });
      await imagePool.close();
      return Buffer.from(image.encodedWith.mozjpeg.binary);
    },
  };
}

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------

function parseArgs(argv) {
  const args = [...argv];
  let outputPath = null;
  while (args.length) {
    const arg = args.shift();
    if (arg === "--output" || arg === "-o") {
      outputPath = args.shift() ?? null;
    }
  }
  return { outputPath };
}

function makeDataset(name, width, height, generator) {
  const rgb = generator(width, height);
  const rgba = toRgba(rgb);
  return { name, width, height, rgb, rgba };
}

function generateGradient(width, height) {
  const out = Buffer.alloc(width * height * 3);
  let idx = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const r = Math.floor((x * 255) / width);
      const g = Math.floor((y * 255) / height);
      const b = Math.floor(((x + y) * 127) / (width + height));
      out[idx++] = r;
      out[idx++] = g;
      out[idx++] = b;
    }
  }
  return out;
}

function generateNoisy(width, height) {
  const out = Buffer.alloc(width * height * 3);
  let idx = 0;
  let seed = 12345 >>> 0;
  for (let i = 0; i < width * height; i++) {
    seed = (seed * 1103515245 + 12345) >>> 0;
    const r = seed >>> 16;
    seed = (seed * 1103515245 + 12345) >>> 0;
    const g = seed >>> 16;
    seed = (seed * 1103515245 + 12345) >>> 0;
    const b = seed >>> 16;
    out[idx++] = r & 0xff;
    out[idx++] = g & 0xff;
    out[idx++] = b & 0xff;
  }
  return out;
}

function toRgba(rgb) {
  const pixelCount = rgb.length / 3;
  const rgba = Buffer.alloc(pixelCount * 4);
  for (let i = 0; i < pixelCount; i++) {
    const src = i * 3;
    const dst = i * 4;
    rgba[dst] = rgb[src];
    rgba[dst + 1] = rgb[src + 1];
    rgba[dst + 2] = rgb[src + 2];
    rgba[dst + 3] = 255;
  }
  return rgba;
}

async function measureEncoder(fn, iterations) {
  // Warm-up
  const sample = await fn();
  const start = performance.now();
  for (let i = 0; i < iterations; i++) {
    await fn();
  }
  const end = performance.now();
  const durationMicros = ((end - start) / iterations) * 1000;
  const bytes = sample?.length ?? sample?.byteLength ?? null;
  return { bytes, durationMicros };
}

async function importOptional(name) {
  try {
    const module = await import(name);
    return { ok: true, module };
  } catch (error) {
    return { ok: false, error };
  }
}

async function computePackageSize(pkg) {
  try {
    const resolved = require.resolve(pkg);
    const root = findPackageRoot(path.dirname(resolved));
    return directorySize(root);
  } catch (err) {
    return null;
  }
}

function findPackageRoot(startDir) {
  let dir = startDir;
  while (dir && dir !== path.dirname(dir)) {
    if (fs.existsSync(path.join(dir, "package.json"))) {
      return dir;
    }
    dir = path.dirname(dir);
  }
  return startDir;
}

function directorySize(dir) {
  let total = 0;
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const entry of entries) {
    // Skip nested node_modules to avoid double-counting shared deps
    if (entry.name === "node_modules") continue;
    const full = path.join(dir, entry.name);
    const stat = fs.statSync(full);
    if (stat.isDirectory()) {
      total += directorySize(full);
    } else {
      total += stat.size;
    }
  }
  return total;
}

await main();
