import fs from "fs";
import path from "path";

function parseArgs(argv) {
  const args = [...argv];
  let rustPath = null;
  let jsPath = null;
  let outputPath = null;
  while (args.length) {
    const arg = args.shift();
    if (arg === "--rust") {
      rustPath = args.shift() ?? null;
    } else if (arg === "--js") {
      jsPath = args.shift() ?? null;
    } else if (arg === "--output" || arg === "-o") {
      outputPath = args.shift() ?? null;
    }
  }
  if (!rustPath || !jsPath) {
    throw new Error("Usage: node aggregate.mjs --rust <rust.json> --js <js.json> [--output <file>]");
  }
  return { rustPath, jsPath, outputPath };
}

function loadJson(filePath) {
  const contents = fs.readFileSync(filePath, "utf8");
  return JSON.parse(contents);
}

function msFromMicros(micros) {
  return micros / 1000;
}

function formatMs(ms) {
  return `${ms.toFixed(2)} ms`;
}

function formatBytes(bytes) {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(2)} KB`;
  return `${bytes} B`;
}

function rankSpeed(entries, format) {
  return entries
    .filter((e) => e.format === format && e.status === "ok")
    .sort((a, b) => a.duration_micros - b.duration_micros)
    .map((e, idx) => ({
      rank: idx + 1,
      ...e,
      duration_ms: msFromMicros(e.duration_micros),
    }));
}

function rankOutputSizes(entries, format, dataset) {
  return entries
    .filter((e) => e.format === format && e.dataset === dataset && e.status === "ok" && e.output_bytes != null)
    .sort((a, b) => a.output_bytes - b.output_bytes)
    .map((e, idx) => ({
      rank: idx + 1,
      ...e,
    }));
}

function rankBinarySizes(entries) {
  return entries
    .filter((e) => e.bytes != null)
    .sort((a, b) => a.bytes - b.bytes)
    .map((e, idx) => ({ rank: idx + 1, ...e }));
}

function toMarkdownTable(rows, headers) {
  const headerLine = `| ${headers.join(" | ")} |`;
  const separator = `| ${headers.map(() => "---").join(" | ")} |`;
  const body = rows
    .map((row) => `| ${headers.map((h) => (row[h] ?? "")).join(" | ")} |`)
    .join("\n");
  return [headerLine, separator, body].join("\n");
}

function normalizeRustSpeed(speed) {
  return speed.map((s) => ({
    library: s.library ?? (s.operation.includes("PNG") ? "comprs" : "image crate"),
    environment: "rust",
    format: s.operation.toLowerCase().includes("png") ? "png" : "jpeg",
    dataset: "gradient",
    duration_micros: s.duration_micros ?? s.duration_us ?? 0,
    status: "ok",
    note: s.subsampling ? `subsampling=${s.subsampling}` : null,
  }));
}

function normalizeRustOutputs(outputs) {
  return outputs.map((o) => ({
    library: o.library ?? "comprs",
    environment: "rust",
    format: o.format.toLowerCase().startsWith("jpeg") ? "jpeg" : "png",
    dataset: "gradient",
    output_bytes: o.bytes ?? o.size ?? o.length ?? 0,
    status: "ok",
    note: o.subsampling ? `subsampling=${o.subsampling}` : null,
  }));
}

function normalizeJsResults(results) {
  return results.map((r) => ({
    library: r.library,
    environment: "js",
    format: r.format,
    dataset: r.dataset,
    duration_micros: r.duration_micros,
    output_bytes: r.output_bytes,
    status: r.status,
    note: r.note ?? null,
  }));
}

function normalizeRustBinarySizes(entries) {
  return entries.map((b) => ({
    library: b.library,
    environment: "rust",
    bytes: b.bytes ? Number(b.bytes) : null,
    notes: `${b.notes} (${b.status})`,
  }));
}

function normalizeJsBinarySizes(entries) {
  return entries.map((b) => ({
    library: b.library,
    environment: "js",
    bytes: b.bytes,
    notes: b.package,
  }));
}

function buildMarkdown(summary) {
  const { speedRankings, outputRankings, binaryRankings, notes } = summary;
  const parts = [];
  parts.push("# Cross-language benchmark summary");
  parts.push("");
  parts.push("## Speed (PNG, 512x512 gradient)");
  parts.push(
    toMarkdownTable(
      speedRankings.png.map((r) => ({
        Rank: r.rank,
        Library: r.library,
        Env: r.environment,
        "Time (ms)": formatMs(r.duration_ms),
        Note: r.note ?? "",
      })),
      ["Rank", "Library", "Env", "Time (ms)", "Note"]
    )
  );
  parts.push("");
  parts.push("## Speed (JPEG q85, 512x512 gradient)");
  parts.push(
    toMarkdownTable(
      speedRankings.jpeg.map((r) => ({
        Rank: r.rank,
        Library: r.library,
        Env: r.environment,
        "Time (ms)": formatMs(r.duration_ms),
        Note: r.note ?? "",
      })),
      ["Rank", "Library", "Env", "Time (ms)", "Note"]
    )
  );
  parts.push("");
  parts.push("## Output size (PNG, 512x512 gradient)");
  parts.push(
    toMarkdownTable(
      outputRankings.png.map((r) => ({
        Rank: r.rank,
        Library: r.library,
        Env: r.environment,
        "Size": formatBytes(r.output_bytes),
        Note: r.note ?? "",
      })),
      ["Rank", "Library", "Env", "Size", "Note"]
    )
  );
  parts.push("");
  parts.push("## Output size (JPEG q85, 512x512 gradient)");
  parts.push(
    toMarkdownTable(
      outputRankings.jpeg.map((r) => ({
        Rank: r.rank,
        Library: r.library,
        Env: r.environment,
        "Size": formatBytes(r.output_bytes),
        Note: r.note ?? "",
      })),
      ["Rank", "Library", "Env", "Size", "Note"]
    )
  );
  parts.push("");
  parts.push("## Binary/package size");
  parts.push(
    toMarkdownTable(
      binaryRankings.map((r) => ({
        Rank: r.rank,
        Library: r.library,
        Env: r.environment,
        Size: formatBytes(r.bytes),
        Note: r.notes ?? "",
      })),
      ["Rank", "Library", "Env", "Size", "Note"]
    )
  );
  parts.push("");
  if (notes?.length) {
    parts.push("## Notes");
    for (const note of notes) {
      parts.push(`- ${note}`);
    }
    parts.push("");
  }
  return parts.join("\n");
}

function main() {
  const { rustPath, jsPath, outputPath } = parseArgs(process.argv.slice(2));

  const rust = loadJson(path.resolve(rustPath));
  const js = loadJson(path.resolve(jsPath));

  const rustSpeed = rust.speed ? normalizeRustSpeed(rust.speed) : [];
  const rustOutputs = rust.output_sizes ? normalizeRustOutputs(rust.output_sizes) : [];
  const rustBinary = rust.binary_sizes ? normalizeRustBinarySizes(rust.binary_sizes) : [];

  const jsSpeed = normalizeJsResults(
    js.results
      .filter((r) => r.dataset === "gradient" && r.duration_micros != null)
      .map((r) => ({
        ...r,
        output_bytes: r.output_bytes,
        status: r.status,
        note: r.note,
      }))
  );
  const jsOutputs = normalizeJsResults(
    js.results.filter((r) => r.dataset === "gradient" && r.output_bytes != null)
  );
  const jsBinary = normalizeJsBinarySizes(js.binary_sizes ?? []);

  const allSpeed = [...rustSpeed, ...jsSpeed];
  const allOutputs = [...rustOutputs, ...jsOutputs];
  const allBinary = [...rustBinary, ...jsBinary];

  const summary = {
    speedRankings: {
      png: rankSpeed(allSpeed, "png"),
      jpeg: rankSpeed(allSpeed, "jpeg"),
    },
    outputRankings: {
      png: rankOutputSizes(allOutputs, "png", "gradient"),
      jpeg: rankOutputSizes(allOutputs, "jpeg", "gradient"),
    },
    binaryRankings: rankBinarySizes(allBinary),
    notes: [
      ...(rust.notes ?? []),
      ...(js.notes ?? []),
      "JS results use quick iterations by default; set BENCH_ITERATIONS for longer runs.",
    ],
  };

  const markdown = buildMarkdown(summary);
  if (outputPath) {
    fs.writeFileSync(outputPath, markdown);
    console.log(`Wrote cross-language summary to ${outputPath}`);
  } else {
    console.log(markdown);
  }
}

main();
