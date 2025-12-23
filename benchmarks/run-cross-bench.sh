#!/usr/bin/env bash
set -euo pipefail

RUST_OUT="${RUST_OUT:-/tmp/rust-summary.json}"
JS_OUT="${JS_OUT:-/tmp/js-bench.json}"
SUMMARY_OUT="${SUMMARY_OUT:-/tmp/cross-bench.md}"

echo "== Rust summary =="
cargo bench --bench comparison -- --summary-only --export-json "${RUST_OUT}"

echo "== JS bench (quick) =="
(
  cd benchmarks/js
  QUICK=1 node run.mjs --output "${JS_OUT}"
)

echo "== Aggregate =="
node benchmarks/aggregate.mjs --rust "${RUST_OUT}" --js "${JS_OUT}" --output "${SUMMARY_OUT}"

echo "Done."
echo "Rust summary:    ${RUST_OUT}"
echo "JS summary:      ${JS_OUT}"
echo "Cross summary:   ${SUMMARY_OUT}"
