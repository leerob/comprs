#!/usr/bin/env bash
set -euo pipefail

RUST_OUT="${RUST_OUT:-/tmp/rust-summary.json}"
JS_OUT="${JS_OUT:-/tmp/js-bench.json}"
SUMMARY_OUT="${SUMMARY_OUT:-/tmp/cross-bench.md}"
SUMMARY_JSON_OUT="${SUMMARY_JSON_OUT:-}"

echo "== Rust summary =="
cargo bench --bench comparison -- --summary-only --export-json "${RUST_OUT}"

echo "== JS bench (quick) =="
(
  cd benchmarks/js
  QUICK=1 node run.mjs --output "${JS_OUT}"
)

echo "== Aggregate =="
if [ -n "${SUMMARY_JSON_OUT}" ]; then
  node benchmarks/aggregate.mjs --rust "${RUST_OUT}" --js "${JS_OUT}" --output "${SUMMARY_OUT}" --json-out "${SUMMARY_JSON_OUT}"
else
  node benchmarks/aggregate.mjs --rust "${RUST_OUT}" --js "${JS_OUT}" --output "${SUMMARY_OUT}"
fi

echo "Done."
echo "Rust summary:    ${RUST_OUT}"
echo "JS summary:      ${JS_OUT}"
echo "Cross summary:   ${SUMMARY_OUT}"
if [ -n "${SUMMARY_JSON_OUT}" ]; then
  echo "Cross summary JSON: ${SUMMARY_JSON_OUT}"
fi
