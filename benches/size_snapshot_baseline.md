# Size snapshot baseline (local)

Command:
```
cargo bench --bench size_snapshot -- --nocapture
```

Environment: x86_64 (release bench profile), Q=85 for JPEG.

Output (post 4:2:0 fix and optimized Huffman option):
```
==> gradient 256x256
PNG comprs: 6.135 ms, 2175 bytes
PNG comprs fast: 4.404 ms, 2483 bytes
PNG comprs max: 8.012 ms, 1778 bytes
PNG image:  2.838 ms, 35214 bytes
JPEG comprs q85 444: 2.490 ms, 6485 bytes
JPEG comprs q85 444 opt: 5.044 ms, 4615 bytes
JPEG comprs fast:   1.504 ms, 3592 bytes
JPEG comprs q85 420: 1.496 ms, 3998 bytes
JPEG image q85:      3.214 ms, 6403 bytes

==> noisy 256x256
PNG comprs: 12.439 ms, 195632 bytes
PNG comprs fast: 11.387 ms, 191989 bytes
PNG comprs max: 13.011 ms, 195632 bytes
PNG image:  3.923 ms, 196947 bytes
JPEG comprs q85 444: 6.172 ms, 104372 bytes
JPEG comprs q85 444 opt: 9.934 ms, 98334 bytes
JPEG comprs fast:   3.751 ms, 39944 bytes
JPEG comprs q85 420: 3.251 ms, 50144 bytes
JPEG image q85:      7.880 ms, 104302 bytes
```

Use this as a rough reference; rerun on your hardware to track regressions.```

WASM size (local, release, feature=wasm, raw/gz):
- Raw: 90,473 bytes
- Gzipped: 30,065 bytes
(wasm-opt not available in this environment)
