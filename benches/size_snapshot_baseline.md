# Size snapshot baseline (local)

Command:
```
cargo bench --bench size_snapshot -- --nocapture
```

Environment: x86_64 (release bench profile), Q=85 for JPEG.

Output (post 4:2:0 fix):
```
==> gradient 256x256
PNG comprs: 6.278 ms, 2175 bytes
PNG comprs fast: 4.389 ms, 2483 bytes
PNG comprs max: 8.514 ms, 1778 bytes
PNG image:  2.804 ms, 35214 bytes
JPEG comprs q85 444: 2.459 ms, 6485 bytes
JPEG comprs fast:   1.466 ms, 3592 bytes
JPEG comprs q85 420: 1.461 ms, 3998 bytes
JPEG image q85:      3.269 ms, 6403 bytes

==> noisy 256x256
PNG comprs: 12.377 ms, 195632 bytes
PNG comprs fast: 11.439 ms, 191989 bytes
PNG comprs max: 13.137 ms, 195632 bytes
PNG image:  3.836 ms, 196947 bytes
JPEG comprs q85 444: 6.110 ms, 104372 bytes
JPEG comprs fast:   2.942 ms, 39944 bytes
JPEG comprs q85 420: 3.153 ms, 50144 bytes
JPEG image q85:      7.630 ms, 104302 bytes
```

Use this as a rough reference; rerun on your hardware to track regressions.```
