# Benchmark records

Each `xpu-kernels bench` run writes one JSONL file here. Keep the raw records with the hardware name, device index, shape, dtype, software versions, Event timing method, warmup/batch parameters, per-batch samples, and matching historical best. Golden is recorded as a correctness reference only. Native PyTorch/torch_npu RMSNorm is optional (`--native-reference`).

Schema 3 identifies `operator` and `implementation`, and records correctness status, maximum absolute error and tolerances. GPU implementations are `cuda`, `triton` and `torch` (GPU primitive composition); NPU uses `ascendc`. Historical best matches operator, implementation, device, hardware, shape, dtype, software and timing settings. Legacy schema 2 GPU records represent Triton; NPU records represent Ascend C.

The v2 files contain 48 shapes and three dtypes: 432 GPU rows and 144 NPU rows. All 576 rows passed the independent golden check. The [performance table](../../docs/performance/README.md) is generated directly from these files. The v1 records remain unchanged.

Only sanitized, compact JSONL and this schema note are intended for Git. Profiler traces, compiler output, and temporary build files remain ignored.
