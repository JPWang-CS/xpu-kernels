# Benchmark records

Each `xpu-kernels bench` run writes one JSONL file here. Keep the raw records with the hardware name, device index, shape, dtype, software versions, Event timing method, warmup/batch parameters, per-batch samples, and matching historical best. Golden is recorded as a correctness reference only. Native PyTorch/torch_npu RMSNorm is optional (`--native-reference`).

Schema 3 identifies `operator` and `implementation`, and records correctness status, maximum absolute error and tolerances. GPU implementations are `cuda`, `triton` and `torch` (GPU primitive composition); NPU uses `ascendc`. Historical best matches operator, implementation, device, hardware, shape, dtype, software and timing settings. Legacy schema 2 GPU records represent Triton; NPU records represent Ascend C.

The v2 files contain 48 shapes and three dtypes: 432 GPU rows and 144 NPU rows. All 576 rows passed the independent golden check. The [performance table](../../docs/performance/README.md) is generated directly from these files. The v1 records remain unchanged.

Only sanitized, compact JSONL and this schema note are intended for Git. Profiler traces, compiler output, and temporary build files remain ignored.

## CANN mainline comparisons

`rmsnorm-npu-mainline.jsonl`: 48 shapes x 3 dtypes x 2 implementations, 288 records. Schema4 records each implementation independently, including output allocation policy. NPU native and Ascend C both allocate outputs per call. Each uses its own documented numerical reference because Pangu and CANN RMSNorm differ in intermediate casting. Historical comparisons must match output allocation and timing settings as well as software/device identity.

`mhc-post-backward-npu-mainline.jsonl`: 16 BF16 TND/BSND shapes, CANN mainline native and final optimized, 32 Event records. Both allocate four outputs per call. Correctness samples boundary/interior tokens with independent CPU FP64 equations; representative full-output tests also run on the device.

`mhc-post-backward-msprof.jsonl`: the same 16 shapes and two implementations, official msprof MIX_AIC Task Duration(us). Each row retains five post-warmup samples and links a compact public CSV preserving all25 launches, Task IDs, stream, shape and source row. Public performance tables use these kernel medians.

The historical `mhc-post-backward-history.jsonl` and `mhc-post-backward-a1b-paired.jsonl` describe earlier private implementations; their predecessor columns are not the CANN mainline baseline. The initial migration Event file is retained as a separate installation/run record.
