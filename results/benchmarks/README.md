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

## GPU mainline records

`rmsnorm-gpu-mainline.jsonl`: BF16, 48 shapes and four implementations (`cuda`, `triton`, `torch` composition, and PyTorch `native`), 192 schema 4 records. Run with `--native-reference`; all four implementations allocate their output per call, matching the NPU native comparison policy. The GPU composition also creates its intermediate tensors per call. All implementation correctness checks passed.

`mhc-post-backward-gpu-tilelang.jsonl`: BF16, the same 16 shape/dtype/distribution cases as the NPU matrix, one TileLang implementation and 16 Event records. Each device creates its own input tensors. Each record includes six-token FP64 golden samples, output allocation per call, 20 warmups, 10 batches and 50 calls per batch; all correctness checks passed.

`mhc-post-backward-gpu-ncu.jsonl`: 16 per-shape NCU summaries for `mhc_post_bwd_cuda_kernel`, using kernel replay and `gpu__time_duration.sum`; each summary contains five measured samples after 20 warmups. [mhc-gpu-tilelang.csv](../../docs/performance/data/mhc-gpu-tilelang.csv) preserves all 400 warmup and measured launches across the 16 shapes. The NCU correctness samples and matching Event results passed.

The combined absolute-latency figures are [RMSNorm](../../docs/assets/benchmark-rmsnorm.png) and [mHC Post Backward](../../docs/assets/benchmark-mhc.png). Their independent panels compare matching shape, dtype and input distribution on each device; each device generates its own random inputs. The [RMSNorm table](../../docs/performance/README.md) and [mHC table](../../docs/performance/mhc-post-backward.md) contain the full cross-device matrices. Historical benchmark JSONL files above remain unchanged.
