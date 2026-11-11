---
name: xpu-kernels
description: Implement, build, validate, benchmark, or profile operators in this repository's GPU CUDA C++/Triton/TileLang and Ascend C backends.
---

# xpu-kernels project workflow

Use this skill for changes to operators and their correctness/performance workflow in this repository. Read the root `README.md` for supported environments and the installed public API before changing an operator.

## RMSNorm contract

- Public API: `xpu_kernels.rms_norm(x, weight, eps=1e-5, out=None, *, implementation=None)`. GPU defaults to CUDA C++; `implementation="triton"` selects Triton. NPU accepts default or `"ascendc"`.
- Input is contiguous float32, bfloat16, or float16. Normalize the last dimension after flattening leading dimensions into rows; `weight` is one-dimensional and matches the width.
- Match `xpu_kernels.golden.rms_norm_golden`: calculate mean-square normalization in float32, cast normalized values to input dtype, then multiply by weight and return input dtype.
- Device dispatch selects only the matching backend. Missing support is an error; never substitute the golden result for a device kernel.
- Preserve the current device and framework stream. Keep backend imports lazy so a GPU-only or NPU-only install loads only its own toolchain.

## Build and validation

From the repository root, install into the selected private project environment, then inspect its actual tools:

```bash
python -m pip install -e '.[gpu,dev]'   # RMSNorm GPU environment
python -m xpu_kernels.cli check-env --backend gpu
python -m xpu_kernels.cli build --backend gpu
python -m xpu_kernels.cli example --device cuda
python -m xpu_kernels.cli test --device cuda
```

GPU mHC adds TileLang: `python -m pip install -e '.[gpu,mhc-gpu,dev]'`.

For NPU, manually source the selected project CANN first. Do not install into a shared CANN/OPP or shared Python environment.

```bash
python -m pip install -e '.[dev]'
python -m xpu_kernels.cli check-env --backend npu
python -m xpu_kernels.cli build --backend npu
python -m xpu_kernels.cli example --device npu
python -m xpu_kernels.cli test --device npu
```

Correctness comes before performance. Compare each device implementation with its golden on matching shapes and dtypes. RMSNorm device tests cover aligned, tail and model-width cases across supported dtypes; its default benchmark matrix has 48 shapes and three dtypes. GPU RMSNorm compares CUDA C++, Triton and PyTorch composition; `--native-reference` adds native RMSNorm and switches every implementation to per-call output allocation. mHC GPU tests compare TileLang outputs with CPU FP64 samples; NPU tests cover native and optimized paths. Device results require executed cases on the requested device.

## Benchmark and profile

Run repeated measurements with matching dtype, shapes, device and workload. Compare against the saved xpu-kernels best only when software versions and measurement settings match. Golden is a correctness reference. Record native RMSNorm separately because its accumulation/cast order can differ. Do not mix cache cold start with steady state, or graph mode with eager mode. RMSNorm uses CUDA/NPU Events on the current framework stream. Default library measurements reuse preallocated `out`; with `--native-reference`, all implementations allocate outputs per call, and GPU composition creates its intermediate tensors per call. mHC GPU TileLang and both NPU implementations also allocate all four outputs per call. Records include hardware/software versions, shape, dtype, method, allocation policy, batches, per-batch samples and median latency under `results/benchmarks/` unless `--output` is supplied. Keep private toolchain and cache paths out of commit-ready records.

```bash
python -m xpu_kernels.cli bench --device npu --warmup 20 --batches 10 --iterations-per-batch 50
python -m xpu_kernels.cli bench --device cuda --warmup 20 --batches 10 --iterations-per-batch 50
```

Capture profiler traces independently, after correctness and matched repeated benchmarks. Use one tool per run; keep trace and temporary files in the project result area.

```bash
python -m xpu_kernels.cli profile --device npu --tool msprof --output results/profiles/npu-rmsnorm
python -m xpu_kernels.cli profile --device cuda --tool nsys --output results/profiles/cuda-rmsnorm
python -m xpu_kernels.cli profile --device cuda --tool ncu --output results/profiles/cuda-rmsnorm-ncu
python -m xpu_kernels.cli profile --device cuda --operator mhc_post_backward --tool ncu --implementation tilelang --shape 2x1024x4x6144 --dtype bfloat16 --output results/profiles/mhc-ncu
```

NCU profiles one matching launch of the selected CUDA C++/Triton/TileLang implementation with kernel replay, no cache or clock control, and `SpeedOfLight`/`LaunchStats` sections. msprof reads its own `--help` output and uses `--application` only when the installed version advertises it. RMSNorm profiler workload is packaged at `xpu_kernels.profile_workload`; mHC profiling routes through `xpu_kernels.mhc_workload`.

## DeepJIT and project paths

DeepJIT is the upstream header-only C++20 runtime in `third-party/deep-jit`; reuse its official compile/load/launch, device, and stream support. Keep Ascend C kernels in `kernels/include/xpu_kernels/` and the PyTorch host binding in `csrc/ascend/`. Its tracked kernel includes use canonical angle-bracket form and the `xpu_kernels/` prefix. Restart the Python process after changing a JIT-included header.

Builds and caches belong under `build/`, `cache/`, and `tmp/`. Keep benchmark JSONL under `results/benchmarks/` and profiler products under `results/profiles/`; Git tracks only the compact benchmark records and their README. `runs/`, `artifacts/`, `.runtime/`, `envs/`, and `tools/remote/` remain ignored/private.

GPU tables use shape rows and hardware-named implementation columns; write “单算子拼接” for primitive composition. NPU native comparisons use `Shape | 原生 | 优化`, with the actual hardware model above the table. Keep dtype, unit, output-allocation policy and measurement method together with each table. Do not add shape-source commentary. Preserve per-implementation raw rows and legacy provenance.

## Homepage performance highlights

Keep the root README focused on library purpose, installation and a minimal API example, concise operator entries, two performance figures and one complete guide link per operator. Feature only implemented, device-validated operators. The homepage figures are `docs/assets/benchmark-rmsnorm.png` and `docs/assets/benchmark-mhc.png`; their axes and units are labeled in the figures. Use grouped bars of absolute latency, zero-based linear axes and hardware/implementation labels. Compare GPU and NPU using matching operator, shape, dtype and input distribution; each device creates its own input tensors. Give each selected shape its own panel, retaining the implementation series that are faster and slower. Keep kernel duration and Event call intervals in separate comparisons. Keep measurement methods, allocation policies, environment versions, raw-data links and detailed result links in the operator/performance guides; the README should not link JSONL, CSV or chart manifests, or repeat sampling conditions and profiler settings. Full latency tables stay in the detailed documents.

`tools/render_homepage_performance.py` updates the README performance block from the committed JSONL; use `--check` to verify it. `tools/render_homepage_charts.py` renders the chart PNGs from those same JSONL records and saves their exact values in `docs/assets/performance-chart-data.json`; install optional plotting dependencies from `tools/requirements-viz.txt`. `tools/render_native_comparisons.py` generates homepage content through `homepage()`; keep that output aligned with the clean README format. Refresh charts and homepage alongside the full-table renderers whenever published measurements change. Keep setup, measurement methods and profiler details in `docs/getting-started.md`, `docs/benchmarking.md` and the detailed result documents. Compute speedups from matching raw rows only in detailed documents; preserve dtype, device, timing method and workload settings.

## mHC post backward

`mhc_post_backward(grad_y, x, h_res, h_out, h_post, out=None, *, implementation="auto")` returns four gradients. Its independent CPU FP64 golden is `xpu_kernels.mhc_post_backward.mhc_post_backward_golden`, also checked against PyTorch autograd.

The shared API accepts contiguous TND/BSND input with four residual streams, FP16/BF16 data, FP32 mixing coefficients, 512..49152 tokens and D<=8192. On CUDA, the DeepSeek TileKernels TileLang implementation supports BF16 and D divisible by 128. `auto`, `optimized` and `tilelang` select that GPU implementation. On Ascend 950, the optimized kernel applies to BF16, 2048..49152 tokens, N=4, h_res present and D in {2560,5120,6144,7168}. NPU `auto` selects that kernel inside the range and CANN mainline outside it; `native` selects `torch_npu.npu_mhc_post_backward` / `aclnnMhcPostBackward`. See `docs/operators/mhc-post-backward.md`.

```bash
python -m pip install -e '.[gpu,mhc-gpu,dev]'
python -m xpu_kernels.cli build --backend gpu --operator mhc_post_backward
python -m xpu_kernels.cli test --device cuda --operator mhc_post_backward
python -m xpu_kernels.cli bench --device cuda --operator mhc_post_backward --implementations tilelang --warmup 20 --batches 10 --iterations-per-batch 50
python -m xpu_kernels.cli profile --device cuda --operator mhc_post_backward --implementation tilelang --shape 2x1024x4x6144 --tool ncu

python -m xpu_kernels.cli build --backend npu --operator mhc_post_backward
python -m xpu_kernels.cli example --device npu --operator mhc_post_backward --shape 2x1024x4x6144
python -m xpu_kernels.cli test --device npu --operator mhc_post_backward
python -m xpu_kernels.cli bench --device npu --operator mhc_post_backward --implementations native,optimized --warmup 20 --batches 10 --iterations-per-batch 50
python -m xpu_kernels.cli profile --device npu --operator mhc_post_backward --implementation optimized --shape 2x1024x4x6144 --tool msprof
```

NPU benchmarks compare CANN native and optimized with matching shape, dtype, generated input distribution, stream, per-call output allocation and sampling method. The GPU and NPU matrices use the same 16 shape/dtype cases; each device generates its own input tensors. Check the installed CANN mainline route with private custom-op overlays disabled; a private predecessor is not the native baseline. NPU tables use `Shape | 原生 | 优化`; cross-device tables use hardware-named implementation columns and keep Event call time distinct from kernel duration. State hardware model, dtype, unit, allocation policy and method with each table. Keep historical experiment names only in raw provenance records and retain slowdowns in published comparisons.

For kernel-only NPU comparisons use official msprof `Task Duration(us)` and retain per-launch CSV samples with extraction rules. GPU mHC NCU captures `mhc_post_bwd_cuda_kernel` with `gpu__time_duration.sum`; keep these kernel durations separate from CUDA Event call intervals. Reuse results only when native route, device, implementation and measurement conditions match. Preserve the CANN license with the NPU source and cost model, and the vendored TileKernels MIT license with its GPU source.
