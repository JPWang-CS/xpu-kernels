---
name: xpu-kernels
description: Implement, build, validate, benchmark, or profile operators in this repository's GPU CUDA C++/Triton and Ascend C backends.
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
python -m pip install -e '.[gpu,dev]'   # GPU environment
python -m xpu_kernels.cli check-env --backend gpu
python -m xpu_kernels.cli build --backend gpu
python -m xpu_kernels.cli example --device cuda
python -m xpu_kernels.cli test --device cuda
```

For NPU, manually source the selected project CANN first. Do not install into a shared CANN/OPP or shared Python environment.

```bash
python -m pip install -e '.[dev]'
python -m xpu_kernels.cli check-env --backend npu
python -m xpu_kernels.cli build --backend npu
python -m xpu_kernels.cli example --device npu
python -m xpu_kernels.cli test --device npu
```

Correctness comes before performance. Run the same shapes and dtypes on the kernel and golden. The built-in suite has 48 aligned/tail/model-width cases; the default GPU benchmark measures cuda, triton and torch composition, and NPU measures Ascend C plus CANN native when --native-reference is enabled. A device test is accepted only if the requested device is available and the relevant tests execute; all-skipped is not a device pass.

## Benchmark and profile

Run repeated measurements with matching dtype, shapes, device, and workload. Compare against the current saved xpu-kernels best when one exists, requiring matching software versions. Golden is only a correctness reference. When available, measure PyTorch/torch_npu native RMSNorm separately and record that its accumulation/cast order may differ. Do not mix cache cold start with steady state, or graph mode with eager mode. Benchmark uses CUDA/NPU Events on the current framework stream, preallocates `out`, and writes hardware/software versions, shape, dtype, event method, batches, per-batch samples, and median latency to `results/benchmarks/` unless `--output` is supplied. Keep private toolchain and cache paths out of commit-ready records.

```bash
python -m xpu_kernels.cli bench --device npu --warmup 20 --batches 10 --iterations-per-batch 50
python -m xpu_kernels.cli bench --device cuda --warmup 20 --batches 10 --iterations-per-batch 50
```

Capture profiler traces independently, after correctness and matched repeated benchmarks. Use one tool per run; keep trace and temporary files in the project result area.

```bash
python -m xpu_kernels.cli profile --device npu --tool msprof --output results/profiles/npu-rmsnorm
python -m xpu_kernels.cli profile --device cuda --tool nsys --output results/profiles/cuda-rmsnorm
python -m xpu_kernels.cli profile --device cuda --tool ncu --output results/profiles/cuda-rmsnorm-ncu
```

NCU profiles one matching launch of the selected CUDA C++/Triton implementation with kernel replay, no cache or clock control, and only `SpeedOfLight`/`LaunchStats` sections. msprof reads its own `--help` output and uses `--application` only when the installed version advertises it. The profiler workload module is packaged at `xpu_kernels.profile_workload`, so the command works after wheel installation.

## DeepJIT and project paths

DeepJIT is the upstream header-only C++20 runtime in `third-party/deep-jit`; reuse its official compile/load/launch, device, and stream support. Keep Ascend C kernels in `kernels/include/xpu_kernels/` and the PyTorch host binding in `csrc/ascend/`. Its tracked kernel includes use canonical angle-bracket form and the `xpu_kernels/` prefix. Restart the Python process after changing a JIT-included header.

Builds and caches belong under `build/`, `cache/`, and `tmp/`. Keep benchmark JSONL under `results/benchmarks/` and profiler products under `results/profiles/`; Git tracks only the compact benchmark records and their README. `runs/`, `artifacts/`, `.runtime/`, `envs/`, and `tools/remote/` remain ignored/private.

GPU tables use shape rows and hardware-named implementation columns; write “单算子拼接” for primitive composition. NPU native comparisons use `Shape | 原生 | 优化`, with the actual hardware model above the table. Keep dtype, unit, output-allocation policy and measurement method together with each table. Do not add shape-source commentary. Preserve per-implementation raw rows and legacy provenance.

## Homepage performance highlights

Keep the root README focused on supported operators, a compact measured comparison, quick start and the Python API. Put long environment/profiler notes in collapsible sections and link the full performance matrix. Feature only implemented, device-validated operators. Select aligned and tail shapes, small and large workloads; retain cases where another implementation is faster. Bold the lowest measured time among implementations on the same GPU, keeping NPU timings separate from this ranking.

`tools/render_homepage_performance.py` updates the README performance block from the committed JSONL; use `--check` to verify it. Refresh it alongside `tools/render_rmsnorm_table.py` whenever published measurements change. Compute speedups from the matching raw rows; preserve dtype, device, timing method and workload settings.

## mHC post backward

`mhc_post_backward(grad_y, x, h_res, h_out, h_post, out=None, *, implementation="auto")` returns four gradients. Its independent CPU FP64 golden is `xpu_kernels.mhc_post_backward.mhc_post_backward_golden`, also checked against PyTorch autograd.

The current API accepts contiguous TND/BSND input with four residual streams, FP16/BF16 data, FP32 mixing coefficients, 512..49152 tokens and D<=8192. The final optimized Ascend 950 kernel applies to BF16, 2048..49152 tokens, N=4, h_res present and D in {2560,5120,6144,7168}. Auto uses that kernel inside the range and CANN mainline outside it; explicit optimized rejects unsupported inputs. Native calls `torch_npu.npu_mhc_post_backward` / `aclnnMhcPostBackward`. See `docs/operators/mhc-post-backward.md`.

```bash
python -m xpu_kernels.cli build --backend npu --operator mhc_post_backward
python -m xpu_kernels.cli example --device npu --operator mhc_post_backward --shape 2x1024x4x6144
python -m xpu_kernels.cli test --device npu --operator mhc_post_backward
python -m xpu_kernels.cli bench --device npu --operator mhc_post_backward --implementations native,optimized --warmup 20 --batches 10 --iterations-per-batch 50
python -m xpu_kernels.cli profile --device npu --operator mhc_post_backward --implementation optimized --shape 2x1024x4x6144 --tool msprof
```

Benchmark both native and optimized with the same input, stream, output allocation policy and sampling method. Check the installed CANN mainline route with private custom-op overlays disabled; a private predecessor is not the native baseline. Public comparison tables use `Shape | 原生 | 优化`; state hardware model, dtype, unit and method once above each table. Keep historical experiment names only in raw provenance records. Compare within a device, retain slowdowns, and restrict the public MHC matrix to the optimized range.

For kernel-only NPU comparisons use official msprof `Task Duration(us)` and retain the per-launch CSV samples and extraction rules. Keep Event stream intervals in a separate table; they can include CPU submission gaps. Reuse existing results only when native route, device, implementation and measurement conditions match. Validate the installed wheel on an actual device after migration. Preserve the CANN license bundled with the NPU source and cost model.
