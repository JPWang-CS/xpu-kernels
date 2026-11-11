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

Correctness comes before performance. Run the same shapes and dtypes on the kernel and golden. The built-in suite has 48 aligned/tail/model-width cases; the default GPU benchmark measures cuda, triton and torch composition, and NPU measures ascendc only. A device test is accepted only if the requested device is available and the relevant tests execute; all-skipped is not a device pass.

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

Performance tables use shape rows with columns `NPU 950DT`, `GPU L20X`, `GPU L20X（Triton）`, `GPU L20X（单算子）`. Omit AscendC/CUDA suffixes and shape-source explanations from the table. Record dtype, device model, unit and measurement method once in table notes. The single-operator column means GPU composition of primitive PyTorch operations, not a framework RMSNorm API. Keep raw rows distinguished by operator and implementation; legacy GPU v1 rows identify Triton, not CUDA C++.
