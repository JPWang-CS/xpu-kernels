# Homepage structure and charts

The README introduces xpu-kernels as a GPU/NPU operator library, then moves through installation and a minimal API example, operator descriptions and their detailed guides, measured performance grouped by operator, and reproduction instructions. Environment setup and operator-specific input constraints stay in linked documents.

The homepage uses a compact masthead and two combined performance figures. The artwork and chart renderers keep product identity separate from measured values. All chart values come from benchmark JSONL records; the source data remains unchanged during rendering.

## Performance figures

### RMSNorm

`docs/assets/benchmark-rmsnorm.png` compares NPU native and xpu-kernels with GPU CUDA, Triton and single-op composition. It reports per-call Event latency in microseconds. Every implementation allocates its output per call; the GPU composition also creates its intermediate tensors per call. The GPU native RMSNorm measurements remain in the full 192-record dataset for consistent allocation-policy comparisons.

### mHC Post Backward

`docs/assets/benchmark-mhc.png` compares NPU CANN native and xpu-kernels optimized kernel duration with GPU TileLang kernel duration. NPU values use msprof `Task Duration(us)`; GPU values use Nsight Compute `gpu__time_duration.sum`. The charts use the same four shapes, dtype and input distribution on each device, with independently generated device inputs.

Both figures use absolute microseconds and linear axes starting at zero. Each is a 2×2 layout with one selected shape per panel; every panel compares the corresponding implementations across both devices. RMSNorm Event call intervals and mHC kernel duration appear in separate figures because they measure different scopes. Each figure's exact selections and values are recorded in `docs/assets/performance-chart-data.json`; the renderer's `--check` mode verifies the manifest against current benchmark records.

## Reading order and references

The README keeps implementation-language details in the operator guides and presents each operator by its purpose and usage. Performance summaries follow the operator descriptions, with installation and measurement steps linked from the reproduction section.

References reviewed:

- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM): concise library scope, clear requirements and implementation interfaces.
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer): direct installation and executable API examples, with detailed documentation linked separately.
- [TileLang](https://github.com/tile-ai/tilelang): benchmark figures grouped by workload and hardware, backed by detailed benchmark sources.

See [visual asset notes](assets/IMAGE_PROMPTS.md) for the masthead and chart production record, and [Benchmark and Profiler](benchmarking.md) for reproduction commands.
