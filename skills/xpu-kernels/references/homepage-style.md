# Homepage structure and charts

The README introduces xpu-kernels as a GPU/NPU operator library, then moves through installation and a minimal API example, operator descriptions and their detailed guides, measured performance grouped by operator, and reproduction instructions. Environment setup and operator-specific input constraints stay in linked documents.

The homepage uses a compact masthead and three performance figures with distinct measurement scopes. The artwork and chart renderers keep product identity separate from measured values. All chart values come from benchmark JSONL records; the source data remains unchanged during rendering.

## Performance figures

### RMSNorm

`docs/assets/benchmark-rmsnorm.png` compares CANN native and optimized RMSNorm on Ascend 950DT. It reports per-call Event latency for BF16 2048×20480 and 4096×20480; both implementations allocate Y/rstd per call. The current figure uses accepted current-contract measurements. Historical GPU comparisons remain in the detailed performance documents under their original numerical convention.

### mHC Post Backward

`docs/assets/benchmark-mhc.png` compares NPU CANN native and xpu-kernels optimized kernel duration with GPU TileLang kernel duration. NPU values use msprof `Task Duration(us)`; GPU values use Nsight Compute `gpu__time_duration.sum`. The charts use the same four shapes, dtype and input distribution on each device, with independently generated device inputs.

All figures use the same Python/Matplotlib renderer and `STYLE` configuration: identical fonts, colors, panel widths, two-column layout, zero-based linear axes, units, legends and two-decimal labels. RMSNorm has two shape panels and mHC has four; additional panels extend the figure vertically. Each panel has its own scale, stated in the footer. Event call intervals and kernel durations are labeled separately. Exact values and style are saved in `docs/assets/performance-chart-data.json` and each PNG's metadata. The renderer's `--check` mode verifies both against current records. New operators add data definitions and reuse this same plotting function.

### Fixed-capacity MoE path

`docs/assets/benchmark-moe.png` shows four selected historical deployment
comparisons from `moe-fixed-capacity-historical.jsonl`. Values are absolute
seconds for the slower rank's whole-model device business interval, one formal
window per arm, on two Ascend 910B3 cards. The candidate combines communication,
three fused kernels, a dedicated EP group and graph adaptation. This chart is
The chart shows whole-model device execution intervals; individual-kernel and
migrated JIT measurements use their own records. The full validation record
stays in the operator design document.

The shared renderer supports chart-specific units and labels while retaining
one style. `tools/render_native_comparisons.py:homepage` also includes this chart,
so `render_homepage_performance.py` preserves it during future README updates.

## Reading order and references

The README keeps implementation-language details in the operator guides and presents each operator by its purpose and usage. Performance summaries follow the operator descriptions, with installation and measurement steps linked from the reproduction section.

References reviewed:

- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM): concise library scope, clear requirements and implementation interfaces.
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer): direct installation and executable API examples, with detailed documentation linked separately.
- [TileLang](https://github.com/tile-ai/tilelang): benchmark figures grouped by workload and hardware, backed by detailed benchmark sources.

See [visual asset notes](visual-assets.md) for the masthead and chart production record, and [Benchmark and Profiler](../../../docs/benchmarking.md) for reproduction commands.
