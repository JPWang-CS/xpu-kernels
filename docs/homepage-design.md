# Homepage structure and charts

The README follows an engineering-library reading order: scope and supported kernels, measured comparisons, a minimal executable quick start, then documentation. Environment diagnostics, compiler details and profiler commands live in dedicated documents.

References reviewed:

- [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM): concise library scope, clear requirements and implementation interfaces.
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer): direct installation and executable API examples, with comprehensive documentation linked separately.
- [TileLang](https://github.com/tile-ai/tilelang): benchmark figures grouped by workload and hardware, backed by detailed benchmark sources.

The homepage uses the project's own artwork and measured data. The masthead is flat and compact; charts share a light background, restrained typography and three stable series colors. GPU and NPU have independent comparisons. Gray denotes the baseline, teal xpu-kernels and violet Triton.

Each graph shows speedup relative to its own same-shape baseline. The four RMSNorm shapes include a small aligned input, an unaligned tail case, a larger row count and a large working set. The four mHC shapes span token counts and feature widths within the optimized dispatch range. Full matrices remain available through links under each graph.

Raw latencies are not edited during plotting. `tools/render_homepage_charts.py` loads the benchmark JSONL, verifies the selected records and computes every bar. `--check` compares the saved plot-data manifest to current measurements. Rebuild instructions are in [Benchmark and Profiler](benchmarking.md).
