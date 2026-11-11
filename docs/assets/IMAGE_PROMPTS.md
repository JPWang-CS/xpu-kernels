# Visual assets

## Brand masthead

`xpu-kernels-masthead.png` was generated with the built-in ImageGen tool. Original resolution: 2172x724; README display width:640. The wordmark, platform labels and abstract matrix-tile motif were visually checked. The artwork contains no performance data or vendor marks.

Prompt: A restrained academic/open-source kernel library masthead, near-white background #FAFCFE, crisp flat editorial art, no shadows, cards, 3D chips or neon. Exact wordmark “xpu-kernels”; subtitle “GPU / NPU Kernel Library”; line “CUDA · Triton · Ascend C”. An abstract X from square matrix tiles and thin grids at the right, using teal #0D9488, violet #7C3AED and slate #94A3B8. Generous whitespace, precise alignment and a thin bottom divider.

## Performance plots

`benchmark-mhc.png`, `benchmark-rms-gpu.png`, and `benchmark-rms-npu.png` are exact-data charts rendered by Matplotlib via `tools/render_homepage_charts.py`. They replace the previous numeric-card illustrations. Plot values, hardware, metric, labels, colors and selected shapes are stored in `performance-chart-data.json` and checked against the original benchmark JSONL.

Each plot uses a linear zero-based axis, fixed per-series colors and a visible1x reference. Bar height is baseline latency divided by implementation latency. Labels round to two decimals while bar geometry uses the full ratio. The NPU RMSNorm plot retains cases below1x. Detailed documents retain absolute timings and measurement conditions.

[Rendering and validation](../benchmarking.md#更新首页图表) · [Design references](../homepage-design.md)
