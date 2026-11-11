# Visual assets

## Brand masthead

`xpu-kernels-masthead.png` was generated with the built-in ImageGen tool. Original resolution: 2172x724; README display width:640. The wordmark, platform labels and abstract matrix-tile motif were visually checked. The artwork contains no performance data or vendor marks.

Prompt: A restrained academic/open-source kernel library masthead, near-white background #FAFCFE, crisp flat editorial art, no shadows, cards, 3D chips or neon. Exact wordmark “xpu-kernels”; subtitle “GPU / NPU Kernel Library”; line “CUDA · Triton · Ascend C”. An abstract X from square matrix tiles and thin grids at the right, using teal #0D9488, violet #7C3AED and slate #94A3B8. Generous whitespace, precise alignment and a thin bottom divider.

## Performance charts

`benchmark-rmsnorm.png` and `benchmark-mhc.png` are the combined, exact-data figures rendered by `tools/render_homepage_charts.py`. The renderer reads committed benchmark JSONL and stores selected shapes, labels, hardware, units and values in `performance-chart-data.json` for `--check` validation.

Both figures show absolute latency in microseconds on linear axes starting at zero. Each figure has four independent shape panels; every panel compares the selected implementations on both devices for that shape. The RMSNorm figure uses per-call CUDA/NPU Event measurements with output allocation on every call. It shows NPU native and xpu-kernels implementations, plus GPU CUDA, Triton and single-op composition. The GPU RMSNorm dataset also includes PyTorch native measurements for the full 48-shape, four-implementation record set.

The mHC figure shows GPU TileLang NCU kernel duration alongside NPU CANN native and optimized msprof `Task Duration(us)`. It compares the same four selected shapes, dtype and input distribution on each device; each device generates its own inputs. Event call intervals and kernel duration remain distinct metrics.

[Rendering and validation](../benchmarking.md#5-记录与图表) · [Homepage structure and chart design](../homepage-design.md)
