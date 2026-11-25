# Visual assets

## Brand masthead

`xpu-kernels-masthead.png` was generated with the built-in ImageGen tool. Original resolution: 2172x724; README display width:640. The wordmark, platform labels and abstract matrix-tile motif were visually checked. The artwork contains no performance data or vendor marks.

Prompt: A restrained academic/open-source kernel library masthead, near-white background #FAFCFE, crisp flat editorial art, no shadows, cards, 3D chips or neon. Exact wordmark “xpu-kernels”; subtitle “GPU / NPU Kernel Library”; line “CUDA · Triton · Ascend C”. An abstract X from square matrix tiles and thin grids at the right, using teal #0D9488, violet #7C3AED and slate #94A3B8. Generous whitespace, precise alignment and a thin bottom divider.

## Performance charts

`tools/render_homepage_charts.py` prepares the selected shapes, labels, hardware, units and values in `performance-chart-data.json` from committed benchmark JSONL. Its `--check` mode checks those values and saved assets. New artwork follows the project's ImageGen workflow and is visually checked against the manifest.

`benchmark-rmsnorm.png` was generated with the built-in ImageGen tool at 1828×860. It uses two horizontal-bar panels with a shared 0..240 μs axis, gray native and teal optimized bars. Ascend 950DT BF16 Event call times are 83.16/48.53 μs for 2048×20480 and 203.16/106.04 μs for 4096×20480. Both implementations allocate Y/rstd per call. The labels, common zero baseline, bar proportions and units were visually checked. The production prompt is saved in [benchmark-rmsnorm.prompt.txt](benchmark-rmsnorm.prompt.txt). Historical GPU values use the prior numerical contract and are retained in detailed historical tables.

The existing mHC figure retains its original four-panel artwork and exact source data. It shows GPU TileLang NCU kernel duration alongside NPU CANN native and optimized msprof `Task Duration(us)`. It compares the same four selected shapes, dtype and input distribution on each device; each device generates its own inputs. Event call intervals and kernel duration remain distinct metrics.

[Rendering and validation](../benchmarking.md#5-记录与图表) · [Homepage structure and chart design](../homepage-design.md)
