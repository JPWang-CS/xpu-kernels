# Homepage visual assets

Generated with the built-in ImageGen tool. Visual assets use the project’s own title and abstract matrix-tile illustration; no vendor logos.

## xpu-kernels-hero.png

- Original PNG: 2172 × 724.
- Use: GitHub README hero, 3:1 landscape, full article width.
- Prompt: Premium open-source engineering banner, deep ink navy background, white exact text “xpu-kernels”, “GPU / NPU Kernel Library”, “CUDA · Triton · Ascend C”. Left-aligned typography and restrained cyan/jade matrix tile assembly on the right, subtle amber accents, generous spacing. No speedup claims or vendor imagery.
- Check: all three text lines verified against the rendered image; no data is encoded in the illustration.

## RMSNorm GPU performance card

File: `rmsnorm-performance.png`. Built-in ImageGen; numeric cards, navy/teal/blue on pale white background. Source: `results/benchmarks/rmsnorm-gpu-v2.jsonl`, BF16, GPU L20X. 128x4096: CUDA 7.011, Triton 16.201, composition 36.486 us; 4096x8192: CUDA 58.522, Triton 37.066, composition 493.193 us. Ratios 5.20x and 13.31x against GPU primitive composition. Event/Eager, warmup20,10x50. Verified every displayed digit and implementation label.

## mHC CANN mainline comparison

File: `mhc-native-performance.png`. Built-in ImageGen. Prompt: premium wide 2:1 engineering comparison, white/pale blue, navy sans-serif Chinese text, two aligned rounded numeric cards, teal highlights, no bar chart. Title mHC Post Backward; subtitle NPU 950DT / BF16 / us per call. Shape 2x1024x4x6144: native295.395, optimized89.943, 3.28x. Shape32768x4x6144: native4678.831, optimized1736.369,2.69x. Footer CANN mainline / msprof Task Duration / warmup20 / median5samples. No experiment-version labels.

Source: `results/benchmarks/mhc-post-backward-msprof.jsonl`. Native is CANN aclnnMhcPostBackward, optimized is the final xpu-kernels implementation. Every digit, shape, Chinese label, unit, sampling method and ratio was visually checked against the raw samples.
