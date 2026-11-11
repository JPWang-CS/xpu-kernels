# Benchmark 与 Profiler

正确性通过后再测量性能。RMSNorm 默认覆盖 48 个 shape 和三种精度；mHC 默认对比 CANN 原生与优化实现，覆盖优化范围内的 16 个 shape。

```bash
python -m xpu_kernels.cli test --device cuda
python -m xpu_kernels.cli bench --device cuda \
  --warmup 20 --batches 10 --iterations-per-batch 50

python -m xpu_kernels.cli test --device npu --operator mhc_post_backward
python -m xpu_kernels.cli bench --device npu --operator mhc_post_backward \
  --implementations native,optimized --warmup 20 --batches 10 --iterations-per-batch 50

# RMSNorm 的 CANN 原生对照
python -m xpu_kernels.cli bench --device npu --native-reference \
  --warmup 20 --batches 10 --iterations-per-batch 50
```

指定 shape 可加 `--shapes 128x4096,129x4097`，指定精度可加 `--dtypes bfloat16`。记录保存设备型号、数值误差、软件版本、采样方式和每次测量，历史最佳只比较相同实现与测量条件。[记录格式](../results/benchmarks/README.md)

<details>
<summary><strong>NCU / Nsight Systems / msprof</strong></summary>

```bash
python -m xpu_kernels.cli profile --device cuda --tool ncu \
  --implementation cuda --shape 128x4096 --dtype bfloat16 \
  --output results/profiles/gpu-ncu
python -m xpu_kernels.cli profile --device cuda --tool nsys \
  --implementation triton --shape 128x4096 --dtype bfloat16 \
  --output results/profiles/gpu-nsys
python -m xpu_kernels.cli profile --device npu --operator mhc_post_backward \
  --implementation optimized --shape 2x1024x4x6144 --dtype bfloat16 --tool msprof \
  --output results/profiles/mhc-msprof
```

NCU 使用 kernel replay，收集一个匹配 launch 的 SpeedOfLight / LaunchStats，保持缓存与时钟设置。msprof 按当前版本支持的命令参数采集。完整 mHC 原生对照采集与提取方法见[性能文档](performance/mhc-post-backward.md)。

</details>


## 更新首页图表

图表直接从已验证的 JSONL 读取延迟。每组柱按同 shape 基线时间除以实现时间计算加速比；保留本库更慢的结果。绘图只读取现有记录。

```bash
python -m pip install -r tools/requirements-viz.txt
python tools/render_homepage_charts.py
PYTHONPATH=src python tools/render_homepage_performance.py
python tools/render_homepage_charts.py --check
```

绘图需要中文字体，Windows 默认使用微软雅黑；其它系统可安装 Noto Sans CJK SC，或传入 `--font /path/to/font.otf`。图表坐标从零开始，柱高按未舍入的加速比绘制，柱顶显示两位小数。源延迟、计算结果、颜色和 shape 保存在 [图表数据](assets/performance-chart-data.json)。[首页设计记录](homepage-design.md)
