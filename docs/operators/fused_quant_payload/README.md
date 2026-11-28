# 量化与通信打包 · `fused_quant_payload`

`x`: BF16 `[16,2560]`；`topk_ids`: INT32 `[16,8]`；`active`: BOOL `[16]`。返回 INT8 `[16,2624]` 的 `payload` 和 FP32 `[16]` 的 `scale`。

## 计算过程

每行先按 `active` 屏蔽填充 token，再取 FP32 最大绝对值。设最大值为 `maximum`，非零行的放大倍数为 `127 / maximum`；将放大后的元素按最近偶数舍入并截断到 INT8，保存恢复比例 `scale`。零行写出零量化值和正零 scale。

| 行内字节偏移 | 内容 |
|---|---|
| 0–2559 | 2560 个 INT8 特征值 |
| 2560–2563 | FP32 scale 的原始字节 |
| 2564–2595 | 8 个 INT32 专家编号的原始字节 |
| 2596 | active 标志 |
| 2597–2623 | 27 字节零填充 |

scale 和专家编号保存原始位型，未做 INT8 数值量化。每行 2624 字节，16 行共 41984 字节。无效行的专家编号也填零。

## 调用

```python
from xpu_kernels import fused_quant_payload
payload, scale = fused_quant_payload(x, topk_ids, active)
```

融合后端使用同一 NPU 上的连续张量。通过 `out=(payload, scale)` 传入输出缓冲区，各缓冲区使用与输入及其他输出分离的内存区域。独立参考函数位于 `fused_quant_payload_golden` 模块。

## 构建与验证

先手动 source 所选 CANN，在项目私有 Python 环境安装本库：

```bash
python -m pip install -e '.[dev]'
python -m xpu_kernels.cli build --backend npu --operator fused_quant_payload
python -m xpu_kernels.cli test --device npu --operator fused_quant_payload
```

三个内核从原固定容量 MoE 路径迁入，公开接口面向固定 BF16 小批量 NPU 路径。CLI 已接入 `build`、`example` 和 `test`；新入口的 NPU 编译、运行与图模式验证待执行，性能验收待完成。

设计、数据流和历史测量见[优化说明](optimization.md)。完整通信与专家计算组合见仓库的 `examples/moe_fixed_capacity.py`，调用方提供预先建立的专用 EP 通信组和专家计算函数。
