# 按 token 加权合并 · `fused_owner_combine`

`received_slots`: BF16 `[2,16,8,2560]`；`topk_ids`: INT32 `[16,8]`；`topk_weights`: FP32 `[16,8]`；`active`: BOOL `[16]`。返回 BF16 `[16,2560]`。有效行的专家编号在 `[0,256)` 内。

## 计算过程

AllToAll 后，第一维区分两个专家来源 rank。对每个 token 的第 `k` 个专家选择，`topk_ids[token,k] // 128` 给出结果所属的来源 rank。内核从该 rank 取出对应槽位，乘以路由权重，再按 `k=0..7` 的顺序累加。

$$
{\mathrm{output}}_{t,d}=\mathrm{BF16}\left(\sum_{k=0}^{7}
{\mathrm{received\_slots}}_{\lfloor {\mathrm{topk\_ids}}_{t,k}/128\rfloor,t,k,d}
\,{\mathrm{topk\_weights}}_{t,k}\right).
$$

有效行使用逐槽 FP32 乘法和加法，无效行输出零。公式表达加权和；实现明确保留逐槽舍入顺序。

## 调用

```python
from xpu_kernels import fused_owner_combine
output = fused_owner_combine(received_slots, topk_ids, topk_weights, active)
```

融合后端使用同一 NPU 上的连续张量。通过 `out=(output,)` 传入输出缓冲区，各缓冲区使用与输入及其他输出分离的内存区域。独立参考函数位于 `fused_owner_combine_golden` 模块。

## 构建与验证

先手动 source 所选 CANN，在项目私有 Python 环境安装本库：

```bash
python -m pip install -e '.[dev]'
python -m xpu_kernels.cli build --backend npu --operator fused_owner_combine
python -m xpu_kernels.cli test --device npu --operator fused_owner_combine
```

三个内核从原固定容量 MoE 路径迁入，公开接口面向固定 BF16 小批量 NPU 路径。CLI 已接入 `build`、`example` 和 `test`；新入口的 NPU 编译、运行与图模式验证待执行，性能验收待完成。

组合测量与精度状态见[验证记录](optimization.md#验证记录)。

设计、数据流和历史测量见[优化说明](optimization.md)。完整通信与专家计算组合见仓库的 `examples/moe_fixed_capacity.py`，调用方提供预先建立的专用 EP 通信组和专家计算函数。
