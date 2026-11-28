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

输入必须连续且位于同一 NPU。可用 `out=(...)` 传入输出缓冲区；量化接口需要两个输出，其余接口需要一个。输出与输入、其他输出不得重叠。GPU 和 CPU 不执行融合实现；独立参考函数在 `fused_owner_combine_golden` 模块中。

## 构建与验证

先手动 source 所选 CANN，在项目私有 Python 环境安装本库：

```bash
python -m pip install -e '.[dev]'
python -m xpu_kernels.cli build --backend npu --operator fused_owner_combine
python -m xpu_kernels.cli test --device npu --operator fused_owner_combine
```

三个内核从原固定容量 MoE 路径迁入，原实验设备为 Ascend 910B3。此处改为本库的 DeepJIT 加载入口，尚未完成新入口的 NPU 编译、运行和图模式验证。CLI 的设备测试是 smoke 对照；正式优化验收继续要求精度标准 2.1、FP64 golden 与真实 GPU 双标杆。bench/profile 在这些验收完成前明确拒绝执行。

原组合路径存在一例长输出 token 不一致，整体精度验收尚未通过。该问题与三个独立内核、通信组和模型图执行分别核对，保留原失败结论。

设计、数据流和历史测量见[优化说明](optimization.md)。完整通信与专家计算组合见仓库的 `examples/moe_fixed_capacity.py`，调用方提供预先建立的专用 EP 通信组和专家计算函数。
