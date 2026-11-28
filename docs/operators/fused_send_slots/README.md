# 专家结果槽位整理 · `fused_send_slots`

`down_proj`: BF16 `[N,2560]`，`1 <= N <= 256`；`inverse_index`: INT32 `[256]`；`route_ids`: INT32 `[32,8]`；`active`: BOOL `[32]`。专家区间只能为 `[0,128)` 或 `[128,256)`。返回 BF16 `[2,16,8,2560]`。

## 计算过程

32 行来自两张卡，每张卡各有 16 行；每行包含 8 个专家选择，共 256 个 `(token, slot)`。`inverse_index` 把每个选择映射到专家计算结果的行号。

对有效 token，若专家编号属于本卡区间，就读取对应的 `down_proj` 行；其余槽位填零。输出的第一维表示原 token 所在的目标 rank，随后依次为 token、专家选择槽和特征。

有效且属于本卡的 `inverse_index` 必须在 `[0,N)` 内。无效或非本卡槽位不读取专家结果。

## 调用

```python
from xpu_kernels import fused_send_slots
send_slots = fused_send_slots(down_proj, inverse_index, route_ids, active, 0, 128)
```

输入必须连续且位于同一 NPU。可用 `out=(...)` 传入输出缓冲区；量化接口需要两个输出，其余接口需要一个。输出与输入、其他输出不得重叠。GPU 和 CPU 不执行融合实现；独立参考函数在 `fused_send_slots_golden` 模块中。

## 构建与验证

先手动 source 所选 CANN，在项目私有 Python 环境安装本库：

```bash
python -m pip install -e '.[dev]'
python -m xpu_kernels.cli build --backend npu --operator fused_send_slots
python -m xpu_kernels.cli test --device npu --operator fused_send_slots
```

三个内核从原固定容量 MoE 路径迁入，原实验设备为 Ascend 910B3。此处改为本库的 DeepJIT 加载入口，尚未完成新入口的 NPU 编译、运行和图模式验证。CLI 的设备测试是 smoke 对照；正式优化验收继续要求精度标准 2.1、FP64 golden 与真实 GPU 双标杆。bench/profile 在这些验收完成前明确拒绝执行。

原组合路径存在一例长输出 token 不一致，整体精度验收尚未通过。该问题与三个独立内核、通信组和模型图执行分别核对，保留原失败结论。

设计、数据流和历史测量见[优化说明](optimization.md)。完整通信与专家计算组合见仓库的 `examples/moe_fixed_capacity.py`，调用方提供预先建立的专用 EP 通信组和专家计算函数。
