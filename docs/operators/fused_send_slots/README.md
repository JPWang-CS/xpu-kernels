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

融合后端使用同一 NPU 上的连续张量。通过 `out=(send_slots,)` 传入输出缓冲区，各缓冲区使用与输入及其他输出分离的内存区域。独立参考函数位于 `fused_send_slots_golden` 模块。

## 构建与验证

先手动 source 所选 CANN，在项目私有 Python 环境安装本库：

```bash
python -m pip install -e '.[dev]'
python -m xpu_kernels.cli build --backend npu --operator fused_send_slots
python -m xpu_kernels.cli test --device npu --operator fused_send_slots
```

三个内核从原固定容量 MoE 路径迁入，公开接口面向固定 BF16 小批量 NPU 路径。CLI 已接入 `build`、`example` 和 `test`；新入口的 NPU 编译、运行与图模式验证待执行，性能验收待完成。

设计、数据流和历史测量见[优化说明](optimization.md)。完整通信与专家计算组合见仓库的 `examples/moe_fixed_capacity.py`，调用方提供预先建立的专用 EP 通信组和专家计算函数。
