# 专家结果槽位整理：设计与数据流

原流程需要局部专家 mask、索引 gather、填零和 reshape。融合实现把它们合成一次读写：每个 Vector 核处理同一 token 的 8 个选择，32 个核覆盖全部 256 个槽位。

```cpp
const uint32_t t = p / 8;
const bool isLocal = act && (ridV >= expertStart) && (ridV < expertEnd);
// isLocal: read down_proj[inverse_index[p]]; otherwise fill zero.
```

本卡专家结果从全局内存读入 UB，再搬出到固定槽位。输出逻辑容量为 1.25 MiB，包含发给本卡的数据块和零槽位；跨卡实际传输字节由调用方的 AllToAll 数据决定。

## 在 MoE 中的位置

![量化打包、固定槽位与加权合并的数据流](../fused_quant_payload/figures/dataflow.png)

图中的 K1、K2、K3 是原实验的三个阶段，依次对应 `fused_quant_payload`、`fused_send_slots`、`fused_owner_combine`。图展示源码数据流；AllGather、原 routing、专家计算和 AllToAll 位于三个接口之间，由组合示例连接。

## 迁移与数值约定

迁移保留 Ascend C 计算主体，将 OPP 的 tiling 指针读取改成 DeepJIT 入口参数构造局部 tiling 结构。Host 层验证 shape、dtype、布局、设备和输出别名，并沿用 PyTorch 当前设备与 stream；构建产物和 JIT 缓存写入项目目录。

独立参考用于核对槽位布局、专家区间和零槽位写入；设备实现保持固定槽位的读取与写出顺序。

## 历史测量范围

组合测量与验证状态统一记录在[按 token 加权合并的历史测量](../fused_owner_combine/optimization.md#历史测量范围)和[验证记录](../fused_owner_combine/optimization.md#验证记录)中。
