# 按 token 加权合并：设计与数据流

原流程先创建 owner 索引，再 gather 两路结果、转 FP32、加权和归约。融合内核让一个 Vector 核处理一个 token，UB 中保留累加器、当前专家向量和乘积。

```cpp
AscendC::Muls(prodUb, vUb, w[k], h); // each product rounds to FP32
AscendC::Add(accUb, accUb, prodUb, h); // then add in slot order
```

上面的实际代码先用标量权重乘向量，再更新累加器。迁移未改为树归约或 FMA。逐槽结果从全局内存进入 UB，乘权重后更新 FP32 累加器，最后转 BF16 写回。

## 在 MoE 中的位置

![量化打包、固定槽位与加权合并的数据流](../fused_quant_payload/figures/dataflow.png)

图中的 K1、K2、K3 是原实验的三个阶段，依次对应 `fused_quant_payload`、`fused_send_slots`、`fused_owner_combine`。图为源码数据流示意，不是硬件采集的流水图。AllGather、原 routing、专家计算和 AllToAll 位于三个接口之间，由组合示例连接。

## 迁移与数值约定

保留原 Ascend C 计算主体，将 OPP 的 tiling 指针读取改成 DeepJIT 入口参数构造局部 tiling 结构。Host 层验证 shape、dtype、布局、设备和输出别名，并沿用 PyTorch 当前设备与 stream；安装过程不覆盖公共 CANN/OPP。

独立参考用于核对布局与数值。`fused_owner_combine_golden(..., fp64=True)` 另外给出未转 BF16 的 FP64 数学结果；默认参考保留逐槽 FP32 舍入。量化参考以 FP64 除法后转 FP32，不把数学参考和特定设备舍入实现混为同一份代码。正式双标杆结果尚待实际 GPU/NPU 执行。

## 历史测量范围

原实验把三个融合内核、固定容量集合通信、独立专家并行（EP）通信组和图模式适配一起测量。这些组合结果不代表本接口的独立加速比。原组合的一例长输出 token 差异仍未关闭；迁移没有扩大其精度覆盖。

下面是盘古 92B 原整网组合方案的设备业务区间。模型使用 INT8 专家权重和 BF16 hidden，设备为同机双 Ascend 910B3，TP2/DP1/EP2、MTP3、并发 1、FULL_DECODE_ONLY；每个正式窗口各测一次。区间按两 rank 的慢端统计，单位秒；它不是客户端响应时间，也不是单内核耗时。

![MoE 原部署与组合路径的历史设备区间](../../assets/benchmark-moe.png)

图中选择 S3、S6、S8、S9 四个输入/输出组合，灰色为原生路径，绿色为固定容量组合路径。数据来自历史官方 msprof `task_time` CSV 区间，保留整网组合口径：固定容量集合通信、三个融合内核、独立专家并行（EP）通信组和图适配均包含在设备业务区间内；它不是本库新 JIT 入口或三个独立内核的单独耗时。长输出一致性问题仍待排查，详细状态见下表最后的说明。

| 输入 / 输出 token 数 | 原生路径（s） | 组合路径（s） |
|---|---:|---:|
| 188 / 1024 | 72.04 | 22.11 |
| 4096 / 1024 | 76.33 | 25.28 |
| 4096 / 2048 | 143.25 | 48.12 |
| 188 / 4096 | 272.88 | 91.25 |
| 15999 / 128 | 29.68 | 20.73 |

8192 / 2048 的旧正式请求曾在输出位置 16 出现 token 差异；后续有限复验不覆盖这一失败。以上数据保留原实验范围，不能替代本库新 JIT 入口的正确性与性能验收。
