# 仓库结构与新增算子

每个算子使用一个稳定的英文名，例如 `rms_norm`、`mhc_post_backward`。Python API、设备实现、测试和文档使用同一个名字。设备名称用于区分实现，来源和优化历史写入说明与测量记录。

## 目录

```text
src/xpu_kernels/
  __init__.py                       # 导出公开 API
  <operator>.py                     # 参数检查、设备选择、公开接口
  <operator>_golden.py              # 独立数值参考
  <operator>_workload.py            # CLI 的测试、计时与采集负载
  backends/
    <operator>_cuda.py              # CUDA C++ 或 TileLang 接入
    <operator>_triton.py            # Triton 实现（使用时添加）
    <operator>_ascend.py            # Ascend C 接入
    _ascend.py                      # 多个算子共用的扩展加载器
    _tilekernels/                   # 第三方源码及其许可证
csrc/
  ascend/
    bindings.cpp                   # 公共 Python 绑定
    <operator>.cpp                 # Host 校验、tiling 与启动
  cuda/
    <operator>.cpp                 # CUDA C++ 绑定（使用时添加）
    <operator>.cu                  # CUDA C++ kernel（使用时添加）
kernels/include/xpu_kernels/
  <operator>/
    kernel.hpp                     # DeepJIT 设备入口
    *.hpp                          # 计算、tiling 等必要实现文件
    LICENSE                        # 对应源码许可证
tests/test_<operator>.py
docs/operators/<operator>.md
docs/performance/<operator>.md      # 有实测结果后添加
```

公共计时、环境检查、CLI 和结果校验保留在 `bench.py`、`environment.py`、`cli.py` 等公共模块。算子通过它们运行相同的流程。仅在使用相应后端时添加实现文件；第三方源码保留原始许可证及必要的原始标识。

Ascend C 每个算子只有一个设备目录，内部按功能平铺。RMSNorm 的 `row_retain.hpp` 和 `row_prefetch.hpp` 分别实现两行保留和三行预取；两者共用 `tiling.hpp` 与归约辅助代码。mHC 的 `cube.hpp`、`vector.hpp`、`tiling.hpp` 和 `cost_model.hpp` 分别承担矩阵计算、向量计算、分块参数和配置选择。

## 新增算子

1. 确定算子名与数学定义，写清输入、输出、dtype、布局、epsilon 等数值条件；在 `<operator>_golden.py` 实现独立参考。
2. 在 `<operator>.py` 建立公开接口和参数检查，按实际设备添加同名后端。Ascend C 使用公共加载器及本算子的 `kernel.hpp`，同时在 `bindings.cpp` 注册接口、在 `_ascend.py` 的源文件列表加入 Host 文件；CUDA C++、Triton、TileLang 按各自工具链构建。
3. 在 `tests/test_<operator>.py` 覆盖对齐、非对齐、小输入和模型输入，检查错误参数及非默认 stream。把采集负载接入 `<operator>_workload.py` 与现有 CLI。
4. 在 `docs/operators/<operator>.md` 写清公式、接口、原生实现、优化范围和执行命令。在包入口导出 API 后运行结构检查、正确性、安装包检查，再运行 benchmark 与 profiler。
5. 实测结果沿用 `results/benchmarks/` 的 JSONL 格式。保留设备型号、shape、dtype、实现、数值合同、计时方式与逐批样本；对照表由这些记录生成。首页只放已验证的优化图与算子入口。

```bash
python tools/check_repository_layout.py
python -m unittest discover -s tests -v
python -m xpu_kernels.cli test --device npu --operator <operator>
python -m xpu_kernels.cli bench --device npu --operator <operator>
python -m xpu_kernels.cli profile --device npu --operator <operator> --tool msprof
```

新增算子时同时在 CLI 中注册其名称与负载。结构检查从公开导出自动发现算子，核对必需文件、后端命名及本地头文件引用；设备测试继续验证实际执行结果。

## 维护约定

当前版本的源码使用算子名及功能名。`native`、`optimized` 是实现选择参数，CANN 等来源写入文档和第三方说明。历史 JSONL、profiler 符号和原始采集文件保留当时的名字，测量时间与当前源码分开解释。

打包沿用一套 `setup.py`，将 `csrc/`、`kernels/include/` 和 DeepJIT 资源一并放入 wheel。新增算子不复制构建系统。源码目录与 wheel 均须核对资源完整性，设备构建和执行记录单独保存。

参考 [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM)、[DeepJIT](https://github.com/deepseek-ai/DeepJIT) 的公共运行时组织，以及 [TileKernels](https://github.com/deepseek-ai/TileKernels/tree/main/tile_kernels/mhc) 的同名多设备实现。这里保留本库的统一 Python API、独立 golden 和验收流程。
