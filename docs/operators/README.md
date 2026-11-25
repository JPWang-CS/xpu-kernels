# 算子文档

每个算子在自己的目录中保存两份文档：README 说明功能、公式、接口、范围和运行方法；优化文档串联原生实现、瓶颈、设计、代码、数据流、精度及实测结果。

| 算子 | 功能与使用 | 优化设计与结果 |
|---|---|---|
| RMSNorm | [README](rms_norm/README.md) | [优化设计](rms_norm/optimization.md) |
| mHC Post Backward | [README](mhc_post_backward/README.md) | [优化设计](mhc_post_backward/optimization.md) |

性能柱状图使用统一的 Python 绘图脚本；数据流示意图与对应的计算、搬运和同步代码一起阅读。各表保留设备、dtype、计时方式和采样条件。
