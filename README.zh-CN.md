# RBCM-Edge

本分支是与论文 V5 同步的 **H-RBCM（HED-lite Retinal Boundary Context
Modulation）**及视网膜 MEA 分析的代码与配置发布版。

仓库有意排除论文文档、已生成图片、结果表、原始记录、图像数据集、权重、
预测结果和所有历史模型版本。运行生成的结果只保留在本地，并由 Git 忽略。

## 最终计算范围

- 一个可训练的四级编码、三分支解码 HED-lite anchor；
- 从同一冻结 anchor 得到 `plain_identity`、`main_surround`、
  `no_surround`、`conv_control` 四种输出；
- 由归一化 Sobel 证据、近远方环、带符号中心-周边对比、不确定性门控和
  logit 校正构成的确定性 H-RBCM；
- 只在源域验证集选择候选，随后冻结并用于同域和跨域评估；
- 最终 UME/CME 局部群体 MEA 分析；
- 图5采用样本内均值中心化、中位绝对残差缩放和对称 1-MAD 阈值的统计代码。

## 源码结构

- `MEA_analysis`：最终 MEA 分析与共享读取代码；
- `MEA_model`：MEA 作图和统计汇总代码；
- `edge_model`：anchor 训练、推理和评估；
- `src/rbcm_edge`：可导入的 HED-lite 模型与损失；
- `scripts`：校准、协议检查、跨域评估、图5统计与发布校验。

大文件和预训练权重按 `DOWNLOADS.md` 单独获取。完整步骤见
`REPRODUCE.zh-CN.md`，英文版为 `REPRODUCE.md`。

```bash
python -m pip install -r requirements-repro.txt
python -m pip install -e .
python scripts/release/verify_paper_release.py --code-root .
python scripts/release/smoke_paper_release.py --checkpoint-root pretrained --dataset all
python scripts/release/reproduce_release.py --checkpoint-root pretrained --stage all
```

统一 Python 评估器会恢复原图尺寸、按配置执行 NMS、扫描固定阈值，并使用
目标数据集对应的空间容差。其 dilation matcher 是统一的近官方实现，不是
BSDS 官方 Matlab 精确二分匹配器，比较时必须统一并明确标注评估后端。

构建脚本不会自动选择代码许可证；数据与记录的公开仍受原始来源及适用的
伦理、数据使用条款约束。
