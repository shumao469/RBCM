# RBCM-Edge V5 复现说明

本文档覆盖论文最终使用的 MEA 分析和 H-RBCM 边缘检测实验，不包含已废弃的历史模型路线。

## 1. 准备代码和资源

克隆公开仓库的 `release` 分支，并把单独发布的资源解压到代码根目录：

```text
RBCM-Edge/
  pretrained/   # checkpoint 与验证集冻结候选
  edge_data/    # 正式边缘数据集和划分文件
  MEA_data/     # 经授权可共享的 MEA 输入
```

解压前使用 `DOWNLOADS.md` 中的 SHA-256 校验压缩包。数据只能在原始许可、伦理和机构共享要求允许时分发。

## 2. 创建环境

严格 NYUDv2 本地环境为 Python 3.10.16、PyTorch 2.7.1+cu126 和 torchvision
0.22.1+cu126；严格 MultiCue 使用服务器镜像 `cuda128_torch280_py312`。安装与本机
CUDA 匹配的 PyTorch 后运行：

```bash
python -m pip install -r requirements-repro.txt
python -m pip install -e .
```

不同 GPU、CUDA、cuDNN 和 NMS 实现可能产生很小的数值差异。划分身份、冻结候选、预测方向和评估后端必须一致。

## 3. 运行发布版检查

```bash
python scripts/release/verify_paper_release.py --code-root .
python scripts/release/smoke_paper_release.py --checkpoint-root pretrained --dataset all
python scripts/checks/audit_multicue_strict_protocol.py \
  --config edge_model/configs/rbcm/multicue_strict.yaml
python scripts/checks/audit_nyud_strict_protocol.py \
  --config edge_model/configs/rbcm/nyudv2_strict.yaml
python scripts/checks/run_rbcm_paper_preflight.py --check-data
```

正式协议为：BIPED 170/30/50 并使用三个固定划分；MultiCue 68/12/20 个源图像互斥；
NYUDv2 RGB 为 381/414/654；BSDS500 和 UDED 只作为评估或迁移目标。

## 4. 一体化复现 V5 内部结果

下面的入口会按依赖顺序执行发布检查、协议检查、三个冻结训练源的推理与正式评估、
标准化结果输入、V5 内部表1-3、图5和最终完整性审计：

```bash
python scripts/release/reproduce_release.py \
  --checkpoint-root pretrained --device cuda
```

长时间运行前可加入 `--dry-run` 查看完整命令计划。没有授权 MEA 包时可加入
`--skip-mea`。这条流水线不训练模型，也不会在测试集或目标域上搜索候选参数。

单张图像目录推理示例：

```bash
python edge_model/infer.py \
  --config edge_model/configs/rbcm/nyudv2_strict.yaml \
  --checkpoint pretrained/nyudv2_strict/best.pt \
  --image-dir edge_data/official_rbcm/NYUDv2/image \
  --output-dir reproduced/nyudv2_main \
  --mode main_surround \
  --candidate-csv pretrained/nyudv2_strict/fixed_candidates.csv
```

论文评估器恢复原图尺寸、执行固定 NMS 和阈值扫描，并使用数据集固定空间容差。
它是统一的近官方 Python dilation matcher，不是 BSDS 官方 Matlab 精确二分匹配器。

## 5. 表格边界

代码、checkpoint 和数据可以独立重建内部表1-3。表4和表5包含第三方模型结果，
需要另外取得外部证据目录，其中至少包含 `source_matched_learned.csv`、
`curve_manifest.csv` 和其引用的外部 PR 曲线：

```bash
python scripts/release/reproduce_release.py --stage tables \
  --external-evidence-root external_evidence
```

没有外部证据时，程序不会用论文中的数字代替真实评估输出。所有生成结果写入被 Git 忽略的 `edge_outputs/`。
`build_v5_result_tables.py` 本身不保存论文分数；内部表1-3来自标准化评估输出，
表4-5只有在提供外部证据包后才生成。

## 6. 从头训练

```bash
python edge_model/train.py \
  --config edge_model/configs/rbcm/nyudv2_strict.yaml
python edge_model/train.py \
  --config edge_model/configs/rbcm/nyudv2_strict.yaml --resume
```

MultiCue 必须使用 `multicue_strict.yaml`。候选参数只能由验证集选择，测试集必须在候选
写入并哈希冻结后才可读取。历史 `multicue.yaml`、`nyudv2.yaml` 和 `bsds500.yaml`
的验证集与测试集完全重合，训练入口会拒绝它们，公开 release 也不包含它们。

新训练保存 `best_validation_metric.pt`；读取旧实验时仍兼容历史名称
`best_train_metric.pt`。`last.pt` 保存模型、优化器、调度器、AMP scaler、配置和 RNG 状态。

## 7. MEA 与图5

```bash
python scripts/analysis/run_mea_pipeline.py
python scripts/analysis/reproduce_figure5_relative_statistics.py \
  --candidate-csv pretrained/multicue_strict/calibration_candidates.csv
python scripts/figures/bridge/render_figure5_relative_panels.py \
  --source-dir edge_outputs/rbcm/analyses/mea_rbcm_bridge/figure5_relative \
  --output-dir edge_outputs/rbcm/figures/mea_rbcm_bridge/figure5_relative
```

图5使用纯 H-RBCM 项 `alpha * U * C`，状态分类不读取目标 GT。图中比较的是样本内
归一化后的相对空间异质性，不表示 MEA 与网络原始绝对效应分布相同。

## 8. 结果一致性

复现结果应与 V5 接近，但不要求不同硬件和库版本下逐字节一致。只有协议、划分身份、
冻结候选、评估后端和指标定义一致时，才可视为同一实验的复现。

MultiCue 会先要求新生成的保存概率图指标与 checkpoint 中的全精度冻结参考值在
`1e-4` 内一致，再使用 canonical 全精度值，避免 GPU 与 uint8 PNG 量化单独造成
第四位小数变化。NYUDv2 浮点概率指标必须在 `5e-8` 内一致。两类差值都会写入
`edge_outputs/rbcm/reproduction/v5/inputs/reproduction_manifest.json`。
