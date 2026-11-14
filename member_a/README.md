# Member A 数据分析与特征实验

本路线直接接入仓库现有公共基础，没有另一套分折、标签构造或评分函数。使用 seed=4486 和协议 `patient5-omics-smape1-v1`。所有结果来自真实运行；代码、实验执行和文字整理均有 Codex 协助，使用前需复现理解并按课程要求披露。

## 快速查看

- `results/comparison.csv`：四组固定模型完整五折结果。
- `results/A0/oof.csv` 至 `results/A3/oof.csv`：每个文件包含完整 12,697 条 `prediction_id,rating` 预测。
- `eda/figures/`：三张数据分析图和一张特征实验图。
- `report_material.md`：英文技术素材。
- `slides.md`：三页展示内容及讲稿。
- `walkthrough.ipynb`：已执行的入门 Notebook。
- `AI_USAGE.md`：真实来源和 AI 协助说明。

## 真实实验结果

| experiment | mean_score | std_score | oof_score | delta_vs_A0 |
| --- | --- | --- | --- | --- |
| A0 | 65.621 | 3.329 | 65.484 | 0.000 |
| A1 | 66.126 | 3.110 | 65.990 | 0.505 |
| A2 | 66.754 | 3.108 | 66.632 | 1.133 |
| A3 | 66.816 | 3.809 | 66.650 | 1.195 |

`mean_score` 是组内主要比较分数，等权平均五折；`oof_score` 是所有请求合并后的分数。不能互换。A0 只使用预测时已知的三个时间字段；A1 增加 14 个当前采样汇总；A2 增加 22 个组学历史特征；A3 增加 16 个训练折内蛋白主成分。

四组固定相同的 HistGradientBoostingRegressor 配置。扩展特征没有改善主要比较分数，保留简单 A0 作为本路线候选。这不是对生物学作用的否定，也不是最终整组模型结论。

## 运行

在仓库根目录运行，推荐 Python 3.11：

```bash
python -m pip install -r member_a/requirements.txt
python -m unittest discover -s tests -v
python -m member_a.verify
```

完整重训并重现保存模型：

```bash
python -m member_a.run --output-dir member_a/results_reproduced
python -m member_a.verify --results-dir member_a/results_reproduced --require-models
```

输出目录必须不存在。训练默认只写本次输出目录中的实验日志。`--append-shared-log` 是显式选项；现有四个 experiment ID 已登记，重复登记会报错，重新运行用于复现时不要加该选项。

joblib 模型不进入 Git，完整重训可生成 20 个折模型包；当前提交提供 OOF、分数和实测验证记录。没有模型包时 `verify` 仍会核验完整请求覆盖、冻结哈希和重算分数，并明确记录模型核验未运行；加 `--require-models` 则要求模型存在并重现每条预测。只加载本人生成的 joblib 文件。

重新生成 EDA 和说明：

```bash
python -m member_a.eda --output-dir member_a/eda_reproduced
python -m member_a.write_materials
```

`write_materials` 读取已验证的默认 `member_a/results` 和默认 EDA 图目录，不执行训练。文档、图和 Notebook 是分析材料，不替代整组报告。

## 与公共协议对齐

1. 不生成或覆盖 `patient_folds.csv`、`samples.csv`、`fold_summary.csv`、`manifest.json`。仅调用 `common.prepare.prepare` 核验它们。
2. 使用 `common.interface.model_folds`，共享同一个 `split_id`：`5445dbd361f26a5fcdc2e1a6eeda1a2d1387942264957eb9e17844ff9163d550`。
3. 标签按单项 `prediction_id` 组织，仅预测已冻结请求。起点是蛋白与肽采样访视并集，包含无同月临床标签但未来有评分的请求。
4. 评分统一调用 `common.metric.evaluate_oof`。A 分支自己将负预测显式裁剪到 0；共享评分函数行为保持原样。
5. 特征仅使用组学历史和已知时间，不使用临床评分、用药状态、临床访视次数或未来采样信息。训练患者确定组学词表；PCA 中位数填补、标准化和基向量全部在训练折拟合。
6. 不使用补充数据训练、预训练模型、融合或验证折早停。

## 与旧版 A 工作包的区别

先前本地工作包使用 seed=42、临床 anchor 和 pooled OOF 主指标，其旧成绩不可与本仓库直接比较。本目录已重新使用当前仓库协议运行四组实验。旧工作包代码、旧折文件、旧分数和旧报告不纳入本次 GitHub 提交。

## 已知边界

离线评分只纳入冻结的有组学起点，不覆盖时序 API 的全部无组学请求分布。未为此分支实现正式的 API 提交循环，也未执行 Kaggle 提交。多组配置共用开发折，选择最优配置后也存在开发选择偏差。不能由这些成绩直接推断官方分数、名次或 Top 5。

基础依赖版本保持根目录 requirements.txt 的 numpy 1.26.4、pandas 2.3.3、scikit-learn 1.6.1、filelock 3.19.1。A 的额外依赖单独列出，没有覆盖公共依赖。实际执行环境记录在 results/manifest.json。
