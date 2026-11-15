# CS4486 Parkinson 项目公共基础

> **Member C 当前进度（2026-11-15）**：参见 [V18 实验交接文档](C_improved_model/PROGRESS_HANDOFF.md)，包含两轮五折结果、第三轮待测方案、已解决的 macOS 环境问题、Top-5 目标与下个对话接续说明。


按上级目录 `CS4486_Three_Person_Parallel_Workplan.docx` 第 2 节建立四个共享接口：验证划分、评价指标、数据输入、实验日志。A / B / C 可以在这些接口上分别开展自己的模型实验。

## 开始使用

在本目录运行，Python 3.9 或以上：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m common.prepare
python -m unittest discover -s tests -v
python -m common.verify
```

真实数据已在 `amp-parkinsons-disease-progression-prediction/`，固定输出已生成在 `artifacts/shared/`。`prepare` 再次运行会核验原始文件及输出的 SHA-256，不重新分折或覆盖结果。需要改变协议或数据时，指定新的 `--output-dir`，并使用独立实验日志。输入路径也可以通过 `--data-dir` 指定。

## Member C 最小 MLP

安装 `requirements.txt` 后，打开 `C_improved_model/train_mlp.ipynb`，选择对应的 Python 内核，从上到下运行。模型由 `MLP` 类定义，训练由 `train_model` 函数定义。训练轮数、学习率和正则化参数在配置单元格直接修改。

默认 `use_proteins=True`、`max_missing_rate=1.0`、`imputation_method="iterative"`，使用三个时间字段、四项评分类型的 one-hot 编码及当前访问的蛋白 NPX。每折仅根据训练部分的唯一访问筛选缺失率不超过阈值的蛋白，按相同列序处理验证数据；`max_missing_rate=1.0` 保留全部 227 种蛋白，`0.0` 仅保留训练部分完全无缺失的蛋白。填补由 `prepare_features` 函数完成：`median` 用唯一训练访问的中位数；`iterative` 先在唯一训练访问上标准化蛋白，再用 `IterativeImputer + BayesianRidge` 拟合蛋白间关系（最多 10 轮，每种蛋白参考 20 个相关特征，tol=0.01）。仅替换缺失值，观测值保持不变，填补的 NPX 裁剪到非负。填补器不使用时间、UPDRS 标签或验证患者。拼接时间特征后，再对模型输入拟合训练折 StandardScaler。NPX 暂不做 log 变换。若本机 OpenBLAS 小矩阵多线程运行缓慢，可在启动 notebook 内核前设置 `OPENBLAS_NUM_THREADS=1`；本次对照使用此设置。MLP 隐藏层为 32、16，以 MSE 损失和 AdamW 优化器完成 CPU 五折训练，预测裁剪至非负，最终按 SMAPE+1 评分。

将 `use_proteins=False` 并从头运行，可复现仅时间和评分类型的 7 维输入参照实验。保留简单随机种子，不强制确定性运算。

结果保存在 `artifacts/member_c/`：每次运行的 OOF CSV，以及通过公共接口追加的 `experiment_results.csv`。日志区分蛋白与时间特征实验，记录缺失阈值及每折保留的蛋白列。阈值对比结果见 `artifacts/member_c/protein_missingness_comparison.csv`；按唯一训练访问拟合的中位数与迭代填补对照见 `artifacts/member_c/protein_imputation_comparison.csv`。不使用肽、历史或补充临床特征，暂不保存推理模型或接入 Kaggle 提交 API。

## Member C 改进对照

`C_improved_model/model_improvements.ipynb` 按顺序运行六组实验：时间/蛋白原模型诊断、相同 Softplus 输出的 MSE 与 SMAPE+1 损失对照、SMAPE+1 下的蛋白实验，以及截至当前的组学访问历史实验。固定 200 轮，不按验证曲线挑选 epoch。实验结果、训练曲线、分目标诊断和总结保存在 `artifacts/member_c/improvements/`。

单 seed 结果中，最佳候选为时间 + Softplus + SMAPE+1，五折平均分 65.1485；蛋白模型明显过拟合，本次访问历史未带来提升。Notebook 保留所有对照，尚未接入 Kaggle 提交或训练最终推理模型。

## Member C 蛋白百分位编码

`C_improved_model/protein_quantiles.ipynb` 比较蛋白五等级数值与五维 one-hot。每种蛋白的分界点由训练折唯一访问的非缺失 NPX 计算（20%、40%、60%、80%），缺失用同一训练部分的中位数填补，验证使用冻结的分界。每种编码对照线性输出 MSE、Softplus MSE、Softplus SMAPE+1，以及加入当前及过去的组学访问历史；共享两个时间模型对照。标签保持连续 UPDRS。

沿用输入标准化；one-hot 每列也使用训练折 StandardScaler。重复分界可能导致某些等级为空，同值不会被强行分到不同等级。本轮没有额外加入缺失标记。每个实验的 OOF、曲线、分目标评分、每折蛋白列和分界点均保存在 `artifacts/member_c/protein_quantiles/`。

## Member C 共享蛋白编码器

`C_improved_model/protein_set.ipynb` 用共享网络编码每种蛋白的 ID embedding 和当前 NPX，按访问做 masked mean pooling，拼接时间与评分类型后预测 UPDRS。缺失蛋白不参与聚合，不做中位数填补；每折仅用唯一训练访问拟合蛋白标准化器。每个请求仍是一条监督样本，同一访问的蛋白表示只计算一次。

固定 200 次全批次更新，使用 Softplus、SMAPE+1 和 AdamW；与纯时间模型同折比较。集合模型参数 1,921，五折平均 65.6205，时间模型为 65.1485：过拟合差距缩小，但未获得整体增益。结果、OOF、曲线、分项诊断及配置保存在 `artifacts/member_c/protein_set/`。本次 CPU 对照设置 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`。

## Member C 相关性筛选

`C_improved_model/protein_selection.ipynb` 在每折训练患者内，按当前访问 NPX 与四项评分的相关性筛选 top 5/10/20，接入共享蛋白集合模型。排名使用控制访问月份秩后的 partial Spearman 绝对值在四项评分上的均值，同时保存 Pearson、Spearman、访问数、患者数和缺失率。每个组合至少有 100 个访问、50 位患者；仅四项评分均合格的蛋白参与排名。

五折平均：纯时间 65.1485、全部蛋白 65.6205、top 5 为 65.2489、top 10 为 65.0331、top 20 为 65.2274。Top 10 收益仅 0.1154 分且只有两折改善，属于待复核的探索性结果。全数据排名仅作解释，不参与 CV 筛选；重复访问相关性不作独立样本显著性推断。结果在 `artifacts/member_c/protein_selection/`。

## Member C Bayesian Ridge PLS 与 SVR

`C_improved_model/classical_models.ipynb` 比较 Bayesian Ridge、PLS 回归、PCA+SVR 和 PLS+SVR，并提供纯时间 Bayesian Ridge/SVR 对照。每项 UPDRS 单独训练，全部 horizon 合并；不使用临床评分作为输入。Bayesian Ridge 和直接 PLS 使用时间与所有蛋白的联合输入；PCA/PLS+SVR 只压缩蛋白，再拼接时间。

外层沿用冻结患者五折；内层使用患者三折，按 SMAPE+1 选择 PLS/PCA 组件数 1/2/5/10 及 RBF-SVR C=0.01/0.1/1/10。SVR epsilon=0.1、gamma=scale 固定。每次内层拟合都重新学习填充、缩放和降维；PCA 在唯一训练访问上拟合，监督 PLS 只使用当前训练部分标签。负预测统一裁剪为零。没有根据外层分数选参。

结果保存在 `artifacts/member_c/classical_models/`，包括 OOF、各目标分数、内层搜索、每折配置、运行时间与 Bayesian Ridge 标准化系数。依赖已包含在现有 scikit-learn 中，无需新增安装。

Final five-fold means: time_bayesian=71.0611, bayesian=74.6222, pls=68.9918, time_svr=66.4686, pca_svr=66.8982, pls_svr=66.8303. Full details: `artifacts/member_c/classical_models/summary.md`.

## Member C 概率分类与 SMAPE 解码

`C_improved_model/probability_smape.ipynb` 调用同目录 `probability_smape.py`，完成固定患者五折的四组特征实验。四项 UPDRS 分别训练固定小容量 LightGBM 多分类器，同一份概率比较最高概率类别、概率均值和最小期望 SMAPE+1 解码；不使用 supplemental、分子丰度或临床评分特征。新增依赖 `lightgbm==4.6.0`。

SMAPE 解码五折平均：时间 64.8009、基础历史 58.2572、扩展历史 58.1587、仅组学可见扩展历史 65.3848。完整历史收益依赖临床与组学访问并集可见的本地假设；扩展相对基础仅改善 0.0985 且折间波动更大，未确认稳定增益。这是开发集结果，无隐藏成绩或 Top-5 结论。

完整 OOF、分折模型与验证概率、分项结果及复现说明见 `artifacts/member_c/probability_smape/summary.md`。17 项单元测试通过，12 份 OOF 独立重评分及 80 份概率解码复核通过。本 notebook 仅训练，不接入比赛 API。

## 1 验证划分

- 协议：`patient5-omics-smape1-v1`，5-fold，seed `4486`。
- 将主队列的患者 ID 排序，在患者列表上使用 `KFold(n_splits=5, shuffle=True, random_state=4486)`，再映射到所有样本；不对访问行随机分折。
- 248 位主队列患者，每折 49–50 人。以同一患者为起点或终点的所有预测请求始终属于同一折。
- 泛化目标是未见过的患者。该设计是本项目的离线验证协议，不等于 Kaggle 隐藏测试集，也不能直接据此声称 leaderboard 名次。
- 原点取蛋白和肽表的访问并集，即有组学样本的访问。按比赛 evaluation 的文字说明，对每个原点预测 0 / 6 / 12 / 24 个月后的四项 UPDRS。未来访问必须精确存在；不插值、不寻找“最近的访问”、不外推标签。某项标签缺失时只删除该项评分请求。
- 主数据中有 45 个组学访问没有同月临床记录；它们仍可作为未来预测原点。实现不通过 inner join 意外丢掉这些原点。
- 当前冻结集有 **12,697** 条有标签请求。它不是原始临床行数，也不是独立患者数；同一个未来标签可能对应不同预测原点，这是不同的预测请求。
- 补充临床数据不进入验证集。默认模型接口只单独返回它，是否用于训练必须记入 `features/config/notes`；使用时自动排除本折验证患者。当前数据中两队列患者没有交集。

| fold | 验证患者 | 有标签请求 |
|---|---:|---:|
| 0 | 50 | 2769 |
| 1 | 50 | 2588 |
| 2 | 50 | 2584 |
| 3 | 49 | 2351 |
| 4 | 49 | 2405 |

共享文件：`patient_folds.csv` 是患者到 fold 的映射，`samples.csv` 是带标签预测请求，`fold_summary.csv` 是各折数量，`manifest.json` 记录协议、环境版本、文件哈希和完整 `split_id`。三人应共享整个输出目录。

## 2 评分约定

统一调用 `common.metric.smape_plus_one(y_true, y_pred)`，返回百分数尺度，越低越好：

```text
SMAPE+1 = mean(200 * abs(y - prediction) / (abs(y + 1) + abs(prediction + 1)))
```

这里是先给真值和预测值加 1，再计算 SMAPE，不是给最终分数加 1。概览页仍显示 SMAPE；比赛的一手参赛者说明和最终获胜方案使用 SMAPE+1，详见下方来源。双方为零时得 0。缺失标签可在数组接口中屏蔽；有标签位置的 NaN / inf 预测会报错。函数不自动裁剪负预测，任何裁剪均须由模型分支显式执行并记录。

完整 CV 使用 `evaluate_oof(samples, predictions)`，其中 `predictions` 必须包含且仅包含所有待评分 `prediction_id`，以及一列 `rating`。行顺序可以不同，重复、遗漏、多余的 ID 均报错。

- `fold_scores`：各折有效标量请求的 SMAPE+1。
- `mean_score`：五折分数的算术平均，作为组内主要比较分数。
- `std_score`：五折分数的总体标准差，`ddof=0`。
- `oof_score`：全部 OOF 请求合并评分，按有效请求数加权；折大小不同，因此它与 `mean_score` 可能不同。报告两者时必须标清。

## 3 数据与模型输入

`load_raw(data_dir)` 返回 `RawData(clinical, proteins, peptides, supplemental)`，四表保持长表结构。读取时校验必需字段、非负整数患者/月、`visit_id = patient_id_visit_month`、唯一键和数值合法性；保留缺失值。不对全数据做均值填充、标准化或缺失值筛选。

| 表 | 唯一键 | 数值或标签 |
|---|---|---|
| clinical / supplemental | visit_id | updrs_1…4；另保留原始 medication 字段 |
| proteins | visit_id, UniProt | NPX |
| peptides | visit_id, UniProt, Peptide | PeptideAbundance |
| samples | prediction_id | rating 为目标月的单项真值 |

`samples` 字段：`visit_id, patient_id, visit_month, horizon, target, target_month, prediction_id, rating, fold`。`visit_month` 是预测原点，`target_month = visit_month + horizon`。`prediction_id` 与 Kaggle 示例格式一致，例如 `55_0_updrs_1_plus_6_months`。

推荐三条路线统一使用 `common.interface.model_folds`：

```python
import json
from pathlib import Path
import pandas as pd
from common.data import load_raw
from common.interface import model_folds
from common.reproducibility import set_seed

set_seed()
shared = Path("artifacts/shared")
manifest = json.loads((shared / "manifest.json").read_text())
raw = load_raw("amp-parkinsons-disease-progression-prediction")
samples = pd.read_csv(shared / "samples.csv")
patients = pd.read_csv(shared / "patient_folds.csv")

for data in model_folds(raw, samples, patients):
    # data.fold, data.train_samples, data.valid_samples
    # data.train_features 与 data.valid_features 已与标签逐行对齐
    # 标签分别是 data.train_samples.rating / data.valid_samples.rating
    # data.supplemental 仅供可选的训练增强
    pass  # 在自己的模型分支内训练并保存本折 validation 的预测
```

输入特征是当前原点的 `protein::UniProt`、`peptide::UniProt::Peptide`，以及已知的 `visit_month, horizon, target, target_month`。接口以训练患者确定组学列，再用相同列变换验证集；验证集独有的分子列被忽略。`patient_id, visit_id, prediction_id` 仅为对齐键，不能直接作为模型数值特征。`target` 需要在模型内编码或用于分目标训练。

特征表不含 `rating`、四项临床评分或 medication。临床表在预测时不提供这些值，不能把当前访问的临床真值作为特征。需要历史/序列特征的路线须在各患者内仅使用 `history.visit_month <= origin.visit_month` 的组学记录，禁止访问未来观测、未来访问次数、终末访问月和未来缺失模式。所有填充、缩放、筛选、目标编码和超参数选择都必须在训练折内拟合。

低层 `visit_features` 支持传入固定的蛋白/肽列清单；若不传，列来自所传原始表，因此训练/验证建模应使用上面的 `model_folds` 入口。

推理时可用 `prediction_requests(test[KEYS])` 生成同格式请求，并用 `visit_features` 对接当批组学。没有组学测量的访问仍保留，特征为 NaN，训练阶段应设计对应的缺失处理。Kaggle 示例确实包含这种访问；当前离线评分协议依照 evaluation 描述，只纳入组学原点，因此不涵盖示例 API 的全部无组学请求分布。真实提交还需各模型实现时序 API 的逐批预测；本公共基础没有包含提交模型。

## 4 实验记录

`artifacts/shared/experiment_results.csv` 已记录 Member A 的四组真实五折特征实验，详见 [Member A 工作说明](member_a/README.md)。公共基础的接口验收分数不作为模型性能。

每个完整五折实验调用一次：

```python
from common.metric import evaluate_oof
from common.experiments import log_experiment

# oof_predictions 是三条路线各自生成的 DataFrame(prediction_id, rating)
evaluation = evaluate_oof(samples, oof_predictions)
log_experiment(
    shared / "experiment_results.csv",
    experiment_id="memberA_method_v1", model="填写实际模型",
    features="填写特征和 supplemental 使用情况",
    evaluation=evaluation, seed=4486, runtime_seconds=elapsed_seconds,
    split_id=manifest["split_id"],
    config={"parameters": {}, "code_revision": "填写版本或提交号"},
    notes="实验假设、效果、失败原因；必要时附模型库及硬件信息",
)
```

字段包含 model、features、fold scores、mean/std、OOF、seed、runtime、config 和 notes。写入加文件锁，禁止重复 experiment ID、不同 split 混入同一表或不完整五折记录。`set_seed` 只固定 Python/NumPy；所用训练库的 random_state、PyTorch seed/确定性设置等由模型分支按实际使用记录。总运行时间从模型分支自己的计时器获得，不使用估算数。

## 验收与来源

`tests/test_common.py` 覆盖手算评分、无效值、请求 ID 对齐、精确 horizon、缺失标签、无临床标签特征、无未来组学特征、固定患者划分、训练折词表、补充患者隔离、文件篡改及日志约束。`common.verify` 在真实数据上跑完五折接口，检查 Kaggle 示例请求及无组学访问，结果写入 `verification.json`。身份预测只用于检验评分器，不代表模型性能。

- 需求：上级目录工作计划第 2 节及 `../Reference/Mini Project Instruction-CS4486.pdf`。
- [Kaggle competition evaluation](https://www.kaggle.com/competitions/amp-parkinsons-disease-progression-prediction/overview/evaluation)：组学原点、预测 horizon、时序 API。
- [Kaggle competition data](https://www.kaggle.com/competitions/amp-parkinsons-disease-progression-prediction/data)：原始下载文件；以本机 CSV 和 example_test_files 核验字段。
- [一手参赛者对 SMAPE+1 更新的说明](https://www.kaggle.com/competitions/amp-parkinsons-disease-progression-prediction/discussion/417124)：真值与预测均先加 1。
- [第一名方案的指标说明](https://www.kaggle.com/competitions/amp-parkinsons-disease-progression-prediction/writeups/connecting-dotts-1st-place-solution)：最终使用 SMAPE+1。这里只核对指标，没有复现其模型。

本次公共基础代码由 Codex 协助实现。课程要求披露 AI 使用，组员需审查、理解并在报告中如实说明这些辅助实现及自行完成的核心设计。`common/` 只提供公共接口；`member_a/` 提供 A 的特征实验路线。当前不包含 B/C 的模型路线，也未声称达到 Top-5 performance。

## Member A 特征实验

在仓库根目录运行：

```bash
python -m pip install -r member_a/requirements.txt
python -m unittest discover -s tests -v
python -m member_a.verify
```

完整重训：`python -m member_a.run --output-dir member_a/results_reproduced`，然后执行 `python -m member_a.verify --results-dir member_a/results_reproduced --require-models`。训练默认不更改共享日志；仅在明确需要登记新实验时使用 `--append-shared-log`，现有 experiment ID 不可重复。

A0 至 A3 均使用当前冻结的 seed=4486 和同一 split_id。主要比较值仍是五折 `mean_score`，不是 pooled OOF。代码、结果、图和 AI 协助说明见 [member_a/README.md](member_a/README.md)。本路线没有使用补充数据、预训练模型或融合，也没有执行官方提交。
