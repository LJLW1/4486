# CS4486 Parkinson 项目公共基础

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
