# Member C / V18 实验进度交接（2026-11-15）

> **下一个对话优先阅读此文件。** 项目为 CS4486 帕金森病 UPDRS 预测，三人小组；2026-11-17 需要提交 PPT 并完成有关实验。**用户目标是争取 Kaggle Top 5，不能因为两轮实验仅有微小收益就停止优化。** 已有结果均为本地五折验证，不代表 Kaggle 名次。项目仓库 `LJLW1/4486`；用户自己的分支实际名称是 **`member_c`**（下划线），不得擅自新建分支、修改 main。

## 当前基线 v18

- Kaggle notebook：`C_improved_model/architecture_residual_submission.ipynb`；可导入实现：`C_improved_model/architectures.py`、`teammate_neural_proteins.py`、`teammate_aligned_model.py`。
- 架构：LightGBM 分类树（300 棵树，类别概率经最小期望 SMAPE+1 决策解码）与三个 PyTorch 残差网络（随机种子 4486/4487/4488）融合。网络的 `ScoreNetwork` 从访问历史特征预测 16 个 target/horizon 输出，残差修正网络加入蛋白 `P05060`、`O15240`；当蛋白缺失时残差被 mask。默认残差强度 λ=1、联合训练 200 epochs。
- 四目标树/神经融合 `ALPHA=[0.625,0.75,0.75,0.375]`。训练使用主队列及 supplemental 临床请求；五折均剔除验证患者。树与神经网络都只使用比赛允许的时间、访问历史和蛋白特征，没有直接读取真实 UPDRS 作为预测输入。
- 基线：**五折平均 SMAPE+1 = 62.1387**，标准差约 **4.4150**；评分越低越好。此处**不等于 Kaggle public/private leaderboard**，尚无经确认的 Top-5 结果。

## 固定验证约定

- 协议 `patient5-omics-smape1-v1`；`artifacts/shared/patient_folds.csv`、`samples.csv`、`manifest.json` 固定分折。248 名主队列患者、5 个患者级 fold、12,697 条请求；fold 请求数依次 2769、2588、2584、2351、2405。
- 指标：`common.metric.smape_plus_one` 和 `evaluate_oof`；首选比较 `mean_fold_smape1`，同时检查 `std_fold_smape1`、逐折增益以及最终 Kaggle 实测。不要把基于同五折的调参结果当作独立测试。
- 默认数据目录：`amp-parkinsons-disease-progression-prediction/`。默认实验使用 200 epochs；`--epochs 2` 仅用于 smoke test。
- 真实 Kaggle 提交必须以官方 `yiyangsame/basic-api-demo` notebook 为基础编辑/覆盖，保留 `docker_image_pinning_type: original`、`/kaggle/input/amp-pd`、Internet off、官方 `amp_pd_peptide.make_env() → env.iter_test() → env.predict()` 工作流。当前五折实验脚本**不会自动提交 Kaggle**；尚不能声称实测 Top 5。

## 已完成实验与决策

| 实验 | 五折平均 SMAPE+1 ↓ | 相对原 v18 | 判断 |
|---|---:|---:|---|
| 原 v18，λ=1，联合训练 | 62.1387 | 0 | 可靠基准 |
| 推理残差 λ=0 | 63.0158 | +0.8771 | 关闭残差显著退步；残差有用 |
| 推理残差 λ=0.25 | 62.4811 | +0.3424 | 退步 |
| 推理残差 λ=0.50 | 62.1750 | +0.0363 | 近似持平 |
| **推理残差 λ=0.75** | **62.0674** | **−0.0713** | 小幅最优、五折中 4 折改善，但探索性 |
| 两阶段残差训练（基础 100 + 残差 100 epochs） | 62.2062 | +0.0675 | 3/5 折改善、整体退步，不替换基线 |

第一轮结果脚本：`C_improved_model/validate_residual_scale.py`；输出 `residual_scale_results/`。λ=.75 相对 v18 在 Fold 0 退步约 0.045，余下四折改善。结果说明残差网络提供信息，但简单调缩放系数收益太小。

第二轮结果脚本：`C_improved_model/validate_two_stage.py`；输出 `two_stage_results/`。两阶段相对 v18 的五折差值约为 **[+0.3472, −0.1865, −0.2557, +0.4831, −0.0506]**，平均比 v18 差 0.0675。相关实现 `architectures.py` 新增 `residual_two_stage` variant、`teammate_neural_proteins.py` 新增训练模式，**原 joint 默认保持不变**。

## 第三轮实验：五折结果已收到，全部回归融合方案退步

**动机：** 不再反复调整残差强度，而是检验 LightGBM 部分是否更适合连续目标回归。原 v18 的树是分类预测+最小期望 SMAPE 解码；新实验分别训练 `regression_l1`（MAE）、`huber`、`poisson` 三种 `LGBMRegressor`，保留原 v18 的神经输出，按比例混合替换原分类树输出。

- 脚本：**`C_improved_model/validate_tree_regression.py`**。
- 说明：`C_improved_model/TREE_REGRESSION_ABLATION.md`。
- 每个目标测试树替换比例 `0.25,0.5,0.75,1.0`，共 12 个候选 + 原 v18 基准；每折只训练一遍原 v18 神经模型，再训练回归树做预测融合。
- 运行：
  ```bash
  git checkout member_c
  git pull origin member_c
  python C_improved_model/validate_tree_regression.py
  ```
- 结果目录：**`C_improved_model/tree_regression_results/`**，包含 `summary.csv`、`folds.csv`、`oof_predictions.csv`、`metadata.json`；目录已加入 `.gitignore`，默认不会提交结果文件。
- **运行状态：第三轮已经完成，用户上传 `summary(2).csv` 与 `folds(2).csv`，真实五折成绩见下表。** macOS `libomp.dylib` 问题已解决。无需重新运行同一实验。
- **第三轮关键成绩（真实用户上传的五折验证）**：

  | 模型 | 五折平均 SMAPE+1 ↓ | delta_vs_v18 |
  |---|---:|---:|
  | v18 分类树原模型 | **62.138694** | 0 |
  | regression_l1_w0p25 | 62.243914 | +0.105220 |
  | regression_l1_w0p5 | 62.324398 | +0.185704 |
  | regression_l1_w0p75 | 62.388628 | +0.249934 |
  | regression_l1_w1p0 | 62.460047 | +0.321353 |
  | poisson_w0p25 | 63.720266 | +1.581572 |
  | huber_w0p25 | 64.798286 | +2.659592 |
  | poisson_w1p0 | 66.491706 | +4.353012 |
  | huber_w1p0 | 68.532315 | +6.393621 |

  剩余融合权重也全部为正 delta。**12 个候选全部退步；不采用回归树替换分类树。** 最好的新方案是 MAE 25%，但仅 Fold 2 改善（−0.045645）；Fold 0/1/3/4 分别退步 +0.231216、+0.081971、+0.093786、+0.164772。继续优化需要转向其他结构或数据方法，不应再将本轮写成待测。

## 已知技术问题与后续工作优先级

1. **第三轮已确认失败，不需要重新运行。** 下一步应寻找新方向：分析不同 UPDRS target/horizon、临床历史特征构造与训练分布，然后实现第四轮方案及冻结五折比较。
2. 如果没有显著改善，继续推进更大收益可能性的方案，而不是宣布完成：检查训练样本构造/样本权重、各 target/horizon 误差分布与偏差、交叉验证与 Kaggle 推理分布差异、树与神经的目标/时间条件融合、蛋白缺失样本与非缺失样本表现。须控制泄漏与过度调参。允许通过有依据的实验淘汰方案。
3. 本地 LightGBM 有时提示 `X does not have valid feature names`，属 sklearn warning，不影响训练完成，但要保持特征顺序一致。此前缺失 `libomp.dylib` 的 macOS 导入报错**已解决**。
4. 正式 Kaggle Notebook 仍默认原 v18。若第三轮或后续模型胜出，需同步修改官方 demo 中模型定义和训练/推理代码，遵守官方环境限制，取得真实 Kaggle 分数才可判断 Top 5。
5. **2026-11-17 PPT 截止**：保留实验数据与负面结果作为结构/训练消融证据；但用户的明确目标仍是继续提高排名。记录 AI 辅助代码实现，按课程要求披露。
6. 仓库 README 记录了更早的成员 C 实验，其中历史特征 LightGBM 的较低本地分数（例如 58.1587）与 v18 可能在特征可见性/训练设置上不同，**不能未经对齐就将其与 v18 62.1387 当成公平的头对头比较**。在优化时优先调查这个差异，而非仅凭数字断言更好。

## 下个对话接续提示

“阅读 `C_improved_model/PROGRESS_HANDOFF.md`。第三轮 LightGBM 回归融合已完成五折验证，12 个候选全部低于原 v18（v18=62.138694；最好新候选 MAE-25%=62.243914）。macOS libomp 已解决。请直接推进第四轮冲击 Top 5 的新实验，不要重跑已失败的第三轮或建议停止。未经验证不得宣称 Kaggle 排名。”

**文件与协作约定：** 不改 `main`；GitHub 文件变更可直接通过连接的 GitHub 工具提交到 `member_c`。交接文档必须随实验结果更新。所有离线结果与真正 Kaggle 成绩要明确区分。代码由 ChatGPT 辅助实现，须人工审查及按课程要求披露。


## 第四轮：Target × Horizon 条件融合（已提交代码，待运行）

- 实现文件：`C_improved_model/validate_conditional_fusion.py`，仅新增独立离线实验，不修改 v18 Notebook、模型架构或 main。
- 假设：原 V18 固定按目标使用 `ALPHA=[0.625,0.75,0.75,0.375]`，但树和神经网络在不同 horizon 的相对误差可能不同；尝试为 16 个 target/horizon 组合估计单独的融合权重。
- 反泄漏设计：外层仍用冻结五折；每个外层训练患者中按 `random_state=4486+fold` 抽出 25% 主队列患者作为内层校准患者；内层训练排除这些患者（包括其 supplemental 记录）。内层校准只使用这些患者的标签，外层验证患者标签从不用于权重学习。随后在全部外层训练患者上重新训练 V18，使用冻结校准权重预测外层验证患者。
- 16 个权重在 [0,1] 的九点网格上选取，按对应 V18 权重施加二次惩罚，校准样本数少于 30 时回退固定权重；保留每目标/horizon 的树、神经、原融合和新融合诊断。
- 输出：`conditional_fusion_results/{summary.csv,folds.csv,head_diagnostics.csv,oof_predictions.csv,metadata.json}`，该目录应保留为本地产物，不要把 smoke test 视为正式五折。
- 完整运行（200 epochs）：
  ```bash
  git checkout member_c
  git pull origin member_c
  python C_improved_model/validate_conditional_fusion.py
  ```
- **当前状态：仅 GitHub 实验代码提交成功；尚未执行完整五折，尚无新模型成绩，也未提交 Kaggle。** 对五折的反复研发仍是探索性选择，不属于独立外部测试。
- 课程 AI 使用限制：辅助构思、代码检查及调试需披露；组员必须自行理解并审查实现。不可将 AI 输出未经审查地当作作业最终答案。


## 第四轮实测结果（用户于 2026-11-15 上传 summary(3).csv / folds(3).csv）

| 方案 | mean fold SMAPE+1 ↓ | std | pooled OOF | Δ vs V18 |
|---|---:|---:|---:|---:|
| V18 | 62.138694 | 4.414987 | 61.958859 | 0 |
| Target × Horizon conditional | 62.647769 | 4.716971 | 62.456089 | **+0.509075** |

- 五折 conditional 相对 V18 的差值：Fold 0 +0.079693、Fold 1 −0.166492、Fold 2 +1.152504、Fold 3 +0.243139、Fold 4 +1.236531；仅 1/5 折改善。**本轮失败，不采纳条件融合，不重跑。**
- 数据来源为用户上传的真实五折输出，尚未独立复现实验，也没有此次模型的 Kaggle 隐藏评分。
- **下一步优先调查历史 LightGBM 58.1587 与 V18 62.1387 之间的数据可见性/请求构造差异。** 旧方案的临床访问历史可能在离线时利用了 Kaggle 在线迭代 API 无法观察到的访问事件；先对齐合法特征和统一验证，再判断模型收益。不得将分数差直接归因于架构改进。
- 进一步实验应优先分析 Tree-only 与 NN-only 的按目标和时间跨度误差、不同来源训练请求数量/分布、全数据标签分布及蛋白缺失。避免继续扩展第 4 轮条件融合的调参搜索。


## 第五轮：训练样本权重、时间条件、概率温度（脚本已提交，待运行）

新脚本 `C_improved_model/validate_rows_time_temperature.py`，两阶段分别执行，不修改原 V18 和 Kaggle demo：

1. `--stage rows`：3 种来源（omics / winner_origins / winner_origins_supplemental）×3 种权重（uniform / unique_label / patient_equal）×2 组时间特征（original / relative），共 18 组。fixed LightGBM / SMAPE+1 最小风险解码，使用原冻结患者五折，分别保存 `rows_summary.csv`、`rows_folds.csv`、`rows_oof.csv`。
2. `--stage temperature`：指定上阶段选择的 `--source`、`--weight`、`--time`；对每个外层 fold 在训练患者内另外留出 25% 校准患者，在每个 UPDRS 目标内用五个候选温度选择最小 SMAPE 解码损失；再用完整外层训练集重训分类器，评估外层验证患者。输出 `temperature_summary.csv`、`temperature_folds.csv`、`temperature_oof.csv`、`temperatures_fold*.json`。
3. 运行：
```bash
git switch member_c && git pull origin member_c
python C_improved_model/validate_rows_time_temperature.py --stage rows
python C_improved_model/validate_rows_time_temperature.py --stage temperature --source winner_origins --weight unique_label --time original
```
第二条命令的 source/weight/time 是预设示例值；应先根据第一阶段成绩和可用性审计确定，而不是默认它必然最优。用相同的外层五折开发选择仍是探索性模型选择，不能视为独立测试。原 v18 62.1387 用不同模型与历史数据来源，暂不视作严格头对头对照。**当前仅 GitHub 提交代码，没有完整运行、没有新分数。** 学生应自行检查逻辑并按课程规则披露 AI 使用。


## 温度校准快速复核（已提交实现，待运行）

由于第一阶段 18 配置 × 五折 × 四个目标需重复训练 360 个分类器、运行过久，之后默认使用单配置的最小针对性对照。新入口保留现有脚本，新增 `--stage temperature-check`：直接读取上次生成的 `temperatures_fold0.json`…`temperatures_fold4.json`，**每个外层 fold 每目标只训练一次最终 LightGBM，共 20 次 fit**，在相同模型的同一批预测概率上同时计算原始 T=1 和所选温度的 SMAPE 决策。无需重新运行 rows 网格，也不用重新训练内层温度选择器。它严格用于隔离温度变化影响，不会声称提供独立 Kaggle 测试。

```bash
git switch member_c
git pull origin member_c
python C_improved_model/validate_rows_time_temperature.py --stage temperature-check --source omics --weight uniform --time original
```

需要在原输出目录 `C_improved_model/row_time_calibration_results/` 中保留之前的五个 `temperatures_fold*.json`，并确保 `--source/--weight/--time` 与这些 JSON 当时的配置一致。输出：`temperature_check_summary.csv`、`temperature_check_folds.csv`、`temperature_check_pairs.csv`（每折即时保存）、`temperature_check_breakdown.csv`（target × horizon 的翻转率、平均损失差）。**目前只提交代码，尚未验证本地执行和分数。**


## 第六轮：Ordinal LightGBM 最小实验（代码已提交，待实测）

- 文件：`C_improved_model/validate_ordinal_lgbm.py`。默认只做 **Fold 0 / updrs_1**（1 个原多分类 LightGBM + 最多 8 个二分类阈值 LightGBM），避免再次默认执行 18 配置的长时间网格。
- 固定数据和特征：原 `winner_rows_experiment.py` 的 omics-origin、uniform、original；每目标只用外层训练患者。阈值从训练标签的等分位点确定，仅拟合二分类 `P(UPDRS>t|X)`；单调化累积概率后，用训练集内部标签频率分摊每个有序分箱的概率质量，最后使用原 `decision()` 最小期望 SMAPE+1 解码。
- 默认运行：
```bash
git switch member_c && git pull origin member_c
python C_improved_model/validate_ordinal_lgbm.py
```
- 完整五折、四目标仅在试验有明确价值后手动运行：
```bash
python C_improved_model/validate_ordinal_lgbm.py --folds 0 1 2 3 4 --targets updrs_1 updrs_2 updrs_3 updrs_4
```
- 输出目录：`C_improved_model/ordinal_results/`，包含 `ordinal_comparison.csv`、`ordinal_predictions.csv`、`ordinal_summary.csv`、`ordinal_metadata.json`；每个 fold/target 运行后即时写出预测与成绩，防止最后汇总失败导致全部实验丢失。
- 状态：**GitHub 已提交实验脚本，未执行本地数据训练，未产生 Ordinal 分数，更未得到 Kaggle 隐藏评分。** 该阶段仅讨论设计/实现与调试，课程要求组员独立审查并披露 AI 使用。


## 第六轮后续：按用户要求只做两步（2026-11-15）

用户已在本地运行 Ordinal LightGBM 五折×四目标，报告每目标五折均值：UPDRS1 54.543531→54.367013（Δ−0.176518，4/5折胜出）；UPDRS2 63.444628→63.233608（Δ−0.211020，4/5）；UPDRS3 64.879515→64.532445（Δ−0.347070，4/5）；UPDRS4 47.387456→47.244424（Δ−0.143032，4/5）。不允许直接把四项目标分数的算术平均当完整五折指标。

1. **无重训总体 OOF 核算**：新文件 `C_improved_model/analyze_ordinal_oof.py` 从已经完成的 `ordinal_results/ordinal_predictions.csv` 读取并与冻结 `artifacts/shared/samples.csv` 对齐，检查完整五折/四目标、无重复请求、label/fold 匹配。输出 `ordinal_overall_summary.csv`、`ordinal_overall_folds.csv`、`ordinal_target_horizon.csv`（五折平均、逐折差值、fold win rate、overall prediction flip rate）；**不要再训练**。
2. **在线特征可用性检查**：新文件 `C_improved_model/audit_ordinal_online_features.py` 不训练，只将离线使用的临床访问+组学历史与 omics-only conservative replay 对照，输出 `ordinal_online_visibility_audit.csv`。本地代理不是 Kaggle 的实际 `iter_test` 输入；完整 clinical visits 对隐藏患者并非可用信息，因此历史特征必须以在线可观测状态重新构造并使用真实官方 demo 验证后才可提出部署成功/真实 Kaggle 成绩的结论。
```bash
git switch member_c && git pull origin member_c
python C_improved_model/analyze_ordinal_oof.py
python C_improved_model/audit_ordinal_online_features.py
```
- 状态：两份脚本已提交 GitHub；本会话仅收到四个目标汇总数字，没有 `ordinal_predictions.csv` 的文件内容，也没有该用户本机 Kaggle CLI 的认证运行环境，因此 **尚未实测整体 OOF / Kaggle 隐藏评分**。不应声称已提交 Kaggle，也不应盲目把离线特征方案直接写入已通过隐藏评分的官方 notebook。
- 根据用户要求，现阶段不做 Horizon-specific 新实验、不重跑 Ordinal 五折、不改原 V18 和 main。课程 AI 使用规则要求独立审查实现并披露辅助使用。


## Top-5 目标恢复为模型优化（2026-11-15）

用户明确要求 **不是制作 PPT，必须优化模型并通过 Kaggle Private Top-5 分数门槛**。官方历史榜单 Top 5 第五名 Private **60.184**（2023 赛季最终榜）；此前已提交 V7 Private **60.092**，数值已在历史 Top-5 区间，但 V7 是 winner-inspired，尚不能作为原创改进成果；V9 Private 60.106。任何新方案必须在同一 Kaggle notebook 环境通过 API 实测，禁止把 CV 视作 Private。

- V18 已有冻结 OOF 残差消融输出（原本由 `validate_residual_scale.py` 生成，路径 `C_improved_model/residual_scale_results/oof_predictions.csv`，含 λ=0/0.25/0.5/0.75/1）。离线 λ=.75 为 62.0674，相对 λ=1 的 62.1387 改善微弱。
- 新增 **零重训原创实验** `C_improved_model/validate_horizon_gated_residual.py`：测试按 horizon 改变残差系数的轻量推理门控，不再训练 V18 模型、不重新五折。每个 outer fold 用另外四折患者的 OOF 损失为 0/6/12/24 月分别从 λ=0.5、0.75、1 中选候选；相对固定 λ=.75 改善至少 0.10 才使用，否则保持 λ=.75，避免更大的搜索空间。分折选择结果、OOF/逐折/整体 SMAPE 和 flip rate 输出至 `horizon_gate_results/`。这仍是对同一患者五折的探索性二级模型选择，不能宣称独立确认或 Top 5。
```bash
git switch member_c && git pull origin member_c
python C_improved_model/validate_horizon_gated_residual.py
```
- **只有明确稳定改善且特征可部署时**，才考虑把轻量 horizon 残差门控接入 V18 Kaggle 官方 demo；如果没改善立即淘汰，不做新超参数网格。比 60.092 更低才是相对最高实测 Private 的真正突破目标；第五名分数阈值 60.184 只是数值参照，晚提交不等于历史实际排名。
- **执行状态**：实验代码已通过 GitHub 提交，但当前未实际执行，本轮没有新 Private 分数。不要将此前已完成的 69 分 Ordinal 失败继续解释为候选改善。


## V7 直接残差组件（2026-11-15，代码已提交）

- 用户明确说：**“直接针对 v7 设计一个残差组件吧”**，不是改 PPT、再重复 V18 OOF，也不是 Ordinal。
- 新脚本 **`C_improved_model/v7_residual_adapter.py`**：轻量 Ridge 后处理残差校正；原 V7 的预测保持完整，通过 `f_\theta(\hat y_{V7},target,horizon,visit_month)` 估计误差，再做幅度封顶 + 小比例加法 `max(0,\hat y_{V7}+0.25*clip(\Delta,-3,3))`。特征仅来自原 V7 预测值、target、horizon、预测起点月份，Kaggle 官方 `prediction_id` 可恢复，不依赖额外不可见临床历史。Ridge `alpha=100`、未额外搜参。
- **反泄漏关键**：只允许用**由未见过相应患者标签的 V7 模型**生成的完整 12,697 条 OOF 预测拟合。评估用 leave-one-fold-out 拟合残差，分折计分；全量 OOF 拟合最终 `residual_corrector.json`。不能使用 V7 在其训练患者上的 in-sample 预测，以免学习乐观残差。需先检查是否已有 V7 OOF；如果没有，本脚本无法凭 Kaggle Private 分数反推残差，也不应伪造 OOF。
- 输入 schema `prediction_id,prediction[,fold]`。运行 `python C_improved_model/v7_residual_adapter.py --oof PATH_TO_V7_OOF.csv`。输出 `C_improved_model/v7_residual_results/{summary.csv,folds.csv,oof_predictions.csv,residual_corrector.json}`。
- 推理插件用法（需确认真正 V7 Notebook 后接入）：`corrector=V7ResidualCorrector.load("residual_corrector.json")`；`output=corrector.correct_api_output(v7_predict_batch(...))`；输出仍由官方 API `env.predict(output)`。不要把未证实的 `teammate_aligned_model.py` 冒充实际 V7 已评分提交版本。
- **状态**：现仅提交源码，未在实际 V7 OOF 上运行、未检查 Kaggle V7 Notebook 精确版本、未提交 Kaggle、没有改进分数。若缺少 V7 OOF，应该请求准确文件或从 V7 代码生成患者级 OOF，而非盲重训其他模型。


## 重大里程碑：V7 + Residual Correction 线上 Private 60.067（2026-11-15）

**用户明确报告：新提交成功，Kaggle Private SMAPE+1 = 60.067。** 原 V7 Private 60.092；新版本改善 **−0.025（越低越好）**，约为 0.0416% 相对下降。记录为 **user-reported Kaggle result**；尚未在此记录中附 Kaggle kernel version、submission ID、成绩页面截图或外部核验，故不声称独立核实/实际榜单排名。

- 用户直接提供了 `winner_reproduction_submission.ipynb`，此前生成并试跑的 `v7_residual_official_api_submission.ipynb` 以 V7 原始结构为基线，在输出阶段加入轻量级 Ridge 残差校正；输出通过原生官方 API。
- 新版本在 Notebook 内通过独立患者划分获取残差训练/校验数据，在校验未达门槛时退回 V7；实际 Kaggle 测试得到更低 Private 分数，证明至少此次线上提交成功改善了 V7 原有数值表现。
- 将此版本作为当前最优 **已报告**线上结果，保留原始 V7 和新版本 notebook、相关 Kaggle 版本及实验日志；不要覆盖这次结果，也不要继续默认使用较差的 Ordinal（Private 约 69）。
- 课程汇报应按真实方法描述为“V7-derived baseline + bounded residual correction”，明确原创增量是残差后处理及其验证，不将原 V7 本身描述为原创。
- 这一步的成果是模型改进，不代表已经独立确认历史榜单名次或达到任何官方实时排名。
