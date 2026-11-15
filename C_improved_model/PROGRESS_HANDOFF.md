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

## 第三轮实验：已提交代码，等待真实数据验证

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
- **运行状态（截至本次交接）：尚未收到第三轮的五折评分。** 用户此前首次执行时遇到 macOS LightGBM 缺少 `libomp.dylib`，**用户随后明确确认“环境已经解决了”**。所以不再重复建议安装 Homebrew/libomp，而应让用户重跑上面命令、读取第三轮 CSV，并根据真实分数继续优化。
- 检验 `delta_vs_v18 < 0` 的候选、逐折是否稳定；若出现值得继续的方案，下一步需复核接口与移植 Kaggle Notebook，不要直接改原 v18 提交而丢失回退路径。第三轮尚没有经证实的提升，不得捏造数字。

## 已知技术问题与后续工作优先级

1. **首要：获得第三轮真实 `summary.csv`、`folds.csv`**，分析 LightGBM 回归与融合是否改善；不要预先断言有效。用户可以直接上传 CSV，也可以强制提交到 `member_c`。
2. 如果没有显著改善，继续推进更大收益可能性的方案，而不是宣布完成：检查训练样本构造/样本权重、各 target/horizon 误差分布与偏差、交叉验证与 Kaggle 推理分布差异、树与神经的目标/时间条件融合、蛋白缺失样本与非缺失样本表现。须控制泄漏与过度调参。允许通过有依据的实验淘汰方案。
3. 本地 LightGBM 有时提示 `X does not have valid feature names`，属 sklearn warning，不影响训练完成，但要保持特征顺序一致。此前缺失 `libomp.dylib` 的 macOS 导入报错**已解决**。
4. 正式 Kaggle Notebook 仍默认原 v18。若第三轮或后续模型胜出，需同步修改官方 demo 中模型定义和训练/推理代码，遵守官方环境限制，取得真实 Kaggle 分数才可判断 Top 5。
5. **2026-11-17 PPT 截止**：保留实验数据与负面结果作为结构/训练消融证据；但用户的明确目标仍是继续提高排名。记录 AI 辅助代码实现，按课程要求披露。
6. 仓库 README 记录了更早的成员 C 实验，其中历史特征 LightGBM 的较低本地分数（例如 58.1587）与 v18 可能在特征可见性/训练设置上不同，**不能未经对齐就将其与 v18 62.1387 当成公平的头对头比较**。在优化时优先调查这个差异，而非仅凭数字断言更好。

## 下个对话接续提示

“阅读 `C_improved_model/PROGRESS_HANDOFF.md`。当前第三轮 LightGBM 连续回归融合五折实验代码已提交到 GitHub `LJLW1/4486` 的 `member_c` 分支；macOS libomp 问题已解决。请查看我新上传的 `tree_regression_results/summary.csv` 和 `folds.csv`，分析相对 v18 的改进与稳定性，继续提出并实现下一轮冲击 Top 5 的实验，不要只推荐停止。未经验证不得宣称 Kaggle 排名。”

**文件与协作约定：** 不改 `main`；GitHub 文件变更可直接通过连接的 GitHub 工具提交到 `member_c`。交接文档必须随实验结果更新。所有离线结果与真正 Kaggle 成绩要明确区分。代码由 ChatGPT 辅助实现，须人工审查及按课程要求披露。
