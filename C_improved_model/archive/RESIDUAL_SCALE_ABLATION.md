# V18 残差强度消融（member_c）

目的：检验已训练 V18 的蛋白残差修正幅度是否过强。保留原 LightGBM + 3 随机种子残差网络、历史和蛋白特征、训练策略、融合权重及原生 Kaggle API，只在验证推理时改变残差倍数 λ。

**公式**：`prediction = clamp(base(features) + λ * correction(features, proteins) * protein_available, min=0)`。默认 `λ=1` 与原 V18 完全一致；`λ=0` 关闭已训练模型的修正，但不等于训练一个无残差模型。

## 变更文件

- `architectures.py`：只新增可调 `residual_scale`，默认 1。
- `architecture_residual_submission.ipynb`：同步上述两处改动，API 及训练配置未变；默认仍为 V18。
- `validate_residual_scale.py`：冻结患者五折交叉验证，每折只训练一次，推理对比 `0, 0.25, 0.5, 0.75, 1.0`。
- `test_residual_scale.py`：无需比赛数据的单元测试。

## 执行

在仓库根目录安装公共依赖、PyTorch 和 LightGBM：

```bash
python -m pip install -r requirements.txt
python -m pip install torch lightgbm
python -m unittest discover -s C_improved_model -p "test_residual_scale.py" -v
python C_improved_model/validate_residual_scale.py
```

正式实验使用默认 `--epochs 200`，种子为 4486、4487、4488；原始数据目录默认为仓库内 `amp-parkinsons-disease-progression-prediction/`。可使用 `--data-dir`、`--output-dir` 指定路径。

运行 `python C_improved_model/validate_residual_scale.py --epochs 2` 只用于快速确认流程，不能将 2-epoch 分数与正式 V18 比较。无需先运行全数据 Kaggle 提交 Notebook。

## 输出与限制

默认输出目录 `C_improved_model/residual_scale_results/`：

- `summary.csv`：五折 SMAPE+1 均值、标准差、合并 OOF 分数、相对 λ=1 的变化。
- `folds.csv`：逐折分数、蛋白可用/缺失样本分数及样本量。
- `oof_predictions.csv`：所有 12,697 条冻结请求的每个 λ 对应的预测。
- `metadata.json`：固定 split ID、训练配置、随机种子和实验局限。

验证请求和患者映射直接使用 `artifacts/shared/` 中的 `samples.csv`、`patient_folds.csv`、`manifest.json`，不会重划分。训练仍用 V18 的临床访问训练请求与 supplemental，排除 held-out 患者；验证使用冻结的组学访问请求。不同 λ 共享完全相同的训练权重。SMAPE+1 越低越好，建议观察逐折改善是否稳定。

这些验证折同时用于选择 λ，因此最优五折结果只能视作探索性证据，不是独立测试结果。脚本不会提交 Kaggle 或改变比赛隐藏成绩；若需提交，按项目 `AGENTS.md` 使用已验证的 `yiyangsame/basic-api-demo` 原始运行环境和原生 API 追加版本。

代码设计与调试有 ChatGPT 辅助；组员须审查并理解实验，按课程要求披露 AI 使用。
