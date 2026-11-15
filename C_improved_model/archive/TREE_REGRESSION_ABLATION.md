# V18 进一步优化：LightGBM 回归目标与融合

原始 v18 将 UPDRS 分数当作分类标签，使用 LightGBMClassifier 的类别概率并选择预期 SMAPE+1 风险最小的整数输出。新实验**不改动已有 v18 模型与 Kaggle Notebook**，仅检验另一个假设：将 UPDRS 视作有序非负连续值，训练回归树，是否能够改善与神经网络的融合。

## 运行

```bash
git pull origin member_c
python C_improved_model/validate_tree_regression.py
```

可用 `--data-dir` 和 `--output-dir` 指定路径；`--epochs 2` 仅用于流程检查，不是可比较成绩。正式运行 `--epochs 200`，每折原 V18 的 3 个残差网络仍按固定种子训练一次，然后分别拟合 MAE（`regression_l1`）、Huber、Poisson 三个回归树，评估新回归树替换原分类树的 25%、50%、75%、100%。

保持原 LightGBM 配置（300 棵树、7 叶、学习率 .03、正则）和固定 5 折患者验证，训练只使用非 held-out 患者。v18 原预测及其神经部分原封不动；融合公式为：

`pred = max(0, v18_pred + (1 - ALPHA[target]) * weight * (regression_tree - original_classification_tree))`

输出 `C_improved_model/tree_regression_results/{summary.csv,folds.csv,oof_predictions.csv,metadata.json}`。观察 `delta_vs_v18`，负值表示提升；逐折检查改善是否稳定。由于多个候选方案共享同一五折验证集，最优组合属于探索性选择而非独立泛化证明。

即便某个候选得分更好，也**不能自动认为达到 Kaggle Top 5**：还需将最终模型移植至官方 demo 的原生推理 API，并取得真实公开/私有榜评分；目前此脚本不提交 Kaggle。使用 AI 辅助实现，请组员复核并按课程要求披露。
