# Member A presentation material

These three content slides are for C to merge into the course deck. Approximately 100 seconds. Review and revise the material, and disclose AI assistance.

## Slide 1 Task and data

- Four UPDRS scores at 0, 6, 12 and 24 months.
- 248 patients, 2,615 clinical visits and 1,113 molecular sampling visits.
- 227 proteins and 968 peptide sequences.
- UPDRS 4 has 39.7% missing labels. Of observed values, 60.9% are zero.

Visual: eda/figures/02_targets.png.

Speaker notes: Our task estimates four UPDRS outcomes over several prediction horizons. The main dataset has only two hundred and forty-eight patients, despite almost one million peptide measurements. Repeated molecular rows are not independent patients. UPDRS four has both substantial missingness and many observed zeros, so we preserve the distinction during scoring.

## Slide 2 Validation and metric

- Existing patient-grouped five folds, seed 4486.
- 12,697 labelled scalar requests from molecular origins.
- Exact future-month labels and time-bounded molecular history.
- Main comparison: five-fold mean SMAPE+1. Pooled OOF is a separate statistic.

Footer: Offline development CV, not an official leaderboard score.

Speaker notes: We reuse the team's frozen patient split and request table. A patient never appears in both training and validation. Predictions use only measurements available at the origin, while PCA uses training patients only. The shared evaluator aligns every prediction ID. Our main comparison is the mean of five fold scores, which differs from pooling every request into one average.

## Slide 3 Controlled feature ablations

| experiment | mean_score | oof_score | delta_vs_A0 |
| --- | --- | --- | --- |
| A0 | 65.621 | 65.484 | 0.000 |
| A1 | 66.126 | 65.990 | 0.505 |
| A2 | 66.754 | 66.632 | 1.133 |
| A3 | 66.816 | 66.650 | 1.195 |

A0 has 3 known time features. A1 adds summaries, A2 adds molecular history, and A3 adds protein PCA. The model is fixed across configurations.

Speaker notes: Time features score sixty-five point six two one on the team's primary metric. Molecular summaries increase that error by about half a point, and history and PCA do not improve the overall score. We retain A zero as the candidate for this feature branch. This result applies to the tested representations and model, rather than proving that the molecular measurements have no biological value.

Sources for slide notes: common modules, artifacts/shared/manifest.json, member_a/eda/audit.json, target_statistics.csv, results/comparison.csv and results/verification.json. Metric-update source: Kaggle discussion/417124. No official submission was performed.

## 答辩要点

1. 分折为什么不是 seed=42？仓库公共协议冻结为 4486，三条路线必须保持一致，旧本地结果不参与比较。
2. 为什么不把 65.484 写成主成绩？它是 pooled OOF；本仓库主要比较五折等权均分 65.621。
3. 当前访视没有临床标签还能预测吗？可以。当前组学采样是原点，未来目标月份有标签就能构成请求，不能通过同月 inner join 丢失。
4. 为什么不把临床访视次数作为特征？本分支只使用公共接口允许的组学历史和已知时间字段，没有依赖预测时未核实可见的临床日程。
5. PCA 为什么每折做？词表、中位数填补、缩放和主成分都不能从验证患者学习。
6. 新特征变差说明什么？仅说明当前固定模型和协议下没有总体收益，不能推断医学或生物学因果结论。
7. 这些是官方成绩吗？不是。这是当前冻结样本的开发 CV，尚未运行本分支的官方提交。
8. 个人贡献怎么写？如实记录本人复现、理解、修改和独立完成的工作，并披露 Codex 对代码、执行和文字的协助，不虚构工时。
