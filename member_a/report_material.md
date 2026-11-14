# Member A technical material

This is the Member A contribution for review and integration. Codex assisted with the design, implementation, execution and writing. The group must disclose that assistance and describe its own actual independent work accurately.

## Task and data analysis

The task estimates four UPDRS outcomes at the current visit and 6, 12 and 24 months ahead. The main clinical table has 2,615 visits from 248 patients. The molecular tables contain 232,741 protein rows for 227 proteins and 981,834 peptide rows for 968 peptide sequences, spanning 1,113 sampling visits. Measurement rows are not independent patients. The supplemental table has 2,223 clinical rows from 771 additional patients, but this branch does not use it for training.

Clinical and molecular visit coverage differ. There are 45 molecular visits without a same-month clinical row. The frozen common protocol retains them as origins for any available future labels rather than losing them through an inner join on current clinical records. There are 12,697 labelled scalar requests after exact month and target matching.

UPDRS 4 has 39.694% missing labels, and 60.938% of its observed scores equal zero. Missing values therefore remain distinct from observed zeros. Although the recorded molecular abundance rows contain no null abundance values, complete visit-by-analyte matrices have approximately 7.880% missing protein cells and 8.869% missing peptide cells. Figures 01 to 03 document repeated follow-up, target distributions and variable panel coverage. These statements describe the main training tables, not the hidden test set.

## Validation and metric

All experiments use the existing frozen five-fold patient partition with seed 4486. The validation folds contain 50, 50, 50, 49 and 49 patients. The exact split_id is `5445dbd361f26a5fcdc2e1a6eeda1a2d1387942264957eb9e17844ff9163d550`. Samples have one observed target per prediction_id, and missing target values have no scoring request. Every patient belongs to one fold. This assesses held-out patients rather than the future of known training patients.

We call common.interface.model_folds without modifying the common modules. Its fold-specific vocabulary supplies the allowed protein and peptide columns. Within a visit, summaries use log1p abundance and only allowed analytes. Historical features use measurements at or before the origin. Clinical score values, medication state, clinical visit frequency, patient IDs and future visit counts do not enter the model. PCA fitting uses distinct training sampling visits, with training-only imputation and scaling. No validation-label early stopping is used.

The competition-aligned SMAPE+1 convention adds one to truth and prediction before the symmetric percentage error. The shared implementation computes 200 times the mean of abs(y-p)/(abs(y+1)+abs(p+1)). Missing truth is masked and real zeros are retained. Prediction clipping is explicit in the A model branch, not hidden in the common metric. All OOF predictions are aligned by the shared evaluator, which rejects missing, duplicate or extra IDs.

The primary comparison is mean_score, the arithmetic average of the five fold scores. The secondary oof_score pools all labelled requests. The fold standard deviation uses ddof=0 and is descriptive rather than a confidence interval.

## Feature branch

Each configuration fits a separate HistGradientBoostingRegressor for each target. Fixed parameters are MAE loss, learning rate 0.05, 120 iterations, 7 leaves, maximum depth 3, minimum leaf size 20, L2 regularization 5, no early stopping and random state 4486. Evaluation uses SMAPE+1. No pretrained model or ensemble is used.

A0 uses visit_month, horizon and target_month. A1 adds 14 current-visit summaries: count, mean, median, standard deviation, minimum, maximum and IQR for each modality. Abundances use log1p, while count remains a count. A2 adds 22 historical features: previous-sample changes, changes from the first sample and cumulative means for mean, median and count, plus sampling count and span for each modality. A3 adds 16 protein principal components fitted within the training fold.

| experiment | mean_score | std_score | oof_score | delta_vs_A0 |
| --- | --- | --- | --- | --- |
| A0 | 65.621 | 3.329 | 65.484 | 0.000 |
| A1 | 66.126 | 3.110 | 65.990 | 0.505 |
| A2 | 66.754 | 3.108 | 66.632 | 1.133 |
| A3 | 66.816 | 3.809 | 66.650 | 1.195 |

The fixed model benefits most from the simple time representation among the tested configurations. Molecular summaries increase the five-fold mean by 0.505205, although fold 2 improves slightly. Adding history increases the primary score by another 0.627866, and adding PCA increases it by another 0.062028. These specific additions provide no overall benefit for the fixed model and split. The experiments do not identify the cause and do not imply that proteins lack biological relevance.

## Reproducibility and limitations

All four OOF files cover exactly 12,697 frozen prediction IDs and are evaluated with the unchanged common evaluator. Existing frozen files retain their original hashes. Eighteen tests pass, including future-perturbation and truncation checks, ignored validation-only analytes, missing-current-visit behavior and duplicate-request-invariant PCA fitting. Twenty saved fold bundles reproduce their OOF predictions within 1e-12 during the recorded verification. Model bundles are ignored by Git and regenerated by the training command.

The local protocol is an offline development proxy and does not include all missing-omics requests exposed by the time-series API. This branch does not implement an official submission loop or claim an official score or rank. Model selection on shared development folds also introduces selection bias. The old seed-42 local branch used a different sample population and metric aggregation, so its scores are excluded from this comparison.

## Sources and disclosure

Data, sample construction, patient folds, metric and model interface come from this repository's common modules and frozen artifacts at base commit 49ce4bf042e982240e9be8f06edbc09e94c9019a. Competition setup: https://www.kaggle.com/competitions/amp-parkinsons-disease-progression-prediction . The participant explanation of the SMAPE+1 update is https://www.kaggle.com/competitions/amp-parkinsons-disease-progression-prediction/discussion/417124 . The overview still calls its metric SMAPE, so the metric-update source is stated separately. No public notebook was copied for this branch. AI assistance and actual student contributions must be disclosed under the course policy.
