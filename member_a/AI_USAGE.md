# AI assistance and provenance

Codex assisted with Member A's implementation, experiment execution, data analysis, tests and writing. The repository's existing common modules and frozen artifacts are the authoritative protocol. No new patient split, sample table or metric implementation is introduced.

The previous local seed-42 A branch informed the feature hypotheses, but its folds, scores, clinical-anchor samples and clinical-schedule features are not included in this branch. After inspecting the actual GitHub repository, Member A was adapted and rerun using the existing seed-4486 omics-origin protocol. The two sets of scores must not be combined.

No public Kaggle notebook was copied, forked or rewritten. Standard scikit-learn regressors and PCA implement four fixed feature configurations. No supplemental training, pretrained model or ensemble is used. No official Kaggle submission was performed.

Course rules require accurate AI disclosure and prohibit taking a complete AI-generated solution verbatim as independent work or copying implementation details without understanding them. Members must review, understand, reproduce and revise these materials before course use. The code's original AI assistance remains part of its provenance even after review. Independent designs, productive hours and contribution statements must describe actual student work; none are fabricated here.

Sources: existing repository common/data.py, common/interface.py, common/metric.py and frozen artifacts at base commit 49ce4bf042e982240e9be8f06edbc09e94c9019a; locally downloaded Kaggle training tables; the course's project instruction and report template. Metric-update explanation: https://www.kaggle.com/competitions/amp-parkinsons-disease-progression-prediction/discussion/417124 . This is a participant-authored explanation, distinct from the competition overview that still labels the metric SMAPE.
