# CS4486 — Parkinson's Disease Progression Prediction
## Presentation storyline and evidence sheet (Member C; 2026-11-15)

**Scope:** Suggested material for group discussion and manual assembly into the official `Final_Presentation_Template_CS4486.pptx`. This is not a completed, submission-ready coursework deck. The team must check every claim, fill in team identities and their own data, approve authorship attribution, and disclose AI assistance as required.

### Presentation thesis
**From a competition-inspired predictor to a protein-aware residual correction model: rigorous patient-level validation, ablation, and a realistic assessment of online generalization.**

Course assignment explicitly asks to *reproduce and improve*, including data formulation, baseline, improvements, validation, reproducibility, and reflection. Do not claim that the V7 winner-inspired implementation or its algorithm is original.

### Suggested 8 content slides mapped to official template

1. **Task and metric** — Predict UPDRS 1–4 at horizons 0,6,12,24 months using longitudinal Parkinson's biomarker visits. Metrics: SMAPE+1 (lower better). Show data-to-prediction diagram. Clarify target labels are not test-time features.
2. **Data and validation** — Longitudinal repeated patient visits, peptide/protein measurements and missingness, additional clinical/supplemental visits. Fixed 5-fold **patient** CV with 248 main patients and 12,697 prediction requests; no patient overlap across fold train/validation. Challenge: visit histories seen offline may differ from online `iter_test`.
3. **Reproducible baseline** — Winner-inspired LightGBM classification + expected SMAPE optimal decoder; known Kaggle private result V7 60.092, V9 60.106 (these are versions in the group's prior run logs, not a score attributed to V18). Acknowledge competition-inspired components explicitly.
4. **Proposed architecture: V18** — Shared temporal input -> baseline `ScoreNetwork` producing 16 UPDRS/horizon scores; two selected proteins P05060, O15240, normalized with presence indicators -> 16-dimensional zero-initialized correction branch -> correction multiplied by protein availability mask -> `clamp_min(0)`; ensemble three seeds [4486,4487,4488], 200 epochs. Final weighted combination with LightGBM, target-specific alpha [0.625,0.75,0.75,0.375]. Cite `C_improved_model/architectures.py`, `teammate_neural_proteins.py`. Do NOT call general residual networks, tree ensembles, or winner methods novel inventions.
5. **Core ablation evidence — residual branch** — All same frozen 5 folds, mean-fold SMAPE+1: residual scale lambda=0: 63.0158; lambda=1 original V18: 62.1387; lambda=0.75: 62.0674 (small exploratory improvement, four folds improved). Main defensible result is lambda=0 vs 1 (+0.8771 worse without correction); avoid claiming statistical significance. Two-stage training: 62.2062 (worse).
6. **Other explored improvements / negative results** — Target×horizon conditional fusion 62.6478 vs V18 62.1387 (+0.5091 worse); LightGBM MAE 25% substitution 62.2439 (+0.1052 worse); all 12 regression replacement options worse. Samples/time ablation: best 58.5600 vs its own LightGBM baseline 58.5716 (−0.0117; weak; **different feature protocol from V18, not comparable**). Ordinal under its original offline protocol 58.3447 vs multiclass 58.5716 (−0.2269, 4/5 fold wins), yet its changed online submission got Private around 69 (**different feature construction and evaluation; do not claim exact causal attribution**).
7. **Results and competition scores** — Keep separate panels: **CV:** V18 62.1387, lambda=.75 62.0674, lambda=0 63.0158; **Kaggle Private:** V7 60.092, V9 60.106; recent ordinal revised deployment ~69. No confirmed Kaggle private score for V18 in current evidence; never fabricate. Do not imply the lowest offline CV candidate equals best Kaggle model, or that private ~60 reaches Top 5 without leaderboard rank proof.
8. **Reflection, limitations, AI usage and reproducibility** — Genuine improvement is zero-initialized protein-conditioned residual correction and its ablation evidence; not every proposed change helps. Online feature audit: seen_6_18_count differed for 6,548/12,697 (~51.57%) when replacing full clinical-visit records with a conservative omics-visit proxy; not direct measurement of actual hidden API exposure. Need honest disclosure of ChatGPT use in experimentation/debugging/presentation support, independent student review, seeds, patient folds, API constraints and Kaggle run versions.

### Presentation-ready evidence table
| Experiment | Comparison | Mean five-fold SMAPE+1 | Defensible interpretation |
|---|---|---:|---|
| V18 baseline | residual lambda=1 | 62.1387 | Frozen reference |
| Residual disabled | lambda=0 | 63.0158 | Residual correction helps within this CV protocol |
| Reduced residual | lambda=.75 | 62.0674 | Small exploratory gain; not proof of external improvement |
| Two-stage | base 100 + correction 100 | 62.2062 | Worse than joint training |
| Conditional fusion | target×horizon weighting | 62.6478 | Worse |
| MAE tree replacement | 25% | 62.2439 | Worse |
| Ordinal offline study | multiclass 58.5716 -> ordinal 58.3447 | 58.3447 | Different protocol; **not directly comparable with V18** |
| Ordinal online revision | updated streaming features | Kaggle Private ~69 | Fails compared with previous Kaggle versions; not same as offline ordinal setup |

### Attribution/accuracy constraints
- Say **“our implementation/experiment”**, not “we invented residual learning”, “we outperform the winner” or “Top 5 achieved”.
- A good **novel contribution** in a class project can be a carefully motivated architecture adaptation + controlled experiments; no evidence of first-in-literature novelty is claimed.
- V18's baseline `ScoreNetwork` and inherited history/ensemble pipeline should be described as adapted/extended, not designed entirely from scratch.
- The provided current evaluation cannot establish whether residual correction improves Kaggle Private versus the strongest V7/V9 version; CV protocols and private scores have different roles.
- The course's AI policy prohibits using a generated, complete solution without understanding or independent revision. This file is an internal evidence sheet, not a slide deck for submission.

### What group must verify and add before Nov 17
- Group member names, IDs, speaking roles, competition link and achieved rank from Kaggle.
- Actual EDA visuals: target distribution, protein missingness, fold counts.
- Visual diagram traced to the submitted V18 notebook, not merely Python draft classes.
- Confirm exact Kaggle version and score for any version described as the final submitted model.
- Full AI disclosure and the actual sources credited in presentation.

### Source code / reproducibility
- `C_improved_model/architectures.py` (zero-init and protein-gated residual).
- `C_improved_model/teammate_neural_proteins.py` (training / tree fusion).
- `C_improved_model/validate_residual_scale.py` (residual ablation).
- `C_improved_model/validate_two_stage.py`.
- `C_improved_model/validate_tree_regression.py`.
- `C_improved_model/validate_conditional_fusion.py`.
- `C_improved_model/validate_ordinal_lgbm.py`, `analyze_ordinal_oof.py`, `audit_ordinal_online_features.py`.
