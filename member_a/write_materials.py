"""Generate A's review materials from this repository's completed experiments."""
import contextlib
import io
import json
import os
import uuid
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/cs4486-member-a-repo-mpl')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def table(frame):
    def fmt(v):
        return f'{v:.3f}' if isinstance(v,(float,np.floating)) else str(v)
    return '\n'.join(['| '+' | '.join(frame.columns)+' |',
        '| '+' | '.join(['---']*len(frame.columns))+' |',
        *['| '+' | '.join(fmt(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None)]])


def main():
    base=Path(__file__).resolve().parent;root=base.parent
    result=pd.read_csv(base/'results/comparison.csv')
    shared=json.loads((root/'artifacts/shared/manifest.json').read_text())
    verification=json.loads((base/'results/verification.json').read_text())
    assert verification['status']=='passed'
    results_table=table(result[['experiment','mean_score','std_score','oof_score','delta_vs_A0']])
    plt.rcParams.update({'font.family':'DejaVu Sans','axes.spines.right':False,'axes.spines.top':False})
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained')
    ax.bar(result.experiment,result.mean_score,color=['#375B82','#2A9D8F','#E9B44C','#C96B5C'],width=.65)
    for i,row in result.iterrows():ax.text(i,row.mean_score+.5,f'{row.mean_score:.3f}',ha='center',fontweight='bold')
    ax.set(ylim=(0,77),ylabel='Five-fold mean SMAPE+1 (lower is better)',title='Member A feature ablations on the frozen shared split')
    fig.savefig(base/'eda/figures/04_ablation.png',dpi=200);plt.close(fig)
    readme=f'''# Member A 数据分析与特征实验

本路线直接接入仓库现有公共基础，没有另一套分折、标签构造或评分函数。使用 seed=4486 和协议 `{shared['protocol']}`。所有结果来自真实运行；代码、实验执行和文字整理均有 Codex 协助，使用前需复现理解并按课程要求披露。

## 快速查看

- `results/comparison.csv`：四组固定模型完整五折结果。
- `results/A0/oof.csv` 至 `results/A3/oof.csv`：每个文件包含完整 12,697 条 `prediction_id,rating` 预测。
- `eda/figures/`：三张数据分析图和一张特征实验图。
- `report_material.md`：英文技术素材。
- `slides.md`：三页展示内容及讲稿。
- `walkthrough.ipynb`：已执行的入门 Notebook。
- `AI_USAGE.md`：真实来源和 AI 协助说明。

## 真实实验结果

{results_table}

`mean_score` 是组内主要比较分数，等权平均五折；`oof_score` 是所有请求合并后的分数。不能互换。A0 只使用预测时已知的三个时间字段；A1 增加 14 个当前采样汇总；A2 增加 22 个组学历史特征；A3 增加 16 个训练折内蛋白主成分。

四组固定相同的 HistGradientBoostingRegressor 配置。扩展特征没有改善主要比较分数，保留简单 A0 作为本路线候选。这不是对生物学作用的否定，也不是最终整组模型结论。

## 运行

在仓库根目录运行，推荐 Python 3.11：

```bash
python -m pip install -r member_a/requirements.txt
python -m unittest discover -s tests -v
python -m member_a.verify
```

完整重训并重现保存模型：

```bash
python -m member_a.run --output-dir member_a/results_reproduced
python -m member_a.verify --results-dir member_a/results_reproduced --require-models
```

输出目录必须不存在。训练默认只写本次输出目录中的实验日志。`--append-shared-log` 是显式选项；现有四个 experiment ID 已登记，重复登记会报错，重新运行用于复现时不要加该选项。

joblib 模型不进入 Git，完整重训可生成 20 个折模型包；当前提交提供 OOF、分数和实测验证记录。没有模型包时 `verify` 仍会核验完整请求覆盖、冻结哈希和重算分数，并明确记录模型核验未运行；加 `--require-models` 则要求模型存在并重现每条预测。只加载本人生成的 joblib 文件。

重新生成 EDA 和说明：

```bash
python -m member_a.eda --output-dir member_a/eda_reproduced
python -m member_a.write_materials
```

`write_materials` 读取已验证的默认 `member_a/results` 和默认 EDA 图目录，不执行训练。文档、图和 Notebook 是分析材料，不替代整组报告。

## 与公共协议对齐

1. 不生成或覆盖 `patient_folds.csv`、`samples.csv`、`fold_summary.csv`、`manifest.json`。仅调用 `common.prepare.prepare` 核验它们。
2. 使用 `common.interface.model_folds`，共享同一个 `split_id`：`{shared['split_id']}`。
3. 标签按单项 `prediction_id` 组织，仅预测已冻结请求。起点是蛋白与肽采样访视并集，包含无同月临床标签但未来有评分的请求。
4. 评分统一调用 `common.metric.evaluate_oof`。A 分支自己将负预测显式裁剪到 0；共享评分函数行为保持原样。
5. 特征仅使用组学历史和已知时间，不使用临床评分、用药状态、临床访视次数或未来采样信息。训练患者确定组学词表；PCA 中位数填补、标准化和基向量全部在训练折拟合。
6. 不使用补充数据训练、预训练模型、融合或验证折早停。

## 与旧版 A 工作包的区别

先前本地工作包使用 seed=42、临床 anchor 和 pooled OOF 主指标，其旧成绩不可与本仓库直接比较。本目录已重新使用当前仓库协议运行四组实验。旧工作包代码、旧折文件、旧分数和旧报告不纳入本次 GitHub 提交。

## 已知边界

离线评分只纳入冻结的有组学起点，不覆盖时序 API 的全部无组学请求分布。未为此分支实现正式的 API 提交循环，也未执行 Kaggle 提交。多组配置共用开发折，选择最优配置后也存在开发选择偏差。不能由这些成绩直接推断官方分数、名次或 Top 5。

基础依赖版本保持根目录 requirements.txt 的 numpy 1.26.4、pandas 2.3.3、scikit-learn 1.6.1、filelock 3.19.1。A 的额外依赖单独列出，没有覆盖公共依赖。实际执行环境记录在 results/manifest.json。
'''
    (base/'README.md').write_text(readme,encoding='utf-8')
    report=f'''# Member A technical material

This is the Member A contribution for review and integration. Codex assisted with the design, implementation, execution and writing. The group must disclose that assistance and describe its own actual independent work accurately.

## Task and data analysis

The task estimates four UPDRS outcomes at the current visit and 6, 12 and 24 months ahead. The main clinical table has 2,615 visits from 248 patients. The molecular tables contain 232,741 protein rows for 227 proteins and 981,834 peptide rows for 968 peptide sequences, spanning 1,113 sampling visits. Measurement rows are not independent patients. The supplemental table has 2,223 clinical rows from 771 additional patients, but this branch does not use it for training.

Clinical and molecular visit coverage differ. There are 45 molecular visits without a same-month clinical row. The frozen common protocol retains them as origins for any available future labels rather than losing them through an inner join on current clinical records. There are 12,697 labelled scalar requests after exact month and target matching.

UPDRS 4 has 39.694% missing labels, and 60.938% of its observed scores equal zero. Missing values therefore remain distinct from observed zeros. Although the recorded molecular abundance rows contain no null abundance values, complete visit-by-analyte matrices have approximately 7.880% missing protein cells and 8.869% missing peptide cells. Figures 01 to 03 document repeated follow-up, target distributions and variable panel coverage. These statements describe the main training tables, not the hidden test set.

## Validation and metric

All experiments use the existing frozen five-fold patient partition with seed 4486. The validation folds contain 50, 50, 50, 49 and 49 patients. The exact split_id is `{shared['split_id']}`. Samples have one observed target per prediction_id, and missing target values have no scoring request. Every patient belongs to one fold. This assesses held-out patients rather than the future of known training patients.

We call common.interface.model_folds without modifying the common modules. Its fold-specific vocabulary supplies the allowed protein and peptide columns. Within a visit, summaries use log1p abundance and only allowed analytes. Historical features use measurements at or before the origin. Clinical score values, medication state, clinical visit frequency, patient IDs and future visit counts do not enter the model. PCA fitting uses distinct training sampling visits, with training-only imputation and scaling. No validation-label early stopping is used.

The competition-aligned SMAPE+1 convention adds one to truth and prediction before the symmetric percentage error. The shared implementation computes 200 times the mean of abs(y-p)/(abs(y+1)+abs(p+1)). Missing truth is masked and real zeros are retained. Prediction clipping is explicit in the A model branch, not hidden in the common metric. All OOF predictions are aligned by the shared evaluator, which rejects missing, duplicate or extra IDs.

The primary comparison is mean_score, the arithmetic average of the five fold scores. The secondary oof_score pools all labelled requests. The fold standard deviation uses ddof=0 and is descriptive rather than a confidence interval.

## Feature branch

Each configuration fits a separate HistGradientBoostingRegressor for each target. Fixed parameters are MAE loss, learning rate 0.05, 120 iterations, 7 leaves, maximum depth 3, minimum leaf size 20, L2 regularization 5, no early stopping and random state 4486. Evaluation uses SMAPE+1. No pretrained model or ensemble is used.

A0 uses visit_month, horizon and target_month. A1 adds 14 current-visit summaries: count, mean, median, standard deviation, minimum, maximum and IQR for each modality. Abundances use log1p, while count remains a count. A2 adds 22 historical features: previous-sample changes, changes from the first sample and cumulative means for mean, median and count, plus sampling count and span for each modality. A3 adds 16 protein principal components fitted within the training fold.

{results_table}

The fixed model benefits most from the simple time representation among the tested configurations. Molecular summaries increase the five-fold mean by 0.505205, although fold 2 improves slightly. Adding history increases the primary score by another 0.627866, and adding PCA increases it by another 0.062028. These specific additions provide no overall benefit for the fixed model and split. The experiments do not identify the cause and do not imply that proteins lack biological relevance.

## Reproducibility and limitations

All four OOF files cover exactly 12,697 frozen prediction IDs and are evaluated with the unchanged common evaluator. Existing frozen files retain their original hashes. Eighteen tests pass, including future-perturbation and truncation checks, ignored validation-only analytes, missing-current-visit behavior and duplicate-request-invariant PCA fitting. Twenty saved fold bundles reproduce their OOF predictions within 1e-12 during the recorded verification. Model bundles are ignored by Git and regenerated by the training command.

The local protocol is an offline development proxy and does not include all missing-omics requests exposed by the time-series API. This branch does not implement an official submission loop or claim an official score or rank. Model selection on shared development folds also introduces selection bias. The old seed-42 local branch used a different sample population and metric aggregation, so its scores are excluded from this comparison.

## Sources and disclosure

Data, sample construction, patient folds, metric and model interface come from this repository's common modules and frozen artifacts at base commit 49ce4bf042e982240e9be8f06edbc09e94c9019a. Competition setup: https://www.kaggle.com/competitions/amp-parkinsons-disease-progression-prediction . The participant explanation of the SMAPE+1 update is https://www.kaggle.com/competitions/amp-parkinsons-disease-progression-prediction/discussion/417124 . The overview still calls its metric SMAPE, so the metric-update source is stated separately. No public notebook was copied for this branch. AI assistance and actual student contributions must be disclosed under the course policy.
'''
    (base/'report_material.md').write_text(report,encoding='utf-8')
    slides=f'''# Member A presentation material

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

{table(result[['experiment','mean_score','oof_score','delta_vs_A0']])}

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
'''
    (base/'slides.md').write_text(slides,encoding='utf-8')
    (base/'AI_USAGE.md').write_text('''# AI assistance and provenance

Codex assisted with Member A's implementation, experiment execution, data analysis, tests and writing. The repository's existing common modules and frozen artifacts are the authoritative protocol. No new patient split, sample table or metric implementation is introduced.

The previous local seed-42 A branch informed the feature hypotheses, but its folds, scores, clinical-anchor samples and clinical-schedule features are not included in this branch. After inspecting the actual GitHub repository, Member A was adapted and rerun using the existing seed-4486 omics-origin protocol. The two sets of scores must not be combined.

No public Kaggle notebook was copied, forked or rewritten. Standard scikit-learn regressors and PCA implement four fixed feature configurations. No supplemental training, pretrained model or ensemble is used. No official Kaggle submission was performed.

Course rules require accurate AI disclosure and prohibit taking a complete AI-generated solution verbatim as independent work or copying implementation details without understanding them. Members must review, understand, reproduce and revise these materials before course use. The code's original AI assistance remains part of its provenance even after review. Independent designs, productive hours and contribution statements must describe actual student work; none are fabricated here.

Sources: existing repository common/data.py, common/interface.py, common/metric.py and frozen artifacts at base commit 49ce4bf042e982240e9be8f06edbc09e94c9019a; locally downloaded Kaggle training tables; the course's project instruction and report template. Metric-update explanation: https://www.kaggle.com/competitions/amp-parkinsons-disease-progression-prediction/discussion/417124 . This is a participant-authored explanation, distinct from the competition overview that still labels the metric SMAPE.
''',encoding='utf-8')
    cells=[]
    def md(s):cells.append({'cell_type':'markdown','id':uuid.uuid4().hex[:8],'metadata':{},'source':s.splitlines(keepends=True)})
    def code(s):cells.append({'cell_type':'code','id':uuid.uuid4().hex[:8],'metadata':{},'source':s.splitlines(keepends=True),'execution_count':None,'outputs':[]})
    md('# Member A 使用仓库公共协议\n\n先认识 patient 患者、visit 访视、origin 预测起点、horizon 预测跨度、target 单项目标、OOF 折外预测。所有实现与文字有 Codex 协助，先阅读 AI_USAGE.md。')
    code('''from pathlib import Path
import sys, json
import pandas as pd
ROOT = Path.cwd()
if not (ROOT / 'common').exists():
    ROOT = ROOT.parent
assert (ROOT / 'common').exists(), '请在仓库根目录或 member_a 中运行'
sys.path.insert(0, str(ROOT))
from common.metric import evaluate_oof, smape_plus_one
shared = ROOT / 'artifacts/shared'
manifest = json.loads((shared / 'manifest.json').read_text())
samples = pd.read_csv(shared / 'samples.csv')
patients = pd.read_csv(shared / 'patient_folds.csv')
print('协议', manifest['protocol'], 'seed', manifest['seed'])
print('患者', len(patients), '标量预测请求', len(samples))''')
    md('## 1 一行样本是什么\n\n当前 samples.csv 一行是一个 prediction_id 对应的单项 UPDRS 评分，不是一行四个目标。不能把旧本地宽表直接拼入。')
    code("print(samples.head(8).to_string(index=False))")
    md('## 2 为什么按患者分折\n\n同一患者所有访视跟随同一个 fold。不能随机拆分请求行。')
    code('''for fold in range(5):
    train = set(samples.loc[samples.fold != fold, 'patient_id'])
    valid = set(samples.loc[samples.fold == fold, 'patient_id'])
    assert train.isdisjoint(valid)
    print(f'fold {fold}: 训练 {len(train)} 位患者，验证 {len(valid)} 位患者')''')
    md('## 3 手算 SMAPE+1\n\n不是最后分数加 1，而是给真实值和预测值先各加 1。')
    code("print('真实 [0,2]、预测 [0,4] 的分数 =', smape_plus_one([0,2], [0,4]))")
    md('## 4 重算所有 OOF\n\nmean_score 是五折等权均分；oof_score 是所有请求合并评分，不能混淆。')
    code('''for name in ['A0','A1','A2','A3']:
    predictions = pd.read_csv(ROOT / 'member_a/results' / name / 'oof.csv')
    result = evaluate_oof(samples, predictions)
    print(name, 'mean_score =', round(result['mean_score'],6),
          'oof_score =', round(result['oof_score'],6), '请求数 =', result['n_predictions'])''')
    md('![目标分布](eda/figures/02_targets.png)\n\n![特征实验](eda/figures/04_ablation.png)\n\n所有扩展在当前模型上没有改善五折均分。查看 features.py 理解时间限制，查看 run.py 理解公共接口的调用。完整重训命令在 README.md。')
    namespace={};previous=Path.cwd();counter=0
    try:
        os.chdir(root)
        for cell in cells:
            if cell['cell_type']!='code':continue
            counter+=1;stream=io.StringIO()
            with contextlib.redirect_stdout(stream):exec(compile(''.join(cell['source']),f'cell_{counter}','exec'),namespace)
            cell['execution_count']=counter;cell['outputs']=[{'name':'stdout','output_type':'stream','text':stream.getvalue().splitlines(keepends=True)}]
    finally:os.chdir(previous)
    nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.11.15'}},'nbformat':4,'nbformat_minor':5}
    (base/'walkthrough.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
    print('Generated Member A materials for split', shared['split_id'])


if __name__=='__main__':main()
