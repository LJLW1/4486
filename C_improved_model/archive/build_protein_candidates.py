"""Generate two V12 candidates using an additional protein similarity predictor."""
import ast
import json
from pathlib import Path

directory = Path(__file__).resolve().parent
base = json.loads((directory / 'teammate_aligned_submission.ipynb').read_text())
visit_source = (directory / 'visit_history_model.py').read_text()
tree = ast.parse(visit_source)
names = {'history_codes', 'training_rows'}
helpers = '\n\n'.join(ast.get_source_segment(visit_source, node) for node in tree.body
                      if isinstance(node, ast.FunctionDef) and node.name in names)
protein_code = '''
MILESTONES = (6, 18)
{helpers}

protein_table = pd.read_csv(Path(DATA_DIR) / 'train_proteins.csv', usecols=KEYS + ['UniProt', 'NPX'])
clinical = pd.read_csv(Path(DATA_DIR) / 'train_clinical_data.csv')
peptides = pd.read_csv(Path(DATA_DIR) / 'train_peptides.csv', usecols=KEYS)
omics = pd.concat([protein_table[KEYS], peptides], ignore_index=True).drop_duplicates()
reference = training_rows(clinical, omics, 'union').reset_index(drop=True)
wide = protein_table[protein_table.UniProt.isin(PROTEINS)].pivot(index=KEYS, columns='UniProt', values='NPX').reindex(columns=PROTEINS)
values = np.log1p(wide)
protein_mean = values.mean()
protein_std = values.std(ddof=0).replace(0, 1).fillna(1)
scaled = ((values - protein_mean) / protein_std).clip(-5, 5)
reference_protein = scaled.reindex(pd.MultiIndex.from_frame(reference[KEYS])).to_numpy()

def protein_prediction(query, protein_values):
    output = np.empty(len(query))
    rating = reference.rating.to_numpy()
    target = reference.target.to_numpy()
    history = reference.history.to_numpy()
    month = reference.target_month.to_numpy()
    horizon = reference.horizon.to_numpy()
    groups = query.reset_index(drop=True).groupby(['target', 'target_month', 'history', 'horizon'], sort=False).indices
    for (tgt, target_month, hist, h), positions in groups.items():
        all_target = target == tgt
        selected = all_target & (history == hist)
        if not selected.any():
            selected = all_target
        truth = rating[selected]
        candidates = np.arange(int(rating[all_target].max()) + 1)
        losses = 200 * abs(truth[:, None] - candidates) / (truth[:, None] + candidates + 2)
        weights = np.exp(-0.5 * ((month[selected] - target_month) / 12) ** 2)
        weights *= np.where((horizon[selected] == 6) == (h == 6), 1, 0.25)
        for start in range(0, len(positions), 64):
            idx = positions[start:start + 64]
            diff = protein_values[idx, None, :] - reference_protein[selected][None, :, :]
            similarity = np.where(np.isfinite(diff), np.exp(-0.5 * diff ** 2), 1).mean(axis=2)
            output[idx] = candidates[((0.5 + 0.5 * similarity) * weights @ losses).argmin(axis=1)]
    return output
'''.format(helpers=helpers)
api = '''import amp_pd_peptide
env = amp_pd_peptide.make_env()
for test, peptides, proteins, sample in env.iter_test():
    prediction = model.predict_batch(test, peptides, proteins, sample)
    parsed = sample.prediction_id.str.extract(r'^(?P<patient_id>\\d+)_(?P<visit_month>\\d+)_(?P<target>updrs_[1-4])_plus_(?P<horizon>0|6|12|24)_months$')
    if parsed.isna().any().any():
        raise ValueError('Unexpected prediction identifier')
    for column in ['patient_id', 'visit_month', 'horizon']:
        parsed[column] = parsed[column].astype(int)
    parsed['target_month'] = parsed.visit_month + parsed.horizon
    visits = pd.DataFrame([(p, t) for p, dates in model.history.visits.items() for t in dates], columns=KEYS)
    parsed['history'] = history_codes(parsed, visits)
    current = proteins[proteins.UniProt.isin(PROTEINS)].pivot(index=KEYS, columns='UniProt', values='NPX').reindex(columns=PROTEINS)
    current = ((np.log1p(current) - protein_mean) / protein_std).clip(-5, 5)
    current = current.reindex(pd.MultiIndex.from_frame(parsed[KEYS])).to_numpy()
    protein_preds = protein_prediction(parsed, current)
    available = np.isfinite(current).any(axis=1)
    prediction.loc[available, 'rating'] = 0.75 * prediction.loc[available, 'rating'].to_numpy() + 0.25 * protein_preds[available]
    env.predict(prediction)
'''
options = {
    'pair': ['P05060', 'O15240'],
    'top10': ['Q06481', 'P17174', 'P04180', 'P05060', 'O15240', 'P14618', 'Q9NQ79', 'P13521', 'P02787', 'P02649'],
}
for name, proteins in options.items():
    nb = json.loads(json.dumps(base))
    nb['cells'][0]['source'].append('\nProtein similarity auxiliary model, 25% blend when current proteins are available.\n')
    text = 'PROTEINS = ' + repr(proteins) + '\n' + protein_code
    compile(text, '<protein>', 'exec')
    compile(api, '<api>', 'exec')
    nb['cells'][-1:] = [{'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': text.splitlines(True)},
                        {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': api.splitlines(True)}]
    path = directory / ('teammate_aligned_protein_' + name + '.ipynb')
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n')
    print(path)
