"""Match archived method/scenario/trial requirements to fresh paired reruns."""
import json
import ast
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'data/envelope_method/results'
REVIEW = ROOT/'quarantine/obsolete_synthetic_2026-09-08'
ALIASES = {'tscp_lwc': 'Old_LWC_union', 'tscp_r': 'TSCP_R', 'tscp_s': 'TSCP_S',
           'tscp_gwc': 'TSCP_GWC', 'point_chr': 'Point_CHR', 'unscaled': 'Unscaled',
           'empirical_copula': 'Empirical_copula', 'bonferroni': 'Bonferroni',
           'naive': 'Naive', 'population_oracle': 'Population_oracle'}


def portable(path):
    return str(path).replace('\\', '/')


def without_protocol(config):
    return {k: v for k, v in config.items() if k != 'redraw'}


def main():
    inventory = []
    for filename in ['settings.json', 'notebook_settings.json', 'repair_settings.json']:
        path = ROOT/'envelope_method'/filename
        if path.exists():
            inventory += json.loads(path.read_text())
    by_source = {}
    for item in inventory:
        assert item['config']['redraw'] is True
        for source in item['sources']:
            by_source.setdefault(portable(source), []).append(item)
    fresh = pd.read_csv(OUT/'simulation_trials.csv', low_memory=False)
    for family in ['cqr_baselines']:
        for path in (OUT/family).glob('*/trials.csv'):
            fresh = pd.concat([fresh, pd.read_csv(path)], ignore_index=True)
    for item in inventory:
        if item['id'] not in set(fresh.config_id):
            path = OUT/item['config']['kind']/item['id']/'trials.csv'
            if path.exists():
                fresh = pd.concat([fresh, pd.read_csv(path)], ignore_index=True)
    fresh = fresh.drop_duplicates(['config_id', 'method', 'trial'])
    assert fresh.redraw_train_test.eq(True).all()
    lookup = {(cid, method): set(g.trial.astype(int))
              for (cid, method), g in fresh.groupby(['config_id', 'method'])}
    toys = []
    for family in ['toys', 'toy_baselines']:
        for path in (OUT/family).glob('*/trials.csv'):
            if path.parent.name != 'boundary':
                toys.append(pd.read_csv(path).assign(study=path.parent.name))
    toy = pd.concat(toys).drop_duplicates(['study', 'method', 'trial'])
    assert toy.redraw_train_test.eq(True).all()
    toy_lookup = {(study, method): set(g.trial.astype(int))
                  for (study, method), g in toy.groupby(['study', 'method'])}
    rows, classified = [], []

    def add(source, method, expected, item=None, study=None, detail=''):
        expected = set(int(x) for x in expected)
        cid = item['id'] if item else ''
        actual = (toy_lookup.get((study, method), set()) if study else
                  lookup.get((cid, method), set()))
        missing = sorted(expected-actual)
        status = 'complete' if not missing else 'missing'
        if missing and method in ['Old_LWC_union', 'Exact_ratio_LWC', 'Exact_LWC_bbox', 'Exact_LWC_union']:
            status = 'deferred_full_lwc'
        rows.append(dict(source=source, config_id=cid, study=study or '', method=method,
                         old_trials=len(expected), fresh_trials=len(actual),
                         missing_trials=len(missing), status=status, detail=detail))

    def matching(items, row):
        fields = {'n_dim': 'd', 'n_cals': 'n_cal', 'n_cal': 'n_cal', 'alpha': 'alpha',
                  'base_interval_alpha': 'base_alpha', 'noise_type': 'noise_type'}
        result = []
        for item in items:
            cfg = item['config']
            if any(k in row and pd.notna(row[k]) and row[k] != cfg.get(v)
                   for k, v in fields.items()):
                continue
            if any(k in row and pd.notna(row[k]) and row[k] != v for k, v in cfg['generator_kwargs'].items()):
                continue
            if 'noise_ratio' in row and pd.notna(row['noise_ratio']):
                if not np.isclose(row['noise_ratio'], cfg['noise_levels'][0]/cfg['noise_levels'][-1]):
                    continue
            result.append(item)
        return result

    files = json.loads((REVIEW/'manifest.json').read_text())['files']
    for record in files:
        source = portable(record['source'])
        if not source.endswith('.csv'):
            continue
        path = REVIEW/record['destination']
        category = 'derived_table'
        if source.startswith('syn_exps/'):
            category = 'original_synthetic_requirements'
            stem = path.stem
            prefix = next(k for k in ALIASES if stem.startswith(k+'_'))
            method = ALIASES[prefix]
            canonical = portable(Path(source).with_name('tscp_r'+stem[len(prefix):]+'.csv'))
            candidates = by_source.get(canonical, [])
            for row in pd.read_csv(path).to_dict('records'):
                found = matching(candidates, row)
                if len(found) != 1:
                    add(source, method, range(int(row['n_trials'])), detail=f'configuration matches: {len(found)}; {row}')
                else:
                    add(source, method, range(int(row['n_trials'])), found[0])
        elif source.startswith('reviewer_exps/') and source.endswith('_trial.csv') and '_coordinate_' not in source:
            category = 'reviewer_synthetic_requirements'
            frame = pd.read_csv(path)
            keys = [k for k in ['alpha','n_dim','n_cal','noise_type','base_interval_alpha','noise_ratio',
                                'correlation','correlation_structure','method','score_transform','shift_constant'] if k in frame]
            for values, group in frame.groupby(keys, dropna=False):
                row = dict(zip(keys, values if isinstance(values, tuple) else (values,)))
                method = row['method']
                transform = row.get('score_transform', '')
                if transform in ['capped', 'shifted']:
                    method += f"_{transform}_{row.get('shift_constant', 0):g}"
                found = matching(by_source.get(source, []), row)
                if len(found) != 1:
                    add(source, method, group.trial, detail=f'configuration matches: {len(found)}; {row}')
                else:
                    add(source, method, group.trial, found[0])
        elif source.startswith('envelope_method/results/') and path.name == 'trials.csv':
            parts = source.split('/')
            family = parts[2]
            if family in ['absolute', 'cqr', 'legacy_auxiliary']:
                category = 'superseded_method_run'
                item = json.loads((path.parent/'config.json').read_text())
                found = [i for i in inventory if without_protocol(i['config']) == without_protocol(item['config'])]
                frame = pd.read_csv(path)
                for method, group in frame.groupby('method'):
                    add(source, method, group.trial, found[0] if len(found)==1 else None,
                        detail='' if len(found)==1 else f'configuration matches: {len(found)}')
            elif family == 'toys':
                if path.parent.name == 'boundary':
                    category = 'mathematical_audit_not_fitted_experiment'
                else:
                    category = 'fresh_fitted_toy_replication'
                    frame = pd.read_csv(path)
                    for method, group in frame.groupby('method'):
                        add(source, method, group.trial, study=path.parent.name,
                            detail='new fitted replication of a former score-only experiment')
        classified.append(dict(source=source, category=category))
    # Mixed old aggregates also contain formerly fresh studies whose individual
    # archives were never quarantined. Check these rows rather than assuming
    # that the archived per-configuration directories exhaust the requirements.
    aggregate = pd.read_csv(REVIEW/'envelope_method/results/simulation_trials.csv',
                            usecols=['config_id','trial','method'])
    old_configs = pd.read_csv(REVIEW/'envelope_method/results/configurations.csv').set_index('config_id')
    for (old_id, method), group in aggregate.groupby(['config_id','method']):
        old = old_configs.loc[old_id]
        config = {k: old[k] for k in ['kind','d','n_cal','alpha','noise_type','n_features','n_train','n_test']}
        config.update(noise_levels=ast.literal_eval(old.noise_levels),
                      generator_kwargs=ast.literal_eval(old.generator_kwargs))
        if config['kind'] == 'cqr':
            config['base_alpha'] = old.base_alpha
        found = [i for i in inventory if without_protocol(i['config']) == config]
        add('envelope_method/results/simulation_trials.csv', method, group.trial,
            found[0] if len(found)==1 else None,
            detail=f'old_config_id={old_id}; matches={len(found)}')
    # The older standalone signed toy included a capped-shortcut plot and exact
    # local regions, in addition to the methods in its aggregate trial CSV.
    add('tmp/signed_lwc_toy_summary.json', 'TSCP_R_capped_0', [0], study='conservative_signed_2d',
        detail='fresh fitted replication, not the original unfitted seed-6 illustration')
    add('tmp/signed_lwc_toy_summary.json', 'Exact_LWC_union', [0], study='conservative_signed_2d',
        detail='signed full-cell enumeration left to manual work as requested')
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/'comparator_rerun_audit.csv', index=False)
    pd.DataFrame(classified).to_csv(OUT/'obsolete_table_classification.csv', index=False)
    summary = dict(archived_csv_files=len(classified), checked_method_scenario_entries=len(frame),
                   statuses=frame.status.value_counts().to_dict(),
                   methods=frame.groupby('method').status.value_counts().unstack(fill_value=0).to_dict('index'),
                   note='Checks method/scenario/trial availability; derived summaries are not independent experiments. Fresh draw and fit integrity is separately verified in final_audit.json and comparator sidecars.')
    (OUT/'comparator_rerun_audit.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print(frame.loc[frame.status != 'complete', ['source','method','old_trials','fresh_trials','status','detail']].to_string(index=False))


if __name__ == '__main__':
    main()
