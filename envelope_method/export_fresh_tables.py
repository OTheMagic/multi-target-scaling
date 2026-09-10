"""Restore notebook-compatible synthetic tables using fresh trial results only."""
import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments import ROOT, OUT
from utility.project_paths import data_path
from utility.exps import _summarize_trial_results, _summarize_coordinate_results
from summarize import read_trial_frame, trial_records


def numeric_list(text):
    if isinstance(text, (list, tuple, np.ndarray)):
        return list(text)
    class Constants(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id not in {'inf','nan'}:
                raise ValueError(f'Unexpected token in saved numeric array: {node.id}')
            return ast.Constant(float(node.id))
    return ast.literal_eval(Constants().visit(ast.parse(text,mode='eval')))


def main():
    inventory=json.loads((ROOT/'envelope_method/settings.json').read_text())
    repair=ROOT/'envelope_method/repair_settings.json'
    if repair.exists():
        inventory+=json.loads(repair.read_text())
    targets={}
    for item in inventory:
        for source in item['sources']:
            if source.startswith(('syn_exps','reviewer_exps')):
                targets.setdefault(source,[]).append(item)
    report=[]
    for source,items in targets.items():
        target=data_path(source)
        target.parent.mkdir(parents=True,exist_ok=True)
        if source.startswith('syn_exps'):
            tables={}
            for item in items:
                cfg=item['config']
                folders=[OUT/'absolute'/item['id'], OUT/'auxiliary'/item['id'],
                         OUT/'full_lwc_scaling'/item['id']]
                data=pd.concat([read_trial_frame(p) for p in folders if p.exists() and
                                ((p/'trials.csv').exists() or next(p.glob('trial_*.json'),None) is not None)],
                               ignore_index=True)
                data=data.drop_duplicates(['method','trial'],keep='first')
                for method,g in data.groupby('method'):
                    # All generated repetitions remain available; one shared
                    # cohort can improve precision in a smaller historical view.
                    volume=g.outcome_volume/2**cfg['d']
                    max_length=g.coordinate_lengths.map(lambda s: max(numeric_list(s))/2)
                    name='TSCP_LWC' if method=='Old_LWC_union' else method
                    row=dict(alpha=cfg['alpha'],n_dim=cfg['d'],n_cals=cfg['n_cal'],n_trials=len(g),
                             noise_type=cfg['noise_type'],test_coverage_avg=g.test_coverage.mean(),
                             test_coverage_1std=g.test_coverage.std(),coverage_vol_avg=volume.mean(),
                             coverage_vol_1std=volume.std(),coverage_max_length_median=max_length.median(),
                             runtime_avg=g.runtime.mean(),redraw_train_test=True,n_train=cfg['n_train'],n_test=cfg['n_test'],
                             config_id=item['id'],**cfg['generator_kwargs'])
                    tables.setdefault(name,[]).append(row)
            for method,rows in tables.items():
                path=target.with_name(target.name.replace('tscp_r_',method.lower()+'_',1))
                pd.DataFrame(rows).to_csv(path,index=False)
                report.append(dict(path=str(path.relative_to(ROOT)),rows=len(rows)))
            continue
        records,coords=[],[]
        for item in items:
            cfg=item['config']
            for path in sorted((OUT/cfg['kind']/item['id']).glob('trial_*.json')):
                saved_records=trial_records(path)
                extra=OUT/'cqr_baselines'/item['id']/path.name
                if cfg['kind']=='cqr' and extra.exists():
                    present={r['method'] for r in saved_records}
                    saved_records += [r for r in trial_records(extra) if r['method'] not in present]
                for row in saved_records:
                    method=row['method']
                    transform='native' if cfg['kind']=='cqr' else None
                    if cfg['kind']=='cqr' and method in ['Unscaled','Empirical_copula']:
                        transform='raw'
                    shift=0.
                    for token in ['_capped_','_shifted_']:
                        if token in method:
                            method,suffix=method.split(token)
                            transform=token.strip('_')
                            shift=float(suffix)
                    if cfg['kind']=='cqr' and method not in ['Envelope_signed','Signed_GWC','CQHR','Base','Unscaled','Empirical_copula']:
                        if ('shift' in target.name) != (transform=='shifted'):
                            continue
                    common=dict(config_id=item['id'],trial=row['trial'],alpha=cfg['alpha'],n_dim=cfg['d'],
                                n_cal=cfg['n_cal'],method=method,noise_type=cfg['noise_type'],
                                score_type='cqr' if cfg['kind']=='cqr' else 'absolute_residual',
                                redraw_train_test=True,n_train=cfg['n_train'],n_test=cfg['n_test'],**cfg['generator_kwargs'])
                    if cfg['kind']=='cqr':
                        common.update(base_interval_alpha=cfg['base_alpha'],score_transform=transform,shift_constant=shift)
                    if 'heterogeneity' in target.name:
                        common['noise_ratio']=cfg['noise_levels'][0]/cfg['noise_levels'][-1]
                    factor=1 if cfg['kind']=='cqr' else 2**cfg['d']
                    records.append(dict(common,test_coverage=row['test_coverage'],coverage_volume=row['outcome_volume']/factor,
                                        coverage_max_length=max(row['coordinate_lengths'])/(1 if cfg['kind']=='cqr' else 2),runtime=row['runtime']))
                    for j,length in enumerate(row['coordinate_lengths'],1):
                        coordinate=dict(common,coordinate=j,coordinate_length=length/(1 if cfg['kind']=='cqr' else 2))
                        if cfg['kind']=='cqr':
                            coordinate.update(coordinate_adjustment=row['mean_adjustments'][j-1],
                                              coordinate_base_length=length-2*row['mean_adjustments'][j-1])
                        coords.append(coordinate)
        data=pd.DataFrame(records)
        coordinates=pd.DataFrame(coords)
        data.to_csv(target,index=False)
        _summarize_trial_results(data).to_csv(target.with_name(target.name.replace('_trial.csv','_summary.csv')),index=False)
        coordinates.to_csv(target.with_name(target.name.replace('_trial.csv','_coordinate_trial.csv')),index=False)
        _summarize_coordinate_results(coordinates).to_csv(target.with_name(target.name.replace('_trial.csv','_coordinate_summary.csv')),index=False)
        report.append(dict(path=str(target.relative_to(ROOT)),rows=len(data)))
    (OUT/'fresh_exports.json').write_text(json.dumps(dict(status='complete',tables=report,
        note='Every row is freshly drawn/refitted. Shared configurations use the full saved repetition count; counts are explicit.'),indent=2),encoding='utf-8')
    print('Fresh notebook-compatible exports:',len(report))


if __name__=='__main__':
    main()
