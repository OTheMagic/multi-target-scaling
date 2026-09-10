"""Combine completed workloads, audit pairing, and produce portable study tables."""
import ast
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
try:
    from .archive_storage import validate_records
except ImportError:
    from archive_storage import validate_records

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/envelope_method/results'


def trial_records(path):
    """Read current or legacy record-only checkpoints without requiring arrays."""
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(payload,dict):
        validate_records(payload)
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        # Real-data split checkpoints predate the storage tiers and keep
        # method measurements under `trials`, alongside coordinate records.
        records = payload.get('records', payload.get('trials'))
    else:
        records = None
    if not isinstance(records, list):
        raise ValueError(f'No trial metric records in {path}')
    return records


def read_trial_frame(folder, *, from_json=False):
    """Use the existing CSV, or rebuild it in memory from retained JSON records."""
    folder = Path(folder)
    csv_path = folder/'trials.csv'
    if csv_path.exists() and not from_json:
        return pd.read_csv(csv_path)
    paths = sorted(folder.glob('trial_*.json')) + sorted(folder.glob('split_*.json'))
    records = [row for path in paths for row in trial_records(path)]
    if not records:
        raise FileNotFoundError(f'No trial CSV or retained metric checkpoints in {folder}')
    return pd.DataFrame(records)


def trial_folders(parent):
    """Find both legacy CSV cohorts and compact cohorts retaining only JSON."""
    if not Path(parent).exists():
        return []
    return [p for p in sorted(Path(parent).iterdir()) if p.is_dir() and
            ((p/'trials.csv').exists() or next(p.glob('trial_*.json'), None) is not None or
             next(p.glob('split_*.json'), None) is not None)]


def summarize_frame(frame, keys, metrics=None):
    rows=[]
    for key,g in frame.groupby(keys,dropna=False):
        key=key if isinstance(key,tuple) else (key,)
        row=dict(zip(keys,key))
        row['n_trials']=len(g)
        for col in metrics or ['test_coverage','outcome_volume','runtime','empty_rate']:
            if col not in g:
                continue
            x=g[col].to_numpy(float)
            row[col+'_mean']=float(np.mean(x))
            row[col+'_se']=float(np.std(x,ddof=1)/np.sqrt(len(x))) if len(x)>1 and np.isfinite(x).all() else np.nan
            # Existing means/SEs retain their exact nonfinite semantics. New
            # finite-only summaries are named explicitly, never substituted.
            finite=x[np.isfinite(x)]
            row[col+'_sd']=float(np.std(x,ddof=1)) if len(x)>1 and np.isfinite(x).all() else np.nan
            row[col+'_median']=float(np.median(x))
            row[col+'_finite_count']=len(finite)
            row[col+'_nan_count']=int(np.isnan(x).sum())
            row[col+'_posinf_count']=int(np.isposinf(x).sum())
            row[col+'_neginf_count']=int(np.isneginf(x).sum())
            row[col+'_zero_count']=int((x==0).sum())
            for label,q in [('q05',.05),('q25',.25),('median',.5),('q75',.75),('q95',.95)]:
                row[col+'_finite_'+label]=float(np.quantile(finite,q)) if len(finite) else np.nan
        if 'outcome_volume' in g:
            row['infinite_volume_fraction']=float(np.isposinf(g.outcome_volume).mean())
            row['invalid_volume_fraction']=float(np.isnan(g.outcome_volume).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def pairs(frame, new, old, keys):
    left=frame.loc[frame.method==new]
    right=frame.loc[frame.method==old]
    merged=left.merge(right,on=keys,suffixes=('_new','_old'),validate='one_to_one')
    if merged.empty:
        return merged
    finite=(np.isfinite(merged.outcome_volume_new)&np.isfinite(merged.outcome_volume_old)&(merged.outcome_volume_old>0))
    merged['volume_ratio']=np.where(finite,merged.outcome_volume_new/merged.outcome_volume_old,np.nan)
    merged['coverage_difference']=merged.test_coverage_new-merged.test_coverage_old
    merged['finite_positive_reference']=finite
    merged['smaller_volume']=merged.outcome_volume_new<merged.outcome_volume_old
    return merged[keys+['volume_ratio','coverage_difference','finite_positive_reference','smaller_volume']]


def refresh_summaries():
    """Rebuild small reports from consolidated tables, without reading raw archives."""
    combined = pd.read_csv(OUT/'simulation_trials.csv', low_memory=False)
    configs = pd.read_csv(OUT/'configurations.csv')
    summarize_frame(combined, ['config_id', 'method']).merge(
        configs, on='config_id').to_csv(OUT/'simulation_summary.csv', index=False)
    for family, keys in [('toy', ['study', 'method']),
                         ('real', ['dataset', 'alpha', 'method'])]:
        frame = pd.read_csv(OUT/(family+'_trials.csv'))
        summarize_frame(frame, keys).to_csv(OUT/(family+'_summary.csv'), index=False)
    paired = pairs(combined, 'Envelope_signed', 'Envelope_capped_0', ['config_id', 'trial'])
    paired.to_csv(OUT/'signed_vs_capped_trials.csv', index=False)
    rows = []
    for config_id, group in paired.groupby('config_id'):
        finite = group.loc[group.finite_positive_reference, 'volume_ratio']
        rows.append(dict(config_id=config_id, trials=len(group), finite_pairs=len(finite),
                         mean_finite_volume_ratio=finite.mean(),
                         finite_ratio_se=finite.std(ddof=1)/np.sqrt(len(finite)) if len(finite)>1 else np.nan,
                         mean_coverage_difference=group.coverage_difference.mean(),
                         smaller_volume_fraction=group.smaller_volume.mean()))
    pd.DataFrame(rows).merge(configs, on='config_id').to_csv(
        OUT/'signed_vs_capped_summary.csv', index=False)
    print('Refreshed summaries and signed-versus-capped pairs; no experiments run.')


def main(*, from_json=False):
    inventory=json.loads((ROOT/'envelope_method/settings.json').read_text())
    inventory+=json.loads((ROOT/'envelope_method/notebook_settings.json').read_text())
    repair=ROOT/'envelope_method/repair_settings.json'
    if repair.exists():
        inventory+=json.loads(repair.read_text())
    assert len({i['id'] for i in inventory}) == len(inventory)
    frames=[]
    statuses=[]
    config_rows=[]
    for item in inventory:
        cfg=item['config']
        folder=OUT/cfg['kind']/item['id']
        path=folder/'trials.csv'
        status=json.loads((folder/'status.json').read_text()) if (folder/'status.json').exists() else {'status':'pending','completed':len(list(folder.glob('trial_*.json')))}
        statuses.append({'config_id':item['id'],'expected':item['trials'],**status})
        config_rows.append(dict(config_id=item['id'],sources=' | '.join(item['sources']),expected_trials=item['trials'],**cfg))
        if path.exists() or next(folder.glob('trial_*.json'),None) is not None:
            frame=read_trial_frame(folder,from_json=from_json)
            assert frame.groupby('method').trial.nunique().eq(item['trials']).all()
            assert not frame.duplicated(['trial','method']).any()
            frames.append(frame)
    for family in ['auxiliary','full_lwc_scaling','cqr_baselines']:
        for folder in trial_folders(OUT/family):
            frames.append(read_trial_frame(folder,from_json=from_json))
    # This previously fitted baseline already obeys the fresh-per-trial design.
    # Its deterministic DGP/model seeds match the four paired envelope cohorts.
    shape_path=ROOT/'data/reviewer_update/data/shape_template_baseline_trial.csv'
    shape=pd.read_csv(shape_path)
    assert shape.redraw_train_test.all()
    for item in inventory:
        if not any('shape_template_standard' in s for s in item['sources']):
            continue
        selected=shape[shape.n_cal==item['config']['n_cal']].copy()
        selected['config_id']=item['id']
        selected['outcome_volume']=4*selected.coverage_volume
        selected['source_table']=str(shape_path.relative_to(ROOT))
        frames.append(selected)
    combined=pd.concat(frames,ignore_index=True).drop_duplicates(['config_id','trial','method'],keep='first')
    combined.to_csv(OUT/'simulation_trials.csv',index=False)
    configs=pd.DataFrame(config_rows)
    configs.to_csv(OUT/'configurations.csv',index=False)
    summary=summarize_frame(combined,['config_id','method'])
    summary.merge(configs,on='config_id').to_csv(OUT/'simulation_summary.csv',index=False)
    paired=pairs(combined,'Envelope','TSCP_R',['config_id','trial'])
    paired.to_csv(OUT/'absolute_paired_trials.csv',index=False)
    ratio=paired.groupby('config_id').agg(trials=('trial','size'),mean_volume_ratio=('volume_ratio','mean'),
          ratio_sd=('volume_ratio','std'),finite_pairs=('finite_positive_reference','sum'),
          smaller_fraction=('smaller_volume','mean'),mean_coverage_difference=('coverage_difference','mean')).reset_index()
    ratio.merge(configs,on='config_id').to_csv(OUT/'absolute_paired_summary.csv',index=False)
    # Extract coordinate tables from the structured per-trial records rather than
    # attempting to interpret Python-list strings in the CSV export.
    coordinate_rows=[]
    containment_violations=0
    for item in inventory:
        if item['config']['kind']!='absolute':
            continue
        folder=OUT/'absolute'/item['id']
        for path in sorted(folder.glob('trial_*.json')):
            records=trial_records(path)
            by_method={r['method']:r for r in records}
            en=np.array(by_method['Envelope']['mean_adjustments'])
            old=np.array(by_method['TSCP_R']['mean_adjustments'])
            containment_violations+=int(np.any(en>old+1e-8*(1+abs(old))))
            for row in records:
                for j,(length,coverage,adj) in enumerate(zip(row['coordinate_lengths'],row['coordinate_coverage'],row['mean_adjustments']),1):
                    coordinate_rows.append(dict(config_id=item['id'],trial=row['trial'],method=row['method'],coordinate=j,
                                                outcome_length=length,coordinate_coverage=coverage,adjustment=adj))
    coordinates=pd.DataFrame(coordinate_rows)
    coordinates.to_csv(OUT/'absolute_coordinate_trials.csv',index=False)
    summarize_frame(coordinates,['config_id','method','coordinate'],
                    ['outcome_length','coordinate_coverage','adjustment']).to_csv(
                        OUT/'absolute_coordinate_summary.csv',index=False)
    toy_frames=[]
    for folder in trial_folders(OUT/'toys'):
        if folder.name=='boundary':
            continue
        toy_frames.append(read_trial_frame(folder,from_json=from_json).assign(study=folder.name))
    for folder in trial_folders(OUT/'toy_baselines'):
        toy_frames.append(read_trial_frame(folder,from_json=from_json).assign(study=folder.name))
    toys=pd.concat(toy_frames,ignore_index=True)
    toys=toys.drop_duplicates(['study','trial','method'],keep='first')
    toys.to_csv(OUT/'toy_trials.csv',index=False)
    summarize_frame(toys,['study','method']).to_csv(OUT/'toy_summary.csv',index=False)
    cqr_pairs=pairs(combined,'Envelope_signed','CQHR',['config_id','trial'])
    cqr_pairs.to_csv(OUT/'cqr_paired_trials.csv',index=False)
    if not cqr_pairs.empty:
        cqr_pairs.groupby('config_id').agg(trials=('trial','size'),volume_ratio=('volume_ratio','mean'),
              coverage_difference=('coverage_difference','mean'),smaller_fraction=('smaller_volume','mean')).reset_index().merge(configs,on='config_id').to_csv(OUT/'cqr_paired_summary.csv',index=False)
    real_frames=[]
    for family in ['real','extra_real']:
        for folder in trial_folders(OUT/family):
            data=read_trial_frame(folder,from_json=from_json)
            if 'runtime' not in data:
                data['runtime']=data.runtime_median
            real_frames.append(data)
    real=pd.concat(real_frames,ignore_index=True)
    real.to_csv(OUT/'real_trials.csv',index=False)
    summarize_frame(real,['dataset','alpha','method']).to_csv(OUT/'real_summary.csv',index=False)
    audit=dict(simulation_statuses=statuses,absolute_containment_violations=containment_violations,
               completed_simulation_configurations=sum(s['status']=='complete' for s in statuses),
               expected_simulation_configurations=len(statuses),
               completed_simulation_trials=sum(s.get('completed',0) for s in statuses if s['status']=='complete'),
               caveat='Auxiliary and extra-real workloads have separate completion files; this audit is not a completion claim for the entire request.')
    (OUT/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k!='simulation_statuses'},indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--summaries-only', action='store_true')
    parser.add_argument('--from-json', action='store_true',
                        help='Rebuild consolidated trials from retained JSON, including compact checkpoints.')
    args = parser.parse_args()
    if args.from_json and args.summaries_only:
        parser.error('--from-json rebuilds trial tables and cannot be combined with --summaries-only')
    if not args.summaries_only:
        main(from_json=args.from_json)
    refresh_summaries()
