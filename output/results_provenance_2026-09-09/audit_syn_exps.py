"""Read-only reconciliation of current synthetic exports to trial CSV sources."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'envelope_method/results'
inventory=json.loads((ROOT/'envelope_method/settings.json').read_text())+json.loads((ROOT/'envelope_method/repair_settings.json').read_text())
views=defaultdict(list)
items={}
for item in inventory:
    for source in item['sources']:
        if source.startswith('syn_exps'):
            views[source.replace('\\','/')].append(item['id'])
            items[item['id']]=item
cache={}
trial_counts=Counter()
family_methods=defaultdict(set)
for identifier,item in items.items():
    parts=[]
    for family in ['absolute','auxiliary','full_lwc_scaling']:
        path=OUT/family/identifier/'trials.csv'
        if path.exists():
            data=pd.read_csv(path)
            data['_family']=family
            parts.append(data)
    combined=pd.concat(parts,ignore_index=True).drop_duplicates(['method','trial'],keep='first')
    for method,group in combined.groupby('method'):
        cache[identifier,method]=group
        trial_counts[method]+=len(group)
        for family in group['_family'].unique():family_methods[family].add(method)

def numeric_list(text):
    if isinstance(text,str):return [float(v.strip()) for v in text.strip('[]').split(',')]
    return text

metrics=['test_coverage_avg','test_coverage_1std','coverage_vol_avg','coverage_vol_1std','coverage_max_length_median','runtime_avg','n_trials']
files=[]
issues=[]
unique_rows={}
cell_checks=0
hashes=defaultdict(list)
for path in sorted((ROOT/'syn_exps').rglob('*')):
    if not path.is_file():continue
    relative=path.relative_to(ROOT).as_posix()
    hashes[hashlib.sha256(path.read_bytes()).hexdigest()].append(relative)
    info=dict(path=relative,bytes=path.stat().st_size)
    if path.suffix!='.csv':
        info['kind']='non_csv'
        files.append(info)
        continue
    frame=pd.read_csv(path)
    info.update(rows=len(frame),columns=list(frame.columns))
    matching=[]
    for source,identifiers in views.items():
        target=Path(source)
        suffix=target.name.removeprefix('tscp_r_')
        if path.parent==ROOT/target.parent and path.name.endswith(suffix):
            matching.append((source,identifiers,path.name[:-len(suffix)].removesuffix('_')))
    # The longest suffix identifies the unit Gaussian or Laplace sample view.
    matching.sort(key=lambda x:len(Path(x[0]).name),reverse=True)
    if not matching:
        issues.append(dict(path=relative,problem='no export view mapping'))
        files.append(info)
        continue
    source,identifiers,prefix=matching[0]
    method='Old_LWC_union' if prefix=='tscp_lwc' else next((m for _,m in cache if m.lower()==prefix),None)
    info.update(view=source,method=method,config_ids=frame.config_id.tolist(),source_families=[],checks=0,mismatches=[])
    expected_ids=[i for i in identifiers if (i,method) in cache]
    if sorted(frame.config_id)!=sorted(expected_ids):
        info['mismatches'].append(dict(field='config_ids',actual=frame.config_id.tolist(),expected=expected_ids))
    for _,row in frame.iterrows():
        key=(row.config_id,method)
        if key not in cache:
            info['mismatches'].append(dict(config_id=row.config_id,field='missing source method'))
            continue
        group=cache[key]
        cfg=items[row.config_id]['config']
        info['source_families']+=list(group['_family'].unique())
        volume=group.outcome_volume.to_numpy()/2**cfg['d']
        lengths=np.array([max(numeric_list(value))/2 for value in group.coordinate_lengths])
        with np.errstate(invalid='ignore'):
            expected=dict(test_coverage_avg=np.mean(group.test_coverage),test_coverage_1std=np.std(group.test_coverage,ddof=1),
                coverage_vol_avg=np.mean(volume),coverage_vol_1std=np.std(volume,ddof=1),
                coverage_max_length_median=np.median(lengths),runtime_avg=np.mean(group.runtime),n_trials=len(group))
        expected.update(alpha=cfg['alpha'],n_dim=cfg['d'],n_cals=cfg['n_cal'],noise_type=cfg['noise_type'],
            redraw_train_test=True,n_train=cfg['n_train'],n_test=cfg['n_test'],config_id=row.config_id,**cfg['generator_kwargs'])
        for field,value in expected.items():
            info['checks']+=1
            cell_checks+=1
            actual=row[field] if field in row.index else None
            ok=(actual==value) if isinstance(value,(str,bool)) else np.isclose(actual,value,rtol=1e-11,atol=1e-12,equal_nan=True)
            if not ok:info['mismatches'].append(dict(config_id=row.config_id,field=field,actual=actual,expected=value))
        unique_rows.setdefault(key,[]).append(relative)
    info['source_families']=sorted(set(info['source_families']))
    files.append(info)
    issues.extend(dict(path=relative,**issue) for issue in info['mismatches'])

# Independently reconcile owning per-config CSV rows to the consolidated trial table.
columns=['config_id','method','trial','test_coverage','outcome_volume','runtime','coordinate_lengths']
simulation=pd.read_csv(OUT/'simulation_trials.csv',usecols=columns)
simulation=simulation[simulation.config_id.isin(items)]
simulation_checks=Counter()
simulation_issues=[]
for (identifier,method),source in cache.items():
    if (identifier,method) not in unique_rows:continue
    target=simulation[(simulation.config_id==identifier)&(simulation.method==method)]
    if len(target)!=len(source):
        simulation_issues.append(dict(config_id=identifier,method=method,source_rows=len(source),simulation_rows=len(target)))
        continue
    source=source.sort_values('trial').reset_index(drop=True)
    target=target.sort_values('trial').reset_index(drop=True)
    for field in ['trial','test_coverage','outcome_volume','runtime']:
        a=source[field].to_numpy();b=target[field].to_numpy()
        good=np.isclose(a,b,rtol=1e-11,atol=1e-12,equal_nan=True)
        simulation_checks[field]+=len(good)
        if not good.all():simulation_issues.append(dict(config_id=identifier,method=method,field=field,mismatches=int((~good).sum())))
    for a,b in zip(source.coordinate_lengths,target.coordinate_lengths):
        aa=numeric_list(a);bb=numeric_list(b)
        simulation_checks['coordinate_values']+=len(aa)
        if len(aa)!=len(bb) or not np.allclose(aa,bb,rtol=1e-11,atol=1e-12,equal_nan=True):
            simulation_issues.append(dict(config_id=identifier,method=method,field='coordinate_lengths'))

view_details={}
for view,identifiers in views.items():
    selected=[f for f in files if f.get('view')==view]
    view_details[view]=dict(configs=len(set(identifiers)),files=len(selected),rows=sum(f['rows'] for f in selected),
        dimensions=sorted({items[i]['config']['d'] for i in identifiers}),calibration_sizes=sorted({items[i]['config']['n_cal'] for i in identifiers}),
        methods={f['method']:dict(rows=f['rows'],families=f['source_families'],n_trials=sorted(pd.read_csv(ROOT/f['path']).n_trials.unique().tolist())) for f in selected})
report=dict(scope='All current syn_exps files; no exporters or experiments run, no research file edits',
    file_count=len(files),bytes=sum(f['bytes'] for f in files),csv_files=sum(f['path'].endswith('.csv') for f in files),
    exported_rows=sum(f.get('rows',0) for f in files),unique_config_method_rows=len(unique_rows),unique_configurations=len(items),
    summary_cell_checks=cell_checks,summary_mismatches=issues,trial_table_cell_checks=dict(simulation_checks),trial_table_mismatches=simulation_issues,
    source_method_trial_counts=dict(trial_counts),source_family_methods={k:sorted(v) for k,v in family_methods.items()},
    exact_duplicate_file_groups=[v for v in hashes.values() if len(v)>1],repeated_config_method_views=[dict(config_id=k[0],method=k[1],files=v) for k,v in unique_rows.items() if len(v)>1],
    views=view_details,files=files)
destination=Path(__file__).with_name('syn_exps_audit.json')
destination.write_text(json.dumps(report,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ['files','repeated_config_method_views']},indent=2))
