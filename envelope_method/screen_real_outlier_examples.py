"""Screen the current real cohorts without choosing deletions by method wins."""
import json,pickle,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/envelope_method/results/real_outlier_screen';OUT.mkdir(exist_ok=True)
all_results=pd.read_csv(ROOT/'data/envelope_method/results/real_comparison_audited.csv')
results=[]
for name in ['stock','rf2','scm1d','scm20d','energy','student','air','crime']:
    if name in ['air','crime']:
        path=ROOT/f'data/envelope_method/results/extra_real/{name}/dataset.pkl'
    else:path=ROOT/f'data/reviewer_update/real_diagnostics/cache/{name}.pkl'
    with path.open('rb') as f:loaded=pickle.load(f)
    X,y=loaded[:2]
    extremes=[]
    for target in y:
        a=y[target].to_numpy(dtype=float);ids=np.argsort(a);med=np.median(a);mad=np.median(abs(a-med));q1,q3=np.quantile(a,[.25,.75])
        unique=np.unique(a);second=unique[-2] if len(unique)>1 else np.nan
        extremes.append(dict(target=target,largest=float(unique[-1]),next_distinct_largest=float(second),largest_over_next=float(unique[-1]/second) if second else None,max_count=int((a==unique[-1]).sum()),max_position=int(ids[-1]),original_index=str(y.index[ids[-1]]),median=float(med),mad=float(mad),max_modified_z=float(.67448975*(unique[-1]-med)/mad) if mad else None,iqr=float(q3-q1)))
    a=all_results[(all_results.dataset==name)&(all_results.alpha==.1)].set_index('method')
    result=dict(dataset=name,rows=len(y),features=X.shape[1],outputs=y.shape[1],envelope_volume=float(a.loc['Envelope','volume']),chr_volume=float(a.loc['Point_CHR','volume']),ratio=float(a.loc['Envelope','volume']/a.loc['Point_CHR','volume']),envelope_coverage=float(a.loc['Envelope','coverage']),chr_coverage=float(a.loc['Point_CHR','coverage']),target_extremes=extremes)
    if name=='crime':
        pop=X['pop'].to_numpy();top=np.argsort(pop)[-10:][::-1]
        result['largest_populations']=[dict(position=int(i),original_index=str(y.index[i]),population=float(pop[i]),targets=y.iloc[i].to_dict()) for i in top]
        log=np.log1p(y.to_numpy());med=np.median(log,axis=0);mad=np.median(abs(log-med),axis=0)
        result['log_target_outliers']=[dict(position=int(i),max_modified_z=float(np.max(.67448975*abs(log[i]-med)/np.maximum(mad,1e-12)))) for i in np.argsort(np.max(.67448975*abs(log-med)/np.maximum(mad,1e-12),axis=1))[-5:][::-1]]
    results.append(result)
(OUT/'screen.json').write_text(json.dumps(results,indent=2))
sys.stdout.reconfigure(encoding='utf-8')
print(pd.DataFrame([{k:r[k] for k in ['dataset','rows','outputs','envelope_coverage','chr_coverage','ratio']} for r in results]).to_string(index=False))
crime=next(r for r in results if r['dataset']=='crime')
print(json.dumps({k:crime[k] for k in ['target_extremes','largest_populations','log_target_outliers']},indent=2))
