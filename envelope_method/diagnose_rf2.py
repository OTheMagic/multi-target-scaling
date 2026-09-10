"""Paired rf2 shape diagnostics on all 200 original fitted splits."""
import json, sys, os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.stdout.reconfigure(encoding='utf-8')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/mpl'))
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utility.conformal_utils import conformal_quantile
from utility.data_splitting import data_spliting_CHR_prediction
from utility.envelope import envelope_prediction
OUT=ROOT/'data/envelope_method/results/rf2_diagnosis'
REPORT_OUT = ROOT / 'envelope_method/results/rf2_diagnosis'
REPORT_OUT.mkdir(parents=True, exist_ok=True)
OUT.mkdir(exist_ok=True)
meta=json.loads((ROOT/'data/reviewer_update/real_diagnostics/cache/rf2_metadata.json').read_text())
historical=pd.read_csv(ROOT/'data/reviewer_update/real_diagnostics/data/real_joint_trials.csv')
records=[]; coordinates=[]; split_seed_rows=[]
for trial in range(200):
    with np.load(ROOT/f'data/reviewer_update/real_diagnostics/cache/rf2/split_{trial:03d}.npz') as z:
        cal,test=z['scores_cal'],z['scores_test']
    with np.load(ROOT/f'data/envelope_method/results/real/rf2/split_{trial:03d}.npz') as z:
        bounds={m:z[f'{m}_0.1'] for m in ['Envelope','TSCP_R','Signed_GWC']}
    chr_u=data_spliting_CHR_prediction(cal,.1).upper
    old=historical[(historical.dataset=='rf2')&(historical.trial==trial)&(historical.method=='Point_CHR')].iloc[0]
    np.testing.assert_allclose(np.prod(chr_u),old.residual_volume,rtol=1e-10)
    np.testing.assert_allclose((test<=chr_u).all(axis=1).mean(),old.joint_coverage)
    bounds['Point_CHR']=chr_u
    mu,sd=cal.mean(0),cal.std(0)
    # This pooled fit/calibration diagnostic has no standalone validity claim.
    bounds['Pooled_mean_std_DIAGNOSTIC']=mu+sd*conformal_quantile(((cal-mu)/sd).max(1),.1)
    for seed in [42,7,19,83,151]:
        a,b=train_test_split(cal,test_size=.5,random_state=seed)
        base=conformal_quantile(a,.1,axis=0)
        qu=base*conformal_quantile((b/base).max(1),.1)
        if seed==42:
            np.testing.assert_allclose(qu,chr_u,rtol=1e-12)
            sm,ss=a.mean(0),a.std(0)
            bounds['Split_mean_std']=sm+ss*conformal_quantile(((b-sm)/ss).max(1),.1)
            bounds['Split_std_no_center']=ss*conformal_quantile((b/ss).max(1),.1)
            robust=conformal_quantile(a,.5,axis=0)
            bounds['Split_median_shape']=robust*conformal_quantile((b/robust).max(1),.1)
        split_seed_rows.append(dict(trial=trial,seed=seed,coverage=(test<=qu).all(1).mean(),volume=np.prod(2*qu)))
    for m,u in bounds.items():
        records.append(dict(trial=trial,method=m,coverage=(test<=u).all(1).mean(),volume=np.prod(2*u)))
    q90=np.quantile(cal,.9,axis=0)
    tail_share=np.sort((cal-mu)**2,axis=0)[-max(1,int(np.ceil(len(cal)*.01))):].sum(0)/((cal-mu)**2).sum(0)
    for j,target in enumerate(meta['target_names']):
        coordinates.append(dict(trial=trial,target=target,mean=mu[j],std=sd[j],q90=q90[j],q99=np.quantile(cal[:,j],.99),maximum=cal[:,j].max(),std_over_q90=sd[j]/q90[j],top1pct_variance_share=tail_share[j],env_length=2*bounds['Envelope'][j],chr_length=2*chr_u[j],width_ratio=bounds['Envelope'][j]/chr_u[j],log_width_ratio=np.log(bounds['Envelope'][j]/chr_u[j]),env_coverage=(test[:,j]<=bounds['Envelope'][j]).mean(),chr_coverage=(test[:,j]<=chr_u[j]).mean()))
f=pd.DataFrame(records); c=pd.DataFrame(coordinates); seedf=pd.DataFrame(split_seed_rows)
f.to_csv(OUT/'paired_trials.csv',index=False); c.to_csv(OUT/'coordinate_trials.csv',index=False);seedf.to_csv(OUT/'chr_seed_trials.csv',index=False)
s=f.groupby('method').agg(coverage=('coverage','mean'),coverage_sd=('coverage','std'),volume=('volume','mean'),volume_median=('volume','median'))
v=f.pivot(index='trial',columns='method',values='volume'); cv=f.pivot(index='trial',columns='method',values='coverage')
comparisons={}
for m in v.columns:
    delta=cv[m]-cv.Point_CHR
    comparisons[m]=dict(ratio_of_means=v[m].mean()/v.Point_CHR.mean(),median_paired_ratio=(v[m]/v.Point_CHR).median(),geometric_paired_ratio=np.exp(np.log(v[m]/v.Point_CHR).mean()),fraction_larger=float((v[m]>v.Point_CHR*(1+1e-10)).mean()),coverage_difference=float(delta.mean()),coverage_difference_mc95=[float(delta.mean()-1.96*delta.std()/np.sqrt(200)),float(delta.mean()+1.96*delta.std()/np.sqrt(200))])
cs=c.groupby('target').mean(numeric_only=True).drop(columns='trial')
cs['geometric_width_ratio']=np.exp(cs.log_width_ratio)
s.to_csv(OUT/'method_summary.csv');cs.to_csv(OUT/'coordinate_summary.csv')
result=dict(metadata=meta,method_summary=s.reset_index().to_dict('records'),paired_comparisons=comparisons,coordinates=cs.reset_index().to_dict('records'),seed_sensitivity=seedf.groupby('seed')[['coverage','volume']].mean().reset_index().to_dict('records'),envelope_over_gwc_ratio_of_means=v.Envelope.mean()/v.Signed_GWC.mean(),envelope_over_old_ratio_of_means=v.Envelope.mean()/v.TSCP_R.mean(),checks='Point CHR independently reconstructed and matched archived coverage and volume on all 200 splits; algebraically equivalent quantile-shape implementation matched all 200 splits.',scope='Post-hoc diagnostics on existing fits, not new model fits or confirmatory experiments. Pooled_mean_std_DIAGNOSTIC is not asserted valid. MC intervals describe random splitting conditional on this fixed dataset.')
(OUT/'diagnosis.json').write_text(json.dumps(result,indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
x=np.arange(8); names=[t.split('_48')[0] for t in cs.index]
axes[0].bar(x-.18,cs.env_length,.36,label='Envelope TSCP',color='#197c80')
axes[0].bar(x+.18,cs.chr_length,.36,label='Point CHR',color='#d38b38')
axes[0].set_yscale('log');axes[0].set_xticks(x,names,rotation=35);axes[0].set_ylabel('Mean full interval length (log scale)');axes[0].legend()
axes[1].bar(x,cs.geometric_width_ratio,color='#197c80');axes[1].axhline(1,color='black',linewidth=1)
axes[1].set_xticks(x,names,rotation=35);axes[1].set_ylabel('Geometric mean paired width ratio\nEnvelope / Point CHR')
fig.suptitle('rf2: shape mismatch across eight coordinates | 200 paired splits')
fig.savefig(REPORT_OUT / 'coordinate_comparison.png',dpi=180);fig.savefig(REPORT_OUT / 'coordinate_comparison.pdf');plt.close(fig)
print(json.dumps(result,indent=2))
