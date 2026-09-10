"""Summarize completed one-row removal and remaining tail sensitivity."""
import os,json,pickle,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/envelope_method/results/rf2_remove_one'
REPORT_OUT = ROOT / 'envelope_method/results/rf2_remove_one'
REPORT_OUT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/mpl'))
sys.stdout.reconfigure(encoding='utf-8')
s=json.loads((OUT/'summary.json').read_text());meta=json.loads((OUT/'metadata.json').read_text())
assert s['status']=='complete'
with (ROOT/'data/reviewer_update/real_diagnostics/cache/rf2.pkl').open('rb') as f:
    X,y,_,_=pickle.load(f)
coords=[]
for p in sorted(OUT.glob('trial_*.npz')):
    with np.load(p) as z:
        a=z['scores_cal'];m=a.mean(0)
        share=np.sort((a-m)**2,axis=0)[-4:].sum(0)/((a-m)**2).sum(0)
        for j,t in enumerate(y.columns):
            coords.append(dict(trial=int(p.stem.split('_')[1]),target=t,env_length=2*z['Envelope'][j],chr_length=2*z['Point_CHR'][j],log_ratio=np.log(z['Envelope'][j]/z['Point_CHR'][j]),cal_std=a[:,j].std(),cal_q90=np.quantile(a[:,j],.9),cal_max=a[:,j].max(),top4_variance_share=share[j]))
c=pd.DataFrame(coords).groupby('target').mean(numeric_only=True).drop(columns='trial')
c['geometric_width_ratio']=np.exp(c.log_ratio)
c.to_csv(OUT/'remaining_coordinate_diagnostics.csv')
before=pd.read_csv(ROOT/'data/envelope_method/results/rf2_diagnosis/paired_trials.csv')
after=pd.read_csv(OUT/'trials.csv')
lines=['# rf2: one-row deletion with 200 refitted comparisons','', 'The specified observation was removed from every split, wherever it occurred; all 200 random forests were refitted. Original data/results are unchanged. The remaining split assignments, model configuration, and model seeds are preserved. Each method shares the reduced fit/calibration/test data. Target joint coverage is 90%.','', '| Dataset | Method | Mean coverage | Coverage SD | Mean full volume | Median full volume |','|---|---|---:|---:|---:|---:|']
for label,frame in [('Original',before),('One row removed',after)]:
    for method in ['Envelope','Point_CHR']:
        g=frame[frame.method==method]
        lines.append(f'| {label} | {method} | {g.coverage.mean():.6f} | {g.coverage.std():.6f} | {g.volume.mean():,.2f} | {g.volume.median():,.2f} |')
lines += ['',f"After deletion: ratio of mean volumes = {s['ratio_of_mean_volumes']:.4f}; median paired volume ratio = {s['median_paired_volume_ratio']:.4f}; envelope smaller in {s['envelope_smaller_fraction']*200:.0f}/200 splits.",'', '## Is it an outlier?', '', f"NASI2 target=78.9 at retained position 3660, original dataframe index 4782, original ARFF line 5371. The next largest target is 5.62, median 3.3, and MAD 0.07. Its modified z-score, 0.67449*(value-median)/MAD, is {meta['modified_z']:.2f}. The upper outer IQR fence is 4.53; 314 observations exceed that fence, so a boxplot flag alone is not unique evidence. The exceptional separation from the next largest observation provides additional evidence. Nearby original rows have NASI2 targets 3.56, 3.53, 3.53, 78.9, 3.54, 3.51, 3.51. Source verification confirms the value is present in the original ARFF, not introduced by the cache. It is a clear statistical outlier, but a recording error cannot be established from these data alone.",'', 'No exact 78.9 values occur in NASI2-prefixed input features. Equal numbers in other target-scale feature families are not evidence of copies of this observation.', '', '## Remaining coordinate shape differences', '', '| Target | Mean envelope length | Mean CHR length | Geometric paired width ratio | Mean largest-four variance share |','|---|---:|---:|---:|---:|']
for t,r in c.iterrows():
    lines.append(f'| {t} | {r.env_length:.4f} | {r.chr_length:.4f} | {r.geometric_width_ratio:.4f} | {r.top4_variance_share:.1%} |')
lines += ['', 'These remaining differences measure residual shape after removing the one specified row. They do not prove that remaining extreme residuals are errors or justify further deletions.', '', '## Verification and interpretation', '', s['verification'], '', 'Calibration has 383 observations in the six trials where the removed row was in calibration, otherwise 384. Point CHR uses the correct conformal order statistic for each half separately (191/192 for those six cases). This avoids the existing helper\'s assumption that the halves have equal sizes. Training size is reduced by one in 158 trials and test size by one in 36 trials. These are matched deletion comparisons rather than a new random repartition of all remaining rows.', '', 'Coverage Monte Carlo intervals and paired before/after differences are saved in summary.json. They describe variability over random splits conditional on this modified dataset. Removing a value identified after inspecting outcomes makes this a sensitivity analysis; it does not establish coverage for the original population or verify that the observation should be removed in the primary benchmark.', '', 'Sources: metadata.json, source_verification.json, summary.json, trials.csv, and 200 trial JSON/NPZ pairs in this directory.']
(REPORT_OUT / 'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
colors=['#197c80','#d38b38'];x=np.arange(2)
for k,method in enumerate(['Envelope','Point_CHR']):
    groups=[a[a.method==method] for a in [before,after]]
    axes[0].bar(x+(k-.5)*.32,[g.volume.mean() for g in groups],.32,label=method,color=colors[k])
    axes[1].errorbar(x+(k-.5)*.15,[100*g.coverage.mean() for g in groups],yerr=[196*g.coverage.std()/np.sqrt(200) for g in groups],fmt='o',capsize=4,label=method,color=colors[k])
axes[0].set_ylabel('Mean full outcome-space volume');axes[0].legend()
axes[1].axhline(90,color='gray',linestyle='--');axes[1].set_ylabel('Joint coverage (%) with 95% Monte Carlo bars')
for a in axes:a.set_xticks(x,['Original','One row removed'])
fig.suptitle('rf2: same 200 split assignments, refitted after deletion')
fig.savefig(REPORT_OUT / 'comparison.png',dpi=180);plt.close(fig)
print('\n'.join(lines))
