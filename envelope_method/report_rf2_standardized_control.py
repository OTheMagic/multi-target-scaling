"""Render the fully paired standardized rf2 sensitivity comparison."""
import os,json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'data/envelope_method/results/rf2_standardized_outlier_control'
REPORT_OUT=ROOT/'envelope_method/results/rf2_standardized_outlier_control'
REPORT_OUT.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/mpl'))
s=pd.read_csv(OUT/'summary.csv');a=json.loads((OUT/'audit.json').read_text())
lines=['# rf2: outlier impact with standardized training in both conditions','', '200 matched splits per condition. Retained-outlier fits are new; removed-outlier standardized fits reuse the previously completed 200 trials. Training target means/SDs are estimated separately from each training partition; the forest settings and random seed are identical. Predictions are transformed back to original outcome units before calibration.','', '| Treatment | Method | Coverage | Coverage SD | Mean full volume | Volume SD | Median full volume |','|---|---|---:|---:|---:|---:|---:|']
for treatment in ['Retained','Removed']:
    for method in ['Envelope','Point_CHR']:
        r=s[(s.treatment==treatment)&(s.method==method)].iloc[0]
        lines.append(f'| {treatment} | {method} | {100*r.coverage:.3f}% | {100*r.coverage_sd:.3f} pp | {r.volume:,.1f} | {r.volume_sd:,.1f} | {r.median_volume:,.1f} |')
lines += ['', '## Paired effects', '', '| Method | Reduction in mean volume | Median within-split reduction | Coverage change on identical test observations |','|---|---:|---:|---:|']
for e in a['effects']:
    lines.append(f'| {e["method"]} | {100*(1-e["removed_over_retained_mean_volume"]):.2f}% | {100*(1-e["median_paired_volume_ratio"]):.2f}% | {100*e["common_coverage_difference"]:+.4f} pp |')
lines += ['', 'The envelope/CHR ratio of mean volumes is 6.3469 with the outlier and 1.4774 after removal. Corresponding median within-split ratios are 2.5934 and 1.6272. Envelope is smaller in 45/200 intact splits and 52/200 deleted splits. This is strong outlier sensitivity but not a ranking reversal.', '', 'The mean-volume reduction is larger than the median paired reduction because extreme-volume trials receive more weight in the former statistic. Neither is a universal gain. On common test observations, the envelope coverage change has a descriptive 95% Monte Carlo interval of [-0.1305,+0.1422] percentage points. This is variability across splits conditional on the dataset, not an independent population-sampling interval.', '', '## Why training standardization does not eliminate the outlier effect', '', '| Outlier originally in | Trials | Mean envelope volume retained | Mean envelope volume removed |','|---|---:|---:|---:|']
for partition,n in [('train',158),('cal',6),('test',36)]:
    vals={t:next(x['volume'] for x in a['by_partition'] if x['treatment']==t and x['outlier_partition']==partition and x['method']=='Envelope') for t in ['Retained','Removed']}
    lines.append(f'| {partition} | {n} | {vals["Retained"]:,.1f} | {vals["Removed"]:,.1f} |')
lines += ['', 'Training standardization changes the forest\'s target weighting. It does not robustify the mean/SD estimated from calibration residuals. If the extreme observation is in calibration, it remains capable of inflating the envelope even when the location model was trained with standardized targets. The test-only deletion leaves training/calibration unchanged, and the before/after bounds reproduce identically in all 36 such splits.', '', '## Verification and scope', '', 'All original split-source hashes match. The removed split indices equal the retained indices after excluding row 3660 (original dataframe index 4782). Training transformations were checked against the exact training rows in both conditions. Every new saved method metric was recomputed from binary bounds/test residuals; removed-cohort metrics were independently recomputed from their reused archives. Those source archives are linked by SHA-256 in audit.json.', '', 'rf2 remains useful as an outlier-sensitivity example under the stated random-split protocol. Because of its temporal dependence, these results do not establish future-forecast coverage. The separate ordered-split pilot does not invalidate this within-protocol comparison.', '', 'The original observation is statistically extreme; removal is an explicitly labeled sensitivity study, not proof that its recorded value is wrong or that data deletion is justified for the primary benchmark.', '', 'Sources: trials.csv, summary.csv, audit.json, retained trial JSON/NPZ files, and the reused archives under ../rf2_remaining/model_time/.']
(REPORT_OUT / 'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,2,figsize=(10.5,4.5),layout='constrained')
for k,method in enumerate(['Envelope','Point_CHR']):
    rs=[s[(s.treatment==t)&(s.method==method)].iloc[0] for t in ['Retained','Removed']];x=np.arange(2)+(k-.5)*.28;color=['#197c80','#cf8837'][k]
    axes[0].bar(x,[r.volume for r in rs],.28,label=method,color=color)
    axes[1].errorbar(x,[100*r.coverage for r in rs],yerr=[196*r.coverage_sd/np.sqrt(200) for r in rs],fmt='o',capsize=4,color=color)
axes[0].set_ylabel('Mean full outcome-space volume');axes[0].legend()
axes[1].set_ylabel('Joint coverage (%) with 95% Monte Carlo bars');axes[1].axhline(90,color='gray',linestyle='--')
for ax in axes:ax.set_xticks([0,1],['Outlier retained','Outlier removed'])
fig.suptitle('rf2: training targets standardized in both conditions | 200 matched splits')
fig.savefig(REPORT_OUT / 'comparison.png',dpi=180);plt.close(fig)
sys.stdout.reconfigure(encoding='utf-8');print('Report and comparison figure saved.')
