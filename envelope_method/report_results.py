"""Big-picture paired plots and a concise interpretation of completed workloads."""
import json
import ast
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'envelope_method'
DATA = ROOT / 'data/envelope_method'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd


def main():
    figs=OUT/'figures'
    figs.mkdir(exist_ok=True)
    (DATA/'figures').mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    cfg=pd.read_csv(DATA / 'results/configurations.csv').set_index('config_id')
    absolute=pd.read_csv(DATA / 'results/absolute_paired_trials.csv')
    cqr=pd.read_csv(DATA / 'results/cqr_paired_trials.csv')
    paired=pd.concat([absolute,cqr],ignore_index=True)
    summary=paired.groupby('config_id').agg(trials=('trial','size'),finite_pairs=('volume_ratio','count'),ratio=('volume_ratio','mean'),
        ratio_sd=('volume_ratio','std'),coverage_difference=('coverage_difference','mean'),
        coverage_sd=('coverage_difference','std')).join(cfg)
    summary['ratio_se']=summary.ratio_sd/np.sqrt(summary.finite_pairs)
    summary['coverage_se']=summary.coverage_sd/np.sqrt(summary.trials)
    summary.to_csv(DATA / 'results/paired_overview.csv')
    sources={}
    inventory=json.loads((OUT/'settings.json').read_text())+json.loads((OUT/'notebook_settings.json').read_text())
    if (OUT/'repair_settings.json').exists():
        inventory+=json.loads((OUT/'repair_settings.json').read_text())
    for item in inventory:
        for source in item['sources']:
            sources.setdefault(source,[]).append(item['id'])
    manifest=[]
    for index,(source,ids) in enumerate(sources.items(),1):
        frame=summary.loc[ids].sort_values(['d','n_cal','alpha','n_features'])
        for part,start in enumerate(range(0,len(frame),12),1):
            group=frame.iloc[start:start+12]
            y=np.arange(len(group))
            labels=[]
            for _,r in group.iterrows():
                extra=ast.literal_eval(r.generator_kwargs) if isinstance(r.generator_kwargs,str) else {}
                detail=', '.join(f'{k}={v}' for k,v in extra.items() if k in ['correlation','df','heteroskedastic_strength','contamination_fraction'])
                label=f"d={r.d}, n={r.n_cal}, alpha={r.alpha:g}, train={r.n_train}"
                if r.kind=='cqr': label+=f", base={r.base_alpha:g}"
                if detail: label+='; '+detail
                labels.append(label)
            fig,axes=plt.subplots(1,2,figsize=(12, max(2.8,.36*len(group)+1.4)),layout='constrained')
            for ax,val,se,ref,xlabel in [(axes[0],'ratio','ratio_se',1.,'Envelope / reference volume (paired mean)'),
                                        (axes[1],'coverage_difference','coverage_se',0.,'Envelope minus reference joint coverage')]:
                ax.errorbar(group[val],y,xerr=1.96*group[se],fmt='o',color='#147d78',capsize=3)
                ax.axvline(ref,color='#777777',linestyle='--',linewidth=1)
                ax.set_yticks(y,labels if ax==axes[0] else [])
                ax.set_xlabel(xlabel); ax.grid(axis='x',alpha=.2)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            fig.suptitle(source.replace('\\',' / ')+'\nReference: old shortcut for absolute scores; CQHR for signed CQR. 95% Monte Carlo bars.',fontsize=10)
            name=f'study_{index:02d}_{part:02d}'
            fig.savefig(figs/(name+'.pdf'),bbox_inches='tight')
            if index<=2:
                fig.savefig(figs/(name+'.png'),dpi=160,bbox_inches='tight')
            plt.close(fig)
            manifest.append(dict(figure=name+'.pdf',source=source,configurations=group.index.tolist()))
    (DATA/'figures/study_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    a=summary[summary.kind=='absolute']
    toys=pd.read_csv(DATA / 'results/fitted_toy_summary.csv')
    miss=toys[toys.study=='misspecified_width'].set_index('method')
    cons=toys[toys.study=='conservative_signed_2d'].set_index('method')
    report=f'''# Signed Envelope: Results and Qualifications

## Completed Fitted Studies

- {len(inventory)} synthetic configurations, {sum(i['trials'] for i in inventory):,} fresh train/calibration/test trials.
- 1,730 fresh fitted toy trials, with 1,000 training observations each.
- Eight real datasets, 200 splits each; six also have five target-alpha levels.
- All primary and pilot synthetic runs refit within each trial. Shared method
  comparisons use identical data and fits within that trial. The existing shape
  baseline uses its already-fresh, matching deterministic cohort.

## Main Findings

Across {len(a)} absolute-score configurations ({int(a.trials.sum()):,} trials), no
coordinate-containment violations were found against the corrected old shortcut.
The median configuration's mean paired volume ratio is {a.ratio.median():.4f};
the smallest is {a.ratio.min():.4f}. Averaging configuration ratios equally gives
{a.ratio.mean():.4f}, which is a descriptive summary, not a universal efficiency gain.

There is no uniform CQHR advantage. In the standard fitted Gaussian toys, signed
envelope volumes are generally similar to, or slightly larger than, CQHR.
In the misspecified-width toy (80 trials, common base alpha 0.1), mean volumes
are {miss.loc['Envelope_signed','volume']:.3f} versus {miss.loc['CQHR','volume']:.3f},
with joint coverages {miss.loc['Envelope_signed','coverage']:.4f} versus
{miss.loc['CQHR','coverage']:.4f}. The ratio of these mean volumes is
{miss.loc['Envelope_signed','volume']/miss.loc['CQHR','volume']:.3f}; this differs
from the mean within-trial ratio reported in the paired table.

Signed shrinkage is visible in the conservative-interval toy (base alpha 0.02,
explicitly separate from the requested 0.1 comparisons). Mean envelope volume
is {cons.loc['Envelope_signed','volume']:.3f} versus base volume
{cons.loc['Base','volume']:.3f}, with coverage {cons.loc['Envelope_signed','coverage']:.4f}.
The plot fixes trial 0 and test covariate 0, rather than selecting a favorable example.

A search of the saved CQR trials found a mixed-sign witness: GWC adjustments
were (5.208, 5.612, 0.622), but envelope adjustments were
(5.208, 5.567, -0.672). Thus expansion of every GWC coordinate does not rule
out shrinkage of one envelope coordinate. This is not evidence that the entire
envelope lies below zero, or that its volume is smaller than the base box.
See `sign_witnesses.json` for
the exact archive and test index. No full-LWC computation was needed.

## Proof Scope and Search

Uniform weak improvement is proved for the same nonnegative score function,
positive calibration scales, and harmonized closed-cell rules. It does not
compare different signed/capped representations, does not dominate CQHR, and
does not say a smaller region has greater coverage. A tied-zero mean-cell bug
in the former shortcut is fixed and documented, including six saved witnesses.

The note proves both a certified binary-search range and a stronger direct
order-statistic localization rule. Optional `search="rank"` matched backward
search in all 9,200 saved formula cases. Backward search remains the default;
extra rank preprocessing need not help when only one surface is inspected.

## Deferred Costly Work

No further full-LWC computation is scheduled. `full_lwc_manual.ipynb` is disabled
by default, runs sequentially when enabled, and checkpoints per trial. Full-LWC
outputs that completed before the pause request are retained, but no claim about
runtime comparability is made: vectorized and historical implementations differ.

## Data and Reading Guide

`signed_envelope.tex` is the full derivation; `signed_envelope.pdf` is the reading
copy. `results/paired_overview.csv` indexes paired comparisons, and
`figures/study_manifest.json` maps study plots to configurations. Trial JSON/NPZ
files contain raw observations, scores, bounds, and seeds. Infinite-volume rates
and invalid numerical outputs must not be mistaken for finite estimates.
`results/signed_vs_capped_summary.csv` separates the score-representation comparison
from the positive-score shortcut comparison and reports the usable finite pairs.
Monte Carlo standard errors are over trials, not individual test points.
Volume-ratio error bars use only finite positive-reference pairs; the exported
summary reports their count. `results/final_audit.json` verifies all {sum(i['trials'] for i in inventory):,}
primary/pilot archive checksums, data shapes, fresh draws, and fit metadata.

Superseded data are in `../quarantine/obsolete_synthetic_2026-09-08/`, with hash
manifests. Fresh tables were restored under the original notebook data paths;
their explicit repetition counts can exceed a smaller historical view because
shared configurations use the full newly generated cohort. Historical manuscript
PDFs/figures are not updated by this standalone-method report.
'''
    (OUT / 'RESULTS.md').write_text(report,encoding='utf-8')
    print('Study plots:',len(manifest),'sources:',len(sources))


if __name__=='__main__':
    main()
