"""Paired Monte Carlo uncertainty and fixed-covariate 2D region illustrations."""
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / 'tmp/mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from archive_storage import load_archive
from summarize import read_trial_frame,trial_folders

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'envelope_method'
DATA = ROOT / 'data/envelope_method'
COLORS = dict(Envelope_signed='#147d78', CQHR='#c94b52', Signed_GWC='#58728b', Base='#666666',
              Envelope='#147d78', TSCP_R='#c94b52')
plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                     'savefig.dpi':180, 'pdf.fonttype':42})


def main():
    dest = OUT / 'figures'
    dest.mkdir(exist_ok=True)
    (DATA/'figures').mkdir(parents=True,exist_ok=True)
    rows, summary = [], []
    for folder in trial_folders(DATA / 'results/toys'):
        if folder.name == 'boundary':
            continue
        df = read_trial_frame(folder)
        for method, group in df.groupby('method'):
            summary.append(dict(study=folder.name, method=method, trials=len(group),
                coverage=group.test_coverage.mean(), coverage_se=group.test_coverage.std()/np.sqrt(len(group)),
                volume=group.outcome_volume.mean(), volume_se=group.outcome_volume.std()/np.sqrt(len(group))))
        if 'CQHR' not in set(df.method):
            continue
        p = df.pivot(index='trial', columns='method', values='outcome_volume')
        ratio = p.Envelope_signed / p.CQHR
        coverage = df.pivot(index='trial', columns='method', values='test_coverage')
        diff = coverage.Envelope_signed - coverage.CQHR
        rows.append(dict(study=folder.name, trials=len(p), volume_ratio=ratio.mean(),
                         ratio_se=ratio.std()/np.sqrt(len(p)), coverage_difference=diff.mean(),
                         coverage_difference_se=diff.std()/np.sqrt(len(p))))
    summary = pd.DataFrame(summary)
    summary.to_csv(DATA / 'results/fitted_toy_summary.csv', index=False)
    paired = pd.DataFrame(rows)
    paired.to_csv(DATA / 'results/toy_paired_summary.csv', index=False)
    main_pairs = paired[paired.study != 'conservative_signed_2d'].reset_index(drop=True)
    labels = main_pairs.study.str.replace('_', ' ')
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.1), layout='constrained')
    y = np.arange(len(main_pairs))
    for ax, value, se, reference, xlabel in [
        (axes[0], 'volume_ratio', 'ratio_se', 1., 'Mean paired volume ratio: envelope / CQHR'),
        (axes[1], 'coverage_difference', 'coverage_difference_se', 0., 'Paired joint-coverage difference')]:
        ax.errorbar(main_pairs[value], y, xerr=1.96*main_pairs[se], fmt='o', color=COLORS['Envelope_signed'], capsize=3)
        ax.axvline(reference, color='#777777', linestyle='--', linewidth=1)
        ax.set_yticks(y, labels if ax == axes[0] else [])
        ax.set_xlabel(xlabel)
        ax.grid(axis='x', alpha=.18)
    fig.suptitle('Fresh fitted toys: common base miscoverage 0.1; 95% Monte Carlo error bars', fontsize=12)
    for ext in ['png','pdf']:
        fig.savefig(dest / ('toy_cqhr_comparison.'+ext), bbox_inches='tight')
    plt.close(fig)
    plot_regions(dest)
    print(summary.to_string(index=False))


def plot_regions(dest):
    studies = ['2d_homoskedastic_independent', 'misspecified_width', 'conservative_signed_2d']
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 4.7), layout='constrained')
    metadata = []
    for ax, study in zip(axes, studies):
        folder = DATA / 'results/toys' / study
        cfg = json.loads((folder / 'config.json').read_text())
        with load_archive(folder / 'trial_0000.npz',required_keys=(
                'base_lengths_test','Signed_GWC','CQHR','Envelope_signed')) as data:
            # Trial and covariate index are fixed in advance, not selected for gain.
            width = data['base_lengths_test'][0]
            methods = ['Base', 'Signed_GWC', 'CQHR', 'Envelope_signed']
            radii = {}
            for method in methods:
                adj = np.zeros(2) if method == 'Base' else data[method]
                if adj.ndim == 2:
                    adj = adj[0]
                radii[method] = width/2 + adj
            finite = [r for r in radii.values() if np.isfinite(r).all() and (r >= 0).all()]
            limit = np.max(finite, axis=0) * 1.16
            for method in methods:
                r = radii[method]
                if not np.isfinite(r).all() or np.any(r < 0):
                    continue
                ax.add_patch(Rectangle(-r, 2*r[0], 2*r[1], fill=False,
                    edgecolor=COLORS[method], linewidth=2 if method == 'Envelope_signed' else 1.5,
                    linestyle='--' if method in ['Base','Signed_GWC'] else '-', label=method.replace('_',' ')))
            ax.set_xlim(-limit[0],limit[0]); ax.set_ylim(-limit[1],limit[1])
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlabel('Outcome 1 minus fitted center')
            ax.set_ylabel('Outcome 2 minus fitted center')
            ax.set_title(study.replace('_',' ')+'\n'+f"base alpha={cfg['base_alpha']:g}", fontsize=10)
            ax.axhline(0,color='#cccccc',linewidth=.6); ax.axvline(0,color='#cccccc',linewidth=.6)
            metadata.append(dict(study=study, trial=0, test_covariate_index=0,
                widths=width.tolist(), radii={k:v.tolist() for k,v in radii.items()}))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=4, frameon=False)
    fig.suptitle('Outcome-space regions at one fixed test covariate (trial 0, test index 0)', fontsize=12)
    for ext in ['png','pdf']:
        fig.savefig(dest / ('signed_regions_2d.'+ext), bbox_inches='tight')
    plt.close(fig)
    (DATA / 'figures/signed_regions_2d.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
