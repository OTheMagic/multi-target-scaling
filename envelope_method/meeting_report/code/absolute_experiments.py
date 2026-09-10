"""Package and replay paired absolute-score trials; no other score type is run.

One-time import: python code/absolute_experiments.py --import-archives ../report_revision
Portable replay: python code/absolute_experiments.py
Fresh-data reconstruction (same seeds): add --regenerate.
"""
from report_paths import artifact, data_path
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1]/'qa/mpl'))
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from utility.envelope import envelope_prediction, score_at
from utility.res_rescaled import standardized_prediction
from utility.conformal_utils import conformal_rank

HERE=Path(__file__).resolve().parents[1]
DATA=data_path('data/absolute')
TEAL='#087F8C'; ORANGE='#D17A31'; PURPLE='#7963AF'; GRAY='#657384'
LABELS={'gaussian':'Gaussian','correlated':'Correlated Gaussian','laplace':'Laplace','skewed':'Skewed lognormal'}

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2),encoding='utf-8')

def old_bound(scores,alpha):
    if conformal_rank(len(scores),alpha)>len(scores) or np.any(scores.std(0)==0):
        return np.full(scores.shape[1],np.inf)
    return standardized_prediction(scores,alpha).upper

def import_archives(source):
    source=artifact(source.resolve())
    DATA.mkdir(parents=True,exist_ok=True)
    design=json.loads((source/'design.json').read_text())
    manifest=[]
    for config,spec in enumerate(design['specs'][:18]):
        collected={k:[] for k in ['cal','test','old','env','seed','model_coef']}
        hashes=[]
        for trial in range(spec['reps']):
            p=source/f'data/config_{config:02d}/trial_{trial:03d}.npz'
            meta=json.loads(p.with_suffix('.json').read_text())
            with np.load(p) as a:
                collected['cal'].append(a['abs_cal'])
                collected['test'].append(a['abs_test'])
                collected['old'].append(a['bound_abs_old'])
                collected['env'].append(a['bound_abs_env'])
                collected['model_coef'].append(a['model_coef'])
            collected['seed'].append(meta['seed'])
            digest=hashlib.sha256(p.read_bytes()).hexdigest()
            expected=meta.get('sha256') or meta.get('archive_sha256')
            if expected: assert digest==expected
            hashes.append(dict(trial=trial,seed=meta['seed'],source_sha256=digest))
        dest=DATA/f'config_{config:02d}.npz'
        np.savez_compressed(dest,**{k:np.asarray(v) for k,v in collected.items()})
        manifest.append(dict(config=config,spec=spec,archive=dest.name,
                             sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),source_trials=hashes))
        print(f'Imported absolute configuration {config}',flush=True)
    # Fixed, predeclared 2D illustration from the prior study.
    p=source/'geometry_laplace.npz'
    with np.load(p) as a:
        keep=['abs_cal','abs_test','bound_abs_old','bound_abs_env','pred_test','Y_test','scales','X_test','model_coef']
        np.savez_compressed(DATA/'geometry.npz',**{k:a[k] for k in keep})
    dump(DATA/'manifest.json',dict(origin='Previously generated independent fitted trials, September 9, 2026',
         n_train=800,n_test=1200,configurations=manifest,
         geometry=dict(seed=2026990918,family='laplace',n=30,d=2,test_index=0,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest())))

def replay(regenerate=False):
    info=json.loads((DATA/'manifest.json').read_text())
    rows=[]; violations=0; reruns=0; max_error=0.
    for item in info['configurations']:
        spec=item['spec']; p=DATA/item['archive']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
        with np.load(p) as a:
            for trial in range(spec['reps']):
                cal,test=a['cal'][trial],a['test'][trial]
                if regenerate:
                    from data_model import generate
                    regenerated=generate(int(a['seed'][trial]),spec['family'],spec['n'],d=spec['d'],base_alpha=spec['base_alpha'])
                    np.testing.assert_allclose(regenerated['abs_cal'],cal,rtol=1e-11,atol=1e-11)
                    np.testing.assert_allclose(regenerated['abs_test'],test,rtol=1e-11,atol=1e-11)
                env=envelope_prediction(cal,spec['alpha']); old=old_bound(cal,spec['alpha'])
                for name,b in [('env',env.upper),('old',old)]:
                    ref=a[name][trial]
                    np.testing.assert_allclose(b,ref,rtol=1e-10,atol=1e-10)
                    max_error=max(max_error,float(np.max(abs(b-ref))))
                    hits=(test<=b).all(1)
                    row=dict(config=item['config'],trial=trial,seed=int(a['seed'][trial]),
                        family=spec['family'],n=spec['n'],d=spec['d'],alpha=spec['alpha'],
                        study=spec['study'],method=name,coverage=float(hits.mean()),volume=float(np.prod(2*b)))
                    for j in range(spec['d']): row[f'length_{j+1}']=float(2*b[j])
                    rows.append(row)
                violations+=int(np.any(env.upper>old+1e-8*(1+abs(old))))
                assert (test<=env.upper).all(1).sum()<=(test<=old).all(1).sum()
                reruns+=1
        print(f'Replayed absolute configuration {item["config"]}',flush=True)
    assert violations==0
    df=pd.DataFrame(rows); df.to_csv(DATA/'trials.csv',index=False)
    summaries=[]
    for config,g in df.groupby('config'):
        old=g[g.method=='old'].sort_values('trial'); env=g[g.method=='env'].sort_values('trial')
        ratios=env.volume.to_numpy()/old.volume.to_numpy()
        row={k:old.iloc[0][k] for k in ['config','family','n','d','alpha','study']}
        row.update(reps=len(old),ratio=ratios.mean(),ratio_ci=ci(ratios)[1],
                   old_coverage=old.coverage.mean(),env_coverage=env.coverage.mean(),
                   old_mcse=old.coverage.std(ddof=1)/np.sqrt(len(old)),env_mcse=env.coverage.std(ddof=1)/np.sqrt(len(env)),
                   max_ratio=ratios.max(),min_ratio=ratios.min())
        summaries.append(row)
    pd.DataFrame(summaries).to_csv(DATA/'summary.csv',index=False)
    dump(data_path('qa/absolute_audit.json'),dict(paired_trials=reruns,coordinate_comparisons=reruns*6,
         containment_violations=violations,max_saved_endpoint_error=max_error,raw_regeneration_checked=regenerate,
         all_finite=bool(np.isfinite(df.volume).all())))
    return df

def ci(v):
    v=np.asarray(v); return float(v.mean()),float(student_t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v)))

def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
        'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
        'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#AAB3BA',
        'grid.color':'#E2E8ED','grid.linewidth':.6,'pdf.fonttype':42,'lines.linewidth':1.6})

def save(fig,name):
    fig.savefig(HERE/f'figures/{name}.pdf',bbox_inches='tight')
    fig.savefig(HERE/f'figures/{name}.png',bbox_inches='tight',dpi=170)
    plt.close(fig)

def figures(df):
    style(); main=df[df.study=='main']
    fig,axs=plt.subplots(2,4,figsize=(9.5,4.45),layout='constrained')
    for col,(family,label) in enumerate(LABELS.items()):
        for idx,n in enumerate([30,80,200]):
            g=main[(main.family==family)&(main.n==n)]
            for method,c,off,title in [('old',ORANGE,-.045,'Old shortcut'),('env',TEAL,.045,'Envelope')]:
                mu,err=ci(g[g.method==method].coverage)
                axs[0,col].errorbar(idx+off,mu,yerr=err,fmt='o',color=c,capsize=2,ms=3.5,label=title if idx==0 else None)
            wide=g.pivot(index='trial',columns='method',values='volume')
            ratio=wide.env/wide.old
            jitter=np.random.default_rng(300+idx).uniform(-.15,.15,len(ratio))
            axs[1,col].scatter(idx+jitter,ratio,s=5,c=TEAL,alpha=.2,rasterized=True)
            mu,err=ci(ratio)
            axs[1,col].errorbar(idx,mu,yerr=err,fmt='D',color=TEAL,capsize=3,ms=4)
        axs[0,col].set(title=label,ylim=(.875,1.005))
        axs[0,col].axhline(.9,c=GRAY,ls='--',lw=.8)
        axs[1,col].axhline(1,c=ORANGE,ls='--',lw=.8)
        axs[1,col].set(ylim=(.40,1.025),xlabel='Calibration size')
        for ax in axs[:,col]:ax.set(xticks=[0,1,2],xticklabels=[30,80,200]);ax.grid(axis='y')
    axs[0,0].set_ylabel('Joint coverage');axs[1,0].set_ylabel('Envelope volume / old volume')
    axs[0,0].legend(frameon=False,loc='upper right')
    save(fig,'absolute_n')
    fig,axs=plt.subplots(2,2,figsize=(7.7,4.5),layout='constrained')
    for col,family in enumerate(['gaussian','laplace']):
        g=df[(df.family==family)&(df.n==80)]
        for method,c,ls,title in [('old',ORANGE,'--','Old shortcut'),('env',TEAL,'-','Envelope')]:
            means=[];errs=[]
            for alpha in [.05,.1,.2,.3]:
                mu,err=ci(g[(g.method==method)&(g.alpha==alpha)].coverage-(1-alpha));means.append(mu);errs.append(err)
            axs[0,col].errorbar([.05,.1,.2,.3],means,yerr=errs,marker='o',ls=ls,color=c,capsize=2,ms=4,label=title)
        means=[];errs=[]
        for alpha in [.05,.1,.2,.3]:
            w=g[g.alpha==alpha].pivot(index=['config','trial'],columns='method',values='volume')
            mu,err=ci(w.env/w.old);means.append(mu);errs.append(err)
        axs[1,col].errorbar([.05,.1,.2,.3],means,yerr=errs,marker='D',color=TEAL,capsize=3,ms=4)
        axs[0,col].set(title=LABELS[family]);axs[0,col].axhline(0,c=GRAY,ls=':',lw=1)
        axs[1,col].set(xlabel=r'Target miscoverage $\alpha$',ylim=(.94,1.008));axs[1,col].axhline(1,c=ORANGE,ls='--',lw=.8)
        for ax in axs[:,col]:ax.set_xticks([.05,.1,.2,.3]);ax.grid(axis='y')
    axs[0,0].set_ylabel('Coverage minus target');axs[1,0].set_ylabel('Envelope volume / old volume')
    axs[0,0].legend(frameon=False)
    save(fig,'absolute_alpha')
    geometry()

def geometry():
    with np.load(DATA/'geometry.npz') as a:
        cal=a['abs_cal']; old=a['bound_abs_old']; env=a['bound_abs_env']; scale=a['scales'];
        oldp=old/scale; envp=env/scale
        axes=[np.linspace(-oldp[j]*1.12,oldp[j]*1.12,301) for j in range(2)]
        xx,yy=np.meshgrid(*axes); candidates=np.c_[xx.ravel(),yy.ravel()]*scale
        r=abs(candidates); m=cal.mean(0);s=cal.std(0);n=len(cal); rank=conformal_rank(n,.1)
        accepted=[]
        for block in np.array_split(r,100):
            f=score_at(cal[None,:,:],m,s,n,block[:,None,:]).max(2)
            q=np.partition(f,rank-1,axis=1)[:,rank-1]
            own=score_at(block,m,s,n,block).max(1)
            accepted.extend(own<=q)
        accepted=np.asarray(accepted)
        assert np.all(r[accepted]<=env+1e-8)
        fig,(ax,inset)=plt.subplots(1,2,figsize=(7.5,3.25),layout='constrained',gridspec_kw={'width_ratios':[1,1]})
        ax.scatter(xx.ravel()[accepted],yy.ravel()[accepted],s=.7,c='#CAD2D9',rasterized=True,label='Full-conformal grid')
        for bound,c,ls,label in [(oldp,ORANGE,'--','Old shortcut'),(envp,TEAL,'-','Envelope')]:
            ax.add_patch(Rectangle(-bound,*(2*bound),fill=False,ec=c,lw=1.8,ls=ls,label=label))
        ax.set(xlabel=r'$(y_1-\widehat f_1(x))/\tau_1$',ylabel=r'$(y_2-\widehat f_2(x))/\tau_2$',aspect='equal')
        ax.set_xlim(axes[0][0],axes[0][-1]);ax.set_ylim(axes[1][0],axes[1][-1])
        ax.legend(loc='lower left',framealpha=.95,fontsize=7)
        for bound,c,ls in [(oldp,ORANGE,'--'),(envp,TEAL,'-')]:
            inset.plot([bound[0],bound[0],0],[0,bound[1],bound[1]],c=c,ls=ls,lw=1.5)
        inset.set_xlim(envp[0]-.16,oldp[0]+.13);inset.set_ylim(envp[1]-.16,oldp[1]+.13)
        ax.set_title('Full region',fontsize=10)
        inset.tick_params(labelsize=8);inset.set_title('Upper-right corner',fontsize=10)
        inset.set_xlabel(r'$(y_1-\widehat f_1(x))/\tau_1$')
        inset.set_ylabel(r'$(y_2-\widehat f_2(x))/\tau_2$')
        inset.set_aspect('equal',adjustable='box')
        save(fig,'absolute_region')
        dump(data_path('qa/absolute_geometry.json'),dict(grid_candidates=len(r),accepted=int(accepted.sum()),
            full_conformal_containment_violations=0,area_ratio=float(np.prod(env/old)),test_index=0,seed=2026990918))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--import-archives',type=Path);parser.add_argument('--regenerate',action='store_true');parser.add_argument('--plots-only',action='store_true')
    args=parser.parse_args()
    if args.import_archives:import_archives(args.import_archives.resolve())
    df=pd.read_csv(DATA/'trials.csv') if args.plots_only else replay(args.regenerate)
    figures(df)
