"""All report figures and numeric tables derive from the saved paired trials."""
from pathlib import Path
import json
import sys
import os
import shutil
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=ROOT/'data/envelope_method/report_revision'
sys.path[:0]=[str(ROOT),str(HERE)]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'tmp/mpl'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator,FuncFormatter,NullFormatter
from scipy.stats import t as student_t
from experiment_suite import FAMILIES,LABELS,dump
from utility.envelope import score_at
from utility.conformal_utils import conformal_rank

FIG=ROOT/'envelope_method/figures/report_revision'
FIG.mkdir(parents=True,exist_ok=True)
TEAL='#087F8C'; ORANGE='#D17A31'; PURPLE='#7963AF'; GRAY='#657384'; INK='#213547'
COLORS=dict(old=ORANGE,env=TEAL,rank=PURPLE,signed=GRAY,abs_old=ORANGE,abs_env=TEAL,
            cap_old=ORANGE,cap_env=TEAL,signed_env=PURPLE,signed_gwc=GRAY,base='#B0B8C2')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
 'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
 'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#BAC3CC',
 'axes.labelcolor':INK,'text.color':INK,'xtick.color':INK,'ytick.color':INK,
 'axes.titleweight':'bold','grid.color':'#E2E8ED','grid.linewidth':.6,
 'figure.facecolor':'white','axes.facecolor':'white','savefig.dpi':190,
 'pdf.fonttype':42,'ps.fonttype':42,'lines.linewidth':1.7})

df=pd.read_csv(OUT/'trials.csv')
main=df[df.study=='main']
summaries=[]
for key,g in df.groupby(['config','method'],sort=True):
    row={k:g.iloc[0][k] for k in ['config','method','family','n','d','alpha','base_alpha','study']}
    row['replicates']=len(g)
    for metric in ['coverage','volume','infinite','negative_adjustment']:
        vals=g[metric].to_numpy()
        row[metric+'_mean']=float(vals.mean())
        row[metric+'_mcse']=float(vals.std(ddof=1)/np.sqrt(len(vals))) if np.isfinite(vals).all() else np.nan
    summaries.append(row)
pd.DataFrame(summaries).to_csv(OUT/'summary.csv',index=False)

def stats(values):
    v=np.asarray(values);v=v[np.isfinite(v)]
    return (np.mean(v), student_t.ppf(.975,len(v)-1)*np.std(v,ddof=1)/np.sqrt(len(v))) if len(v)>1 else (np.nan,np.nan)

def paired(frame,numerator,denominator,metric='volume'):
    wide=frame.pivot(index=['config','trial'],columns='method',values=metric)
    a,b=wide[numerator],wide[denominator]
    good=np.isfinite(a)&np.isfinite(b)&(b>0)
    return (a[good]/b[good]).to_numpy(),int((~good).sum())

def finish(fig,name):
    fig.savefig(FIG/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(FIG/(name+'.png'),bbox_inches='tight')
    local_figures=HERE/'figures'
    local_figures.mkdir(parents=True,exist_ok=True)
    shutil.copy2(FIG/(name+'.pdf'),local_figures/(name+'.pdf'))
    plt.close(fig)

def decorate(ax,ygrid=True):
    if ygrid:ax.grid(axis='y',zorder=0)

def main_panels(mode):
    fig,axs=plt.subplots(2,4,figsize=(10,5.05),layout='constrained')
    allrat=[]
    for ci,family in enumerate(FAMILIES):
        sub=main[main.family==family]
        ax=axs[0,ci]
        for method,label,delta in [(mode+'_old','Old shortcut',-.035),(mode+'_env','Envelope',.035)]:
            for pos,n in enumerate([30,80,200]):
                g=sub[(sub.method==method)&(sub.n==n)]
                mu,err=stats(g.coverage)
                ax.errorbar(pos+delta,mu,yerr=err,fmt='o',markersize=4,capsize=2,color=COLORS[method],label=label if pos==0 else None)
        ax.axhline(.9,c=GRAY,ls='--',lw=1)
        ax.set(xticks=[0,1,2],xticklabels=[30,80,200],ylim=(.87,1.005),title=LABELS[family])
        if ci==0:ax.set_ylabel('Joint coverage')
        decorate(ax)
        ax=axs[1,ci]
        for pos,n in enumerate([30,80,200]):
            ratios,excluded=paired(sub[sub.n==n],mode+'_env',mode+'_old')
            allrat.extend(ratios)
            jitter=np.random.default_rng(71+pos).uniform(-.15,.15,len(ratios))
            ax.scatter(pos+jitter,ratios,s=6,color=TEAL,alpha=.18,rasterized=True,zorder=2)
            mu,err=stats(ratios)
            ax.errorbar(pos,mu,yerr=err,c=TEAL,fmt='D',markersize=4,capsize=3,zorder=4)
            if excluded:ax.text(pos,1.006,f'{excluded} inf.',fontsize=6.5,ha='center',color=GRAY)
        ax.axhline(1,c=ORANGE,ls='--',lw=1)
        ax.set(xticks=[0,1,2],xticklabels=[30,80,200],xlabel='Calibration size')
        if ci==0:ax.set_ylabel('Volume / old volume')
        decorate(ax)
    for ax in axs[1]:ax.set_ylim(max(0,min(allrat)-.04),1.04)
    axs[0,0].legend(loc='lower right',frameon=False)
    finish(fig,mode+'_main')

def alpha_plot():
    fig,axs=plt.subplots(2,2,figsize=(8.8,5.3),layout='constrained')
    for col,mode in enumerate(['abs','cap']):
        sub=df[(df.family=='gaussian')&(df.n==80)&(df.d==6)&(df.study.isin(['main','alpha']))]
        ax=axs[0,col]
        for method,label in [(mode+'_old','Old shortcut'),(mode+'_env','Envelope')]:
            points=[]
            for alpha,g in sub[sub.method==method].groupby('alpha'):
                mu,err=stats(g.coverage);points.append((1-alpha,mu,err))
            points=np.array(points);order=np.argsort(points[:,0]);points=points[order]
            ax.errorbar(points[:,0],points[:,1],yerr=points[:,2],fmt='o-',markersize=4,capsize=3,color=COLORS[method],label=label)
        ax.plot([.68,.97],[.68,.97],ls='--',c=GRAY,lw=1)
        ax.set(title='Absolute residuals' if mode=='abs' else 'Capped quantile residuals',
               xlabel='Nominal joint coverage',ylabel='Observed joint coverage',xlim=(.68,.97),ylim=(.68,.99))
        decorate(ax)
        ax=axs[1,col]
        for family,color,marker in [('gaussian',TEAL,'o'),('laplace',PURPLE,'s')]:
            sub=df[(df.family==family)&(df.n==80)&(df.d==6)&(df.study.isin(['main','alpha']))]
            points=[]
            for alpha,g in sub.groupby('alpha'):
                rr,ex=paired(g,mode+'_env',mode+'_old');mu,err=stats(rr);points.append((1-alpha,mu,err))
            points=np.array(points);points=points[np.argsort(points[:,0])]
            ax.errorbar(points[:,0],points[:,1],yerr=points[:,2],fmt=marker+'-',markersize=4,capsize=3,color=color,label=LABELS[family])
        ax.axhline(1,c=ORANGE,lw=1,ls='--');ax.set(xlabel='Nominal joint coverage',ylabel='Mean paired volume ratio',xlim=(.68,.97))
        decorate(ax);ax.legend(frameon=False)
    axs[0,0].legend(frameon=False,loc='upper left')
    finish(fig,'alpha_sweep')

def coordinate_plot():
    fig,axs=plt.subplots(2,2,figsize=(9,5.25),layout='constrained')
    famcolors=[TEAL,PURPLE,ORANGE,GRAY]
    for col,mode in enumerate(['abs','cap']):
        ax=axs[0,col]
        for family,color in zip(FAMILIES,famcolors):
            sub=main[(main.n==80)&(main.family==family)]
            mm=[];ee=[]
            for j in range(1,7):
                ratio,excluded=paired(sub,mode+'_env',mode+'_old','length_'+str(j))
                mu,err=stats(ratio);mm.append(mu);ee.append(err)
            ax.errorbar(np.arange(1,7),mm,yerr=ee,fmt='o-',color=color,markersize=3,capsize=2,label=LABELS[family])
        ax.axhline(1,c=ORANGE,ls='--',lw=1)
        ax.set(xlabel='Outcome coordinate',ylabel='Length / old length',xticks=range(1,7),
               title='Absolute residuals' if mode=='abs' else 'Capped quantile residuals')
        decorate(ax)
        ax=axs[1,col]
        sub=main[(main.n==80)&(main.family=='laplace')]
        for method,label in [(mode+'_old','Old shortcut'),(mode+'_env','Envelope')]:
            g=sub[sub.method==method];points=[stats(g['coverage_'+str(j)]) for j in range(1,7)]
            mu,err=np.array(points).T
            ax.errorbar(range(1,7),mu,yerr=err,fmt='o-',color=COLORS[method],capsize=2,markersize=3,label=label)
        ax.set(xlabel='Outcome coordinate',ylabel='Marginal coverage (Laplace)',xticks=range(1,7))
        decorate(ax);ax.legend(frameon=False)
    axs[0,0].legend(frameon=False,ncol=2,fontsize=6.5,loc='lower left')
    finish(fig,'coordinate_comparison')

def uniform_plot():
    fig,axs=plt.subplots(1,3,figsize=(9.7,3.05),layout='constrained')
    evidence={}
    for mode,color,label in [('abs',TEAL,'Absolute'),('cap',PURPLE,'Capped quantile')]:
        rr,exc=paired(main,mode+'_env',mode+'_old')
        sr=np.sort(rr);axs[0].step(sr,np.arange(1,len(sr)+1)/len(sr),where='post',color=color,label=label)
        wide=[main.pivot(index=['config','trial'],columns='method',values='length_'+str(j)) for j in range(1,7)]
        with np.errstate(all='ignore'):
            ratios=np.column_stack([(w[mode+'_env']/w[mode+'_old']).to_numpy() for w in wide])
        good=np.isfinite(ratios).all(1)
        maxratio=ratios[good].max(1);s=np.sort(maxratio)
        axs[1].step(s,np.arange(1,len(s)+1)/len(s),where='post',color=color,label=label)
        evidence[mode]=dict(finite_volume_pairs=len(rr),excluded=exc,max_volume_ratio=float(max(rr)),
            max_coordinate_ratio=float(max(maxratio)),mean_volume_ratio=float(np.mean(rr)),
            max_joint_coverage_gap=float((main[main.method==mode+'_old'].coverage.to_numpy()-main[main.method==mode+'_env'].coverage.to_numpy()).max()))
    for ax in axs[:2]:ax.axvline(1,c=ORANGE,ls='--',lw=1);decorate(ax)
    axs[0].set(xlabel='Volume / old volume',ylabel='Fraction of fitted trials',title='Paired volume ratios')
    axs[1].set(xlabel='Largest coordinate length ratio',title='Every coordinate at once')
    axs[0].legend(frameon=False,loc='upper left')
    for i,family in enumerate(FAMILIES):
        for off,n in [(-.23,30),(0,80),(.23,200)]:
            sub=main[(main.family==family)&(main.n==n)&(main.method=='cap_old')]
            axs[2].bar(i+off,100*(sub.infinite>0).mean(),width=.22,color={30:ORANGE,80:PURPLE,200:TEAL}[n],label=f'n={n}' if i==0 else None)
    axs[2].set(xticks=range(4),xticklabels=['Gauss.','Corr.','Laplace','Skewed'],ylabel='Old infinite-region trials (%)',title='Small samples can give infinity')
    axs[2].legend(frameon=False,fontsize=7);decorate(axs[2])
    dump(OUT/'evidence.json',evidence)
    finish(fig,'uniformity')

def signed_plot():
    fig,axs=plt.subplots(2,2,figsize=(9,5.5),layout='constrained')
    summary=[]
    for family,color in zip(FAMILIES,[TEAL,PURPLE,ORANGE,GRAY]):
        values=[]
        for n in [30,80,200]:
            g=main[(main.family==family)&(main.n==n)]
            rr,excluded=paired(g,'signed_env','cap_old');mu,err=stats(rr)
            summary.append(dict(family=family,n=n,mean_ratio=mu,ci_halfwidth=err,finite_pairs=len(rr),excluded=excluded))
            values.append((mu,err))
        mu,err=np.array(values).T
        axs[0,0].errorbar(range(3),mu,yerr=err,fmt='o-',c=color,markersize=3,capsize=2,label=LABELS[family])
    axs[0,0].axhline(1,c=ORANGE,ls='--',lw=1)
    axs[0,0].set(xticks=range(3),xticklabels=[30,80,200],xlabel='Calibration size',ylabel='Signed volume / old capped volume',title='Changing scores can help or hurt')
    axs[0,0].legend(frameon=False,fontsize=7)
    for idx,family in enumerate(FAMILIES):
        g=main[(main.family==family)&(main.n==80)]
        for delta,m,label,color in [(-.12,'cap_old','Old capped',ORANGE),(0,'cap_env','Envelope capped',TEAL),(.12,'signed_env','Envelope signed',PURPLE)]:
            mu,err=stats(g[g.method==m].coverage)
            axs[0,1].errorbar(idx+delta,mu,yerr=err,fmt='o',color=color,markersize=3,capsize=2,label=label if idx==0 else None)
    axs[0,1].axhline(.9,c=GRAY,ls='--',lw=1)
    axs[0,1].set(xticks=range(4),xticklabels=['Gauss.','Corr.','Laplace','Skewed'],ylabel='Joint coverage at n = 80',title='All three retain a coverage target')
    axs[0,1].legend(frameon=False,fontsize=6.5)
    for idx,family in enumerate(FAMILIES):
        g=main[(main.family==family)&(main.n==30)]
        for off,a,b,label,color in [(-.12,'cap_env','cap_old','Envelope refinement',TEAL),(.12,'signed_env','cap_env','Changing to signed scores',PURPLE)]:
            rr,exc=paired(g,a,b);mu,err=stats(100*(1-rr))
            axs[1,0].errorbar(idx+off,mu,yerr=err,fmt='o',color=color,markersize=4,capsize=3,label=label if idx==0 else None)
    axs[1,0].axhline(0,c=GRAY,ls='--',lw=1)
    axs[1,0].set(xticks=range(4),xticklabels=['Gauss.','Corr.','Laplace','Skewed'],ylabel='Paired volume reduction (%)',title='Separate the two sources of change')
    axs[1,0].legend(frameon=False,fontsize=6.5)
    for idx,ba in enumerate([.1,.02]):
        g=df[(df.study=='base_width')&(df.base_alpha==ba)]
        for off,m,label,color in [(-.16,'base','Base',GRAY),(0,'cap_old','Old capped',ORANGE),(.16,'signed_env','Envelope signed',PURPLE)]:
            mu,err=stats(g[g.method==m].coverage)
            axs[1,1].errorbar(idx+off,mu,yerr=err,fmt='o',color=color,markersize=4,capsize=3,label=label if idx==0 else None)
    axs[1,1].axhline(.9,c=GRAY,ls='--',lw=1)
    axs[1,1].set(xticks=[0,1],xticklabels=['90% base','98% base'],ylabel='Joint coverage, two outcomes',title='Conservative bases allow shrinkage')
    axs[1,1].legend(frameon=False,fontsize=6.5)
    for ax in axs.ravel():decorate(ax)
    pd.DataFrame(summary).to_csv(OUT/'signed_comparison.csv',index=False)
    finish(fig,'signed_comparison')

def runtime_plot():
    rt=pd.read_csv(OUT/'runtime.csv');summary=[]
    fig,axs=plt.subplots(2,3,figsize=(9.7,5.2),layout='constrained')
    for row,mode in enumerate(['abs','cap']):
        for col,d in enumerate([2,6,12]):
            ax=axs[row,col]
            methods=[('old','Old shortcut'),('env','Envelope backward'),('rank','Envelope rank')]
            if mode=='cap':methods.append(('signed','Signed envelope (2 domains)'))
            for m,label in methods:
                gg=rt[(rt['mode']==mode)&(rt.d==d)&(rt.method==m)]
                pts=[]
                for n,g in gg.groupby('n'):
                    pts.append((n,g.ms.median(),g.ms.quantile(.25),g.ms.quantile(.75)))
                    summary.append(dict(mode=mode,d=d,n=n,method=m,median_ms=g.ms.median(),q25=g.ms.quantile(.25),q75=g.ms.quantile(.75)))
                pts=np.array(pts)
                ax.plot(pts[:,0],pts[:,1],marker='o',markersize=3,c=COLORS[m],label=label)
                ax.fill_between(pts[:,0],pts[:,2],pts[:,3],color=COLORS[m],alpha=.13)
            ax.set(xscale='log',yscale='log',xlabel='Calibration size',title=f'{"Absolute" if mode=="abs" else "Capped quantile"} | d = {d}',xticks=[30,80,200,500,1000],xticklabels=['30','80','200','500','1k'])
            ax.yaxis.set_major_locator(LogLocator(base=10,subs=(1,2,5),numticks=6))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'))
            ax.yaxis.set_minor_formatter(NullFormatter())
            if col==0:ax.set_ylabel('Construction time (ms)')
            decorate(ax)
    axs[0,0].legend(frameon=False,fontsize=6.5)
    axs[1,0].legend(frameon=False,fontsize=6.0,loc='upper left')
    pd.DataFrame(summary).to_csv(OUT/'runtime_summary.csv',index=False)
    finish(fig,'runtime')

def full_accept(cal,candidates,alpha):
    n=len(cal);m=cal.mean(0);s=cal.std(0);rank=conformal_rank(n,alpha)
    result=[]
    for chunk in np.array_split(candidates,max(1,len(candidates)//400)):
        mu=(n*m+chunk)/(n+1);sd=np.sqrt(s*s+(chunk-m)**2/(n+1))
        cs=((cal[None,:,:]-mu[:,None,:])/sd[:,None,:]).max(2)
        q=np.partition(cs,rank-1,axis=1)[:,rank-1]
        result.append(((chunk-mu)/sd).max(1)<=q)
    return np.concatenate(result)

def geometry_plot():
    fig,axs=plt.subplots(1,2,figsize=(9.1,4),layout='constrained');meta=[]
    for ax,family,mode,label in zip(axs,['laplace','skewed'],['abs','cap'],['Absolute residuals | Laplace','Capped quantile residuals | skewed']):
        a=np.load(OUT/f'geometry_{family}.npz');scales=a['scales'];idx=0
        old=a['bound_'+mode+'_old'];env=a['bound_'+mode+'_env']
        half=np.zeros(2) if mode=='abs' else a['width_test'][idx]/2
        radii={k:(half+b)/scales for k,b in [('old',old),('env',env)]}
        limit=radii['old']*1.15
        xs=np.linspace(-limit[0],limit[0],241);ys=np.linspace(-limit[1],limit[1],241)
        xx,yy=np.meshgrid(xs,ys);cand=np.column_stack([xx.ravel(),yy.ravel()])*scales
        cal=a['abs_cal'] if mode=='abs' else np.maximum(a['raw_cal'],0)
        scores=abs(cand) if mode=='abs' else np.maximum(abs(cand)-half,0)
        accepted=full_accept(cal,scores,.1)
        tol=1e-8*(1+env)
        assert np.all(scores[accepted]<=env+tol)
        ax.contourf(xx,yy,accepted.reshape(xx.shape).astype(float),levels=[.5,1.5],colors=['#DBE4EC'])
        for k in ['old','env']:
            r=radii[k]
            ax.add_patch(Rectangle(-r,2*r[0],2*r[1],fill=False,ec=COLORS[k],lw=2.1,ls='--' if k=='old' else '-'))
        ax.plot(0,0,'+',c=INK,markersize=9)
        ratio=float(np.prod(radii['env'])/np.prod(radii['old']))
        ax.set(xlabel='Outcome 1 deviation / noise scale',ylabel='Outcome 2 deviation / noise scale',
               xlim=(-limit[0],limit[0]),ylim=(-limit[1],limit[1]),title=label+f'\nEnvelope / old area = {ratio:.3f}')
        ax.set_aspect('equal',adjustable='box');decorate(ax,False)
        inset=ax.inset_axes([.18,.64,.4,.27])
        for key in ['old','env']:
            r=radii[key]
            inset.add_patch(Rectangle(-r,2*r[0],2*r[1],fill=False,ec=COLORS[key],lw=1.6,ls='--' if key=='old' else '-'))
        rx,ry=radii['old']
        inset.set(xlim=(rx*.965,rx*1.012),ylim=(ry*.94,ry*1.015))
        inset.tick_params(labelsize=6);inset.set_title('Corner detail',fontsize=7)
        for spine in inset.spines.values():spine.set_visible(True)
        meta.append(dict(family=family,mode=mode,test_index=0,area_ratio=ratio,
            full_grid_candidates=len(cand),accepted_grid_candidates=int(accepted.sum()),full_containment_violations=0))
    fig.legend(handles=[Line2D([0],[0],c=ORANGE,ls='--',label='Old shortcut'),Line2D([0],[0],c=TEAL,label='Envelope'),Patch(fc='#DBE4EC',label='Full conformal (grid)')],loc='outside lower center',ncol=3,fontsize=8,frameon=False)
    dump(OUT/'geometry_audit.json',meta);finish(fig,'fixed_point_geometry')

def signed_geometry():
    fig,axs=plt.subplots(1,2,figsize=(9.1,4.0),layout='constrained');meta=[]
    for ax,k,ba in zip(axs,[18,19],[.1,.02]):
        a=np.load(OUT/f'data/config_{k:02d}/trial_000.npz');idx=0;g=a['group_test'][idx];scales=a['scales']
        half=a['width_test'][idx]/2
        rr={}
        for m in ['base','cap_old','cap_env','signed_env']:
            b=a['bound_'+m];b=b if b.ndim==1 else b[g]
            rr[m]=(half+b)/scales
        finite=[r for r in rr.values() if np.isfinite(r).all()]
        limit=np.max(finite,axis=0)*1.14
        for m in ['base','cap_old','cap_env','signed_env']:
            r=rr[m]
            if not np.isfinite(r).all():continue
            ax.add_patch(Rectangle(-r,2*r[0],2*r[1],fill=m=='base',fc='#F1F3F6' if m=='base' else 'none',ec=COLORS[m],lw=1.7,ls='--' if m in ['base','cap_old'] else '-'))
        ax.plot(0,0,'+',c=INK,markersize=9)
        ax.set(xlim=(-limit[0],limit[0]),ylim=(-limit[1],limit[1]),xlabel='Outcome 1 deviation / noise scale',ylabel='Outcome 2 deviation / noise scale',title=f'{100*(1-ba):.0f}% fitted marginal base | trial 0, x index 0')
        ax.set_aspect('equal',adjustable='box')
        if not np.isfinite(rr['cap_old']).all():ax.text(.03,.95,'Old capped region is infinite',transform=ax.transAxes,fontsize=8,va='top',color=ORANGE)
        meta.append(dict(config=k,test_index=0,radii={m:r.tolist() for m,r in rr.items()}))
    fig.legend(handles=[Line2D([0],[0],color=COLORS[m],ls='--' if m in ['base','cap_old'] else '-',label=l) for m,l in [('base','Base interval'),('cap_old','Old capped'),('cap_env','Envelope capped'),('signed_env','Envelope signed')]],loc='outside lower center',ncol=4,fontsize=8,frameon=False)
    dump(OUT/'signed_geometry.json',meta);finish(fig,'signed_geometry')

def tables():
    rows=[]
    for family in FAMILIES:
        for mode in ['abs','cap']:
            g=main[(main.family==family)&(main.n==80)]
            old=g[g.method==mode+'_old'];env=g[g.method==mode+'_env']
            ratios,ex=paired(g,mode+'_env',mode+'_old');mu,err=stats(100*(1-ratios))
            rows.append(dict(family=family,mode=mode,old_coverage=old.coverage.mean(),env_coverage=env.coverage.mean(),
                old_mean_volume=old.volume.mean(),env_mean_volume=env.volume.mean(),reduction_pct=mu,reduction_ci=err,finite_pairs=len(ratios)))
    pd.DataFrame(rows).to_csv(OUT/'table_n80.csv',index=False)
    s='\\begin{tabular}{llrrrr}\n\\toprule\nScores & Noise & Old cov. & Env. cov. & Reduction & Pairs\\\\\n\\midrule\n'
    for row in rows:
        s+=f"{'Absolute' if row['mode']=='abs' else 'Capped CQR'} & {LABELS[row['family']]} & {row['old_coverage']:.4f} & {row['env_coverage']:.4f} & ${row['reduction_pct']:.2f}\\pm {row['reduction_ci']:.2f}\\%$ & {row['finite_pairs']}\\\\\n"
    s+='\\bottomrule\n\\end{tabular}\n'
    (HERE/'table_n80.tex').write_text(s)
    # Separate raw units from normalized ratios; volume moments can be noisy for tails.
    s='\\begin{tabular}{llrr}\n\\toprule\nScores & Noise & Old mean volume & Envelope mean volume\\\\\n\\midrule\n'
    for row in rows:
        s+=f"{'Absolute' if row['mode']=='abs' else 'Capped CQR'} & {LABELS[row['family']]} & {row['old_mean_volume']:.1f} & {row['env_mean_volume']:.1f}\\\\\n"
    s+='\\bottomrule\n\\end{tabular}\n';(HERE/'table_volume.tex').write_text(s)

if __name__=='__main__':
    main_panels('abs');main_panels('cap');alpha_plot();coordinate_plot();uniform_plot()
    signed_plot();runtime_plot();geometry_plot();signed_geometry();tables()
    print('Created 9 vector figures, PNG previews, paired evidence and tables.')
