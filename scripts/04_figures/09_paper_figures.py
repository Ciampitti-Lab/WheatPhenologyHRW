"""Canonical, FT-capable regenerator for the manuscript figures F3-F6.

Supersedes the per-figure scripts 03/04/05/06 for the current 7-model
pipeline: it uses the adopted best model per stage (FT-Transformer at
anthesis/maturity, tree/linear elsewhere), computed live under the
identical LOYO/LOSO protocol via scripts.utils.deep_models. Writes
PDF + PNG into docs/figures/. Deep stages need a GPU.

  F3  per-stage predicted vs observed (DOS), best model per stage
  F4  best-model-per-strategy comparison (panel A bars + panel B 5/8)
  F5  per-stage grouped feature importance (deep stages: permutation)
  F6  leave-one-state-out transferability (adopted best per stage)
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.config import CFG, REPO_ROOT
from scripts.utils.deep_models import (ORDER, ADOPT, LOSO_STATES, WE,
    load_cohort, stage_frame, loyo, fold_pred_ensemble, fold_pred_ft, r2)
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

WORK = REPO_ROOT / CFG.paths.work_dir
PHENO = str(REPO_ROOT / CFG.paths.phenology_matched)
FIG = REPO_ROOT / 'docs' / 'figures'; FIG.mkdir(parents=True, exist_ok=True)
GRID = WORK / 'v3_results' / 'phase_e_grid.parquet'
GOLD='#CEB888'; ACCENT='#8E6F3E'; DARK='#1B1B1B'; GREY='#8a8a8a'
TITLE={s:s.replace('_',' ').title() for s in ORDER}
GROUPS=['WES','HLS phenometrics','MODIS LST','Thermal-time','Daymet meteo','State encoders']
GCOL={'WES':ACCENT,'HLS phenometrics':GOLD,'MODIS LST':'#6E8E9E','Thermal-time':'#9E6E8E',
      'Daymet meteo':'#7E9E6E','State encoders':'#BBB'}
plt.rcParams.update({'font.size':16,'axes.titlesize':19,'axes.labelsize':17,
    'xtick.labelsize':15,'ytick.labelsize':15,'legend.fontsize':14,
    'figure.titlesize':22,'axes.titleweight':'bold'})
SEEDS=[0,1,2,3,4]

def group_of(c):
    if c in WE: return 'WES'
    if c.startswith(('NDVI','EVI','GCVI','DL_')) or 'phenometr' in c: return 'HLS phenometrics'
    if c.startswith('lst_'): return 'MODIS LST'
    if c.startswith('state_'): return 'State encoders'
    if c.startswith(('GDD','VD','fV','PPT','photoperiod')): return 'Thermal-time'
    return 'Daymet meteo'

def pooled(d,feat,tgt,mod):
    P,T=[],[]
    for yr in sorted(d['year'].unique()):
        tr,te=d[d['year']!=yr],d[d['year']==yr]
        if len(tr)<50 or len(te)<5: continue
        if mod=='FT':
            P.extend(np.mean([fold_pred_ft(tr,te,feat,tgt,s) for s in SEEDS],axis=0))
        else:
            P.extend(fold_pred_ensemble(tr,te,feat,tgt,mod))
        T.extend(te[tgt].values)
    return np.array(T),np.array(P)

def F3(fe,cols):
    fig,ax=plt.subplots(2,4,figsize=(20.7,10.4),dpi=150)
    for i,s in enumerate(ORDER):
        a=ax.flat[i]; strat,mod=ADOPT[s]; d,tgt=stage_frame(fe,s)
        T,P=pooled(d,cols(s,strat=='C_Hybrid'),tgt,mod)
        R=r2(T,P); rmse=np.sqrt(np.mean((T-P)**2)); lo,hi=min(T.min(),P.min()),max(T.max(),P.max())
        a.scatter(T,P,s=9,alpha=.32,color=ACCENT,edgecolors='none')
        a.plot([lo,hi],[lo,hi],'-',color=DARK,lw=1)
        a.fill_between([lo,hi],[lo-10,hi-10],[lo+10,hi+10],color=GOLD,alpha=.22,lw=0)
        a.set_title(f'{TITLE[s]}  ({"FT-Transformer" if mod=="FT" else mod})',fontsize=18)
        a.text(.04,.96,f'$R^2$={R:.2f}\nRMSE={rmse:.1f} d\n$n$={len(T)}',transform=a.transAxes,
               va='top',fontsize=15,bbox=dict(boxstyle='round',fc='white',ec=GREY,alpha=.9))
        a.set_xlabel('Observed DOS',fontsize=14); a.set_ylabel('Predicted DOS',fontsize=14)
        a.grid(alpha=.18,lw=.5)
        for sp in('top','right'): a.spines[sp].set_visible(False)
    fig.suptitle('Per-stage predicted vs. observed timing (LOYO, best model per stage)',
                 fontsize=22,fontweight='bold',y=1.005)
    fig.tight_layout()
    fig.savefig(FIG/'F3_per_stage_scatter.pdf',bbox_inches='tight')
    fig.savefig(FIG/'F3_per_stage_scatter.png',dpi=200,bbox_inches='tight'); plt.close(fig); print('F3 ok')

def F4():
    g=pd.read_parquet(GRID)
    bv=np.array([g[(g.stage==s)&(g.strategy=='B_ML-only')].R2.max() for s in ORDER])
    cv=np.array([g[(g.stage==s)&(g.strategy=='C_Hybrid')].R2.max() for s in ORDER])
    fig,(a1,a2)=plt.subplots(1,2,figsize=(16.1,6.4),dpi=150,gridspec_kw={'width_ratios':[3,2]})
    x=np.arange(8); w=.36
    a1.bar(x-w/2,bv,w,label='ML-only (no WES)',color=GREY,edgecolor=DARK,lw=.6)
    a1.bar(x+w/2,cv,w,label='Hybrid (WES + ML)',color=ACCENT,edgecolor=DARK,lw=.6)
    for i,(b,c) in enumerate(zip(bv,cv)):
        dr=round(c-b,2)
        mk,col=((f'▲{dr:.02f}','#1B6E63') if dr>0 else
                (f'▼{abs(dr):.02f}','#A03939') if dr<0 else ('≈0.00',GREY))
        a1.text(i,max(b,c)+.03,mk,ha='center',fontsize=12,color=col,fontweight='bold')
    a1.set_xticks(x); a1.set_xticklabels([TITLE[s] for s in ORDER],rotation=30,ha='right',fontsize=15)
    a1.set_ylabel('LOYO $R^2$',fontsize=17); a1.set_ylim(-.05,1.0); a1.axhline(0,color='k',lw=.4)
    a1.set_title('A. Best model per strategy, by stage',fontsize=18,pad=10)
    a1.legend(loc='upper left',fontsize=14); a1.grid(True,axis='y',alpha=.2,lw=.5)
    for sp in('top','right'): a1.spines[sp].set_visible(False)
    dd=np.round(cv-bv,2); nh=int((dd>0).sum()); nm=int((dd<0).sum()); nt=int((dd==0).sum())
    bars=a2.bar(['Hybrid\nhigher','ML-only\nhigher','Tie'],[nh,nm,nt],
                color=[ACCENT,GREY,'#CCC'],edgecolor=DARK,lw=.6)
    for bb,vv in zip(bars,[nh,nm,nt]):
        a2.text(bb.get_x()+bb.get_width()/2,vv+.1,str(vv),ha='center',fontsize=19,fontweight='bold')
    a2.set_ylim(0,max(nh,nm,nt)+1.5); a2.set_ylabel('Stages (of 8)',fontsize=17)
    a2.set_title(f'B. Hybrid higher in {nh}/8 (best-vs-best;\nmean $\\Delta R^2$={np.mean(cv-bv):+.2f})',
                 fontsize=18,pad=10)
    a2.grid(True,axis='y',alpha=.2,lw=.5)
    for sp in('top','right'): a2.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG/'F4_strategy_comparison.pdf',bbox_inches='tight')
    fig.savefig(FIG/'F4_strategy_comparison.png',dpi=200,bbox_inches='tight'); plt.close(fig)
    print(f'F4 ok (Hybrid {nh}/8)')

def _imp(fe,cols,s):
    strat,mod=ADOPT[s]; d,tgt=stage_frame(fe,s); feat=cols(s,strat=='C_Hybrid')
    if mod=='FT':
        groups=sorted({group_of(c) for c in feat}); base=[]; perm={g:[] for g in groups}
        rng=np.random.RandomState(0)
        for yr in sorted(d['year'].unique()):
            tr,te=d[d['year']!=yr],d[d['year']==yr]
            if len(tr)<50 or len(te)<5: continue
            base.append(r2(te[tgt].values,fold_pred_ft(tr,te,feat,tgt,0)))
            for g in groups:
                cs=[c for c in feat if group_of(c)==g]; te2=te.copy()
                for c in cs: te2[c]=rng.permutation(te2[c].values)
                perm[g].append(r2(te2[tgt].values,fold_pred_ft(tr,te2,feat,tgt,0)))
        b=float(np.mean(base)); drop={g:max(0.,b-float(np.mean(perm[g]))) for g in groups}
        t=sum(drop.values()) or 1.
        return {g:100*drop.get(g,0)/t for g in GROUPS}
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    pp=Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())])
    X=pp.fit_transform(d[feat]); y=d[tgt].values
    if mod in('ElasticNet','Ridge'):
        from sklearn.linear_model import ElasticNetCV,RidgeCV
        m=(ElasticNetCV(l1_ratio=[.1,.5,.9,1.],n_alphas=20,max_iter=20000,cv=5)
           if mod=='ElasticNet' else RidgeCV(alphas=np.logspace(-3,3,30),cv=5))
        m.fit(X,y); imp=np.abs(m.coef_)
    else:
        from xgboost import XGBRegressor
        from lightgbm import LGBMRegressor
        m=(XGBRegressor(n_estimators=300,max_depth=5,learning_rate=0.05,random_state=42,verbosity=0)
           if mod=='XGBoost' else LGBMRegressor(n_estimators=300,max_depth=5,learning_rate=0.05,
                                                random_state=42,verbose=-1))
        m.fit(X,y); imp=m.feature_importances_.astype(float)
    agg={g:0. for g in GROUPS}
    for c,v in zip(feat,imp): agg[group_of(c)]=agg.get(group_of(c),0.)+float(v)
    t=sum(agg.values()) or 1.
    return {g:100*agg[g]/t for g in GROUPS}

def F5(fe,cols):
    mat=np.array([[_imp(fe,cols,s)[g] for g in GROUPS] for s in ORDER])
    fig,ax=plt.subplots(figsize=(13.8,8.0),dpi=150); y=np.arange(8); left=np.zeros(8)
    for gi,g in enumerate(GROUPS):
        ax.barh(y,mat[:,gi],left=left,color=GCOL[g],edgecolor='white',lw=.5,label=g); left+=mat[:,gi]
    ax.set_yticks(y); ax.set_yticklabels([TITLE[s] for s in ORDER],fontsize=16); ax.invert_yaxis()
    ax.set_xlabel('Feature-group importance (%)',fontsize=18,labelpad=10); ax.set_xlim(0,100)
    ax.set_title('Per-stage feature-group importance (best model per stage;\n'
                 'deep-model stages use grouped permutation importance)',fontsize=19,pad=10)
    ax.legend(ncol=6,fontsize=12,loc='upper center',bbox_to_anchor=(.5,-.12),
              columnspacing=1.0,handletextpad=.5)
    for sp in('top','right'): ax.spines[sp].set_visible(False)
    fig.subplots_adjust(bottom=.20)
    fig.savefig(FIG/'F5_feature_importance.pdf',bbox_inches='tight')
    fig.savefig(FIG/'F5_feature_importance.png',dpi=200,bbox_inches='tight'); plt.close(fig); print('F5 ok')

def F6(fe,cols):
    M=np.full((8,5),np.nan)
    for i,s in enumerate(ORDER):
        strat,mod=ADOPT[s]; d,tgt=stage_frame(fe,s); fc=cols(s,strat=='C_Hybrid')
        for j,st in enumerate(LOSO_STATES):
            T,P=loyo(d,fc,tgt,mod,split='state',holdout=st)
            M[i,j]=r2(T,P) if len(T)>5 else np.nan
    Mc=np.clip(M,-1,1)
    fig,ax=plt.subplots(figsize=(9.8,9.2),dpi=150)
    im=ax.imshow(Mc,cmap='RdYlGn',norm=TwoSlopeNorm(vmin=-1,vcenter=.3,vmax=1),aspect='auto')
    ax.set_xticks(range(5)); ax.set_xticklabels(LOSO_STATES,fontsize=15)
    ax.set_yticks(range(8)); ax.set_yticklabels([TITLE[s] for s in ORDER],fontsize=15)
    ax.set_xlabel('Held-out state',fontsize=17)
    for i in range(8):
        for j in range(5):
            v=M[i,j]
            ax.text(j,i,'—' if np.isnan(v) else f'{v:.2f}',ha='center',va='center',
                    fontsize=15,fontweight='bold',
                    color=DARK if (not np.isnan(v) and v>-.2) else 'white')
    ax.set_title('Leave-one-state-out transferability ($R^2$;\nbest model per stage)',
                 fontsize=19,pad=10)
    fig.colorbar(im,ax=ax,shrink=.8,label='$R^2$ (clipped to $[-1,1]$)')
    fig.tight_layout()
    fig.savefig(FIG/'F6_loso_transferability.pdf',bbox_inches='tight')
    fig.savefig(FIG/'F6_loso_transferability.png',dpi=200,bbox_inches='tight'); plt.close(fig); print('F6 ok')

if __name__=='__main__':
    fe,cols=load_cohort(WORK,PHENO)
    F4(); F3(fe,cols); F5(fe,cols); F6(fe,cols)
    print('F3-F6 regenerated into docs/figures/')
