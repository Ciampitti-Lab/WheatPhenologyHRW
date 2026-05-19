"""Shared 7-model Phase-E core (incl. TabNet, FT-Transformer).

Validated logic carried over verbatim from the analysis that produced
the manuscript figures/tables; the only change is that the working
directory is passed in (config-driven via scripts.utils.config), so the
public repo stays portable. Deep models: inputs+target standardised on
the training fold, inner-year early stopping, predictions averaged over
five seeds; identical leave-one-year-out protocol for every model.

Used by:
  scripts/03_modeling/01_phase_e_loyo.py        (7-model grid + best + LOSO)
  scripts/04_figures/09_paper_figures.py        (F3-F6)
  scripts/05_analysis/0{5,6}_*                  (ablations / negative control)
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

STAGE_MAP = {
 'emergence':['Emerging','Emerging - Seedling','Shoot - Emerging','Shoot','Seedling',
              'Seedling - 1 Leaf','1 Leaf','2 Leaf','3 Leaf','4 Leaf'],
 'tillering':['Begin Tillering','Tillering','1-2 Tiller','2-4 Tiller','4-6 Tiller',
              '6-8 Tiller','8+ Tiller','Full Tillering','End Tillering'],
 'jointing':['Jointing','1st Node Visible','2nd Node Visible','3rd Node Visible','Spring Vegetative'],
 'flag_leaf':['Flag Leaf Emerging','Flag Leaf Emerged'],
 'boot':['Early Boot','Boot'],'heading':['Head Emerging','Heading','Complete Heading'],
 'anthesis':['Early Bloom','Bloom'],'maturity':['Maturity','Harvest Ready','Ready For Harvesting']}
SPRING={'tillering','jointing'}; EARLY={'emergence','tillering','jointing'}
META_FIXED=['field_id','year','flag_true_doy','n_obs','sowing_doy_used']
REDUND=['GDD_M2_at_SOS','VD_at_SOS','emergence_doy','VD_from_emergence_at_SOS',
        'fV_from_emergence_at_SOS','days_emergence_to_SOS']
WE=['WE_emergence_doy','WE_tillering_doy','WE_jointing_doy','WE_flag_leaf_doy',
    'WE_boot_doy','WE_heading_doy','WE_anthesis_doy','WE_maturity_doy']
STATE_OH=['state_TX','state_OK','state_KS','state_NE','state_CO','state_NM']
LOSO_STATES=['TX','OK','KS','NE','CO']
K_GRID=(20,40,60,80,None); LIN={'ElasticNet','Ridge'}; DEEP={'TabNet','FT'}
MODELS=['ElasticNet','Ridge','RandomForest','XGBoost','LightGBM','TabNet','FT']
ORDER=['emergence','tillering','jointing','flag_leaf','boot','heading','anthesis','maturity']
# Best (strategy, model) per stage carried to the manuscript Table; FT
# adopted only where it genuinely wins (anthesis, maturity); flag_leaf /
# heading kept tree/linear (FT only ties them at two decimals).
ADOPT={'emergence':('C_Hybrid','LightGBM'),'tillering':('B_ML-only','ElasticNet'),
 'jointing':('C_Hybrid','LightGBM'),'flag_leaf':('C_Hybrid','XGBoost'),
 'boot':('C_Hybrid','LightGBM'),'heading':('C_Hybrid','ElasticNet'),
 'anthesis':('C_Hybrid','FT'),'maturity':('C_Hybrid','FT')}
SEEDS=[0,1,2,3,4]

def is_win(c): return (c.endswith(('_gf','_pa','_pa_late'))
                       or c.startswith(('heat_days_','hot_days_','frost_days_')))
def r2(T,P):
    T,P=np.asarray(T,float),np.asarray(P,float)
    d=np.sum((T-T.mean())**2); return float(1-np.sum((T-P)**2)/d) if d>0 else 0.0
def boot_ci(T,P,nit=2000,sd=42):
    rng=np.random.RandomState(sd); n=len(T); o=[]
    for _ in range(nit):
        i=rng.choice(n,n,True); o.append(r2(T[i],P[i]))
    return float(np.percentile(o,2.5)),float(np.percentile(o,97.5))
def metrics(T,P):
    T,P=np.asarray(T,float),np.asarray(P,float); lo,hi=boot_ci(T,P)
    return dict(R2=r2(T,P),RMSE=float(np.sqrt(np.mean((T-P)**2))),
                w10=float(np.mean(np.abs(T-P)<=10)*100),R2_lo=lo,R2_hi=hi,n=len(T))

def _ens(mn):
    return {'ElasticNet':lambda:ElasticNetCV(l1_ratio=[.1,.3,.5,.7,.9,.95,1.0],n_alphas=20,
              max_iter=20000,cv=5,n_jobs=8),
     'Ridge':lambda:RidgeCV(alphas=np.logspace(-3,3,30),cv=5),
     'RandomForest':lambda:RandomForestRegressor(n_estimators=200,n_jobs=8,random_state=42),
     'XGBoost':lambda:XGBRegressor(n_estimators=300,max_depth=5,learning_rate=0.05,
              n_jobs=8,random_state=42,verbosity=0),
     'LightGBM':lambda:LGBMRegressor(n_estimators=300,max_depth=5,learning_rate=0.05,
              n_jobs=8,random_state=42,verbose=-1)}[mn]

def _prep(tr,te,feat,tg):
    imp=SimpleImputer(strategy='median').fit(tr[feat]); xs=StandardScaler()
    Xtr=xs.fit_transform(imp.transform(tr[feat])).astype('float32')
    Xte=xs.transform(imp.transform(te[feat])).astype('float32')
    ys=StandardScaler().fit(tr[[tg]].values)
    ytr=ys.transform(tr[[tg]].values).astype('float32').ravel()
    return Xtr,Xte,ytr,ys

def fold_pred_ensemble(tr,te,feat,tg,mn):
    if mn in LIN:
        bk,bs=None,-np.inf
        for k in K_GRID:
            st=[('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())]
            if k is not None: st.append(('sel',SelectKBest(f_regression,k=min(k,len(feat)))))
            st.append(('m',_ens(mn)())); pp=Pipeline(st); iy=sorted(tr['year'].unique())
            if len(iy)<2: pp.fit(tr[feat],tr[tg]); sc=0
            else:
                vy=iy[-1]; it,iv=tr[tr['year']!=vy],tr[tr['year']==vy]
                pp.fit(it[feat],it[tg]); sc=r2(iv[tg].values,pp.predict(iv[feat]))
            if sc>bs: bs,bk=sc,k
        st=[('imp',SimpleImputer(strategy='median')),('sc',StandardScaler())]
        if bk is not None: st.append(('sel',SelectKBest(f_regression,k=min(bk,len(feat)))))
        st.append(('m',_ens(mn)())); pp=Pipeline(st)
    else:
        pp=Pipeline([('imp',SimpleImputer(strategy='median')),('sc',StandardScaler()),('m',_ens(mn)())])
    pp.fit(tr[feat],tr[tg]); return np.ravel(pp.predict(te[feat]))

def fold_pred_tabnet(tr,te,feat,tg,seed):
    from pytorch_tabnet.tab_model import TabNetRegressor
    import torch
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    iy=sorted(tr['year'].unique()); vy=iy[-1]
    it,iv=tr[tr['year']!=vy],tr[tr['year']==vy]
    Xi,Xv,yi,ys=_prep(it,iv,feat,tg); yv=ys.transform(iv[[tg]].values).astype('float32').ravel()
    m=TabNetRegressor(n_d=16,n_a=16,n_steps=4,gamma=1.5,seed=seed,
                      optimizer_params=dict(lr=2e-2),verbose=0,device_name=dev)
    m.fit(Xi,yi.reshape(-1,1),eval_set=[(Xv,yv.reshape(-1,1))],
          max_epochs=200,patience=30,batch_size=1024,virtual_batch_size=256)
    ne=max(30,len(m.history['loss']))
    Xa,Xt,_,ys2=_prep(tr,te,feat,tg)
    m2=TabNetRegressor(n_d=16,n_a=16,n_steps=4,gamma=1.5,seed=seed,
                       optimizer_params=dict(lr=2e-2),verbose=0,device_name=dev)
    m2.fit(Xa,ys2.transform(tr[[tg]].values).astype('float32'),
           max_epochs=ne,patience=ne,batch_size=1024,virtual_batch_size=256)
    return ys2.inverse_transform(m2.predict(Xt)).ravel()

def fold_pred_ft(tr,te,feat,tg,seed):
    import torch, torch.nn as nn
    from rtdl_revisiting_models import FTTransformer
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(seed); np.random.seed(seed)
    iy=sorted(tr['year'].unique()); vy=iy[-1]
    it,iv=tr[tr['year']!=vy],tr[tr['year']==vy]
    Xi,Xv,yi,ys=_prep(it,iv,feat,tg); yv=ys.transform(iv[[tg]].values).astype('float32').ravel()
    ncf=Xi.shape[1]
    mk=lambda:FTTransformer(n_cont_features=ncf,cat_cardinalities=[],d_out=1,
                            **FTTransformer.get_default_kwargs()).to(dev)
    def train(model,X,y,Xv_,yv_,mx):
        opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)
        Xt=torch.tensor(X,device=dev); yt=torch.tensor(y,device=dev).view(-1,1)
        Xvt=torch.tensor(Xv_,device=dev); best=1e18; bad=0; bep=mx; n=len(Xt); bs=256
        for ep in range(mx):
            model.train(); perm=torch.randperm(n,device=dev)
            for i in range(0,n,bs):
                idx=perm[i:i+bs]; opt.zero_grad()
                loss=nn.functional.mse_loss(model(Xt[idx],None),yt[idx]); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                v=nn.functional.mse_loss(model(Xvt,None).cpu().view(-1),torch.tensor(yv_)).item()
            if v<best-1e-4: best=v; bad=0; bep=ep+1
            else:
                bad+=1
                if bad>=30: break
        return bep
    mdl=mk(); ep=train(mdl,Xi,yi,Xv,yv,200)
    Xa,Xt,_,ys2=_prep(tr,te,feat,tg); ya=ys2.transform(tr[[tg]].values).astype('float32').ravel()
    fin=mk(); train(fin,Xa,ya,Xv,yv,max(30,ep)); fin.eval()
    with torch.no_grad():
        pr=fin(torch.tensor(Xt,device=dev),None).cpu().numpy().ravel()
    return ys2.inverse_transform(pr.reshape(-1,1)).ravel()

def loyo(d,feat,tg,mn,split='year',holdout=None):
    """split='year' -> LOYO; split='state' -> one LOSO fold (holdout state,
    its one-hot encoder zeroed in the test set). Deep models are averaged
    over five seeds."""
    P,T=[],[]
    if split=='year':
        folds=[(d[d['year']!=y],d[d['year']==y]) for y in sorted(d['year'].unique())]
    else:
        tr=d[d['state']!=holdout].copy(); te=d[d['state']==holdout].copy()
        for c in STATE_OH:
            if c in feat: te[c]=0.0
        folds=[(tr,te)]
    for tr,te in folds:
        if len(tr)<50 or len(te)<5: continue
        if mn in DEEP:
            fn=fold_pred_tabnet if mn=='TabNet' else fold_pred_ft
            ps=[]
            for s in SEEDS:
                try: ps.append(fn(tr,te,feat,tg,s))
                except Exception: pass
            if not ps: continue
            pr=np.mean(ps,axis=0)
        else:
            pr=fold_pred_ensemble(tr,te,feat,tg,mn)
        P.extend(pr); T.extend(te[tg].values)
    return np.array(T),np.array(P)

def load_cohort(work_dir, pheno_path):
    """Build the feature+target+state frame and a per-(stage,strategy)
    feature-column selector. work_dir / pheno_path come from config."""
    from pathlib import Path
    work_dir=Path(work_dir)
    ph=pd.read_parquet(pheno_path)
    ph['year']=ph['growing_season'].str.split('-').str[1].astype(int)
    ph['field_id']=ph['FIELDID'].astype(str)
    tg=None
    for s,l in STAGE_MAP.items():
        x=ph[ph['growth_stage'].isin(l)].copy()
        if s in SPRING: x=x[x['dos']>200]
        if s=='maturity': x=x[x['dos']>=280]
        e=x.groupby(['field_id','year'])['dos'].min().reset_index().rename(columns={'dos':s+'_dos_obs'})
        tg=e if tg is None else tg.merge(e,on=['field_id','year'],how='outer')
    fe=pd.read_parquet(work_dir/'features_v3_realsowing_train.parquet')
    fe['field_id']=fe['field_id'].astype(str); fe['year']=fe['year'].astype(int)
    if 'state' in fe.columns: fe=fe.drop(columns=['state'])
    fe=fe.merge(tg,on=['field_id','year'],how='left')
    fe['state']=fe[STATE_OH].idxmax(axis=1).str.replace('state_','')
    tc=[s+'_dos_obs' for s in STAGE_MAP]; M=META_FIXED+tc+['state']
    ndre=[c for c in fe.columns if c.startswith('NDRE')]
    allc=[c for c in fe.columns if c not in M and c not in ndre and c not in REDUND
          and pd.api.types.is_numeric_dtype(fe[c])]
    mlc=[c for c in allc if c not in WE]
    winc=[c for c in fe.columns if is_win(c) and c not in M]
    def cols(stage,wes):
        b=allc if wes else mlc
        return [c for c in b if c not in winc] if stage in EARLY else b
    return fe,cols

def stage_frame(fe,s):
    tgt=s+'_dos_obs'; d=fe.dropna(subset=[tgt]).copy()
    q1,q9=d[tgt].quantile([.01,.99]); return d[(d[tgt]>=q1)&(d[tgt]<=q9)].copy(),tgt
