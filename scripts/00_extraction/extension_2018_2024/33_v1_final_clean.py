"""V1 FINAL — clean retrain on the published mixed-coordinate features
without `ph_top` (SoilGrids soil pH, mostly NaN and near-zero ablation
contribution). All other features and the EARLY/late stage split are
identical to the published modeling pipeline.

Outputs:
    v1_final/multi_stage_models.parquet
    v1_final/multi_stage_best.csv
    v1_final/_run.log
"""
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

EXT = Path('/depot/ciampitti/data/WheatPhenologyHRW/data/raw/satellite/extension_2018_2024')
V1_FEAT = '/depot/ciampitti/data/WheatPhenologyHRW/data/processed/features/features_a6.parquet'
PHENO   = '/depot/ciampitti/data/WheatPhenologyHRW/data/processed/buffer_300m/wheat_hrw_phenology_buffer_matched.parquet'

OUT_DIR = EXT / 'v1_final'
OUT_DIR.mkdir(exist_ok=True)
OUT_RES = OUT_DIR / 'multi_stage_models.parquet'
OUT_BEST = OUT_DIR / 'multi_stage_best.csv'
LOG_PATH = OUT_DIR / '_run.log'

STAGE_MAP = {
    'emergence':  ['Emerging','Emerging - Seedling','Shoot - Emerging','Shoot',
                   'Seedling','Seedling - 1 Leaf','1 Leaf','2 Leaf','3 Leaf','4 Leaf'],
    'tillering':  ['Begin Tillering','Tillering','1-2 Tiller','2-4 Tiller','4-6 Tiller',
                   '6-8 Tiller','8+ Tiller','Full Tillering','End Tillering'],
    'jointing':   ['Jointing','1st Node Visible','2nd Node Visible','3rd Node Visible',
                   'Spring Vegetative'],
    'flag_leaf':  ['Flag Leaf Emerging','Flag Leaf Emerged'],
    'boot':       ['Early Boot','Boot'],
    'heading':    ['Head Emerging','Heading','Complete Heading'],
    'anthesis':   ['Early Bloom','Bloom'],
    'maturity':   ['Maturity','Harvest Ready','Ready For Harvesting'],
}
SPRING_ONLY = {'tillering','jointing'}
LINEAR = {'ElasticNet','Ridge'}
K_GRID = (20,40,60,80,None)

# Features to DROP — explicit list for paper transparency
DROP_NDRE = True            # S2-only, partial coverage prior to 2015 onboarding
DROP_REDUND = True          # multicollinear with primary features
DROP_PH_TOP = True          # mostly NaN, near-zero ablation contribution
REDUND = ['GDD_M2_at_SOS','VD_at_SOS','emergence_doy',
          'VD_from_emergence_at_SOS','fV_from_emergence_at_SOS','days_emergence_to_SOS']


def log(msg, fh):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line); fh.write(line + '\n'); fh.flush()


def build_targets(pheno):
    out = None
    for stage, labels in STAGE_MAP.items():
        s = pheno[pheno['growth_stage'].isin(labels)].copy()
        if stage in SPRING_ONLY:
            s = s[s['dos'] > 200]
        s['harvest_year'] = s['growing_season'].str.split('-').str[1].astype(int)
        s['field_id'] = s['FIELDID'].astype(str)
        e = s.groupby(['field_id','harvest_year'])['dos'].min().reset_index()
        e = e.rename(columns={'harvest_year':'year','dos':f'{stage}_dos_obs'})
        out = e if out is None else out.merge(e, on=['field_id','year'], how='outer')
    return out


def loyo(df, feat_cols, target, factory, model_name):
    df2 = df.dropna(subset=[target]).copy()
    q1, q99 = df2[target].quantile([0.01, 0.99])
    df2 = df2[(df2[target] >= q1) & (df2[target] <= q99)]
    is_linear = model_name in LINEAR
    pred_all, true_all = [], []
    for yr in sorted(df2['year'].unique()):
        tr = df2[df2['year'] != yr]; te = df2[df2['year'] == yr]
        if len(tr) < 50 or len(te) < 5: continue
        if is_linear:
            inner_yrs = sorted(tr['year'].unique())
            best_k, best_s = None, -np.inf
            for k in K_GRID:
                steps = [('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]
                if k is not None:
                    steps.append(('sel', SelectKBest(f_regression, k=min(k, len(feat_cols)))))
                steps.append(('m', factory()))
                pipe = Pipeline(steps)
                if len(inner_yrs) < 2:
                    pipe.fit(tr[feat_cols], tr[target]); s_val = 0
                else:
                    vy = inner_yrs[-1]
                    pipe.fit(tr[tr['year']!=vy][feat_cols], tr[tr['year']!=vy][target])
                    pinner = pipe.predict(tr[tr['year']==vy][feat_cols])
                    yv = tr[tr['year']==vy][target].values
                    denom = np.sum((yv-yv.mean())**2)
                    s_val = 1 - np.sum((yv-pinner)**2)/denom if denom>0 else 0
                if s_val > best_s:
                    best_s, best_k = s_val, k
            steps = [('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]
            if best_k is not None:
                steps.append(('sel', SelectKBest(f_regression, k=min(best_k, len(feat_cols)))))
            steps.append(('m', factory()))
            pipe = Pipeline(steps)
        else:
            pipe = Pipeline([('imp', SimpleImputer(strategy='median')),
                             ('sc',  StandardScaler()),
                             ('m',   factory())])
        pipe.fit(tr[feat_cols], tr[target])
        p = pipe.predict(te[feat_cols])
        if p.ndim > 1: p = p.ravel()
        pred_all.extend(p); true_all.extend(te[target].values)
    return np.array(pred_all), np.array(true_all)


def metrics(yt, yp):
    rmse = float(np.sqrt(np.mean((yt-yp)**2)))
    denom = np.sum((yt-yt.mean())**2)
    r2 = 1 - np.sum((yt-yp)**2)/denom if denom > 0 else 0
    w10 = float(np.mean(np.abs(yt-yp)<=10)*100)
    return float(r2), rmse, w10


def main():
    fh = open(LOG_PATH, 'w', buffering=1)
    t0 = time.time()

    log('Loading V1 features...', fh)
    feat = pd.read_parquet(V1_FEAT)
    feat['field_id'] = feat['field_id'].astype(str); feat['year'] = feat['year'].astype(int)
    if 'state' in feat.columns: feat = feat.drop(columns=['state'])
    log(f'Features shape: {feat.shape}', fh)

    pheno = pd.read_parquet(PHENO)
    targets = build_targets(pheno)
    feat = feat.merge(targets, on=['field_id','year'], how='left')

    # Define feature columns with explicit drops
    META = ['field_id','year','flag_true_doy','n_obs','sowing_doy_used'] + \
           [f'{s}_dos_obs' for s in STAGE_MAP]
    we_multi = ['WE_emergence_doy','WE_tillering_doy','WE_jointing_doy','WE_flag_leaf_doy',
                'WE_boot_doy','WE_heading_doy','WE_anthesis_doy','WE_maturity_doy']

    drop_set = set(META)
    if DROP_NDRE:
        drop_set |= {c for c in feat.columns if c.startswith('NDRE')}
    if DROP_REDUND:
        drop_set |= set(REDUND)
    if DROP_PH_TOP:
        drop_set.add('ph_top')

    log(f'\nFeature pipeline drops:', fh)
    log(f'  NDRE features:   {DROP_NDRE}  ({len([c for c in feat.columns if c.startswith("NDRE")])} columns)', fh)
    log(f'  REDUND set:      {DROP_REDUND}  ({len(REDUND)} columns)', fh)
    log(f'  ph_top:          {DROP_PH_TOP}', fh)

    all_feat = [c for c in feat.columns
                if c not in drop_set
                and pd.api.types.is_numeric_dtype(feat[c])]
    ml_only = [c for c in all_feat if c not in we_multi]

    WIN_PREF = ('heat_days_','hot_days_','frost_days_')
    WIN_SUFF = ('_gf','_pa','_pa_late')
    windowed = [c for c in feat.columns
                if (c.endswith(WIN_SUFF) or c.startswith(WIN_PREF)) and c not in META]
    EARLY = {'emergence','tillering','jointing'}

    log(f'\nFeature counts (after cleanup):', fh)
    log(f'  C_Hybrid (with WES): {len(all_feat)} features', fh)
    log(f'  B_ML-only (no WES):  {len(ml_only)} features', fh)
    log(f'  Early stages drop:   {len(windowed)} windowed features', fh)

    def stage_cols(stage, include_wes=True):
        base = all_feat if include_wes else ml_only
        if stage in EARLY:
            return [c for c in base if c not in windowed]
        return base

    factories = {
        'ElasticNet':   lambda: ElasticNetCV(l1_ratio=[.1,.3,.5,.7,.9,.95,1.0],
                                              n_alphas=20, max_iter=20000, cv=5, n_jobs=1),
        'Ridge':        lambda: RidgeCV(alphas=np.logspace(-3,3,30), cv=5),
        'RandomForest': lambda: RandomForestRegressor(n_estimators=200, n_jobs=1, random_state=42),
        'XGBoost':      lambda: XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                              n_jobs=1, random_state=42, verbosity=0),
        'LightGBM':     lambda: LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                                n_jobs=1, random_state=42, verbose=-1),
    }
    strategies = [('B_ML-only', False), ('C_Hybrid', True)]

    rows = []
    counter = 0
    total = len(STAGE_MAP) * 2 * 5
    for stage in STAGE_MAP:
        target = f'{stage}_dos_obs'
        if target not in feat.columns: continue
        n = feat[target].notna().sum()
        if n < 30:
            log(f'[skip] {stage}: only {n} labels', fh); continue
        for sname, inc_wes in strategies:
            cols = stage_cols(stage, include_wes=inc_wes)
            for mname, fac in factories.items():
                counter += 1
                t1 = time.time()
                try:
                    p, t = loyo(feat, cols, target, fac, mname)
                    r2, rmse, w10 = metrics(t, p)
                    rows.append({'stage':stage,'strategy':sname,'model':mname,
                                 'n':len(t),'R2':r2,'RMSE':rmse,'w10':w10,
                                 'fit_s':time.time()-t1})
                    log(f'[{counter}/{total}] {stage:<10} | {sname:<10} | {mname:<13}: '
                        f'R²={r2:+.3f} RMSE={rmse:.1f}d n={len(t)}  ({time.time()-t1:.1f}s)', fh)
                    pd.DataFrame(rows).to_parquet(OUT_RES, index=False)
                except Exception as e:
                    log(f'[{counter}/{total}] {stage} | {sname} | {mname}: FAILED {e}', fh)

    df = pd.DataFrame(rows)
    if len(df):
        best = df.loc[df.groupby('stage')['R2'].idxmax()]
        best.to_csv(OUT_BEST, index=False)
        log(f'\n=== BEST per stage (V1 final — ph_top removed) ===', fh)
        for _, r in best.iterrows():
            log(f'  {r["stage"]:<10}  {r["strategy"]:<10}  {r["model"]:<13}  R²={r["R2"]:.3f}', fh)
        log(f'\nMean R²: {best["R2"].mean():.3f}', fh)
    log(f'\nTotal wall time: {(time.time()-t0)/60:.1f} min', fh)
    fh.close()


if __name__ == '__main__':
    main()
