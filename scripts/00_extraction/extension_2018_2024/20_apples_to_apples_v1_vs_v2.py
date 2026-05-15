"""Apples-to-apples comparison: train both v1 and v2 models on the
SAME 1,830 (field_id, harvest_year) tuples that survived the v1
flag-leaf filter and are also in v2 (2014-2017).

This isolates the methodology change (calendar-year vs growing-season
features) from the sample-size change.
"""
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

V1_FEAT = '/depot/ciampitti/data/WheatPhenologyHRW/data/processed/features/features_a6.parquet'
V2_FEAT = '/depot/ciampitti/data/WheatPhenologyHRW/data/raw/satellite/extension_2018_2024/features_gs_train_2014_2017.parquet'
PHENO   = '/depot/ciampitti/data/WheatPhenologyHRW/data/processed/buffer_300m/wheat_hrw_phenology_buffer_matched.parquet'

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


def loyo_with_select(df, feat_cols, target):
    """LOYO with ElasticNet + SelectKBest tuned over k."""
    df2 = df.dropna(subset=[target]).copy()
    q1, q99 = df2[target].quantile([0.01, 0.99])
    df2 = df2[(df2[target] >= q1) & (df2[target] <= q99)]
    K_GRID = (20, 40, 60, 80, None)
    pred_all, true_all = [], []
    for yr in sorted(df2['year'].unique()):
        tr = df2[df2['year'] != yr]
        te = df2[df2['year'] == yr]
        if len(tr) < 50 or len(te) < 5:
            continue
        # Inner-CV tune k via held-out year
        inner_yrs = sorted(tr['year'].unique())
        best_k, best_score = None, -np.inf
        for k in K_GRID:
            steps = [('imp', SimpleImputer(strategy='median')),
                     ('sc',  StandardScaler())]
            if k is not None:
                steps.append(('sel', SelectKBest(f_regression, k=min(k, len(feat_cols)))))
            steps.append(('m', ElasticNetCV(l1_ratio=[.1,.3,.5,.7,.9,.95,1.0],
                                            n_alphas=20, max_iter=10000, cv=5, n_jobs=1)))
            pipe = Pipeline(steps)
            if len(inner_yrs) < 2:
                pipe.fit(tr[feat_cols], tr[target]); score = 0
            else:
                val_y = inner_yrs[-1]
                inner_tr = tr[tr['year'] != val_y]
                inner_va = tr[tr['year'] == val_y]
                pipe.fit(inner_tr[feat_cols], inner_tr[target])
                pred_inner = pipe.predict(inner_va[feat_cols])
                yv = inner_va[target].values
                denom = np.sum((yv - yv.mean()) ** 2)
                score = 1 - np.sum((yv - pred_inner) ** 2) / denom if denom > 0 else 0
            if score > best_score:
                best_score, best_k = score, k
        steps = [('imp', SimpleImputer(strategy='median')),
                 ('sc',  StandardScaler())]
        if best_k is not None:
            steps.append(('sel', SelectKBest(f_regression, k=min(best_k, len(feat_cols)))))
        steps.append(('m', ElasticNetCV(l1_ratio=[.1,.3,.5,.7,.9,.95,1.0],
                                        n_alphas=20, max_iter=10000, cv=5, n_jobs=1)))
        pipe = Pipeline(steps)
        pipe.fit(tr[feat_cols], tr[target])
        pred = pipe.predict(te[feat_cols])
        if pred.ndim > 1: pred = pred.ravel()
        pred_all.extend(pred); true_all.extend(te[target].values)
    return np.array(pred_all), np.array(true_all)


def metrics(y_true, y_pred):
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / denom if denom > 0 else 0
    return float(r2), rmse


def main():
    pheno = pd.read_parquet(PHENO)
    targets = build_targets(pheno)

    v1 = pd.read_parquet(V1_FEAT)
    v2 = pd.read_parquet(V2_FEAT)
    v1['field_id'] = v1['field_id'].astype(str); v1['year'] = v1['year'].astype(int)
    v2['field_id'] = v2['field_id'].astype(str); v2['year'] = v2['year'].astype(int)
    if 'state' in v1.columns: v1 = v1.drop(columns=['state'])
    if 'state' in v2.columns: v2 = v2.drop(columns=['state'])

    # Get overlap pairs (fields in both, years 2014-2017)
    v1_pairs = set(zip(v1['field_id'], v1['year']))
    v2_pairs = set(zip(v2['field_id'], v2['year']))
    overlap = v1_pairs & v2_pairs
    print(f'Overlap field-years: {len(overlap):,}')

    # Filter both to overlap
    ov_df = pd.DataFrame(list(overlap), columns=['field_id','year'])
    v1_o = v1.merge(ov_df, on=['field_id','year']).merge(targets, on=['field_id','year'], how='left')
    v2_o = v2.merge(ov_df, on=['field_id','year']).merge(targets, on=['field_id','year'], how='left')
    print(f'v1 overlap: {len(v1_o):,}  v2 overlap: {len(v2_o):,}')

    # Feature columns (drop ndre, redundant, meta)
    META = ['field_id','year','flag_true_doy','n_obs','sowing_doy_used'] + \
           [f'{s}_dos_obs' for s in STAGE_MAP]
    REDUND = ['GDD_M2_at_SOS','VD_at_SOS','emergence_doy',
              'VD_from_emergence_at_SOS','fV_from_emergence_at_SOS','days_emergence_to_SOS']
    def cols(df):
        ndre = [c for c in df.columns if c.startswith('NDRE')]
        return [c for c in df.columns if c not in META and c not in REDUND
                and c not in ndre and pd.api.types.is_numeric_dtype(df[c])]

    feat_cols_v1 = cols(v1_o); feat_cols_v2 = cols(v2_o)
    print(f'v1 features: {len(feat_cols_v1)}, v2 features: {len(feat_cols_v2)}')

    rows = []
    for stage in STAGE_MAP:
        target = f'{stage}_dos_obs'
        if target not in v1_o.columns or target not in v2_o.columns:
            continue
        n1 = v1_o[target].notna().sum(); n2 = v2_o[target].notna().sum()
        print(f'\n=== {stage} (n_v1={n1}, n_v2={n2}) ===')
        if n1 < 50 or n2 < 50:
            print(f'  skipping (too few labels)')
            continue
        try:
            p1, t1 = loyo_with_select(v1_o, feat_cols_v1, target)
            r1, rmse1 = metrics(t1, p1)
            print(f'  v1: R²={r1:.3f}  RMSE={rmse1:.1f}')
        except Exception as e:
            print(f'  v1 FAILED: {e}'); r1, rmse1 = np.nan, np.nan
        try:
            p2, t2 = loyo_with_select(v2_o, feat_cols_v2, target)
            r2, rmse2 = metrics(t2, p2)
            print(f'  v2: R²={r2:.3f}  RMSE={rmse2:.1f}')
        except Exception as e:
            print(f'  v2 FAILED: {e}'); r2, rmse2 = np.nan, np.nan
        rows.append({'stage': stage, 'n': len(t1) if not np.isnan(r1) else 0,
                     'R2_v1': r1, 'R2_v2': r2, 'RMSE_v1': rmse1, 'RMSE_v2': rmse2})

    df = pd.DataFrame(rows)
    df['ΔR²'] = df['R2_v2'] - df['R2_v1']
    print(f'\n=== APPLES-TO-APPLES (same field-years, ElasticNet+SelectKBest) ===')
    print(df.round(3).to_string(index=False))
    print(f'\nMean R²: v1 {df["R2_v1"].mean():.3f}  →  v2 {df["R2_v2"].mean():.3f}  '
          f'(Δ={df["ΔR²"].mean():+.3f})')


if __name__ == '__main__':
    main()
