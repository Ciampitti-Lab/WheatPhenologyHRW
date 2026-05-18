"""Tillering target-definition robustness (Supplement S7).

Does the modest tillering skill come from the earliest-label target
collapsing the whole tillering phase into one number? We rebuild the
target as a consistent developmental anchor (each count-bearing
sub-label mapped to its Zadoks tiller code; per field-year the
observation closest to mid-tillering, Z25, is taken; the ambiguous
generic "Tillering" label is dropped) and re-run the identical LOYO
pipeline. Only the target differs between the two arms, so the delta
is what matters. Read-only; prints a report.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.config import CFG, REPO_ROOT

import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
warnings.filterwarnings('ignore')

WORK = REPO_ROOT / CFG.paths.work_dir
TRAIN_FEAT = WORK / 'features_v3_realsowing_train.parquet'
PHENO = str(REPO_ROOT / CFG.paths.phenology_matched)

VOCAB = ['Begin Tillering', 'Tillering', '1-2 Tiller', '2-4 Tiller',
         '4-6 Tiller', '6-8 Tiller', '8+ Tiller', 'Full Tillering',
         'End Tillering']
# Zadoks tiller code per count-bearing sub-label; the generic
# "Tillering" is ambiguous and excluded from the developmental target.
ZAD = {'Begin Tillering': 21, '1-2 Tiller': 21.5, '2-4 Tiller': 23,
       '4-6 Tiller': 25, '6-8 Tiller': 27, '8+ Tiller': 29,
       'Full Tillering': 29, 'End Tillering': 30}
ZMID = 25.0  # mid-tillering developmental code

META_FIXED = ['field_id', 'year', 'flag_true_doy', 'n_obs', 'sowing_doy_used']
REDUND = ['GDD_M2_at_SOS', 'VD_at_SOS', 'emergence_doy',
          'VD_from_emergence_at_SOS', 'fV_from_emergence_at_SOS',
          'days_emergence_to_SOS']
WE = ['WE_emergence_doy', 'WE_tillering_doy', 'WE_jointing_doy',
      'WE_flag_leaf_doy', 'WE_boot_doy', 'WE_heading_doy',
      'WE_anthesis_doy', 'WE_maturity_doy']
K_GRID = (20, 40, 60, 80, None)
LINEAR = {'ElasticNet', 'Ridge'}


def is_windowed(c):
    return (c.endswith(('_gf', '_pa', '_pa_late'))
            or c.startswith(('heat_days_', 'hot_days_', 'frost_days_')))


def factories():
    return {
        'ElasticNet': lambda: ElasticNetCV(l1_ratio=[.1, .3, .5, .7, .9, .95, 1.0],
                                           n_alphas=20, max_iter=20000, cv=5, n_jobs=1),
        'Ridge': lambda: RidgeCV(alphas=np.logspace(-3, 3, 30), cv=5),
        'RandomForest': lambda: RandomForestRegressor(n_estimators=200, n_jobs=1,
                                                      random_state=42),
        'XGBoost': lambda: XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                        n_jobs=1, random_state=42, verbosity=0),
        'LightGBM': lambda: LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                          n_jobs=1, random_state=42, verbose=-1),
    }


def loyo(df, feat, tg, factory, mname):
    """Leave-one-year-out, mirroring scripts/03_modeling/01_phase_e_loyo.py."""
    d = df.dropna(subset=[tg]).copy()
    q1, q99 = d[tg].quantile([.01, .99])
    d = d[(d[tg] >= q1) & (d[tg] <= q99)].copy()
    is_lin = mname in LINEAR
    pred, true = [], []
    for yr in sorted(d['year'].unique()):
        tr, te = d[d['year'] != yr], d[d['year'] == yr]
        if len(tr) < 50 or len(te) < 5:
            continue
        if is_lin:
            best_k, best = None, -np.inf
            for k in K_GRID:
                st = [('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]
                if k is not None:
                    st.append(('sel', SelectKBest(f_regression, k=min(k, len(feat)))))
                st.append(('m', factory()))
                pp = Pipeline(st)
                iy = sorted(tr['year'].unique())
                if len(iy) < 2:
                    pp.fit(tr[feat], tr[tg])
                    sc = 0
                else:
                    vy = iy[-1]
                    it, iv = tr[tr['year'] != vy], tr[tr['year'] == vy]
                    pp.fit(it[feat], it[tg])
                    pr = pp.predict(iv[feat])
                    yv = iv[tg].values
                    dn = np.sum((yv - yv.mean()) ** 2)
                    sc = 1 - np.sum((yv - pr) ** 2) / dn if dn > 0 else 0
                if sc > best:
                    best, best_k = sc, k
            st = [('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]
            if best_k is not None:
                st.append(('sel', SelectKBest(f_regression, k=min(best_k, len(feat)))))
            st.append(('m', factory()))
            pp = Pipeline(st)
        else:
            pp = Pipeline([('imp', SimpleImputer(strategy='median')),
                           ('sc', StandardScaler()), ('m', factory())])
        pp.fit(tr[feat], tr[tg])
        pred.extend(np.ravel(pp.predict(te[feat])))
        true.extend(te[tg].values)
    pred, true = np.array(pred), np.array(true)
    dn = np.sum((true - true.mean()) ** 2)
    r2 = 1 - np.sum((true - pred) ** 2) / dn if dn > 0 else 0
    return r2, float(np.sqrt(np.mean((true - pred) ** 2))), len(true)


def main():
    ph = pd.read_parquet(PHENO)
    ph['year'] = ph['growing_season'].str.split('-').str[1].astype(int)
    ph['field_id'] = ph['FIELDID'].astype(str)
    t = ph[ph['growth_stage'].isin(VOCAB) & (ph['dos'] > 200)].copy()

    # Arm A: current target = earliest of any tillering label.
    A = (t.groupby(['field_id', 'year'])['dos'].min().reset_index()
           .rename(columns={'dos': 'tillering_dos_obs'}))

    # Arm B: Zadoks-midpoint = DOS of the obs closest to Z25.
    r = t[t['growth_stage'].isin(ZAD)].copy()
    r['z'] = r['growth_stage'].map(ZAD)
    r['d2'] = (r['z'] - ZMID).abs()
    r = r.sort_values(['field_id', 'year', 'd2', 'dos'])
    B = (r.groupby(['field_id', 'year']).first().reset_index()
          [['field_id', 'year', 'dos']]
          .rename(columns={'dos': 'tillering_dos_obs'}))

    fe = pd.read_parquet(TRAIN_FEAT)
    fe['field_id'] = fe['field_id'].astype(str)
    fe['year'] = fe['year'].astype(int)
    if 'state' in fe.columns:
        fe = fe.drop(columns=['state'])
    meta = META_FIXED + ['tillering_dos_obs']
    ndre = [c for c in fe.columns if c.startswith('NDRE')]
    allc = [c for c in fe.columns if c not in meta and c not in ndre
            and c not in REDUND and pd.api.types.is_numeric_dtype(fe[c])]
    mlc = [c for c in allc if c not in WE]
    winc = [c for c in fe.columns if is_windowed(c) and c not in meta]

    def cols(wes):                      # tillering is an early stage
        base = allc if wes else mlc
        return [c for c in base if c not in winc]

    for name, tg in [('A_current_earliest', A), ('B_zadoks_midpoint', B)]:
        df = fe.merge(tg, on=['field_id', 'year'], how='left')
        nlab = df['tillering_dos_obs'].notna().sum()
        best = None
        for strat, wes in [('ML-only', False), ('Hybrid', True)]:
            fc = cols(wes)
            for mn, fa in factories().items():
                r2, rmse, n = loyo(df, fc, 'tillering_dos_obs', fa, mn)
                if best is None or r2 > best[0]:
                    best = (r2, rmse, n, strat, mn)
        print(f'{name}: n_labelled={nlab}  best R2={best[0]:.3f}  '
              f'RMSE={best[1]:.1f} d  n={best[2]}  ({best[3]}/{best[4]})')


if __name__ == '__main__':
    main()
