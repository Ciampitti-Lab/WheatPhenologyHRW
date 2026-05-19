"""Sowing-date fallback sensitivity analysis (reviewer de-risking, SI).
[NOTE] anthesis here uses the old ElasticNet best; the manuscript value
is recomputed with the adopted FT-Transformer in
scripts/05_analysis/06_anthesis_ft_ablation.py. flag_leaf/boot/heading
(tree/linear best, unchanged) keep the numbers reported here.

Question a JAG reviewer will ask: 85.3% of training field-years use the
state-median sowing-date fallback to anchor the Wang--Engel--Streck (WES)
simulator. If those fallback sowing dates are perturbed with random noise,
how much of the physiology-informed gain (Delta R^2 = R2[hybrid] - R2[ML-only])
on the reproductive stages survives?

Design
------
* Identify fallback vs observed field-years from sowing_lookup.parquet
  (source == 'state_median' -> fallback; else observed, left untouched).
* For sigma in {7, 14, 21} days and N replicate seeds, add round(N(0,sigma))
  to the fallback sowing DOY only, re-run the *published* simulate_wes()
  for every perturbed field-year, and overwrite its WE_<stage>_doy columns.
  Observed-sowing rows keep their original WE values.
* Re-run the exact Phase-E LOYO procedure (copied verbatim from
  39_phase_e_v3.py) for the C_Hybrid strategy using each reproductive
  stage's best model. B_ML-only does not use WE features, so its R^2 is
  invariant under perturbation and is computed once as the baseline.
* Report R2[hybrid] and the surviving gain at each sigma vs the
  unperturbed control.

Outputs
-------
  v3_results/sowing_sensitivity.csv          (per stage x sigma x rep)
  v3_results/sowing_sensitivity_summary.csv  (mean +/- sd, % gain retained)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.config import CFG, REPO_ROOT
_WORK = REPO_ROOT / CFG.paths.work_dir
_PHENO = str(REPO_ROOT / CFG.paths.phenology_matched)

from pathlib import Path
import time
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

import sys
from scripts.utils.thermal import simulate_wes

EXT = _WORK
TRAIN_FEAT = EXT / 'features_v3_realsowing_train.parquet'
SOW_LOOKUP = EXT / 'sowing_lookup.parquet'
DAYMET     = EXT / 'daymet_full_2013_2024.parquet'
PHENO_PATH = _PHENO
OUT_DIR = EXT / 'v3_results'
OUT_DIR.mkdir(parents=True, exist_ok=True)

SIGMAS = [7, 14, 21, 28]      # days, std of Gaussian perturbation
N_REPS = 3
# Best (strategy, model) per reproductive stage, from multi_stage_best_a6_gs
BEST_MODEL = {'flag_leaf': 'XGBoost', 'boot': 'LightGBM',
              'heading': 'ElasticNet', 'anthesis': 'ElasticNet'}
REPRO = list(BEST_MODEL)

# ─── Phase-E constants (verbatim from 39_phase_e_v3.py) ───
STAGE_MAP = {
    'flag_leaf':  ['Flag Leaf Emerging', 'Flag Leaf Emerged'],
    'boot':       ['Early Boot', 'Boot'],
    'heading':    ['Head Emerging', 'Heading', 'Complete Heading'],
    'anthesis':   ['Early Bloom', 'Bloom'],
}
META_FIXED = ['field_id', 'year', 'flag_true_doy', 'n_obs', 'sowing_doy_used']
REDUND = ['GDD_M2_at_SOS', 'VD_at_SOS', 'emergence_doy',
          'VD_from_emergence_at_SOS', 'fV_from_emergence_at_SOS',
          'days_emergence_to_SOS']
WE_OUTPUTS = ['WE_emergence_doy', 'WE_tillering_doy', 'WE_jointing_doy',
              'WE_flag_leaf_doy', 'WE_boot_doy', 'WE_heading_doy',
              'WE_anthesis_doy', 'WE_maturity_doy']
LINEAR_MODELS = {'ElasticNet', 'Ridge'}
K_GRID = (20, 40, 60, 80, None)


def model_factory(name):
    return {
        'ElasticNet': lambda: ElasticNetCV(l1_ratio=[.1, .3, .5, .7, .9, .95, 1.0],
                                           n_alphas=20, max_iter=20000, cv=5, n_jobs=-1),
        'Ridge':      lambda: RidgeCV(alphas=np.logspace(-3, 3, 30), cv=5),
        'RandomForest': lambda: RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42),
        'XGBoost':    lambda: XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                           n_jobs=-1, random_state=42, verbosity=0),
        'LightGBM':   lambda: LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                            n_jobs=-1, random_state=42, verbose=-1),
    }[name]


def build_targets(pheno):
    out = None
    for stage, labels in STAGE_MAP.items():
        s = pheno[pheno['growth_stage'].isin(labels)].copy()
        s['harvest_year'] = s['growing_season'].str.split('-').str[1].astype(int)
        s['field_id'] = s['FIELDID'].astype(str)
        e = (s.groupby(['field_id', 'harvest_year'])['dos'].min().reset_index()
              .rename(columns={'harvest_year': 'year', 'dos': f'{stage}_dos_obs'}))
        out = e if out is None else out.merge(e, on=['field_id', 'year'], how='outer')
    return out


def loyo_predict(df, feat_cols, target, factory, model_name):
    """Verbatim LOYO from 39_phase_e_v3.py (linear models tune SelectKBest k)."""
    df2 = df.dropna(subset=[target]).copy()
    q1, q99 = df2[target].quantile([0.01, 0.99])
    df2 = df2[(df2[target] >= q1) & (df2[target] <= q99)].copy()
    is_linear = model_name in LINEAR_MODELS
    pred_all, true_all = [], []
    for yr in sorted(df2['year'].unique()):
        tr = df2[df2['year'] != yr]
        te = df2[df2['year'] == yr]
        if len(tr) < 50 or len(te) < 5:
            continue
        if is_linear:
            best_k, best_score = None, -np.inf
            for k in K_GRID:
                steps = [('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]
                if k is not None:
                    steps.append(('sel', SelectKBest(f_regression, k=min(k, len(feat_cols)))))
                steps.append(('m', factory()))
                pipe = Pipeline(steps)
                inner_yrs = sorted(tr['year'].unique())
                if len(inner_yrs) < 2:
                    pipe.fit(tr[feat_cols], tr[target]); score = 0
                else:
                    val_y = inner_yrs[-1]
                    itr = tr[tr['year'] != val_y]; iva = tr[tr['year'] == val_y]
                    pipe.fit(itr[feat_cols], itr[target])
                    pv = pipe.predict(iva[feat_cols]); yv = iva[target].values
                    den = np.sum((yv - yv.mean()) ** 2)
                    score = 1 - np.sum((yv - pv) ** 2) / den if den > 0 else 0
                if score > best_score:
                    best_score, best_k = score, k
            steps = [('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]
            if best_k is not None:
                steps.append(('sel', SelectKBest(f_regression, k=min(best_k, len(feat_cols)))))
            steps.append(('m', factory()))
            pipe = Pipeline(steps)
        else:
            pipe = Pipeline([('imp', SimpleImputer(strategy='median')),
                             ('sc', StandardScaler()), ('m', factory())])
        pipe.fit(tr[feat_cols], tr[target])
        pred = pipe.predict(te[feat_cols])
        if pred.ndim > 1:
            pred = pred.ravel()
        pred_all.extend(pred); true_all.extend(te[target].values)
    return np.array(pred_all), np.array(true_all)


def r2_of(y, p):
    den = np.sum((y - y.mean()) ** 2)
    return float(1 - np.sum((y - p) ** 2) / den) if den > 0 else 0.0


def main():
    t0 = time.time()
    print('=== Loading inputs ===', flush=True)
    feat = pd.read_parquet(TRAIN_FEAT)
    feat['field_id'] = feat['field_id'].astype(str)
    feat['year'] = feat['year'].astype(int)
    if 'state' in feat.columns:
        feat = feat.drop(columns=['state'])
    pheno = pd.read_parquet(PHENO_PATH)
    feat = feat.merge(build_targets(pheno), on=['field_id', 'year'], how='left')

    sl = pd.read_parquet(SOW_LOOKUP)
    sl['field_id'] = sl['field_id'].astype(str)
    sl = sl.rename(columns={'harvest_year': 'year'})
    feat = feat.merge(sl[['field_id', 'year', 'source', 'sowing_doy_used']]
                      .rename(columns={'sowing_doy_used': 'sow_base'}),
                      on=['field_id', 'year'], how='left')
    is_fb = (feat['source'] == 'state_median').values
    print(f'Training field-years: {len(feat)}  fallback: {is_fb.sum()} '
          f'({100*is_fb.mean():.1f}%)  observed: {(~is_fb).sum()}', flush=True)

    # Pre-build per-field-year weather frames for simulate_wes (date, doy, T_mean)
    print('=== Indexing daymet ===', flush=True)
    dm = pd.read_parquet(DAYMET, columns=['FIELDID', 'date', 'Tmin', 'Tmax', 'harvest_year'])
    dm['FIELDID'] = dm['FIELDID'].astype(str)
    dm['date'] = pd.to_datetime(dm['date'])
    dm['doy'] = dm['date'].dt.dayofyear
    dm['T_mean'] = ((dm['Tmin'] + dm['Tmax']) / 2.0).astype('float32')
    wx_by = {k: g.sort_values('date')[['date', 'doy', 'T_mean']].reset_index(drop=True)
             for k, g in dm.groupby(['FIELDID', 'harvest_year'])}
    del dm
    lat_by = dict(zip(zip(feat['field_id'], feat['year']), feat['latitude']))

    # Feature column groups (verbatim Phase-E logic)
    target_cols = [f'{s}_dos_obs' for s in STAGE_MAP]
    META = META_FIXED + target_cols + ['source', 'sow_base']
    ndre = [c for c in feat.columns if c.startswith('NDRE')]
    all_cols = [c for c in feat.columns if c not in META and c not in ndre
                and c not in REDUND and pd.api.types.is_numeric_dtype(feat[c])]
    hybrid_cols = all_cols
    mlonly_cols = [c for c in all_cols if c not in WE_OUTPUTS]

    # ── Baseline: ML-only (invariant) + control hybrid (unperturbed WE) ──
    rows = []
    base_gain = {}
    for st in REPRO:
        tgt = f'{st}_dos_obs'
        mdl = BEST_MODEL[st]
        yp, yt = loyo_predict(feat, mlonly_cols, tgt, model_factory(mdl), mdl)
        r2_ml = r2_of(yt, yp)
        yp, yt = loyo_predict(feat, hybrid_cols, tgt, model_factory(mdl), mdl)
        r2_hy0 = r2_of(yt, yp)
        base_gain[st] = (r2_ml, r2_hy0, r2_hy0 - r2_ml)
        rows.append(dict(stage=st, model=mdl, sigma=0, rep=0,
                         r2_mlonly=r2_ml, r2_hybrid=r2_hy0, gain=r2_hy0 - r2_ml))
        print(f'[control] {st:9s} ML={r2_ml:.3f}  hybrid={r2_hy0:.3f}  '
              f'gain={r2_hy0 - r2_ml:+.3f}', flush=True)

    fb_idx = np.where(is_fb)[0]
    fb_keys = list(zip(feat['field_id'].values[fb_idx],
                       feat['year'].values[fb_idx],
                       feat['sow_base'].values[fb_idx]))

    for sigma in SIGMAS:
        for rep in range(1, N_REPS + 1):
            rng = np.random.RandomState(1000 * sigma + rep)
            pert = feat.copy()
            we_new = {c: pert[c].values.copy() for c in WE_OUTPUTS}
            noise = np.round(rng.normal(0, sigma, len(fb_idx))).astype(int)
            n_ok = 0
            for j, (fid, hy, base) in enumerate(fb_keys):
                wx = wx_by.get((fid, hy))
                if wx is None or len(wx) == 0:
                    continue
                new_sow = int(np.clip(base + noise[j], 220, 330))
                we = simulate_wes(wx, lat=float(lat_by.get((fid, hy), 38.0)),
                                  sowing_doy=new_sow, sowing_year=int(hy),
                                  return_dos=False)
                ridx = fb_idx[j]
                for c in WE_OUTPUTS:
                    we_new[c][ridx] = we.get(c, np.nan)
                n_ok += 1
            for c in WE_OUTPUTS:
                pert[c] = we_new[c]
            for st in REPRO:
                tgt = f'{st}_dos_obs'
                mdl = BEST_MODEL[st]
                yp, yt = loyo_predict(pert, hybrid_cols, tgt, model_factory(mdl), mdl)
                r2_hy = r2_of(yt, yp)
                r2_ml = base_gain[st][0]
                rows.append(dict(stage=st, model=mdl, sigma=sigma, rep=rep,
                                 r2_mlonly=r2_ml, r2_hybrid=r2_hy, gain=r2_hy - r2_ml))
                print(f'[s={sigma:2d} r{rep}] {st:9s} resim={n_ok} '
                      f'hybrid={r2_hy:.3f} gain={r2_hy - r2_ml:+.3f}', flush=True)
            pd.DataFrame(rows).to_csv(OUT_DIR / 'sowing_sensitivity.csv', index=False)

    df = pd.DataFrame(rows)
    summ = []
    for st in REPRO:
        g0 = base_gain[st][2]
        for sigma in SIGMAS:
            sub = df[(df.stage == st) & (df.sigma == sigma)]
            gm, gs = sub.gain.mean(), sub.gain.std()
            summ.append(dict(stage=st, sigma=sigma, gain_control=round(g0, 4),
                             gain_mean=round(gm, 4), gain_sd=round(gs, 4),
                             pct_retained=round(100 * gm / g0, 1) if g0 != 0 else np.nan))
    sdf = pd.DataFrame(summ)
    sdf.to_csv(OUT_DIR / 'sowing_sensitivity_summary.csv', index=False)
    print('\n=== SUMMARY (gain = R2 hybrid - R2 ML-only) ===', flush=True)
    print(sdf.to_string(index=False), flush=True)
    print(f'\nDone in {(time.time() - t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
