"""Phase E — full re-train of all 5 models × 8 stages × 3 strategies on
v2 (growing-season) features, with per-stage checkpointing.

Resumable: if killed mid-run, re-running picks up from the last
saved per-stage results CSV. Each stage writes its results
incrementally to the multi-stage results parquet.

Outputs:
    multi_stage_models_a6_gs.parquet  (all model × strategy × stage rows)
    multi_stage_best_a6_gs.csv         (best per stage)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.config import CFG, REPO_ROOT
_WORK = REPO_ROOT / CFG.paths.work_dir
_PHENO = str(REPO_ROOT / CFG.paths.phenology_matched)

from pathlib import Path
import sys
import time
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
import warnings
warnings.filterwarnings('ignore')

EXT = _WORK
TRAIN_FEAT = EXT / 'features_v3_realsowing_train.parquet'
PHENO_PATH = _PHENO

OUT_DIR = EXT / "v3_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_RESULTS = OUT_DIR / 'multi_stage_models_a6_gs.parquet'
OUT_BEST    = OUT_DIR / 'multi_stage_best_a6_gs.csv'
OUT_PROGRESS = OUT_DIR / '_progress.txt'  # resume marker

# ── Stage-label mapping (matches scripts/03_modeling/01_multi_stage_ml.ipynb)
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
META_FIXED = ['field_id','year','flag_true_doy','n_obs','sowing_doy_used']
REDUND = ['GDD_M2_at_SOS','VD_at_SOS','emergence_doy',
          'VD_from_emergence_at_SOS','fV_from_emergence_at_SOS',
          'days_emergence_to_SOS']
WE_OUTPUTS = ['WE_emergence_doy','WE_tillering_doy','WE_jointing_doy','WE_flag_leaf_doy',
              'WE_boot_doy','WE_heading_doy','WE_anthesis_doy','WE_maturity_doy']
WINDOWED_SUFFIXES = ('_gf','_pa','_pa_late')
WINDOWED_PREFIXES = ('heat_days_','hot_days_','frost_days_')
EARLY_STAGES_NO_WIN = {'emergence','tillering','jointing'}


def is_windowed(c):
    return c.endswith(WINDOWED_SUFFIXES) or c.startswith(WINDOWED_PREFIXES)


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


LINEAR_MODELS = {'ElasticNet', 'Ridge'}
K_GRID = (20, 40, 60, 80, None)


def loyo_predict(df, feat_cols, target, factory, model_name=''):
    """LOYO CV. For linear models, tunes SelectKBest k via inner CV."""
    df2 = df.dropna(subset=[target]).copy()

    # Per-stage sanity filter on targets — drop physiologically implausible DOS
    # (e.g., maturity labels at DOS 7 are mistakes; mass clip via IQR).
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
            best_pipe, best_score = None, -np.inf
            for k in K_GRID:
                steps = [('imp', SimpleImputer(strategy='median')),
                         ('sc',  StandardScaler())]
                if k is not None:
                    steps.append(('sel', SelectKBest(f_regression, k=min(k, len(feat_cols)))))
                steps.append(('m', factory()))
                pipe = Pipeline(steps)
                # Inner-fold-like score: split train into earlier (train) + last year (val)
                inner_yrs = sorted(tr['year'].unique())
                if len(inner_yrs) < 2:
                    pipe.fit(tr[feat_cols], tr[target])
                    score = 0
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
                    best_score = score
                    best_k = k
            # Refit best k on full train
            steps = [('imp', SimpleImputer(strategy='median')),
                     ('sc',  StandardScaler())]
            if best_k is not None:
                steps.append(('sel', SelectKBest(f_regression, k=min(best_k, len(feat_cols)))))
            steps.append(('m', factory()))
            pipe = Pipeline(steps)
        else:
            steps = [('imp', SimpleImputer(strategy='median')),
                     ('sc',  StandardScaler())]
            steps.append(('m', factory()))
            pipe = Pipeline(steps)
        pipe.fit(tr[feat_cols], tr[target])
        pred = pipe.predict(te[feat_cols])
        if pred.ndim > 1:
            pred = pred.ravel()
        pred_all.extend(pred); true_all.extend(te[target].values)
    return np.array(pred_all), np.array(true_all)


def metrics(y_true, y_pred):
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / denom if denom > 0 else 0
    w10 = float(np.mean(np.abs(y_true - y_pred) <= 10) * 100)
    return float(r2), rmse, w10


def boot_ci(y_true, y_pred, n_iter=200, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    r2s = []
    for _ in range(n_iter):
        idx = rng.choice(n, n, replace=True)
        yt, yp = y_true[idx], y_pred[idx]
        denom = np.sum((yt - yt.mean()) ** 2)
        r2 = 1 - np.sum((yt - yp) ** 2) / denom if denom > 0 else 0
        r2s.append(r2)
    return float(np.percentile(r2s, 2.5)), float(np.percentile(r2s, 97.5))


def model_factories():
    return {
        'ElasticNet':   lambda: ElasticNetCV(l1_ratio=[.1,.3,.5,.7,.9,.95,1.0],
                                             n_alphas=20, max_iter=20000, cv=5, n_jobs=1),
        'Ridge':        lambda: RidgeCV(alphas=np.logspace(-3, 3, 30), cv=5),
        'RandomForest': lambda: RandomForestRegressor(n_estimators=200, n_jobs=1, random_state=42),
        'XGBoost':      lambda: XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                              n_jobs=1, random_state=42, verbosity=0),
        'LightGBM':     lambda: LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                               n_jobs=1, random_state=42, verbose=-1),
    }


def main():
    t0 = time.time()
    print('=== Loading training features (v2, growing season) ===')
    feat = pd.read_parquet(TRAIN_FEAT)
    feat['field_id'] = feat['field_id'].astype(str)
    feat['year'] = feat['year'].astype(int)
    if 'state' in feat.columns:
        feat = feat.drop(columns=['state'])
    print(f'Features shape: {feat.shape}')

    pheno = pd.read_parquet(PHENO_PATH)
    targets = build_targets(pheno)
    feat = feat.merge(targets, on=['field_id','year'], how='left')

    # Feature column groups
    target_cols = [f'{s}_dos_obs' for s in STAGE_MAP]
    META = META_FIXED + target_cols
    ndre = [c for c in feat.columns if c.startswith('NDRE')]
    all_feat_cols = [c for c in feat.columns
                     if c not in META and c not in ndre and c not in REDUND
                     and pd.api.types.is_numeric_dtype(feat[c])]
    ml_only_cols = [c for c in all_feat_cols if c not in WE_OUTPUTS]
    windowed_cols = [c for c in feat.columns if is_windowed(c) and c not in META]

    def stage_cols(stage, include_wes=True):
        base = all_feat_cols if include_wes else ml_only_cols
        if stage in EARLY_STAGES_NO_WIN:
            return [c for c in base if c not in windowed_cols]
        return base

    # Resume from checkpoint
    done = set()
    if OUT_RESULTS.exists():
        prev = pd.read_parquet(OUT_RESULTS)
        done = set(zip(prev['stage'], prev['strategy'], prev['model']))
        print(f'Resuming — {len(done)} (stage, strategy, model) tuples done')
    rows = list(prev.to_dict('records')) if OUT_RESULTS.exists() else []

    factories = model_factories()
    strategies = [
        ('B_ML-only', False),  # ML features only (no WES outputs)
        ('C_Hybrid',  True),   # WES + ML features (the production strategy)
    ]
    # NOTE: Strategy A (WES alone) skipped here for time — it's a direct
    #       passthrough of WES outputs and can be added later if needed.

    total_combos = len(STAGE_MAP) * len(strategies) * len(factories)
    print(f'\n=== Training {total_combos} combinations ===')

    counter = 0
    for stage in STAGE_MAP:
        target = f'{stage}_dos_obs'
        if target not in feat.columns:
            print(f'  [{stage}] no labels, skipping')
            continue
        n_labels = feat[target].notna().sum()
        if n_labels < 30:
            print(f'  [{stage}] only {n_labels} labels, skipping')
            continue

        for strat_name, include_wes in strategies:
            cols = stage_cols(stage, include_wes=include_wes)
            for model_name, fac in factories.items():
                counter += 1
                key = (stage, strat_name, model_name)
                if key in done:
                    print(f'  [{counter}/{total_combos}] {stage} | {strat_name} | {model_name}  (already done, skip)')
                    continue
                try:
                    t1 = time.time()
                    y_pred, y_true = loyo_predict(feat, cols, target, fac, model_name=model_name)
                    r2, rmse, w10 = metrics(y_true, y_pred)
                    r2_lo, r2_hi = boot_ci(y_true, y_pred)
                    elapsed = time.time() - t1
                    rows.append({
                        'stage': stage, 'strategy': strat_name, 'model': model_name,
                        'n': len(y_true), 'R2': r2, 'R2_lo': r2_lo, 'R2_hi': r2_hi,
                        'RMSE': rmse, 'w10': w10, 'fit_time_s': elapsed,
                    })
                    print(f'  [{counter}/{total_combos}] {stage} | {strat_name} | {model_name}: '
                          f'R²={r2:.3f} CI[{r2_lo:.3f}, {r2_hi:.3f}]  RMSE={rmse:.1f}d  '
                          f'(n={len(y_true)}, {elapsed:.1f}s)')
                    # Persist after every model fit (resumable)
                    pd.DataFrame(rows).to_parquet(OUT_RESULTS, index=False)
                except Exception as e:
                    print(f'  [{counter}/{total_combos}] {stage} | {strat_name} | {model_name}: FAILED — {e}')

        # Per-stage summary print
        stage_rows = [r for r in rows if r['stage'] == stage]
        if stage_rows:
            best = max(stage_rows, key=lambda r: r['R2'])
            print(f'  → BEST for {stage}: {best["strategy"]} {best["model"]} '
                  f'R²={best["R2"]:.3f}')

    # Build "best per stage" CSV
    if rows:
        df = pd.DataFrame(rows)
        best_per = df.loc[df.groupby('stage')['R2'].idxmax()]
        best_per.to_csv(OUT_BEST, index=False)
        print(f'\n→ {OUT_RESULTS}: {len(df)} rows')
        print(f'→ {OUT_BEST}')
        print(f'\nBest model per stage:')
        print(best_per[['stage','strategy','model','n','R2','R2_lo','R2_hi','RMSE','w10']].to_string(index=False))

    print(f'\nWall time: {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
