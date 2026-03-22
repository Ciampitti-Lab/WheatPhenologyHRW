"""Fix the maturity stage by filtering implausible labels (DOS<280) and re-running
the full evaluation pipeline (Phase E LOYO, LOSO, inference 2018-2024, feature
importance, trend analysis) for maturity only. Updates the canonical artefact
files in-place so the downstream figures pick up corrected numbers automatically.

Implausible labels are observations of maturity stages ("Maturity",
"Harvest Ready", "Ready For Harvesting") with DOS<280 (~May 1). These are
prior-season harvest residue observations mis-attributed to the next growing
season; in HRW wheat, biological maturity is reached in late May to early July
(DOS >= 320 typically; we use 280 as a conservative cutoff).
"""

# --- repo-portable paths (no hardcoded cluster paths) --------------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from scripts.utils.config import CFG, REPO_ROOT
_WORK = REPO_ROOT / CFG.paths.work_dir
_PHENO = str(REPO_ROOT / CFG.paths.phenology_matched)
# ------------------------------------------------------------------------

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

EXT = _WORK
PHENO = _PHENO

TRAIN_FEAT = EXT / 'features_v3_realsowing_train.parquet'
EXT_FEAT_2018 = EXT / 'features_v3_realsowing_2018.parquet'
EXT_FEAT_19_24 = EXT / 'features_v3_realsowing_extension.parquet'

# Output artefacts to update (replace maturity rows)
PHASE_E       = EXT / 'v3_results' / 'multi_stage_models_a6_gs.parquet'
LOYO_PREDS    = EXT / 'v3_loyo_predictions.parquet'
LOSO_RES      = EXT / 'v3_loso_results.csv'
EXT_PREDS     = EXT / 'predictions_all_stages_2018_2024.parquet'
TRENDS        = EXT / 'v3_trends_per_stage_per_state.csv'
FEAT_IMP      = ROOT / 'docs' / 'figures' / 'feature_importance_per_stage.csv'

MATURITY_LABELS = ['Maturity', 'Harvest Ready', 'Ready For Harvesting']
MATURITY_DOS_MIN = 280   # filter prior-year residue observations
STATES = ['TX', 'OK', 'KS', 'NE', 'CO']
LINEAR = {'ElasticNet', 'Ridge'}


# ────────── feature setup (matches existing Phase E pipeline) ─────────
def feature_cols(feat_df, include_wes):
    META = ['field_id', 'year', 'flag_true_doy', 'n_obs', 'sowing_doy_used',
            'maturity_dos_obs']
    REDUND = ['GDD_M2_at_SOS', 'VD_at_SOS', 'emergence_doy',
              'VD_from_emergence_at_SOS', 'fV_from_emergence_at_SOS',
              'days_emergence_to_SOS']
    we_multi = ['WE_emergence_doy', 'WE_tillering_doy', 'WE_jointing_doy',
                'WE_flag_leaf_doy', 'WE_boot_doy', 'WE_heading_doy',
                'WE_anthesis_doy', 'WE_maturity_doy']
    ndre = [c for c in feat_df.columns if c.startswith('NDRE')]
    drop_set = set(META + REDUND + ndre + ['ph_top', 'state'])
    cols = [c for c in feat_df.columns
            if c not in drop_set and pd.api.types.is_numeric_dtype(feat_df[c])]
    if not include_wes:
        cols = [c for c in cols if c not in we_multi]
    return cols


def get_factory(name):
    return {
        'ElasticNet':   lambda: ElasticNetCV(l1_ratio=[.1, .3, .5, .7, .9, .95, 1.0],
                                              n_alphas=20, max_iter=20000, cv=5, n_jobs=-1),
        'Ridge':        lambda: RidgeCV(alphas=np.logspace(-3, 3, 30), cv=5),
        'RandomForest': lambda: RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42),
        'XGBoost':      lambda: XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                              n_jobs=-1, random_state=42, verbosity=0),
        'LightGBM':     lambda: LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                                n_jobs=-1, random_state=42, verbose=-1),
    }[name]


def build_pipeline(mdl, cols):
    steps = [('imp', SimpleImputer(strategy='median')),
             ('sc',  StandardScaler())]
    if mdl in LINEAR:
        steps.append(('sel', SelectKBest(f_regression, k=min(60, len(cols)))))
    steps.append(('m', get_factory(mdl)()))
    return Pipeline(steps)


# ────────── load + filter targets ─────────
def main():
    print('=== Loading data ===')
    pheno = pd.read_parquet(PHENO)
    feat = pd.read_parquet(TRAIN_FEAT)
    feat['field_id'] = feat['field_id'].astype(str)
    feat['year'] = feat['year'].astype(int)
    if 'state' in feat.columns:
        feat = feat.drop(columns=['state'])

    print('\n=== Re-building maturity target with DOS>=280 filter ===')
    m = pheno[pheno['growth_stage'].isin(MATURITY_LABELS)].copy()
    n_before = len(m)
    m = m[m['dos'] >= MATURITY_DOS_MIN]
    print(f'Maturity observations: {n_before} → {len(m)} after DOS>=280 filter')
    m['harvest_year'] = m['growing_season'].str.split('-').str[1].astype(int)
    m['field_id'] = m['FIELDID'].astype(str)
    targets = (m.groupby(['field_id', 'harvest_year'])['dos'].min().reset_index()
                 .rename(columns={'harvest_year': 'year', 'dos': 'maturity_dos_obs'}))
    feat = feat.merge(targets, on=['field_id', 'year'], how='left')

    train = feat.dropna(subset=['maturity_dos_obs']).copy()
    q1, q99 = train['maturity_dos_obs'].quantile([0.01, 0.99])
    train = train[(train['maturity_dos_obs'] >= q1) &
                  (train['maturity_dos_obs'] <= q99)]
    print(f'Training field-years for maturity: {len(train)}')

    # ────────── 1. Phase E LOYO across 5 models × 2 strategies ─────────
    print('\n=== Phase E LOYO (maturity only) ===')
    strategies = [('B_ML-only', False), ('C_Hybrid', True)]
    models = ['ElasticNet', 'Ridge', 'RandomForest', 'XGBoost', 'LightGBM']
    years = sorted(train['year'].unique())
    rows = []
    loyo_preds = {(strat, mdl): [] for strat, _ in strategies for mdl in models}

    for strat_name, include_wes in strategies:
        cols = feature_cols(feat, include_wes)
        for mdl in models:
            ys, yhats = [], []
            for hold_y in years:
                tr = train[train['year'] != hold_y]
                te = train[train['year'] == hold_y]
                if len(tr) < 30 or len(te) < 5:
                    continue
                pipe = build_pipeline(mdl, cols)
                pipe.fit(tr[cols], tr['maturity_dos_obs'])
                p = pipe.predict(te[cols])
                if p.ndim > 1:
                    p = p.ravel()
                ys.append(te['maturity_dos_obs'].values)
                yhats.append(p)
                loyo_preds[(strat_name, mdl)].append(
                    pd.DataFrame({'stage': 'maturity',
                                  'observed': te['maturity_dos_obs'].values,
                                  'predicted': p}))
            if not ys:
                continue
            y_arr = np.concatenate(ys)
            p_arr = np.concatenate(yhats)
            ss = np.sum((y_arr - p_arr) ** 2)
            st = np.sum((y_arr - y_arr.mean()) ** 2)
            r2 = 1 - ss / st if st > 0 else np.nan
            rmse = float(np.sqrt(np.mean((y_arr - p_arr) ** 2)))
            within10 = float(np.mean(np.abs(y_arr - p_arr) <= 10))
            rows.append({'stage': 'maturity', 'strategy': strat_name,
                         'model': mdl, 'n': int(len(y_arr)),
                         'R2': float(r2), 'R2_lo': np.nan, 'R2_hi': np.nan,
                         'RMSE': rmse, 'w10': within10,
                         'fit_time_s': np.nan})
            print(f'  {strat_name:10s} | {mdl:13s}  R²={r2:+.3f}  RMSE={rmse:.2f}d  n={len(y_arr)}')

    phase_e_new = pd.DataFrame(rows)
    print('\nBest maturity (strategy, model):')
    best = phase_e_new.loc[phase_e_new['R2'].idxmax()]
    BEST_STRAT, BEST_MDL = best['strategy'], best['model']
    print(f'  → {BEST_STRAT} | {BEST_MDL}  R²={best["R2"]:+.3f}')

    # ────────── 2. Update Phase E results parquet ─────────
    pe = pd.read_parquet(PHASE_E)
    pe = pe[pe['stage'] != 'maturity']
    pe = pd.concat([pe, phase_e_new], ignore_index=True)
    pe.to_parquet(PHASE_E, index=False)
    print(f'  → {PHASE_E} updated ({len(phase_e_new)} maturity rows)')

    # ────────── 3. Update LOYO predictions parquet ─────────
    best_loyo = pd.concat(loyo_preds[(BEST_STRAT, BEST_MDL)], ignore_index=True)
    lp = pd.read_parquet(LOYO_PREDS)
    lp = lp[lp['stage'] != 'maturity']
    lp = pd.concat([lp, best_loyo], ignore_index=True)
    lp.to_parquet(LOYO_PREDS, index=False)
    print(f'  → {LOYO_PREDS} updated ({len(best_loyo)} maturity rows)')

    # ────────── 4. LOSO transferability for maturity ─────────
    print('\n=== LOSO maturity ===')
    cols_best = feature_cols(feat, BEST_STRAT == 'C_Hybrid')
    loso_rows = []
    for state in STATES:
        col_st = f'state_{state}'
        if col_st not in train.columns:
            continue
        tr = train[train[col_st] != 1]
        te = train[train[col_st] == 1]
        if len(tr) < 50 or len(te) < 10:
            loso_rows.append({'stage': 'maturity', 'state': state,
                              'R2': np.nan, 'n': len(te)})
            continue
        pipe = build_pipeline(BEST_MDL, cols_best)
        pipe.fit(tr[cols_best], tr['maturity_dos_obs'])
        p = pipe.predict(te[cols_best])
        if p.ndim > 1:
            p = p.ravel()
        y = te['maturity_dos_obs'].values
        denom = np.sum((y - y.mean()) ** 2)
        r2 = 1 - np.sum((y - p) ** 2) / denom if denom > 0 else np.nan
        loso_rows.append({'stage': 'maturity', 'state': state,
                          'R2': float(r2), 'n': int(len(y))})
        print(f'  {state}: R²={r2:+.3f}  n={len(y)}')

    loso_old = pd.read_csv(LOSO_RES)
    loso_old = loso_old[loso_old['stage'] != 'maturity']
    loso_new = pd.concat([loso_old, pd.DataFrame(loso_rows)], ignore_index=True)
    loso_new.to_csv(LOSO_RES, index=False)
    print(f'  → {LOSO_RES} updated')

    # ────────── 5. Inference 2018-2024 for maturity ─────────
    print('\n=== Refitting + inference 2018-2024 (maturity) ===')
    pipe = build_pipeline(BEST_MDL, cols_best)
    pipe.fit(train[cols_best], train['maturity_dos_obs'])

    ext1 = pd.read_parquet(EXT_FEAT_2018)
    ext2 = pd.read_parquet(EXT_FEAT_19_24)
    ext = pd.concat([ext1, ext2], ignore_index=True)
    ext['field_id'] = ext['field_id'].astype(str)
    ext['year'] = ext['year'].astype(int)
    if 'state' in ext.columns:
        ext = ext.drop(columns=['state'])
    for c in cols_best:
        if c not in ext.columns:
            ext[c] = np.nan

    ext['maturity_dos_pred_new'] = pipe.predict(ext[cols_best])
    ext['maturity_date_new'] = (pd.to_datetime((ext['year'] - 1).astype(str) + '-07-01')
                                + pd.to_timedelta(ext['maturity_dos_pred_new'] - 1, unit='D'))
    ext['maturity_doy_pred_new'] = ext['maturity_date_new'].dt.dayofyear

    state_cols = [c for c in ext.columns if c.startswith('state_') and c != 'state']
    ext['state'] = ext[state_cols].idxmax(axis=1).str.replace('state_', '')

    ep = pd.read_parquet(EXT_PREDS)
    ep = ep.merge(ext[['field_id', 'year', 'maturity_dos_pred_new',
                        'maturity_doy_pred_new']],
                  on=['field_id', 'year'], how='left')
    # Replace old maturity columns
    if 'maturity_dos_pred' in ep.columns:
        ep['maturity_dos_pred'] = ep['maturity_dos_pred_new']
    if 'maturity_doy_pred' in ep.columns:
        ep['maturity_doy_pred'] = ep['maturity_doy_pred_new']
    ep = ep.drop(columns=['maturity_dos_pred_new', 'maturity_doy_pred_new'])
    ep.to_parquet(EXT_PREDS, index=False)
    print(f'  → {EXT_PREDS} updated')

    # ────────── 6. Re-compute trend slopes for maturity ─────────
    print('\n=== Trends for maturity ===')
    PLAUS_HI, PLAUS_LO = 210, 150
    sub_all = ep[ep['maturity_doy_pred'].between(PLAUS_LO, PLAUS_HI)]
    rng = np.random.RandomState(42)
    trend_rows = []
    for st in STATES:
        sst = sub_all[sub_all['state'] == st]
        if len(sst) < 30:
            continue
        x = sst['year'].values.astype(float)
        y = sst['maturity_doy_pred'].values.astype(float)
        slope, intercept = np.polyfit(x, y, 1)
        n = len(x)
        slopes = [np.polyfit(x[idx], y[idx], 1)[0]
                  for idx in (rng.choice(n, n, replace=True) for _ in range(500))]
        lo, hi = np.percentile(slopes, [2.5, 97.5])
        trend_rows.append({'stage': 'maturity', 'state': st,
                           'slope_d_per_yr': float(slope),
                           'ci_lo': float(lo), 'ci_hi': float(hi), 'n': int(n)})
        sig = '*' if (lo > 0 or hi < 0) else ''
        print(f'  {st}: slope = {slope:+.2f} d/yr  (95% CI [{lo:+.2f}, {hi:+.2f}])  n={n}  {sig}')

    trends_old = pd.read_csv(TRENDS)
    trends_old = trends_old[trends_old['stage'] != 'maturity']
    trends_new = pd.concat([trends_old, pd.DataFrame(trend_rows)], ignore_index=True)
    trends_new.to_csv(TRENDS, index=False)
    print(f'  → {TRENDS} updated')

    # ────────── 7. Feature importance for maturity ─────────
    print('\n=== Feature importance for maturity ===')
    pipe = build_pipeline(BEST_MDL, cols_best)
    pipe.fit(train[cols_best], train['maturity_dos_obs'])
    m = pipe.named_steps['m']
    imp_records = []
    if BEST_MDL in LINEAR:
        sel = pipe.named_steps['sel']
        mask = sel.get_support()
        selected = [c for c, s in zip(cols_best, mask) if s]
        coefs = np.abs(m.coef_)
        for f, imp in zip(selected, coefs):
            imp_records.append({'stage': 'maturity', 'feature': f,
                                'importance': float(imp)})
    else:
        imps = m.feature_importances_
        for f, imp in zip(cols_best, imps):
            imp_records.append({'stage': 'maturity', 'feature': f,
                                'importance': float(imp)})
    imp_new = pd.DataFrame(imp_records)
    imp_new['importance_norm'] = imp_new['importance'] / imp_new['importance'].sum() \
        if imp_new['importance'].sum() > 0 else imp_new['importance']

    # Re-categorize using same function as 41_feature_importance_f5.py
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'fi5', ROOT / 'scripts' / '00_extraction' / 'extension_2018_2024' /
                  '41_feature_importance_f5.py')
    fi5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(fi5)
    imp_new['group'] = imp_new['feature'].map(fi5.categorize)

    imp_old = pd.read_csv(FEAT_IMP)
    imp_old = imp_old[imp_old['stage'] != 'maturity']
    imp_all = pd.concat([imp_old, imp_new[['stage', 'feature', 'importance',
                                            'importance_norm', 'group']]],
                        ignore_index=True)
    imp_all.to_csv(FEAT_IMP, index=False)
    print(f'  → {FEAT_IMP} updated')

    print('\nAll artefacts updated. Re-run 42_paper_figures.py to regenerate figures.')


if __name__ == '__main__':
    main()
