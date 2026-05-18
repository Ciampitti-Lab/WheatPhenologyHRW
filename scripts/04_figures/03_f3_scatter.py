"""F3 — Per-stage predicted vs observed scatter (8 panels).
The money figure for the paper.

Uses V2 vanilla features (growing-season DOS), runs LOYO with the
best-performing model per stage from Phase E V2 vanilla results,
saves predictions, and plots an 8-panel figure with R², RMSE, n,
and ±10-day reference bands.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.config import CFG, REPO_ROOT
_WORK = REPO_ROOT / CFG.paths.work_dir
_PHENO = str(REPO_ROOT / CFG.paths.phenology_matched)

import time
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

EXT = _WORK
V2_FEAT = EXT / 'features_v3_realsowing_train.parquet'
PHENO = _PHENO
OUT_FIG = ROOT / 'docs' / 'figures' / 'F3_per_stage_scatter_v3.png'
OUT_PRED = EXT / 'v3_loyo_predictions.parquet'

# Best model per stage (from Phase E V2 vanilla results)
BEST_MODELS = {
    'emergence':  ('C_Hybrid',   'LightGBM'),
    'tillering':  ('B_ML-only',  'ElasticNet'),
    'jointing':   ('C_Hybrid',   'LightGBM'),
    'flag_leaf':  ('C_Hybrid',   'XGBoost'),
    'boot':       ('C_Hybrid',   'LightGBM'),
    'heading':    ('C_Hybrid',   'ElasticNet'),
    'anthesis':   ('C_Hybrid',   'ElasticNet'),
    'maturity':   ('B_ML-only',  'LightGBM'),
}

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


def get_factory(name):
    return {
        'ElasticNet':   lambda: ElasticNetCV(l1_ratio=[.1,.3,.5,.7,.9,.95,1.0],
                                              n_alphas=20, max_iter=20000, cv=5, n_jobs=1),
        'Ridge':        lambda: RidgeCV(alphas=np.logspace(-3,3,30), cv=5),
        'RandomForest': lambda: RandomForestRegressor(n_estimators=200, n_jobs=1, random_state=42),
        'XGBoost':      lambda: XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                              n_jobs=1, random_state=42, verbosity=0),
        'LightGBM':     lambda: LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                                n_jobs=1, random_state=42, verbose=-1),
    }[name]


def main():
    t0 = time.time()
    print('Loading V2 vanilla features...')
    feat = pd.read_parquet(V2_FEAT)
    feat['field_id'] = feat['field_id'].astype(str)
    feat['year'] = feat['year'].astype(int)
    if 'state' in feat.columns: feat = feat.drop(columns=['state'])

    pheno = pd.read_parquet(PHENO)
    targets = build_targets(pheno)
    feat = feat.merge(targets, on=['field_id','year'], how='left')

    META = ['field_id','year','flag_true_doy','n_obs','sowing_doy_used'] + \
           [f'{s}_dos_obs' for s in STAGE_MAP]
    REDUND = ['GDD_M2_at_SOS','VD_at_SOS','emergence_doy',
              'VD_from_emergence_at_SOS','fV_from_emergence_at_SOS','days_emergence_to_SOS']
    we_multi = ['WE_emergence_doy','WE_tillering_doy','WE_jointing_doy','WE_flag_leaf_doy',
                'WE_boot_doy','WE_heading_doy','WE_anthesis_doy','WE_maturity_doy']
    ndre = [c for c in feat.columns if c.startswith('NDRE')]
    drop_set = set(META + REDUND + ndre + ['ph_top'])
    all_feat = [c for c in feat.columns
                if c not in drop_set and pd.api.types.is_numeric_dtype(feat[c])]
    ml_only = [c for c in all_feat if c not in we_multi]
    WIN_PREF = ('heat_days_','hot_days_','frost_days_')
    WIN_SUFF = ('_gf','_pa','_pa_late')
    windowed = [c for c in feat.columns
                if (c.endswith(WIN_SUFF) or c.startswith(WIN_PREF)) and c not in META]
    EARLY = {'emergence','tillering','jointing'}

    def stage_cols(stage, include_wes):
        base = all_feat if include_wes else ml_only
        if stage in EARLY:
            return [c for c in base if c not in windowed]
        return base

    # Run LOYO for each stage's best model and collect predictions
    all_preds = []
    metrics_per_stage = {}
    for stage, (strat, mdl) in BEST_MODELS.items():
        include_wes = (strat == 'C_Hybrid')
        cols = stage_cols(stage, include_wes)
        target = f'{stage}_dos_obs'
        print(f'\n[{stage}] {strat} | {mdl} ({len(cols)} features)')
        y_pred, y_true = loyo(feat, cols, target, get_factory(mdl), mdl)
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        denom = np.sum((y_true - y_true.mean()) ** 2)
        r2 = float(1 - np.sum((y_true - y_pred) ** 2) / denom) if denom > 0 else 0
        w10 = float(np.mean(np.abs(y_true - y_pred) <= 10) * 100)
        metrics_per_stage[stage] = {'r2': r2, 'rmse': rmse, 'w10': w10, 'n': len(y_true)}
        print(f'    R²={r2:.3f}  RMSE={rmse:.1f}d  ±10d={w10:.1f}%  n={len(y_true)}')
        for p, t in zip(y_pred, y_true):
            all_preds.append({'stage': stage, 'observed': t, 'predicted': p})

    preds_df = pd.DataFrame(all_preds)
    preds_df.to_parquet(OUT_PRED, index=False)
    print(f'\nPredictions saved: {OUT_PRED}')

    # ── 8-panel scatter ──
    print('\nBuilding F3 figure...')
    stages_order = ['emergence','tillering','jointing','flag_leaf',
                    'boot','heading','anthesis','maturity']
    GOLD = '#CEB888'; ACCENT = '#8E6F3E'; DARK = '#1B1B1B'
    CRITICAL_STAGES = {'flag_leaf','boot','heading','anthesis'}  # highlighted

    fig, axes = plt.subplots(2, 4, figsize=(15, 8), dpi=140, sharex=False, sharey=False)
    for i, stage in enumerate(stages_order):
        ax = axes[i // 4][i % 4]
        sub = preds_df[preds_df['stage'] == stage]
        m = metrics_per_stage[stage]
        is_crit = stage in CRITICAL_STAGES

        col = ACCENT if is_crit else '#888'
        ax.scatter(sub['observed'], sub['predicted'], s=8, c=col, alpha=0.35,
                   edgecolors='none')

        # 1:1 line + ±10 band
        all_vals = pd.concat([sub['observed'], sub['predicted']])
        lo, hi = all_vals.quantile([0.02, 0.98]).values
        margin = (hi - lo) * 0.05
        x0, x1 = lo - margin, hi + margin
        ax.plot([x0, x1], [x0, x1], 'k-', lw=1.3, alpha=0.8)
        ax.fill_between([x0, x1], [x0 - 10, x1 - 10], [x0 + 10, x1 + 10],
                        color='gray', alpha=0.08, edgecolor=None)
        ax.set_xlim(x0, x1); ax.set_ylim(x0, x1)

        # Stats box
        txt = (f'R² = {m["r2"]:.3f}\n'
               f'RMSE = {m["rmse"]:.1f} d\n'
               f'±10d = {m["w10"]:.1f}%\n'
               f'n = {m["n"]:,}')
        ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=9,
                va='top', ha='left',
                bbox=dict(boxstyle='round,pad=0.4',
                          facecolor='white' if not is_crit else GOLD,
                          edgecolor=ACCENT if is_crit else '#666', alpha=0.92))

        title = stage.replace('_', ' ').title()
        if is_crit:
            title = f'★ {title}'
        ax.set_title(title, fontsize=11, fontweight='bold' if is_crit else 'normal',
                     color=DARK)
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        if i % 4 == 0:
            ax.set_ylabel('Predicted DOS', fontsize=10)
        if i // 4 == 1:
            ax.set_xlabel('Observed DOS', fontsize=10)

    plt.suptitle('Per-stage prediction accuracy across 8 winter-wheat developmental stages\n'
                 '(Leave-One-Year-Out CV, ★ = agronomically critical reproductive stages)',
                 fontsize=12, fontweight='bold', color=DARK, y=1.00)
    plt.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FIG, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'\n→ {OUT_FIG}')
    print(f'Wall time: {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
