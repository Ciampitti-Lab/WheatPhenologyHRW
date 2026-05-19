"""F6 — Leave-One-State-Out (LOSO) transferability heatmap.
[NOTE] Reflects the original 5-model best map. For the current 7-model
pipeline (FT-Transformer adopted at anthesis/maturity) the canonical,
FT-capable generator is scripts/04_figures/09_paper_figures.py.


For each (stage, state-held-out), train on the other 4 states and test
on the held-out state. Reveals spatial generalization of the framework.

Output: docs/figures/F6_loso_transferability_v3.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.config import CFG, REPO_ROOT
_WORK = REPO_ROOT / CFG.paths.work_dir
_PHENO = str(REPO_ROOT / CFG.paths.phenology_matched)

import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

EXT = _WORK
V2_FEAT = EXT / "features_v3_realsowing_train.parquet"
PHENO = _PHENO
OUT_FIG = ROOT / 'docs' / 'figures' / 'F6_loso_transferability_v3.png'
OUT_TABLE = EXT / 'v3_loso_results.csv'

# Best model per stage (from V2 vanilla Phase E)
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
STATES = ['TX', 'OK', 'KS', 'NE', 'CO']


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


def loso_for_stage(df, feat_cols, target, factory, model_name):
    df2 = df.dropna(subset=[target]).copy()
    q1, q99 = df2[target].quantile([0.01, 0.99])
    df2 = df2[(df2[target] >= q1) & (df2[target] <= q99)]
    results = {}
    for state in STATES:
        col = f'state_{state}'
        if col not in df2.columns:
            results[state] = (np.nan, 0); continue
        tr = df2[df2[col] != 1]
        te = df2[df2[col] == 1]
        if len(tr) < 50 or len(te) < 10:
            results[state] = (np.nan, len(te)); continue
        steps = [('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]
        if model_name in LINEAR:
            steps.append(('sel', SelectKBest(f_regression, k=min(60, len(feat_cols)))))
        steps.append(('m', factory()))
        pipe = Pipeline(steps)
        pipe.fit(tr[feat_cols], tr[target])
        p = pipe.predict(te[feat_cols])
        if p.ndim > 1: p = p.ravel()
        yt = te[target].values
        denom = np.sum((yt - yt.mean()) ** 2)
        r2 = 1 - np.sum((yt - p) ** 2) / denom if denom > 0 else np.nan
        results[state] = (float(r2), len(yt))
    return results


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

    rows = []
    print('\n=== LOSO per stage ===')
    for stage, (strat, mdl) in BEST_MODELS.items():
        include_wes = (strat == 'C_Hybrid')
        cols = stage_cols(stage, include_wes)
        target = f'{stage}_dos_obs'
        print(f'\n[{stage}] {strat} | {mdl}')
        res = loso_for_stage(feat, cols, target, get_factory(mdl), mdl)
        for state, (r2, n) in res.items():
            rows.append({'stage': stage, 'state': state, 'R2': r2, 'n': n})
            print(f'   {state}: R²={r2:.3f}  n={n}')

    df = pd.DataFrame(rows)
    df.to_csv(OUT_TABLE, index=False)
    print(f'\n→ {OUT_TABLE}')

    # ─── Heatmap ──
    stages_order = ['emergence','tillering','jointing','flag_leaf',
                    'boot','heading','anthesis','maturity']
    heat = df.pivot_table(index='stage', columns='state', values='R2')
    heat = heat.reindex(stages_order)[STATES]

    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=140)
    cmap = LinearSegmentedColormap.from_list('rwg', ['#A03939', '#FFFFFF', '#1B6E63'], N=256)
    im = ax.imshow(heat.values, cmap=cmap, vmin=-0.2, vmax=0.85, aspect='auto')

    for i, stage in enumerate(stages_order):
        for j, state in enumerate(STATES):
            v = heat.iloc[i, j]
            if pd.isna(v):
                txt = '—'; col = '#888'
            else:
                txt = f'{v:.2f}'
                col = 'white' if abs(v) > 0.4 else 'black'
            ax.text(j, i, txt, ha='center', va='center', fontsize=9, color=col,
                    fontweight='bold' if (not pd.isna(v) and v >= 0.7) else 'normal')

    ax.set_xticks(range(len(STATES)))
    ax.set_xticklabels([f'{s}\nheld out' for s in STATES], fontsize=10)
    ax.set_yticks(range(len(stages_order)))
    ax.set_yticklabels([s.replace('_',' ').title() for s in stages_order], fontsize=10)
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label('R² (LOSO)', fontsize=10)

    ax.set_title('F6. Leave-One-State-Out transferability\n'
                 '(model trained on 4 states, evaluated on the held-out state)',
                 fontsize=11, fontweight='bold', pad=12)
    plt.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FIG, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'\n→ {OUT_FIG}')
    print(f'Wall time: {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
