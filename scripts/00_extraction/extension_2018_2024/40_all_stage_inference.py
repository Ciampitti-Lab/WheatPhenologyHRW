"""All-stage inference — train 8 best-per-stage V3 models on training
features (2014-2017), apply to extension (2018-2024), produce a
complete table of predicted stage dates per (field_id, harvest_year).

Also produces F7-style trends per stage.

Inputs:
    features_v3_realsowing_train.parquet
    features_v3_realsowing_2018.parquet + features_v3_realsowing_extension.parquet
        (concat → 2018-2024 inference set)

Output:
    predictions_all_stages_2018_2024.parquet
    docs/figures/F7b_all_stages_trends_2018_2024.png
"""
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

ROOT = Path('/home/vmangidi/repositories/WheatPhenologyHRW')
EXT = Path('/depot/ciampitti/data/WheatPhenologyHRW/data/raw/satellite/extension_2018_2024')
TRAIN_FEAT = EXT / 'features_v3_realsowing_train.parquet'
EXT_FEAT_2018 = EXT / 'features_v3_realsowing_2018.parquet'
EXT_FEAT_19_24 = EXT / 'features_v3_realsowing_extension.parquet'
PHENO = '/depot/ciampitti/data/WheatPhenologyHRW/data/processed/buffer_300m/wheat_hrw_phenology_buffer_matched.parquet'

OUT_PRED = EXT / 'predictions_all_stages_2018_2024.parquet'
OUT_FIG  = ROOT / 'docs' / 'figures' / 'F7b_all_stages_trends_2018_2024.png'

BEST_MODELS = {
    'emergence':  ('B_ML-only',  'LightGBM',     True),   # use_wes_features=False
    'tillering':  ('B_ML-only',  'ElasticNet',   False),
    'jointing':   ('C_Hybrid',   'LightGBM',     True),
    'flag_leaf':  ('C_Hybrid',   'XGBoost',      True),
    'boot':       ('C_Hybrid',   'LightGBM',     True),
    'heading':    ('C_Hybrid',   'ElasticNet',   True),
    'anthesis':   ('C_Hybrid',   'ElasticNet',   True),
    'maturity':   ('B_ML-only',  'LightGBM',     False),
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
                                              n_alphas=20, max_iter=20000, cv=5, n_jobs=-1),
        'Ridge':        lambda: RidgeCV(alphas=np.logspace(-3,3,30), cv=5),
        'RandomForest': lambda: RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42),
        'XGBoost':      lambda: XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                              n_jobs=-1, random_state=42, verbosity=0),
        'LightGBM':     lambda: LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                                n_jobs=-1, random_state=42, verbose=-1),
    }[name]


def main():
    # Load training features
    train = pd.read_parquet(TRAIN_FEAT)
    train['field_id'] = train['field_id'].astype(str)
    train['year'] = train['year'].astype(int)
    if 'state' in train.columns: train = train.drop(columns=['state'])
    print(f'Training features: {train.shape}')

    # Concat 2018 + 2019-2024 extension features
    ext_dfs = []
    if EXT_FEAT_2018.exists():
        ext_dfs.append(pd.read_parquet(EXT_FEAT_2018))
        print(f'  2018 features: {len(ext_dfs[0]):,} field-years')
    ext_dfs.append(pd.read_parquet(EXT_FEAT_19_24))
    print(f'  2019-2024 features: {len(ext_dfs[-1]):,} field-years')
    ext = pd.concat(ext_dfs, ignore_index=True)
    ext['field_id'] = ext['field_id'].astype(str)
    ext['year'] = ext['year'].astype(int)
    if 'state' in ext.columns: ext = ext.drop(columns=['state'])
    print(f'Combined extension: {len(ext):,} field-years across {sorted(ext["year"].unique())}')

    pheno = pd.read_parquet(PHENO)
    targets = build_targets(pheno)
    train_full = train.merge(targets, on=['field_id','year'], how='left')

    # Feature columns (same logic as Phase E)
    META = ['field_id','year','flag_true_doy','n_obs','sowing_doy_used'] + \
           [f'{s}_dos_obs' for s in STAGE_MAP]
    REDUND = ['GDD_M2_at_SOS','VD_at_SOS','emergence_doy',
              'VD_from_emergence_at_SOS','fV_from_emergence_at_SOS','days_emergence_to_SOS']
    we_multi = ['WE_emergence_doy','WE_tillering_doy','WE_jointing_doy','WE_flag_leaf_doy',
                'WE_boot_doy','WE_heading_doy','WE_anthesis_doy','WE_maturity_doy']
    ndre = [c for c in train.columns if c.startswith('NDRE')]
    drop_set = set(META + REDUND + ndre + ['ph_top'])
    all_feat = [c for c in train.columns
                if c not in drop_set and pd.api.types.is_numeric_dtype(train[c])]
    ml_only = [c for c in all_feat if c not in we_multi]
    WIN_PREF = ('heat_days_','hot_days_','frost_days_')
    WIN_SUFF = ('_gf','_pa','_pa_late')
    windowed = [c for c in train.columns
                if (c.endswith(WIN_SUFF) or c.startswith(WIN_PREF)) and c not in META]
    EARLY = {'emergence','tillering','jointing'}

    def stage_cols(stage, include_wes):
        base = all_feat if include_wes else ml_only
        if stage in EARLY:
            return [c for c in base if c not in windowed]
        return base

    # Train + predict per stage
    pred_rows = []
    for stage, (strat, mdl, _) in BEST_MODELS.items():
        include_wes = (strat == 'C_Hybrid')
        cols = stage_cols(stage, include_wes)
        target = f'{stage}_dos_obs'
        print(f'\n[{stage}] {strat} | {mdl} ({len(cols)} features)')
        df_tr = train_full.dropna(subset=[target]).copy()
        q1, q99 = df_tr[target].quantile([0.01, 0.99])
        df_tr = df_tr[(df_tr[target] >= q1) & (df_tr[target] <= q99)]
        print(f'  training rows: {len(df_tr):,}')

        # Train final model on full training set (SelectKBest for linear)
        steps = [('imp', SimpleImputer(strategy='median')),
                 ('sc',  StandardScaler())]
        if mdl in ('ElasticNet', 'Ridge'):
            steps.append(('sel', SelectKBest(f_regression, k=min(60, len(cols)))))
        steps.append(('m', get_factory(mdl)()))
        pipe = Pipeline(steps)
        pipe.fit(df_tr[cols], df_tr[target])

        # Apply to extension (ensure all cols exist)
        for c in cols:
            if c not in ext.columns:
                ext[c] = np.nan
        ext[f'{stage}_dos_pred'] = pipe.predict(ext[cols])

    # Build predictions table
    pred_cols = ['field_id', 'year', 'latitude', 'longitude']
    state_cols = [c for c in ext.columns if c.startswith('state_') and c != 'state']
    pred_cols += state_cols
    for stage in BEST_MODELS:
        pred_cols.append(f'{stage}_dos_pred')
    preds = ext[pred_cols].copy()

    # Add state column from one-hot
    preds['state'] = preds[state_cols].idxmax(axis=1).str.replace('state_', '')
    preds = preds.drop(columns=state_cols)

    # Convert each stage's DOS to calendar DOY
    for stage in BEST_MODELS:
        dos_col = f'{stage}_dos_pred'
        date = pd.to_datetime((preds['year'] - 1).astype(str) + '-07-01') + \
               pd.to_timedelta(preds[dos_col] - 1, unit='D')
        preds[f'{stage}_doy_pred'] = date.dt.dayofyear

    preds.to_parquet(OUT_PRED, index=False)
    print(f'\n→ {OUT_PRED}: {len(preds):,} rows')
    print(f'\nMedian predicted DOS per stage per year:')
    for stage in BEST_MODELS:
        med = preds.groupby('year')[f'{stage}_dos_pred'].median().round(0)
        print(f'  {stage:<12}', med.to_string().replace('\n', ' | '))

    # ── F7b: trends per stage per state (8 panels) ──
    GOLD = '#CEB888'; ACCENT = '#8E6F3E'
    palette = {'CO': '#CEB888', 'KS': '#8E6F3E', 'NE': '#1B6E63',
               'OK': '#A03939', 'TX': '#3F6FAA', 'NM': '#704A8A'}
    stages_order = ['emergence','tillering','jointing','flag_leaf',
                    'boot','heading','anthesis','maturity']
    # Per-stage plausibility bounds (DOY) — drops extrapolated outliers
    PLAUS = {'emergence': (260, 340), 'tillering': (30, 110), 'jointing': (50, 120),
             'flag_leaf': (90, 140), 'boot': (95, 150), 'heading': (100, 160),
             'anthesis': (110, 175), 'maturity': (150, 210)}

    rng = np.random.RandomState(42)
    trend_rows = []
    print('\n=== Per-stage per-state climate trends (OLS + 500x bootstrap) ===')
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), dpi=140, sharex=True)
    for i, stage in enumerate(stages_order):
        ax = axes[i // 4][i % 4]
        col = f'{stage}_doy_pred'
        lo, hi = PLAUS[stage]
        sub = preds[preds[col].between(lo, hi)]
        print(f'\n[{stage}]')
        years_x = np.array(sorted(sub['year'].unique()))
        for st in sorted(sub['state'].unique()):
            sst = sub[sub['state'] == st]
            if len(sst) < 30: continue
            agg = sst.groupby('year')[col].agg(['mean','std','count']).reset_index()
            agg['sem'] = agg['std'] / np.sqrt(agg['count'])
            c = palette.get(st, 'gray')
            ax.errorbar(agg['year'], agg['mean'], yerr=1.96*agg['sem'],
                        fmt='o-', color=c, lw=1.4, ms=5, capsize=2, label=f'{st}')
            # OLS + bootstrap CI on slope
            x = sst['year'].values.astype(float)
            y = sst[col].values.astype(float)
            slope, intercept = np.polyfit(x, y, 1)
            n = len(x)
            slopes = [np.polyfit(x[idx], y[idx], 1)[0]
                      for idx in (rng.choice(n, n, replace=True) for _ in range(500))]
            slo, shi = np.percentile(slopes, [2.5, 97.5])
            ax.plot(years_x, slope * years_x + intercept, '--', color=c, alpha=0.4, lw=1.0)
            trend_rows.append({'stage': stage, 'state': st, 'slope_d_per_yr': slope,
                               'ci_lo': slo, 'ci_hi': shi, 'n': n})
            sig = '*' if (slo > 0 or shi < 0) else ''
            print(f'  {st}: slope = {slope:+.2f} d/yr  (95% CI [{slo:+.2f}, {shi:+.2f}])  n={n}  {sig}')
        ax.set_title(stage.replace('_',' ').title(), fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.25)
        if i % 4 == 0:
            ax.set_ylabel('Predicted DOY', fontsize=10)
        if i // 4 == 1:
            ax.set_xlabel('Harvest year', fontsize=10)
        if i == 0:
            ax.legend(loc='best', fontsize=8, framealpha=0.92)
    plt.suptitle('All-stage predictions across the HRW belt, 2018–2024\n'
                 '(V3 framework: real sowing dates + WES fix; dashed = OLS trend)',
                 fontsize=12, fontweight='bold', y=1.00)
    plt.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FIG, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'\n→ {OUT_FIG}')

    trends_df = pd.DataFrame(trend_rows)
    OUT_TRENDS = EXT / 'v3_trends_per_stage_per_state.csv'
    trends_df.to_csv(OUT_TRENDS, index=False)
    print(f'→ {OUT_TRENDS}')


if __name__ == '__main__':
    main()
