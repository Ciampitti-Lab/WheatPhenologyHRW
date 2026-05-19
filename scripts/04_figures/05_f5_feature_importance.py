"""F5 — Per-stage feature importance.
[NOTE] Reflects the original 5-model best map. For the current 7-model
pipeline (FT-Transformer adopted at anthesis/maturity) the canonical,
FT-capable generator is scripts/04_figures/09_paper_figures.py.


For each of the 8 stages, refit the best model (per V3 Phase E) on the full
training set, extract feature importance (|coef| for linear models after
SelectKBest+StandardScaler; gain for tree models), categorize features into
physiological groups, and visualize.

Outputs:
    docs/figures/F5_feature_importance_v3.png
    docs/figures/feature_importance_per_stage.csv
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.config import CFG, REPO_ROOT
_WORK = REPO_ROOT / CFG.paths.work_dir
_PHENO = str(REPO_ROOT / CFG.paths.phenology_matched)

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
V3_FEAT = EXT / "features_v3_realsowing_train.parquet"
PHENO = _PHENO
OUT_FIG = ROOT / 'docs' / 'figures' / 'F5_feature_importance_v3.png'
OUT_CSV = ROOT / 'docs' / 'figures' / 'feature_importance_per_stage.csv'

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


def categorize(f):
    fl = f.lower()
    # Wang-Engel-Streck simulator outputs
    if f.startswith('WE_'):
        return 'WES (thermal-time)'
    # MODIS LST (cover LST_ and lst_)
    if fl.startswith('lst_'):
        return 'MODIS LST'
    # Daymet & derived meteo: temperature, precipitation, radiation, vapor pressure,
    # daylength, photothermal quotient, frost / heat / hot days, etc.
    if (fl.startswith(('heat_days_','hot_days_','frost_days_','frost_events_',
                       'prcp_','ppt_','tmax_','tmin_','tmean_','tavg_','temp_',
                       'srad_','rad_','vp_','dayl_','dl_','swe_','ptq_'))
        or fl.startswith('days_above') or fl.startswith('days_below')):
        return 'Daymet'
    # GDD / vernalization / dormancy
    if (f.startswith(('GDD','VD','fV'))
        or fl.startswith(('dormancy','vernalization','vd_','fv_'))):
        return 'GDD/Vernalization'
    # HLS windowed (any phenology-window suffix)
    if f.endswith(('_gf','_pa','_pa_late','_early','_mid','_late')):
        return 'HLS windowed'
    # HLS-derived phenometrics: explicit phenology keywords
    if any(k in fl for k in ['sos','pos','eos','peak','integrated','amplitude','slope',
                              'greenup','senesc','midpoint','steepness','duration',
                              'shoulder','base','rate']):
        return 'HLS phenometrics'
    # Raw VIs without explicit phenometric suffix
    if any(f.startswith(p) for p in ['NDVI','EVI','GCVI','GCC','GNDVI','REIP','NDWI','NIRv']):
        return 'HLS phenometrics'
    # Site / management
    if f in ('latitude','longitude','elevation','sowing_doy_used_actual') or f.startswith('state_'):
        return 'Site/state'
    return 'Other'


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


def main():
    print('Loading features and building targets...')
    feat = pd.read_parquet(V3_FEAT)
    feat['field_id'] = feat['field_id'].astype(str)
    feat['year'] = feat['year'].astype(int)
    if 'state' in feat.columns:
        feat = feat.drop(columns=['state'])
    pheno = pd.read_parquet(PHENO)
    targets = build_targets(pheno)
    feat = feat.merge(targets, on=['field_id','year'], how='left')

    META = ['field_id','year','flag_true_doy','n_obs','sowing_doy_used'] + \
           [f'{s}_dos_obs' for s in STAGE_MAP]
    REDUND = ['GDD_M2_at_SOS','VD_at_SOS','emergence_doy',
              'VD_from_emergence_at_SOS','fV_from_emergence_at_SOS','days_emergence_to_SOS']
    we_multi = [f'WE_{s}_doy' for s in STAGE_MAP]
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

    records = []
    print('\n=== Refitting best model per stage and extracting importances ===')
    for stage, (strat, mdl) in BEST_MODELS.items():
        include_wes = (strat == 'C_Hybrid')
        cols = stage_cols(stage, include_wes)
        target = f'{stage}_dos_obs'
        df = feat.dropna(subset=[target]).copy()
        q1, q99 = df[target].quantile([0.01, 0.99])
        df = df[(df[target] >= q1) & (df[target] <= q99)]

        print(f'\n[{stage}] {strat} | {mdl}  (n={len(df)}, features={len(cols)})')
        steps = [('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())]
        if mdl in LINEAR:
            steps.append(('sel', SelectKBest(f_regression, k=min(60, len(cols)))))
        steps.append(('m', get_factory(mdl)()))
        pipe = Pipeline(steps)
        pipe.fit(df[cols], df[target])

        m = pipe.named_steps['m']
        if mdl in LINEAR:
            sel = pipe.named_steps['sel']
            mask = sel.get_support()
            selected = [c for c, s in zip(cols, mask) if s]
            coefs = np.abs(m.coef_)
            for f, imp in zip(selected, coefs):
                records.append({'stage': stage, 'feature': f,
                                'importance': float(imp), 'group': categorize(f)})
        else:
            imps = m.feature_importances_
            for f, imp in zip(cols, imps):
                records.append({'stage': stage, 'feature': f,
                                'importance': float(imp), 'group': categorize(f)})

    imp_df = pd.DataFrame(records)
    imp_df['importance_norm'] = imp_df.groupby('stage')['importance'].transform(
        lambda x: x / x.sum() if x.sum() > 0 else x)
    imp_df.to_csv(OUT_CSV, index=False)
    print(f'\n→ {OUT_CSV}')

    group_df = imp_df.groupby(['stage','group'])['importance_norm'].sum().reset_index()
    pivot = group_df.pivot_table(index='stage', columns='group',
                                  values='importance_norm', fill_value=0)
    print('\n=== Per-stage feature-group breakdown (fraction of total) ===')
    print(pivot.round(2).to_string())

    # ─── F5 figure ─────────────────────────────────────────────
    stages_order = ['emergence','tillering','jointing','flag_leaf',
                    'boot','heading','anthesis','maturity']
    groups_order = ['WES (thermal-time)','GDD/Vernalization','Daymet','MODIS LST',
                    'HLS phenometrics','HLS windowed','Site/state','Other']
    pivot = pivot.reindex(stages_order)
    cols_present = [g for g in groups_order if g in pivot.columns]
    pivot = pivot.reindex(columns=cols_present)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), dpi=140,
                                    gridspec_kw={'width_ratios': [2, 1.7]})
    cmap = LinearSegmentedColormap.from_list('cream', ['#FFFFFF', '#CEB888', '#8E6F3E'], N=256)
    im = ax1.imshow(pivot.values * 100, cmap=cmap, vmin=0, vmax=60, aspect='auto')
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            v = pivot.iloc[i, j] * 100
            if v >= 1:
                col = 'white' if v > 35 else 'black'
                ax1.text(j, i, f'{v:.0f}', ha='center', va='center',
                         fontsize=9, color=col,
                         fontweight='bold' if v > 35 else 'normal')
    ax1.set_xticks(range(len(pivot.columns)))
    ax1.set_xticklabels(pivot.columns, rotation=30, ha='right', fontsize=9)
    ax1.set_yticks(range(len(pivot.index)))
    ax1.set_yticklabels([s.replace('_',' ').title() for s in pivot.index], fontsize=10)
    cbar = plt.colorbar(im, ax=ax1, fraction=0.04, pad=0.04)
    cbar.set_label('% of total importance', fontsize=9)
    ax1.set_title('A. Feature-group importance per stage',
                  fontsize=11, fontweight='bold', pad=10)

    ax2.set_title('B. Top 5 features per stage', fontsize=11, fontweight='bold', pad=10)
    ax2.axis('off')
    lines = []
    for stage in stages_order:
        sub = imp_df[imp_df['stage'] == stage].sort_values('importance', ascending=False).head(5)
        feats_str = '  '.join(sub['feature'].tolist())
        lines.append(f'{stage.replace("_"," ").title():<12s}: {feats_str}')
    ax2.text(0.0, 0.97, '\n\n'.join(lines), ha='left', va='top',
             fontsize=7.5, family='monospace', transform=ax2.transAxes)

    plt.suptitle('Feature importance across the 8 phenology stages\n'
                 '(best model per stage; |coef| for linear, gain for trees; normalized per stage)',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FIG, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'\n→ {OUT_FIG}')


if __name__ == '__main__':
    main()
