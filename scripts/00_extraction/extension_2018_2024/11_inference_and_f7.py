"""STEP 5+6 — train final A6 ElasticNet for anthesis on the full
training set, apply to the 2019-2024 extension features, and build F7
(climate-trend figure showing per-state mean anthesis DOY across years).

Target stage: anthesis (best A6 model: ElasticNet Hybrid, R²=0.81 LOYO).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression

ROOT = Path('/home/vmangidi/repositories/WheatPhenologyHRW')
TRAIN_FEAT = '/depot/ciampitti/data/WheatPhenologyHRW/data/processed/features/features_a6.parquet'
PHENO_PATH = '/depot/ciampitti/data/WheatPhenologyHRW/data/processed/buffer_300m/wheat_hrw_phenology_buffer_matched.parquet'
EXT = Path('/depot/ciampitti/data/WheatPhenologyHRW/data/raw/satellite/extension_2018_2024')
EXT_FEAT = EXT / 'features_extension_2019_2024.parquet'
OUT_PRED = EXT / 'predictions_anthesis_2019_2024.parquet'
OUT_FIG  = ROOT / 'docs' / 'figures' / 'F7_anthesis_trend_2019_2024.png'

# Stage label mapping (matches scripts/03_modeling/01_multi_stage_ml.ipynb)
STAGE_MAP_ANTHESIS = ['Early Bloom', 'Bloom']
TARGET_STAGE = 'anthesis'

# Column hygiene — same drops the modelling notebook applies
META_COLS = ['field_id', 'year', 'flag_true_doy', 'n_obs', 'sowing_doy_used',
             f'{TARGET_STAGE}_dos_obs']
REDUND   = ['GDD_M2_at_SOS', 'VD_at_SOS', 'emergence_doy',
            'VD_from_emergence_at_SOS', 'fV_from_emergence_at_SOS',
            'days_emergence_to_SOS']


def build_anthesis_target(pheno):
    """Pull min DOS per (field_id, harvest_year) for anthesis-class stages."""
    s = pheno[pheno['growth_stage'].isin(STAGE_MAP_ANTHESIS)].copy()
    s['harvest_year'] = s['growing_season'].str.split('-').str[1].astype(int)
    s['field_id'] = s['FIELDID'].astype(str)
    e = s.groupby(['field_id', 'harvest_year'])['dos'].min().reset_index()
    e = e.rename(columns={'harvest_year': 'year', 'dos': f'{TARGET_STAGE}_dos_obs'})
    return e


def feature_columns(df):
    """All numeric features minus meta/redund/state-string/NDRE."""
    ndre = [c for c in df.columns if c.startswith('NDRE')]
    skip = set(META_COLS + REDUND + ndre + ['state'])
    return [c for c in df.columns if c not in skip
            and pd.api.types.is_numeric_dtype(df[c])]


def main():
    print('=== Loading training features + phenology labels ===')
    feat = pd.read_parquet(TRAIN_FEAT)
    pheno = pd.read_parquet(PHENO_PATH)
    feat['field_id'] = feat['field_id'].astype(str)
    feat['year'] = feat['year'].astype(int)
    if 'state' in feat.columns:
        feat = feat.drop(columns=['state'])

    targets = build_anthesis_target(pheno)
    feat = feat.merge(targets, on=['field_id', 'year'], how='left')
    train = feat.dropna(subset=[f'{TARGET_STAGE}_dos_obs']).copy()
    print(f'Training rows with anthesis label: {len(train):,}')

    feat_cols = feature_columns(feat)
    print(f'Feature count: {len(feat_cols)}')

    # ── Train final model on full training set ─────────────────────────────
    print('\n=== Training ElasticNet (final model, no CV split) ===')
    pipe = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc',  StandardScaler()),
        ('sel', SelectKBest(f_regression, k=80)),  # K from best A6 ElasticNet
        ('m',   ElasticNetCV(l1_ratio=[.1, .3, .5, .7, .9, .95, 1.0],
                              n_alphas=20, max_iter=20000, cv=5, n_jobs=-1)),
    ])
    pipe.fit(train[feat_cols], train[f'{TARGET_STAGE}_dos_obs'])
    print(f'  alpha:      {pipe.named_steps["m"].alpha_:.4f}')
    print(f'  l1_ratio:   {pipe.named_steps["m"].l1_ratio_:.2f}')
    print(f'  n features kept after SelectKBest: {pipe.named_steps["sel"].k}')

    # ── Apply to extension features ───────────────────────────────────────
    print('\n=== Applying to 2019-2024 extension features ===')
    ext = pd.read_parquet(EXT_FEAT)
    ext['field_id'] = ext['field_id'].astype(str)
    ext['year'] = ext['year'].astype(int)
    if 'state' in ext.columns:
        ext = ext.drop(columns=['state'])
    # Ensure same column order (some columns may be missing → fill NaN)
    for c in feat_cols:
        if c not in ext.columns:
            ext[c] = np.nan
    pred = pipe.predict(ext[feat_cols])
    ext['anthesis_dos_pred'] = pred

    # Convert DOS → DOY (calendar day of harvest_year)
    # gs_start = Jul 1 of (harvest_year - 1); date = gs_start + (dos - 1) days
    ext['anthesis_date'] = pd.to_datetime(
        (ext['year'] - 1).astype(str) + '-07-01') + pd.to_timedelta(
        ext['anthesis_dos_pred'] - 1, unit='D')
    ext['anthesis_doy'] = ext['anthesis_date'].dt.dayofyear

    # Recover state from one-hot
    state_cols = [c for c in ext.columns if c.startswith('state_') and c != 'state']
    ext['state'] = ext[state_cols].idxmax(axis=1).str.replace('state_', '')

    # Sanity: keep only physiologically plausible anthesis predictions
    # (DOY 90-180 = April-June for HRW wheat in US Plains)
    plausible = ext[(ext['anthesis_doy'] >= 90) & (ext['anthesis_doy'] <= 200)].copy()
    print(f'  predicted: n={len(ext):,}, plausible (DOY 90-200): {len(plausible):,}')
    print(f'  median anthesis DOY by year:')
    print(plausible.groupby('year')['anthesis_doy'].median().round(1).to_string())

    plausible[['field_id', 'year', 'state', 'anthesis_dos_pred',
               'anthesis_doy', 'latitude', 'longitude']].to_parquet(OUT_PRED, index=False)
    print(f'\n→ predictions: {OUT_PRED}')

    # ── F7 figure ─────────────────────────────────────────────────────────
    build_f7(plausible)


def build_f7(df):
    print('\n=== Building F7 climate-trend figure ===')
    # Per-state per-year median + IQR
    summ = (df.groupby(['state', 'year'])['anthesis_doy']
              .agg(['median', 'mean', 'std', 'count'])
              .reset_index())
    print(summ.to_string())

    # Per-state linear trend (DOY/year) + 95 % CI from bootstrap
    rng = np.random.RandomState(42)
    states = sorted(df['state'].unique())
    palette = {s: c for s, c in zip(states,
                  ['#CEB888', '#8E6F3E', '#1B6E63', '#704A8A', '#A03939', '#3F6FAA'])}

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)
    for st in states:
        sub = df[df['state'] == st]
        if len(sub) < 50:
            continue
        agg = sub.groupby('year')['anthesis_doy'].agg(['mean', 'std', 'count']).reset_index()
        agg['sem'] = agg['std'] / np.sqrt(agg['count'])
        col = palette.get(st, 'gray')
        ax.errorbar(agg['year'], agg['mean'], yerr=1.96 * agg['sem'],
                    fmt='o-', color=col, lw=1.6, ms=6, capsize=3,
                    label=f'{st} (n={int(agg["count"].sum()):,})')
        # OLS trend
        x = sub['year'].values.astype(float)
        y = sub['anthesis_doy'].values.astype(float)
        slope, intercept = np.polyfit(x, y, 1)
        # Bootstrap CI on slope
        n = len(x)
        slopes = []
        for _ in range(500):
            idx = rng.choice(n, n, replace=True)
            slopes.append(np.polyfit(x[idx], y[idx], 1)[0])
        lo, hi = np.percentile(slopes, [2.5, 97.5])
        years_x = np.array(sorted(df['year'].unique()))
        ax.plot(years_x, slope * years_x + intercept,
                '--', color=col, alpha=0.45, lw=1.2)
        print(f'  {st}: slope = {slope:+.2f} d/yr  (95 % CI [{lo:+.2f}, {hi:+.2f}])')

    ax.set_xlabel('Harvest year', fontsize=12)
    ax.set_ylabel('Predicted anthesis DOY (day of year)', fontsize=12)
    ax.set_title('Field-scale anthesis timing across the HRW belt, 2019–2024\n'
                 '(out-of-training-window inference using the trained A6 ElasticNet model)',
                 fontsize=12, pad=14)
    ax.grid(True, alpha=0.25)
    ax.legend(loc='best', fontsize=9, framealpha=0.92)
    ax.set_xticks(sorted(df['year'].unique()))
    plt.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_FIG, dpi=160, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'\n→ {OUT_FIG}')


if __name__ == '__main__':
    main()
