"""Phase G-mini — train anthesis ElasticNet on v2 (growing-season)
training features, apply to v2 extension features, build F7 v2.

This is the minimal pipeline needed to verify the v2 framework gives
sensible results. Full 8-stage re-training is in Phase E (separate
checkpointed script).

Outputs:
    predictions_anthesis_v3_2019_2024.parquet
    docs/figures/F7_v3_anthesis_trend_2019_2024.png
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression

ROOT = Path('/home/vmangidi/repositories/WheatPhenologyHRW')
EXT = Path('/depot/ciampitti/data/WheatPhenologyHRW/data/raw/satellite/extension_2018_2024')

TRAIN_FEAT = EXT / 'features_v3_realsowing_train.parquet'
EXT_FEAT   = EXT / 'features_v3_realsowing_extension.parquet'
PHENO_PATH = ('/depot/ciampitti/data/WheatPhenologyHRW/data/processed/'
              'buffer_300m/wheat_hrw_phenology_buffer_matched.parquet')

OUT_PRED = EXT / 'predictions_anthesis_v3_2019_2024.parquet'
OUT_FIG  = ROOT / 'docs' / 'figures' / 'F7_v3_anthesis_trend_2019_2024.png'

STAGE_MAP_ANTHESIS = ['Early Bloom', 'Bloom']
TARGET = 'anthesis_dos_obs'

META = ['field_id','year','flag_true_doy','n_obs','sowing_doy_used', TARGET]
REDUND = ['GDD_M2_at_SOS','VD_at_SOS','emergence_doy',
          'VD_from_emergence_at_SOS','fV_from_emergence_at_SOS',
          'days_emergence_to_SOS']


def build_anthesis_target(pheno):
    s = pheno[pheno['growth_stage'].isin(STAGE_MAP_ANTHESIS)].copy()
    s['harvest_year'] = s['growing_season'].str.split('-').str[1].astype(int)
    s['field_id'] = s['FIELDID'].astype(str)
    e = s.groupby(['field_id','harvest_year'])['dos'].min().reset_index()
    return e.rename(columns={'harvest_year':'year','dos':TARGET})


def feature_columns(df):
    ndre = [c for c in df.columns if c.startswith('NDRE')]
    skip = set(META + REDUND + ndre + ['state'])
    return [c for c in df.columns
            if c not in skip and pd.api.types.is_numeric_dtype(df[c])]


def main():
    print('=== Phase G-mini: train anthesis + F7 v2 ===')
    feat = pd.read_parquet(TRAIN_FEAT)
    print(f'Training features (v2): {feat.shape}')
    feat['field_id'] = feat['field_id'].astype(str)
    feat['year'] = feat['year'].astype(int)
    if 'state' in feat.columns:
        feat = feat.drop(columns=['state'])

    pheno = pd.read_parquet(PHENO_PATH)
    targets = build_anthesis_target(pheno)
    feat = feat.merge(targets, on=['field_id','year'], how='left')
    train = feat.dropna(subset=[TARGET]).copy()
    print(f'Training rows with anthesis label: {len(train):,}')

    feat_cols = feature_columns(feat)
    print(f'Feature columns: {len(feat_cols)}')

    print('\n=== Training ElasticNetCV on full set ===')
    pipe = Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('sc',  StandardScaler()),
        ('sel', SelectKBest(f_regression, k=80)),
        ('m',   ElasticNetCV(l1_ratio=[.1,.3,.5,.7,.9,.95,1.0],
                              n_alphas=20, max_iter=20000, cv=5, n_jobs=-1)),
    ])
    pipe.fit(train[feat_cols], train[TARGET])
    print(f'  alpha={pipe.named_steps["m"].alpha_:.4f}, '
          f'l1_ratio={pipe.named_steps["m"].l1_ratio_:.2f}')

    # In-sample R² (sanity)
    in_pred = pipe.predict(train[feat_cols])
    ss_res = np.sum((train[TARGET].values - in_pred) ** 2)
    ss_tot = np.sum((train[TARGET].values - train[TARGET].mean()) ** 2)
    print(f'  In-sample R² (sanity): {1 - ss_res/ss_tot:.3f}')

    # ── Apply to extension ──
    print('\n=== Applying to extension 2019-2024 ===')
    ext = pd.read_parquet(EXT_FEAT)
    ext['field_id'] = ext['field_id'].astype(str)
    ext['year'] = ext['year'].astype(int)
    if 'state' in ext.columns:
        ext = ext.drop(columns=['state'])
    for c in feat_cols:
        if c not in ext.columns:
            ext[c] = np.nan
    ext['anthesis_dos_pred'] = pipe.predict(ext[feat_cols])

    # DOS → DOY conversion
    ext['anthesis_date'] = pd.to_datetime(
        (ext['year'] - 1).astype(str) + '-07-01') + pd.to_timedelta(
        ext['anthesis_dos_pred'] - 1, unit='D')
    ext['anthesis_doy'] = ext['anthesis_date'].dt.dayofyear

    # Recover state from one-hot
    state_cols = [c for c in ext.columns if c.startswith('state_') and c != 'state']
    ext['state'] = ext[state_cols].idxmax(axis=1).str.replace('state_', '')

    plausible = ext[(ext['anthesis_doy'] >= 90) & (ext['anthesis_doy'] <= 200)].copy()
    print(f'  total: {len(ext):,}  plausible (DOY 90-200): {len(plausible):,}')
    print(f'\nMedian anthesis DOY by year:')
    print(plausible.groupby('year')['anthesis_doy'].median().round(1).to_string())

    plausible[['field_id','year','state','anthesis_dos_pred','anthesis_doy',
               'latitude','longitude']].to_parquet(OUT_PRED, index=False)
    print(f'\n→ {OUT_PRED}')

    # ── F7 figure ──
    build_f7(plausible)


def build_f7(df):
    print('\n=== Building F7 v2 ===')
    rng = np.random.RandomState(42)
    states = sorted(df['state'].unique())
    palette = dict(zip(states,
        ['#CEB888', '#8E6F3E', '#1B6E63', '#704A8A', '#A03939', '#3F6FAA']))

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=150)
    for st in states:
        sub = df[df['state'] == st]
        if len(sub) < 50:
            continue
        agg = sub.groupby('year')['anthesis_doy'].agg(['mean','std','count']).reset_index()
        agg['sem'] = agg['std'] / np.sqrt(agg['count'])
        col = palette.get(st, 'gray')
        ax.errorbar(agg['year'], agg['mean'], yerr=1.96*agg['sem'],
                    fmt='o-', color=col, lw=1.6, ms=6, capsize=3,
                    label=f'{st} (n={int(agg["count"].sum()):,})')
        # OLS trend
        x = sub['year'].values.astype(float)
        y = sub['anthesis_doy'].values.astype(float)
        slope, intercept = np.polyfit(x, y, 1)
        slopes = []
        n = len(x)
        for _ in range(500):
            idx = rng.choice(n, n, replace=True)
            slopes.append(np.polyfit(x[idx], y[idx], 1)[0])
        lo, hi = np.percentile(slopes, [2.5, 97.5])
        years_x = np.array(sorted(df['year'].unique()))
        ax.plot(years_x, slope*years_x + intercept, '--', color=col, alpha=0.45, lw=1.2)
        print(f'  {st}: slope = {slope:+.2f} d/yr  (95% CI [{lo:+.2f}, {hi:+.2f}])')

    ax.set_xlabel('Harvest year', fontsize=12)
    ax.set_ylabel('Predicted anthesis DOY (day of year)', fontsize=12)
    ax.set_title('Field-scale anthesis timing across the HRW belt, 2019-2024 (v3 — real sowing dates + WES fix)\n'
                 '(out-of-training-window inference using A6 ElasticNet trained on 2014-2017)',
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
