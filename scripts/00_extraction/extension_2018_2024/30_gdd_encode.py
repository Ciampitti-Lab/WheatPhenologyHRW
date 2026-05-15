"""Thermal-time (GDD) encoding of date features.

For each (field_id, harvest_year), compute cumulative GDD over the
growing season (DOS 1-365) and convert each date-based feature from
"day-of-season" to "GDD accumulated by that day".

Rationale: TX and CO fields with very different DOS for the same
physiological event may accumulate similar GDD by that event. GDD-space
features should therefore have lower cross-field variance and be more
informative for the model.

REPLACE strategy: original date features overwritten with their GDD
equivalents (model is forced to learn from thermal-time coordinates).

Output: features_gs_*_gdd.parquet
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

EXT = Path('/depot/ciampitti/data/WheatPhenologyHRW/data/raw/satellite/extension_2018_2024')
DAYMET_PATH = EXT / 'daymet_full_2013_2024.parquet'

# Date features in DOS coordinate (already growing-season)
DATE_FEATURES_DOS = [
    'NDVI_POS', 'NDVI_SOS', 'NDVI_LeftShoulder', 'NDVI_greenup_midpoint',
    'EVI_POS',  'EVI_SOS',  'EVI_LeftShoulder',  'EVI_greenup_midpoint',
    'GCVI_POS', 'GCVI_SOS', 'GCVI_LeftShoulder', 'GCVI_greenup_midpoint',
    'NDRE_POS', 'NDRE_SOS', 'NDRE_LeftShoulder', 'NDRE_greenup_midpoint',
    'DL_c4_greenup_midpoint', 'DL_c6_senesc_midpoint',
    'dormancy_break_DOS',
]

# Date features in calendar DOY (WES outputs); need DOY→DOS conversion
DATE_FEATURES_DOY = [
    'WE_emergence_doy', 'WE_tillering_doy', 'WE_jointing_doy',
    'WE_flag_leaf_doy', 'WE_boot_doy', 'WE_heading_doy',
    'WE_anthesis_doy', 'WE_maturity_doy',
]


def doy_to_dos(doy):
    """Calendar DOY → growing-season DOS."""
    if pd.isna(doy):
        return np.nan
    return doy - 181 if doy >= 182 else doy + 184


def build_gdd_lookup(wx_fy):
    """For one field-year's weather, return GDD_cum array indexed 1..365.

    GDD = max(0, (Tmin + Tmax)/2 - 0) (Method 1 GDD, base 0°C)
    Cumulative from DOS 1 (1 July of harvest_year - 1).
    """
    wx_fy = wx_fy.sort_values('dos')
    t_mean = (wx_fy['Tmin'].values + wx_fy['Tmax'].values) / 2.0
    gdd = np.maximum(0.0, t_mean)
    gdd_cum = np.cumsum(gdd)
    # Build daily lookup: index 0 = DOS 1, index 364 = DOS 365
    lookup = np.full(365, np.nan)
    for d, g in zip(wx_fy['dos'].values, gdd_cum):
        if 1 <= d <= 365:
            lookup[int(d) - 1] = g
    # Forward-fill any gaps
    last_good = 0.0
    for i in range(365):
        if np.isnan(lookup[i]):
            lookup[i] = last_good
        else:
            last_good = lookup[i]
    return lookup


def gdd_at_dos(lookup, dos_value):
    if pd.isna(dos_value):
        return np.nan
    d = int(round(dos_value))
    if d < 1: return 0.0
    if d > 365: d = 365
    return float(lookup[d - 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='in_path', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    print(f'Loading features: {args.in_path}')
    feat = pd.read_parquet(args.in_path)
    feat['field_id'] = feat['field_id'].astype(str)
    feat['year'] = feat['year'].astype(int)
    print(f'  shape: {feat.shape}')

    print(f'Loading Daymet (filtered to {feat["field_id"].nunique():,} fields)...')
    valid_fields = set(feat['field_id'].unique())
    valid_years = sorted(feat['year'].unique().tolist())
    hy_min = min(valid_years)
    hy_max = max(valid_years)
    date_lo = pd.Timestamp(f'{hy_min - 1}-07-01')
    date_hi = pd.Timestamp(f'{hy_max}-07-31')

    wx_t = pq.read_table(
        DAYMET_PATH,
        columns=['FIELDID','date','Tmin','Tmax','harvest_year','dos'],
        filters=[('date', '>=', date_lo), ('date', '<=', date_hi)])
    wx = wx_t.to_pandas()
    del wx_t
    wx = wx.rename(columns={'FIELDID': 'field_id'})
    wx['field_id'] = wx['field_id'].astype(str)
    wx = wx[wx['field_id'].isin(valid_fields)].copy()
    print(f'  Daymet rows: {len(wx):,}')

    # Index Daymet by (field_id, harvest_year) for fast lookup
    print('Building per-field-year GDD lookups...')
    wx_grouped = dict(tuple(wx.groupby(['field_id', 'harvest_year'])))
    n_groups = len(wx_grouped)
    print(f'  {n_groups:,} field-year groups indexed')

    # Replace each date feature with GDD-encoded version
    print('\nReplacing date features with GDD-encoded versions...')
    counts = {}
    for col in DATE_FEATURES_DOS:
        if col in feat.columns:
            counts[col] = 0
    for col in DATE_FEATURES_DOY:
        if col in feat.columns:
            counts[col] = 0

    for idx, row in feat.iterrows():
        key = (row['field_id'], row['year'])
        if key not in wx_grouped:
            continue
        lookup = build_gdd_lookup(wx_grouped[key])

        # DOS-encoded features
        for col in DATE_FEATURES_DOS:
            if col in feat.columns and not pd.isna(row[col]):
                feat.at[idx, col] = gdd_at_dos(lookup, row[col])
                counts[col] += 1

        # DOY-encoded features (WES) - convert to DOS first
        for col in DATE_FEATURES_DOY:
            if col in feat.columns and not pd.isna(row[col]):
                dos = doy_to_dos(row[col])
                feat.at[idx, col] = gdd_at_dos(lookup, dos)
                counts[col] += 1

    print(f'\nFeature conversion counts:')
    for col, c in counts.items():
        print(f'  {col:<28}  {c:>5} rows converted')

    print('\nSample distributions (new GDD values):')
    for col in ['NDVI_POS', 'NDVI_SOS', 'WE_anthesis_doy', 'WE_flag_leaf_doy', 'GCVI_POS']:
        if col in feat.columns:
            s = feat[col].dropna()
            if len(s):
                print(f'  {col:<28}  n={len(s):>5}  median={s.median():>8.1f}  p5..p95=[{s.quantile(0.05):>8.1f}, {s.quantile(0.95):>8.1f}]')

    feat.to_parquet(args.out, index=False)
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
