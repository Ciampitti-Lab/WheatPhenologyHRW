"""Add temporal phase-alignment features to V2 parquets.

For each (field, harvest_year), anchor all date-based features to the
GCVI peak-of-season (GCVI_POS) and add relative-to-peak versions:

    NDVI_SOS_rel        = NDVI_SOS - GCVI_POS_dos
    NDVI_LeftShoulder_rel
    WE_anthesis_doy_rel = WE_anthesis_dos - GCVI_POS_dos
    ...

WE outputs are in calendar DOY (return_dos=False in pipeline); they
must be converted to DOS before subtraction. Conversion:
    if DOY ≥ 182 (Jul-Dec) → year Y-1, DOS = DOY - 181
    if DOY <  182 (Jan-Jun) → year Y,   DOS = DOY + 184

Output:
    features_gs_train_2014_2017_rel.parquet
    features_gs_extension_2019_2024_rel.parquet
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

EXT = Path('/depot/ciampitti/data/WheatPhenologyHRW/data/raw/satellite/extension_2018_2024')

# Features that are dates in DOS coordinate (already DOS-encoded)
DATE_FEATURES_DOS = [
    'NDVI_POS', 'NDVI_SOS', 'NDVI_LeftShoulder', 'NDVI_greenup_midpoint',
    'EVI_POS',  'EVI_SOS',  'EVI_LeftShoulder',  'EVI_greenup_midpoint',
    'GCVI_SOS', 'GCVI_LeftShoulder', 'GCVI_greenup_midpoint',  # skip GCVI_POS (anchor)
    'NDRE_POS', 'NDRE_SOS', 'NDRE_LeftShoulder', 'NDRE_greenup_midpoint',
    'DL_c4_greenup_midpoint', 'DL_c6_senesc_midpoint',
    'dormancy_break_DOS',
]

# Features that are calendar DOY (need conversion to DOS first)
DATE_FEATURES_DOY = [
    'WE_emergence_doy', 'WE_tillering_doy', 'WE_jointing_doy',
    'WE_flag_leaf_doy', 'WE_boot_doy', 'WE_heading_doy',
    'WE_anthesis_doy', 'WE_maturity_doy',
]


def doy_to_dos(doy_value):
    """Convert calendar DOY → DOS (Jul 1 of year-1 anchor).

    Heuristic: WE outputs in Jul-Dec of year-1 have DOY ≥ 182.
    """
    if pd.isna(doy_value):
        return np.nan
    if doy_value >= 182:
        # Year Y-1 (Jul-Dec)
        return doy_value - 181
    else:
        # Year Y (Jan-Jun) — add 184 days (Jul-Dec span)
        return doy_value + 184


def add_phase_alignment(df):
    """Add *_rel columns anchored at GCVI_POS (DOS-based)."""
    if 'GCVI_POS' not in df.columns:
        raise ValueError('GCVI_POS not in dataframe — cannot anchor.')
    anchor = df['GCVI_POS'].astype(float)
    n_added = 0

    # DOS-encoded features → simple subtraction
    for col in DATE_FEATURES_DOS:
        if col in df.columns:
            df[f'{col}_rel'] = df[col].astype(float) - anchor
            n_added += 1

    # DOY-encoded features → convert to DOS, then subtract
    for col in DATE_FEATURES_DOY:
        if col in df.columns:
            dos_value = df[col].apply(doy_to_dos)
            df[f'{col.replace("_doy", "")}_rel'] = dos_value - anchor
            n_added += 1

    return df, n_added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='in_path', required=True,
                    help='Input features parquet')
    ap.add_argument('--out', required=True, help='Output parquet')
    args = ap.parse_args()

    print(f'Loading {args.in_path}...')
    df = pd.read_parquet(args.in_path)
    print(f'  shape before: {df.shape}')

    df, n_added = add_phase_alignment(df)
    print(f'  added {n_added} relative-to-GCVI-POS features')
    print(f'  shape after: {df.shape}')

    # Sanity: GCVI_POS_rel should be 0 (skipped); show range of others
    print(f'\nSample _rel feature distributions:')
    rel_cols = [c for c in df.columns if c.endswith('_rel')]
    for c in rel_cols[:8]:
        s = df[c].dropna()
        print(f'  {c:<35}  n={len(s):>5}  median={s.median():>+7.1f}  p5..p95=[{s.quantile(0.05):>+6.1f}, {s.quantile(0.95):>+6.1f}]')

    df.to_parquet(args.out, index=False)
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
