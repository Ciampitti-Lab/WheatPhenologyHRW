"""Replace raw date features with peak-normalized (relative-to-GCVI-POS)
versions. Unlike script 25 (which ADDS *_rel columns), this script
DROPS the raw date columns so the model is forced to learn relative
timing.

Kept as anchor reference: GCVI_POS (the only absolute date feature that
remains).

Output: features_gs_*_rel_replaced.parquet
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

EXT = Path('/depot/ciampitti/data/WheatPhenologyHRW/data/raw/satellite/extension_2018_2024')

DATE_FEATURES_DOS = [
    'NDVI_POS', 'NDVI_SOS', 'NDVI_LeftShoulder', 'NDVI_greenup_midpoint',
    'EVI_POS',  'EVI_SOS',  'EVI_LeftShoulder',  'EVI_greenup_midpoint',
    'GCVI_SOS', 'GCVI_LeftShoulder', 'GCVI_greenup_midpoint',
    'NDRE_POS', 'NDRE_SOS', 'NDRE_LeftShoulder', 'NDRE_greenup_midpoint',
    'DL_c4_greenup_midpoint', 'DL_c6_senesc_midpoint',
    'dormancy_break_DOS',
]
DATE_FEATURES_DOY = [
    'WE_emergence_doy', 'WE_tillering_doy', 'WE_jointing_doy',
    'WE_flag_leaf_doy', 'WE_boot_doy', 'WE_heading_doy',
    'WE_anthesis_doy', 'WE_maturity_doy',
]


def doy_to_dos(doy):
    if pd.isna(doy):
        return np.nan
    return doy - 181 if doy >= 182 else doy + 184


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='in_path', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.in_path)
    print(f'In shape: {df.shape}')

    if 'GCVI_POS' not in df.columns:
        raise ValueError('GCVI_POS missing')
    anchor = df['GCVI_POS'].astype(float)

    # Replace DOS-encoded date features with their relative versions
    for col in DATE_FEATURES_DOS:
        if col == 'GCVI_POS':
            continue   # keep anchor
        if col in df.columns:
            df[col] = df[col].astype(float) - anchor

    # Replace DOY-encoded WE_* features with relative-to-GCVI-POS-dos
    for col in DATE_FEATURES_DOY:
        if col in df.columns:
            dos_val = df[col].apply(doy_to_dos)
            df[col] = dos_val - anchor

    print(f'Out shape: {df.shape}')
    print(f'\nSample replaced feature distributions:')
    sample = ['NDVI_SOS', 'NDVI_LeftShoulder', 'WE_anthesis_doy', 'WE_flag_leaf_doy',
              'WE_emergence_doy', 'dormancy_break_DOS', 'DL_c4_greenup_midpoint']
    for c in sample:
        if c in df.columns:
            s = df[c].dropna()
            if len(s):
                print(f'  {c:<28} n={len(s):>5}  median={s.median():>+7.1f}  '
                      f'p5..p95=[{s.quantile(0.05):>+7.1f}, {s.quantile(0.95):>+7.1f}]')

    df.to_parquet(args.out, index=False)
    print(f'\n→ {args.out}')


if __name__ == '__main__':
    main()
