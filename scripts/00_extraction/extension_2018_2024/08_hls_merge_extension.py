"""Merge HLS L8 + S2 for 2018-2024, mirroring scripts/01_data_prep/02_hls_merge.ipynb.

Replicates the production merge:
    - concat L8 + S2 by FIELDID/date
    - normalise FIELDID → field_id
    - apply the ×10000 band correction the original notebook used (GEE
      reflectance was stored as int16 ×0.0001; we put it back on the
      integer scale so the EVI/MTCI constants are dimensionally right)
    - recompute NDVI/EVI/GCVI/NDRE/CIre/MTCI from corrected bands
    - attach lat/lon from the matched phenology parquet (same 6,120
      fields as 2013-2017, no labels needed for extension years)

Output: data/raw/satellite/extension_2018_2024/hls_phenology_merged_2018_2024.parquet
"""
from pathlib import Path
import numpy as np
import pandas as pd

EXT = Path('/depot/ciampitti/data/WheatPhenologyHRW/data/raw/satellite/extension_2018_2024')
PHENO_MATCHED = ('/depot/ciampitti/data/WheatPhenologyHRW/data/processed/buffer_300m/'
                 'wheat_hrw_phenology_buffer_matched.parquet')
OUT = EXT / 'hls_phenology_merged_2018_2024.parquet'


def main():
    print('=== Loading per-source extension parquets ===')
    df_l8 = pd.read_parquet(EXT / 'hls_l8_2018_2024.parquet')
    df_s2 = pd.read_parquet(EXT / 'hls_s2_2018_2024.parquet')
    print(f'  L8: {len(df_l8):,} rows, {df_l8["FIELDID"].nunique()} fields')
    print(f'  S2: {len(df_s2):,} rows, {df_s2["FIELDID"].nunique()} fields')

    df = pd.concat([df_l8, df_s2], ignore_index=True)
    df = df.rename(columns={'FIELDID': 'field_id'})
    df['field_id'] = df['field_id'].astype(str)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['doy'] = df['date'].dt.dayofyear
    df = df.sort_values(['field_id', 'date']).reset_index(drop=True)
    print(f'  Combined: {len(df):,} rows')

    # Same band-scale correction as the original merge — bands come out
    # of GEE in [0,1], we put them back on the int16 scale so EVI/MTCI
    # constants line up with the training-set features.
    band_cols = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2']
    re_cols   = ['RE1', 'RE2', 'RE3']
    for col in band_cols + re_cols:
        if col in df.columns:
            df[col] = df[col] * 10000

    # Recompute indices on corrected bands
    df['NDVI'] = (df['NIR'] - df['Red']) / (df['NIR'] + df['Red'])
    df['EVI']  = 2.5 * (df['NIR'] - df['Red']) / (df['NIR'] + 6*df['Red']
                                                  - 7.5*df['Blue'] + 1)
    df['GCVI'] = df['NIR'] / df['Green'] - 1
    if 'RE1' in df.columns:
        m = df['RE1'].notna()
        df.loc[m, 'NDRE'] = ((df.loc[m, 'NIR'] - df.loc[m, 'RE1'])
                             / (df.loc[m, 'NIR'] + df.loc[m, 'RE1']))
        df.loc[m, 'CIre'] = df.loc[m, 'NIR'] / df.loc[m, 'RE1'] - 1
        df.loc[m, 'MTCI'] = ((df.loc[m, 'RE2'] - df.loc[m, 'RE1'])
                             / (df.loc[m, 'RE1'] - df.loc[m, 'Red']))

    # Attach lat/lon (same fields as training set, coords don't change)
    print('\n=== Attaching lat/lon from matched phenology parquet ===')
    pheno = pd.read_parquet(PHENO_MATCHED, columns=['FIELDID', 'lat', 'lon'])
    pheno = pheno.rename(columns={'FIELDID': 'field_id'})
    pheno['field_id'] = pheno['field_id'].astype(str)
    coords = pheno.groupby('field_id', as_index=False).agg(
        lat=('lat', 'median'), lon=('lon', 'median'))
    df = df.merge(coords, on='field_id', how='left')
    print(f'  fields with coords: {df["lat"].notna().sum() / len(df) * 100:.1f}%')

    df.to_parquet(OUT, index=False)
    print(f'\n→ {OUT}')
    print(f'   {len(df):,} rows, {df["field_id"].nunique()} fields, '
          f'years {sorted(df["year"].unique().tolist())}')


if __name__ == '__main__':
    main()
