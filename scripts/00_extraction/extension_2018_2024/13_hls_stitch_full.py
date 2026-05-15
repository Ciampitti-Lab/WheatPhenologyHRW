"""Stitch training (2013-2017) and extension (2018-2024) HLS into a
single time-series parquet covering 2013-04-12 → 2024-12-31. Adds the
growing-season coordinate (DOS) and harvest_year for the v2 pipeline.

For winter wheat:
    harvest_year = year if month >= 7 else year + 1
    gs_start     = Jul 1 of (harvest_year - 1)
    DOS          = (date - gs_start).days + 1     # 1..365

Output: data/raw/satellite/extension_2018_2024/hls_full_2013_2024.parquet
"""
from pathlib import Path
import pandas as pd
import numpy as np

EXT = Path('/depot/ciampitti/data/WheatPhenologyHRW/data/raw/satellite/extension_2018_2024')
TRAIN_HLS = '/depot/ciampitti/data/WheatPhenologyHRW/data/processed/buffer/hls_phenology_merged.parquet'
EXT_HLS   = EXT / 'hls_phenology_merged_2018_2024.parquet'
OUT       = EXT / 'hls_full_2013_2024.parquet'

print('=== HLS stitch (training + extension) ===')
train = pd.read_parquet(TRAIN_HLS)
print(f'  training rows: {len(train):,}, '
      f'date range: {pd.to_datetime(train["date"]).min()} → '
      f'{pd.to_datetime(train["date"]).max()}')
ext = pd.read_parquet(EXT_HLS)
print(f'  extension rows: {len(ext):,}, '
      f'date range: {pd.to_datetime(ext["date"]).min()} → '
      f'{pd.to_datetime(ext["date"]).max()}')

# Keep only common columns
common = [c for c in train.columns if c in ext.columns]
print(f'  common columns ({len(common)}): {common}')
train = train[common].copy()
ext = ext[common].copy()

both = pd.concat([train, ext], ignore_index=True)
both['field_id'] = both['field_id'].astype(str)
both['date'] = pd.to_datetime(both['date'])
both = both.sort_values(['field_id', 'date']).reset_index(drop=True)
print(f'  combined rows: {len(both):,}')

# Add growing-season coordinates
both['cy'] = both['date'].dt.year
both['month'] = both['date'].dt.month.astype('int8')
both['harvest_year'] = (both['cy'] + (both['month'] >= 7).astype(int)).astype('int16')
both['gs_start'] = pd.to_datetime((both['harvest_year'] - 1).astype(str) + '-07-01')
both['dos'] = (both['date'] - both['gs_start']).dt.days.astype('int16') + 1
print(f'  harvest_years: {sorted(both["harvest_year"].unique().tolist())}')
print(f'  dos range: {both["dos"].min()} → {both["dos"].max()}')

both = both.drop(columns=['gs_start', 'cy'])
both.to_parquet(OUT, index=False)
print(f'\n→ {OUT}')
print(f'   {len(both):,} rows × {len(both.columns)} cols')
