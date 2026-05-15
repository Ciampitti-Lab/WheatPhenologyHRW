"""Stitch training-set 2017 Daymet/LST onto the extension 2018-2024
files so that harvest_year=2018 has the missing fall-2017 half of its
growing season available.
"""
from pathlib import Path
import pandas as pd

EXT = Path('/depot/ciampitti/data/WheatPhenologyHRW/data/raw/satellite/extension_2018_2024')
TRAIN_DAYMET = '/depot/ciampitti/data/WheatPhenologyHRW/data/raw/satellite/daymet_daily_weather_full.csv'
TRAIN_LST    = '/depot/ciampitti/data/WheatPhenologyHRW/data/raw/satellite/modis_lst_buffer.csv'

# ─── Daymet ─────────────────────────────────────────────────────────────────
print('=== Daymet stitch (training Jul-Dec 2017 + extension 2018-2024) ===')
ext = pd.read_parquet(EXT / 'daymet_full_2018_2024.parquet')
ext['FIELDID'] = ext['FIELDID'].astype(str) if 'FIELDID' in ext.columns else ext['field_id'].astype(str)
print(f'  extension rows: {len(ext):,}')

train_chunks = []
for chunk in pd.read_csv(TRAIN_DAYMET, dtype={'FIELDID': str}, chunksize=500_000):
    chunk['date'] = pd.to_datetime(chunk['date'], format='mixed')
    chunk = chunk[(chunk['date'] >= '2017-07-01') & (chunk['date'] <= '2017-12-31')]
    if len(chunk):
        train_chunks.append(chunk)
train_2017_h2 = pd.concat(train_chunks, ignore_index=True)
print(f'  training Jul-Dec 2017 rows: {len(train_2017_h2):,}')

# Standardise column names — the training CSV may use lowercase tmin/tmax
if 'tmin' in train_2017_h2.columns:
    train_2017_h2 = train_2017_h2.rename(columns={'tmin':'Tmin', 'tmax':'Tmax'})
# Match the extension parquet's column set
common = [c for c in ext.columns if c in train_2017_h2.columns]
print(f'  common columns kept: {common}')

stitched = pd.concat([train_2017_h2[common], ext[common]], ignore_index=True)
stitched = stitched.sort_values(['FIELDID','date']).reset_index(drop=True)
out_d = EXT / 'daymet_full_2017_2024.parquet'
stitched.to_parquet(out_d, index=False)
print(f'  → {out_d}: {len(stitched):,} rows '
      f'(spans {stitched["date"].min()} → {stitched["date"].max()})')

# ─── MODIS LST ──────────────────────────────────────────────────────────────
print('\n=== MODIS LST stitch (training Jul-Dec 2017 + extension 2018-2024) ===')
ext_l = pd.read_parquet(EXT / 'modis_lst_2018_2024.parquet')
ext_l['FIELDID'] = ext_l['FIELDID'].astype(str)
print(f'  extension rows: {len(ext_l):,}')

lst_chunks = []
for chunk in pd.read_csv(TRAIN_LST, dtype={'FIELDID': str}, chunksize=500_000):
    chunk['date'] = pd.to_datetime(chunk['date'], format='mixed')
    chunk = chunk[(chunk['date'] >= '2017-07-01') & (chunk['date'] <= '2017-12-31')]
    if len(chunk):
        lst_chunks.append(chunk)
train_l_2017_h2 = pd.concat(lst_chunks, ignore_index=True)
print(f'  training Jul-Dec 2017 rows: {len(train_l_2017_h2):,}')

common_l = [c for c in ext_l.columns if c in train_l_2017_h2.columns]
print(f'  common columns kept: {common_l}')

stitched_l = pd.concat([train_l_2017_h2[common_l], ext_l[common_l]], ignore_index=True)
stitched_l = stitched_l.sort_values(['FIELDID','date']).reset_index(drop=True)
out_l = EXT / 'modis_lst_2017_2024.parquet'
stitched_l.to_parquet(out_l, index=False)
print(f'  → {out_l}: {len(stitched_l):,} rows '
      f'(spans {stitched_l["date"].min()} → {stitched_l["date"].max()})')

# ─── HLS ────────────────────────────────────────────────────────────────────
# For HLS we use calendar-year smoothing, so 2018 extension HLS alone is
# sufficient for harvest_year=2018. No stitch needed there.

# ─── Build 2018-only valid field-years ──────────────────────────────────────
v = pd.read_parquet(EXT / 'valid_field_years_2018_2024.parquet')
v_2018 = v[v['harvest_year'] == 2018].copy()
v_2018.to_parquet(EXT / 'valid_field_years_2018_only.parquet')
print(f'\n→ {EXT / "valid_field_years_2018_only.parquet"}: '
      f'{len(v_2018):,} field-years for harvest_year=2018')
