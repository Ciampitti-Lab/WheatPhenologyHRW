"""Build canonical Daymet + LST parquets covering 2013-2024 (training +
extension), to feed the growing-season feature pipeline.

Memory-conscious: streams training CSVs through chunked reads,
downcasts to float32 immediately, writes intermediate parquets,
then concats with the extension parquets via pyarrow tables (no
single-pass full DataFrame).

Outputs:
    extension_2018_2024/daymet_full_2013_2024.parquet
    extension_2018_2024/modis_lst_2012_2024.parquet
"""
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

EXT = Path('/depot/ciampitti/data/WheatPhenologyHRW/data/raw/satellite/extension_2018_2024')
TRAIN_DAYMET = '/depot/ciampitti/data/WheatPhenologyHRW/data/raw/satellite/daymet_daily_weather_full.csv'
TRAIN_LST    = '/depot/ciampitti/data/WheatPhenologyHRW/data/raw/satellite/modis_lst_buffer.csv'


def add_dos(df):
    df['cy']           = df['date'].dt.year.astype('int16')
    df['month']        = df['date'].dt.month.astype('int8')
    df['harvest_year'] = (df['cy'] + (df['month'] >= 7).astype(int)).astype('int16')
    df['gs_start']     = pd.to_datetime((df['harvest_year'] - 1).astype(str) + '-07-01')
    df['dos']          = ((df['date'] - df['gs_start']).dt.days + 1).astype('int16')
    return df.drop(columns=['gs_start', 'cy'])


# ─── Daymet ─────────────────────────────────────────────────────────────────
print('=== Daymet 2013-2024 (chunked, low-memory) ===')

writer = None
out_d = EXT / 'daymet_full_2013_2024.parquet'

# Stream training CSV
print('  Streaming training CSV...')
train_total = 0
for i, chunk in enumerate(pd.read_csv(TRAIN_DAYMET, dtype={'FIELDID': str},
                                      chunksize=2_000_000)):
    chunk['date'] = pd.to_datetime(chunk['date'], format='mixed')
    if 'tmin' in chunk.columns:
        chunk = chunk.rename(columns={'tmin': 'Tmin', 'tmax': 'Tmax'})
    for c in ['Tmin','Tmax','prcp','srad','vp','swe','vpd','ptq']:
        if c in chunk.columns:
            chunk[c] = chunk[c].astype('float32')
    chunk = add_dos(chunk)
    keep_cols = ['FIELDID','date','Tmin','Tmax','prcp','srad','vp','swe','vpd','ptq',
                 'month','harvest_year','dos']
    keep_cols = [c for c in keep_cols if c in chunk.columns]
    chunk = chunk[keep_cols]
    table = pa.Table.from_pandas(chunk, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(out_d, table.schema, compression='snappy')
    else:
        # align schema if needed
        chunk = chunk.reindex(columns=writer.schema.names)
        table = pa.Table.from_pandas(chunk, schema=writer.schema, preserve_index=False)
    writer.write_table(table)
    train_total += len(chunk)
    print(f'    chunk {i}: +{len(chunk):,} rows (total training: {train_total:,})')

# Now stream extension parquet
print('  Streaming extension parquet...')
ext = pd.read_parquet(EXT / 'daymet_full_2018_2024.parquet')
for c in ['Tmin','Tmax','prcp','srad','vp','swe','vpd','ptq']:
    if c in ext.columns:
        ext[c] = ext[c].astype('float32')
ext['date'] = pd.to_datetime(ext['date'])
ext = add_dos(ext)
ext = ext.reindex(columns=writer.schema.names)
table = pa.Table.from_pandas(ext, schema=writer.schema, preserve_index=False)
writer.write_table(table)
writer.close()
print(f'  → {out_d}: {train_total + len(ext):,} rows')

# ─── MODIS LST ──────────────────────────────────────────────────────────────
print('\n=== MODIS LST 2012-2024 (chunked) ===')
out_l = EXT / 'modis_lst_2012_2024.parquet'
writer_l = None
train_total = 0
for i, chunk in enumerate(pd.read_csv(TRAIN_LST, dtype={'FIELDID': str},
                                      chunksize=2_000_000)):
    chunk['date'] = pd.to_datetime(chunk['date'], format='mixed')
    for c in ['lst_day_C', 'lst_night_C']:
        if c in chunk.columns:
            chunk[c] = chunk[c].astype('float32')
    chunk = add_dos(chunk)
    keep_cols = ['FIELDID','date','lst_day_C','lst_night_C','month','harvest_year','dos']
    chunk = chunk[[c for c in keep_cols if c in chunk.columns]]
    table = pa.Table.from_pandas(chunk, preserve_index=False)
    if writer_l is None:
        writer_l = pq.ParquetWriter(out_l, table.schema, compression='snappy')
    else:
        chunk = chunk.reindex(columns=writer_l.schema.names)
        table = pa.Table.from_pandas(chunk, schema=writer_l.schema, preserve_index=False)
    writer_l.write_table(table)
    train_total += len(chunk)
    print(f'    chunk {i}: +{len(chunk):,} rows (total training: {train_total:,})')

ext_l = pd.read_parquet(EXT / 'modis_lst_2018_2024.parquet')
for c in ['lst_day_C', 'lst_night_C']:
    if c in ext_l.columns:
        ext_l[c] = ext_l[c].astype('float32')
ext_l['date'] = pd.to_datetime(ext_l['date'])
ext_l = add_dos(ext_l)
ext_l = ext_l.reindex(columns=writer_l.schema.names)
table = pa.Table.from_pandas(ext_l, schema=writer_l.schema, preserve_index=False)
writer_l.write_table(table)
writer_l.close()
print(f'  → {out_l}: {train_total + len(ext_l):,} rows')
