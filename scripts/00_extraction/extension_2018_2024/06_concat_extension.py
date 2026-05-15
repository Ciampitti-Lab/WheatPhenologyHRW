"""Concat 2018–2024 extension CSVs into per-source parquet files.

Inputs (from gdown bulk-download):
    extension_2018_2024/
        buffer_l8_timeseries_<YYYY>.csv      (7 files)
        buffer_s2_timeseries_<YYYY>.csv      (7 files)
        modis_lst_buffer_<YYYY>.csv          (7 files)
        cdl_wheat_fraction_2018_2024.csv     (1 file, all years)

Outputs (parquet for fast downstream loading):
    hls_l8_2018_2024.parquet
    hls_s2_2018_2024.parquet
    modis_lst_2018_2024.parquet
    cdl_wheat_fraction_2018_2024.parquet

Usage:
    python 06_concat_extension.py
"""
from pathlib import Path
import pandas as pd

DIR = Path('/depot/ciampitti/data/WheatPhenologyHRW/data/raw/satellite/extension_2018_2024')


def concat_yearly(prefix: str, out_name: str):
    files = sorted(DIR.glob(f'{prefix}_*.csv'))
    print(f'\n=== {prefix} ===  ({len(files)} files)')
    parts = []
    for f in files:
        df = pd.read_csv(f, dtype={'FIELDID': str})
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['year'] = df['date'].dt.year
        print(f'  {f.name:<45} {len(df):>10,} rows')
        parts.append(df)
    big = pd.concat(parts, ignore_index=True)
    out = DIR / out_name
    big.to_parquet(out, index=False)
    print(f'  -> {out.name}: {len(big):,} rows, {big["FIELDID"].nunique():,} fields')


def main():
    concat_yearly('buffer_l8_timeseries',  'hls_l8_2018_2024.parquet')
    concat_yearly('buffer_s2_timeseries',  'hls_s2_2018_2024.parquet')
    concat_yearly('modis_lst_buffer',      'modis_lst_2018_2024.parquet')

    # CDL is already a single file but convert to parquet for consistency
    print('\n=== CDL ===')
    cdl = pd.read_csv(DIR / 'cdl_wheat_fraction_2018_2024.csv',
                      dtype={'FIELDID': str})
    print(f'  rows: {len(cdl):,}  fields: {cdl["FIELDID"].nunique():,}'
          f'  wheat-frac non-null: {cdl["wheat"].notna().sum():,}')
    cdl.to_parquet(DIR / 'cdl_wheat_fraction_2018_2024.parquet', index=False)
    print(f'  -> cdl_wheat_fraction_2018_2024.parquet')


if __name__ == '__main__':
    main()
