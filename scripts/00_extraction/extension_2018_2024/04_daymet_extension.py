"""Daymet backfill for 2018–2024 — all variables in one pass.

Mirrors the existing Daymet downloads (scripts/01_data_prep/03_daymet_temperature.py
and 05_daymet_extra_vars.py) but pulls tmin, tmax, prcp, srad, vp, swe in
a single REST call per field × per phase. Phased to avoid hitting the
Daymet API hard with all 6,120 × 7 years at once.

Usage:
    python 04_daymet_extension.py --years 2018
    python 04_daymet_extension.py --years 2019,2020,2021
    python 04_daymet_extension.py --years 2022,2023,2024

Output: data/raw/satellite/daymet_extension_<years>.csv
Checkpoint: resumable — if it dies, re-run with same args.
"""
import argparse
import io
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

# Reuse field coordinates from the matched phenology parquet — that's
# where (FIELDID, lat, lon) live. The Daymet CSV only carries the
# weather columns, not coordinates.
PHENO_MATCHED = ('/depot/ciampitti/data/WheatPhenologyHRW/data/processed/'
                 'buffer_300m/wheat_hrw_phenology_buffer_matched.parquet')
OUTDIR = Path('/depot/ciampitti/data/WheatPhenologyHRW/data/raw/satellite')

DAYMET_API = 'https://daymet.ornl.gov/single-pixel/api/data'
VARS = 'tmin,tmax,prcp,srad,vp,swe'
MAX_WORKERS = 8
BATCH_SIZE = 100


def fetch(lat, lon, field_id, years):
    url = (f"{DAYMET_API}?lat={lat}&lon={lon}&vars={VARS}"
           f"&years={','.join(str(y) for y in years)}")
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return None
        lines = r.text.strip().split('\n')
        data_lines = [l for l in lines if l and l[0].isdigit()]
        header_line = [l for l in lines if l.startswith('year,')]
        if not data_lines or not header_line:
            return None
        df = pd.read_csv(io.StringIO(header_line[0] + '\n' + '\n'.join(data_lines)))
        df.columns = [c.strip().split(' ')[0] for c in df.columns]
        df['FIELDID'] = field_id
        df = df.rename(columns={'yday': 'doy'})
        df['date'] = pd.to_datetime(
            df['year'].astype(str) + '-' + df['doy'].astype(str), format='%Y-%j')
        return df[['FIELDID', 'date', 'year', 'doy',
                   'tmin', 'tmax', 'prcp', 'srad', 'vp', 'swe']]
    except Exception:
        return None


def load_field_coords():
    """Unique (FIELDID, lat, lon) from the matched phenology parquet."""
    df = pd.read_parquet(PHENO_MATCHED, columns=['FIELDID', 'lat', 'lon'])
    df['FIELDID'] = df['FIELDID'].astype(str)
    coords = (df.groupby('FIELDID', as_index=False)
                .agg(lat=('lat', 'median'), lon=('lon', 'median')))
    return coords


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', required=True,
                    help='Comma-separated calendar years, e.g. 2018 or 2019,2020,2021')
    args = ap.parse_args()
    years = [int(y) for y in args.years.split(',')]
    tag = '_'.join(str(y) for y in years)
    out_path = OUTDIR / f'daymet_extension_{tag}.csv'
    chk_path = OUTDIR / f'daymet_extension_{tag}_checkpoint.csv'

    coords = load_field_coords()
    print(f'Years: {years}')
    print(f'Fields: {len(coords):,}')
    print(f'Output: {out_path}')

    done = set()
    if chk_path.exists():
        done = set(pd.read_csv(chk_path, usecols=['FIELDID'])['FIELDID'].astype(str))
        print(f'Resuming — {len(done):,} fields already done')

    remaining = coords[~coords['FIELDID'].isin(done)].reset_index(drop=True)
    if len(remaining) == 0:
        print('All fields already done — finalising output…')
        pd.read_csv(chk_path).to_csv(out_path, index=False)
        return

    t0 = time.time()
    buffer = []
    for start in range(0, len(remaining), BATCH_SIZE):
        batch = remaining.iloc[start:start + BATCH_SIZE]
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = [ex.submit(fetch, r.lat, r.lon, r.FIELDID, years)
                       for r in batch.itertuples()]
            for f in as_completed(futures):
                res = f.result()
                if res is not None:
                    buffer.append(res)

        elapsed = time.time() - t0
        seen = start + len(batch)
        rate = seen / elapsed * 60 if elapsed > 0 else 0
        print(f'  [{seen}/{len(remaining)}]  {rate:.0f} fields/min')

        if len(buffer) >= 500 or seen >= len(remaining):
            df_b = pd.concat(buffer, ignore_index=True)
            if chk_path.exists():
                df_b = pd.concat([pd.read_csv(chk_path), df_b], ignore_index=True)
            df_b.to_csv(chk_path, index=False)
            buffer = []

    # Finalise: copy checkpoint to canonical output
    df_final = pd.read_csv(chk_path)
    df_final.to_csv(out_path, index=False)
    print(f'\nDone — {len(df_final):,} rows, {df_final["FIELDID"].nunique()} fields')
    print(f'Saved: {out_path}')


if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f'Wall time: {(time.time() - t0) / 60:.1f} min')
