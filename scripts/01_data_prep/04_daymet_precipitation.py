"""
02d - Download Daymet Daily Precipitation (1km) via REST API

Downloads prcp for all field locations (2013-2017) and merges with existing
daymet_daily_temperature.csv -> daymet_daily_weather.csv (combined output).

Usage:
    python 02d_daymet_precip_download.py
"""

import pandas as pd
import numpy as np
import requests
import time
import os
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

# === CONFIG ===
PHENOLOGY_PATH = 'data/processed/buffer_300m/wheat_hrw_phenology_buffer_matched.parquet'
TEMP_CSV = 'data/raw/satellite/daymet_daily_temperature.csv'
OUTPUT_CSV = 'data/raw/satellite/daymet_daily_weather.csv'
CHECKPOINT_CSV = 'data/raw/satellite/daymet_prcp_checkpoint.csv'

DAYMET_API = 'https://daymet.ornl.gov/single-pixel/api/data'
START_YEAR = 2013
END_YEAR = 2017
VARS = 'prcp'
MAX_WORKERS = 8


def fetch_prcp(lat, lon, field_id, start_year, end_year):
    url = f"{DAYMET_API}?lat={lat}&lon={lon}&vars={VARS}&years={','.join(str(y) for y in range(start_year, end_year+1))}"
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
        df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['doy'].astype(str), format='%Y-%j')
        return df[['FIELDID', 'date', 'year', 'doy', 'prcp']]
    except Exception:
        return None


def main():
    df_pheno = pd.read_parquet(PHENOLOGY_PATH)
    fields = df_pheno.groupby('FIELDID').agg(lat=('lat', 'median'), lon=('lon', 'median')).reset_index()
    print(f"Total fields: {len(fields)}")

    done_ids = set()
    if os.path.exists(CHECKPOINT_CSV):
        done_ids = set(pd.read_csv(CHECKPOINT_CSV)['FIELDID'].astype(str).unique())
        print(f"Checkpoint: {len(done_ids)} fields already done")

    remaining = fields[~fields['FIELDID'].astype(str).isin(done_ids)]
    print(f"Remaining: {len(remaining)} fields")

    if len(remaining) > 0:
        all_results = []
        batch_size = 100
        t0 = time.time()

        for batch_start in range(0, len(remaining), batch_size):
            batch = remaining.iloc[batch_start:batch_start + batch_size]
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                futures = [ex.submit(fetch_prcp, r['lat'], r['lon'], r['FIELDID'], START_YEAR, END_YEAR)
                           for _, r in batch.iterrows()]
                for f in as_completed(futures):
                    res = f.result()
                    if res is not None:
                        all_results.append(res)

            done = batch_start + len(batch)
            rate = done / (time.time() - t0) * 60
            print(f"  [{done}/{len(remaining)}] {rate:.0f} fields/min")

            if len(all_results) >= 500 or done >= len(remaining):
                df_b = pd.concat(all_results, ignore_index=True)
                if os.path.exists(CHECKPOINT_CSV):
                    df_b = pd.concat([pd.read_csv(CHECKPOINT_CSV), df_b], ignore_index=True)
                df_b.to_csv(CHECKPOINT_CSV, index=False)
                all_results = []

    # Merge with existing tmin/tmax
    print("\nMerging prcp with existing tmin/tmax...")
    df_temp = pd.read_csv(TEMP_CSV)
    df_prcp = pd.read_csv(CHECKPOINT_CSV)
    df_temp['FIELDID'] = df_temp['FIELDID'].astype(str)
    df_prcp['FIELDID'] = df_prcp['FIELDID'].astype(str)
    df = df_temp.merge(df_prcp[['FIELDID', 'date', 'prcp']], on=['FIELDID', 'date'], how='left')
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved: {OUTPUT_CSV} ({len(df):,} rows)")
    print(f"prcp missing: {df['prcp'].isna().sum()}")


if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f"\nTotal: {(time.time()-t0)/60:.1f} min")
