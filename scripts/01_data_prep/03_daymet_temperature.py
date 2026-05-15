"""
02c - Download Daymet Daily Temperature (1km) via REST API
No GEE needed. Direct download from ORNL DAAC.

Usage:
    python 02c_daymet_download.py

Downloads tmin/tmax for all field locations, 2013-2017.
Output: daymet_daily_temperature.csv
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
OUTPUT_PATH = 'data/raw/satellite/daymet_daily_temperature.csv'
CHECKPOINT_PATH = 'data/raw/satellite/daymet_checkpoint.csv'

DAYMET_API = 'https://daymet.ornl.gov/single-pixel/api/data'
START_YEAR = 2013
END_YEAR = 2017
VARS = 'tmin,tmax'
MAX_WORKERS = 8  # parallel requests


def fetch_daymet_point(lat, lon, field_id, start_year, end_year):
    """Fetch Daymet tmin/tmax for a single point, all years."""
    url = f"{DAYMET_API}?lat={lat}&lon={lon}&vars={VARS}&years={','.join(str(y) for y in range(start_year, end_year+1))}"

    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200:
            return None

        # Parse CSV response (skip header lines starting with non-numeric)
        lines = r.text.strip().split('\n')
        data_lines = [l for l in lines if l and l[0].isdigit()]
        header_line = [l for l in lines if l.startswith('year,')]

        if not data_lines or not header_line:
            return None

        df = pd.read_csv(io.StringIO(header_line[0] + '\n' + '\n'.join(data_lines)))
        df.columns = [c.strip().split(' ')[0] for c in df.columns]  # clean column names
        df['FIELDID'] = field_id
        df = df.rename(columns={'yday': 'doy'})

        # Create date from year + doy
        df['date'] = pd.to_datetime(df['year'].astype(str) + '-' + df['doy'].astype(str), format='%Y-%j')

        return df[['FIELDID', 'date', 'year', 'doy', 'tmin', 'tmax']]

    except Exception as e:
        return None


def main():
    # Load field locations
    df = pd.read_parquet(PHENOLOGY_PATH)
    fields = df.groupby('FIELDID').agg(
        lat=('lat', 'median'),
        lon=('lon', 'median')
    ).reset_index()

    print(f"Total fields: {len(fields)}")

    # Check for checkpoint (resume from where we left off)
    done_ids = set()
    if os.path.exists(CHECKPOINT_PATH):
        df_done = pd.read_csv(CHECKPOINT_PATH)
        done_ids = set(df_done['FIELDID'].astype(str).unique())
        print(f"Checkpoint found: {len(done_ids)} fields already done")

    remaining = fields[~fields['FIELDID'].astype(str).isin(done_ids)]
    print(f"Remaining: {len(remaining)} fields")

    if len(remaining) == 0:
        print("All done!")
        # Combine checkpoint into final output
        df_final = pd.read_csv(CHECKPOINT_PATH)
        df_final.to_csv(OUTPUT_PATH, index=False)
        print(f"Saved to {OUTPUT_PATH}")
        return

    # Download in parallel batches
    all_results = []
    batch_size = 100
    total = len(remaining)

    for batch_start in range(0, total, batch_size):
        batch = remaining.iloc[batch_start:batch_start + batch_size]

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {}
            for _, row in batch.iterrows():
                f = executor.submit(
                    fetch_daymet_point,
                    row['lat'], row['lon'], row['FIELDID'],
                    START_YEAR, END_YEAR
                )
                futures[f] = row['FIELDID']

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    all_results.append(result)

        # Progress
        done = batch_start + len(batch)
        rate = done / (time.time() - t0) * 60 if 'time' in dir() else 0
        print(f"  [{done}/{total}] {len(all_results)} fields downloaded")

        # Save checkpoint every 500 fields
        if len(all_results) >= 500 or done >= total:
            df_batch = pd.concat(all_results, ignore_index=True)
            if os.path.exists(CHECKPOINT_PATH):
                df_existing = pd.read_csv(CHECKPOINT_PATH)
                df_batch = pd.concat([df_existing, df_batch], ignore_index=True)
            df_batch.to_csv(CHECKPOINT_PATH, index=False)
            all_results = []
            print(f"  Checkpoint saved ({df_batch['FIELDID'].nunique()} fields)")

    # Final save
    df_final = pd.read_csv(CHECKPOINT_PATH)
    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDone! Saved {len(df_final):,} rows, {df_final['FIELDID'].nunique()} fields")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == '__main__':
    import time
    t0 = time.time()
    main()
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} minutes")