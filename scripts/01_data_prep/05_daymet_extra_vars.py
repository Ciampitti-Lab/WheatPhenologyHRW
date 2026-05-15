"""
06 - Download Extra Daymet Variables: srad, vp, swe
====================================================
Adds three new variables to our weather dataset:
  - srad: solar radiation (W/m²) — for PTQ (photothermal quotient)
  - vp:   vapor pressure (Pa) — for VPD calculation
  - swe:  snow water equivalent (kg/m²) — winter dormancy indicator

Uses same REST API + checkpointing pattern as 04_daymet_temperature.py.

Usage: python 06_daymet_extra_vars.py
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
WEATHER_PATH = 'data/raw/satellite/daymet_daily_weather.csv'
OUTPUT_PATH = 'data/raw/satellite/daymet_daily_weather_full.csv'
CHECKPOINT_PATH = 'data/raw/satellite/daymet_extra_checkpoint.csv'

DAYMET_API = 'https://daymet.ornl.gov/single-pixel/api/data'
START_YEAR = 2013
END_YEAR = 2017
VARS = 'srad,vp,swe'
MAX_WORKERS = 8


def fetch_extra(lat, lon, field_id, start_year, end_year):
    """Download srad, vp, swe for one field."""
    url = (f"{DAYMET_API}?lat={lat}&lon={lon}&vars={VARS}"
           f"&years={','.join(str(y) for y in range(start_year, end_year+1))}")
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
        return df[['FIELDID', 'date', 'year', 'doy', 'srad', 'vp', 'swe']]
    except Exception:
        return None


def main():
    df_pheno = pd.read_parquet(PHENOLOGY_PATH)
    fields = df_pheno.groupby('FIELDID').agg(
        lat=('lat', 'median'), lon=('lon', 'median')
    ).reset_index()
    print(f"Total fields: {len(fields)}")

    done_ids = set()
    if os.path.exists(CHECKPOINT_PATH):
        done_ids = set(pd.read_csv(CHECKPOINT_PATH)['FIELDID'].astype(str).unique())
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
                futures = [
                    ex.submit(fetch_extra, r['lat'], r['lon'], r['FIELDID'], START_YEAR, END_YEAR)
                    for _, r in batch.iterrows()
                ]
                for f in as_completed(futures):
                    res = f.result()
                    if res is not None:
                        all_results.append(res)

            done = batch_start + len(batch)
            rate = done / (time.time() - t0) * 60 if (time.time() - t0) > 0 else 0
            print(f"  [{done}/{len(remaining)}] {rate:.0f} fields/min")

            if len(all_results) >= 500 or done >= len(remaining):
                df_b = pd.concat(all_results, ignore_index=True)
                if os.path.exists(CHECKPOINT_PATH):
                    df_b = pd.concat([pd.read_csv(CHECKPOINT_PATH), df_b], ignore_index=True)
                df_b.to_csv(CHECKPOINT_PATH, index=False)
                all_results = []

    # Merge srad/vp/swe with existing weather (Tmin/Tmax/prcp)
    print("\nMerging extra vars with existing weather...")
    df_existing = pd.read_csv(WEATHER_PATH)
    df_extra = pd.read_csv(CHECKPOINT_PATH)
    df_existing['FIELDID'] = df_existing['FIELDID'].astype(str)
    df_extra['FIELDID'] = df_extra['FIELDID'].astype(str)
    df_existing['date'] = pd.to_datetime(df_existing['date'], format='mixed')
    df_extra['date'] = pd.to_datetime(df_extra['date'], format='mixed')

    df = df_existing.merge(
        df_extra[['FIELDID', 'date', 'srad', 'vp', 'swe']],
        on=['FIELDID', 'date'], how='left'
    )

    # Normalize temperature column names (existing CSV has lowercase tmin/tmax)
    if 'tmin' in df.columns and 'Tmin' not in df.columns:
        df = df.rename(columns={'tmin': 'Tmin', 'tmax': 'Tmax'})

    # Compute derived features:
    #   VPD (Vapor Pressure Deficit) = saturation_vp - actual_vp
    #   saturation_vp uses Magnus formula on T_mean
    T_mean = (df['Tmin'] + df['Tmax']) / 2
    es = 611.2 * np.exp(17.62 * T_mean / (243.12 + T_mean))  # saturation vp in Pa
    df['vpd'] = (es - df['vp']).clip(lower=0)

    #   PTQ (Photothermal Quotient) = solar / GDD
    #   Useful for grain-filling stress detection
    gdd = np.maximum(0, T_mean)  # Method 1 GDD base 0
    df['ptq'] = df['srad'] / np.where(gdd > 0, gdd, np.nan)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved: {OUTPUT_PATH} ({len(df):,} rows)")
    print(f"\nNew columns: srad, vp, swe, vpd (derived), ptq (derived)")
    print(f"  srad missing: {df['srad'].isna().sum()}")
    print(f"  vp missing:   {df['vp'].isna().sum()}")
    print(f"  swe missing:  {df['swe'].isna().sum()}")


if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f"\nTotal: {(time.time() - t0) / 60:.1f} min")
