"""Concat Daymet extension years (2018-2024) and add derived VPD + PTQ.

Schema matches the original training-set Daymet parquet so downstream
feature notebooks can swap data sources without other changes.

Output: data/raw/satellite/extension_2018_2024/daymet_full_2018_2024.parquet
"""
from pathlib import Path
import numpy as np
import pandas as pd

DAYMET_DIR = Path('/depot/ciampitti/data/WheatPhenologyHRW/data/raw/satellite')
OUT = DAYMET_DIR / 'extension_2018_2024' / 'daymet_full_2018_2024.parquet'

YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024]

print(f'=== Daymet concat + derive  (years {YEARS}) ===')
parts = []
for y in YEARS:
    p = DAYMET_DIR / f'daymet_extension_{y}.csv'
    df = pd.read_csv(p, dtype={'FIELDID': str})
    print(f'  {p.name:<35} {len(df):>10,} rows')
    parts.append(df)

df = pd.concat(parts, ignore_index=True)
df['date'] = pd.to_datetime(df['date'], format='mixed')
df = df.rename(columns={'tmin': 'Tmin', 'tmax': 'Tmax'})

# Derived: VPD (saturation vapor pressure - actual vapor pressure)
T_mean = (df['Tmin'] + df['Tmax']) / 2
es = 611.2 * np.exp(17.62 * T_mean / (243.12 + T_mean))   # Pa, Magnus formula
df['vpd'] = (es - df['vp']).clip(lower=0)

# Derived: PTQ (photothermal quotient = srad / GDD_base0)
gdd = np.maximum(0, T_mean)
df['ptq'] = df['srad'] / np.where(gdd > 0, gdd, np.nan)

df.to_parquet(OUT, index=False)
print(f'\n→ {OUT}')
print(f'   {len(df):,} rows, {df["FIELDID"].nunique():,} fields')
print(f'   columns: {list(df.columns)}')
print(f'   missing — vpd: {df["vpd"].isna().sum():,}  ptq: {df["ptq"].isna().sum():,}')
