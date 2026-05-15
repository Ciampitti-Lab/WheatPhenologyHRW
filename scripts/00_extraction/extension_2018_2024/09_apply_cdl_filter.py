"""Apply CDL winter-wheat filter to produce the (field_id, harvest_year)
tuples we want to predict on.

Logic for winter wheat:
    harvest_year = Y → growing season ≈ Sep Y-1 → Jul Y. CDL Y captures
    the crop that was on the ground for most of that growing period, so
    we keep field-years where CDL[year=Y] wheat-fraction ≥ 0.5.

Output: extension_2018_2024/valid_field_years_2018_2024.parquet
        (columns: field_id, harvest_year, wheat_frac, lat, lon)
"""
from pathlib import Path
import pandas as pd

EXT = Path('/depot/ciampitti/data/WheatPhenologyHRW/data/raw/satellite/extension_2018_2024')
PHENO_MATCHED = ('/depot/ciampitti/data/WheatPhenologyHRW/data/processed/buffer_300m/'
                 'wheat_hrw_phenology_buffer_matched.parquet')
THRESHOLD = 0.5
OUT = EXT / 'valid_field_years_2018_2024.parquet'


def main():
    cdl = pd.read_parquet(EXT / 'cdl_wheat_fraction_2018_2024.parquet')
    cdl = cdl.rename(columns={'FIELDID': 'field_id', 'year': 'harvest_year',
                              'wheat': 'wheat_frac'})
    cdl['field_id'] = cdl['field_id'].astype(str)
    print(f'CDL rows: {len(cdl):,}  field-years before filter')

    keep = cdl[cdl['wheat_frac'] >= THRESHOLD].copy()
    print(f'After wheat_frac ≥ {THRESHOLD}: {len(keep):,} field-years')
    print('\nField-years per harvest_year (after filter):')
    print(keep.groupby('harvest_year').size().to_string())

    # Attach lat/lon from matched phenology parquet
    pheno = pd.read_parquet(PHENO_MATCHED, columns=['FIELDID', 'lat', 'lon'])
    pheno = pheno.rename(columns={'FIELDID': 'field_id'})
    pheno['field_id'] = pheno['field_id'].astype(str)
    coords = pheno.groupby('field_id', as_index=False).agg(
        lat=('lat', 'median'), lon=('lon', 'median'))
    keep = keep.merge(coords, on='field_id', how='left')

    keep.to_parquet(OUT, index=False)
    print(f'\n→ {OUT}')
    print(f'   {len(keep):,} valid (field_id, harvest_year) tuples ready '
          'for feature pipeline.')


if __name__ == '__main__':
    main()
