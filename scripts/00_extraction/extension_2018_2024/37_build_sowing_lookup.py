"""Build per-(field_id, harvest_year) sowing-date lookup.

Priority:
    1. Actual labelled sowing date if available (Seed > Germinating >
       Seed Swell > Preplant)
    2. Per-state median sowing date (from labelled fields only)
    3. Global median if a state has no labels at all (rare)

Output: extension_2018_2024/sowing_lookup.parquet
   columns: field_id, harvest_year, sowing_doy_used, source
   source: 'observed' | 'state_median' | 'global_median'
"""
from pathlib import Path
import pandas as pd
import numpy as np

EXT = Path('/depot/ciampitti/data/WheatPhenologyHRW/data/raw/satellite/extension_2018_2024')
PHENO = '/depot/ciampitti/data/WheatPhenologyHRW/data/processed/buffer_300m/wheat_hrw_phenology_buffer_matched.parquet'
OUT = EXT / 'sowing_lookup.parquet'

# Priority list — earlier = closer to actual sowing event
SOW_PRIORITY = ['Seed', 'Preplant', 'Seed Swell', 'Germinating']

# Offset (days) to convert each stage to "sowing DOY"
# Preplant happens ~5 days before sowing → add 5
# Seed = sowing → 0
# Seed Swell = 3-5 days after sowing → subtract 3
# Germinating = 5-10 days after sowing → subtract 7
STAGE_OFFSET = {
    'Preplant': 5,
    'Seed': 0,
    'Seed Swell': -3,
    'Germinating': -7,
}


def assign_state(lat, lon):
    if pd.isna(lat) or pd.isna(lon):  return None
    if lon < -103.5 and lat < 37.0:   return 'NM'
    if lon < -103.5:                  return 'CO'
    if lat < 34.5:                    return 'TX'
    if lat < 37.0:                    return 'OK'
    if lat < 40.0:                    return 'KS'
    return 'NE'


def main():
    ph = pd.read_parquet(PHENO)
    ph['FIELDID'] = ph['FIELDID'].astype(str)
    ph['harvest_year'] = ph['growing_season'].str.split('-').str[1].astype(int)
    ph['state'] = [assign_state(la, lo) for la, lo in zip(ph['lat'], ph['lon'])]

    # All (field, harvest_year) tuples for which we want a sowing date
    all_fy = ph.groupby(['FIELDID', 'harvest_year', 'state']).first().reset_index()
    print(f'Total field-years in phenology: {len(all_fy):,}')

    # Observed sowing dates: best (highest-priority) stage per field-year
    sow_records = []
    for stage in SOW_PRIORITY:
        sub = ph[ph['growth_stage'] == stage].copy()
        if len(sub) == 0: continue
        offset = STAGE_OFFSET[stage]
        sub['sow_doy_candidate'] = sub['doy'] + offset
        # Earliest occurrence within (field, harvest_year)
        per_fy = sub.groupby(['FIELDID', 'harvest_year'])['sow_doy_candidate'].min().reset_index()
        per_fy['source_stage'] = stage
        sow_records.append(per_fy)

    obs = pd.concat(sow_records, ignore_index=True)
    # Keep priority order: take Seed first if available
    stage_rank = {s: i for i, s in enumerate(SOW_PRIORITY)}
    obs['priority'] = obs['source_stage'].map(stage_rank)
    obs = obs.sort_values(['FIELDID', 'harvest_year', 'priority']).drop_duplicates(
        ['FIELDID', 'harvest_year'], keep='first')
    obs = obs[['FIELDID', 'harvest_year', 'sow_doy_candidate', 'source_stage']]
    obs = obs.rename(columns={'sow_doy_candidate': 'sowing_doy_observed'})
    print(f'Field-years with observed sowing date: {len(obs):,}')
    print(obs['source_stage'].value_counts().to_string())

    # Merge into the full field-year list
    full = all_fy[['FIELDID', 'harvest_year', 'state']].merge(
        obs, on=['FIELDID', 'harvest_year'], how='left')

    # Per-state median (from observed only)
    state_median = obs.merge(all_fy[['FIELDID', 'harvest_year', 'state']],
                              on=['FIELDID', 'harvest_year'])
    state_med = state_median.groupby('state')['sowing_doy_observed'].median().to_dict()
    print(f'\nPer-state median sowing DOY (observed):')
    for s, m in sorted(state_med.items()):
        print(f'  {s}: {m:.0f}')

    global_med = float(obs['sowing_doy_observed'].median())
    print(f'Global median: {global_med:.0f}')

    # Assign
    def resolve(row):
        if not pd.isna(row['sowing_doy_observed']):
            return row['sowing_doy_observed'], 'observed'
        if row['state'] in state_med and not pd.isna(state_med[row['state']]):
            return state_med[row['state']], 'state_median'
        return global_med, 'global_median'

    full[['sowing_doy_used', 'source']] = full.apply(
        lambda r: pd.Series(resolve(r)), axis=1)
    full['sowing_doy_used'] = full['sowing_doy_used'].astype(int)
    full = full.rename(columns={'FIELDID': 'field_id'})

    print(f'\nSource breakdown:')
    print(full['source'].value_counts().to_string())

    print(f'\nFinal sowing DOY distribution:')
    print(f'  p5..p95: [{full["sowing_doy_used"].quantile(0.05):.0f}, '
          f'{full["sowing_doy_used"].quantile(0.95):.0f}]')
    print(f'  median:  {full["sowing_doy_used"].median():.0f}')

    full = full[['field_id', 'harvest_year', 'state',
                 'sowing_doy_used', 'source', 'source_stage']]
    full.to_parquet(OUT, index=False)
    print(f'\n→ {OUT}  ({len(full):,} rows)')


if __name__ == '__main__':
    main()
