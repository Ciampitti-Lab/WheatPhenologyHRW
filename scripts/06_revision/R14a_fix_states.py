"""R14a - Correct the per-field state assignment.

The pipeline assigns each field to a state with a latitude/longitude box
(scripts/01_sowing/01_build_sowing_lookup.py):

    if lat < 34.5: 'TX'   elif lat < 37.0: 'OK'   elif lat < 40.0: 'KS' ...

The Texas Panhandle reaches 36.5 N, so the second rule hands the whole
Panhandle to Oklahoma, and the -103.5 longitude cut misplaces fields along
the Colorado/Kansas border, which actually lies at -102.05.

This replaces the box with a spatial join against Natural Earth state
polygons and writes a corrected lookup. It also flags fields whose
coordinates fall outside the five study states.

Needs the geospatial environment:
  /depot/ciampitti/apps/envs/vmangidi_ww_protein_prediction/bin/python

Output: data/revision/R14_state_lookup.csv
"""
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

ROOT = Path(__file__).resolve().parents[2]
REV = ROOT / 'data' / 'revision'
W = Path('/depot/ciampitti/data/WheatPhenologyHRW/data/raw/satellite/'
         'extension_2018_2024')
SHP = Path('/home/vmangidi/.local/share/cartopy/shapefiles/natural_earth/'
           'cultural/ne_50m_admin_1_states_provinces.shp')
STUDY = ['TX', 'OK', 'KS', 'NE', 'CO']


def main():
    fe = pd.read_parquet(W / 'features_v3_realsowing_train.parquet',
                         columns=['field_id', 'latitude', 'longitude',
                                  'state_TX', 'state_OK', 'state_KS',
                                  'state_NE', 'state_CO', 'state_NM'])
    oh = [c for c in fe.columns if c.startswith('state_')]
    fe['state_rule'] = fe[oh].idxmax(axis=1).str.replace('state_', '')
    fld = fe.drop_duplicates('field_id')[
        ['field_id', 'latitude', 'longitude', 'state_rule']].copy()

    st = gpd.read_file(SHP)
    st = st[st.admin == 'United States of America'][['postal', 'geometry']]
    g = gpd.GeoDataFrame(
        fld, geometry=[Point(x, y) for x, y in zip(fld.longitude, fld.latitude)],
        crs='EPSG:4326')
    j = gpd.sjoin(g, st, how='left', predicate='within')
    j = j.rename(columns={'postal': 'state_true'}).drop(columns='index_right')
    j = j.drop_duplicates('field_id')

    j['in_study_area'] = j.state_true.isin(STUDY)
    j['changed'] = j.state_true.notna() & (j.state_rule != j.state_true)

    out = j[['field_id', 'latitude', 'longitude', 'state_rule',
             'state_true', 'in_study_area', 'changed']]
    REV.mkdir(parents=True, exist_ok=True)
    out.to_csv(REV / 'R14_state_lookup.csv', index=False)

    print(f'fields: {len(out)}   reassigned: {out.changed.sum()} '
          f'({100 * out.changed.mean():.1f} %)\n')
    cmp = pd.DataFrame({'published': out.state_rule.value_counts(),
                        'corrected': out.state_true.value_counts()}).fillna(0).astype(int)
    cmp['diff'] = cmp.corrected - cmp.published
    print(cmp.sort_values('corrected', ascending=False).to_string())
    print(f'\noutside the five study states: {(~out.in_study_area).sum()} fields')
    print(out[~out.in_study_area][['field_id', 'latitude', 'longitude',
                                   'state_true']].to_string(index=False))
    print(f'\n-> {REV / "R14_state_lookup.csv"}')


if __name__ == '__main__':
    main()
