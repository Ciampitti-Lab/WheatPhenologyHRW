"""Build the de-identified public data subset.

Remaps field identifiers to anonymous integers (consistent across
files), coarsens coordinates, keeps state, and restricts to the four
training seasons. Writes data_public/processed/. The id mapping is not
saved. Run from the repo root.

Coordinates
-----------
Latitude and longitude are *model inputs*, so simply deleting them
leaves the release unable to reproduce the published results: dropping
them moves LOYO R^2 by up to 0.10 (scripts/06_revision/R06). They are
therefore snapped to the centre of a GRID_DEG cell instead of removed.
At 0.05 deg (~5.5 x 4.5 km, roughly 300x the area of the 300 m field
buffer) the released matrix reproduces every published value to within
0.01 R^2 while no field centroid is recoverable. The data-sharing
agreement is satisfied because the released coordinate identifies a
grid cell, not a field.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.utils.config import CFG, REPO_ROOT

WORK = REPO_ROOT / CFG.paths.work_dir
PHENO = REPO_ROOT / CFG.paths.phenology_matched
OUT = REPO_ROOT / 'data_public' / 'processed'

TRAIN_HARVEST_YEARS = {2014, 2015, 2016, 2017}
# Exact field geometry is never published.
DROP_COLS = {'geometry', 'centroid_lat', 'centroid_lon', 'x', 'y'}
# Coordinate columns are coarsened rather than dropped (see module docstring).
COARSEN_COLS = {'lat': 'lat', 'lon': 'lon',
                'latitude': 'latitude', 'longitude': 'longitude'}
GRID_DEG = 0.05


# Corrected per-field state assignment (spatial join; see R14a_fix_states.py).
def _load_state_lookup():
    import csv, os
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     '..', 'data', 'revision', 'R14_state_lookup.csv')
    d = {}
    if os.path.exists(p):
        with open(p) as f:
            for r in csv.DictReader(f):
                if r.get('state_true'):
                    d[(round(float(r['latitude']), 6),
                       round(float(r['longitude']), 6))] = r['state_true']
    return d


_STATE_LOOKUP = _load_state_lookup()


def assign_state(lat, lon):
    """State of a field centroid.

    NOTE. Earlier versions of this function used a latitude/longitude box:

        if lat < 34.5: 'TX'  elif lat < 37.0: 'OK'  elif lat < 40.0: 'KS'

    That rule assigns the whole Texas Panhandle, which reaches 36.5 N, to
    Oklahoma, and the -103.5 longitude cut misplaces fields along the
    Colorado/Kansas border, which lies at -102.05. It misassigned 348 of
    5293 fields (6.6 %). State is now read from the spatial join produced
    by scripts/06_revision/R14a_fix_states.py; the box is retained only as
    a fallback for points the join cannot resolve.
    """
    if pd.isna(lat) or pd.isna(lon):
        return None
    s = _STATE_LOOKUP.get((round(float(lat), 6), round(float(lon), 6)))
    if s is not None:
        return s
    if lon < -103.5 and lat < 37.0:   return 'NM'
    if lon < -103.5:                  return 'CO'
    if lat < 34.5:                    return 'TX'
    if lat < 37.0:                    return 'OK'
    if lat < 40.0:                    return 'KS'
    return 'NE'


def build_field_id_mapping(*frames) -> dict:
    """Deterministic original-FIELDID -> anonymous-int map.

    Built from the union of every frame's id column so the same field
    gets the same integer across all output files. Integers are
    assigned in sorted order of the string id for full reproducibility.
    """
    keys: set = set()
    for df in frames:
        col = 'FIELDID' if 'FIELDID' in df.columns else 'field_id'
        keys.update(df[col].dropna().astype(str).tolist())
    return {k: i + 1 for i, k in enumerate(sorted(keys))}


def _anonymise(df: pd.DataFrame, key_map: dict) -> pd.DataFrame:
    df = df.copy()
    col = 'FIELDID' if 'FIELDID' in df.columns else 'field_id'
    df['field_id'] = df[col].astype(str).map(key_map).astype('Int64')
    if col != 'field_id':
        df = df.drop(columns=[col])
    df = df.drop(columns=[c for c in df.columns if c in DROP_COLS])
    for c in df.columns:
        if c in COARSEN_COLS:
            df[c] = (df[c] // GRID_DEG) * GRID_DEG + GRID_DEG / 2
    front = [c for c in ('field_id', 'state', 'harvest_year', 'year')
             if c in df.columns]
    return df[front + [c for c in df.columns if c not in front]]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # --- per-field phenology labels (training seasons only) ----------
    ph = pd.read_parquet(PHENO)
    ph['harvest_year'] = ph['growing_season'].str.split('-').str[1].astype(int)
    ph = ph[ph['harvest_year'].isin(TRAIN_HARVEST_YEARS)].copy()
    ph['state'] = [assign_state(la, lo) for la, lo in zip(ph['lat'], ph['lon'])]

    # --- sowing lookup (training cohort) -----------------------------
    sl = pd.read_parquet(WORK / 'sowing_lookup.parquet')
    sl = sl[sl['harvest_year'].isin(TRAIN_HARVEST_YEARS)].copy()

    # --- DOS-anchored v3 features (already the 8,465 training cohort) -
    fe = pd.read_parquet(WORK / 'features_v3_realsowing_train.parquet')

    key_map = build_field_id_mapping(ph, sl, fe)

    out = {
        'phenology_labels.parquet': _anonymise(ph, key_map),
        'sowing_lookup.parquet':    _anonymise(sl, key_map),
        'features_v3_train.parquet': _anonymise(fe, key_map),
    }
    for name, df in out.items():
        df.to_parquet(OUT / name, index=False)
        ident = [c for c in df.columns
                 if c.lower() in DROP_COLS or c in ('FIELDID',)]
        print(f'  {name:28s} {df.shape}  '
              f'fields={df["field_id"].nunique()}  '
              f'leak={ident or "none"}')

    (OUT / 'field_id_mapping.json').write_text(json.dumps(
        {'description': ('Anonymous field_id integer assignments. The '
                         'original partner-issued field identifiers are '
                         'omitted from the public release, and field '
                         'coordinates are snapped to the centre of a '
                         f'{GRID_DEG}-degree grid cell so that the released '
                         'matrix reproduces the published models without '
                         'identifying any field.'),
         'n_fields': len(key_map), 'coordinate_grid_deg': GRID_DEG}, indent=2))
    print(f'\n-> {OUT.relative_to(REPO_ROOT)}/  '
          f'({len(key_map)} anonymous fields; coordinates snapped to {GRID_DEG} deg)')


if __name__ == '__main__':
    main()
