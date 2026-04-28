"""De-identify the per-field artefacts for the public release.

Mirrors the de-identification logic of the companion WheatGPCPipeline:
the partner-issued field identifiers are remapped to anonymous integers
with a deterministic, file-consistent mapping, and exact field
coordinates are dropped while state-level geographic context is kept.

Produces, under ``data_public/processed/`` (committed to the public
repo, restricted to the paper's four training seasons 2013/14–2016/17):

  * phenology_labels.parquet  — cleaned per-field stage observations
  * sowing_lookup.parquet     — per field-year WES sowing anchor
  * features_v3_train.parquet — DOS-anchored model feature matrix
  * field_id_mapping.json     — anonymous-id count only (original
                                identifiers are intentionally omitted)

The original partner-issued keys and the raw lat/lon are NEVER written.
Read-only on the inputs. Run from the repo root:

    python3 scripts/deidentify_public_release.py
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
# Exact field coordinates / geometry are never published.
DROP_COLS = {'lat', 'lon', 'latitude', 'longitude',
             'geometry', 'centroid_lat', 'centroid_lon', 'x', 'y'}


def assign_state(lat, lon):
    """Replicates the state assignment used to build the sowing lookup."""
    if pd.isna(lat) or pd.isna(lon):
        return None
    if lon < -103.5 and lat < 37.0:
        return 'NM'
    if lon < -103.5:
        return 'CO'
    if lat < 34.5:
        return 'TX'
    if lat < 37.0:
        return 'OK'
    if lat < 40.0:
        return 'KS'
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
                         'original partner-issued field identifiers and '
                         'exact coordinates are intentionally omitted '
                         'from the public release.'),
         'n_fields': len(key_map)}, indent=2))
    print(f'\n-> {OUT.relative_to(REPO_ROOT)}/  '
          f'({len(key_map)} anonymous fields; coordinates dropped)')


if __name__ == '__main__':
    main()
