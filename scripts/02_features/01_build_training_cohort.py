"""Build the (field_id, harvest_year) tuples for v2 training set.

Restricts to harvest_year 2014-2017 because harvest_year=2013 lacks
the pre-2013 HLS coverage needed for full growing-season smoothing.
Targets are computed elsewhere (in modelling notebook) from phenology
labels — this script just enumerates the field-years.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.config import CFG, REPO_ROOT
_WORK = REPO_ROOT / CFG.paths.work_dir
_PHENO = str(REPO_ROOT / CFG.paths.phenology_matched)

from pathlib import Path
import pandas as pd

PHENO_PATH = _PHENO
EXT = _WORK
OUT = EXT / 'valid_field_years_2014_2017_train.parquet'

ph = pd.read_parquet(PHENO_PATH)
ph['FIELDID'] = ph['FIELDID'].astype(str)
# growing_season looks like "2013-2014" — split second is harvest_year
ph['harvest_year'] = ph['growing_season'].str.split('-').str[1].astype(int)

# Build unique (field_id, harvest_year, lat, lon) tuples for harvest_year >= 2014
keep = (ph[ph['harvest_year'] >= 2014]
        .groupby(['FIELDID', 'harvest_year'], as_index=False)
        .agg(lat=('lat', 'median'), lon=('lon', 'median')))
keep = keep.rename(columns={'FIELDID': 'field_id'})

print(f'Training (field_id, harvest_year) tuples (2014-2017): {len(keep):,}')
print(f'Per harvest_year:\n{keep.groupby("harvest_year").size().to_string()}')
print(f'Unique fields: {keep["field_id"].nunique():,}')

keep.to_parquet(OUT, index=False)
print(f'\n→ {OUT}')
