"""R06 - What can the released de-identified matrix actually reproduce?
        (journal research-data policy; Co-EIC "all instructions in the
        Guide for Authors must be followed").

The submitted Data-availability statement claims the deposited matrix is
"sufficient to reproduce every result, table, and figure in this paper,
with field identifiers and precise coordinates removed". But latitude and
longitude are *active model inputs* (Sec. 2.3.2), so removing them cannot
leave the results reproducible. This script measures the gap and tests
the coarsening that closes it.

Variants
--------
  full        latitude/longitude at native precision (what we ran)
  dropped     both removed  (what we released)
  coarse_XX   both snapped to an XX-degree grid, so no field centroid is
              recoverable but the continuous spatial gradient survives

A 0.10-degree cell is roughly 11 km north-south, two orders of magnitude
coarser than the 300 m field buffer, and the cell occupancy statistic
printed below shows how many distinct fields share each cell.

Output: data/revision/R06_public_reproducibility.csv
"""
import warnings

warnings.filterwarnings('ignore')

from _common import ORDER, STAGE_LABEL, cohort, metrics, save, stage_frame
from scripts.utils.deep_models import ADOPT, loyo
import numpy as np
import pandas as pd

# stages whose adopted model runs on CPU (deep-model stages are handled by
# the GPU companion job); LightGBM is used for the deep-model stages here so
# the variant contrast is still measured, and is labelled as such.
CPU_SUB = {'emergence': 'LightGBM', 'tillering': 'ElasticNet',
           'jointing': 'LightGBM', 'flag_leaf': 'XGBoost',
           'boot': 'LightGBM', 'heading': 'ElasticNet',
           'anthesis': 'LightGBM', 'maturity': 'LightGBM'}
GRIDS = [0.05, 0.10, 0.25]


def main():
    fe, cols = cohort()

    print('=== cell occupancy of the coordinate grids ===')
    nf = fe['field_id'].nunique()
    for g in GRIDS:
        cell = (np.floor(fe.latitude / g).astype(int).astype(str) + '_'
                + np.floor(fe.longitude / g).astype(int).astype(str))
        occ = fe.assign(cell=cell).groupby('cell')['field_id'].nunique()
        print(f'  {g:.2f} deg : {occ.size:4d} cells, '
              f'median {occ.median():.0f} fields/cell, '
              f'{100 * (occ == 1).sum() / occ.size:.1f} % singleton cells')
    print(f'  ({nf} unique fields in total)\n')

    rows = []
    for s in ORDER:
        d0, tgt = stage_frame(fe, s)
        model = CPU_SUB[s]
        strat, adopted = ADOPT[s]
        wes = strat == 'C_Hybrid'
        feat = cols(s, wes)
        for variant in ['full', 'dropped'] + [f'coarse_{g}' for g in GRIDS]:
            d = d0.copy()
            f = list(feat)
            if variant == 'dropped':
                f = [c for c in f if c not in ('latitude', 'longitude')]
            elif variant.startswith('coarse'):
                g = float(variant.split('_')[1])
                d['latitude'] = np.floor(d.latitude / g) * g + g / 2
                d['longitude'] = np.floor(d.longitude / g) * g + g / 2
            T, P = loyo(d, f, tgt, model)
            if len(T) == 0:
                continue
            r = metrics(T, P)
            r.update(stage=s, variant=variant, model=model,
                     is_adopted_model=(model == adopted))
            rows.append(r)
            print(f'  {STAGE_LABEL[s]:10s} {variant:11s} {model:11s} '
                  f'R2={r["R2"]:+.3f}', flush=True)

    R = pd.DataFrame(rows)
    save(R.round(4), 'R06_public_reproducibility.csv')
    piv = R.pivot_table(index='stage', columns='variant', values='R2').reindex(ORDER)
    print('\n=== LOYO R2 by coordinate treatment ===')
    print(piv.round(3).to_string())
    print('\n=== deviation from the native-precision run ===')
    dev = piv.sub(piv['full'], axis=0).drop(columns='full')
    print(dev.round(3).to_string())
    print('\nmax |deviation| per variant:')
    print(dev.abs().max().round(3).to_string())


if __name__ == '__main__':
    main()
