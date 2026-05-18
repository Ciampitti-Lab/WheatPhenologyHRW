"""Deviation of observed sowing dates from the state-median fallback.

Reports the mean absolute deviation, median and SD over the training
cohort. The state-median is the fixed fallback value from the sowing
lookup. The full-period figures are printed too as a check that the
method reproduces the published numbers. Read-only.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.config import CFG, REPO_ROOT
_WORK = REPO_ROOT / CFG.paths.work_dir
_PHENO = str(REPO_ROOT / CFG.paths.phenology_matched)

from pathlib import Path
import numpy as np
import pandas as pd

EXT = _WORK
SL = EXT / 'sowing_lookup.parquet'
TRAIN_YEARS = [2014, 2015, 2016, 2017]


def stats(dev):
    a = np.abs(dev)
    return dict(n=len(dev),
                mad=a.mean(), med=np.median(a),
                sd0=dev.std(ddof=0), sd1=dev.std(ddof=1))


def report(tag, df, state_med):
    obs = df[df['source'] == 'observed'].copy()
    obs['dev'] = obs['sowing_doy_used'] - obs['state'].map(state_med)
    s = stats(obs['dev'].values)
    print(f'{tag}: n={s["n"]}  MAD={s["mad"]:.1f} d  '
          f'median|dev|={s["med"]:.1f} d  '
          f'SD(signed)={s["sd0"]:.1f} d (ddof0) / {s["sd1"]:.1f} d (ddof1)')
    return s


def main():
    sl = pd.read_parquet(SL)
    # Fixed fallback state-median = value stored on state_median rows.
    state_med = (sl[sl['source'] == 'state_median']
                 .groupby('state')['sowing_doy_used'].agg(
                     lambda v: int(v.iloc[0])).to_dict())
    print('State-median fallback DOY (fixed, full-period):')
    for k in sorted(state_med):
        print(f'  {k}: {state_med[k]}')
    print('(paper: TX 286, OK 290, KS 279, NE 263, CO 280, NM 295)\n')

    full = report('FULL HLS period (validation)', sl, state_med)
    train = report('TRAINING cohort  (recompute) ',
                    sl[sl['harvest_year'].isin(TRAIN_YEARS)], state_med)

    ok = (abs(full['mad'] - 16.5) < 1.0 and abs(full['med'] - 12) < 1.5
          and min(abs(full['sd0'] - 28), abs(full['sd1'] - 28)) < 2.0)
    print(f'\nVALIDATION GATE (full-period reproduces 16.5/12/28): '
          f'{"PASS" if ok else "FAIL"}')
    if ok:
        sd = train['sd0'] if abs(full['sd0'] - 28) < abs(full['sd1'] - 28) \
            else train['sd1']
        print(f'-> Training-cohort numbers to use in the manuscript: '
              f'MAD {train["mad"]:.1f} d, median {train["med"]:.0f} d, '
              f'SD {sd:.0f} d  (n={train["n"]})')
    else:
        print('-> Method does NOT match the published numbers; do not '
              'substitute. Inspect definition before editing the paper.')


if __name__ == '__main__':
    main()
