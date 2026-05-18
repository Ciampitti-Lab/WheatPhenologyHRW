"""Recompute the sowing-anchor calibration stat for the training cohort.

The manuscript S4 / Results 3.3 sentence quotes the deviation of observed
sowing dates from their state-median fallback as MAD 16.5 d, median 12 d,
SD 28 d, originally over the FULL-period observed set (n=1679). The
supplement is now scoped to the training cohort only, so this recomputes
the same statistic over the training-cohort observed set (n=1382).

Method is identical to 37_build_sowing_lookup.py: the state-median is the
fixed fallback value stored in the lookup (median of ALL observed per
state, full period); deviation = observed sowing DOY - that state median.

VALIDATION GATE: the script first reproduces the full-period numbers and
only the run that recovers ~16.5/12/28 certifies the training-cohort
numbers as methodologically consistent. Read-only; prints a report.
"""
from pathlib import Path
import numpy as np
import pandas as pd

EXT = Path('/depot/ciampitti/data/WheatPhenologyHRW/data/raw/satellite/'
           'extension_2018_2024')
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
