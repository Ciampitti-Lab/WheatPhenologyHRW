"""Provenance for the numbers added in the reviewer-2 proofing pass.

Recomputes, from the canonical v3 artefacts, every figure quoted in the
manuscript that was added to answer reviewer-2 attack vectors:

  1. TX/OK/KS reproductive-stage trend range  -> abstract/conclusion
     "advance by 0.4-1.6 d/yr ... largest in Texas"
  2. Colorado significant-stage count          -> Colorado framing
  3. Maturity R^2 bootstrap CI                  -> Results 3.1 / SI A.4
  4. Label-intrinsic scatter (repeat same-stage obs within 21 d)
     -> Discussion noise-floor lower bound
  5. HLS usable composite density per month     -> Methods 2.4.1 / 2.4.6
     ("~2 clear obs/field/month, roughly uniform across the season")

Run: python3 46_reviewer2_stats.py    (read-only; prints a report)
"""

# --- repo-portable paths (no hardcoded cluster paths) --------------------
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from scripts.utils.config import CFG, REPO_ROOT
_WORK = REPO_ROOT / CFG.paths.work_dir
_PHENO = str(REPO_ROOT / CFG.paths.phenology_matched)
# ------------------------------------------------------------------------

from pathlib import Path
import numpy as np
import pandas as pd

EXT = _WORK
PHENO = _PHENO
TRAIN_YEARS = [2014, 2015, 2016, 2017]
REPRO = ['flag_leaf', 'boot', 'heading', 'anthesis']


def r2(yt, yp):
    d = np.sum((yt - yt.mean()) ** 2)
    return 1 - np.sum((yt - yp) ** 2) / d if d > 0 else 0.0


def main():
    # 1 & 2 — trends
    t = pd.read_csv(EXT / 'v3_trends_per_stage_per_state.csv')
    sub = t[t.stage.isin(REPRO) & t.state.isin(['TX', 'OK', 'KS'])]
    adv = sub[sub.slope_d_per_yr < 0].slope_d_per_yr
    print(f'[1] TX/OK/KS reproductive advancing slopes: '
          f'{-adv.max():.2f} to {-adv.min():.2f} d/yr  '
          f'(largest state mean: '
          f'{sub.groupby("state").slope_d_per_yr.mean().idxmin()})')
    co = t[t.state == 'CO'].copy()
    co['sig_pos'] = ((co.ci_lo > 0) & (co.slope_d_per_yr > 0))
    print(f'[2] Colorado significant-delay stages: {int(co.sig_pos.sum())}/8 '
          f'({", ".join(sorted(co[co.sig_pos].stage))}); '
          f'all 4 reproductive significant: '
          f'{co[co.stage.isin(REPRO)].sig_pos.all()}')

    # 3 — maturity R^2 bootstrap CI from LOYO predictions
    lp = pd.read_parquet(EXT / 'v3_loyo_predictions.parquet')
    m = lp[lp.stage == 'maturity']
    y, p = m['observed'].values, m['predicted'].values
    rng = np.random.RandomState(42)
    bs = [r2(y[i], p[i]) for i in
          (rng.choice(len(y), len(y), True) for _ in range(2000))]
    print(f'[3] maturity n={len(y)}  R2={r2(y, p):.3f}  '
          f'95% CI [{np.percentile(bs, 2.5):.3f}, {np.percentile(bs, 97.5):.3f}]')

    # 4 — label-intrinsic scatter
    ph = pd.read_parquet(PHENO)
    ph['fid'] = ph['FIELDID'].astype(str)
    ph['hy'] = ph['growing_season'].str.split('-').str[1].astype(int)
    ph = ph[ph.hy.isin(TRAIN_YEARS)]
    vocab = {
        'tillering': ['Begin Tillering', 'Tillering', '1-2 Tiller', '2-4 Tiller',
                      '4-6 Tiller', '6-8 Tiller', '8+ Tiller', 'Full Tillering',
                      'End Tillering'],
        'jointing': ['Jointing', '1st Node Visible', '2nd Node Visible',
                     '3rd Node Visible', 'Spring Vegetative'],
        'emergence': ['Emerging', 'Emerging - Seedling', 'Shoot - Emerging',
                      'Shoot', 'Seedling', 'Seedling - 1 Leaf', '1 Leaf',
                      '2 Leaf', '3 Leaf', '4 Leaf'],
    }
    for st, labs in vocab.items():
        g = (ph[ph.growth_stage.isin(labs)]
             .groupby(['fid', 'hy'])['dos'].agg(['count', 'std', 'min', 'max']))
        w = g[(g['count'] >= 2) & (g['max'] - g['min'] <= 21)]
        print(f'[4] {st}: n={len(w)} field-years with >=2 obs within 21 d, '
              f'within-FY SD = {w["std"].mean():.1f} d')

    # 5 — HLS usable composite density per month
    h = pd.read_parquet(EXT / 'hls_full_2013_2024.parquet',
                        columns=['field_id', 'date', 'NDVI'])
    h['date'] = pd.to_datetime(h['date'])
    h = h[h.date.dt.year.isin([2013] + TRAIN_YEARS) & h.NDVI.notna()]
    u = h.drop_duplicates(['field_id', 'date']).copy()
    u['m'] = u.date.dt.month
    u['fy'] = u.field_id.astype(str) + '_' + u.date.dt.year.astype(str)
    permo = u.groupby(['fy', 'm']).size().groupby('m').mean()
    print(f'[5] HLS usable obs/field/month: Oct-Feb '
          f'{permo.reindex([10, 11, 12, 1, 2]).mean():.1f}  vs  '
          f'Apr-Jun {permo.reindex([4, 5, 6]).mean():.1f}  '
          f'(season-wide ~{permo.mean():.1f}/mo)')


if __name__ == '__main__':
    main()
