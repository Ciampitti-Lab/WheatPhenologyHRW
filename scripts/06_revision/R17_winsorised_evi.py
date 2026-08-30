"""R17 - Refit the reproductive stages with the EVI tails clipped.

Audit item 13 records that \\num{498} of \\num{8461} field-years (5.9 %) carry
an |EVI base, peak or amplitude| above 2, the worst 3.8e4, because the EVI
denominator approaches zero over sparse canopy and the ratio explodes. We left
them unfiltered by choice: gradient-boosted trees split rather than
extrapolate, so the adopted models are unaffected. A reviewer reading item 13
is entitled to ask what happens if they are removed, so this answers it in
advance.

EVI is bounded on [-1, 1] for real surfaces. We clip the affected columns to
[-2, 2], a physical bound rather than a data-derived percentile, so nothing
leaks from the held-out fold. Adopted model per stage, identical LOYO
protocol, everything else untouched.

Output: data/revision/R17_winsorised_evi.csv
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np, pandas as pd
from scripts.utils.config import CFG, REPO_ROOT
from scripts.utils.deep_models import (ADOPT, load_cohort, stage_frame, r2,
                                       fold_pred_ensemble, fold_pred_ft, SEEDS)

WORK = REPO_ROOT / CFG.paths.work_dir
PHENO = str(REPO_ROOT / CFG.paths.phenology_matched)
OUT = REPO_ROOT / 'data' / 'revision'
REPRO = ['flag_leaf', 'boot', 'heading', 'anthesis']
BOUND = 2.0


def pooled(d, feat, tgt, mod):
    P, T = [], []
    for yr in sorted(d['year'].unique()):
        tr, te = d[d['year'] != yr], d[d['year'] == yr]
        if len(tr) < 50 or len(te) < 5:
            continue
        if mod == 'FT':
            P.extend(np.mean([fold_pred_ft(tr, te, feat, tgt, s) for s in SEEDS], axis=0))
        else:
            P.extend(fold_pred_ensemble(tr, te, feat, tgt, mod))
        T.extend(te[tgt].values)
    return np.array(T), np.array(P)


def main():
    fe, cols = load_cohort(WORK, PHENO)
    # Only the three columns audit item 13 names. The other EVI features are
    # day-of-season dates (SOS, POS, midpoint, shoulder, duration) and an
    # integrated area, all of which legitimately exceed 2 and must not be
    # clipped: doing so would destroy real features rather than tails.
    evi = ['EVI_base', 'EVI_peak', 'EVI_amplitude']
    assert all(c in fe.columns for c in evi), [c for c in evi if c not in fe.columns]
    hit = fe[evi].abs().gt(BOUND).any(axis=1)
    print(f'EVI columns: {len(evi)}   field-years beyond +/-{BOUND}: {int(hit.sum())} '
          f'({100 * hit.mean():.1f} %)   worst |value| {fe[evi].abs().max().max():.3g}',
          flush=True)

    rows = []
    for clipped in (False, True):
        f = fe.copy()
        if clipped:
            f[evi] = f[evi].clip(-BOUND, BOUND)
        for s in REPRO:
            strat, mod = ADOPT[s]
            d, tgt = stage_frame(f, s)
            t0 = time.time()
            T, P = pooled(d, cols(s, strat == 'C_Hybrid'), tgt, mod)
            R = r2(T, P); rm = float(np.sqrt(np.mean((T - P) ** 2)))
            rows.append(dict(stage=s, model=mod, variant='clipped' if clipped else 'control',
                             R2=round(R, 4), RMSE=round(rm, 3), n=len(T)))
            print(f'  {s:10s} {mod:11s} {"clipped" if clipped else "control":8s} '
                  f'R2={R:+.4f} RMSE={rm:.2f} ({time.time() - t0:.0f}s)', flush=True)

    R = pd.DataFrame(rows)
    p = R.pivot_table(index='stage', columns='variant', values='R2')
    p['delta'] = (p['clipped'] - p['control']).round(4)
    print('\n=== effect of clipping the EVI tails ===')
    print(p.round(4).to_string())
    print(f'\nlargest |change| in LOYO R2: {p.delta.abs().max():.4f}')
    OUT.mkdir(parents=True, exist_ok=True)
    R.to_csv(OUT / 'R17_winsorised_evi.csv', index=False)
    print(f'-> data/revision/R17_winsorised_evi.csv ({len(R)} rows)')


if __name__ == '__main__':
    main()
