"""R13d - Buffer/mask refit with one fixed, outlier-robust model.

R13b's first pass used each stage's adopted model, and heading (ElasticNet)
returned R^2 of order -1e16. That is not a buffer effect: the rebuilt EVI
phenometrics contain denominator artifacts (EVI is bounded near [-1, 1] for
real surfaces, but the ratio explodes when NIR + 6*RED - 7.5*BLUE + 1
approaches zero over sparse canopy), and a regularised linear model
extrapolates catastrophically from them. The published feature matrix carries
the same pathology at smaller magnitude: 500 of 8465 field-years (5.9 %) have
|EVI base/peak/amplitude| > 2, the worst 3.8e4.

We therefore fix the model at LightGBM for every stage, as in R01 and R08.
Trees split rather than extrapolate, so the contrast stays a feature-set
contrast and is not dominated by one fragile estimator. The 300 m unmasked
arm remains the positive control.

Output: data/revision/R13_buffer_sensitivity.csv
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np, pandas as pd
from _common import cohort, metrics, stage_frame
from scripts.utils.deep_models import fold_pred_ensemble

REV = ROOT / 'data' / 'revision'
MODEL = 'LightGBM'
REPRO = ['flag_leaf', 'boot', 'heading', 'anthesis']
RADII, MASKS = [150, 300, 500], ['raw', 'cdl']


def main():
    ph = pd.read_parquet(REV / 'R13_buffer_phenometrics.parquet')
    fe, cols = cohort()
    fe['FIELDID'] = fe['field_id'].astype(str)
    hls_cols = [c for c in fe.columns if c.startswith(('NDVI', 'EVI', 'GCVI', 'DL_c'))]

    rows = []
    for radius in RADII:
        for mask in MASKS:
            sub = ph[(ph.radius == radius) & (ph['mask'] == mask)].copy()
            sub = sub.rename(columns={'harvest_year': 'year'})
            keep = [c for c in hls_cols if c in sub.columns]
            sub = sub[['FIELDID', 'year'] + keep]
            f2 = fe.drop(columns=keep).merge(sub, on=['FIELDID', 'year'], how='inner')
            for stage in REPRO:
                d, tgt = stage_frame(f2, stage)
                if len(d) < 100:
                    continue
                feat = [c for c in cols(stage, True) if c in d.columns]
                T, P = [], []
                for y in sorted(d.year.unique()):
                    tr, te = d[d.year != y], d[d.year == y]
                    if len(tr) < 50 or len(te) < 5:
                        continue
                    P.extend(fold_pred_ensemble(tr, te, feat, tgt, MODEL))
                    T.extend(te[tgt].values)
                if not T:
                    continue
                m = metrics(np.asarray(T, float), np.asarray(P, float))
                m.update(stage=stage, radius=radius, mask=mask, n=len(d), model=MODEL)
                rows.append(m)
                print(f'  r={radius:3d} {mask:3s} {stage:10s} R2={m["R2"]:+.3f} '
                      f'RMSE={m["RMSE"]:5.2f} n={len(d)}', flush=True)

    R = pd.DataFrame(rows)
    ref = R[(R.radius == 300) & (R['mask'] == 'raw')].set_index('stage')['R2']
    R['delta_vs_300raw'] = R.apply(lambda x: x.R2 - ref.get(x.stage, np.nan), axis=1)
    R.to_csv(REV / 'R13_buffer_sensitivity.csv', index=False)

    print('\n=== LOYO R2 by radius and mask (LightGBM fixed) ===')
    print(R.pivot_table(index='stage', columns=['mask', 'radius'],
                        values='R2').reindex(REPRO).round(3).to_string())
    print('\n=== deviation from the 300 m unmasked control ===')
    print(R.pivot_table(index='stage', columns=['mask', 'radius'],
                        values='delta_vs_300raw').reindex(REPRO).round(3).to_string())
    print(f'\nlargest |deviation| across all configurations: '
          f'{R.delta_vs_300raw.abs().max():.3f} R2')


if __name__ == '__main__':
    main()
