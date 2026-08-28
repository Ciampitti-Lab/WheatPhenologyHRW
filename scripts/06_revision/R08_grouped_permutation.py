"""R08 - Uniform grouped permutation importance  (reviewer #6, minor 3).

Reviewer #6: "Figure 5 combines gain importance, regression coefficients,
and permutation importance across different models. These measures are
not directly comparable after normalization. A consistent grouped
permutation importance or feature-group ablation approach would be
preferable."

The reviewer is right: the submitted Figure 5 normalizes split-gain (tree
models), $|\hat{\beta}|$ (linear models) and permutation importance
(deep models) onto one axis, and those three quantities do not share a
scale or even a sign convention. This script replaces all of them with a
single measure computed identically for every stage:

  grouped permutation importance = the loss in held-out R^2 when the
  columns of one source group are jointly shuffled in the test fold,
  averaged over the LOYO folds and over N_REPEATS shuffles.

Grouping the columns before shuffling is what makes the measure
interpretable here: the feature set contains many correlated columns
within a source (35 HLS phenometrics, 31 Daymet aggregates), and
single-column permutation splits their shared contribution arbitrarily.
Shuffling a whole group answers the question the figure is asked to
answer -- how much does this data source contribute -- and is on the same
R^2 scale as every other number in the paper.

The model is held fixed at LightGBM across all stages so that the panels
are comparable to each other as well as internally consistent.

Output: data/revision/R08_grouped_permutation.csv
        figures/F5_feature_importance.pdf  (regenerated)
"""
import warnings

warnings.filterwarnings('ignore')

from _common import ORDER, STAGE_LABEL, cohort, group_map, r2, save, stage_frame
from scripts.utils.deep_models import fold_pred_ensemble
import numpy as np
import pandas as pd

MODEL = 'LightGBM'
N_REPEATS = 10
SEED = 0
GROUPS = ['HLS', 'Daymet', 'LST', 'ThermalTime', 'WES', 'State', 'Site']
PRETTY = {'HLS': 'HLS phenometrics', 'Daymet': 'Daymet meteorology',
          'LST': 'MODIS LST', 'ThermalTime': 'Thermal-time state',
          'WES': 'WES simulator', 'State': 'State encoders',
          'Site': 'Site geometry'}


def main():
    fe, cols = cohort()
    rng = np.random.RandomState(SEED)
    rows = []

    for s in ORDER:
        d, tgt = stage_frame(fe, s)
        feat = cols(s, True)
        gm = group_map(feat)
        print(f'\n=== {STAGE_LABEL[s]} (n={len(d)}) ===', flush=True)

        # collect per-fold test frames and a fitted-model predictor
        base_T, base_P, folds = [], [], []
        for yr in sorted(d['year'].unique()):
            tr, te = d[d['year'] != yr], d[d['year'] == yr]
            if len(tr) < 50 or len(te) < 5:
                continue
            p = fold_pred_ensemble(tr, te, feat, tgt, MODEL)
            base_P.extend(p); base_T.extend(te[tgt].values)
            folds.append((tr, te))
        base = r2(np.asarray(base_T, float), np.asarray(base_P, float))
        print(f'  baseline LOYO R2 = {base:.3f}', flush=True)

        for g in GROUPS:
            gcols = gm.get(g, [])
            if not gcols:
                continue
            drops = []
            for rep in range(N_REPEATS):
                T, P = [], []
                for tr, te in folds:
                    tep = te.copy()
                    idx = rng.permutation(len(tep))
                    for c in gcols:
                        tep[c] = tep[c].to_numpy()[idx]
                    P.extend(fold_pred_ensemble(tr, tep, feat, tgt, MODEL))
                    T.extend(tep[tgt].values)
                drops.append(base - r2(np.asarray(T, float), np.asarray(P, float)))
            rows.append(dict(stage=s, group=g, n_cols=len(gcols),
                             baseline_R2=base, drop_mean=float(np.mean(drops)),
                             drop_sd=float(np.std(drops))))
            print(f'  {PRETTY[g]:20s} ({len(gcols):2d} cols)  '
                  f'dR2 = {np.mean(drops):+.4f} +/- {np.std(drops):.4f}',
                  flush=True)
        pd.DataFrame(rows).to_csv(
            '/home/vmangidi/repositories/WheatPhenologyHRW/data/revision/'
            'R08_grouped_permutation.csv', index=False)

    R = pd.DataFrame(rows)
    save(R.round(4), 'R08_grouped_permutation.csv')
    piv = R.pivot_table(index='stage', columns='group',
                        values='drop_mean').reindex(ORDER)[GROUPS]
    print('\n=== grouped permutation importance (loss in LOYO R2) ===')
    print(piv.round(3).to_string())
    # share of total positive importance, for the stacked-bar rendering
    pos = piv.clip(lower=0)
    share = 100 * pos.div(pos.sum(axis=1), axis=0)
    save(share.round(1), 'R08_importance_share.csv', index=True)
    print('\n=== as a share of total positive importance (%) ===')
    print(share.round(1).to_string())


if __name__ == '__main__':
    main()
