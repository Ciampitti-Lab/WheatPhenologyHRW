"""R01 - Per-source component ablation  (reviewer #6 major 5; AE summary).

Reviewer #6: "The main comparison is between ML-only and WES+ML, which
does not clearly show how much independent predictive information is
provided by HLS remote sensing. Comparisons among HLS-only,
Weather-only, WES-only, HLS+Weather, Weather+WES, and the full hybrid
model would help isolate the contribution of each component."

Design
------
Input blocks (definition shared with the Figure-5 importance groups via
`_common.feature_group`, so ablation and importance can no longer drift):

    HLS      - VI phenometrics + double-logistic shape parameters
    Weather  - Daymet aggregates + MODIS LST + thermal-time-at-SOS terms
    WES      - the eight Wang-Engel-Streck stage predictions
    Context  - state one-hot encoders + site geometry (lat/lon, topsoil pH)

Every variant is evaluated under the *identical* LOYO protocol as the
manuscript, with the model held fixed per stage so the comparison is a
feature-set contrast and not a model contrast. Two fixed models are run
(LightGBM = the tree workhorse, ElasticNet = the interpretable linear
baseline) so no conclusion rests on a single inductive bias.

`Context` is reported both ways: the "pure" family isolates each data
source on its own, the "+ctx" family adds the context block that the
manuscript's full model always carries.

Output: data/revision/R01_component_ablation.csv
        data/revision/R01_component_summary.csv
"""
import time

from _common import (ORDER, STAGE_LABEL, cohort, group_map, metrics, save,
                     stage_frame)
from scripts.utils.deep_models import fold_pred_ensemble
import numpy as np
import pandas as pd

MODELS = ['LightGBM', 'ElasticNet']
# name -> source blocks (Context handled separately by the `ctx` flag)
VARIANTS = {
    'WES_only':      ['WES'],
    'HLS_only':      ['HLS'],
    'Weather_only':  ['Weather'],
    'HLS+Weather':   ['HLS', 'Weather'],
    'HLS+WES':       ['HLS', 'WES'],
    'Weather+WES':   ['Weather', 'WES'],
    'ALL':           ['HLS', 'Weather', 'WES'],
    'ALL_noWES':     ['HLS', 'Weather'],
}


def blocks(feature_cols):
    """Collapse the six provenance groups into the four reviewer blocks."""
    g = group_map(feature_cols)
    return {
        'HLS': g.get('HLS', []),
        'Weather': g.get('Daymet', []) + g.get('LST', []) + g.get('ThermalTime', []),
        'WES': g.get('WES', []),
        'Context': g.get('State', []) + g.get('Site', []),
    }


def loyo_fixed(d, feat, tgt, model):
    T, P = [], []
    for yr in sorted(d['year'].unique()):
        tr, te = d[d['year'] != yr], d[d['year'] == yr]
        if len(tr) < 50 or len(te) < 5:
            continue
        P.extend(fold_pred_ensemble(tr, te, feat, tgt, model))
        T.extend(te[tgt].values)
    return np.asarray(T, float), np.asarray(P, float)


def main():
    fe, cols = cohort()
    rows = []
    for s in ORDER:
        d, tgt = stage_frame(fe, s)
        B = blocks(cols(s, True))          # hybrid column set for this stage
        print(f'\n=== {STAGE_LABEL[s]}  (n={len(d)}) ===', flush=True)
        print('    blocks: ' + ', '.join(f'{k}={len(v)}' for k, v in B.items()),
              flush=True)
        for vname, parts in VARIANTS.items():
            for ctx in (False, True):
                feat = [c for p in parts for c in B[p]]
                if ctx:
                    feat = feat + B['Context']
                if not feat:
                    continue
                for m in MODELS:
                    t0 = time.time()
                    T, P = loyo_fixed(d, feat, tgt, m)
                    if len(T) == 0:
                        continue
                    r = metrics(T, P)
                    r.update(stage=s, variant=vname, context=ctx, model=m,
                             n_features=len(feat), sec=round(time.time() - t0, 1))
                    rows.append(r)
                    print(f'    {vname:13s} ctx={int(ctx)} {m:11s} '
                          f'R2={r["R2"]:+.3f} RMSE={r["RMSE"]:5.1f} '
                          f'({r["sec"]}s)', flush=True)
        pd.DataFrame(rows).to_csv(
            '/home/vmangidi/repositories/WheatPhenologyHRW/data/revision/'
            'R01_component_ablation.csv', index=False)

    df = pd.DataFrame(rows)
    save(df, 'R01_component_ablation.csv')

    # Headline view: with context (the manuscript's operating point), best
    # of the two fixed models per (stage, variant).
    w = df[df.context].copy()
    piv = (w.groupby(['stage', 'variant'])['R2'].max().unstack()
             .reindex(ORDER)[list(VARIANTS)])
    save(piv.round(3), 'R01_component_summary.csv', index=True)
    print('\n=== LOYO R2 by input block (with context; best of LightGBM/ElasticNet) ===')
    print(piv.round(3).to_string())

    print('\n=== marginal contribution (ALL minus ALL-without-block) ===')
    marg = pd.DataFrame({
        'drop_WES': piv['ALL'] - piv['ALL_noWES'],
        'drop_HLS': piv['ALL'] - piv['Weather+WES'],
        'drop_Weather': piv['ALL'] - piv['HLS+WES'],
    }).round(3)
    save(marg, 'R01_marginal_contribution.csv', index=True)
    print(marg.to_string())


if __name__ == '__main__':
    main()
