"""R03 - Selection bias and a pre-specified single model
        (reviewers #6 major 4, #1 comment 3; AE summary).

Reviewer #6: "The best model and strategy for each stage are selected
according to LOYO R2, and the corresponding LOYO R2 values are then
reported as the final performance. This may introduce selection bias.
Nested cross-validation ... would provide a more reliable estimate."

Reviewer #1: "A more meaningful strategy would be to identify a single
model with strong and stable overall performance across all
phenological stages, or to establish a unified and clearly justified
model-selection criterion."

Both are answered by the same experiment, which reports three estimates
per stage:

  selected    max over the (strategy x model) grid, i.e. what the
              submitted Table 2 reports - optimistically biased by
              construction, because the maximum of correlated noisy
              estimates exceeds the expected performance of any one cell
  nested      nested leave-one-year-out: the (strategy, model) choice is
              re-made inside each outer fold, using only that fold's
              training years, and never sees the held-out season. This is
              an unbiased estimate of the *selection procedure*.
  prespecified a single (strategy, model) fixed a priori for all eight
              stages - no per-stage selection at all, therefore no
              selection bias, and it is reviewer #1's "single model with
              strong and stable overall performance".

The gap `selected - nested` is the selection optimism the reviewer asks
us to quantify.

Scope: the five tree/linear models, which run on CPU and cover six of the
eight adopted stages. TabNet/FT-Transformer are handled by the companion
GPU job (R03b) so that the nested estimate can include them at anthesis
and maturity.

Output: data/revision/R03_selection_integrity.csv
        data/revision/R03_prespecified_grid.csv
"""
import time

from _common import ORDER, STAGE_LABEL, cohort, metrics, r2, save, stage_frame
from scripts.utils.deep_models import fold_pred_ensemble
import numpy as np
import pandas as pd

CPU_MODELS = ['ElasticNet', 'Ridge', 'RandomForest', 'XGBoost', 'LightGBM']
STRATS = [('B_ML-only', False), ('C_Hybrid', True)]


def fit_predict(tr, te, feat, tgt, model):
    return fold_pred_ensemble(tr, te, feat, tgt, model)


def loyo_pooled(d, cols_fn, stage, tgt, strategy_wes, model):
    """Plain LOYO for one (strategy, model) cell."""
    feat = cols_fn(stage, strategy_wes)
    T, P = [], []
    for yr in sorted(d['year'].unique()):
        tr, te = d[d['year'] != yr], d[d['year'] == yr]
        if len(tr) < 50 or len(te) < 5:
            continue
        P.extend(fit_predict(tr, te, feat, tgt, model))
        T.extend(te[tgt].values)
    return np.asarray(T, float), np.asarray(P, float)


def nested_loyo(d, cols_fn, stage, tgt):
    """Outer LOYO; inside each outer fold, re-select (strategy, model) by
    an inner LOYO over the remaining seasons only."""
    T, P, picks = [], [], []
    years = sorted(d['year'].unique())
    for yr in years:
        tr_out, te_out = d[d['year'] != yr], d[d['year'] == yr]
        if len(tr_out) < 50 or len(te_out) < 5:
            continue
        inner_years = sorted(tr_out['year'].unique())
        best, best_score = None, -np.inf
        for sname, wes in STRATS:
            feat = cols_fn(stage, wes)
            for m in CPU_MODELS:
                it, ip = [], []
                for iy in inner_years:
                    itr, ite = tr_out[tr_out['year'] != iy], tr_out[tr_out['year'] == iy]
                    if len(itr) < 50 or len(ite) < 5:
                        continue
                    ip.extend(fit_predict(itr, ite, feat, tgt, m))
                    it.extend(ite[tgt].values)
                if not it:
                    continue
                sc = r2(np.asarray(it, float), np.asarray(ip, float))
                if sc > best_score:
                    best_score, best = sc, (sname, wes, m)
        if best is None:
            continue
        sname, wes, m = best
        picks.append(f'{yr}:{sname.split("_")[1]}/{m}')
        P.extend(fit_predict(tr_out, te_out, cols_fn(stage, wes), tgt, m))
        T.extend(te_out[tgt].values)
    return np.asarray(T, float), np.asarray(P, float), picks


def main():
    fe, cols = cohort()

    # ---- 1. full CPU grid (for `selected` and for the pre-specified scan)
    print('=== grid (5 CPU models x 2 strategies) ===', flush=True)
    grid = []
    for s in ORDER:
        d, tgt = stage_frame(fe, s)
        for sname, wes in STRATS:
            for m in CPU_MODELS:
                t0 = time.time()
                T, P = loyo_pooled(d, cols, s, tgt, wes, m)
                if len(T) == 0:
                    continue
                r = metrics(T, P)
                r.update(stage=s, strategy=sname, model=m,
                         sec=round(time.time() - t0, 1))
                grid.append(r)
        print(f'  {STAGE_LABEL[s]:10s} done', flush=True)
    G = pd.DataFrame(grid)
    save(G, 'R03_cpu_grid.csv')

    # ---- 2. pre-specified single (strategy, model) across all stages
    print('\n=== pre-specified single model, all stages ===')
    presc = []
    for sname, _ in STRATS:
        for m in CPU_MODELS:
            sub = G[(G.strategy == sname) & (G.model == m)].set_index('stage')
            if len(sub) < len(ORDER):
                continue
            r2s = sub.reindex(ORDER)['R2']
            presc.append(dict(strategy=sname, model=m, mean_R2=r2s.mean(),
                              median_R2=r2s.median(), min_R2=r2s.min(),
                              n_negative=int((r2s < 0).sum()),
                              **{f'R2_{k}': v for k, v in r2s.items()}))
    PS = pd.DataFrame(presc).sort_values('median_R2', ascending=False)
    save(PS.round(3), 'R03_prespecified_grid.csv')
    print(PS[['strategy', 'model', 'mean_R2', 'median_R2', 'min_R2',
              'n_negative']].round(3).to_string(index=False))
    champ = PS.iloc[0]
    print(f'\n  most stable single choice: {champ.strategy} / {champ.model}')

    # ---- 3. nested LOYO
    print('\n=== nested LOYO (selection re-made inside every outer fold) ===',
          flush=True)
    rows = []
    for s in ORDER:
        d, tgt = stage_frame(fe, s)
        t0 = time.time()
        T, P, picks = nested_loyo(d, cols, s, tgt)
        nested = metrics(T, P) if len(T) else dict(R2=np.nan, RMSE=np.nan)
        sel = G[G.stage == s].R2.max()
        pre = G[(G.stage == s) & (G.strategy == champ.strategy)
                & (G.model == champ.model)].R2
        rows.append(dict(stage=s,
                         R2_selected=sel,
                         R2_nested=nested['R2'],
                         optimism=sel - nested['R2'],
                         RMSE_nested=nested['RMSE'],
                         R2_prespecified=float(pre.iloc[0]) if len(pre) else np.nan,
                         inner_picks='; '.join(picks),
                         sec=round(time.time() - t0, 1)))
        print(f'  {STAGE_LABEL[s]:10s} selected={sel:+.3f} nested={nested["R2"]:+.3f} '
              f'optimism={sel - nested["R2"]:+.3f}  [{"; ".join(picks)}]', flush=True)
    R = pd.DataFrame(rows)
    save(R.round(3), 'R03_selection_integrity.csv')
    print(f'\nmean selection optimism (CPU grid): '
          f'{R.optimism.mean():+.3f} R2 units')


if __name__ == '__main__':
    main()
