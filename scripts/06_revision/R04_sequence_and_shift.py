"""R04 - (a) sequential consistency of the per-stage predictions
        (b) why Colorado fails: covariate shift, not small samples
        (reviewers #5/1, #4/1, #2/4, #6/7).

(a) Reviewer #5: "do the stage predictions respect the sequential
    structure of the crop cycle? ... If the framework does not enforce
    that jointing occurs after tillering, then 'multi-stage' mainly
    describes the product, not the method."

    The eight regressors are indeed fitted independently. Rather than
    assert that this is harmless, we measure it: for every field-year we
    predict all eight stages with one fixed model and count how often the
    predicted sequence inverts the canonical developmental order, and how
    deep the inversions are. The same statistic was computed on the
    ground labels in R02 (0.5 %), which gives the reference the reviewer
    would want.

(b) Reviewers #4/#2: Colorado's negative LOSO R2 is dismissed in the
    submission as "small-sample instability of the R2 statistic".
    Reviewer #4: "n=333 is not an extremely small sample size in this
    field. I suspect there is a systematic distribution shift in the
    spatial feature space."

    We test the reviewer's hypothesis directly rather than defend the
    original wording:
      - a state-vs-rest classifier AUC on the feature matrix quantifies
        how separable each held-out state is (covariate shift);
      - the standardized target shift shows whether the held-out state's
        phenology dates lie outside the training range (concept shift);
      - the variance ratio shows whether R2 is being deflated simply
        because the held-out fold has a narrow target spread, which is
        the mechanism behind an "unstable" R2.

Output: data/revision/R04_sequence_consistency.csv
        data/revision/R04_state_shift.csv
"""
from _common import (LOSO_STATES, ORDER, STAGE_LABEL, cohort, group_map, save,
                     stage_frame)
from scripts.utils.deep_models import fold_pred_ensemble
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

FIXED = 'LightGBM'          # one model for all stages, so the sequence test
                            # is a property of the framework, not of a mix


def stage_predictions(fe, cols):
    """LOYO predictions for all eight stages, keyed by (field_id, year)."""
    out = {}
    for s in ORDER:
        d, tgt = stage_frame(fe, s)
        feat = cols(s, True)
        recs = []
        for yr in sorted(d['year'].unique()):
            tr, te = d[d['year'] != yr], d[d['year'] == yr]
            if len(tr) < 50 or len(te) < 5:
                continue
            pr = fold_pred_ensemble(tr, te, feat, tgt, FIXED)
            recs.append(pd.DataFrame(dict(field_id=te['field_id'].values,
                                          year=te['year'].values,
                                          obs=te[tgt].values, pred=pr)))
        if recs:
            out[s] = pd.concat(recs, ignore_index=True)
        print(f'  {STAGE_LABEL[s]:10s} predicted n={len(out.get(s, [])):5d}',
              flush=True)
    return out


def sequence_consistency(preds):
    wide = None
    for s, df in preds.items():
        c = df[['field_id', 'year', 'pred']].rename(columns={'pred': s})
        wide = c if wide is None else wide.merge(c, on=['field_id', 'year'], how='outer')
    rows, tot, viol = [], 0, 0
    depths = []
    pairs = {}
    for i in range(len(ORDER) - 1):
        for j in range(i + 1, len(ORDER)):
            a, b = ORDER[i], ORDER[j]
            if a not in wide or b not in wide:
                continue
            m = wide[[a, b]].dropna()
            n = len(m)
            v = int((m[a] > m[b]).sum())
            tot += n; viol += v
            depths.extend((m[a] - m[b])[m[a] > m[b]].tolist())
            pairs[f'{a}>{b}'] = dict(n=n, violations=v,
                                     pct=100 * v / n if n else np.nan)
            rows.append(dict(earlier=a, later=b, n_pairs=n, violations=v,
                             pct=100 * v / n if n else np.nan))
    return pd.DataFrame(rows), tot, viol, np.array(depths), wide


def state_shift(fe, cols):
    rows = []
    for s in ORDER:
        d, tgt = stage_frame(fe, s)
        feat = [c for c in cols(s, True)
                if c not in group_map(cols(s, True)).get('State', [])]
        X = d[feat]
        for st in LOSO_STATES:
            y = (d['state'] == st).astype(int).values
            n_in = int(y.sum())
            if n_in < 10 or n_in == len(y):
                rows.append(dict(stage=s, state=st, n=n_in, auc=np.nan,
                                 target_shift=np.nan, var_ratio=np.nan))
                continue
            pipe = Pipeline([('imp', SimpleImputer(strategy='median')),
                             ('sc', StandardScaler()),
                             ('clf', RandomForestClassifier(
                                 n_estimators=200, n_jobs=8, random_state=0,
                                 class_weight='balanced'))])
            p = cross_val_predict(pipe, X, y, cv=4, method='predict_proba',
                                  n_jobs=1)[:, 1]
            auc = roc_auc_score(y, p)
            t_in, t_out = d[tgt][y == 1], d[tgt][y == 0]
            pooled = np.sqrt((t_in.var() + t_out.var()) / 2)
            rows.append(dict(
                stage=s, state=st, n=n_in, auc=auc,
                target_shift=(t_in.mean() - t_out.mean()) / pooled if pooled else np.nan,
                var_ratio=t_in.var() / t_out.var() if t_out.var() else np.nan))
            print(f'  {STAGE_LABEL[s]:10s} {st}  n={n_in:5d} AUC={auc:.3f} '
                  f'shift={rows[-1]["target_shift"]:+.2f} '
                  f'var_ratio={rows[-1]["var_ratio"]:.2f}', flush=True)
    return pd.DataFrame(rows)


def main():
    fe, cols = cohort()

    print('=== (a) per-stage LOYO predictions (fixed model: %s) ===' % FIXED,
          flush=True)
    preds = stage_predictions(fe, cols)
    seq, tot, viol, depths, wide = sequence_consistency(preds)
    save(seq.round(2), 'R04_sequence_consistency.csv')
    print(f'\n  predicted ordered stage pairs: {tot}')
    print(f'  sequence inversions: {viol}  ({100 * viol / tot:.1f} %)')
    if len(depths):
        print(f'  median inversion depth: {np.median(depths):.1f} d  '
              f'(p90 {np.percentile(depths, 90):.1f} d)')
    print('\n  worst pairs:')
    print(seq.sort_values("pct", ascending=False).head(6).round(2).to_string(index=False))

    print('\n=== (b) held-out-state shift diagnostics ===', flush=True)
    sh = state_shift(fe, cols)
    save(sh.round(3), 'R04_state_shift.csv')
    print('\n  mean separability (AUC) by state, over stages:')
    print(sh.groupby('state')[['auc', 'var_ratio']].mean().round(3).to_string())


if __name__ == '__main__':
    main()
