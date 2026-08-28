"""R11 - Harder temporal-transfer tests  (reviewer #6 major 4, second half).

Reviewer #6 asked for "nested cross-validation or a truly independent
temporal evaluation using the 2018/19-2023/24 data already mentioned in
the manuscript". The nested cross-validation is R03. This script deals
with the second option, and with a claim in the submitted manuscript
that does not survive checking.

The ground phenology record ends at harvest 2018. Harvest years present
in the label file are 2013-2018, and harvest 2018 carries emergence
labels only (863 field-years; zero for every other stage). The
"extension cohort" for 2018/19-2023/24 contains predictors - HLS,
Daymet, MODIS LST - but no ground observations, so a reproductive-stage
evaluation on those seasons cannot be run by us or by anyone else with
this dataset. The submitted text implied otherwise and is corrected.

What is possible within the labelled record is a stricter temporal test
than the leave-one-year-out headline: leave-TWO-years-out. Each of the
six ways of splitting the four seasons 2v2 trains on half the seasons
and predicts the other half, so the model sees 3 to 4 fewer seasons of
climate variation than the LOYO protocol allows. The drop from LOYO to
L2YO measures how much of the reported skill depends on having three
training seasons rather than two, which is the practical question behind
the reviewer's request: how well does this transfer to a season unlike
those it was trained on?

Output: data/revision/R11_temporal_stress.csv
        data/revision/R11_label_coverage.csv
"""
import itertools
import warnings

warnings.filterwarnings('ignore')

from _common import (ORDER, PHENO, SPRING, STAGE_LABEL, STAGE_MAP, cohort,
                     metrics, r2, save, stage_frame)
from scripts.utils.deep_models import fold_pred_ensemble
import numpy as np
import pandas as pd

MODEL = 'LightGBM'          # pre-specified single model (R03)
YEARS = [2014, 2015, 2016, 2017]


def label_coverage():
    """Per-harvest-year, per-stage field-year counts in the raw record."""
    ph = pd.read_parquet(PHENO)
    ph['hy'] = ph['growing_season'].str.split('-').str[1].astype(int)
    ph['field_id'] = ph['FIELDID'].astype(str)
    rows = []
    for hy in sorted(ph.hy.unique()):
        h = ph[ph.hy == hy]
        rec = dict(harvest_year=hy, observations=len(h),
                   fields=h.FIELDID.nunique())
        for s in ORDER:
            x = h[h.growth_stage.isin(STAGE_MAP[s])]
            if s in SPRING:
                x = x[x.dos > 200]
            if s == 'maturity':
                x = x[x.dos >= 280]
            rec[s] = x.groupby(['field_id', 'hy']).ngroups
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    print('=== label coverage of the ground record ===')
    cov = label_coverage()
    save(cov, 'R11_label_coverage.csv')
    print(cov.to_string(index=False))
    print('\nThe record ends at harvest 2018, which carries emergence only.')
    print('A reproductive-stage evaluation on 2018/19-2023/24 is not possible.\n')

    fe, cols = cohort()
    rows = []
    for s in ORDER:
        d, tgt = stage_frame(fe, s)
        feat = cols(s, True)

        # LOYO reference under the same fixed model
        T, P = [], []
        for y in YEARS:
            tr, te = d[d.year != y], d[d.year == y]
            if len(tr) < 50 or len(te) < 5:
                continue
            P.extend(fold_pred_ensemble(tr, te, feat, tgt, MODEL))
            T.extend(te[tgt].values)
        loyo = metrics(np.asarray(T, float), np.asarray(P, float))

        # leave-two-years-out: all six 2v2 splits
        scores = []
        for test_years in itertools.combinations(YEARS, 2):
            tr = d[~d.year.isin(test_years)]
            te = d[d.year.isin(test_years)]
            if len(tr) < 50 or len(te) < 5:
                continue
            p = fold_pred_ensemble(tr, te, feat, tgt, MODEL)
            scores.append(dict(test_years='+'.join(map(str, test_years)),
                               r2=r2(te[tgt].values, p), n=len(te)))
        if not scores:
            continue
        sc = pd.DataFrame(scores)
        rows.append(dict(stage=s, R2_loyo=loyo['R2'], RMSE_loyo=loyo['RMSE'],
                         R2_l2yo_mean=sc.r2.mean(), R2_l2yo_min=sc.r2.min(),
                         R2_l2yo_max=sc.r2.max(), drop=loyo['R2'] - sc.r2.mean(),
                         n_splits=len(sc)))
        print(f'  {STAGE_LABEL[s]:10s} LOYO={loyo["R2"]:+.3f}  '
              f'L2YO mean={sc.r2.mean():+.3f} '
              f'[{sc.r2.min():+.3f}, {sc.r2.max():+.3f}]  '
              f'drop={loyo["R2"] - sc.r2.mean():+.3f}', flush=True)
        pd.DataFrame(rows).to_csv(
            '/home/vmangidi/repositories/WheatPhenologyHRW/data/revision/'
            'R11_temporal_stress.csv', index=False)

    R = pd.DataFrame(rows)
    save(R.round(3), 'R11_temporal_stress.csv')
    print(f'\nmean drop from LOYO to leave-two-years-out: '
          f'{R.drop.mean():+.3f} R2')
    rep = R[R.stage.isin(['flag_leaf', 'boot', 'heading', 'anthesis'])]
    print(f'reproductive stages only: {rep.drop.mean():+.3f} R2')


if __name__ == '__main__':
    main()
