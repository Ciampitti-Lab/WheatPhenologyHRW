"""R14f - Re-run the analyses that depend on the state assignment.

Three of the revision analyses use state directly and must be redone after
the correction (R14a):

  Table 8   held-out-state shift diagnostics (R04)
  Supp S15  fold-derived sowing anchor, whose state medians change (R05)
  Table 4   per-source ablation, whose Context block contains the encoders

The ablation is expected to be unchanged, since the encoders carry ~0 % of
the grouped permutation importance, but that is a prediction and is checked
here rather than assumed.

Output: data/revision/R14_state_shift.csv
        data/revision/R14_component_check.csv
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np, pandas as pd
from _common import ORDER, REPRODUCTIVE, STAGE_LABEL, cohort, group_map, metrics, save, stage_frame
from scripts.utils.deep_models import fold_pred_ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

REV = ROOT / 'data' / 'revision'
STUDY = ['TX', 'OK', 'KS', 'NE', 'CO']


def shift(fe, cols):
    rows = []
    for st in ORDER:
        d, tgt = stage_frame(fe, st)
        gm = group_map(cols(st, True))
        feat = [c for c in cols(st, True) if c not in gm.get('State', [])]
        X = d[feat]
        for s in STUDY:
            y = (d['state'] == s).astype(int).values
            if y.sum() < 10 or y.sum() == len(y):
                continue
            pipe = Pipeline([('imp', SimpleImputer(strategy='median')),
                             ('sc', StandardScaler()),
                             ('clf', RandomForestClassifier(
                                 n_estimators=200, n_jobs=8, random_state=0,
                                 class_weight='balanced'))])
            p = cross_val_predict(pipe, X, y, cv=4, method='predict_proba')[:, 1]
            ti, to = d[tgt][y == 1], d[tgt][y == 0]
            pooled = np.sqrt((ti.var() + to.var()) / 2)
            rows.append(dict(stage=st, state=s, n=int(y.sum()),
                             auc=roc_auc_score(y, p),
                             target_shift=(ti.mean() - to.mean()) / pooled if pooled else np.nan,
                             var_ratio=ti.var() / to.var() if to.var() else np.nan))
        print(f'  shift {STAGE_LABEL[st]:10s} done', flush=True)
    return pd.DataFrame(rows)


def components(fe, cols):
    """Table 4 check: LightGBM with context, all six input subsets."""
    VAR = {'WES_only': ['WES'], 'HLS_only': ['HLS'], 'Weather_only': ['Weather'],
           'HLS+Weather': ['HLS', 'Weather'], 'ALL': ['HLS', 'Weather', 'WES']}
    rows = []
    for st in ORDER:
        d, tgt = stage_frame(fe, st)
        gm = group_map(cols(st, True))
        B = {'HLS': gm.get('HLS', []),
             'Weather': gm.get('Daymet', []) + gm.get('LST', []) + gm.get('ThermalTime', []),
             'WES': gm.get('WES', []),
             'Context': gm.get('State', []) + gm.get('Site', [])}
        for name, parts in VAR.items():
            feat = [c for p in parts for c in B[p]] + B['Context']
            T, P = [], []
            for y in sorted(d.year.unique()):
                tr, te = d[d.year != y], d[d.year == y]
                if len(tr) < 50 or len(te) < 5:
                    continue
                P.extend(fold_pred_ensemble(tr, te, feat, tgt, 'LightGBM'))
                T.extend(te[tgt].values)
            rows.append(dict(stage=st, variant=name,
                             R2=metrics(np.asarray(T, float), np.asarray(P, float))['R2']))
        print(f'  components {STAGE_LABEL[st]:10s} done', flush=True)
    return pd.DataFrame(rows)


def main():
    fe, cols = cohort()
    print('=== held-out-state shift diagnostics ===', flush=True)
    S = shift(fe, cols)
    save(S.round(3), 'R14_state_shift.csv')
    print('\nmean over the reproductive stages:')
    print(S[S.stage.isin(REPRODUCTIVE)].groupby('state')[
        ['auc', 'target_shift', 'var_ratio']].mean().round(3).to_string())

    print('\n=== per-source ablation check ===', flush=True)
    C = components(fe, cols)
    save(C.round(3), 'R14_component_check.csv')
    piv = C.pivot_table(index='stage', columns='variant', values='R2').reindex(ORDER)
    print(piv[['WES_only', 'HLS_only', 'Weather_only', 'HLS+Weather', 'ALL']].round(2).to_string())
    old = pd.read_csv(REV / 'R01_component_ablation.csv')
    o = old[(old.context) & (old.model == 'LightGBM')].pivot_table(
        index='stage', columns='variant', values='R2').reindex(ORDER)
    d = (piv[['WES_only', 'HLS_only', 'Weather_only', 'ALL']]
         - o[['WES_only', 'HLS_only', 'Weather_only', 'ALL']])
    print('\ndeviation from the published-state ablation:')
    print(d.round(3).to_string())
    print(f'\nmax |deviation| = {d.abs().max().max():.3f}')


if __name__ == '__main__':
    main()
