"""R14c - Full Phase-E grid and LOSO with the corrected state assignment.

Re-runs the grid that Table 3 adopts from, now that load_cohort applies the
spatial-join state assignment (R14a). The five tree and linear models run
here; TabNet and the FT-Transformer run in R14d on a GPU.

Output: data/revision/R14_grid_cpu.csv
        data/revision/R14_loso_cpu.csv
"""
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np, pandas as pd
from _common import ORDER, STAGE_LABEL, cohort, metrics, r2, save, stage_frame
from scripts.utils.deep_models import fold_pred_ensemble, STATE_OH

REV = ROOT / 'data' / 'revision'
MODELS = ['ElasticNet', 'Ridge', 'RandomForest', 'XGBoost', 'LightGBM']
STRATS = [('B_ML-only', False), ('C_Hybrid', True)]
STUDY = ['TX', 'OK', 'KS', 'NE', 'CO']


def loyo(d, feat, tgt, model):
    T, P = [], []
    for y in sorted(d.year.unique()):
        tr, te = d[d.year != y], d[d.year == y]
        if len(tr) < 50 or len(te) < 5:
            continue
        P.extend(fold_pred_ensemble(tr, te, feat, tgt, model))
        T.extend(te[tgt].values)
    return np.asarray(T, float), np.asarray(P, float)


def main():
    fe, cols = cohort()
    print(f'cohort: {fe.field_id.nunique()} fields, {len(fe)} field-years')
    print(fe.drop_duplicates('field_id').state.value_counts().to_string(), '\n')

    rows = []
    for st in ORDER:
        d, tgt = stage_frame(fe, st)
        for sname, wes in STRATS:
            feat = [c for c in cols(st, wes) if c in d.columns]
            for m in MODELS:
                t0 = time.time()
                T, P = loyo(d, feat, tgt, m)
                if len(T) == 0:
                    continue
                r = metrics(T, P)
                r.update(stage=st, strategy=sname, model=m,
                         sec=round(time.time() - t0, 1))
                rows.append(r)
        pd.DataFrame(rows).to_csv(REV / 'R14_grid_cpu.csv', index=False)
        best = max((r for r in rows if r['stage'] == st), key=lambda r: r['R2'])
        print(f'  {STAGE_LABEL[st]:10s} best {best["strategy"]:9s}/'
              f'{best["model"]:12s} R2={best["R2"]:+.3f} RMSE={best["RMSE"]:5.2f}',
              flush=True)

    G = pd.DataFrame(rows)
    save(G.round(4), 'R14_grid_cpu.csv')

    print('\n=== best CPU cell per stage (corrected states) ===')
    b = G.loc[G.groupby('stage').R2.idxmax()].set_index('stage').reindex(ORDER)
    print(b[['strategy', 'model', 'R2', 'RMSE', 'n']].round(3).to_string())

    print('\n=== LOSO for the best CPU cell ===')
    lo = []
    for st in ORDER:
        d, tgt = stage_frame(fe, st)
        row = b.loc[st]
        wes = row.strategy == 'C_Hybrid'
        feat = [c for c in cols(st, wes) if c in d.columns]
        for held in STUDY:
            tr, te = d[d.state != held].copy(), d[d.state == held].copy()
            if len(tr) < 50 or len(te) < 5:
                lo.append(dict(stage=st, state=held, R2=np.nan, n=len(te)))
                continue
            for c in STATE_OH:
                if c in feat:
                    te[c] = 0.0
            p = fold_pred_ensemble(tr, te, feat, tgt, row.model)
            lo.append(dict(stage=st, state=held, model=row.model,
                           R2=r2(te[tgt].values, p), n=len(te)))
        print(f'  {STAGE_LABEL[st]:10s} done', flush=True)
    L = pd.DataFrame(lo)
    save(L.round(4), 'R14_loso_cpu.csv')
    print('\n=== LOSO R2 (corrected states) ===')
    print(L.pivot_table(index='stage', columns='state',
                        values='R2').reindex(ORDER)[STUDY].round(3).to_string())


if __name__ == '__main__':
    main()
