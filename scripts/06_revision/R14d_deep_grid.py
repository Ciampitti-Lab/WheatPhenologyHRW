"""R14d - TabNet and FT-Transformer with the corrected state assignment.

Completes the R14c grid. Anthesis and maturity adopt the FT-Transformer,
so these runs decide whether the published 0.82 and 0.44 survive the state
correction. Needs a GPU.

Output: data/revision/R14_grid_deep.csv
        data/revision/R14_loso_deep.csv
"""
import sys, time, warnings
from pathlib import Path
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np, pandas as pd
from _common import ORDER, STAGE_LABEL, cohort, metrics, r2, save, stage_frame
from scripts.utils.deep_models import loyo, STATE_OH, SEEDS, fold_pred_ft, fold_pred_tabnet

REV = ROOT / 'data' / 'revision'
STRATS = [('B_ML-only', False), ('C_Hybrid', True)]
STUDY = ['TX', 'OK', 'KS', 'NE', 'CO']


def main():
    import torch
    print('CUDA:', torch.cuda.is_available(), flush=True)
    fe, cols = cohort()
    rows = []
    for st in ORDER:
        d, tgt = stage_frame(fe, st)
        for sname, wes in STRATS:
            feat = [c for c in cols(st, wes) if c in d.columns]
            for m in ['TabNet', 'FT']:
                t0 = time.time()
                T, P = loyo(d, feat, tgt, m)
                if len(T) == 0:
                    continue
                r = metrics(T, P)
                r.update(stage=st, strategy=sname, model=m,
                         sec=round(time.time() - t0, 1))
                rows.append(r)
                print(f'  {STAGE_LABEL[st]:10s} {sname:9s} {m:7s} '
                      f'R2={r["R2"]:+.3f} ({r["sec"]}s)', flush=True)
                pd.DataFrame(rows).to_csv(REV / 'R14_grid_deep.csv', index=False)
    G = pd.DataFrame(rows)
    save(G.round(4), 'R14_grid_deep.csv')

    # LOSO for the FT arm at the two stages that adopt it
    lo = []
    for st in ['anthesis', 'maturity', 'flag_leaf', 'heading']:
        d, tgt = stage_frame(fe, st)
        feat = [c for c in cols(st, True) if c in d.columns]
        for held in STUDY:
            tr, te = d[d.state != held].copy(), d[d.state == held].copy()
            if len(tr) < 50 or len(te) < 5:
                continue
            for c in STATE_OH:
                if c in feat:
                    te[c] = 0.0
            ps = []
            for s in SEEDS:
                try:
                    ps.append(fold_pred_ft(tr, te, feat, tgt, s))
                except Exception:
                    pass
            if not ps:
                continue
            lo.append(dict(stage=st, state=held, model='FT',
                           R2=r2(te[tgt].values, np.mean(ps, axis=0)), n=len(te)))
            print(f'  LOSO {st:10s} {held} R2={lo[-1]["R2"]:+.3f}', flush=True)
            pd.DataFrame(lo).to_csv(REV / 'R14_loso_deep.csv', index=False)
    save(pd.DataFrame(lo).round(4), 'R14_loso_deep.csv')


if __name__ == '__main__':
    main()
