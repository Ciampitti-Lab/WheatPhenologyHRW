"""R14e - Final Table 3, Figure 4 tally and LOSO under the corrected states.

Assembles the numbers the manuscript reports, from the corrected grid
(R14c + R14d) and the adoption declared in deep_models.ADOPT.

Output: data/revision/R14_table3.csv
        data/revision/R14_strategy_tally.csv
        data/revision/R14_loso_final.csv
"""
import sys, warnings
from pathlib import Path
warnings.filterwarnings('ignore')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np, pandas as pd
from _common import ORDER, STAGE_LABEL, cohort, r2, save, stage_frame
from scripts.utils.deep_models import (ADOPT, STATE_OH, SEEDS, DEEP,
                                       fold_pred_ensemble, fold_pred_ft,
                                       fold_pred_tabnet)

REV = ROOT / 'data' / 'revision'
STUDY = ['TX', 'OK', 'KS', 'NE', 'CO']
LABEL = {'C_Hybrid': 'Physiology-informed', 'B_ML-only': 'Machine-learning-only'}


def main():
    g = pd.read_csv(REV / 'R14_grid_full.csv')

    # --- Table 3 -----------------------------------------------------------
    rows = []
    for st in ORDER:
        strat, mod = ADOPT[st]
        r = g[(g.stage == st) & (g.strategy == strat) & (g.model == mod)].iloc[0]
        rows.append(dict(stage=st, strategy=LABEL[strat], model=mod,
                         R2=r.R2, R2_lo=r.R2_lo, R2_hi=r.R2_hi,
                         RMSE=r.RMSE, n=int(r.n)))
    T = pd.DataFrame(rows)
    save(T.round(3), 'R14_table3.csv')
    print('=== Table 3 (corrected states) ===')
    print(T.round(3).to_string(index=False))

    # --- Figure 4 tally: best of seven per strategy -------------------------
    p = g.pivot_table(index='stage', columns='strategy', values='R2',
                      aggfunc='max').reindex(ORDER)
    p['delta'] = p['C_Hybrid'] - p['B_ML-only']
    p['winner'] = ['Hybrid' if d > 0.005 else ('ML-only' if d < -0.005 else 'tie')
                   for d in p.delta]
    save(p.round(3), 'R14_strategy_tally.csv', index=True)
    print('\n=== Figure 4: best per strategy ===')
    print(p.round(3).to_string())
    print('tally:', dict(p.winner.value_counts()))

    # --- LOSO for the adopted model ----------------------------------------
    fe, cols = cohort()
    lo = []
    for st in ORDER:
        strat, mod = ADOPT[st]
        d, tgt = stage_frame(fe, st)
        feat = [c for c in cols(st, strat == 'C_Hybrid') if c in d.columns]
        for held in STUDY:
            tr, te = d[d.state != held].copy(), d[d.state == held].copy()
            if len(tr) < 50 or len(te) < 5:
                lo.append(dict(stage=st, state=held, model=mod,
                               R2=np.nan, n=len(te)))
                continue
            for c in STATE_OH:
                if c in feat:
                    te[c] = 0.0
            if mod in DEEP:
                fn = fold_pred_tabnet if mod == 'TabNet' else fold_pred_ft
                ps = []
                for sd in SEEDS:
                    try:
                        ps.append(fn(tr, te, feat, tgt, sd))
                    except Exception:
                        pass
                if not ps:
                    continue
                pred = np.mean(ps, axis=0)
            else:
                pred = fold_pred_ensemble(tr, te, feat, tgt, mod)
            lo.append(dict(stage=st, state=held, model=mod,
                           R2=r2(te[tgt].values, pred), n=len(te)))
        print(f'  LOSO {STAGE_LABEL[st]:10s} done', flush=True)
    L = pd.DataFrame(lo)
    save(L.round(3), 'R14_loso_final.csv')
    print('\n=== LOSO, adopted model, corrected states ===')
    print(L.pivot_table(index='stage', columns='state',
                        values='R2').reindex(ORDER)[STUDY].round(3).to_string())


if __name__ == '__main__':
    main()
