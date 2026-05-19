"""Phase E — leave-one-year-out grid over 8 stages x 2 strategies x 7
models (ElasticNet, Ridge, RandomForest, XGBoost, LightGBM, TabNet,
FT-Transformer), with bootstrap 95% CIs. Deep models are evaluated
under the identical LOYO protocol (standardised I/O, inner-year early
stopping, 5-seed averaged). The per-stage best is the (strategy, model)
carried to the manuscript Table; FT-Transformer is adopted only where
it genuinely wins (anthesis, maturity) -- flag leaf / heading keep the
interpretable tree/linear model (FT only ties them at two decimals).
Then leave-one-state-out (LOSO) for the adopted best per stage.

Resumable: re-running picks up cells already in the grid parquet.

Outputs (in <work_dir>/v3_results/):
  phase_e_grid.parquet  /  multi_stage_models_a6_gs.parquet   (full grid + CIs)
  phase_e_best.csv                                            (adopted best/stage)
  phase_e_loso.csv                                            (LOSO, adopted best)
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.config import CFG, REPO_ROOT
from scripts.utils.deep_models import (MODELS, ORDER, ADOPT, LOSO_STATES,
    load_cohort, stage_frame, loyo, metrics, r2)
import numpy as np, pandas as pd

WORK = REPO_ROOT / CFG.paths.work_dir
PHENO = str(REPO_ROOT / CFG.paths.phenology_matched)
OUT = WORK / 'v3_results'; OUT.mkdir(parents=True, exist_ok=True)
GRID = OUT / 'phase_e_grid.parquet'
STRAT = [('B_ML-only', False), ('C_Hybrid', True)]


def main():
    t0 = time.time()
    fe, cols = load_cohort(WORK, PHENO)
    rows, done = [], set()
    if GRID.exists():
        rows = pd.read_parquet(GRID).to_dict('records')
        done = {(r['stage'], r['strategy'], r['model']) for r in rows}
        print(f'resume: {len(done)} cells already done', flush=True)

    for s in ORDER:
        d, tgt = stage_frame(fe, s)
        for sn, wes in STRAT:
            fc = cols(s, wes)
            for mn in MODELS:
                if (s, sn, mn) in done:
                    continue
                t1 = time.time()
                T, P = loyo(d, fc, tgt, mn)
                if len(T) == 0:
                    continue
                m = metrics(T, P)
                m.update(stage=s, strategy=sn, model=mn,
                         sec=round(time.time() - t1, 1))
                rows.append(m)
                pd.DataFrame(rows).to_parquet(GRID, index=False)
                pd.DataFrame(rows).to_parquet(
                    OUT / 'multi_stage_models_a6_gs.parquet', index=False)
                print(f"{s:10s} {sn:9s} {mn:11s} R2={m['R2']:.3f} "
                      f"CI[{m['R2_lo']:.3f},{m['R2_hi']:.3f}] n={m['n']} "
                      f"({m['sec']}s)", flush=True)

    g = pd.DataFrame(rows)
    best = []
    for s in ORDER:
        strat, mod = ADOPT[s]
        r = g[(g.stage == s) & (g.strategy == strat) & (g.model == mod)]
        if len(r):
            best.append(r.iloc[0].to_dict())
    bdf = pd.DataFrame(best)
    bdf.to_csv(OUT / 'phase_e_best.csv', index=False)
    print('\n=== adopted best per stage ===', flush=True)
    print(bdf[['stage', 'strategy', 'model', 'R2', 'R2_lo', 'R2_hi',
               'RMSE', 'n']].round(3).to_string(index=False), flush=True)

    lo = []
    for s in ORDER:
        strat, mod = ADOPT[s]
        d, tgt = stage_frame(fe, s)
        fc = cols(s, strat == 'C_Hybrid')
        for st in LOSO_STATES:
            T, P = loyo(d, fc, tgt, mod, split='state', holdout=st)
            rr = r2(T, P) if len(T) > 5 else np.nan
            lo.append(dict(stage=s, model=mod, held_out=st,
                           R2=rr, n=int(len(T))))
            print(f"LOSO {s:10s} {st}: "
                  f"R2={rr if np.isnan(rr) else round(rr,3)} n={len(T)}",
                  flush=True)
    pd.DataFrame(lo).to_csv(OUT / 'phase_e_loso.csv', index=False)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == '__main__':
    main()
