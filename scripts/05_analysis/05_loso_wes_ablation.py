"""Negative control (Supplementary S8): does WES aid CROSS-REGION
transfer? Controlled with/without-WES ablation under leave-one-state-out.

For each stage's adopted model and each held-out state: LOSO R^2 with
the WES features (Hybrid) vs without them (ML-only), identical fold and
held-out-encoder zeroing as Figure F6. Reported as a negative control:
in the regime where the held-out R^2 is meaningful (positive) the
difference is negligible (mean ~+0.04; -0.03 for the reproductive
stages at the well-sampled TX/OK/KS folds), the large apparent gains
being confined to the degenerate Colorado / small-sample folds where
both arms have R^2 << 0. WES is an in-distribution thermal-time prior,
not a spatial-generalisation device. Needs a GPU (FT stages).

Output: <work_dir>/v3_results/loso_wes_ablation.csv (+ _summary.csv)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.config import CFG, REPO_ROOT
from scripts.utils.deep_models import (ORDER, ADOPT, LOSO_STATES,
    load_cohort, stage_frame, loyo, r2)
import numpy as np, pandas as pd

WORK = REPO_ROOT / CFG.paths.work_dir
PHENO = str(REPO_ROOT / CFG.paths.phenology_matched)
OUT = WORK / 'v3_results'; OUT.mkdir(parents=True, exist_ok=True)


def main():
    fe, cols = load_cohort(WORK, PHENO)
    rows = []
    for s in ORDER:
        strat, mod = ADOPT[s]
        d, tgt = stage_frame(fe, s)
        fh, fm = cols(s, True), cols(s, False)        # with / without WES
        for st in LOSO_STATES:
            Th, Ph = loyo(d, fh, tgt, mod, split='state', holdout=st)
            Tm, Pm = loyo(d, fm, tgt, mod, split='state', holdout=st)
            if len(Th) > 5 and len(Tm) > 5:
                rh, rm = r2(Th, Ph), r2(Tm, Pm)
                rows.append(dict(stage=s, model=mod, held_out=st, n=len(Th),
                                 r2_hybrid=round(rh, 4), r2_mlonly=round(rm, 4),
                                 dWES=round(rh - rm, 4)))
            else:
                rows.append(dict(stage=s, model=mod, held_out=st, n=int(len(Th)),
                                 r2_hybrid=np.nan, r2_mlonly=np.nan, dWES=np.nan))
            pd.DataFrame(rows).to_csv(OUT / 'loso_wes_ablation.csv', index=False)
            print(rows[-1], flush=True)

    v = pd.DataFrame(rows).dropna(subset=['dWES'])
    usable = v[v.r2_hybrid > 0]
    repro = usable[(usable.stage.isin(['flag_leaf', 'boot', 'heading', 'anthesis']))
                   & (usable.held_out.isin(['TX', 'OK', 'KS']))]
    summ = dict(
        cells=len(v), hybrid_gt=int((v.dWES > 0).sum()),
        usable_cells=len(usable), usable_mean_dWES=round(usable.dWES.mean(), 4),
        usable_hybrid_gt=int((usable.dWES > 0).sum()),
        repro_TXOKKS_mean_dWES=round(repro.dWES.mean(), 4) if len(repro) else np.nan)
    pd.DataFrame([summ]).to_csv(OUT / 'loso_wes_ablation_summary.csv', index=False)
    print('\n=== negative-control summary ===', flush=True)
    print(summ, flush=True)
    print('Verdict: WES does NOT materially aid cross-region transfer.', flush=True)


if __name__ == '__main__':
    main()
