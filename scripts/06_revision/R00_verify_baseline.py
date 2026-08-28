"""R00 - Baseline audit: does the submitted manuscript match the pipeline?

Run before any revision analysis. Checks, against the code that produced
the submission:

  A. Table 1 cohort sizes, including the effect of the 1-99 percentile
     target trim in `stage_frame` (undocumented in the manuscript).
  B. Table 2 adopted (strategy, model) versus the true per-stage argmax
     of the Phase-E grid  ->  the manuscript says selection is "strictly
     by held-out R^2"; the code applies an interpretability tie-break.
  C. The Discussion's model-stability claim against the actual grid.
  D. The declared feature inventory versus the columns actually fed to
     the models (manuscript Sec. 2.3.2 lists elevation; `ph_top` is used
     but never described).
  E. Whether the early stages really use "the full-season vector"
     (manuscript Sec. 2.3.5) or drop the window-aggregated features.

Output: data/revision/R00_audit.csv  (one row per finding)
"""
from _common import (ADOPT, EARLY, ORDER, STAGE_LABEL, WE, cohort, feature_group,
                     is_win, save, stage_frame)
import pandas as pd

GRID = ('/depot/ciampitti/data/WheatPhenologyHRW/data/raw/satellite/'
        'extension_2018_2024/deep_baseline/phaseE_full_grid.parquet')
findings = []


def add(check, verdict, manuscript, pipeline, note=''):
    findings.append(dict(check=check, verdict=verdict, manuscript=manuscript,
                         pipeline=pipeline, note=note))


def main():
    fe, cols = cohort()
    print(f'cohort: {fe.shape[0]} field-years, {fe["field_id"].nunique()} fields\n')

    # ---- A. cohort sizes / percentile trim -------------------------------
    TAB1 = dict(emergence=4145, tillering=4715, jointing=5124, flag_leaf=2657,
                boot=2320, heading=2714, anthesis=1464, maturity=357)
    print('=== A. cohort sizes ===')
    for s in ORDER:
        tgt = s + '_dos_obs'
        n_lab = int(fe[tgt].notna().sum())
        d, _ = stage_frame(fe, s)
        n_mod = len(d)
        ok = 'OK' if n_mod == TAB1[s] else 'MISMATCH'
        print(f'  {STAGE_LABEL[s]:10s} labelled={n_lab:5d}  after-trim={n_mod:5d} '
              f' Table1={TAB1[s]:5d}  {ok}')
        if n_mod != TAB1[s]:
            add(f'Table 1 n ({s})', 'MISMATCH', TAB1[s], n_mod,
                'stage_frame 1-99 percentile trim')
        if n_lab != n_mod:
            add(f'percentile trim ({s})', 'UNDOCUMENTED', 'no trim described',
                f'{n_lab} -> {n_mod}',
                'stage_frame() drops target outside [q01, q99]')

    # ---- B. adopted model vs argmax --------------------------------------
    print('\n=== B. adopted vs argmax (Phase-E grid) ===')
    g = pd.read_parquet(GRID)
    for s in ORDER:
        sub = g[g.stage == s]
        top = sub.loc[sub.R2.idxmax()]
        strat, mod = ADOPT[s]
        row = sub[(sub.strategy == strat) & (sub.model == mod)].iloc[0]
        same = (top.strategy == strat) and (top.model == mod)
        flag = 'argmax' if same else 'TIE-BREAK'
        print(f'  {STAGE_LABEL[s]:10s} adopted={strat:9s}/{mod:11s} R2={row.R2:.3f}'
              f'   argmax={top.strategy:9s}/{top.model:11s} R2={top.R2:.3f}   {flag}')
        if not same:
            add(f'Table 2 model ({s})', 'RULE NOT AS STATED',
                'selected strictly by held-out R^2',
                f'adopted {mod} (R2={row.R2:.3f}) over {top.model} (R2={top.R2:.3f})',
                f'delta={top.R2 - row.R2:.4f}; interpretability tie-break')

    # ---- C. stability claim ----------------------------------------------
    print('\n=== C. Discussion stability claim ===')
    fam = {'ElasticNet': 'regularized-linear', 'Ridge': 'regularized-linear',
           'LightGBM': 'gradient-boosted-tree', 'XGBoost': 'gradient-boosted-tree'}
    bad = g[(g.model.isin(fam)) & (g.R2 < 0)]
    for _, r in bad.sort_values('R2').iterrows():
        print(f'  NEGATIVE  {r.model:11s} {r.strategy:9s} {r.stage:10s} R2={r.R2:.3f}')
    if len(bad):
        worst = bad.loc[bad.R2.idxmin()]
        add('Discussion stability claim', 'CONTRADICTED',
            'only GBT and regularized-linear families are stable at every stage',
            f'{len(bad)} negative cells; worst {worst.model}/{worst.strategy}'
            f'/{worst.stage} R2={worst.R2:.2f}',
            'linear baselines collapse at emergence')

    # ---- D. feature inventory --------------------------------------------
    print('\n=== D. feature inventory ===')
    feats = cols('heading', True)
    groups = {}
    for c in feats:
        groups.setdefault(feature_group(c), []).append(c)
    for gname in sorted(groups):
        print(f'  {gname:12s} {len(groups[gname]):3d}')
    declared_site = {'latitude', 'longitude', 'elevation'}
    present = set(feats)
    missing = sorted(declared_site - present)
    if missing:
        add('Sec. 2.3.2 site geometry', 'DESCRIBES ABSENT FEATURE',
            'latitude, longitude, elevation', 'latitude, longitude only',
            f'not in feature matrix: {missing}')
        print(f'  DECLARED BUT ABSENT: {missing}')
    if 'ph_top' in present:
        add('Sec. 2.3 feature list', 'UNDECLARED FEATURE', 'no soil data described',
            'ph_top (topsoil pH) is an active model input',
            'no soil source cited in Methods; absent from Figure 5 groups')
        print('  UNDECLARED: ph_top (topsoil pH) is an active feature')

    # ---- E. "full-season vector" for the early stages ---------------------
    print('\n=== E. early-stage feature set ===')
    for s in ['emergence', 'heading']:
        n_all, n_win = len(cols(s, True)), len([c for c in cols(s, True) if is_win(c)])
        print(f'  {STAGE_LABEL[s]:10s} n_features={n_all:3d}  window-aggregated={n_win}')
    dropped = sorted(set(cols('heading', True)) - set(cols('emergence', True)))
    print(f'  dropped for early stages ({len(dropped)}): {dropped[:6]} ...')
    add('Sec. 2.3.5', 'CONTRADICTED',
        'Every stage still uses the full-season vector',
        f'{len(dropped)} window features dropped for {sorted(EARLY)}',
        'config early_stages_no_window; causally correct, but undescribed')

    print()
    out = pd.DataFrame(findings)
    save(out, 'R00_audit.csv')
    print('\n=== FINDINGS ===')
    for _, r in out.iterrows():
        print(f'  [{r.verdict}] {r.check}')


if __name__ == '__main__':
    main()
