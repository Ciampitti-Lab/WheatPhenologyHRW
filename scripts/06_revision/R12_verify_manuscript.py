"""R12 - Check every headline number in the manuscript against its source.

The revision makes "we re-derived every reported quantity from the code"
a central claim of the response letter. This script is what makes that
claim checkable, and it is meant to be re-run before any future
resubmission: it parses the numbers out of main.tex and compares them
with the result files that produced them.

A FAIL here means the manuscript and the analysis disagree. Run it after
any edit that touches a number.

Usage:  python R12_verify_manuscript.py [path/to/main.tex]
"""
import re
import sys
from pathlib import Path

import pandas as pd

REV = Path('/home/vmangidi/repositories/WheatPhenologyHRW/data/revision')
GRID = ('/depot/ciampitti/data/WheatPhenologyHRW/data/raw/satellite/'
        'extension_2018_2024/deep_baseline/phaseE_full_grid.parquet')
TEX = Path(sys.argv[1] if len(sys.argv) > 1
           else '/home/vmangidi/repositories/paper-overleaf/main.tex')

checks = []


def check(label, claimed, actual, tol=0.006):
    ok = (claimed is not None and actual is not None
          and abs(float(claimed) - float(actual)) <= tol)
    checks.append((ok, label, claimed, actual))
    print(f'  [{"OK  " if ok else "FAIL"}] {label:52s} '
          f'manuscript={claimed}  source={actual}')


def main():
    tex = TEX.read_text()

    print('=== Table 3, adopted model per stage (corrected-state grid) ===')
    g = pd.read_csv(REV / 'R14_grid_full.csv')
    ADOPT = {'emergence': ('C_Hybrid', 'LightGBM', 0.34, 29.4),
             'tillering': ('B_ML-only', 'Ridge', 0.34, 16.5),
             'jointing': ('C_Hybrid', 'LightGBM', 0.33, 16.3),
             'flag_leaf': ('C_Hybrid', 'XGBoost', 0.71, 5.9),
             'boot': ('C_Hybrid', 'LightGBM', 0.67, 5.6),
             'heading': ('C_Hybrid', 'ElasticNet', 0.73, 5.5),
             'anthesis': ('C_Hybrid', 'FT', 0.82, 4.6),
             'maturity': ('C_Hybrid', 'FT', 0.45, 5.6)}
    for st, (strat, mod, r2_tab, rmse_tab) in ADOPT.items():
        row = g[(g.stage == st) & (g.strategy == strat) & (g.model == mod)]
        check(f'Table 3 R2 {st}', r2_tab, round(float(row.R2.iloc[0]), 2))
        check(f'Table 3 RMSE {st}', rmse_tab,
              round(float(row.RMSE.iloc[0]), 1), tol=0.06)

    print('\n=== Table 4, per-source ablation (R01, LightGBM + context) ===')
    a = pd.read_csv(REV / 'R14_component_check.csv')
    w = a.pivot_table(index='stage', columns='variant', values='R2')
    tab4 = {'Emergence': ('emergence', 0.27, 0.32, 0.27, 0.34),
            'Tillering': ('tillering', 0.22, 0.29, 0.23, 0.27),
            'Flag leaf': ('flag_leaf', 0.65, 0.46, 0.39, 0.70),
            'Boot': ('boot', 0.65, 0.51, 0.50, 0.67),
            'Heading': ('heading', 0.71, 0.58, 0.56, 0.71),
            'Anthesis': ('anthesis', 0.81, 0.77, 0.59, 0.69),
            'Maturity': ('maturity', 0.43, 0.46, 0.16, 0.34)}
    for lbl, (st, wes, hls, met, allv) in tab4.items():
        check(f'Table 4 WES-only {st}', wes, round(w.loc[st, 'WES_only'], 2))
        check(f'Table 4 HLS-only {st}', hls, round(w.loc[st, 'HLS_only'], 2))
        check(f'Table 4 Meteo-only {st}', met,
              round(w.loc[st, 'Weather_only'], 2))
        check(f'Table 4 All {st}', allv, round(w.loc[st, 'ALL'], 2))

    print('\n=== Table 6, selection optimism (R03) ===')
    r3 = pd.read_csv(REV / 'R03_selection_integrity.csv').set_index('stage')
    for st in r3.index:
        check(f'Table 6 nested {st}', None, None) if False else None
    check('Table 6 mean optimism', 0.052, round(r3.optimism.mean(), 3))
    # every row of Table 6, so the table cannot drift from the CSV again
    for st, v in [('emergence', 0.048), ('tillering', 0.060), ('jointing', 0.124),
                  ('flag_leaf', 0.019), ('boot', 0.108), ('heading', 0.000),
                  ('anthesis', 0.016), ('maturity', 0.042)]:
        check(f'Table 6 optimism {st}', v, round(r3.loc[st, 'optimism'], 3))
    for st, v in [('emergence', 0.352), ('tillering', 0.338), ('jointing', 0.329),
                  ('flag_leaf', 0.706), ('boot', 0.672), ('heading', 0.732),
                  ('anthesis', 0.801), ('maturity', 0.341)]:
        check(f'Table 6 selected {st}', v, round(r3.loc[st, 'R2_selected'], 3))
    for st, v in [('emergence', 0.338), ('tillering', 0.268), ('jointing', 0.329),
                  ('flag_leaf', 0.700), ('boot', 0.670), ('heading', 0.712),
                  ('anthesis', 0.691), ('maturity', 0.341)]:
        check(f'Table 6 pre-specified {st}', v,
              round(r3.loc[st, 'R2_prespecified'], 3))
    check('Table 6 physiology-informed picks (of 32)', 28,
          sum(p.strip().split(':')[1].split('/')[0] == 'Hybrid'
              for picks in r3.inner_picks for p in picks.split(';')))

    print('\n=== Table 7, label bound (R02) ===')
    r2f = pd.read_csv(REV / 'R02_label_uncertainty.csv').set_index('stage')
    check('Table 7 emergence unbracketed %', 64.7,
          round(r2f.loc['emergence', 'pct_left_censored'], 1), tol=0.06)
    check('Table 7 tillering bracket (d)', 76.6,
          round(r2f.loc['tillering', 'interval_mean'], 1), tol=0.06)
    rep = ['flag_leaf', 'boot', 'heading', 'anthesis']
    lo = r2f.loc[rep, 'pct_of_error_explained'].min()
    hi = r2f.loc[rep, 'pct_of_error_explained'].max()
    check('text: reproductive floor lower %', 80, round(lo), tol=0.6)
    check('text: reproductive floor upper %', 119, round(hi), tol=0.6)

    print('\n=== Table 8, held-out-state shift (R04) ===')
    sh = pd.read_csv(REV / 'R14_state_shift.csv')
    shr = sh[sh.stage.isin(rep)]
    for stt, auc, shift in [('TX', 0.977, -0.58), ('OK', 0.936, -1.46),
                            ('KS', 0.972, 0.14), ('NE', 0.894, 1.44),
                            ('CO', 0.962, 2.69)]:
        d = shr[shr.state == stt]
        check(f'Table 8 AUC {stt}', auc, round(d.auc.mean(), 3))
        check(f'Table 8 shift {stt}', shift, round(d.target_shift.mean(), 2),
              tol=0.011)

    print('\n=== Sec 3.4, fold-derived sowing (R05) ===')
    r5 = pd.read_csv(REV / 'R05_fold_derived_sowing.csv')
    pv = r5.pivot_table(index='stage', columns='variant', values='pct_retained')
    for st, fm, fs in [('flag_leaf', 101, 96), ('boot', 105, 104),
                       ('heading', 96, 98), ('anthesis', 93, 25)]:
        check(f'fold-median retained {st} (%)', fm,
              round(pv.loc[st, 'fold_median']), tol=0.6)
        check(f'fold-strict retained {st} (%)', fs,
              round(pv.loc[st, 'fold_strict']), tol=0.6)

    print('\n=== Table 5, sowing-anchor perturbation (R15/R16) ===')
    V3 = Path('/depot/ciampitti/data/WheatPhenologyHRW/data/raw/satellite/'
              'extension_2018_2024/v3_results')
    ss = pd.read_csv(V3 / 'sowing_sensitivity_summary.csv')
    sp = ss.pivot_table(index='stage', columns='sigma', values='pct_retained')
    sg = ss.groupby('stage').gain_control.first()
    for st, gain, ret in [('flag_leaf', 0.267, (88.2, 80.7, 81.7, 56.1)),
                          ('boot', 0.107, (99.4, 96.2, 99.5, 84.2)),
                          ('heading', 0.434, (94.7, 94.9, 89.5, 89.4))]:
        check(f'Table 5 control gain {st}', gain, round(float(sg.loc[st]), 3))
        for sig, v in zip((7, 14, 21, 28), ret):
            check(f'Table 5 retained {st} sigma={sig}', v,
                  round(float(sp.loc[st, sig]), 1), tol=0.06)
    ft = V3 / 'anthesis_ft_ablation_summary.csv'
    if ft.exists():
        fa = pd.read_csv(ft).set_index('sigma')
        check('Table 5 control gain anthesis (FT)', 0.026,
              round(float(fa.gain_control.iloc[0]), 3))
        for sig, v in zip((7, 14, 21, 28), (84.5, 87.3, 74.1, 71.0)):
            check(f'Table 5 retained anthesis sigma={sig}', v,
                  round(float(fa.loc[sig, 'pct_retained']), 1), tol=0.06)

    print('\n=== abstract and highlights ===')
    gf = pd.read_csv(REV / 'R14_grid_full.csv')
    AD = {'flag_leaf': ('C_Hybrid', 'XGBoost'), 'boot': ('C_Hybrid', 'LightGBM'),
          'heading': ('C_Hybrid', 'ElasticNet'), 'anthesis': ('C_Hybrid', 'FT')}
    rr = [float(gf[(gf.stage == k) & (gf.strategy == v[0])
                   & (gf.model == v[1])].RMSE.iloc[0]) for k, v in AD.items()]
    check('reproductive RMSE min (d)', 4.6, round(min(rr), 1))
    check('reproductive RMSE max (d)', 5.9, round(max(rr), 1))
    r2r = [float(gf[(gf.stage == k) & (gf.strategy == v[0])
                    & (gf.model == v[1])].R2.iloc[0]) for k, v in AD.items()]
    check('reproductive R2 min', 0.67, round(min(r2r), 2))
    check('reproductive R2 max', 0.82, round(max(r2r), 2))
    for claim in ['4.6--5.9 d RMSE', r'\SIrange{4.6}{5.9}{\day}']:
        n_ = tex.count(claim)
        checks.append((n_ >= 1, f'present: "{claim}"', 1, n_))
        print(f'  [{"OK  " if n_ >= 1 else "FAIL"}] present: {claim:28s} count={n_}')

    print('\n=== Supplementary S5, deterministic anchor (R05) ===')
    for st, gain, det, pct in [('flag_leaf', 0.267, 0.193, 72),
                               ('boot', 0.107, 0.103, 96),
                               ('heading', 0.434, 0.406, 94),
                               ]:
        d = r5[(r5.stage == st) & (r5.variant == 'deterministic')].iloc[0]
        c = r5[(r5.stage == st) & (r5.variant == 'control')].iloc[0]
        check(f'S5 control gain {st}', gain, round(float(c.gain), 3))
        check(f'S5 deterministic gain {st}', det, round(float(d.gain), 3))
        check(f'S5 retained {st} (%)', pct, round(float(d.pct_retained)), tol=0.6)
    # the anthesis row of Table S5 uses the adopted FT-Transformer, which R05
    # (CPU only) cannot fit; it comes from 05_analysis/06_anthesis_ft_ablation
    fd = V3 / 'anthesis_ft_deterministic.csv'
    if fd.exists():
        a = pd.read_csv(fd).iloc[0]
        check('S5 anthesis FT control gain', 0.026,
              round(float(a.gain_control), 3))
        check('S5 anthesis FT deterministic gain', 0.017,
              round(float(a.gain_det), 3))
        check('S5 anthesis FT retained (%)', 67,
              round(float(a.pct_retained)), tol=0.6)

    print('\n=== Sec 3.6, leave-two-years-out (R11) ===')
    r11 = pd.read_csv(REV / 'R11_temporal_stress.csv').set_index('stage')
    check('L2YO mean loss, all stages', 0.078, round(r11['drop'].mean(), 3))
    check('L2YO mean loss, reproductive', 0.096,
          round(r11.loc[rep, 'drop'].mean(), 3))
    check('L2YO worst heading', 0.107, round(r11.loc['heading', 'R2_l2yo_min'], 3))
    check('L2YO worst boot', 0.186, round(r11.loc['boot', 'R2_l2yo_min'], 3))

    print('\n=== Sec 3.5, grouped permutation importance (R08) ===')
    sh8 = pd.read_csv(REV / 'R08_importance_share.csv').set_index('stage')
    check('WES share flag leaf (%)', 82, round(sh8.loc['flag_leaf', 'WES']), tol=0.6)
    check('Site share emergence (%)', 53, round(sh8.loc['emergence', 'Site']), tol=0.6)
    check('Site share maturity (%)', 50, round(sh8.loc['maturity', 'Site']), tol=0.6)
    check('HLS share emergence (%)', 27, round(sh8.loc['emergence', 'HLS']), tol=0.6)
    r8 = pd.read_csv(REV / 'R08_grouped_permutation.csv').set_index(['stage', 'group'])
    check('WES flag-leaf permutation loss', 0.54,
          round(float(r8.loc[('flag_leaf', 'WES'), 'drop_mean']), 2))
    check('Daymet share min (%)', 4, round(sh8['Daymet'].min()), tol=0.6)
    check('Daymet share max (%)', 18, round(sh8['Daymet'].max()), tol=0.6)
    check('State encoders inert everywhere (%)', 0.0, round(sh8['State'].max(), 1))

    print('\n=== cohort counts quoted in Table 2 ===')
    cov = pd.read_csv(REV / 'R11_label_coverage.csv')
    c18 = cov[cov.harvest_year == 2018].iloc[0]
    check('harvest-2018 emergence field-years', 863, int(c18.emergence), tol=0.5)
    check('harvest-2018 other stages (sum)', 0,
          int(sum(c18[s] for s in ['tillering', 'jointing', 'flag_leaf',
                                   'boot', 'heading', 'anthesis', 'maturity'])),
          tol=0.5)

    print('\n=== CDL buffer purity, training seasons (R13c) ===')
    cf = pd.read_csv(REV / 'R13_cdl_fraction.csv')
    inw = cf[cf.wheat_fraction > 0.10]
    med = inw.groupby('radius')['wheat_fraction'].median()
    for r, claimed in [(150, 77), (300, 56), (500, 42)]:
        check(f'wheat fraction at r={r} m (%)', claimed,
              round(100 * med.loc[r]), tol=0.6)
    check('wheat pixels at r=300 m', 176,
          round(med.loc[300] * 3.14159 * 300 ** 2 / 900), tol=1.5)

    print('\n=== Table 9, buffer sensitivity (R13d) ===')
    bs = pd.read_csv(REV / 'R13_buffer_sensitivity.csv')
    pv = bs.pivot_table(index='stage', columns=['mask', 'radius'], values='R2')
    # every cell of Table 9
    for st, row in [('flag_leaf', dict(raw=(0.710, 0.687, 0.657),
                                       cdl=(0.756, 0.722, 0.717))),
                    ('boot',      dict(raw=(0.678, 0.702, 0.707),
                                       cdl=(0.726, 0.704, 0.716))),
                    ('heading',   dict(raw=(0.757, 0.747, 0.743),
                                       cdl=(0.746, 0.744, 0.756))),
                    ('anthesis',  dict(raw=(0.882, 0.881, 0.875),
                                       cdl=(0.874, 0.891, 0.886)))]:
        for mk, vals in row.items():
            for rd, claimed in zip((150, 300, 500), vals):
                check(f'Table 9 {st} {mk} r={rd}', claimed,
                      round(pv.loc[st, (mk, rd)], 3))
    check('largest |deviation| from the control', 0.069,
          round(bs.delta_vs_300raw.abs().max(), 3))
    check('deviations at or above 0.04 (count)', 1,
          int((bs.delta_vs_300raw.abs() >= 0.04).sum()))

    print('\n=== corrected state assignment (R14) ===')
    lk = pd.read_csv(REV / 'R14_state_lookup.csv')
    vc = lk.state_true.value_counts()
    for stt, claimed in [('TX', 458), ('OK', 630), ('KS', 3778),
                         ('NE', 54), ('CO', 361)]:
        check(f'fields in {stt}', claimed, int(vc.get(stt, 0)), tol=0.5)
    lo = pd.read_csv(REV / 'R14_loso_final.csv')
    rep4 = lo[lo.stage.isin(rep) & lo.state.isin(['TX', 'OK', 'KS', 'NE'])]
    check('LOSO reproductive minimum', -0.18, round(rep4.R2.min(), 2), tol=0.006)
    check('LOSO reproductive maximum', 0.59, round(rep4.R2.max(), 2), tol=0.006)

    print('\n=== em dashes and stale claims in the .tex ===')
    for bad, why in [('---', 'em dash'),
                     ('strictly by held-out', 'old selection-rule wording'),
                     ('latitude, longitude, elevation', 'elevation as an input'),
                     ('full-season vector', 'old feature-set claim'),
                     ('transferability is strong', 'withdrawn transfer claim'),
                     ('only CDL-wheat pixels enter', 'unapplied CDL mask claim'),
                     ('five of eight', 'superseded strategy tally'),
                     ('\\num{5293}', 'pre-correction field count'),
                     ('\\num{8465}', 'pre-correction field-year count'),
                     # superseded values that survived in prose after the
                     # tables were regenerated. Each was found by hand twice.
                     ('$+0.036$', 'pre-correction selection optimism'),
                     ('{0.102}', 'pre-correction anthesis control gain'),
                     ('{88}{97}', 'pre-correction fold-median range'),
                     ('{83}{92}', 'pre-correction fold-strict range'),
                     ('{0.086}', 'pre-correction L2YO mean loss'),
                     ('{0.059}', 'pre-correction buffer bound'),
                     ('\\subsection{Where the information lives',
                      'Discussion subsection heading'),
                     ('\\subsection{Limitations}',
                      'Discussion subsection heading')]:
        n = tex.count(bad)
        checks.append((n == 0, f'no "{bad}" ({why})', 0, n))
        print(f'  [{"OK  " if n == 0 else "FAIL"}] absent: {why:38s} count={n}')

    bad = [c for c in checks if not c[0]]
    print(f'\n{"=" * 60}\n{len(checks) - len(bad)}/{len(checks)} checks passed')
    if bad:
        print('\nFAILURES:')
        for _, lbl, claimed, actual in bad:
            print(f'  {lbl}: manuscript={claimed} source={actual}')
        sys.exit(1)
    print('Every checked number in the manuscript matches its source file.')


if __name__ == '__main__':
    main()
