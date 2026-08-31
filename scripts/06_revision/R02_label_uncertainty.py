"""R02 - Quantifying ground-label uncertainty  (reviewers #2/2, #5/4, #6/2).

Reviewer #6: "These dates do not necessarily represent the actual
phenological transition, particularly given differences in observation
frequency, label definition, and sample size among stages. The
uncertainty in these reference labels should be quantified."
Reviewer #2: "provide no estimate of inter-observer agreement or label
reliability."
Reviewer #5: the repeat-observation proxy "cannot separate label noise
from real within-field phenological variation."

We have no repeat visits by independent observers, so inter-observer
agreement is not estimable from this dataset - and we say so. What *is*
estimable, and is the operative uncertainty for a date-valued target, is
the **visit-interval censoring bound**: the earliest-label rule places the
target at the first visit that reported stage S, while the true onset lies
somewhere in the open interval since the previous visit to that field.
That interval width is a hard, observer-independent lower bound on the
uncertainty of every target date, and unlike the repeat-observation proxy
of Supplementary S5 it does not confound label noise with genuine
within-field spread (reviewer #5's objection).

Three diagnostics:

  (a) censoring interval  - width of the bracket containing the true onset
  (b) same-stage span     - within-(field, year) spread of repeat labels,
                            i.e. reported stage duration + timing scatter
  (c) order violations    - field-years whose observed stage sequence
                            contradicts the canonical developmental order,
                            a direct, assumption-free label-consistency rate

Output: data/revision/R02_label_uncertainty.csv
        data/revision/R02_order_violations.csv
"""
from _common import ORDER, PHENO, STAGE_LABEL, STAGE_MAP, SPRING, save, OUT
from scripts.utils.deep_models import ADOPT
import numpy as np
import pandas as pd

CANON = {s: i for i, s in enumerate(ORDER)}


def load_obs():
    ph = pd.read_parquet(PHENO)
    ph['year'] = ph['growing_season'].str.split('-').str[1].astype(int)
    ph['field_id'] = ph['FIELDID'].astype(str)
    ph = ph[ph['year'].between(2014, 2017)].copy()
    stage_of = {}
    for s, labels in STAGE_MAP.items():
        for l in labels:
            stage_of[l] = s
    ph['stage'] = ph['growth_stage'].map(stage_of)
    return ph


def main():
    ph = load_obs()
    print(f'observations 2014-2017: {len(ph)}   '
          f'field-years: {ph.groupby(["field_id", "year"]).ngroups}')
    print(f'mapped to a modelled stage: {ph["stage"].notna().sum()} '
          f'({ph["stage"].notna().mean():.1%})\n')

    # visit calendar per field-year (all observations, whatever the label)
    visits = (ph.groupby(['field_id', 'year'])['dos']
                .apply(lambda x: np.sort(x.unique())).to_dict())
    nvis = ph.groupby(['field_id', 'year'])['dos'].nunique()
    print(f'visits per field-year: median={nvis.median():.0f} '
          f'mean={nvis.mean():.1f}  p90={nvis.quantile(.9):.0f}\n')

    rows = []
    for s in ORDER:
        sub = ph[ph['stage'] == s].copy()
        if s in SPRING:
            sub = sub[sub['dos'] > 200]
        if s == 'maturity':
            sub = sub[sub['dos'] >= 280]
        if sub.empty:
            continue

        # (a) censoring interval for the earliest-label target
        first = sub.groupby(['field_id', 'year'])['dos'].min()
        widths, censored = [], 0
        for (f, y), t1 in first.items():
            v = visits[(f, y)]
            prev = v[v < t1]
            if len(prev) == 0:
                censored += 1            # no prior visit -> left-censored
            else:
                widths.append(t1 - prev.max())
        widths = np.array(widths, float)

        # (b) same-stage span within a field-year
        span = sub.groupby(['field_id', 'year'])['dos'].agg(lambda x: x.max() - x.min())
        rep = span[sub.groupby(['field_id', 'year'])['dos'].count() > 1]

        rows.append(dict(
            stage=s, n_field_years=len(first),
            interval_median=np.median(widths) if len(widths) else np.nan,
            interval_mean=widths.mean() if len(widths) else np.nan,
            interval_p90=np.percentile(widths, 90) if len(widths) else np.nan,
            pct_left_censored=100 * censored / len(first),
            n_repeat=len(rep),
            repeat_span_median=rep.median() if len(rep) else np.nan,
            repeat_span_mean=rep.mean() if len(rep) else np.nan))

    u = pd.DataFrame(rows)
    # Model RMSE for the noise-floor comparison. This was hardcoded from the
    # table as submitted, so it survived the state correction of audit item 14
    # and left Table 7 disagreeing with Table 3 at five stages. Read it from
    # the adopted cell of the grid instead, so it cannot drift again.
    grid = pd.read_csv(OUT / 'R14_grid_full.csv')
    rmse = {}
    for st, (strat, mod) in ADOPT.items():
        c = grid[(grid.stage == st) & (grid.strategy == strat) & (grid.model == mod)]
        rmse[st] = round(float(c.RMSE.iloc[0]), 1)
    u['model_RMSE'] = u.stage.map(rmse)
    # a visit-interval-limited RMSE floor: onset uniform on the bracket
    u['rmse_floor'] = u.interval_mean / np.sqrt(12)
    u['pct_of_error_explained'] = 100 * u.rmse_floor / u.model_RMSE
    save(u.round(2), 'R02_label_uncertainty.csv')
    print('=== (a)/(b) label uncertainty by stage ===')
    print(u.round(2).to_string(index=False))

    # (c) canonical-order violations
    lab = ph.dropna(subset=['stage']).copy()
    lab = lab[~((lab.stage.isin(SPRING)) & (lab.dos <= 200))]
    lab = lab[~((lab.stage == 'maturity') & (lab.dos < 280))]
    firsts = (lab.groupby(['field_id', 'year', 'stage'])['dos'].min()
                 .reset_index())
    firsts['rank'] = firsts.stage.map(CANON)
    viol, tot = [], 0
    for (f, y), g in firsts.groupby(['field_id', 'year']):
        g = g.sort_values('rank')
        if len(g) < 2:
            continue
        for i in range(len(g) - 1):
            for j in range(i + 1, len(g)):
                tot += 1
                a, b = g.iloc[i], g.iloc[j]
                if a.dos > b.dos:        # earlier stage observed later
                    viol.append(dict(field_id=f, year=y, earlier=a.stage,
                                     later=b.stage, dos_earlier=a.dos,
                                     dos_later=b.dos, inversion=a.dos - b.dos))
    v = pd.DataFrame(viol)
    save(v, 'R02_order_violations.csv')
    print(f'\n=== (c) canonical-order consistency ===')
    print(f'ordered stage pairs within a field-year: {tot}')
    print(f'inversions: {len(v)}  ({100 * len(v) / max(tot, 1):.1f} %)')
    if len(v):
        print(f'median inversion depth: {v.inversion.median():.0f} d')
        print('\nmost frequent inverted pairs:')
        print(v.groupby(['earlier', 'later']).size().sort_values(ascending=False)
               .head(8).to_string())


if __name__ == '__main__':
    main()
