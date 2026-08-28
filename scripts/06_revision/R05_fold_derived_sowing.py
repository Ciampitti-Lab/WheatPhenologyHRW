"""R05 - Sowing anchor derived strictly inside the training fold
        (reviewer #6 major 3; reviewer #2 comment 3).

Reviewer #6: "approximately 16.3% of the sowing dates are derived from
the same field-survey records used to define the phenology targets. The
current perturbation analysis mainly addresses the remaining 83.7% of
state-median fallback dates and therefore does not fully rule out
circular information introduced by the observed sowing dates. Additional
sensitivity tests using independent sowing dates or training-fold-derived
sowing dates are needed."

Two leakage channels exist and the submission closed only one:

  (1) the *state median* used for the 83.7 % fallback rows is computed
      over the whole cohort, so it carries information from the held-out
      season. Never addressed in the submission.
  (2) the 16.3 % observed anchors come from the same survey that produced
      the targets. Supplementary S4 already replaces every observed
      anchor by the state median globally (88-98 % of the gain retained),
      but not fold-wise.

Variants (the anchor recipe is applied to *every* row of a fold, training
and held-out alike, so the two arms never see differently-scaled WES
features -- the recipe changes, not just some rows):

  control      published WE features
  fold_median  fallback rows use a state median computed only from the
               training-fold observed rows                       -> (1)
  fold_strict  additionally, observed anchors are discarded and every row
               uses that training-fold median                    -> (1)+(2)

Under `fold_strict` no sowing information from the held-out season, and
no per-field survey information of any kind, reaches the WES features.

Note the WE_* columns store *calendar DOY* (simulate_wes(..., return_dos=
False) in 02_build_features.py), so the re-simulation must match.

Output: data/revision/R05_fold_medians.csv
        data/revision/R05_fold_derived_sowing.csv
"""
import time
import warnings

warnings.filterwarnings('ignore')

from _common import OUT, REPRODUCTIVE, STAGE_LABEL, WORK, cohort, r2, save, stage_frame
from scripts.utils.deep_models import fold_pred_ensemble
from scripts.utils.thermal import simulate_wes
import numpy as np
import pandas as pd

SOW = WORK / 'sowing_lookup.parquet'
DAYMET = WORK / 'daymet_full_2013_2024.parquet'
BEST = {'flag_leaf': 'XGBoost', 'boot': 'LightGBM',
        'heading': 'ElasticNet', 'anthesis': 'ElasticNet'}
WE_COLS = [f'WE_{s}_doy' for s in
           ['emergence', 'tillering', 'jointing', 'flag_leaf', 'boot',
            'heading', 'anthesis', 'maturity']]
YEARS = [2014, 2015, 2016, 2017]


def main():
    fe, cols = cohort()
    sw = pd.read_parquet(SOW)
    sw = sw[sw.harvest_year.between(2014, 2017)].copy()
    sw['field_id'] = sw['field_id'].astype(str)

    # ---- step 0: how far do the fold-derived medians move? ----------------
    obs = sw[sw.source != 'state_median']
    glob = obs.groupby('state')['sowing_doy_used'].median()
    rows = []
    for y in YEARS:
        med = obs[obs.harvest_year != y].groupby('state')['sowing_doy_used'].median()
        for st in glob.index:
            rows.append(dict(held_out_year=y, state=st,
                             median_global=glob.get(st, np.nan),
                             median_trainfold=med.get(st, np.nan)))
    fm = pd.DataFrame(rows)
    fm['shift_days'] = fm.median_trainfold - fm.median_global
    save(fm.round(2), 'R05_fold_medians.csv')
    print(f'fold-derived vs global state median: |shift| mean='
          f'{fm.shift_days.abs().mean():.2f} d  max={fm.shift_days.abs().max():.1f} d\n')

    # ---- weather + latitude, once -----------------------------------------
    print('loading Daymet ...', flush=True)
    wx = pd.read_parquet(DAYMET, columns=['FIELDID', 'date', 'Tmin', 'Tmax'])
    wx = wx.rename(columns={'FIELDID': 'field_id'})
    wx['field_id'] = wx['field_id'].astype(str)
    wx['date'] = pd.to_datetime(wx['date'])
    wx['T_mean'] = (wx.Tmax + wx.Tmin) / 2.0
    wx['doy'] = wx['date'].dt.dayofyear
    wx = wx[wx.date.dt.year.between(2013, 2017)]
    wxf = {k: v.sort_values('date') for k, v in wx.groupby('field_id')}
    lat = (fe[['field_id', 'latitude']].drop_duplicates('field_id')
             .set_index('field_id')['latitude'].to_dict())
    print(f'  weather for {len(wxf)} fields\n', flush=True)

    src = sw.set_index(['field_id', 'harvest_year'])['source'].to_dict()
    anchor = sw.set_index(['field_id', 'harvest_year'])['sowing_doy_used'].to_dict()
    keys = fe[['field_id', 'year', 'state']].drop_duplicates()

    def resimulate(year, variant):
        """WE_* table for every field-year, under a training-fold anchor."""
        med = (sw[(sw.harvest_year != year) & (sw.source != 'state_median')]
               .groupby('state')['sowing_doy_used'].median())
        out = []
        for fid, yr, st in keys.itertuples(index=False):
            s = src.get((fid, yr), 'state_median')
            if variant == 'fold_median' and s != 'state_median':
                a = anchor.get((fid, yr))            # keep observed anchor
            else:
                a = med.get(st, np.nan)
            w = wxf.get(fid)
            if w is None or fid not in lat or not np.isfinite(a):
                continue
            w = w[(w.date >= pd.Timestamp(f'{yr - 1}-07-01'))
                  & (w.date <= pd.Timestamp(f'{yr}-07-31'))]
            if len(w) < 200:
                continue
            r = simulate_wes(w, lat[fid], int(a), int(yr), return_dos=False)
            r.update(field_id=fid, year=yr)
            out.append(r)
        return pd.DataFrame(out)

    # cache the re-simulation: it is shared by all four stages
    cache = {}
    for y in YEARS:
        for v in ['fold_median', 'fold_strict']:
            t0 = time.time()
            cache[(y, v)] = resimulate(y, v).set_index(['field_id', 'year'])
            print(f'  resimulated {v:12s} holdout {y}: '
                  f'{len(cache[(y, v)])} field-years ({time.time() - t0:.0f}s)',
                  flush=True)

    # ---- evaluate ----------------------------------------------------------
    rows = []
    for stage in REPRODUCTIVE:
        d, tgt = stage_frame(fe, stage)
        model = BEST[stage]
        feat_h, feat_m = cols(stage, True), cols(stage, False)

        T0, P0 = [], []
        for y in sorted(d.year.unique()):
            tr, te = d[d.year != y], d[d.year == y]
            if len(tr) < 50 or len(te) < 5:
                continue
            P0.extend(fold_pred_ensemble(tr, te, feat_m, tgt, model))
            T0.extend(te[tgt].values)
        r2_ml = r2(np.asarray(T0, float), np.asarray(P0, float))

        for variant in ['control', 'fold_median', 'fold_strict']:
            t0 = time.time()
            T, P = [], []
            for y in sorted(d.year.unique()):
                dd = d
                if variant != 'control':
                    dd = d.copy()
                    new = cache[(y, variant)]
                    idx = pd.MultiIndex.from_frame(dd[['field_id', 'year']])
                    for c in WE_COLS:
                        if c in dd.columns and c in new.columns:
                            dd[c] = new[c].reindex(idx).to_numpy()
                tr, te = dd[dd.year != y], dd[dd.year == y]
                if len(tr) < 50 or len(te) < 5:
                    continue
                P.extend(fold_pred_ensemble(tr, te, feat_h, tgt, model))
                T.extend(te[tgt].values)
            r2_h = r2(np.asarray(T, float), np.asarray(P, float))
            rows.append(dict(stage=stage, model=model, variant=variant,
                             r2_mlonly=r2_ml, r2_hybrid=r2_h,
                             gain=r2_h - r2_ml, sec=round(time.time() - t0, 1)))
            print(f'  {STAGE_LABEL[stage]:10s} {variant:12s} '
                  f'hybrid={r2_h:+.3f} gain={r2_h - r2_ml:+.3f} '
                  f'({rows[-1]["sec"]}s)', flush=True)
            pd.DataFrame(rows).to_csv(OUT / 'R05_fold_derived_sowing.csv',
                                      index=False)

    R = pd.DataFrame(rows)
    ctrl = R[R.variant == 'control'].set_index('stage')['gain']
    R['pct_retained'] = R.apply(
        lambda r: 100 * r.gain / ctrl[r.stage] if ctrl[r.stage] else np.nan, axis=1)
    save(R.round(3), 'R05_fold_derived_sowing.csv')
    print('\n=== gain retained under a training-fold-derived anchor (%) ===')
    print(R.pivot_table(index='stage', columns='variant',
                        values='pct_retained').round(1).to_string())


if __name__ == '__main__':
    main()
