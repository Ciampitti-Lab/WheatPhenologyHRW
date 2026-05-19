"""Anthesis row of the sowing-sensitivity ablation, recomputed with the
ADOPTED model (FT-Transformer) to match the manuscript.

Same perturbation harness as 01_sowing_sensitivity.py (perturb fallback
sowing -> re-run simulate_wes -> rebuild WE_* -> LOYO) but anthesis uses
the FT-Transformer (5-seed averaged). Result: control gain 0.025
(vs 0.102 for the old ElasticNet best) but robust to perturbation
(90-119% retained) -- a small but genuine, non-leakage contribution;
the high-capacity FT re-learns much of the WES signal. The other three
reproductive stages are unchanged (their best models did not change).
Needs a GPU.

Output: <work_dir>/v3_results/anthesis_ft_ablation_summary.csv
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.utils.config import CFG, REPO_ROOT
from scripts.utils.thermal import simulate_wes
from scripts.utils.deep_models import WE, fold_pred_ft, r2
import numpy as np, pandas as pd

WORK = REPO_ROOT / CFG.paths.work_dir
PHENO = str(REPO_ROOT / CFG.paths.phenology_matched)
OUT = WORK / 'v3_results'; OUT.mkdir(parents=True, exist_ok=True)
SIGMAS = [7, 14, 21, 28]; N_REPS = 3; SEEDS = [0, 1, 2, 3, 4]
LBL = ['Early Bloom', 'Bloom']
META_FIXED = ['field_id', 'year', 'flag_true_doy', 'n_obs', 'sowing_doy_used']
REDUND = ['GDD_M2_at_SOS', 'VD_at_SOS', 'emergence_doy', 'VD_from_emergence_at_SOS',
          'fV_from_emergence_at_SOS', 'days_emergence_to_SOS']


def ft_loyo(d, feat, tgt):
    P, T = [], []
    for yr in sorted(d['year'].unique()):
        tr, te = d[d['year'] != yr], d[d['year'] == yr]
        if len(tr) < 50 or len(te) < 5:
            continue
        P.extend(np.mean([fold_pred_ft(tr, te, feat, tgt, s) for s in SEEDS], axis=0))
        T.extend(te[tgt].values)
    return r2(np.array(T), np.array(P))


def main():
    t0 = time.time()
    ph = pd.read_parquet(PHENO)
    ph['year'] = ph['growing_season'].str.split('-').str[1].astype(int)
    ph['field_id'] = ph['FIELDID'].astype(str)
    tgt = 'anthesis_dos_obs'
    e = (ph[ph['growth_stage'].isin(LBL)].groupby(['field_id', 'year'])['dos']
         .min().reset_index().rename(columns={'dos': tgt}))
    fe = pd.read_parquet(WORK / 'features_v3_realsowing_train.parquet')
    fe['field_id'] = fe['field_id'].astype(str); fe['year'] = fe['year'].astype(int)
    if 'state' in fe.columns:
        fe = fe.drop(columns=['state'])
    fe = fe.merge(e, on=['field_id', 'year'], how='left')
    sl = pd.read_parquet(WORK / 'sowing_lookup.parquet')
    sl['field_id'] = sl['field_id'].astype(str)
    sl = sl.rename(columns={'harvest_year': 'year'})
    fe = fe.merge(sl[['field_id', 'year', 'source', 'sowing_doy_used']]
                  .rename(columns={'sowing_doy_used': 'sow_base'}),
                  on=['field_id', 'year'], how='left')
    is_fb = (fe['source'] == 'state_median').values

    META = META_FIXED + [tgt, 'source', 'sow_base']
    ndre = [c for c in fe.columns if c.startswith('NDRE')]
    allc = [c for c in fe.columns if c not in META and c not in ndre
            and c not in REDUND and pd.api.types.is_numeric_dtype(fe[c])]
    hyb, mlo = allc, [c for c in allc if c not in WE]

    def trim(df):
        x = df.dropna(subset=[tgt]).copy()
        q1, q9 = x[tgt].quantile([.01, .99])
        return x[(x[tgt] >= q1) & (x[tgt] <= q9)].copy()

    dm = pd.read_parquet(WORK / 'daymet_full_2013_2024.parquet',
                         columns=['FIELDID', 'date', 'Tmin', 'Tmax', 'harvest_year'])
    dm['FIELDID'] = dm['FIELDID'].astype(str); dm['date'] = pd.to_datetime(dm['date'])
    dm['doy'] = dm['date'].dt.dayofyear
    dm['T_mean'] = ((dm['Tmin'] + dm['Tmax']) / 2).astype('float32')
    wx_by = {k: g.sort_values('date')[['date', 'doy', 'T_mean']].reset_index(drop=True)
             for k, g in dm.groupby(['FIELDID', 'harvest_year'])}
    del dm
    lat_by = dict(zip(zip(fe['field_id'], fe['year']),
                      fe.get('latitude', pd.Series(38.0, index=fe.index))))

    rows = []
    r2_ml = ft_loyo(trim(fe), mlo, tgt)
    g0 = ft_loyo(trim(fe), hyb, tgt) - r2_ml
    print(f'[control] anthesis FT  gain={g0:+.4f}', flush=True)
    fb_idx = np.where(is_fb)[0]
    fb_keys = list(zip(fe['field_id'].values[fb_idx], fe['year'].values[fb_idx],
                       fe['sow_base'].values[fb_idx]))
    for sigma in SIGMAS:
        for rep in range(1, N_REPS + 1):
            rng = np.random.RandomState(1000 * sigma + rep)
            pert = fe.copy(); wenew = {c: pert[c].values.copy() for c in WE}
            noise = np.round(rng.normal(0, sigma, len(fb_idx))).astype(int)
            for j, (fid, hy, base) in enumerate(fb_keys):
                wx = wx_by.get((fid, hy))
                if wx is None or len(wx) == 0 or pd.isna(base):
                    continue
                ns = int(np.clip(int(base) + noise[j], 220, 330))
                we = simulate_wes(wx, lat=float(lat_by.get((fid, hy), 38.0)),
                                  sowing_doy=ns, sowing_year=int(hy), return_dos=False)
                ridx = fb_idx[j]
                for c in WE:
                    wenew[c][ridx] = we.get(c, np.nan)
            for c in WE:
                pert[c] = wenew[c]
            g = ft_loyo(trim(pert), hyb, tgt) - r2_ml
            rows.append(dict(sigma=sigma, rep=rep, gain=g))
            print(f'[s={sigma:2d} r{rep}] gain={g:+.4f}', flush=True)

    df = pd.DataFrame(rows); summ = []
    for sigma in SIGMAS:
        gm = df[df.sigma == sigma].gain.mean()
        summ.append(dict(stage='anthesis', model='FT', sigma=sigma,
                         gain_control=round(g0, 4), gain_mean=round(gm, 4),
                         pct_retained=round(100 * gm / g0, 1) if g0 else np.nan))
    pd.DataFrame(summ).to_csv(OUT / 'anthesis_ft_ablation_summary.csv', index=False)
    print(pd.DataFrame(summ).to_string(index=False), flush=True)
    print(f'Done in {(time.time()-t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
