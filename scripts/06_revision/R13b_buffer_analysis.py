"""R13b - Buffer-radius and CDL-mask sensitivity  (reviewer #6 major 7).

Reads the assets exported by R13a, rebuilds the HLS phenometrics exactly as
scripts/02_features/02_build_features.py does (the smoothing and
phenometric functions are imported from it, not reimplemented), substitutes
them into the published feature matrix, and refits.

Six configurations: radius in {150, 300, 500} m, crossed with an unmasked
buffer mean and a CDL winter-wheat mask. The 300 m unmasked arm is the
positive control: it is what the submitted results used, so it should
reproduce the published accuracy on this subsample, and any deviation is a
check on the whole re-extraction chain rather than a finding.

Only HLS-derived columns are replaced. Daymet, MODIS LST, WES and the site
and state descriptors are carried over unchanged, so the contrast isolates
the reflectance aggregation.

Output: data/revision/R13_buffer_phenometrics.parquet
        data/revision/R13_buffer_sensitivity.csv
        data/revision/R13_vi_agreement.csv
"""
import importlib.util
import sys
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

import ee
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

PROJECT = 'propane-primacy-481403-u3'
OUT_ROOT = f'projects/{PROJECT}/assets/buffer_sensitivity'
REV = ROOT / 'data' / 'revision'
RADII = [150, 300, 500]
MASKS = ['raw', 'cdl']
VIS = ['NDVI', 'EVI', 'GCVI']
REPRO = ['flag_leaf', 'boot', 'heading', 'anthesis']
BEST = {'flag_leaf': 'XGBoost', 'boot': 'LightGBM',
        'heading': 'ElasticNet', 'anthesis': 'ElasticNet'}

_spec = importlib.util.spec_from_file_location(
    'bf', ROOT / 'scripts' / '02_features' / '02_build_features.py')
_bf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bf)
smooth_vi_dos = _bf.smooth_vi_dos
extract_phenometrics_gs = _bf.extract_phenometrics_gs
fit_double_logistic = _bf.fit_double_logistic


def read_asset(asset_id, page=5000):
    """Page an exported FeatureCollection back out of Earth Engine."""
    rows, token, n = [], None, 0
    while True:
        req = {'assetId': asset_id, 'pageSize': page}
        if token:
            req['pageToken'] = token
        r = ee.data.listFeatures(req)
        for f in r.get('features', []):
            rows.append(f['properties'])
        n += len(r.get('features', []))
        token = r.get('nextPageToken')
        print(f'    {n} rows', end='\r', flush=True)
        if not token:
            break
    print(f'    {n} rows read from {asset_id.split("/")[-1]}')
    return pd.DataFrame(rows)


def load_all():
    cache = REV / 'R13_raw_timeseries.parquet'
    if cache.exists():
        print('using cached time series')
        return pd.read_parquet(cache)
    ee.Initialize(project=PROJECT)
    parts = []
    for sensor in ['L30', 'S30']:
        for mask in MASKS:
            parts.append(read_asset(f'{OUT_ROOT}_{sensor}_{mask}'))
    d = pd.concat(parts, ignore_index=True)
    d['date'] = pd.to_datetime(d['date'])
    d['radius'] = d['radius'].astype(int)
    d['harvest_year'] = d['date'].dt.year + (d['date'].dt.month >= 7).astype(int)
    origin = pd.to_datetime(dict(year=d.harvest_year - 1, month=7, day=1))
    d['dos'] = (d['date'] - origin).dt.days + 1
    d = d[d.harvest_year.between(2014, 2017) & d.dos.between(1, 365)]
    d.to_parquet(cache, index=False)
    return d


def phenometrics(ts):
    """Per (field, harvest_year, radius, mask) HLS phenometric row."""
    out = []
    keys = ['FIELDID', 'harvest_year', 'radius', 'mask']
    g = ts.groupby(keys, sort=False)
    tot = g.ngroups
    for i, (k, d) in enumerate(g):
        if i % 500 == 0:
            print(f'    {i}/{tot}', end='\r', flush=True)
        row = dict(zip(keys, k))
        ndvi_pair = (None, None)
        ok = True
        for vi in VIS:
            sub = d[['dos', vi]].dropna()
            if len(sub) < 5:
                ok = False
                break
            grid, sm = smooth_vi_dos(sub.dos.values, sub[vi].values)
            if sm is None:
                ok = False
                break
            row.update(extract_phenometrics_gs(grid, sm, vi))
            if vi == 'NDVI':
                ndvi_pair = (grid, sm)
        if not ok:
            continue
        popt = fit_double_logistic(*ndvi_pair) if ndvi_pair[1] is not None else None
        for j, nm in enumerate(['DL_c3_greenup_steepness', 'DL_c4_greenup_midpoint',
                                'DL_c5_senesc_steepness', 'DL_c6_senesc_midpoint']):
            row[nm] = float(popt[j + 2]) if popt is not None else np.nan
        out.append(row)
    print()
    return pd.DataFrame(out)


def main():
    print('=== reading exported time series ===')
    ts = load_all()
    print(f'{len(ts):,} observations | fields={ts.FIELDID.nunique()} '
          f'| radii={sorted(ts.radius.unique())} | masks={sorted(ts["mask"].unique())}')

    print('\n=== VI agreement across radii (unmasked) ===')
    piv = (ts[ts['mask'] == 'raw']
           .pivot_table(index=['FIELDID', 'date'], columns='radius', values='NDVI'))
    piv = piv.dropna()
    agree = []
    for r in [150, 500]:
        if r in piv.columns and 300 in piv.columns:
            d = piv[r] - piv[300]
            agree.append(dict(comparison=f'NDVI r={r} vs r=300', n=len(d),
                              bias=d.mean(), mad=d.abs().mean(),
                              rmse=float(np.sqrt((d ** 2).mean())),
                              corr=float(piv[r].corr(piv[300]))))
    A = pd.DataFrame(agree)
    if len(A):
        A.to_csv(REV / 'R13_vi_agreement.csv', index=False)
        print(A.round(4).to_string(index=False))

    print('\n=== rebuilding phenometrics ===')
    pcache = REV / 'R13_buffer_phenometrics.parquet'
    if pcache.exists():
        ph = pd.read_parquet(pcache)
        print('  using cached phenometrics')
    else:
        ph = phenometrics(ts)
        ph.to_parquet(pcache, index=False)
    print(f'  {len(ph):,} phenometric rows')

    print('\n=== refitting the reproductive stages ===')
    from _common import cohort, metrics, stage_frame
    from scripts.utils.deep_models import fold_pred_ensemble
    fe, cols = cohort()
    fe['FIELDID'] = fe['field_id'].astype(str)

    hls_cols = [c for c in fe.columns
                if c.startswith(('NDVI', 'EVI', 'GCVI', 'DL_c'))]
    rows = []
    for radius in RADII:
        for mask in MASKS:
            sub = ph[(ph.radius == radius) & (ph['mask'] == mask)].copy()
            sub = sub.rename(columns={'harvest_year': 'year'})
            keep = [c for c in hls_cols if c in sub.columns]
            sub = sub[['FIELDID', 'year'] + keep]
            f2 = fe.drop(columns=keep).merge(sub, on=['FIELDID', 'year'], how='inner')
            for stage in REPRO:
                d, tgt = stage_frame(f2, stage)
                if len(d) < 100:
                    continue
                feat = cols(stage, True)
                feat = [c for c in feat if c in d.columns]
                T, P = [], []
                for y in sorted(d.year.unique()):
                    tr, te = d[d.year != y], d[d.year == y]
                    if len(tr) < 50 or len(te) < 5:
                        continue
                    P.extend(fold_pred_ensemble(tr, te, feat, tgt, BEST[stage]))
                    T.extend(te[tgt].values)
                if not T:
                    continue
                m = metrics(np.asarray(T, float), np.asarray(P, float))
                m.update(stage=stage, radius=radius, mask=mask, n=len(d))
                rows.append(m)
                print(f'  r={radius:3d} {mask:3s} {stage:10s} '
                      f'R2={m["R2"]:+.3f} RMSE={m["RMSE"]:.2f} n={len(d)}',
                      flush=True)
            pd.DataFrame(rows).to_csv(REV / 'R13_buffer_sensitivity.csv', index=False)

    R = pd.DataFrame(rows)
    R.to_csv(REV / 'R13_buffer_sensitivity.csv', index=False)
    print('\n=== LOYO R2 by buffer radius and mask ===')
    print(R.pivot_table(index='stage', columns=['mask', 'radius'],
                        values='R2').reindex(REPRO).round(3).to_string())
    ref = R[(R.radius == 300) & (R['mask'] == 'raw')].set_index('stage')['R2']
    R['delta_vs_300raw'] = R.apply(lambda x: x.R2 - ref.get(x.stage, np.nan), axis=1)
    R.to_csv(REV / 'R13_buffer_sensitivity.csv', index=False)
    print('\n=== deviation from the 300 m unmasked control ===')
    print(R.pivot_table(index='stage', columns=['mask', 'radius'],
                        values='delta_vs_300raw').reindex(REPRO).round(3).to_string())


if __name__ == '__main__':
    main()
