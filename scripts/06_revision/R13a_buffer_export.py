"""R13a - Launch the buffer-radius HLS re-extraction  (reviewer #6 major 7).

Reviewer #6: "The rationale for the 300-m buffer should be better justified,
and information on actual field size and buffer sensitivity should be
provided."

Answering this properly needs the HLS archive re-extracted at several radii,
which is what this script starts. It also settles a second question the audit
raised: manuscript Section 2.2.4 states that only CDL winter-wheat pixels
enter the feature engineering, but no CDL mask exists anywhere in the
pipeline that produced the submitted results. We therefore extract both ways
and can report what the mask would have done.

Design
------
Field points are recovered as the centroids of the published 300 m buffer
polygons (the buffer is a circle, so its centroid is the original point).
Each sampled field contributes three features, buffered at 150, 300 and
500 m, carrying a `radius` property; one reduceRegions pass per image then
covers all three radii at once. Processing (cloud/shadow masking, scaling,
index formulas) is copied from scripts/00_extraction/01_hls.js so the
300 m arm reproduces the published extraction.

Four export tasks are created: {L30, S30} x {unmasked, CDL winter wheat}.
They run server-side; R13b reads the resulting assets back.

Sampling: a stratified subsample keeps the export and read-back tractable.
Fields carrying a reproductive-stage label are preferred, because that is
where the accuracy claims live.
"""
import sys, time
from pathlib import Path

import ee
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

PROJECT = 'propane-primacy-481403-u3'
ASSET = f'projects/{PROJECT}/assets/wheat_fields_buffer300m_polygons_new'
OUT_ROOT = f'projects/{PROJECT}/assets/buffer_sensitivity'
RADII = [150, 300, 500]
START, END = '2013-07-01', '2017-08-01'
N_PER_STATE = {'KS': 360, 'OK': 200, 'CO': 140, 'TX': 120, 'NE': 57}
SEED = 7
CDL_WINTER_WHEAT = 24


def stratified_fields():
    """Sample fields by state, preferring those with reproductive labels."""
    # inlined from scripts.utils.deep_models.STAGE_MAP to keep this script
    # importable in the Earth Engine environment (no xgboost/lightgbm there)
    PHENO = ('/depot/ciampitti/data/WheatPhenologyHRW/data/processed/'
             'buffer_300m/wheat_hrw_phenology_buffer_matched.parquet')
    ph = pd.read_parquet(PHENO, columns=['FIELDID', 'growth_stage'])
    rep = {'Flag Leaf Emerging', 'Flag Leaf Emerged', 'Early Boot', 'Boot',
           'Head Emerging', 'Heading', 'Complete Heading', 'Early Bloom', 'Bloom'}
    has_rep = set(ph.loc[ph.growth_stage.isin(rep), 'FIELDID'].astype(str))

    fe = pd.read_parquet(
        '/depot/ciampitti/data/WheatPhenologyHRW/data/raw/satellite/'
        'extension_2018_2024/features_v3_realsowing_train.parquet',
        columns=['field_id', 'state_TX', 'state_OK', 'state_KS',
                 'state_NE', 'state_CO', 'state_NM'])
    oh = [c for c in fe.columns if c.startswith('state_')]
    fe['state'] = fe[oh].idxmax(axis=1).str.replace('state_', '')
    fe['state'] = fe['state'].replace({'NM': 'TX'})
    fld = fe.drop_duplicates('field_id')[['field_id', 'state']]
    fld['field_id'] = fld['field_id'].astype(str)
    fld['rep'] = fld.field_id.isin(has_rep)

    picks = []
    for st, n in N_PER_STATE.items():
        d = fld[fld.state == st]
        pref = d[d.rep]
        take = pref.sample(min(n, len(pref)), random_state=SEED)
        if len(take) < n:
            rest = d[~d.field_id.isin(take.field_id)]
            take = pd.concat([take, rest.sample(min(n - len(take), len(rest)),
                                                random_state=SEED)])
        picks.append(take)
    out = pd.concat(picks)
    print(out.groupby('state').agg(n=('field_id', 'size'),
                                   with_repro=('rep', 'sum')).to_string())
    return out.field_id.tolist()


def prep(img, sensor):
    """Cloud/shadow mask, band rename, indices.

    NOTE ON SCALING. scripts/00_extraction/01_hls.js multiplies the bands by
    1e-4, which was correct when it was run: the Earth Engine HLS collection
    then served integer digital numbers. It now serves reflectance already
    scaled to 0-1, so applying the factor again drives EVI to ~1e-5. NDVI and
    GCVI are ratios and survive the error, which is why it is easy to miss.
    We therefore normalise defensively rather than assume either convention,
    and export the reflectance bands as well so any index can be recomputed
    without re-running the extraction.
    """
    fm = img.select('Fmask')
    m = (fm.bitwiseAnd(1 << 1).eq(0)
         .And(fm.bitwiseAnd(1 << 2).eq(0))
         .And(fm.bitwiseAnd(1 << 3).eq(0)))
    src = (['B2', 'B3', 'B4', 'B5', 'B6', 'B7'] if sensor == 'L30'
           else ['B2', 'B3', 'B4', 'B8A', 'B11', 'B12'])
    s = img.updateMask(m).select(src, ['Blue', 'Green', 'Red', 'NIR',
                                       'SWIR1', 'SWIR2'])
    # scale only if the values are digital numbers
    s = s.where(s.gt(2.0), s.multiply(1e-4))
    ndvi = s.normalizedDifference(['NIR', 'Red']).rename('NDVI')
    evi = s.expression(
        '2.5 * ((NIR - RED) / (NIR + 6.0 * RED - 7.5 * BLUE + 1.0))',
        {'NIR': s.select('NIR'), 'RED': s.select('Red'),
         'BLUE': s.select('Blue')}).rename('EVI')
    gcvi = s.select('NIR').divide(s.select('Green')).subtract(1).rename('GCVI')
    return (ndvi.addBands([evi, gcvi, s.select(['Blue', 'Green', 'Red', 'NIR'])])
            .copyProperties(img, ['system:time_start']))


def main():
    ee.Initialize(project=PROJECT)
    ids = stratified_fields()
    print(f'\nsampled {len(ids)} fields')

    src = ee.FeatureCollection(ASSET).filter(ee.Filter.inList('FIELDID', ids))
    pts = src.map(lambda f: ee.Feature(f.geometry().centroid(1),
                                       {'FIELDID': f.get('FIELDID')}))
    rings = ee.FeatureCollection([
        pts.map(lambda f: f.buffer(r).set('radius', r)) for r in RADII
    ]).flatten()
    print('features (fields x radii):', rings.size().getInfo())

    tasks = []
    for sensor, coll in [('L30', 'NASA/HLS/HLSL30/v002'),
                         ('S30', 'NASA/HLS/HLSS30/v002')]:
        base = (ee.ImageCollection(coll).filterDate(START, END)
                .filterBounds(rings).map(lambda i: prep(i, sensor)))
        for mask_name in ['raw', 'cdl']:
            def per_image(im, mask_name=mask_name):
                im = ee.Image(im)
                d = im.date()
                if mask_name == 'cdl':
                    # CDL of the harvest year: HLS dates after 1 July belong
                    # to the following harvest year's season.
                    hy = ee.Number(d.get('year')).add(
                        ee.Number(d.getRelative('day', 'year')).gte(181).multiply(1))
                    cdl = (ee.ImageCollection('USDA/NASS/CDL')
                           .filter(ee.Filter.calendarRange(hy, hy, 'year'))
                           .first().select('cropland'))
                    im = im.updateMask(cdl.eq(CDL_WINTER_WHEAT))
                return (im.reduceRegions(rings, ee.Reducer.mean(), 30)
                        .filter(ee.Filter.notNull(['NDVI']))
                        .map(lambda f: f.set('date', d.format('YYYY-MM-dd'),
                                             'sensor', sensor,
                                             'mask', mask_name)))
            out = base.map(per_image).flatten()
            desc = f'bufsens_{sensor}_{mask_name}'
            t = ee.batch.Export.table.toAsset(
                collection=out, description=desc,
                assetId=f'{OUT_ROOT}_{sensor}_{mask_name}')
            t.start()
            tasks.append((desc, t.id))
            print('started', desc, t.id)

    pd.DataFrame(tasks, columns=['description', 'task_id']).to_csv(
        Path(__file__).resolve().parents[2] / 'data' / 'revision'
        / 'R13_export_tasks.csv', index=False)
    print('\n4 export tasks running server-side. Poll with R13b.')


if __name__ == '__main__':
    main()
