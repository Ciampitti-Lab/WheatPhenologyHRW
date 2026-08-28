"""R13c - CDL winter-wheat fraction per buffer, training seasons, by radius.

Two gaps this closes. The manuscript quoted a wheat-fraction figure computed
on the extension cohort (2018-2024) because that was the only CDL extraction
we held, and flagged it as indicative. And the buffer-radius question needs
to know how the wheat fraction itself changes with radius, independently of
what the reflectance does.

Computed directly in Earth Engine over the four training harvest years at
150, 300 and 500 m. Cheap: one CDL image per year.

Output: data/revision/R13_cdl_fraction.csv
"""
from pathlib import Path
import ee
import pandas as pd

PROJECT = 'propane-primacy-481403-u3'
ASSET = f'projects/{PROJECT}/assets/wheat_fields_buffer300m_polygons_new'
REV = Path(__file__).resolve().parents[2] / 'data' / 'revision'
RADII = [150, 300, 500]
YEARS = [2014, 2015, 2016, 2017]
WINTER_WHEAT = 24


def main():
    ee.Initialize(project=PROJECT)
    pts = ee.FeatureCollection(ASSET).map(
        lambda f: ee.Feature(f.geometry().centroid(1),
                             {'FIELDID': f.get('FIELDID')}))
    rows = []
    for year in YEARS:
        cdl = (ee.ImageCollection('USDA/NASS/CDL')
               .filter(ee.Filter.calendarRange(year, year, 'year'))
               .first().select('cropland'))
        wheat = cdl.eq(WINTER_WHEAT).rename('wheat')
        for r in RADII:
            buf = pts.map(lambda f, r=r: f.buffer(r))
            fc = wheat.reduceRegions(buf, ee.Reducer.mean(), 30)
            # page the result out rather than one huge getInfo
            n, token = 0, None
            got = []
            while True:
                req = {'expression': fc, 'pageSize': 3000}
                if token:
                    req['pageToken'] = token
                res = ee.data.computeFeatures(req)
                for f in res.get('features', []):
                    p = f['properties']
                    if p.get('mean') is not None:
                        got.append((p['FIELDID'], float(p['mean'])))
                n += len(res.get('features', []))
                token = res.get('nextPageToken')
                print(f'  {year} r={r}m  {n} fields', end='\r', flush=True)
                if not token:
                    break
            for fid, frac in got:
                rows.append(dict(FIELDID=fid, harvest_year=year,
                                 radius=r, wheat_fraction=frac))
            print(f'  {year} r={r}m  {len(got)} fields                ')
    d = pd.DataFrame(rows)
    d.to_csv(REV / 'R13_cdl_fraction.csv', index=False)
    print(f'\n{len(d):,} rows -> R13_cdl_fraction.csv\n')

    npix = {r: 3.14159 * r * r / 900 for r in RADII}
    s = (d.groupby('radius')['wheat_fraction']
           .agg(['count', 'mean', 'median',
                 lambda x: x.quantile(.25), lambda x: x.quantile(.75)]))
    s.columns = ['n', 'mean', 'median', 'q25', 'q75']
    s['buffer_pixels'] = [round(npix[r]) for r in s.index]
    s['wheat_pixels_median'] = (s['median'] * s['buffer_pixels']).round()
    print('=== CDL winter-wheat fraction within the buffer, all field-years ===')
    print(s.round(3).to_string())

    inw = d[d.wheat_fraction > 0.10]
    s2 = (inw.groupby('radius')['wheat_fraction']
             .agg(['count', 'mean', 'median',
                   lambda x: x.quantile(.25), lambda x: x.quantile(.75)]))
    s2.columns = ['n', 'mean', 'median', 'q25', 'q75']
    s2['buffer_pixels'] = [round(npix[r]) for r in s2.index]
    s2['wheat_pixels_median'] = (s2['median'] * s2['buffer_pixels']).round()
    print('\n=== restricted to field-years the mask calls wheat (frac > 0.10) ===')
    print(s2.round(3).to_string())


if __name__ == '__main__':
    main()
