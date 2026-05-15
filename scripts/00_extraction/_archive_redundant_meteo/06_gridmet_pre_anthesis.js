// =================================================================
// 14 — GRIDMET pre-anthesis stress features (per field-year aggregates)
// =================================================================
// Window: Mar 1 – Apr 14 (DOY 60–104) of harvest year
//   Captures pre-anthesis stress (stem elongation → booting):
//   late frosts, early heat events, drought entering grain-fill
//
// Output CSV: gridmet_pre_anthesis_buffer.csv
//   Per (FIELDID, harvest_year):
//     vpd_max_pa, vpd_mean_pa, vpd_cum_pa
//     eto_cum_pa
//     tmmx_max_pa  (max daytime Tmax, K)
//     tmmn_min_pa  (min nighttime Tmin, K — late frost intensity)
//     frost_days_pa  (# days Tmin < 0°C = 273.15K)
//     heat_days_25_pa  (# days Tmax > 25°C = 298.15K)
//     heat_days_30_pa  (# days Tmax > 30°C = 303.15K)
//     gdd_cum_pa  (cumulative GDD base 5°C, capped at 0)
//     srad_cum_pa
//
// Run in: code.earthengine.google.com
// =================================================================

var fields = ee.FeatureCollection('projects/propane-primacy-481403-u3/assets/wheat_fields_buffer300m_polygons_new');

var centroids = fields.map(function(f) {
  var feat = ee.Feature(f);
  var geom = ee.Geometry(feat.geometry()).centroid({maxError: 1});
  return feat.setGeometry(geom);
});

print('Total fields:', fields.size());

var driveFolder = 'WheatFlagLeaf_HLS';
var harvestYears = [2013, 2014, 2015, 2016, 2017];

var allYearSamples = ee.FeatureCollection(harvestYears.map(function(yr) {
  yr = ee.Number(yr);
  var startDate = ee.Date.fromYMD(yr, 3, 1);
  var endDate   = ee.Date.fromYMD(yr, 4, 15);

  var gm = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET')
    .filterDate(startDate, endDate)
    .select(['vpd', 'eto', 'tmmx', 'tmmn', 'srad']);

  var frost = gm.map(function(im) { return im.select('tmmn').lt(273.15).rename('frost'); });
  var hot25 = gm.map(function(im) { return im.select('tmmx').gt(298.15).rename('hot25'); });
  var hot30 = gm.map(function(im) { return im.select('tmmx').gt(303.15).rename('hot30'); });

  // GDD base 5°C: max(0, ((Tmax + Tmin)/2 - 5°C))
  var gdd = gm.map(function(im) {
    var tmean = im.select('tmmx').add(im.select('tmmn')).divide(2);
    var gddi = tmean.subtract(273.15 + 5.0).max(0).rename('gdd');
    return gddi;
  });

  var stack = ee.Image.cat([
    gm.select('vpd').max().rename('vpd_max_pa'),
    gm.select('vpd').mean().rename('vpd_mean_pa'),
    gm.select('vpd').sum().rename('vpd_cum_pa'),
    gm.select('eto').sum().rename('eto_cum_pa'),
    gm.select('tmmx').max().rename('tmmx_max_pa'),
    gm.select('tmmn').min().rename('tmmn_min_pa'),
    frost.sum().rename('frost_days_pa'),
    hot25.sum().rename('heat_days_25_pa'),
    hot30.sum().rename('heat_days_30_pa'),
    gdd.sum().rename('gdd_cum_pa'),
    gm.select('srad').sum().rename('srad_cum_pa'),
  ]);

  return stack.reduceRegions({
    collection: centroids,
    reducer: ee.Reducer.first(),
    scale: 4000
  }).map(function(f) { return ee.Feature(f).set('harvest_year', yr); });
})).flatten();

print('Total samples:', allYearSamples.size());
print('First sample:', allYearSamples.first());

Export.table.toDrive({
  collection: allYearSamples,
  description: 'GRIDMET_pre_anthesis_buffer',
  folder: driveFolder,
  fileNamePrefix: 'gridmet_pre_anthesis_buffer',
  fileFormat: 'CSV',
  selectors: ['FIELDID', 'harvest_year',
              'vpd_max_pa', 'vpd_mean_pa', 'vpd_cum_pa', 'eto_cum_pa',
              'tmmx_max_pa', 'tmmn_min_pa',
              'frost_days_pa', 'heat_days_25_pa', 'heat_days_30_pa',
              'gdd_cum_pa', 'srad_cum_pa']
});

print('Export ready. Run from Tasks tab.');
print('Expected: ~30,600 rows (6,120 fields × 5 years)');
