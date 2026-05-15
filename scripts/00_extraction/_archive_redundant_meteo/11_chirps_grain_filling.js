// =================================================================
// 13 — CHIRPS daily precipitation grain-filling features
// =================================================================
// Source: UCSB-CHG/CHIRPS/DAILY — daily, 5km, 1981-now
// Use case: Precipitation patterns, drought intensity, dry-spell length
//   during grain filling. CHIRPS is widely cited and more accurate than
//   Daymet at high temporal resolution for water-limited environments.
//
// Window: DOY 105–175 (mid-Apr to late Jun)
//
// Output CSV: chirps_grain_filling_buffer.csv
//   Per (FIELDID, harvest_year):
//     prcp_cum_gf       — cumulative precipitation (mm)
//     prcp_max_day_gf   — max single-day precipitation (mm)
//     wet_days_gf       — # days with precipitation > 1mm
//     dry_days_gf       — # days with precipitation < 1mm
//     prcp_2week_gf     — cumulative precipitation in last 2 weeks of window
//                         (latest = closest to maturity)
//
// Run in: code.earthengine.google.com
// =================================================================

var fields = ee.FeatureCollection('projects/propane-primacy-481403-u3/assets/wheat_fields_buffer300m_polygons');

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
  var startDate     = ee.Date.fromYMD(yr, 4, 15);
  var endDate       = ee.Date.fromYMD(yr, 6, 30);
  var late2weekStart= ee.Date.fromYMD(yr, 6, 16);

  var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
    .filterDate(startDate, endDate)
    .select('precipitation');

  var chirpsLate2w = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
    .filterDate(late2weekStart, endDate)
    .select('precipitation');

  var wet = chirps.map(function(im) { return im.gt(1).rename('wet'); });
  var dry = chirps.map(function(im) { return im.lte(1).rename('dry'); });

  var prcp_sum  = chirps.sum().rename('prcp_cum_gf');
  var prcp_max  = chirps.max().rename('prcp_max_day_gf');
  var wet_days  = wet.sum().rename('wet_days_gf');
  var dry_days  = dry.sum().rename('dry_days_gf');
  var prcp_2w   = chirpsLate2w.sum().rename('prcp_2week_gf');

  var stack = ee.Image.cat([prcp_sum, prcp_max, wet_days, dry_days, prcp_2w]);

  return stack.reduceRegions({
    collection: centroids,
    reducer: ee.Reducer.first(),
    scale: 5000
  }).map(function(f) {
    return ee.Feature(f).set('harvest_year', yr);
  });
})).flatten();

print('Total samples:', allYearSamples.size());
print('First sample:', allYearSamples.first());

Export.table.toDrive({
  collection: allYearSamples,
  description: 'CHIRPS_grain_filling_buffer',
  folder: driveFolder,
  fileNamePrefix: 'chirps_grain_filling_buffer',
  fileFormat: 'CSV',
  selectors: ['FIELDID', 'harvest_year',
              'prcp_cum_gf', 'prcp_max_day_gf',
              'wet_days_gf', 'dry_days_gf', 'prcp_2week_gf']
});

print('Export ready. Run from Tasks tab.');
print('Expected: ~30,600 rows (6,120 fields × 5 years)');
