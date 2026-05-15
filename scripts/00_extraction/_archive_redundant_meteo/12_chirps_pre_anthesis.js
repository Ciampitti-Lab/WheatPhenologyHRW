// =================================================================
// 17 — CHIRPS pre-anthesis precipitation features
// =================================================================
// Window: Mar 1 – Apr 14 of harvest year
//
// Output CSV: chirps_pre_anthesis_buffer.csv
//   Per (FIELDID, harvest_year):
//     prcp_cum_pa
//     prcp_max_day_pa
//     wet_days_pa, dry_days_pa
//     prcp_2week_late_pa  (last 2 weeks before window end — closest to anthesis)
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
  var startDate     = ee.Date.fromYMD(yr, 3, 1);
  var endDate       = ee.Date.fromYMD(yr, 4, 15);
  var late2weekStart= ee.Date.fromYMD(yr, 4, 1);

  var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
    .filterDate(startDate, endDate)
    .select('precipitation');

  var chirpsLate2w = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
    .filterDate(late2weekStart, endDate)
    .select('precipitation');

  var wet = chirps.map(function(im) { return im.gt(1).rename('wet'); });
  var dry = chirps.map(function(im) { return im.lte(1).rename('dry'); });

  var stack = ee.Image.cat([
    chirps.sum().rename('prcp_cum_pa'),
    chirps.max().rename('prcp_max_day_pa'),
    wet.sum().rename('wet_days_pa'),
    dry.sum().rename('dry_days_pa'),
    chirpsLate2w.sum().rename('prcp_2week_late_pa'),
  ]);

  return stack.reduceRegions({
    collection: centroids,
    reducer: ee.Reducer.first(),
    scale: 5000
  }).map(function(f) { return ee.Feature(f).set('harvest_year', yr); });
})).flatten();

print('Total samples:', allYearSamples.size());
print('First sample:', allYearSamples.first());

Export.table.toDrive({
  collection: allYearSamples,
  description: 'CHIRPS_pre_anthesis_buffer',
  folder: driveFolder,
  fileNamePrefix: 'chirps_pre_anthesis_buffer',
  fileFormat: 'CSV',
  selectors: ['FIELDID', 'harvest_year',
              'prcp_cum_pa', 'prcp_max_day_pa',
              'wet_days_pa', 'dry_days_pa', 'prcp_2week_late_pa']
});

print('Export ready. Run from Tasks tab.');
print('Expected: ~30,600 rows (6,120 fields × 5 years)');
