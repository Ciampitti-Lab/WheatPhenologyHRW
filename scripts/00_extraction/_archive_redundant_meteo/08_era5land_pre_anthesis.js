// =================================================================
// 15 — ERA5-Land pre-anthesis stress features
// =================================================================
// Window: Mar 1 – Apr 14 of harvest year
//
// Output CSV: era5land_pre_anthesis_buffer.csv
//   Per (FIELDID, harvest_year):
//     t2m_max_mean_pa, t2m_min_mean_pa
//     t2m_min_min_pa  (extreme cold during pre-anthesis)
//     skin_T_min_pa  (extreme cold land surface)
//     soil_T_L1_mean_pa
//     soil_T_L1_min_pa  (frozen soil events)
//     swvl1_mean_pa, swvl2_mean_pa  (entering pre-anthesis soil moisture state)
//     swvl_decline_pa  (drying rate during window)
//     pe_cum_pa
//     frost_days_2m_pa     (# days t2m_min < 0°C)
//     heat_days_25_2m_pa   (# days t2m_max > 25°C)
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

  var era = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
    .filterDate(startDate, endDate);

  var frost  = era.map(function(im) { return im.select('temperature_2m_min').lt(273.15).rename('frost'); });
  var hot25_2m = era.map(function(im) { return im.select('temperature_2m_max').gt(298.15).rename('hot25_2m'); });

  var swvl1_first = ee.Image(era.select('volumetric_soil_water_layer_1').first());
  var swvl1_last  = ee.Image(era.select('volumetric_soil_water_layer_1').sort('system:time_start', false).first());
  var swvl_decline = swvl1_first.subtract(swvl1_last).rename('swvl_decline_pa');

  var stack = ee.Image.cat([
    era.select('temperature_2m_max').mean().rename('t2m_max_mean_pa'),
    era.select('temperature_2m_min').mean().rename('t2m_min_mean_pa'),
    era.select('temperature_2m_min').min().rename('t2m_min_min_pa'),
    era.select('skin_temperature_min').min().rename('skin_T_min_pa'),
    era.select('soil_temperature_level_1').mean().rename('soil_T_L1_mean_pa'),
    era.select('soil_temperature_level_1').min().rename('soil_T_L1_min_pa'),
    era.select('volumetric_soil_water_layer_1').mean().rename('swvl1_mean_pa'),
    era.select('volumetric_soil_water_layer_2').mean().rename('swvl2_mean_pa'),
    swvl_decline,
    era.select('potential_evaporation_sum').sum().rename('pe_cum_pa'),
    frost.sum().rename('frost_days_2m_pa'),
    hot25_2m.sum().rename('heat_days_25_2m_pa'),
  ]);

  return stack.reduceRegions({
    collection: centroids,
    reducer: ee.Reducer.first(),
    scale: 9000
  }).map(function(f) { return ee.Feature(f).set('harvest_year', yr); });
})).flatten();

print('Total samples:', allYearSamples.size());
print('First sample:', allYearSamples.first());

Export.table.toDrive({
  collection: allYearSamples,
  description: 'ERA5Land_pre_anthesis_buffer',
  folder: driveFolder,
  fileNamePrefix: 'era5land_pre_anthesis_buffer',
  fileFormat: 'CSV',
  selectors: ['FIELDID', 'harvest_year',
              't2m_max_mean_pa', 't2m_min_mean_pa', 't2m_min_min_pa',
              'skin_T_min_pa', 'soil_T_L1_mean_pa', 'soil_T_L1_min_pa',
              'swvl1_mean_pa', 'swvl2_mean_pa', 'swvl_decline_pa',
              'pe_cum_pa', 'frost_days_2m_pa', 'heat_days_25_2m_pa']
});

print('Export ready. Run from Tasks tab.');
print('Expected: ~30,600 rows (6,120 fields × 5 years)');
