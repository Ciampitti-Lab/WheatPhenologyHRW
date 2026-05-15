// =================================================================
// 11 — ERA5-Land grain-filling features (per field-year aggregates)
// =================================================================
// Source: ECMWF/ERA5_LAND/DAILY_AGGR — daily, ~9km, global, 1950-now
// Daily aggregates from hourly ERA5-Land
// Use case: Soil T at depth, root-zone soil moisture, heat stress events
//
// Window: DOY 105–175 (mid-Apr to late Jun) of harvest year
//
// Output CSV: era5land_grain_filling_buffer.csv
//   Per (FIELDID, harvest_year):
//     t2m_max_gf            — daily mean of max 2m air T (K)
//     t2m_max_max_gf        — single hottest day Tmax (K)
//     skin_T_max_gf         — max land surface T (K) — proxy for heat stress
//     soil_T_L1_mean_gf     — mean 0-7cm soil T (K)
//     soil_T_L2_mean_gf     — mean 7-28cm soil T (K)
//     swvl1_mean_gf         — mean 0-7cm volumetric soil water (m³/m³)
//     swvl2_mean_gf         — mean 7-28cm volumetric soil water (m³/m³)
//     swvl_decline_gf       — drop in swvl1 over window (initial - final, m³/m³)
//     pe_cum_gf             — cumulative potential evaporation (m)
//     u10_v10_speed_max_gf  — max 10m wind speed (m/s)
//     hot_days_t2m_30       — days t2m_max > 303.15K
//     hot_days_t2m_35       — days t2m_max > 308.15K
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
  var startDate = ee.Date.fromYMD(yr, 4, 15);
  var endDate   = ee.Date.fromYMD(yr, 6, 30);

  var era = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR')
    .filterDate(startDate, endDate);

  // Wind speed = sqrt(u² + v²)
  var windCol = era.map(function(im) {
    var u = im.select('u_component_of_wind_10m_max');
    var v = im.select('v_component_of_wind_10m_max');
    return u.pow(2).add(v.pow(2)).sqrt().rename('wind_max');
  });

  // Heat-day indicators
  var hot30 = era.map(function(im) {
    return im.select('temperature_2m_max').gt(303.15).rename('hot30');
  });
  var hot35 = era.map(function(im) {
    return im.select('temperature_2m_max').gt(308.15).rename('hot35');
  });

  // Aggregates
  var t2m_max_mean = era.select('temperature_2m_max').mean().rename('t2m_max_gf');
  var t2m_max_max  = era.select('temperature_2m_max').max().rename('t2m_max_max_gf');
  var skin_T_max   = era.select('skin_temperature_max').max().rename('skin_T_max_gf');
  var soil_T_L1    = era.select('soil_temperature_level_1').mean().rename('soil_T_L1_mean_gf');
  var soil_T_L2    = era.select('soil_temperature_level_2').mean().rename('soil_T_L2_mean_gf');
  var swvl1_mean   = era.select('volumetric_soil_water_layer_1').mean().rename('swvl1_mean_gf');
  var swvl2_mean   = era.select('volumetric_soil_water_layer_2').mean().rename('swvl2_mean_gf');

  // Soil water decline: first day - last day
  var swvl1_first = ee.Image(era.select('volumetric_soil_water_layer_1').first());
  var swvl1_last  = ee.Image(era.select('volumetric_soil_water_layer_1').sort('system:time_start', false).first());
  var swvl_decline = swvl1_first.subtract(swvl1_last).rename('swvl_decline_gf');

  var pe_cum     = era.select('potential_evaporation_sum').sum().rename('pe_cum_gf');
  var wind_max   = windCol.max().rename('u10_v10_speed_max_gf');
  var heat30     = hot30.sum().rename('hot_days_t2m_30');
  var heat35     = hot35.sum().rename('hot_days_t2m_35');

  var stack = ee.Image.cat([t2m_max_mean, t2m_max_max, skin_T_max,
                            soil_T_L1, soil_T_L2, swvl1_mean, swvl2_mean,
                            swvl_decline, pe_cum, wind_max, heat30, heat35]);

  return stack.reduceRegions({
    collection: centroids,
    reducer: ee.Reducer.first(),
    scale: 9000  // ERA5-Land native ~9km
  }).map(function(f) {
    return ee.Feature(f).set('harvest_year', yr);
  });
})).flatten();

print('Total samples:', allYearSamples.size());
print('First sample:', allYearSamples.first());

Export.table.toDrive({
  collection: allYearSamples,
  description: 'ERA5Land_grain_filling_buffer',
  folder: driveFolder,
  fileNamePrefix: 'era5land_grain_filling_buffer',
  fileFormat: 'CSV',
  selectors: ['FIELDID', 'harvest_year',
              't2m_max_gf', 't2m_max_max_gf', 'skin_T_max_gf',
              'soil_T_L1_mean_gf', 'soil_T_L2_mean_gf',
              'swvl1_mean_gf', 'swvl2_mean_gf', 'swvl_decline_gf',
              'pe_cum_gf', 'u10_v10_speed_max_gf',
              'hot_days_t2m_30', 'hot_days_t2m_35']
});

print('Export ready. Run from Tasks tab.');
print('Expected: ~30,600 rows (6,120 fields × 5 years)');
