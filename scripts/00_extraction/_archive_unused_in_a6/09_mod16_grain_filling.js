// =================================================================
// 12 — MOD16 Evapotranspiration grain-filling features
// =================================================================
// Source: MODIS/061/MOD16A2GF — 8-day composite, 500m, ET/PET/LE/PLE
//   Gap-filled product (suffix GF), better quality than MOD16A2
// Use case: Actual ET, PET, water stress (1 - ET/PET) during grain-filling
//
// Window: DOY 105–175 (mid-Apr to late Jun)
//
// Output CSV: mod16_grain_filling_buffer.csv
//   Per (FIELDID, harvest_year):
//     ET_cum_gf       — cumulative actual ET (kg/m² over window)
//     PET_cum_gf      — cumulative potential ET
//     LE_mean_gf      — mean latent heat flux (J/m²/day)
//     PLE_mean_gf     — mean potential latent heat flux
//     ET_PET_ratio    — ET_cum / PET_cum (water stress index, 0=fully stressed, 1=no stress)
//     ET_deficit_gf   — PET_cum - ET_cum (mm water deficit)
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

  // MOD16 has scale factor 0.1 for ET/PET; bands are kg/m²/8day (sum within composite)
  // We sum the 8-day composites to get total over window
  var mod16 = ee.ImageCollection('MODIS/061/MOD16A2GF')
    .filterDate(startDate, endDate)
    .select(['ET', 'PET', 'LE', 'PLE'])
    .map(function(im) {
      // Apply scale factor 0.1 for ET/PET (kg/m²/8day → mm/8day)
      var et  = im.select('ET').multiply(0.1).rename('ET');
      var pet = im.select('PET').multiply(0.1).rename('PET');
      var le  = im.select('LE').multiply(10000).rename('LE');   // scale 0.0001
      var ple = im.select('PLE').multiply(10000).rename('PLE'); // scale 0.0001
      return et.addBands([pet, le, ple]);
    });

  var et_sum  = mod16.select('ET').sum().rename('ET_cum_gf');
  var pet_sum = mod16.select('PET').sum().rename('PET_cum_gf');
  var le_mean = mod16.select('LE').mean().rename('LE_mean_gf');
  var ple_mean= mod16.select('PLE').mean().rename('PLE_mean_gf');

  var et_pet_ratio = et_sum.divide(pet_sum.max(1)).rename('ET_PET_ratio');
  var et_deficit   = pet_sum.subtract(et_sum).rename('ET_deficit_gf');

  var stack = ee.Image.cat([et_sum, pet_sum, le_mean, ple_mean,
                            et_pet_ratio, et_deficit]);

  return stack.reduceRegions({
    collection: centroids,
    reducer: ee.Reducer.first(),
    scale: 500
  }).map(function(f) {
    return ee.Feature(f).set('harvest_year', yr);
  });
})).flatten();

print('Total samples:', allYearSamples.size());
print('First sample:', allYearSamples.first());

Export.table.toDrive({
  collection: allYearSamples,
  description: 'MOD16_grain_filling_buffer',
  folder: driveFolder,
  fileNamePrefix: 'mod16_grain_filling_buffer',
  fileFormat: 'CSV',
  selectors: ['FIELDID', 'harvest_year',
              'ET_cum_gf', 'PET_cum_gf', 'LE_mean_gf', 'PLE_mean_gf',
              'ET_PET_ratio', 'ET_deficit_gf']
});

print('Export ready. Run from Tasks tab.');
print('Expected: ~30,600 rows (6,120 fields × 5 years)');
