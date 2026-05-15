// =================================================================
// 16 — MOD16 pre-anthesis ET features
// =================================================================
// Window: Mar 1 – Apr 14 of harvest year
//
// Output CSV: mod16_pre_anthesis_buffer.csv
//   Per (FIELDID, harvest_year):
//     ET_cum_pa, PET_cum_pa
//     ET_PET_ratio_pa  (water stress entering pre-anthesis: 0=stressed, 1=full)
//     ET_deficit_pa  (PET - ET, mm)
//     LE_mean_pa
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

  var mod16 = ee.ImageCollection('MODIS/061/MOD16A2GF')
    .filterDate(startDate, endDate)
    .select(['ET', 'PET', 'LE'])
    .map(function(im) {
      var et  = im.select('ET').multiply(0.1).rename('ET');
      var pet = im.select('PET').multiply(0.1).rename('PET');
      var le  = im.select('LE').multiply(10000).rename('LE');
      return et.addBands([pet, le]);
    });

  var et_sum  = mod16.select('ET').sum().rename('ET_cum_pa');
  var pet_sum = mod16.select('PET').sum().rename('PET_cum_pa');
  var le_mean = mod16.select('LE').mean().rename('LE_mean_pa');
  var et_pet_ratio = et_sum.divide(pet_sum.max(1)).rename('ET_PET_ratio_pa');
  var et_deficit   = pet_sum.subtract(et_sum).rename('ET_deficit_pa');

  var stack = ee.Image.cat([et_sum, pet_sum, le_mean, et_pet_ratio, et_deficit]);

  return stack.reduceRegions({
    collection: centroids,
    reducer: ee.Reducer.first(),
    scale: 500
  }).map(function(f) { return ee.Feature(f).set('harvest_year', yr); });
})).flatten();

print('Total samples:', allYearSamples.size());
print('First sample:', allYearSamples.first());

Export.table.toDrive({
  collection: allYearSamples,
  description: 'MOD16_pre_anthesis_buffer',
  folder: driveFolder,
  fileNamePrefix: 'mod16_pre_anthesis_buffer',
  fileFormat: 'CSV',
  selectors: ['FIELDID', 'harvest_year',
              'ET_cum_pa', 'PET_cum_pa', 'LE_mean_pa',
              'ET_PET_ratio_pa', 'ET_deficit_pa']
});

print('Export ready. Run from Tasks tab.');
print('Expected: ~30,600 rows (6,120 fields × 5 years)');
