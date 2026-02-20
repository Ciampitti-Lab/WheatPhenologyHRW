// =================================================================
// 07 — MODIS LST (Land Surface Temperature) Extraction
// =================================================================
// Source: MOD11A2 (Terra MODIS, 1km, 8-day composites, day & night LST)
// Use case: Soil temperature proxy → emergence/tillering features
//
// Output CSV: modis_lst_buffer.csv
//   Columns: FIELDID (BUF_xxxxx), date, lst_day_C, lst_night_C
//
// Run in: code.earthengine.google.com
// =================================================================

// BUF asset has FIELDID = BUF_xxxxx natively
var fields = ee.FeatureCollection('projects/propane-primacy-481403-u3/assets/wheat_fields_buffer300m_polygons');

var centroids = fields.map(function(f) {
  var feat = ee.Feature(f);
  var geom = ee.Geometry(feat.geometry()).centroid({maxError: 1});
  return feat.setGeometry(geom);
});

print('Total fields:', fields.size());
print('First FIELDID:', ee.Feature(fields.first()).get('FIELDID'));

var startDate = '2012-09-01';
var endDate   = '2017-12-31';
var driveFolder = 'WheatFlagLeaf_HLS';

// MODIS Terra Land Surface Temperature & Emissivity 8-Day Global 1km
var modisLST = ee.ImageCollection('MODIS/061/MOD11A2')
  .filterDate(startDate, endDate)
  .filterBounds(fields)
  .select(['LST_Day_1km', 'LST_Night_1km']);

print('MODIS LST images:', modisLST.size());

// Convert to Celsius (raw scale: Kelvin × 0.02)
var lstScaled = modisLST.map(function(img) {
  var lst_day_C   = img.select('LST_Day_1km').multiply(0.02).subtract(273.15).rename('lst_day_C');
  var lst_night_C = img.select('LST_Night_1km').multiply(0.02).subtract(273.15).rename('lst_night_C');
  return lst_day_C.addBands(lst_night_C)
    .set('date', img.date().format('YYYY-MM-dd'));
});

// reduceRegions on centroids (FIELDID preserved via setGeometry), Reducer.first
var samples = lstScaled.map(function(img) {
  return img.reduceRegions({
    collection: centroids,
    reducer: ee.Reducer.first(),
    scale: 1000
  }).filter(ee.Filter.notNull(['lst_day_C']))
    .map(function(f) {
      return ee.Feature(f).set('date', img.get('date'));
    });
}).flatten();

print('First samples:', samples.limit(3));

Export.table.toDrive({
  collection: samples,
  description: 'MODIS_LST_buffer',
  folder: driveFolder,
  fileNamePrefix: 'modis_lst_buffer',
  fileFormat: 'CSV',
  selectors: ['FIELDID', 'date', 'lst_day_C', 'lst_night_C']
});

print('Export ready. Run from Tasks tab.');
print('Expected: ~6,120 fields × ~230 8-day periods = ~1.4M rows');
