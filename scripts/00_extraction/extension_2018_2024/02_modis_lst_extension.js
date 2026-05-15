// =================================================================
// MODIS LST Extension — single year, parameterized
// =================================================================
// Same logic as scripts/00_extraction/02_modis_lst.js but parameterized
// for one calendar year. Used to backfill LST data for 2018–2024 to
// support out-of-training-window inference (climate-trend figure F7).
// =================================================================

var YEAR = 2018;  // <-- change between phases

var startDate = YEAR + '-01-01';
var endDate   = (YEAR + 1) + '-01-01';
var driveFolder = 'WheatFlagLeaf_HLS_extension';
var suffix = '_' + YEAR;

var fields = ee.FeatureCollection(
  'projects/propane-primacy-481403-u3/assets/wheat_fields_buffer300m_polygons_new');

var centroids = fields.map(function(f) {
  var feat = ee.Feature(f);
  var geom = ee.Geometry(feat.geometry()).centroid({maxError: 1});
  return feat.setGeometry(geom);
});

var modisLST = ee.ImageCollection('MODIS/061/MOD11A2')
  .filterDate(startDate, endDate)
  .filterBounds(fields)
  .select(['LST_Day_1km', 'LST_Night_1km']);

print('Year:', YEAR);
print('MODIS LST images:', modisLST.size());

var lstScaled = modisLST.map(function(img) {
  var lst_day_C   = img.select('LST_Day_1km').multiply(0.02).subtract(273.15).rename('lst_day_C');
  var lst_night_C = img.select('LST_Night_1km').multiply(0.02).subtract(273.15).rename('lst_night_C');
  return lst_day_C.addBands(lst_night_C)
    .set('date', img.date().format('YYYY-MM-dd'));
});

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

Export.table.toDrive({
  collection: samples,
  description: 'MODIS_LST_buffer' + suffix,
  folder: driveFolder,
  fileNamePrefix: 'modis_lst_buffer' + suffix,
  fileFormat: 'CSV',
  selectors: ['FIELDID', 'date', 'lst_day_C', 'lst_night_C']
});

print('MODIS LST export ready for', YEAR, '— go to Tasks tab and RUN.');
