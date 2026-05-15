// =================================================================
// 09 — SMAP Soil Moisture Extraction
// =================================================================
// Source: NASA-USDA SMAP Global Soil Moisture (NASA_USDA/HSL/SMAP10KM_soil_moisture)
// Coverage: April 2015+ (missing 2013-2014!)
// Resolution: 10km, 3-day composites
// Use case: Drought stress → maturity prediction
//
// Output CSV: smap_soil_moisture_buffer.csv
//   Columns: FIELDID (BUF_xxxxx), date, ssm, susm, smp, ssma, susma
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

var startDate = '2015-04-01';
var endDate   = '2017-12-31';
var driveFolder = 'WheatFlagLeaf_HLS';

var smap = ee.ImageCollection('NASA_USDA/HSL/SMAP10KM_soil_moisture')
  .filterDate(startDate, endDate)
  .filterBounds(fields)
  .select(['ssm', 'susm', 'smp', 'ssma', 'susma'])
  .map(function(img) {
    return img.set('date', img.date().format('YYYY-MM-dd'));
  });

print('SMAP images:', smap.size());

var samples = smap.map(function(img) {
  return img.reduceRegions({
    collection: centroids,
    reducer: ee.Reducer.first(),
    scale: 10000
  }).filter(ee.Filter.notNull(['ssm']))
    .map(function(f) {
      return ee.Feature(f).set('date', img.get('date'));
    });
}).flatten();

print('First samples:', samples.limit(3));

Export.table.toDrive({
  collection: samples,
  description: 'SMAP_buffer',
  folder: driveFolder,
  fileNamePrefix: 'smap_soil_moisture_buffer',
  fileFormat: 'CSV',
  selectors: ['FIELDID', 'date', 'ssm', 'susm', 'smp', 'ssma', 'susma']
});

print('Export ready. Run from Tasks tab.');
print('Expected: ~6,120 fields × ~330 3-day periods = ~2M rows');
print('NOTE: 2013-2014 NOT covered (SMAP launched April 2015)');
