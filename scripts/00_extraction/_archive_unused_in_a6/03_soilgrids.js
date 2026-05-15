// =================================================================
// 08 — gNATSGO Soil Properties Extraction (via SoilGrids 250m)
// =================================================================
// Source: SoilGrids 250m (global, ISRIC)
// Static (one-time) — soil texture, OC, pH, bulk density at top layer
//
// Output CSV: gnatsgo_soil_buffer.csv
//   Columns: FIELDID (BUF_xxxxx), clay_top, sand_top, silt_top, soc_top, ph_top, bdod_top
//
// Run in: code.earthengine.google.com
// =================================================================

// BUF asset has FIELDID = BUF_xxxxx natively. No alias needed.
var fields = ee.FeatureCollection('projects/propane-primacy-481403-u3/assets/wheat_fields_buffer300m_polygons');

var centroids = fields.map(function(f) {
  var feat = ee.Feature(f);
  var geom = ee.Geometry(feat.geometry()).centroid({maxError: 1});
  return feat.setGeometry(geom);
});

print('Total fields:', fields.size());
print('First FIELDID:', ee.Feature(fields.first()).get('FIELDID'));
print('First centroid FIELDID:', ee.Feature(centroids.first()).get('FIELDID'));

// SoilGrids 250m
var soilgrids = ee.Image('projects/soilgrids-isric/clay_mean')
  .addBands(ee.Image('projects/soilgrids-isric/sand_mean'))
  .addBands(ee.Image('projects/soilgrids-isric/silt_mean'))
  .addBands(ee.Image('projects/soilgrids-isric/soc_mean'))
  .addBands(ee.Image('projects/soilgrids-isric/phh2o_mean'))
  .addBands(ee.Image('projects/soilgrids-isric/bdod_mean'));

var topsoil = soilgrids
  .select(['clay_0-5cm_mean', 'sand_0-5cm_mean', 'silt_0-5cm_mean',
           'soc_0-5cm_mean', 'phh2o_0-5cm_mean', 'bdod_0-5cm_mean'])
  .rename(['clay_top', 'sand_top', 'silt_top', 'soc_top', 'ph_top', 'bdod_top']);

var samples = topsoil.reduceRegions({
  collection: centroids,
  reducer: ee.Reducer.first(),
  scale: 250
});

print('=== DEBUG ===');
print('First sample propertyNames:', ee.Feature(samples.first()).propertyNames());
print('First sample FIELDID:', ee.Feature(samples.first()).get('FIELDID'));
print('First sample clay_top:', ee.Feature(samples.first()).get('clay_top'));

Export.table.toDrive({
  collection: samples,
  description: 'gNATSGO_SoilGrids_buffer',
  folder: 'WheatFlagLeaf_HLS',
  fileNamePrefix: 'gnatsgo_soil_buffer',
  fileFormat: 'CSV',
  selectors: ['FIELDID', 'clay_top', 'sand_top', 'silt_top', 'soc_top', 'ph_top', 'bdod_top']
});

print('Export ready. Run from Tasks tab.');
