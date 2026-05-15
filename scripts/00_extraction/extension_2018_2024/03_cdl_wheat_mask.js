// =================================================================
// CDL Winter-Wheat Mask — flag which fields were wheat in each year
// =================================================================
// USDA NASS Cropland Data Layer (CDL) is a 30 m annual classification.
// Class 24 = winter wheat, 26 = double-crop winter wheat / soybeans.
//
// For each (FIELDID, year) we compute the fraction of the buffer
// polygon classified as winter wheat. Downstream we keep only
// field-years with wheat_frac >= 0.5 — i.e. the dominant land cover
// inside the 300 m buffer was wheat that year.
//
// One run covers all extension years; output is a long-format
// FeatureCollection (one row per FIELDID × year).
// =================================================================

var YEARS = [2018, 2019, 2020, 2021, 2022, 2023, 2024];
var driveFolder = 'WheatFlagLeaf_HLS_extension';

var fields = ee.FeatureCollection(
  'projects/propane-primacy-481403-u3/assets/wheat_fields_buffer300m_polygons_new');

print('Total fields:', fields.size());

// Build the per-year FC inside a plain JS for-loop. Mixing
// ee.List.map with image-ID string concatenation runs into the
// "mapped function's arguments cannot be used in client-side
// operations" error, because ee.Image() needs a client-side string.
var combined = null;
for (var i = 0; i < YEARS.length; i++) {
  var yr = YEARS[i];
  var cdl = ee.Image('USDA/NASS/CDL/' + yr).select('cropland');
  var wheat = cdl.eq(24).or(cdl.eq(26)).rename('wheat');
  // setOutputs(['wheat']) forces the reducer's output property name to
  // be 'wheat' instead of the default 'mean' — otherwise the export
  // selector below sees nothing and writes empty cells.
  var fc = wheat.reduceRegions({
    collection: fields,
    reducer: ee.Reducer.mean().setOutputs(['wheat']),
    scale: 30
  }).map(function(f) {
    return ee.Feature(f).set('year', yr);
  });
  combined = (combined === null) ? fc : combined.merge(fc);
}

print('Combined size (FIELDID × year rows):', combined.size());

Export.table.toDrive({
  collection: combined,
  description: 'CDL_wheat_fraction_2018_2024',
  folder: driveFolder,
  fileNamePrefix: 'cdl_wheat_fraction_2018_2024',
  fileFormat: 'CSV',
  selectors: ['FIELDID', 'year', 'wheat']  // 'wheat' = polygon-mean wheat fraction
});

print('CDL wheat-fraction export ready — RUN from Tasks tab.');
print('Output column "wheat" = fraction of buffer pixels classified as winter wheat that year.');
