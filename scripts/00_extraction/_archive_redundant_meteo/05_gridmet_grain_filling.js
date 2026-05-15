// =================================================================
// 10 — GRIDMET grain-filling features (per field-year aggregates)
// =================================================================
// Source: IDAHO_EPSCOR/GRIDMET — daily, 4km, US
// Computes grain-filling-window aggregates for atmospheric demand & heat stress
//
// Window: DOY 105–175 (mid-Apr to late Jun) of harvest year — covers
// pre-anthesis, anthesis, grain-filling for HRW Wheat in US Plains
//
// Output CSV: gridmet_grain_filling_buffer.csv
//   Per (FIELDID, harvest_year):
//     vpd_max_gf    — max VPD during window (kPa)
//     vpd_mean_gf   — mean VPD (kPa)
//     vpd_cum_gf    — sum VPD (kPa·day)
//     eto_cum_gf    — cumulative reference ET (mm)
//     tmmx_max_gf   — max daily Tmax (K → convert downstream)
//     heat_days_30  — days Tmax > 303.15K (30°C)
//     heat_days_32  — days Tmax > 305.15K (32°C)
//     heat_days_35  — days Tmax > 308.15K (35°C)
//     srad_cum_gf   — cumulative solar radiation (W/m²·day)
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
print('First FIELDID:', ee.Feature(fields.first()).get('FIELDID'));

var driveFolder = 'WheatFlagLeaf_HLS';
var harvestYears = [2013, 2014, 2015, 2016, 2017];

// For each harvest year, filter GRIDMET to grain-filling window and reduce to per-field stats
var allYearSamples = ee.FeatureCollection(harvestYears.map(function(yr) {
  yr = ee.Number(yr);
  var startDate = ee.Date.fromYMD(yr, 4, 15);  // DOY ~105
  var endDate   = ee.Date.fromYMD(yr, 6, 30);  // DOY ~181

  var gm = ee.ImageCollection('IDAHO_EPSCOR/GRIDMET')
    .filterDate(startDate, endDate)
    .select(['vpd', 'eto', 'tmmx', 'srad']);

  // Threshold-based heat day counts (1 if Tmax exceeds, else 0)
  var hot30 = gm.map(function(im) { return im.select('tmmx').gt(303.15).rename('hot30'); });
  var hot32 = gm.map(function(im) { return im.select('tmmx').gt(305.15).rename('hot32'); });
  var hot35 = gm.map(function(im) { return im.select('tmmx').gt(308.15).rename('hot35'); });

  // Aggregates
  var vpd_max  = gm.select('vpd').max().rename('vpd_max_gf');
  var vpd_mean = gm.select('vpd').mean().rename('vpd_mean_gf');
  var vpd_sum  = gm.select('vpd').sum().rename('vpd_cum_gf');
  var eto_sum  = gm.select('eto').sum().rename('eto_cum_gf');
  var tmmx_max = gm.select('tmmx').max().rename('tmmx_max_gf');
  var srad_sum = gm.select('srad').sum().rename('srad_cum_gf');
  var heat30   = hot30.sum().rename('heat_days_30');
  var heat32   = hot32.sum().rename('heat_days_32');
  var heat35   = hot35.sum().rename('heat_days_35');

  var stack = ee.Image.cat([vpd_max, vpd_mean, vpd_sum, eto_sum,
                            tmmx_max, srad_sum, heat30, heat32, heat35]);

  return stack.reduceRegions({
    collection: centroids,
    reducer: ee.Reducer.first(),
    scale: 4000
  }).map(function(f) {
    return ee.Feature(f).set('harvest_year', yr);
  });
})).flatten();

print('Total samples (5 yr × ~6120 fields):', allYearSamples.size());
print('First sample:', allYearSamples.first());

Export.table.toDrive({
  collection: allYearSamples,
  description: 'GRIDMET_grain_filling_buffer',
  folder: driveFolder,
  fileNamePrefix: 'gridmet_grain_filling_buffer',
  fileFormat: 'CSV',
  selectors: ['FIELDID', 'harvest_year',
              'vpd_max_gf', 'vpd_mean_gf', 'vpd_cum_gf', 'eto_cum_gf',
              'tmmx_max_gf', 'srad_cum_gf',
              'heat_days_30', 'heat_days_32', 'heat_days_35']
});

print('Export ready. Run from Tasks tab.');
print('Expected: ~30,600 rows (6,120 fields × 5 years)');
