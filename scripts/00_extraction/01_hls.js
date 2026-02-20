// ============================================================
// 01b - HLS Buffer-Only Export (6,120 fields, ALL points)
// ============================================================
// Upload the NEW wheat_fields_buffer300m_polygons.zip first
// ============================================================

var fieldsBuffer = ee.FeatureCollection('projects/propane-primacy-481403-u3/assets/wheat_fields_buffer300m_polygons');

var startL8 = '2013-04-01';
var startS2 = '2015-07-01';
var endDate = '2017-12-31';
var driveFolder = 'WheatFlagLeaf_HLS';

// Cloud masking
function maskHLS(image) {
  var fmask = image.select('Fmask');
  var cloud = fmask.bitwiseAnd(1 << 1);
  var adjacent = fmask.bitwiseAnd(1 << 2);
  var shadow = fmask.bitwiseAnd(1 << 3);
  var mask = cloud.eq(0).and(adjacent.eq(0)).and(shadow.eq(0));
  return image.updateMask(mask);
}

// Rename bands
function renameL8(image) {
  return image.select(
    ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'Fmask'],
    ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'Fmask']
  );
}

function renameS2(image) {
  return image.select(
    ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12', 'Fmask'],
    ['Blue', 'Green', 'Red', 'RE1', 'RE2', 'RE3', 'NIR', 'SWIR1', 'SWIR2', 'Fmask']
  );
}

// Scale
function scaleHLS(image) {
  var scaled = image.select(['Blue','Green','Red','NIR','SWIR1','SWIR2'])
    .multiply(0.0001);
  var bandNames = image.bandNames();
  var hasRE = bandNames.contains('RE1');
  scaled = ee.Algorithms.If(hasRE,
    scaled.addBands(image.select(['RE1','RE2','RE3']).multiply(0.0001)),
    scaled
  );
  return ee.Image(scaled).copyProperties(image, image.propertyNames());
}

// Indices L8
function addIndicesL8(image) {
  var ndvi = image.normalizedDifference(['NIR', 'Red']).rename('NDVI');
  var evi = image.expression(
    '2.5 * ((NIR - RED) / (NIR + 6.0 * RED - 7.5 * BLUE + 1.0))', {
      'NIR': image.select('NIR'),
      'RED': image.select('Red'),
      'BLUE': image.select('Blue')
    }).rename('EVI');
  var gcvi = image.select('NIR').divide(image.select('Green')).subtract(1).rename('GCVI');
  return image.addBands([ndvi, evi, gcvi])
    .set('date', image.date().format('YYYY-MM-dd'))
    .set('doy', image.date().getRelative('day', 'year').add(1))
    .set('sensor', 'L8');
}

// Indices S2
function addIndicesS2(image) {
  var ndvi = image.normalizedDifference(['NIR', 'Red']).rename('NDVI');
  var evi = image.expression(
    '2.5 * ((NIR - RED) / (NIR + 6.0 * RED - 7.5 * BLUE + 1.0))', {
      'NIR': image.select('NIR'),
      'RED': image.select('Red'),
      'BLUE': image.select('Blue')
    }).rename('EVI');
  var gcvi = image.select('NIR').divide(image.select('Green')).subtract(1).rename('GCVI');
  var ndre = image.normalizedDifference(['NIR', 'RE1']).rename('NDRE');
  var cire = image.select('NIR').divide(image.select('RE1')).subtract(1).rename('CIre');
  var mtci = image.expression(
    '(RE2 - RE1) / (RE1 - RED)', {
      'RE2': image.select('RE2'),
      'RE1': image.select('RE1'),
      'RED': image.select('Red')
    }).rename('MTCI');
  return image.addBands([ndvi, evi, gcvi, ndre, cire, mtci])
    .set('date', image.date().format('YYYY-MM-dd'))
    .set('doy', image.date().getRelative('day', 'year').add(1))
    .set('sensor', 'S2');
}

// Zonal mean
function extractZonalMean(image, polygons, bands) {
  return image.select(bands).reduceRegions({
    collection: polygons,
    reducer: ee.Reducer.mean(),
    scale: 30
  }).filter(ee.Filter.notNull(['NDVI']))
    .map(function(f) {
      return f.set('date', image.get('date'))
              .set('doy', image.get('doy'))
              .set('sensor', image.get('sensor'));
    });
}

// Load HLS
var hlsL8 = ee.ImageCollection('NASA/HLS/HLSL30/v002')
  .filterDate(startL8, endDate)
  .filterBounds(fieldsBuffer)
  .map(renameL8).map(maskHLS).map(scaleHLS).map(addIndicesL8);

var hlsS2 = ee.ImageCollection('NASA/HLS/HLSS30/v002')
  .filterDate(startS2, endDate)
  .filterBounds(fieldsBuffer)
  .map(renameS2).map(maskHLS).map(scaleHLS).map(addIndicesS2);

print('L8 images:', hlsL8.size());
print('S2 images:', hlsS2.size());

// Extract
var bandsL8 = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'NDVI', 'EVI', 'GCVI'];
var bandsS2 = ['Blue', 'Green', 'Red', 'RE1', 'RE2', 'RE3', 'NIR', 'SWIR1', 'SWIR2',
               'NDVI', 'EVI', 'GCVI', 'NDRE', 'CIre', 'MTCI'];

var bufL8 = hlsL8.map(function(img) { return extractZonalMean(img, fieldsBuffer, bandsL8); }).flatten();
var bufS2 = hlsS2.map(function(img) { return extractZonalMean(img, fieldsBuffer, bandsS2); }).flatten();

print('Buffer L8 observations:', bufL8.size());
print('Buffer S2 observations:', bufS2.size());

// Export
Export.table.toDrive({
  collection: bufL8,
  description: 'Buffer_L8_timeseries',
  folder: driveFolder,
  fileNamePrefix: 'buffer_l8_timeseries',
  fileFormat: 'CSV',
  selectors: ['FIELDID', 'date', 'doy', 'sensor',
              'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2',
              'NDVI', 'EVI', 'GCVI']
});

Export.table.toDrive({
  collection: bufS2,
  description: 'Buffer_S2_timeseries',
  folder: driveFolder,
  fileNamePrefix: 'buffer_s2_timeseries',
  fileFormat: 'CSV',
  selectors: ['FIELDID', 'date', 'doy', 'sensor',
              'Blue', 'Green', 'Red', 'RE1', 'RE2', 'RE3', 'NIR', 'SWIR1', 'SWIR2',
              'NDVI', 'EVI', 'GCVI', 'NDRE', 'CIre', 'MTCI']
});

print('2 Buffer export tasks ready. Go to Tasks tab and click RUN.');
