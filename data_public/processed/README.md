# De-identified public data subset

This directory is the **de-identified subset** released with the
manuscript so that the modelling and the label-noise figure (FA1) are
reproducible without access to the restricted source data. Released with
the permission of the phenology-data provider.

## Files

| File | Rows | Content |
|---|---|---|
| `phenology_labels.parquet`  | 58,303 | Per-field, per-observation growth-stage labels (`field_id`, `state`, `date`, `doy`, `dos`, `growth_stage`, `crop_condition`) |
| `sowing_lookup.parquet`     | 8,466  | Per field-year WES sowing anchor (`field_id`, `state`, `harvest_year`, `sowing_doy_used`, `source`) |
| `features_v3_train.parquet` | 8,465  | DOS-anchored model feature matrix (114 cols: HLS phenometrics, Daymet/MODIS-LST stress windows, WES outputs, `state_*` one-hots) |
| `field_id_mapping.json`     | —      | Anonymous-id **count only** |

Scope: the four training seasons **2013/14–2016/17** (harvest years
2014–2017), 5,294 fields, the cohort analysed in the paper.

## How it was de-identified

Produced by `scripts/deidentify_public_release.py` from the restricted
source:

1. The partner-issued field identifier (e.g. `BUF_00332`) is replaced by
   an **anonymous integer** (1–5294) via a deterministic, file-consistent
   map. **The original→anonymous mapping is discarded** and never
   written, so the anonymisation cannot be reversed from this release.
2. Exact field coordinates and geometry (`lat`, `lon`, `latitude`,
   `longitude`, `geometry`, centroids) are **dropped entirely**.
3. Only the four training seasons are kept.

What remains carries **no direct identifier and no field location**: the
coarsest geographic field is the U.S. state (one of TX, OK, KS, NE, CO,
NM). Re-identifying a record would require already holding the original
provider dataset.

## Use & citation

Provided under the repository's MIT licence for research reproduction.
If you use this subset, cite the associated paper (see `CITATION.cff`).
The underlying raw field phenology observations and field-polygon
geometries remain subject to the provider's data-sharing agreement and
are **not** redistributed here.
