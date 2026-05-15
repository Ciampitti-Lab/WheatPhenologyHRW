# 2018–2024 Extension Scripts

These scripts extend the original 2013–2017 extraction (`scripts/00_extraction/01_hls.js`,
`02_modis_lst.js`, `scripts/01_data_prep/03_daymet_temperature.py`,
`05_daymet_extra_vars.py`) to additional calendar years for the climate-trend
analysis (paper figure F7). They do **not** replace the original scripts —
the training set stays as published.

## Phased plan

| Phase | Years     | HLS GEE | MODIS LST GEE | Daymet REST  |
|------:|-----------|---------|---------------|--------------|
| 1     | 2018      | YEAR=2018 | YEAR=2018   | `--years 2018` |
| 2     | 2019–2021 | each year | each year   | `--years 2019,2020,2021` |
| 3     | 2022–2024 | each year | each year   | `--years 2022,2023,2024` |

Phase 1 first to confirm the pipeline still works against current GEE
collection IDs and Daymet's API. Then 2 and 3 in parallel once Phase 1
is verified.

## Files

- **`01_hls_extension.js`** — HLS L8 + S2 zonal-mean per buffer polygon,
  one calendar year per run (`YEAR` variable). Drive folder
  `WheatPhenologyHRW_HLS_extension`. Output: `buffer_l8_timeseries_<YEAR>.csv`
  + `buffer_s2_timeseries_<YEAR>.csv`.

- **`02_modis_lst_extension.js`** — MOD11A2 day/night LST, single year.
  Output: `modis_lst_buffer_<YEAR>.csv`.

- **`03_cdl_wheat_mask.js`** — USDA NASS CDL winter-wheat fraction per
  field × year, all 7 years in one run. Output:
  `cdl_wheat_fraction_2018_2024.csv`. Downstream we keep field-years
  with `wheat >= 0.5`.

- **`04_daymet_extension.py`** — Daymet REST pull (tmin, tmax, prcp, srad,
  vp, swe), one or more years per run, resumable via per-batch checkpoint
  CSV.

## After extraction

1. Concat all yearly HLS/MODIS CSVs into a single time-series parquet.
2. Apply the CDL wheat-fraction filter (`wheat >= 0.5`) per field-year.
3. Run the existing feature pipeline (`scripts/02_features/`) on the
   filtered (FIELDID, harvest_year) tuples.
4. Run the trained A6 multi-stage model in inference mode.
5. Build F7 — per-state mean anthesis DOY vs harvest year, with linear
   trend slope ± 95 % CI.

## Note on validation

The model is trained on harvest years 2013–2017. Predictions for
2018–2024 are out-of-training-window inferences without ground-truth
phenology labels. F7 will report this caveat explicitly: trend is
descriptive, not validated point-by-point.
