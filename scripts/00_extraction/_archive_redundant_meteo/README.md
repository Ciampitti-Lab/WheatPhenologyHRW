# Archived — redundant meteo extractions

These six GEE extraction scripts pulled grain-filling and pre-anthesis windowed
features from **GridMET (4 km)**, **ERA5-Land (~9 km)** and **CHIRPS (5 km)**.

A leave-one-year-out ablation (`data/results/meteo_ablation.csv`) showed that
dropping all three sources — keeping only **Daymet (1 km, daily)** as the
canonical weather source plus **MOD16 (500 m)** for satellite ET — produces
**equal or better R²** on every stage:

| Stage     | FULL R² | SLIM R² | Δ      |
|-----------|--------:|--------:|-------:|
| jointing  | 0.347   | 0.365   | +0.018 |
| heading   | 0.780   | 0.806   | +0.027 |
| anthesis  | 0.823   | 0.834   | +0.011 |
| flag leaf | 0.818   | 0.808   | −0.010 |
| boot      | 0.797   | 0.798   | +0.001 |
| maturity  | 0.561   | 0.569   | +0.007 |

The redundant lower-resolution datasets were inflating the feature space and
hurting `SelectKBest` ranking. Daymet provides T_max/T_min/precip/srad/VPD at
finer spatial resolution; PET/heat-stress windows are computed in-script via
Hargreaves directly from Daymet daily.

These scripts are kept here for provenance only. They are no longer wired into
the active pipeline.
