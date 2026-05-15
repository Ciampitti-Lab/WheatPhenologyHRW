# Archived — sources not used in the final 3-source pipeline (A6)

These four GEE extraction scripts pulled features that, after a per-source
ablation (`data/results/per_source_ablation_summary.csv`), were shown to add
**negligible R² gain** over the HLS + Daymet + MODIS LST core. They are
kept here for provenance only.

| Source | Features extracted | Mean R² contribution |
|--------|-------------------|---------------------:|
| **SoilGrids 250 m** | clay/sand/silt/SOC/pH/BD top layer | +0.000 |
| **SMAP** soil moisture | ssm at SOS / pre-anthesis / grain-filling | +0.000 |
| **MOD16** ET | ET_cum, PET_cum, ET_deficit, LE_mean (gf + pa windows) | +0.000 |

**Why MODIS LST was kept:** it contributed +0.019 mean R² across 8 stages,
including a large +0.083 gain on tillering, +0.029 on maturity, and +0.023
on anthesis. It captures real surface-temperature drought stress (bare-soil
heating, canopy water status) that air-temperature features in Daymet cannot.

The final pipeline uses three sources:

1. **HLS** (Landsat 8 + Sentinel-2) — 30 m, 2-4 day, NASA
2. **Daymet V4** — 1 km, daily, ORNL
3. **MODIS LST** (MOD11A2) — 1 km, 8-day, NASA

These scripts can be restored if a future analysis needs the dropped sources;
no methodological dependency was lost.
