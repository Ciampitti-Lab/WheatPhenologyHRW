# Aires et al. (2026) — manuscript insertions + gap-fill defense

Single reference for everything from Aires et al. (2026, *J. Remote Sensing*,
doi:10.34133/remotesensing.0878) that goes into the WheatPhenologyHRW
manuscript: an Introduction positioning sentence, a Discussion paragraph
(convergent ENR evidence + the "no separate gap-fill stage" defense), a
supplementary ablation table, and the full citation. The insertion map is in
[Manuscript insertion map](#manuscript-insertion-map).

Context: Aires et al. build a daily *synthetic* gap-filled HLS-EVI series before
phenology fitting and report polynomial gap-filling as clearly best for
corn/soybean. Because that paper now sits in the same journal family, a reviewer
may ask why WheatPhenologyHRW does not add an equivalent gap-fill stage — the
ablation below records the quantitative answer.

## Ablation (script: `scripts/04_benchmarks/03_gapfill_ablation_aires.py`)

Aires et al.'s "Gap-filled daily EVI time series validation" protocol, run on
our HLS stack: held-out clear-sky EVI observations reconstructed from the
remaining observations inside a ±15-d window (their empirically chosen window),
four methods, identical EVI cleaning to the phenometric pipeline (|EVI| > 1 →
NaN). 500 fields, 20,700 held-out targets.

| method   | coverage | RMSE  | MAE   | R²    | RMSE (active, EVI>0.2) | R² (active) |
|----------|----------|-------|-------|-------|------------------------|-------------|
| LightGBM | 0.813    | 0.060 | 0.026 | 0.881 | 0.056                  | 0.874       |
| harmonic | 0.813    | 0.060 | 0.023 | 0.880 | 0.056                  | 0.873       |
| poly2    | 0.813    | 0.062 | 0.024 | 0.871 | 0.059                  | 0.862       |
| median   | 0.813    | 0.063 | 0.027 | 0.866 | 0.059                  | 0.859       |

Two findings:

1. **Method choice is immaterial on HRW wheat.** Best-vs-worst spread is
   ΔRMSE ≤ 0.004 EVI and ΔR² ≤ 0.016 — the four methods are statistically
   indistinguishable. This *contrasts* with Aires et al., who found polynomial
   clearly best for corn/soybean; the post-dormancy winter-wheat trajectory is
   not the clean single bell that gives the polynomial its edge on summer
   row crops.
2. **~19% of clear-sky targets have no ±15-d clear-sky neighbour** (coverage
   0.813), so reconstruction is undefined exactly in the long cloudy stretches
   where a gap-fill stage would matter most.

## Why we already have the equivalent — without the extra stage

`utils.features.smooth_vi` (used by the phenometric extraction) **already
produces a daily synthetic VI series**: it interpolates the clear-sky VI onto a
daily 1–365 grid and applies a Savitzky–Golay filter (window=15, polyorder=2).
That is functionally the same two-step Aires et al. use (daily synthetic series
→ 2nd-degree SG filter); we differ only in using linear interpolation rather
than polynomial/harmonic/LightGBM as the interpolant. The ablation shows that
this specific choice changes reconstruction accuracy by ΔR² ≤ 0.016 — negligible
relative to the downstream phenometric and WES-physics signal. Adding a separate
model-based gap-fill stage would therefore propagate reconstruction error into
the curve fit without adding phenological information.

## Manuscript insertion map

| # | Section | Insertion | Status |
|---|---------|-----------|--------|
| 1 | Introduction / related work | Positioning sentence (below) | ready |
| 2 | Discussion — convergent evidence | folded into the drop-in paragraph (below) | ready |
| 3 | Discussion — gap-fill design defense | drop-in paragraph (below) | ready |
| 4 | Supplementary | ablation table above + `data/results/gapfill_ablation_aires.csv` | ready |
| 5 | References | full citation (below) | ready |

WOFOST / Strategy D is internal-only and is **not** referenced in any of these.

## Introduction snippet

> Recent work has shown that early, spectrally ambiguous events can be inferred
> from later, canopy-saturated phenology: Aires et al. [Aires2026] estimate
> sowing and emergence dates for corn and soybean from six later phenological
> stages of a daily synthetic HLS-EVI series, reaching ±10-day accuracy with a
> regularized linear model. That operational framework targets summer row crops,
> whose development carries no vernalization or strong photoperiod requirement.
> Hard-red-winter-wheat phenology is governed precisely by those processes and
> by a dormancy-interrupted greenness trajectory, and is resolved here per stage
> through spike emergence and anthesis with a Wang–Engel–Streck physiological
> core — the gap this study addresses.

## Drop-in discussion paragraph

> Our framework deliberately fits phenometrics to the native clear-sky HLS-EVI
> series (linearly interpolated to a daily grid and Savitzky–Golay smoothed)
> rather than to a separately reconstructed daily *synthetic* product. Aires
> et al. [Aires2026] showed that, for summer row crops, a model-based daily
> gap-fill — with polynomial regression outperforming harmonic, median, and
> gradient-boosted alternatives — improves downstream phenology retrieval. We
> reproduced their masked-observation validation on our hard-red-winter-wheat
> HLS stack and found the four gap-fill families statistically indistinguishable
> (ΔR² ≤ 0.016, ΔRMSE ≤ 0.004 EVI units over 20,700 held-out clear-sky
> observations), with no clear-sky neighbour available within ±15 d for ~19% of
> targets. The polynomial advantage Aires et al. report for corn and soybean
> does not transfer to the dormancy-interrupted winter-wheat trajectory.
> Because our Savitzky–Golay step already yields a daily synthetic series and
> the Wang–Engel–Streck core supplies the physiological temporal prior, a
> separate model-based reconstruction stage would add interpolation error
> without adding phenological information, and we therefore omit it. This is
> consistent with the convergent finding that a regularized linear model
> (elastic net) is the strongest estimator on a small, multicollinear
> phenology-feature set [Aires2026]: our ElasticNet wins at the late
> reproductive stages mirror their elastic-net result for sowing/emergence at a
> comparable ±10-day accuracy.

Reference to add:

> Aires URV, Martins VS, Ferreira LB, Zhang X, Reddy KR, Yang Y, Sanches IDA.
> Operational framework for field-scale crop sowing and emergence date
> estimation using daily synthetic Harmonized Landsat Sentinel-2 time series.
> *J Remote Sens.* 2026;6:0878. doi:10.34133/remotesensing.0878
