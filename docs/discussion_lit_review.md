# Discussion — literature gap analysis (CORRECTED)

> **Correction.** An earlier version of this file assumed no Discussion was
> written. That was wrong: a full, mature Discussion exists at
> `paper-overleaf/sections/discussion.tex` (target journal: Remote Sensing of
> Environment). This version assesses the *actual* text. Most "gaps" claimed
> earlier are already handled and are retracted below.

## Verdict: the Discussion is strong

It already does, well, the things the first draft of this file wrongly flagged
as missing:

- **LOYO/LOSO rigor is defended** — `roberts2017`, `ploton2020` are cited in
  `methods.tex` for spatial-CV-under-autocorrelation; LOSO strips the state
  encoder so transferability is measured after the regional baseline is removed.
  *(Retract earlier "Gap 3".)*
- **Weak early stages are framed rigorously** — not apologised for: physical
  cause (bare-soil regime, sparse ~2-composite/month cadence) **plus an
  empirical label-noise lower bound** (σ = 8.8 d tillering, 8.1 d jointing,
  6.8 d emergence) shown to be of the same order as the reproductive-to-
  vegetative RMSE gap. This is stronger than anything the earlier file
  proposed. *(Retract earlier "Gap 4/5".)*
- **Ground-truth limitation** — descriptive vs Zadoks labels, maturity
  mis-attribution correction, HLS gaps, state-median sowing fallback + a
  Gaussian-perturbation robustness experiment: all already in the limitations
  paragraph. *(Retract.)*
- **DL is addressed** — `methods.tex` explicitly justifies *not* using
  TabNet/FT-Transformer/LSTM/transformer (`grinsztajn2022`, `shwartzziv2022`).

So: well-structured (context → reproductive/vegetative gap → selective value of
physiology → LOSO/Colorado → climate trends → operational use → limitations),
honest, and self-critical. It does not need more bibliography to be *defensible*.

## What genuinely would strengthen it (3 items, not 7)

The one real weakness: the **"performance in context" opening relies entirely
on 2010–2020 MODIS / Landsat–MODIS-fusion work** (`gao2017`, `sakamoto2010`,
`bolton2013`, `zeng2020`). For a 2026 RSE submission a reviewer will
immediately ask about recent field-scale HLS comparators. None are cited.

| Pri | Reference | Where in `discussion.tex` | Why it is needed |
|----|-----------|---------------------------|------------------|
| **High** | **Aires et al. 2026, *J. Remote Sens.* 6:0878** | opening context para + limitations (HLS-gap sentence) | Same crop family of problem (early-stage dating from later HLS phenology), same journal family, 2026. Also supplies the gap-fill-defense angle already prepared in `aires2026_gapfill_defense.md`. Its absence is the most exposed omission. |
| **High** | **Zhou et al. 2024, *ISPRS J. P&RS* 216:259** (field-level planting dates, corn/soy, HLS phenometrics; corn R²≈0.77/MAE 4.6 d, soy 0.71/5.4 d) | opening context para | The closest *field-scale, modern* quantitative benchmark. Lets the ±10-day claim be anchored to a 2024 field-level number instead of decade-old pixel-level MODIS. |
| **Med** | **Bandaru et al. 2020 (PhenoCrop)** + one PBM-DL framing cite (e.g. hybrid PBM–DL review, arXiv 2504.16141, 2025) | the "value of the physiology-informed strategy" para | Intro claims novelty (iii) "fuses physiology priors with ML corrections" but the Discussion never positions this against the direct PhenoCrop lineage or the crowded 2024–25 hybrid-modeling field → "incremental" risk at review. Bandaru 2020 is already tracked in `pdfs/METHODOLOGY_LINES.md` but is **not** in `references.bib`. |
| Low (optional) | A shape-model-fitting phenology-retrieval ref | one sentence in context para | SMF is the dominant alternative paradigm; one sentence ("we use phenometric+physiology+ML rather than per-crop reference-curve SMF") closes a predictable reviewer question. Skip if length-constrained. |

## Recommended concrete edits

1. `references.bib`: add `aires2026`, `zhou2024`, `bandaru2020` (+ optional
   `hybridpbmdl2025`). Verify volume/page/DOI before final insertion.
2. `discussion.tex`, opening sentence — extend the comparator list from
   "{gao2017, sakamoto2010, bolton2013}" to include the two modern field-scale
   HLS works, with one clause noting both report early-stage dating as the
   hardest target (convergent with our emergence result).
3. `discussion.tex`, physiology-strategy paragraph — one sentence placing
   WES+ML in the PBM-informed-ML lineage relative to PhenoCrop.
4. The Aires gap-fill defense paragraph (ready in
   `aires2026_gapfill_defense.md`) slots into the **limitations** paragraph
   right after the existing Savitzky–Golay / HLS-gap sentence.

WOFOST/Strategy D stays out (internal-only). Lobert 2023 / Zhao 2025 (in
`METHODOLOGY_LINES.md`) are **not** in this manuscript's bib and are **not**
required for the Discussion — they were methodology-comparison notes, not
Discussion comparators; do not force them in.
