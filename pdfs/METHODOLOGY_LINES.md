# Methodology highlights (line-level)

Specific sentences/clauses we adopted from each paper, highlighted in **Boilermaker Gold (#CEB888)** — marker-style, no paragraph bubbles.

---

## Bandaru 2020 — PhenoCrop framework
`bandaru_2020_phenocrop_LINES.pdf`

| WHY we cite | Highlighted phrase | Page |
|---|---|---:|
| 3-component architecture (KFR + RS-phenology + APTT) — we extend it with f(V) | *PhenoCrop framework constitutes three components* | 3 |
| The exact f(T) construction (beta function with Tmin/Topt/Tmax) we use | *built using a beta function with three* | 6 |
| The APTT accumulation principle — we extend with vernalization to APTT-V | *daily photo-thermal time* | 6 |

## Lobert 2023 — DL benchmark we compare against
`lobert_2023_winter_wheat_DL_LINES.pdf`

| WHY we cite | Highlighted phrase | Page |
|---|---|---:|
| Their architecture — contrast with our linear + tree-ensemble approach | *temporal U-Net* | 2 |
| Their headline ±6 day claim — we report ±10 day with ≥94% on 4 critical stages | *absolute error of less than six days* | 2 |

## McMaster & Wilhelm 1997 — GDD Method 2
`mcmaster_wilhelm_1997_gdd_LINES.pdf`

| WHY we cite | Highlighted phrase | Page |
|---|---|---:|
| The Method 1 vs Method 2 distinction — we use Method 2 throughout | *Method 1 accumulates fewer GDD than Method 2* | 1 |
| The Tmax capping rule that defines Method 2 — we cap at Topt per phase | *incorporating an upper threshold* | 1 |

## Porter & Gawith 1999 — wheat cardinal temperatures (source values)
`porter_gawith_1999_LINES.pdf`

| WHY we cite | Highlighted phrase | Page |
|---|---|---:|
| Source of our vernalization Tmin = -1.3 °C value | *Vernalization Tmin* | 3 |
| Source of our vernalization Topt = 4.9 °C value | *Topt 4.9* | 3 |
| Source of our vernalization Tmax = 15.7 °C value | *Tmax 15.7* | 3 |

## Streck 2003 — generalized f(V) = VD^5 / (22.5^5 + VD^5)
`streck_2003_vernalization_LINES.pdf`

| WHY we cite | Highlighted phrase | Page |
|---|---|---:|
| The 22.5-day half-saturation constant in the f(V) sigmoidal — exact value we use | *22.5 (half of the maximum re-* | 2 |
| Vernalization cardinal temperatures (Tmin, Topt, Tmax) we use in our simulator | *1.3, 4.9, and 15.7* | 1 |
| The recipe for accumulating vernalization days — directly implemented in thermal.py | *VD was calculated with Eq. [5]* | 3 |

## Wang & Engel 1998 — phenology model with f(T)·f(V)·f(P)
`wang_engel_1998_LINES.pdf`

| WHY we cite | Highlighted phrase | Page |
|---|---|---:|
| The foundational dDVS/dt = R_max · f(T) · f(V) · f(P) equation | *A wheat phenology model based on the effects of temperature* | 1 |
| Their f(V) (linear three-stage) — Streck 2003 later generalized this to sigmoidal | *vernalization function is given* | 6 |

## Zhao 2025 — Australian SOTA we compare against
`zhao_2025_australian_wheat_LINES.pdf`

| WHY we cite | Highlighted phrase | Page |
|---|---|---:|
| Their target stages — the same set we predict but in U.S. Plains | *main development stages of cereal growth* | 11 |
| Their flag-leaf R²=0.70 vs our R²=0.82 — direct head-to-head | *flag leaf* | 1 |
