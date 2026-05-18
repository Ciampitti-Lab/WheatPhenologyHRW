"""Generate F2_framework — the multi-stage hybrid framework schematic.

Reproducible replacement for the previously hand-drawn diagram. The
climate-trend output was removed when that analysis was dropped
(year-block bootstrap + BH: 0/40 cells significant); outputs are now
per-stage timing and the retrospective phenology atlas only, matching
the Figure 2 caption in sections/methods.tex.

Run:  python scripts/05_visualization/make_framework_figure.py
Writes F2_framework.{pdf,png} into the paper-overleaf figures/ dir.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

_DEFAULT_OUT = Path(__file__).resolve().parents[2] / "docs" / "figures"
_ap = argparse.ArgumentParser(description="Generate the F2 framework schematic.")
_ap.add_argument("--out", type=Path, default=_DEFAULT_OUT,
                 help="output directory for F2_framework.{pdf,png} "
                      "(default: <repo>/docs/figures; pass e.g. a "
                      "paper-overleaf/figures path to update the manuscript)")
OUT = _ap.parse_args().out

C_IN = ["#2f6090", "#a9842f", "#2f6090", "#1f7a6b", "#9e3b3b"]
C_FEAT = "#d9c79f"
C_CENTRE = "#5d4a86"
C_OUT = "#1f1f1f"

inputs = [
    "HLS L30/S30\n30 m surface reflectance",
    "Daymet 1 km\ndaily weather",
    "MODIS land-surface\ntemperature",
    "Per-field sowing dates\n(observed + fallback)",
    "Per-field phenology\nground observations",
]
feats = [
    "HLS phenometrics &\nVI time series",
    "Window stats\n(GDD, frost / heat days)",
    "LST aggregates\n(day & night)",
    "Wang-Engel-Streck\nthermal-time simulator",
    "Per-stage target dates\n(8 stages, DOS)",
]
centre = ("Five ML regressors\n$\\times$ two strategies\nper stage\n\n"
          "LOYO and LOSO\ncross-validation")
outputs = ["Per-stage timing\n(8 stages, field scale)",
           "Retrospective phenology\natlas (HRW belt)"]

fig, ax = plt.subplots(figsize=(11.0, 6.1))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def box(x, y, w, h, text, fc, tc="white", fs=10, bold=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.6,rounding_size=2",
        linewidth=0, facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=tc, fontsize=fs, fontweight="bold" if bold else "normal",
            linespacing=1.25)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14,
        linewidth=1.4, color="#555555", shrinkA=0, shrinkB=0))


ax.text(50, 97, "Multi-stage hybrid framework for winter-wheat phenology",
        ha="center", va="center", fontsize=13, fontweight="bold")

n = 5
ys = [78, 60.5, 43, 25.5, 8]
bw, bh = 21, 13
x_in, x_ft, x_ce, x_ou = 2, 28, 55, 79

for i in range(n):
    box(x_in, ys[i], bw, bh, inputs[i], C_IN[i], fs=9.5)
    box(x_ft, ys[i], bw, bh, feats[i], C_FEAT, tc="#222222", fs=9.5)
    arrow(x_in + bw, ys[i] + bh / 2, x_ft, ys[i] + bh / 2)

cy, ch = 26, 48
box(x_ce, cy, 20, ch, centre, C_CENTRE, fs=10.5, bold=True)
for i in range(n):
    arrow(x_ft + bw, ys[i] + bh / 2, x_ce, cy + ch / 2)

oy = [56, 20]
ow, oh = 19, 17
for i, t in enumerate(outputs):
    box(x_ou, oy[i], ow, oh, t, C_OUT, fs=9.5, bold=True)
    arrow(x_ce + 20, cy + ch / 2, x_ou, oy[i] + oh / 2)

fig.tight_layout(pad=0.4)
OUT.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"F2_framework.{ext}", dpi=200, bbox_inches="tight")
print("wrote", OUT / "F2_framework.{pdf,png}")
