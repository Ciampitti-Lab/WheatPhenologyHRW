#!/bin/bash
# Single entry point: rebuild every manuscript figure and every table source
# from the current feature matrix, then verify the manuscript against them.
#
#   ./regenerate_manuscript_artifacts.sh [--figures-only]
#
# Why this exists. Figure 3 of the R1 submission was generated from the
# pre-correction run and survived into the PDF, because the figure code and
# the table code were reached by different commands and only one of them was
# re-run after audit item 14. Anything that produces a number the manuscript
# reports belongs in this file.
#
# Deep stages need a GPU: submit with
#   sbatch --account=ciampitti --partition=smallgpu --gpus-per-node=1 \
#          --cpus-per-task=64 --time=8:00:00 --wrap="./regenerate_manuscript_artifacts.sh"
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
PY=/depot/ciampitti/apps/envs/wheatphen_deep/bin/python
GEO=/depot/ciampitti/apps/envs/vmangidi_ww_protein_prediction/bin/python   # cartopy
PAPER=/home/vmangidi/repositories/paper-overleaf/figures

step () { printf '\n=== %s ===\n' "$1"; }

step 'F3, F4  (adopted model per stage; needs a GPU for the FT stages)'
$PY -m scripts.04_figures.09_paper_figures F3 F4
cp docs/figures/F3_per_stage_scatter.{pdf,png} "$PAPER/"
cp docs/figures/F4_strategy_comparison.{pdf,png} "$PAPER/"

step 'F1  study area (cartopy env; writes straight into the paper)'
$GEO scripts/06_revision/R09_figure1_studyarea.py

step 'F5  grouped permutation importance'
$PY scripts/06_revision/R10_figure5_importance.py

# F2 is a hand-drawn schematic with no computed numbers, and F6 is regenerated
# by 09_paper_figures F6 when the LOSO grid changes; neither is rebuilt here
# by default because neither reads the feature matrix on every run.

if [ "${1:-}" != "--figures-only" ]; then
  step 'verify every reported number, and every figure panel, against source'
  $PY scripts/06_revision/R12_verify_manuscript.py
fi
