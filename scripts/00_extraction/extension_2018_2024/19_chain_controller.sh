#!/bin/bash
# Phase D → F → G-mini → E auto-chain.
# Polls for Phase D completion, then runs each phase sequentially.
# All outputs land in /depot (persistent). Logs in /tmp.

set -e
cd /home/vmangidi/repositories/WheatPhenologyHRW
EXT=/depot/ciampitti/data/WheatPhenologyHRW/data/raw/satellite/extension_2018_2024
SCRIPTS=scripts/00_extraction/extension_2018_2024

LOG_F=/tmp/feat_ext_gs.log
LOG_G=/tmp/g_mini_anthesis.log
LOG_E=/tmp/full_retrain.log

# ─── Wait for Phase D (training features) ────────────────────────────────────
echo "[chain] $(date) Waiting for Phase D (15_growing_season_pipeline.py) to finish..."
until ! pgrep -f "15_growing_season_pipeline.py" > /dev/null 2>&1; do
    sleep 60
done
echo "[chain] $(date) Phase D ended."

# Verify training features file exists
if [ ! -f "$EXT/features_gs_train_2014_2017.parquet" ]; then
    echo "[chain] ERROR: training features parquet not found after Phase D!"
    exit 1
fi
echo "[chain] Training features file confirmed."

# ─── Phase F: extension features (2019-2024) ────────────────────────────────
echo "[chain] $(date) Starting Phase F: extension features..."
python -u $SCRIPTS/15_growing_season_pipeline.py \
    --valid-path $EXT/valid_field_years_2019_2024.parquet \
    --out $EXT/features_gs_extension_2019_2024.parquet \
    > $LOG_F 2>&1
echo "[chain] $(date) Phase F done."

# ─── Phase G-mini: train anthesis + apply + F7 v2 ────────────────────────────
echo "[chain] $(date) Starting Phase G-mini: anthesis-only train + F7 v2..."
python -u $SCRIPTS/17_train_anthesis_apply_f7.py > $LOG_G 2>&1
echo "[chain] $(date) Phase G-mini done."

# ─── Phase E: full re-train (will run until killed by node shutdown) ─────────
echo "[chain] $(date) Starting Phase E: full re-train (checkpointed)..."
python -u $SCRIPTS/18_full_retrain_checkpointed.py > $LOG_E 2>&1
echo "[chain] $(date) Phase E ended (or killed)."

echo "[chain] $(date) Chain complete."
