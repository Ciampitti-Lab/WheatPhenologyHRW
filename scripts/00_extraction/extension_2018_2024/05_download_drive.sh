#!/bin/bash
# Bulk-download the 22 GEE-export CSVs from Google Drive into the
# canonical extension data directory. File names come back from Drive
# verbatim (e.g. buffer_l8_timeseries_2018.csv), so no renaming needed.
#
# Usage: bash 05_download_drive.sh
#
# Requires: gdown (pip install gdown)
set -e

OUT=/depot/ciampitti/data/WheatPhenologyHRW/data/raw/satellite/extension_2018_2024
mkdir -p "$OUT"
cd "$OUT"

IDS=(
  10boYOkpEUL-gC0hVJlUIbZEvoWmgSM3f
  11nBru2-OvsgYoPXPmq2IgZPjidRUqHWg
  15Zg9OhLY0Fbs8B2SnhxvgU6xgXgbb78w
  166X1ZPLOzYT7FZFsIwwZJvgro8Z488It
  1C3GXE7pPcSt8DDjcyGjJKKalA9CpwssO
  1HET2u4OtVVZx424ICPuyUJx1W-hzlF0u
  1KyYYo5m3iD36c59pAHlJG8O43H8xDhZz
  1Mct9FLG_knVdn27Ab0xS1R7fhmBkPW-1
  1Mku95v42vADd6NMwvH5AtmNTkoJEtExK
  1OPfuthiJxCSvsRQ9KWPdrxG8zCObztCI
  1OU42uuHvHEHAGofXqoo5uM66EiJqEtmt
  1Orf8QzwsOy96oFRz_8mfl76v3BMYI-1t
  1P1F7Nn-9Qts7sUkYUYCL0XUar8bTbgRK
  1QKsDeOyO3-OY4liHCliP_ubMoKbfuryz
  1RBjKPbrbJzEcWcQZb6DxwstWTKGZv34B
  1S2UgSLaaAKHXn2tcf2xMcdTNUTl8auPz
  1VuMkXcg3QCj-3IMUETTb1gnTzjwEgDHo
  1Zu2fb-09OzdPPTgaqj4JDbzwuPnZ-BBJ
  1jE2o4eZJgs_TdtGaxT8H5A2X3UzoDL8t
  1mE9u72WPjdPzOAaxUfp7COsjorixK_8s
  1qyPc5K45uK6wor_qjwCEkd-GPaBMry6c
  1v_-U48cFV5pTPCXCQip1aA3v_dPwq6Y9
)

echo "Downloading ${#IDS[@]} files into $OUT"

i=0
for id in "${IDS[@]}"; do
  i=$((i+1))
  echo "[$i/${#IDS[@]}] gdown $id"
  gdown --fuzzy "https://drive.google.com/uc?id=$id" || echo "  ! failed: $id"
done

echo
echo "Done. Files in $OUT:"
ls -lh "$OUT"
