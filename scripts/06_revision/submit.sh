#!/bin/bash
# Submit one revision analysis to SLURM.
#   ./submit.sh R03_selection_integrity.py [hours] [cores]
# Logs land in scripts/06_revision/logs/.
set -euo pipefail

SCRIPT="${1:?usage: ./submit.sh <script.py> [hours] [cores]}"
HOURS="${2:-8}"
CORES="${3:-32}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME="$(basename "$SCRIPT" .py)"
mkdir -p "$HERE/logs"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${NAME}
#SBATCH --account=ciampitti
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CORES}
#SBATCH --time=${HOURS}:00:00
#SBATCH --output=${HERE}/logs/${NAME}-%j.out
#SBATCH --error=${HERE}/logs/${NAME}-%j.err

cd "${HERE}"
export OMP_NUM_THREADS=${CORES}
export MKL_NUM_THREADS=${CORES}
srun /depot/ciampitti/apps/envs/wheatphen_deep/bin/python -u "${SCRIPT}"
EOF
