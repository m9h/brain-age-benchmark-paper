#!/bin/bash
#SBATCH --job-name=nsrr_dl
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/home/mhough/dev/meeg-brain-age-benchmark-paper/logs/nsrr_dl_%j.out
#SBATCH --error=/home/mhough/dev/meeg-brain-age-benchmark-paper/logs/nsrr_dl_%j.err
# No --time: gx10-dgx-spark is single-user / infinite walltime.
#
# Usage (after NSRR account + per-dataset Data Use Agreement is APPROVED):
#   1. Get your download token: sleepdata.org -> account -> "Download token"
#   2. Save it (never commit):  printf '%s' '<token>' > ~/.nsrr_token
#   3. Submit for a dataset/subpath, e.g. just the EDFs + demographics:
#        sbatch --export=ALL,NSRR_TOKEN="$(cat ~/.nsrr_token)" \
#               download_nsrr_dgx.sbatch shhs/polysomnography/edfs
#        sbatch --export=ALL,NSRR_TOKEN="$(cat ~/.nsrr_token)" \
#               download_nsrr_dgx.sbatch shhs/datasets
#
# Pick narrow subpaths: full cohorts are multi-TB. For brain-age you mainly
# need the EDF signals + the demographics/age tables under <dataset>/datasets.

set -euo pipefail
: "${NSRR_TOKEN:?set NSRR_TOKEN (sleepdata.org download token)}"
DATASET_PATH="${1:?usage: sbatch download_nsrr_dgx.sbatch <dataset>[/<subpath>]}"

export PATH="$(ruby -e 'puts Gem.user_dir')/bin:$PATH"
DEST=/data/datasets/nsrr
mkdir -p "$DEST"
cd "$DEST"

echo "[nsrr] downloading ${DATASET_PATH} -> ${DEST}/${DATASET_PATH%%/*}"
nsrr download "$DATASET_PATH" --token="$NSRR_TOKEN"
echo "[nsrr] done: $DATASET_PATH"
