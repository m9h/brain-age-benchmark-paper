#!/bin/bash
#SBATCH --job-name=on_dl
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/home/mhough/dev/meeg-brain-age-benchmark-paper/logs/on_dl_%j.out
#SBATCH --error=/home/mhough/dev/meeg-brain-age-benchmark-paper/logs/on_dl_%j.err
# No --time: gx10-dgx-spark is single-user / infinite walltime.
#
# Generic OpenNeuro / NEMAR dataset downloader. OpenNeuro mirrors every dataset
# on a public (no-sign-request) S3 bucket, so no credentials/DUA are needed.
#
# Usage:
#   sbatch --export=ALL,DS=ds005385 download_openneuro_dgx.sbatch   # Dortmund Vital resting EEG
#   sbatch --export=ALL,DS=ds005555 download_openneuro_dgx.sbatch   # Bitbrain BOAS healthy sleep PSG
#
# Optional: EXCLUDE_DERIV=1 to skip a derivatives/ subtree (raw BIDS only).

set -euo pipefail
: "${DS:?set DS=dsXXXXXX (OpenNeuro accession)}"

AWS=/home/mhough/miniforge3/bin/aws
DEST="/data/datasets/openneuro/${DS}"
SRC="s3://openneuro.org/${DS}"
mkdir -p "$DEST"

echo "[openneuro] syncing ${SRC} -> ${DEST}"
EXTRA=()
if [[ "${EXCLUDE_DERIV:-0}" == "1" ]]; then
    EXTRA+=(--exclude "derivatives/*")
    echo "[openneuro] excluding derivatives/*"
fi

# Public bucket: --no-sign-request, us-east-1.
"$AWS" s3 sync "$SRC" "$DEST" \
    --no-sign-request --region us-east-1 \
    "${EXTRA[@]}"

echo "[openneuro] done. Top-level contents:"
ls -1 "$DEST" | head -40
echo "[openneuro] participants.tsv head (age check):"
head -5 "$DEST/participants.tsv" 2>/dev/null || echo "  (no participants.tsv at root)"
