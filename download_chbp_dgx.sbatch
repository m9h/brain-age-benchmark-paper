#!/bin/bash
#SBATCH --job-name=chbp_dl
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/home/mhough/dev/meeg-brain-age-benchmark-paper/logs/chbp_dl_%j.out
#SBATCH --error=/home/mhough/dev/meeg-brain-age-benchmark-paper/logs/chbp_dl_%j.err
# No --time: gx10-dgx-spark is single-user / infinite walltime.

set -euo pipefail

# Auth: export a Synapse Personal Access Token (synapse.org -> Account Settings
# -> Personal Access Tokens) before submitting, e.g.
#   sbatch --export=ALL,SYNAPSE_AUTH_TOKEN="$(cat ~/.synapse_token)" download_chbp_dgx.sbatch
# Never commit the token. synapseclient reads SYNAPSE_AUTH_TOKEN automatically.
: "${SYNAPSE_AUTH_TOKEN:?set SYNAPSE_AUTH_TOKEN (Synapse personal access token)}"

DEST=/data/datasets/chbp/CHBP_RAW
mkdir -p "$DEST"
cd /home/mhough/dev/meeg-brain-age-benchmark-paper

# CHBP project on Synapse: syn22324937 (doi:10.7303/syn22324937)
.venv/bin/synapse get -r syn22324937 --downloadLocation "$DEST"
echo "[chbp] download complete -> $DEST"
