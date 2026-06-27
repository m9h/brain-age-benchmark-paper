#!/bin/bash
#SBATCH --job-name=lemon_prep
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --output=/home/mhough/dev/meeg-brain-age-benchmark-paper/logs/lemon_prep_%j.out
#SBATCH --error=/home/mhough/dev/meeg-brain-age-benchmark-paper/logs/lemon_prep_%j.err
# No --time: gx10-dgx-spark is single-user / infinite walltime.
#
# LEMON coffeine baseline preprocessing chain (filterbank-riemann path, which
# does NOT need the source/FreeSurfer step):
#   1. MNE-BIDS-Pipeline preprocessing  -> proc-clean epochs
#   2. compute_autoreject.py            -> proc-autoreject epochs (+ re-ref)
#   3. compute_features.py (fb_covs)    -> features_fb_covs_pooled.h5

set -euo pipefail
cd /home/mhough/dev/meeg-brain-age-benchmark-paper
PY=.venv/bin/python
NJ=16

echo "[1/3] MNE-BIDS-Pipeline preprocessing"
.venv/bin/mne_bids_pipeline --config config_lemon_eeg.py \
    --steps=preprocessing --n_jobs "$NJ"

echo "[2/3] AutoReject + average re-reference"
"$PY" compute_autoreject.py -d lemon --n_jobs "$NJ"

echo "[3/3] coffeine filterbank covariance features (fb_covs)"
"$PY" compute_features.py -d lemon -t fb_covs --n_jobs "$NJ"

echo "[done] LEMON preprocessing -> features_fb_covs_pooled.h5"
