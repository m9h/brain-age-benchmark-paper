#!/bin/bash
#SBATCH --job-name=tueg_prep
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=18
#SBATCH --mem=110G
#SBATCH --output=/home/mhough/dev/meeg-brain-age-benchmark-paper/logs/tueg_prep_%j.out
#SBATCH --error=/home/mhough/dev/meeg-brain-age-benchmark-paper/logs/tueg_prep_%j.err
# No --time: gx10-dgx-spark is single-user / infinite walltime.
#
# Full-TUEG (ar-reference, all-comers) coffeine baseline preprocessing chain
# (filterbank-riemann path; no source/FreeSurfer step). Mirrors the LEMON/TUAB
# chain against config_tueg_eeg.py + the /mnt/t9/TUEG-bids tree:
#   1. MNE-BIDS-Pipeline preprocessing  -> proc-clean epochs
#   2. compute_autoreject.py            -> proc-autoreject epochs (+ re-ref)
#   3. compute_features.py (fb_covs)    -> features_fb_covs_pooled.h5

set -euo pipefail
cd /home/mhough/dev/meeg-brain-age-benchmark-paper
PY=.venv/bin/python
# All-comers TUEG has multi-hour recordings (one subject = 4966 10s epochs ≈
# 1.7 GB/epochs-object); 18 workers each holding one OOM-killed the PTP step on
# the first run (110 GB ceiling). Preprocessing (PTP + multitaper PSD report) is
# the memory-heavy step → run it with few workers; autoreject/features are
# lighter per-subject so keep more parallelism. The pipeline is resumable
# (already-written outputs are skipped), so the ~8.2k done subjects are reused.
NJ_PRE=4
NJ=12

echo "[1/3] MNE-BIDS-Pipeline preprocessing"
.venv/bin/mne_bids_pipeline --config config_tueg_eeg.py \
    --steps=preprocessing --n_jobs "$NJ_PRE"

echo "[2/3] AutoReject + average re-reference"
"$PY" compute_autoreject.py -d tueg --n_jobs "$NJ"

echo "[3/3] coffeine filterbank covariance features (fb_covs)"
"$PY" compute_features.py -d tueg -t fb_covs --n_jobs "$NJ"

echo "[done] TUEG preprocessing -> features_fb_covs_pooled.h5"
