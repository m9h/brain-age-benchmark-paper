#!/bin/bash
#SBATCH --job-name=hbn_prep
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=70G
#SBATCH --output=/home/mhough/dev/meeg-brain-age-benchmark-paper/logs/hbn_prep_%j.out
#SBATCH --error=/home/mhough/dev/meeg-brain-age-benchmark-paper/logs/hbn_prep_%j.err
# No --time: gx10-dgx-spark is single-user / infinite walltime.
#
# CMI Healthy Brain Network (HBN-EEG) coffeine brain-age baseline. 2466 subjects
# with RestingState + age (5.0-21.9 yr), 129-ch EGI GSN-HydroCel net @ 500 Hz.
# Classical filterbank-riemann path (no source/FreeSurfer step):
#   1. MNE-BIDS-Pipeline preprocessing  -> proc-clean epochs (fixed-length rest)
#   2. compute_autoreject.py            -> proc-autoreject epochs (+ avg re-ref)
#   3. compute_features.py (fb_covs)    -> features_fb_covs_rest.h5
#   4. compute_benchmark_age_prediction -> 10-fold MAE (filterbank-riemann)
#
# HBN BIDS is on /data (read-only, already staged); derivatives go to
# /data/derivatives/brain_age/HBN_EEG. Verify /data is mounted first.

set -euo pipefail
cd /home/mhough/dev/meeg-brain-age-benchmark-paper
PY=.venv/bin/python
# Keep across-subject parallelism tied to the slurm allocation so the two can't
# drift apart (oversubscribing cores makes joblib + AutoReject thrash).
NJ=${SLURM_CPUS_PER_TASK:-12}

# Pin BLAS to 1 thread/process: AutoReject otherwise oversubscribes every core
# per subject (~9x on n_jobs=1), so joblib's across-subject parallelism thrashes.
# One core per subject + NJ subjects in parallel is the right throughput trade.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

if [ "$(ls /data | wc -l)" -lt 3 ]; then
    echo "[fatal] /data looks unmounted (run: sudo mount /data)"; exit 1
fi

echo "[1/4] MNE-BIDS-Pipeline preprocessing"
.venv/bin/mne_bids_pipeline --config config_hbn_eeg.py \
    --steps=preprocessing --n_jobs "$NJ"

echo "[2/4] AutoReject + average re-reference"
"$PY" compute_autoreject.py -d hbn --n_jobs "$NJ"

echo "[3/4] coffeine filterbank covariance features (fb_covs)"
"$PY" compute_features.py -d hbn -t fb_covs --n_jobs "$NJ"

echo "[4/4] filterbank-riemann age prediction (10-fold MAE)"
# NB: --n_jobs 1 here (NOT $NJ). The benchmark collects per-subject y_true/y_pred
# via a scorer that appends to a list; with cross_validate n_jobs>1 those appends
# happen in worker processes and are lost in the parent -> np.concatenate([])
# crashes. The 10-fold ridge CV is cheap, so run it serially.
"$PY" compute_benchmark_age_prediction.py -d hbn -b filterbank-riemann \
    --n_jobs 1

echo "[done] HBN baseline -> results/benchmark-filterbank-riemann_dataset-hbn.csv"
