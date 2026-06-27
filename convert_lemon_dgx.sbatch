#!/bin/bash
#SBATCH --job-name=lemon_bids
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/mhough/dev/meeg-brain-age-benchmark-paper/logs/lemon_bids_%j.out
#SBATCH --error=/home/mhough/dev/meeg-brain-age-benchmark-paper/logs/lemon_bids_%j.err
# No --time: gx10-dgx-spark is single-user / infinite walltime.

set -euo pipefail
cd /home/mhough/dev/meeg-brain-age-benchmark-paper
.venv/bin/python convert_lemon_to_bids.py \
    --lemon_data_dir /data/datasets/lemon/LEMON_RAW \
    --bids_data_dir  /data/datasets/lemon/LEMON_EEG_BIDS \
    --n_jobs 8
