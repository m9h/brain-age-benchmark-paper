#!/bin/bash
#SBATCH --job-name=tueg_bids
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=10
#SBATCH --mem=56G
#SBATCH --output=/home/mhough/dev/meeg-brain-age-benchmark-paper/logs/tueg_bids_%j.out
#SBATCH --error=/home/mhough/dev/meeg-brain-age-benchmark-paper/logs/tueg_bids_%j.err
# No --time: gx10-dgx-spark is single-user / infinite walltime.
#
# Convert the ar-reference TUEG brain-age cohort (one resting recording per
# subject, ~9.2k subjects) to BIDS on local SSD. EDF reads from /data NFS run
# in parallel; BIDS writes to /mnt/t9 stay serial (participants.tsv-race-safe).

set -euo pipefail
cd /home/mhough/dev/meeg-brain-age-benchmark-paper
.venv/bin/python convert_tueg_to_bids.py \
    --cohort-csv /mnt/t9/tueg_cohort_ar.csv \
    --bids-root  /mnt/t9/TUEG-bids \
    --n-jobs 10

echo "[done] TUEG ar-cohort BIDS -> /mnt/t9/TUEG-bids"
