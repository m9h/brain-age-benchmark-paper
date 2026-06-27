#!/bin/bash
#SBATCH --job-name=lemon_dl
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/home/mhough/dev/meeg-brain-age-benchmark-paper/logs/lemon_dl_%j.out
#SBATCH --error=/home/mhough/dev/meeg-brain-age-benchmark-paper/logs/lemon_dl_%j.err
# No --time: gx10-dgx-spark is single-user / infinite walltime.

set -euo pipefail
cd /home/mhough/dev/meeg-brain-age-benchmark-paper
# Plain stdlib + pandas downloader; conda base has pandas, no venv dependency.
/home/mhough/miniforge3/bin/python download_lemon_dgx.py
