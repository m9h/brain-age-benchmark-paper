#!/bin/bash
#SBATCH --job-name=tuheeg_dl
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=/home/mhough/dev/meeg-brain-age-benchmark-paper/logs/tuheeg_dl_%j.out
#SBATCH --error=/home/mhough/dev/meeg-brain-age-benchmark-paper/logs/tuheeg_dl_%j.err
# No --time: gx10-dgx-spark is single-user / infinite walltime.
#
# TUH EEG Corpus (TUEG) v2.0.2 from the ISIP/Picone server.
#   14,987 patients / 70,831 EDF files / ~1.6 TB. Avg file ~23 MB (NOT a
#   small-file corpus, so writing straight to /data NFS is fine).
# Clinical EEG; subject age lives in the EDF headers (PatientID field).
#
# Access is via a shared corpus credential (documented/public, not a personal
# secret). Override with TUH_USER / TUH_PASS env if it ever rotates.

set -euo pipefail

TUH_USER="${TUH_USER:-nedc_tuh_eeg}"
TUH_PASS="${TUH_PASS:-nedc_tuh_eeg}"
URL="https://isip.piconepress.com/projects/nedc/data/tuh_eeg/tuh_eeg/v2.0.2/"
DEST="/data/datasets/tuh_eeg/v2.0.2"
mkdir -p "$DEST"

# Optional bandwidth cap to cede the shared 1 GbE link / NFS to other jobs
# (e.g. FastSurfer). Set TUH_RATE to a wget --limit-rate value (e.g. 8m) to
# throttle; leave unset/empty for unthrottled (the remote server self-limits at
# ~13 MB/s anyway). A controlled resume is: scancel the job, then resubmit with
# TUH_RATE set -- the -c/-N flags make resumption safe and lossless.
RATE_ARG=()
if [[ -n "${TUH_RATE:-}" ]]; then
    RATE_ARG=(--limit-rate="${TUH_RATE}")
    echo "[tuheeg] throttling download to ${TUH_RATE}"
fi

echo "[tuheeg] mirroring ${URL} -> ${DEST}  (~1.6 TB, expect many hours)"

# Single stream (don't parallelize: kinder to Picone's server and to NFS).
# -c/-N: resumable + timestamp-skip so re-running continues where it left off.
# -nH --cut-dirs=6: strip /projects/nedc/data/tuh_eeg/tuh_eeg/v2.0.2/ prefix.
wget -r -np -nH --cut-dirs=6 -e robots=off -nv \
     -R "index.html*" \
     -c -N \
     "${RATE_ARG[@]}" \
     --retry-connrefused --tries=0 --waitretry=15 --read-timeout=60 \
     --user="$TUH_USER" --password="$TUH_PASS" \
     -P "$DEST" \
     "$URL"

echo "[tuheeg] done. EDF count:"
find "$DEST/edf" -name '*.edf' 2>/dev/null | wc -l
du -sh "$DEST" 2>/dev/null
