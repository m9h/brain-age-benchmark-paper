"""Convert the selected TUEG brain-age cohort to BIDS (BrainVision).

Standalone mne -> mne-bids converter (no braindecode): the on-disk corpus is
the *general* TUEG, not TUAB, so the ``braindecode.datasets.TUHAbnormal`` path
in ``convert_tuh_to_bids.py`` does not apply. We instead drive conversion from
the one-recording-per-subject cohort CSV produced by ``select_tueg_cohort.py``.

Channel/reference handling mirrors ``convert_tuh_to_bids.py`` so the resulting
BIDS tree is compatible with ``config_tuab_eeg.py``-style preprocessing:
TUH ``*-REF`` channel names are renamed to 10-05 labels, non-10-05 channels are
dropped, and a ``standard_1005`` montage is set. Age is recovered as a birthday
relative to the recording year (TUEG headers give year + age, not full DOB).

Output goes to a BIDS root on local SSD (``/mnt/t9``) — never /data NFS, where
writing tens of thousands of small BIDS files wedges in D-state.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import re
from pathlib import Path

import mne
import numpy as np
from mne_bids import BIDSPath, write_raw_bids

SEX_TO_MNE = {"n/a": 0, "m": 1, "f": 2, "": 0, None: 0}


def rename_tuh_channels(ch_name: str) -> str:
    """Rename TUH channels to MNE-readable 10-05 labels.

    TUEG names look like ``EEG FP1-REF`` (average ref) or ``EEG FP1-LE``
    (linked ears). Strip the ``EEG `` prefix and either reference suffix, then
    apply TUH's casing rules (``FP`` -> ``Fp``, trailing ``Z`` -> ``z``).
    """
    exclude = ["LOC", "ROC", "EKG1"]
    match = re.findall(r"^(?:EEG\s+)?([A-Z][\w']*)-(?:REF|LE)$", ch_name)
    if len(match) == 1:
        out = match[0].replace("FP", "Fp").replace("Z", "z")
    else:
        out = ch_name
    return ch_name if out in exclude else out


def _read_one(row: dict, montage):
    """Read + prep a single EDF into a montaged Raw (parallel-safe, no write).

    Returns ``(row, raw)`` on success or ``(row, None)`` if unreadable / too
    few channels. The NFS EDF read dominates cost, so this is the part we fan
    out across workers; the BIDS write stays serial in the parent to avoid
    racing on the shared ``participants.tsv``.
    """
    edf_path = row["path"]
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    except Exception as exc:  # noqa: BLE001 - log and skip unreadable EDFs
        print(f"[skip] {row['subject']}: read failed: {exc}", flush=True)
        return row, None

    raw.pick_types(eeg=True, verbose="ERROR")
    raw.rename_channels(rename_tuh_channels)
    keep = np.intersect1d(raw.ch_names, montage.ch_names)
    if len(keep) < 8:
        print(f"[skip] {row['subject']}: only {len(keep)} 10-05 channels",
              flush=True)
        return row, None
    raw.pick_channels(list(keep))
    raw.set_montage(montage, verbose="ERROR")

    age = int(row["age"])
    year = int(row["year"]) if row["year"] else 2010
    birthday = (datetime.date(year - age, 1, 1)
                - datetime.timedelta(weeks=4))
    sex = (row.get("sex") or "").lower()

    raw.info["line_freq"] = 60  # North America
    with raw.info._unlock():
        raw.info["subject_info"] = {
            "his_id": row["subject"],
            "birthday": birthday,
            "sex": SEX_TO_MNE.get(sex, 0),
        }
    return row, raw


def _write_one(row: dict, raw, bids_root: Path) -> bool:
    # One recording per subject -> uniform ses-001, run-01. With
    # task_is_rest=True the pipeline builds fixed-length rest epochs via the
    # *run* path (get_runs_tasks "runs" branch); that path needs run != None,
    # because _get_run_rest_noise_path routes a runless (run=None, task="rest")
    # job to _get_rest_path, which is disabled when task_is_rest=True and
    # returns no file (KeyError in _01_data_quality). A concrete run entity
    # keeps every job on the working run path. Real TUH session/segment stay in
    # the cohort CSV for provenance.
    bids_path = BIDSPath(
        subject=row["subject"], session="001", run="01",
        task="rest", root=bids_root, datatype="eeg", check=True)
    try:
        write_raw_bids(raw, bids_path, overwrite=True, allow_preload=True,
                       format="BrainVision", verbose="ERROR")
    except Exception as exc:  # noqa: BLE001
        print(f"[skip] {row['subject']}: write failed: {exc}", flush=True)
        return False
    return True


def convert_cohort(cohort_csv: Path, bids_root: Path, limit: int | None = None,
                   n_jobs: int = 1):
    montage = mne.channels.make_standard_montage("standard_1005")
    bids_root.mkdir(parents=True, exist_ok=True)
    with open(cohort_csv) as fh:
        rows = list(csv.DictReader(fh))
    if limit:
        rows = rows[:limit]

    ok = 0
    if n_jobs == 1:
        for i, row in enumerate(rows, 1):
            r, raw = _read_one(row, montage)
            if raw is not None and _write_one(r, raw, bids_root):
                ok += 1
            if i % 100 == 0:
                print(f"[convert] {i}/{len(rows)} ({ok} written)", flush=True)
    else:
        # Parallel NFS reads, serial local-SSD writes (TSV-race-safe).
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs, backend="loky",
                           return_as="generator")(
            delayed(_read_one)(row, montage) for row in rows)
        for i, (r, raw) in enumerate(results, 1):
            if raw is not None and _write_one(r, raw, bids_root):
                ok += 1
            if i % 100 == 0:
                print(f"[convert] {i}/{len(rows)} ({ok} written)", flush=True)
    print(f"[convert] done: {ok}/{len(rows)} subjects -> {bids_root}",
          flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort-csv", type=Path,
                   default=Path("/mnt/t9/tueg_cohort.csv"))
    p.add_argument("--bids-root", type=Path,
                   default=Path("/mnt/t9/TUEG-bids"))
    p.add_argument("--limit", type=int, default=None,
                   help="convert only the first N subjects (smoke test)")
    p.add_argument("--n-jobs", type=int, default=1,
                   help="parallel EDF-read workers (writes stay serial)")
    args = p.parse_args()
    convert_cohort(args.cohort_csv, args.bids_root, args.limit, args.n_jobs)
