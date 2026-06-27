#!/usr/bin/env python
"""Resumable LEMON raw-EEG download for the DGX Spark.

Adapted from the upstream ``download_data_lemon.py`` (which hard-codes Inria
``/storage/store3`` paths and expects a META csv we don't have). We instead
drive the download from the subject list shipped in the repo
(``lemon_eeg_subjects.csv``) and land the data on ``/data`` (TrueNAS), skipping
any file already present at the correct size so the job is restart-safe.

Layout written (matches what convert_lemon_to_bids.py expects upstream)::

    /data/datasets/lemon/LEMON_RAW/<sub>/RSEEG/<sub>.{eeg,vhdr,vmrk}
"""
from __future__ import annotations

import pathlib
import sys
import urllib.request

import pandas as pd

URL_LEMON = ("https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON"
             "/EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID")
DATA_PATH = pathlib.Path("/data/datasets/lemon/LEMON_RAW")
EXTS = ("vhdr", "vmrk", "eeg")        # small headers first, big .eeg last


def _content_length(url: str) -> int | None:
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            cl = r.headers.get("Content-Length")
            return int(cl) if cl is not None else None
    except Exception:
        return None


def main() -> int:
    subj_csv = pathlib.Path(__file__).with_name("lemon_eeg_subjects.csv")
    subjects = sorted(pd.read_csv(subj_csv)["subject"].astype(str))
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    print(f"[lemon] {len(subjects)} subjects -> {DATA_PATH}", flush=True)

    ok, skipped, failed = 0, 0, []
    for i, sub in enumerate(subjects, 1):
        out_dir = DATA_PATH / sub / "RSEEG"
        out_dir.mkdir(parents=True, exist_ok=True)
        for ext in EXTS:
            url = f"{URL_LEMON}/{sub}/RSEEG/{sub}.{ext}"
            out = out_dir / f"{sub}.{ext}"
            remote = _content_length(url)
            if out.exists() and remote is not None and out.stat().st_size == remote:
                skipped += 1
                continue
            try:
                urllib.request.urlretrieve(url, out)
                ok += 1
            except Exception as err:  # noqa: BLE001 — report and continue
                failed.append((sub, ext, str(err)))
                print(f"[lemon] FAIL {sub}.{ext}: {err}", flush=True)
        if i % 10 == 0 or i == len(subjects):
            print(f"[lemon] {i}/{len(subjects)} subjects "
                  f"(downloaded={ok} skipped={skipped} failed={len(failed)})",
                  flush=True)

    if failed:
        print(f"[lemon] {len(failed)} files failed; rerun to retry", flush=True)
    print(f"[lemon] done: downloaded={ok} skipped={skipped} "
          f"failed={len(failed)}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
