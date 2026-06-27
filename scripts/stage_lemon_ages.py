#!/usr/bin/env python3
"""Fill LEMON ages into the BIDS participants.tsv (the brain-age target).

The LEMON BIDS ``participants.tsv`` ships with an empty ``age`` column. Age lives
in a separate MPI metadata CSV, keyed by the *original* IDs (``sub-032xxx``),
while the BIDS tree uses anonymised IDs (``sub-010xxx``); ``name_match.csv`` maps
between them. Ages are given as 5-year bins ("20-25"); following Sabbagh 2020 /
Engemann 2022 we use the bin midpoint as the regression target.

Pulls both metadata files from the public FCP-INDI S3 mirror (anonymous), writes
the midpoint age back into ``participants.tsv``, and caches the source CSVs next
to the dataset for provenance.
"""

from __future__ import annotations

import io
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config_lemon_eeg as cfg

S3 = "https://fcp-indi.s3.amazonaws.com/data/Projects/INDI/MPI-LEMON"
META = f"{S3}/Behavioural_Data_MPILMBB_LEMON/META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv"
NAME_MATCH = f"{S3}/name_match.csv"


def _fetch_csv(url):
    with urllib.request.urlopen(url) as r:
        return pd.read_csv(io.BytesIO(r.read()))


def _bin_midpoint(s):
    """'20-25' -> 22.5; tolerate stray whitespace and single values."""
    s = str(s).strip()
    if "-" in s:
        lo, hi = s.split("-")
        return (float(lo) + float(hi)) / 2.0
    return float(s)


def main():
    bids_root = cfg.bids_root
    meta = _fetch_csv(META)
    name_match = _fetch_csv(NAME_MATCH)  # Initial_ID (BIDS) <-> INDI_ID (meta)

    # meta is keyed by INDI_ID in its first (unnamed) column.
    meta = meta.rename(columns={meta.columns[0]: "INDI_ID"})
    indi_to_age = dict(zip(meta["INDI_ID"], meta["Age"].map(_bin_midpoint)))
    bids_to_indi = dict(zip(name_match["Initial_ID"], name_match["INDI_ID"]))

    parts = pd.read_csv(bids_root / "participants.tsv", sep="\t")
    parts["age"] = parts["participant_id"].map(
        lambda pid: indi_to_age.get(bids_to_indi.get(pid), np.nan))
    n_ok = parts["age"].notna().sum()
    parts.to_csv(bids_root / "participants.tsv", sep="\t", index=False)

    # Cache sources for provenance.
    meta.to_csv(bids_root / "LEMON_meta_source.csv", index=False)
    name_match.to_csv(bids_root / "LEMON_name_match.csv", index=False)
    print(f"wrote age for {n_ok}/{len(parts)} subjects -> "
          f"{bids_root / 'participants.tsv'}")
    print(parts[["participant_id", "age"]].dropna().head().to_string(index=False))


if __name__ == "__main__":
    main()
