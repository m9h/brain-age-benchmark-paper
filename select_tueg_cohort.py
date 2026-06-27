"""Select a one-recording-per-subject brain-age cohort from the TUEG manifest.

TUEG has many sessions/segments per subject; brain-age wants a single resting
recording per subject. Selection policy (deterministic):

    1. Keep rows with a plausible age (``--min-age`` <= age <= ``--max-age``).
    2. Prefer the average-reference montage (``ar``) over linked-ears (``le``)
       — ``ar`` is the most common and matches the TUAB benchmark config.
    3. Take the earliest session, then the first segment (token t000), so the
       chosen recording is the subject's first available EEG.

Emits a cohort CSV (one row per subject) the converter consumes, plus prints
age/sex/montage summary so we can size the BIDS conversion before launching it.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

_MONTAGE_RANK = {"ar": 0, "ar_a": 1, "le": 2}


def _sort_key(row: dict) -> tuple:
    montage = row.get("montage") or ""
    session = row.get("session") or "999"
    segment = row.get("segment") or "999"
    return (_MONTAGE_RANK.get(montage, 9), session, segment)


def select_cohort(manifest_csv: Path, out_csv: Path,
                  min_age: int = 1, max_age: int = 95,
                  refs: tuple[str, ...] | None = None) -> int:
    by_subject: dict[str, list[dict]] = defaultdict(list)
    with open(manifest_csv) as fh:
        for row in csv.DictReader(fh):
            age = row.get("age")
            if not age:
                continue
            try:
                age_i = int(age)
            except ValueError:
                continue
            if not (min_age <= age_i <= max_age):
                continue
            if refs and (row.get("montage") not in refs):
                continue
            row["age"] = age_i
            by_subject[row["subject"]].append(row)

    cohort = []
    for subject, rows in by_subject.items():
        rows.sort(key=_sort_key)
        cohort.append(rows[0])
    cohort.sort(key=lambda r: r["subject"])

    fields = ["subject", "age", "sex", "session", "year", "montage",
              "segment", "path"]
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in cohort:
            writer.writerow({k: row.get(k) for k in fields})

    # Summary.
    ages = [r["age"] for r in cohort]
    sexes = defaultdict(int)
    montages = defaultdict(int)
    for r in cohort:
        sexes[r.get("sex") or "?"] += 1
        montages[r.get("montage") or "?"] += 1
    print(f"[cohort] subjects={len(cohort)} -> {out_csv}")
    if ages:
        ages_sorted = sorted(ages)
        med = ages_sorted[len(ages_sorted) // 2]
        print(f"[cohort] age: min={min(ages)} median={med} max={max(ages)} "
              f"mean={sum(ages)/len(ages):.1f}")
    print(f"[cohort] sex: {dict(sexes)}")
    print(f"[cohort] montage: {dict(montages)}")
    return len(cohort)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest-csv", type=Path,
                   default=Path("/mnt/t9/tueg_manifest.csv"))
    p.add_argument("--out-csv", type=Path,
                   default=Path("/mnt/t9/tueg_cohort.csv"))
    p.add_argument("--min-age", type=int, default=1)
    p.add_argument("--max-age", type=int, default=95)
    p.add_argument("--refs", nargs="+", default=None,
                   help="keep only these montage refs (e.g. --refs ar). "
                        "Default: all. Use 'ar' for TUAB-parity average ref.")
    args = p.parse_args()
    select_cohort(args.manifest_csv, args.out_csv, args.min_age, args.max_age,
                  refs=tuple(args.refs) if args.refs else None)
