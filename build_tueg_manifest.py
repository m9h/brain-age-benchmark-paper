"""Build a brain-age manifest for the full TUEG v2.0.2 corpus.

TUEG (the *general* Temple corpus) is not the TUAB abnormal subset, so the
braindecode ``TUHAbnormal`` path in ``convert_tuh_to_bids.py`` does not apply
(no normal/abnormal labels). For brain-age we only need (subject, session,
age, sex, recording date) — all of which TUH stores in the EDF *header*'s
local-patient-identification field, e.g.::

    aaaaakhr F 01-JAN-0000 aaaaakhr Age:55

Reading only the first 256 header bytes (no signal decode) lets us scan all
~70k recordings cheaply and reads-only — important because the corpus lives on
NFS and full decodes / small-file writes there are pathologically slow.

Path layout (v2.0.2)::

    edf/<NNN>/<subject>/s<NNN>_<YYYY>/<NN_tcp_ar|le>/<subject>_s<NNN>_t<NNN>.edf

Output: a CSV manifest (one row per EDF) with the fields the BIDS converter and
the brain-age cohort selection need. Selection of one recording per subject is
left to a downstream step so this stays a pure, cacheable scan.
"""

from __future__ import annotations

import argparse
import csv
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Local patient id field is EDF bytes [8:88); we read a bit more for safety.
_HEADER_BYTES = 256
_AGE_RE = re.compile(r"Age:(\d+)", re.IGNORECASE)
_SESSION_RE = re.compile(r"^s(\d+)_(\d{4})$")  # s001_2010 -> (001, 2010)
_MONTAGE_RE = re.compile(r"^\d+_tcp_(ar|le|ar_a)$")  # 01_tcp_ar -> ar
_TOKEN_RE = re.compile(r"_t(\d+)\.edf$", re.IGNORECASE)


def _parse_header(path: Path) -> dict | None:
    """Read age/sex from the EDF local-patient-id field (header only)."""
    try:
        with open(path, "rb") as fh:
            hdr = fh.read(_HEADER_BYTES)
    except OSError:
        return None
    patient = hdr[8:88].decode("latin-1", errors="replace")
    m = _AGE_RE.search(patient)
    age = int(m.group(1)) if m else None
    # Sex token: second whitespace-separated field is typically 'M'/'F'.
    sex = None
    toks = patient.split()
    for t in toks[1:3]:
        if t.upper() in ("M", "F"):
            sex = t.upper()
            break
    return {"age": age, "sex": sex}


def _parse_path(path: Path, edf_root: Path) -> dict:
    """Derive subject/session/year/montage/segment from the TUEG path."""
    rel = path.relative_to(edf_root)
    parts = rel.parts  # (<NNN>, subject, s<NNN>_<YYYY>, <NN_tcp_xx>, file.edf)
    subject = parts[1] if len(parts) > 1 else None
    session, year = None, None
    if len(parts) > 2:
        sm = _SESSION_RE.match(parts[2])
        if sm:
            session, year = sm.group(1), int(sm.group(2))
    montage = None
    if len(parts) > 3:
        mm = _MONTAGE_RE.match(parts[3])
        montage = mm.group(1) if mm else parts[3]
    tm = _TOKEN_RE.search(path.name)
    segment = tm.group(1) if tm else None
    return {
        "subject": subject,
        "session": session,
        "year": year,
        "montage": montage,
        "segment": segment,
    }


def _scan_one(path: Path, edf_root: Path) -> dict:
    row = {"path": str(path)}
    row.update(_parse_path(path, edf_root))
    hdr = _parse_header(path)
    row.update(hdr or {"age": None, "sex": None})
    return row


def build_manifest(edf_root: Path, out_csv: Path, n_workers: int = 16) -> int:
    edf_files = sorted(edf_root.rglob("*.edf"))
    fields = ["path", "subject", "session", "year", "montage",
              "segment", "age", "sex"]
    n = 0
    with open(out_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = [ex.submit(_scan_one, p, edf_root) for p in edf_files]
            for fut in as_completed(futs):
                writer.writerow(fut.result())
                n += 1
                if n % 5000 == 0:
                    print(f"[manifest] {n}/{len(edf_files)} scanned", flush=True)
    print(f"[manifest] done: {n} recordings -> {out_csv}", flush=True)
    return n


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--edf-root", type=Path,
                   default=Path("/data/datasets/tuh_eeg/v2.0.2/edf"))
    p.add_argument("--out-csv", type=Path,
                   default=Path("/mnt/t9/tueg_manifest.csv"))
    p.add_argument("--n-workers", type=int, default=16)
    args = p.parse_args()
    build_manifest(args.edf_root, args.out_csv, args.n_workers)
