#!/usr/bin/env python3
"""Stage the TDBRAIN brain-age dataset from Synapse onto local SSD.

TDBRAIN (Brainclinics, syn25671079) is distributed on Synapse, *not* only via
the brainclinics.com ORCID download. We already hold a working Synapse token
(``~/.synapse_token``), so the data can be pulled non-interactively here — the
only gate is download access on the project ACL, which Brainclinics grants after
the one-time registration at https://brainclinics.com/resources (submit your
Synapse username + sign the data-use agreement). Until that lands, every file
under the project returns HTTP 403 ("lack DOWNLOAD access").

Everything stages on /mnt/t9 (local SSD): the dataset ships as one large zip, and
unzipping its thousands of small per-subject files onto /data NFS would wedge the
mount in D-state for hours — so we download AND unzip on the SSD.

Entities (children of syn25671079):
    syn26253372  TD-BRAIN-DATASET.zip   full dataset
    syn26241849  TD-BRAIN-SAMPLE.zip    small sample (smoke-test the pipeline)
    syn26468893  participants.tsv       age / sex / indication labels
    syn26241847  TD_BRAIN_code.zip      official preprocessing code
"""

from __future__ import annotations

import argparse
import pathlib
import zipfile

ENTITIES = {
    "full": "syn26253372",
    "sample": "syn26241849",
    "participants": "syn26468893",
    "code": "syn26241847",
}
TOKEN_PATH = pathlib.Path.home() / ".synapse_token"


def _login():
    import synapseclient

    token = TOKEN_PATH.read_text().strip()
    syn = synapseclient.Synapse()
    syn.login(authToken=token, silent=True)
    return syn


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="/mnt/t9/tdbrain",
                    help="staging dir on local SSD (default: /mnt/t9/tdbrain)")
    ap.add_argument("--which", nargs="+", default=["participants", "sample"],
                    choices=list(ENTITIES),
                    help="which entities to fetch (default: participants sample)")
    ap.add_argument("--unzip", action="store_true",
                    help="unzip downloaded archives in place (on the SSD)")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    syn = _login()
    print(f"logged in as {syn.getUserProfile()['userName']}")

    for key in args.which:
        sid = ENTITIES[key]
        print(f"-> {key} ({sid}) into {out}")
        ent = syn.get(sid, downloadLocation=str(out))
        path = pathlib.Path(ent.path)
        print(f"   {path} ({path.stat().st_size/1e9:.2f} GB)")
        if args.unzip and path.suffix == ".zip":
            dest = out / path.stem
            dest.mkdir(exist_ok=True)
            print(f"   unzipping -> {dest}")
            with zipfile.ZipFile(path) as zf:
                zf.extractall(dest)
            print("   done")


if __name__ == "__main__":
    main()
