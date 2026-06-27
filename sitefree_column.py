"""Site-free brain-age column: pooled multi-site MAE with per-fold LEACE erasure.

Reports, on identical 10-fold splits, the pooled brain-age MAE *before* and
*after* erasing the site label from the features (``leace.LeaceEraser``). The gap
between the two is the size of the pooled-site shortcut the naive pooled number
was quietly exploiting; the site-free number is the honest cross-cohort skill.

The eraser is refit inside every fold on the training rows only
(``leace.erase_per_fold``), so the site-free column carries no leakage.

Feature commensurability: NEOBA fooof features are flattened per-channel, so two
cohorts only pool if their features were extracted on the *same* channel set.
LEMON is 61 ch (1037-dim), TDBRAIN 26 ch (442-dim) -> they must both be
re-extracted on the common-channel intersection before they pool here. This
script asserts equal dimensionality and tells you to re-extract if not.

Run (single site is a no-op sanity check):
    .venv/bin/python sitefree_column.py LEMON
    .venv/bin/python sitefree_column.py LEMON TDBRAIN   # once TDBRAIN extracted
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import h5io
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score

import neoba
from leace import erase_per_fold

CONDITION = "pooled"

# name -> config module (deriv_root + bids_root come from the benchmark config)
SITE_CONFIGS = {
    "LEMON": "config_lemon_eeg",
    "TDBRAIN": "config_tdbrain_eeg",
}


def _load_site(name):
    import importlib
    cfg = importlib.import_module(SITE_CONFIGS[name])
    deriv = str(cfg.deriv_root)
    parts_path = str(cfg.bids_root / "participants.tsv")
    log = pd.read_csv(f"{deriv}/feature_fooof_{CONDITION}-log.csv")
    good = list(log.query('ok == "OK"').subject)
    feat = h5io.read_hdf5(f"{deriv}/features_fooof_{CONDITION}.h5")
    parts = pd.read_csv(parts_path, sep="\t").set_index("participant_id")
    subs = [s for s in good if s in feat and s in parts.index]
    X = np.array([np.asarray(feat[s]["feats"], float) for s in subs])
    y = parts.loc[subs, "age"].astype(float).values
    groups = np.asarray(feat[subs[0]]["groups"])
    return X, y, groups, subs


def main(site_names):
    Xs, ys, sites, groups_ref = [], [], [], None
    for nm in site_names:
        X, y, groups, _ = _load_site(nm)
        if groups_ref is None:
            groups_ref = groups
        elif X.shape[1] != Xs[0].shape[1]:
            raise SystemExit(
                f"feature dim mismatch: {site_names[0]}={Xs[0].shape[1]} vs "
                f"{nm}={X.shape[1]}. Re-extract both on the common-channel "
                f"intersection before pooling (see compute_common_channels.py).")
        Xs.append(X)
        ys.append(y)
        sites.append(np.full(len(y), nm))
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    site = np.concatenate(sites)
    n_sites = len(set(site))
    print(f"pooled N={len(y)} across {n_sites} site(s): "
          f"{dict(zip(*np.unique(site, return_counts=True)))}", flush=True)

    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    rows = []
    for tag, erase in [("pooled (raw)", False), ("site-free (LEACE)", True)]:
        maes, r2s = [], []
        for tr, te in cv.split(X):
            if erase:
                Xtr, Xte = erase_per_fold(X, site, tr, te)
            else:
                Xtr, Xte = X[tr], X[te]
            model = neoba.make_neoba_model(groups_ref, cv=5)
            model.fit(Xtr, y[tr])
            pred = model.predict(Xte)
            maes.append(mean_absolute_error(y[te], pred))
            r2s.append(r2_score(y[te], pred))
        rows.append((tag, np.mean(maes), np.std(maes), np.mean(r2s)))
        print(f"{tag:20s} MAE={np.mean(maes):.3f} +/- {np.std(maes):.3f}  "
              f"R2={np.mean(r2s):.3f}", flush=True)

    if n_sites == 1:
        d = abs(rows[0][1] - rows[1][1])
        print(f"\nsingle-site sanity: |raw - site-free| MAE = {d:.4f} "
              f"(expect ~0; LEACE is a no-op with one site)")
    else:
        print(f"\nshortcut size: site-free MAE - raw MAE = "
              f"{rows[1][1] - rows[0][1]:+.3f} yr "
              f"(positive = pooled number was leaning on site)")


if __name__ == "__main__":
    main(sys.argv[1:] or ["LEMON"])
