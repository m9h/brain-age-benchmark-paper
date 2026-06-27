"""Unified LEMON brain-age comparison: coffeine vs NEOBA vs REVE on one CV.

All three feature families are evaluated on the *same* 120 LEMON subjects in the
*same* fold assignment (``KFold(10, shuffle, random_state=0)``) with a *uniform*
RidgeCV head, so the only thing that varies between rows is the feature family.

Feature sources (all precomputed, aligned by subject id):
  * coffeine  -- 7-band filterbank covariances -> Riemann tangent space
                 (``features_fb_covs_pooled.h5`` in the benchmark deriv tree).
  * NEOBA     -- our clean-room OSF+ODC features (``neoba_lemon_feats_pcc.npz``
                 'fixed' and 'fixed+xspec').
  * REVE      -- frozen REVE-base layer-6 mean-pooled embeddings
                 (``reve_lemon_emb.npz``).
  * NEOBA+REVE-- z-scored concatenation, to test whether the FM adds signal on
                 top of the classical oscillatory features.

Row order of the two npz caches is ``sorted(glob)`` over the proc-autoreject
epochs with a finite-age filter; we recover that exact ordered subject list, index
the coffeine h5 by it, and assert the per-method age vectors match element-wise
before trusting the alignment.
"""

from __future__ import annotations

import re
from pathlib import Path

import h5io
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import coffeine
import config_lemon_eeg as cfg

NEOBA_NPZ = Path("/mnt/t9/lemon_epo/neoba_lemon_feats_pcc.npz")
REVE_NPZ = Path("/mnt/t9/lemon_epo/reve_lemon_emb.npz")
EPOCHS_GLOB = "/data/derivatives/brain_age/LEMON_EEG/sub-*/eeg/*proc-autoreject_epo.fif"
SUBJECT_RE = re.compile(r"sub-([A-Za-z0-9]+)")
BANDS = {"low": (0.1, 1), "delta": (1, 4), "theta": (4.0, 8.0),
         "alpha": (8.0, 15.0), "beta_low": (15.0, 26.0),
         "beta_mid": (26.0, 35.0), "beta_high": (35.0, 49)}
ALPHAS = np.logspace(-5, 10, 100)


def _load_ages():
    df = pd.read_csv(cfg.bids_root / "participants.tsv", sep="\t")
    ids = df["participant_id"].astype(str).str.replace("sub-", "", regex=False)
    return dict(zip(ids, df["age"].astype(float)))


def _ordered_sids(age):
    """Recover the npz row order: sorted glob + finite-age filter."""
    sids = []
    for f in sorted(Path("/").glob(EPOCHS_GLOB.lstrip("/"))):
        m = SUBJECT_RE.search(f.name)
        if not m:
            continue
        sid = m.group(1)
        if sid in age and np.isfinite(age[sid]):
            sids.append(sid)
    return sids


def main():
    age = _load_ages()
    sids = _ordered_sids(age)
    ages = np.array([age[s] for s in sids], dtype=float)

    neoba = np.load(NEOBA_NPZ)
    reve = np.load(REVE_NPZ)

    # Alignment guard: per-method age vectors must match the recovered order.
    assert np.allclose(ages, neoba["ages"]), "NEOBA age order mismatch"
    assert np.allclose(ages, reve["ages"]), "REVE age order mismatch"
    print(f"aligned {len(sids)} subjects; age vectors match across all sources")

    # coffeine filterbank covariances, indexed by the same ordered subject list.
    featr = h5io.read_hdf5(cfg.deriv_root / "features_fb_covs_pooled.h5")
    covs = np.array([featr[f"sub-{s}"]["covs"] for s in sids])
    Xcoff = pd.DataFrame({b: list(covs[:, i]) for i, b in enumerate(BANDS)})
    rank = len(cfg.analyze_channels) - 1
    fbt = coffeine.make_filter_bank_transformer(
        names=list(BANDS), method="riemann",
        projection_params=dict(scale="auto", n_compo=rank))

    Xneoba = neoba["fixed"]
    Xxspec = neoba["fixed+xspec"]
    Xreve = reve["X"].astype(np.float64)
    Xfuse = np.hstack([
        StandardScaler().fit_transform(Xneoba),
        StandardScaler().fit_transform(Xreve),
    ])

    ridge = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
    coff_model = make_pipeline(fbt, StandardScaler(), RidgeCV(alphas=ALPHAS))

    runs = [
        ("coffeine (fb-riemann)", Xcoff, coff_model),
        ("NEOBA fixed (OSF+ODC)", Xneoba, ridge()),
        ("NEOBA fixed+xspec", Xxspec, ridge()),
        ("REVE-base L6 (d=512)", Xreve, ridge()),
        ("NEOBA (+) REVE fusion", Xfuse, ridge()),
    ]

    cv = KFold(n_splits=10, shuffle=True, random_state=0)
    print(f"\ndummy-mean MAE = {np.mean(np.abs(ages - ages.mean())):.2f} yr\n")
    print(f"{'feature family':<24} {'n_feat':>7} {'MAE (yr)':>16} {'R^2':>16}")
    print("-" * 68)
    for name, X, model in runs:
        sc = cross_validate(model, X, ages, cv=cv,
                            scoring=("neg_mean_absolute_error", "r2"), n_jobs=10)
        mae = -sc["test_neg_mean_absolute_error"]
        r2 = sc["test_r2"]
        nf = X.shape[1] if hasattr(X, "shape") else len(X.columns)
        print(f"{name:<24} {nf:>7} "
              f"{mae.mean():>7.2f} +/- {mae.std():<5.2f} "
              f"{r2.mean():>7.3f} +/- {r2.std():<5.3f}")


if __name__ == "__main__":
    main()
