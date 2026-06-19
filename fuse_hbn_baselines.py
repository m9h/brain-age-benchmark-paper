"""Unified HBN brain-age comparison: coffeine vs NEOBA vs REVE vs fusion.

All families evaluated on the *same* HBN subjects in the *same* folds
(``KFold(10, shuffle, random_state=42)`` — the splits behind the classical
benchmark result) with a uniform RidgeCV head, so only the feature family
varies. HBN analogue of unify_lemon_baselines.py.

Feature sources (precomputed, aligned by sorted-glob subject order + finite-age):
  * coffeine  -- 7-band filterbank covariances -> Riemann tangent space
                 (features_fb_covs_rest.h5 in the HBN deriv tree).
  * NEOBA     -- clean-room OSF+ODC features (neoba_hbn_feats.npz: 'fixed',
                 'fixed+xspec'); montage-agnostic, so EGI needs no remap.
  * REVE      -- frozen REVE-base layer-6 mean-pooled embeddings, EGI labels
                 remapped to REVE's vocab (reve_hbn_emb.npz).
  * NEOBA+REVE-- z-scored concatenation (does the FM add signal on top?).
"""
from __future__ import annotations

import re
from pathlib import Path

import h5io
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler

import coffeine
import config_hbn_eeg as cfg

NEOBA_NPZ = Path("/mnt/t9/neoba_hbn_feats.npz")
REVE_NPZ = Path("/mnt/t9/reve_hbn_emb.npz")
EPOCHS_GLOB = "/data/derivatives/brain_age/HBN_EEG/sub-*/eeg/*proc-autoreject_epo.fif"
SUBJECT_RE = re.compile(r"sub-([A-Za-z0-9]+)")
# Same 7 bands as compute_benchmark_age_prediction's filterbank-riemann config
# (so they line up with the bundled features_fb_covs_rest.h5).
BANDS = {"low": (0.1, 1), "delta": (1, 4), "theta": (4.0, 8.0),
         "alpha": (8.0, 15.0), "beta_low": (15.0, 26.0),
         "beta_mid": (26.0, 35.0), "beta_high": (35.0, 49)}
ALPHAS = np.logspace(-5, 10, 100)
SEED = 42  # match the classical HBN benchmark + reve_brain_age_hbn splits


def _load_ages():
    df = pd.read_csv(cfg.bids_root / "participants.tsv", sep="\t")
    ids = df["participant_id"].astype(str).str.replace("sub-", "", regex=False)
    return dict(zip(ids, df["age"].astype(float)))


def _ordered_sids(age):
    """Recover the npz row order: sorted glob + finite-age filter."""
    sids = []
    for f in sorted(Path("/").glob(EPOCHS_GLOB.lstrip("/"))):
        m = SUBJECT_RE.search(f.name)
        if m and m.group(1) in age and np.isfinite(age[m.group(1)]):
            sids.append(m.group(1))
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

    featr = h5io.read_hdf5(cfg.deriv_root / "features_fb_covs_rest.h5")
    covs = np.array([featr[f"sub-{s}"]["covs"] for s in sids])
    Xcoff = pd.DataFrame({b: list(covs[:, i]) for i, b in enumerate(BANDS)})
    rank = min(len(cfg.analyze_channels) - 1, covs.shape[-1] - 1)
    fbt = coffeine.make_filter_bank_transformer(
        names=list(BANDS), method="riemann",
        projection_params=dict(scale="auto", n_compo=rank))

    Xneoba = neoba["fixed"]
    Xxspec = neoba["fixed+xspec"]
    Xreve = reve["X"].astype(np.float64)

    # NEOBA OSF/ODC are heavy-tailed on HBN (values up to ~1.9e5 from specparam
    # on some pediatric/EGI recordings); StandardScaler turns those into giant
    # z-scores and folds containing an outlier subject blow up (R^2 SD > 1).
    # A rank->normal QuantileTransformer bounds the tails and restores stable
    # signal (RobustScaler does NOT — IQR-scaling leaves the tails huge). REVE
    # embeddings are well-behaved, so they keep StandardScaler.
    def qt():
        return QuantileTransformer(output_distribution="normal", random_state=0)
    ridge = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS))
    qridge = lambda: make_pipeline(qt(), RidgeCV(alphas=ALPHAS))

    n_neoba = Xneoba.shape[1]
    Xfuse = np.hstack([Xneoba, Xreve])  # raw; transforms applied inside the CV
    fuse_model = make_pipeline(
        ColumnTransformer([("neoba", qt(), list(range(n_neoba))),
                           ("reve", StandardScaler(),
                            list(range(n_neoba, Xfuse.shape[1])))]),
        RidgeCV(alphas=ALPHAS))

    runs = [
        ("coffeine (fb-riemann)", Xcoff,
         make_pipeline(fbt, StandardScaler(), RidgeCV(alphas=ALPHAS))),
        ("NEOBA fixed (OSF+ODC)", Xneoba, qridge()),
        ("NEOBA fixed+xspec", Xxspec, qridge()),
        ("REVE-base L6 (d=512)", Xreve, ridge()),
        ("NEOBA (+) REVE fusion", Xfuse, fuse_model),
    ]

    cv = KFold(n_splits=10, shuffle=True, random_state=SEED)
    print(f"\nHBN n={len(sids)}, KFold(10, seed={SEED}); "
          f"dummy-mean MAE = {np.mean(np.abs(ages - ages.mean())):.2f} yr\n")
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
