"""Site-free brain age via LEACE (LEAst-squares Concept Erasure, Belrose 2023).

Pooling EEG brain-age features across cohorts (LEMON, TDBRAIN, CHBP, ...) turns
*site* into a confound: each cohort has its own amplifier, montage idiosyncrasies
and — crucially — its own age distribution, so a regressor can lower its apparent
MAE by silently reading off "which dataset is this" instead of the brain. LEACE
removes exactly the part of the feature space that a *linear* probe could use to
recover the site label, while perturbing the features as little as possible
(it is the least-squares-optimal such projection, in closed form).

After erasure, no linear classifier can predict site above chance, yet the brain
signal that is *not* linearly site-coded is preserved — so a "site-free" brain-age
column reports skill that cannot be a pooled-site shortcut.

Closed form (Belrose et al., "LEACE: Perfect linear concept erasure in closed
form", NeurIPS 2023). With centred features X (n x d) and centred one-hot site
Z (n x k):

    Sxx = Cov(X),  Sxz = Cov(X, Z)
    W   = Sxx^{-1/2}                          (symmetric whitening)
    M   = W @ Sxz                              (whitened cross-covariance)
    P_M = orthogonal projector onto col(M)
    r(x) = x - W^{-1} @ P_M @ W @ (x - mean_X)

The erase matrix ``E = W^{-1} P_M W`` is fit on *training* rows only and then
applied to train and test alike, so it introduces no label leakage across CV
folds. With a single site Sxz = 0, so E = 0 and the eraser is the identity — i.e.
running the site-free column on one cohort is a no-op, as it should be.

Pure NumPy (no torch / concept-erasure dependency) so it stays inside the
CPU-only brain-age venv and composes with the existing coffeine / NEOBA models.

This is an independent clean-room implementation of the Belrose closed form. The
emeg-fm identity-trap audit erases a *subject* axis with its own LEACE (vendored
in ``fmscope/diagnostics/erasure.py``); the two are kept separate on purpose —
LEACE is a small stable primitive, and coupling this benchmark to emeg-fm's
vendored paper code would be the wrong dependency direction.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


def _onehot(z):
    """Site labels -> centred one-hot, dropping all-constant columns."""
    z = np.asarray(z)
    if z.ndim == 2:
        Z = z.astype(float)
    else:
        cats = np.unique(z)
        Z = (z[:, None] == cats[None, :]).astype(float)
    return Z


def _sym_inv_sqrt(S, tol):
    """(S^{-1/2}, S^{1/2}) via eigh, pseudo-inverting near-null directions."""
    w, V = np.linalg.eigh(S)
    cutoff = tol * max(w.max(), 0.0) if w.size else 0.0
    keep = w > cutoff
    inv_sqrt = np.zeros_like(w)
    sqrt = np.zeros_like(w)
    inv_sqrt[keep] = 1.0 / np.sqrt(w[keep])
    sqrt[keep] = np.sqrt(w[keep])
    W = (V * inv_sqrt) @ V.T
    W_inv = (V * sqrt) @ V.T
    return W, W_inv


class LeaceEraser(BaseEstimator, TransformerMixin):
    """Least-squares linear concept (site) eraser.

    Fit with the concept labels: ``fit(X, site)`` where ``site`` is a length-n
    array of site/cohort ids (or an n x k one-hot matrix). ``transform(X)`` maps
    features into the site-erased subspace. Composes in an sklearn pipeline as
    long as the caller supplies ``site`` to ``fit`` (see ``erase_per_fold`` for a
    leak-free CV helper).

    Parameters
    ----------
    tol : float
        Relative eigenvalue cutoff for the whitening pseudo-inverse and for the
        rank of the cross-covariance projector.
    """

    def __init__(self, tol=1e-6):
        self.tol = tol

    def fit(self, X, site):
        X = np.asarray(X, dtype=float)
        Z = _onehot(site)
        n = len(X)
        self.mean_ = X.mean(0)
        Xc = X - self.mean_
        Zc = Z - Z.mean(0)
        Sxx = (Xc.T @ Xc) / n
        Sxz = (Xc.T @ Zc) / n
        W, W_inv = _sym_inv_sqrt(Sxx, self.tol)
        M = W @ Sxz
        # orthogonal projector onto the column space of M
        U, s, _ = np.linalg.svd(M, full_matrices=False)
        if s.size:
            r = int((s > self.tol * s.max()).sum())
            Ur = U[:, :r]
            P_M = Ur @ Ur.T
        else:
            P_M = np.zeros((X.shape[1], X.shape[1]))
        self.erase_ = W_inv @ P_M @ W
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X - (X - self.mean_) @ self.erase_.T

    def fit_transform(self, X, site=None, **kw):
        return self.fit(X, site).transform(X)


def erase_per_fold(X, site, train_idx, test_idx, tol=1e-6):
    """Fit the eraser on train rows, apply to both — no cross-fold leakage."""
    er = LeaceEraser(tol=tol).fit(X[train_idx], np.asarray(site)[train_idx])
    return er.transform(X[train_idx]), er.transform(X[test_idx])


def _selftest():
    """Synthetic 2-site check: erasure drops site-probe accuracy to chance,
    a single site is a no-op, and signal orthogonal to site survives."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    rng = np.random.default_rng(0)
    n, d = 600, 20
    site = rng.integers(0, 2, n)
    # site-coded shift along a fixed direction + shared brain signal
    u = rng.standard_normal(d)
    u /= np.linalg.norm(u)
    brain = rng.standard_normal((n, d))
    X = brain + 3.0 * site[:, None] * u[None, :]

    def probe_acc(M):
        return cross_val_score(
            LogisticRegression(max_iter=1000), M, site, cv=5).mean()

    acc_raw = probe_acc(X)
    Xe = LeaceEraser().fit_transform(X, site)
    acc_erased = probe_acc(Xe)

    # single-site: eraser must be ~identity
    one = LeaceEraser().fit_transform(X, np.zeros(n, dtype=int))
    noop_err = np.abs(one - X).max()

    # variance along the site direction must be flattened out
    var_along_raw = float(np.var(X @ u))
    var_along_erased = float(np.var(Xe @ u))

    print(f"site-probe accuracy  raw={acc_raw:.3f}  erased={acc_erased:.3f} "
          f"(chance=0.500; below-chance = no usable site signal)")
    print(f"variance along site direction  raw={var_along_raw:.3f}  "
          f"erased={var_along_erased:.3f}")
    print(f"single-site no-op max|dX| = {noop_err:.2e}")
    assert acc_raw > 0.9, acc_raw
    assert acc_erased < 0.58, acc_erased          # site no longer recoverable
    assert var_along_erased < 0.05 * var_along_raw, var_along_erased
    assert noop_err < 1e-8, noop_err
    print("LEACE self-test PASSED")


if __name__ == "__main__":
    _selftest()
