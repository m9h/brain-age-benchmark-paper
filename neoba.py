"""NEOBA-style oscillatory brain-age baseline (Hu, Valdés-Sosa et al. 2025).

A stronger, interpretable *classical* baseline than coffeine's filterbank-Riemann,
on the same 10-fold CV — the bar a frozen EEG-FM (REVE) has to clear. Following
"Lifespan brain age prediction based on multiple EEG oscillatory features and
sparse group lasso" (Front. Aging Neurosci. 2025), each subject is described by
four families of oscillatory features:

1. **aperiodic** — FOOOF offset + exponent per channel (the 1/f background);
2. **periodic** — centre frequency, power, and bandwidth of each channel's
   single dominant Gaussian peak (above the aperiodic fit);
3. **relative power** — per-channel band power (delta/theta/alpha/beta) as a
   fraction of total power;
4. **ratios** — the four NEOBA power-ratio pairs (delta/theta, delta/alpha,
   theta/beta, theta/alpha).

These four families are the *groups* of a **sparse group lasso** (skglm): the
group penalty drops whole uninformative families while the L1 term keeps the
surviving families sparse — NEOBA's interpretability lever. We use the
sparse-group-lasso *linear* model rather than NEOBA's FCNN+LRP head so the
comparison to coffeine stays a clean linear-vs-linear, feature-driven contrast.

The feature extractor here is pure (NumPy arrays in, feature vector + group
labels out) so it is unit-testable and reusable outside the benchmark loop.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

# Canonical bands, per the NEOBA paper: delta, theta, alpha (8-12), beta (12-20).
# No gamma — NEOBA's source MNCS cross-spectra are capped at ~19 Hz.
CANONICAL_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (12.0, 20.0),
}
# The four NEOBA power-ratio pairs (numerator, denominator): delta/theta,
# delta/alpha, theta/beta, theta/alpha.
RATIO_PAIRS = [
    ("delta", "theta"),
    ("delta", "alpha"),
    ("theta", "beta"),
    ("theta", "alpha"),
]
# FOOOF fit range. NEOBA's cross-spectra are limited to ~1.17-19.14 Hz; we round
# to the band edges (delta floor 1 Hz, beta ceiling 20 Hz) so no gamma leaks in.
FOOOF_RANGE = (1.0, 20.0)


def _band_mask(freqs, lo, hi):
    return (freqs >= lo) & (freqs < hi)


def extract_fooof_features(data, sfreq, *, n_per_seg=None, max_n_peaks=6):
    """Compute NEOBA oscillatory features for one recording.

    Parameters
    ----------
    data : ndarray, shape (n_epochs, n_channels, n_samples)
        Preprocessed epoch data for a single subject/condition.
    sfreq : float
        Sampling frequency (Hz).
    n_per_seg : int or None
        Welch segment length; default ``2 * sfreq`` (~0.5 Hz resolution).
    max_n_peaks : int
        FOOOF max peaks per channel.

    Returns
    -------
    feats : ndarray, shape (n_features,)
        Concatenated four-family feature vector for the subject.
    groups : ndarray, shape (n_features,)
        Integer group id per feature (0=aperiodic, 1=periodic, 2=relative power,
        3=ratios) — the group structure for the sparse group lasso.
    names : list[str]
        Human-readable feature names, aligned with ``feats``.
    """
    from fooof import FOOOFGroup
    from scipy.signal import welch

    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 3:
        raise ValueError(f"expected (n_epochs, n_channels, n_samples), got {data.shape}")
    n_ch = data.shape[1]
    nps = int(n_per_seg or 2 * sfreq)
    nps = min(nps, data.shape[-1])

    # Mean PSD across epochs -> one robust spectrum per channel.
    freqs, psd = welch(data, fs=sfreq, nperseg=nps, axis=-1)
    psd = psd.mean(axis=0)  # (n_channels, n_freq)

    # --- FOOOF fit per channel (aperiodic + Gaussian peaks). ---
    fg = FOOOFGroup(
        peak_width_limits=(1.0, 12.0),
        max_n_peaks=max_n_peaks,
        aperiodic_mode="fixed",
        verbose=False,
    )
    fg.fit(freqs, psd, FOOOF_RANGE)
    aper = fg.get_params("aperiodic_params")  # (n_ch, 2): offset, exponent
    peak_params = fg.get_params("peak_params")  # (n_peaks, 4): CF, PW, BW, ch_idx

    band_names = list(CANONICAL_BANDS)

    # --- Family 1: aperiodic (offset, exponent per channel). ---
    ap_feats, ap_names = [], []
    for ch in range(n_ch):
        ap_feats += [aper[ch, 0], aper[ch, 1]]
        ap_names += [f"aperiodic_offset_ch{ch}", f"aperiodic_exponent_ch{ch}"]

    # --- Family 2: periodic — the single dominant Gaussian peak per channel. ---
    # NEOBA describes each channel's periodic activity by the centre frequency,
    # power, and bandwidth of its strongest peak (max PW), not a per-band sum.
    dominant = np.zeros((n_ch, 3))  # (CF, PW, BW); zeros if the channel has none
    if peak_params.size:
        for ch in range(n_ch):
            ch_peaks = peak_params[peak_params[:, 3].astype(int) == ch]
            if ch_peaks.size:
                top = ch_peaks[np.argmax(ch_peaks[:, 1])]  # max power (PW)
                dominant[ch] = top[:3]  # CF, PW, BW
    pe_feats, pe_names = [], []
    for ch in range(n_ch):
        pe_feats += [dominant[ch, 0], dominant[ch, 1], dominant[ch, 2]]
        pe_names += [f"peak_cf_ch{ch}", f"peak_pw_ch{ch}", f"peak_bw_ch{ch}"]

    # --- Family 3: relative band power per channel (from raw PSD). ---
    fit_mask = _band_mask(freqs, *FOOOF_RANGE)
    total = psd[:, fit_mask].sum(axis=1) + 1e-20  # (n_ch,)
    relpow = np.zeros((n_ch, len(band_names)))
    for bi, b in enumerate(band_names):
        m = _band_mask(freqs, *CANONICAL_BANDS[b])
        relpow[:, bi] = psd[:, m].sum(axis=1) / total
    rp_feats, rp_names = [], []
    for ch in range(n_ch):
        for bi, b in enumerate(band_names):
            rp_feats.append(relpow[ch, bi])
            rp_names.append(f"relpow_{b}_ch{ch}")

    # --- Family 4: inter-band relative-power ratios per channel. ---
    bidx = {b: i for i, b in enumerate(band_names)}
    ra_feats, ra_names = [], []
    for ch in range(n_ch):
        for num, den in RATIO_PAIRS:
            r = relpow[ch, bidx[num]] / (relpow[ch, bidx[den]] + 1e-20)
            ra_feats.append(r)
            ra_names.append(f"ratio_{num}_{den}_ch{ch}")

    feats = np.concatenate([ap_feats, pe_feats, rp_feats, ra_feats]).astype(np.float64)
    groups = np.concatenate([
        np.full(len(ap_feats), 0),
        np.full(len(pe_feats), 1),
        np.full(len(rp_feats), 2),
        np.full(len(ra_feats), 3),
    ]).astype(int)
    names = ap_names + pe_names + rp_names + ra_names
    # Guard against FOOOF NaNs (failed fits) so downstream scaling is safe.
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats, groups, names


def _group_sizes(groups):
    """Contiguous group sizes from an integer group-label vector.

    NEOBA features are laid out family-by-family, so each group is a contiguous
    block. skglm's ``grp_converter`` accepts that as a list of block sizes.
    """
    groups = np.asarray(groups)
    if np.any(np.diff(groups) < 0):
        raise ValueError("group labels must be non-decreasing (contiguous families)")
    _, sizes = np.unique(groups, return_counts=True)
    return [int(s) for s in sizes]


class SparseGroupLasso(RegressorMixin, BaseEstimator):
    """Sparse group lasso (Simon et al. 2013) over NEOBA feature families.

    Thin sklearn-compatible wrapper around skglm's group BCD solver. The penalty
    is ``alpha * [ l1_ratio * ||w||_1 + (1 - l1_ratio) * Σ_g √p_g ||w_g||_2 ]``:
    the group term drops whole uninformative families, the L1 term sparsifies the
    survivors. ``l1_ratio=0`` is a pure group lasso, ``l1_ratio=1`` a plain lasso.
    """

    def __init__(self, groups, alpha=0.01, l1_ratio=0.5, max_iter=1000,
                 tol=1e-4, fit_intercept=True):
        self.groups = groups
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        from skglm.utils.data import grp_converter
        from skglm.estimators import GeneralizedLinearEstimator
        from skglm.datafits import QuadraticGroup
        from skglm.penalties import WeightedL1GroupL2
        from skglm.solvers import GroupBCD

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        # Handle the intercept by centering: WeightedL1GroupL2 doesn't account
        # for the solver's appended intercept column (its weights_features would
        # then be one short), and the intercept should not be penalised anyway.
        if self.fit_intercept:
            self._x_mean = X.mean(axis=0)
            self._y_mean = float(y.mean())
        else:
            self._x_mean = np.zeros(X.shape[1])
            self._y_mean = 0.0
        Xc = X - self._x_mean
        yc = y - self._y_mean
        sizes = _group_sizes(self.groups)
        grp_indices, grp_ptr = grp_converter(sizes, X.shape[1])
        n_grp = len(grp_ptr) - 1
        w_grp = (1.0 - self.l1_ratio) * np.sqrt(np.asarray(sizes, dtype=np.float64))
        w_feat = np.full(X.shape[1], self.l1_ratio, dtype=np.float64)
        df = QuadraticGroup(grp_ptr=grp_ptr, grp_indices=grp_indices)
        pen = WeightedL1GroupL2(alpha=self.alpha, weights_groups=w_grp,
                                weights_features=w_feat, grp_ptr=grp_ptr,
                                grp_indices=grp_indices)
        self._est = GeneralizedLinearEstimator(
            datafit=df, penalty=pen,
            solver=GroupBCD(ws_strategy="fixpoint", max_iter=self.max_iter,
                            tol=self.tol, fit_intercept=False),
        )
        self._est.fit(Xc, yc)
        self.coef_ = self._est.coef_
        self.intercept_ = self._y_mean - float(self._x_mean @ self.coef_)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X - self._x_mean) @ self.coef_ + self._y_mean

    def score(self, X, y):
        from sklearn.metrics import r2_score
        return r2_score(y, self.predict(X))


def make_neoba_model(groups, *, alphas=None, l1_ratio=0.5, cv=5, n_jobs=1):
    """NEOBA sparse-group-lasso pipeline with internal alpha CV.

    Returns a scaled, cross-validated estimator ready to drop into the brain-age
    benchmark loop. ``groups`` is the integer family-label vector from
    :func:`extract_fooof_features`. Scaling is essential: the four feature
    families live on wildly different scales (exponents vs. power ratios).
    """
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV

    if alphas is None:
        alphas = np.logspace(-3, 0, 8)
    base = SparseGroupLasso(groups=groups, l1_ratio=l1_ratio)
    search = GridSearchCV(
        base, {"alpha": list(alphas)}, cv=cv,
        scoring="neg_mean_absolute_error", n_jobs=n_jobs,
    )
    return make_pipeline(StandardScaler(), search)
