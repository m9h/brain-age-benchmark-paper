import numpy as np
import pandas as pd
import h5io
import coffeine
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import make_scorer, mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

import config_lemon_eeg as cfg
import neoba

bids_root = cfg.bids_root
deriv_root = cfg.deriv_root
analyze_channels = cfg.analyze_channels
condition_ = 'pooled'

bands = {"low": (0.1, 1), "delta": (1, 4), "theta": (4.0, 8.0),
         "alpha": (8.0, 15.0), "beta_low": (15.0, 26.0),
         "beta_mid": (26.0, 35.0), "beta_high": (35.0, 49)}

df = pd.read_csv(bids_root / "participants.tsv", sep='\t').set_index('participant_id').sort_index()

cv = KFold(n_splits=10, shuffle=True, random_state=42)
scoring = {'MAE': make_scorer(mean_absolute_error), 'r2': make_scorer(r2_score)}

def common_subjects(feature_label):
    log = pd.read_csv(deriv_root / f'feature_{feature_label}_{condition_}-log.csv')
    good = log.query('ok == "OK"').subject
    return df.loc[good]

# ---- filterbank-riemann ----
dfr = common_subjects('fb_covs')
featr = h5io.read_hdf5(deriv_root / f'features_fb_covs_{condition_}.h5')
covs = np.array([featr[s]['covs'] for s in dfr.index])
Xr = pd.DataFrame({b: list(covs[:, i]) for i, b in enumerate(bands)})
yr = dfr.age.values
rank = len(analyze_channels) - 1
fbt = coffeine.make_filter_bank_transformer(
    names=list(bands), method='riemann',
    projection_params=dict(scale='auto', n_compo=rank))
mr = make_pipeline(fbt, StandardScaler(), RidgeCV(alphas=np.logspace(-5, 10, 100)))

# ---- fooof-sparse-gl (NEOBA) ----
dff = common_subjects('fooof')
featf = h5io.read_hdf5(deriv_root / f'features_fooof_{condition_}.h5')
subs = list(dff.index)
Xf = np.array([np.asarray(featf[s]['feats'], dtype=float) for s in subs])
groups = np.asarray(featf[subs[0]]['groups'])
yf = dff.age.values
mf = neoba.make_neoba_model(groups, cv=5)

print(f"riemann subjects={len(yr)}  neoba subjects={len(yf)}  "
      f"same set={set(dfr.index)==set(dff.index)}", flush=True)

for name, X, y, model in [('filterbank-riemann', Xr, yr, mr),
                          ('fooof-sparse-gl', Xf, yf, mf)]:
    sc = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=10)
    mae, r2 = sc['test_MAE'], sc['test_r2']
    print(f"{name:20s} N={len(y):3d}  MAE={mae.mean():.3f} +/- {mae.std():.3f}  "
          f"R2={r2.mean():.3f}", flush=True)
