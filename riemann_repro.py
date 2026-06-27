import numpy as np, pandas as pd, h5io, coffeine
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
import config_lemon_eeg as cfg
bands = {"low":(0.1,1),"delta":(1,4),"theta":(4.,8.),"alpha":(8.,15.),
         "beta_low":(15.,26.),"beta_mid":(26.,35.),"beta_high":(35.,49)}
deriv=cfg.deriv_root
df=pd.read_csv(cfg.bids_root/"participants.tsv",sep='\t').set_index('participant_id').sort_index()
log=pd.read_csv(deriv/'feature_fb_covs_pooled-log.csv'); good=log.query('ok=="OK"').subject
df=df.loc[good]
feat=h5io.read_hdf5(deriv/'features_fb_covs_pooled.h5')
covs=np.array([feat[s]['covs'] for s in df.index])
X=pd.DataFrame({b:list(covs[:,i]) for i,b in enumerate(bands)})
y=df.age.values
rank=len(cfg.analyze_channels)-1
fbt=coffeine.make_filter_bank_transformer(names=list(bands),method='riemann',
     projection_params=dict(scale='auto',n_compo=rank))
m=make_pipeline(fbt,StandardScaler(),RidgeCV(alphas=np.logspace(-5,10,100)))
m.fit(X.iloc[:100],y[:100])
p=m.predict(X.iloc[100:])
print("predict OK, mae", np.abs(p-y[100:]).mean())
