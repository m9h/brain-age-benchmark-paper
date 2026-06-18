from pathlib import Path
import mne

study_name = "age-prediction-benchmark"

# DGX Spark paths. HBN BIDS lives on /data (already staged, read-only); the
# many-small-file tree is only read by the pipeline, derivatives go under the
# shared brain_age root.
n_jobs = 8
N_JOBS = 8
bids_root = Path("/data/datasets/hbn-eeg")
deriv_root = Path("/data/derivatives/brain_age/HBN_EEG")
subjects_dir = Path("/data/derivatives/brain_age/freesurfer")

source_info_path_update = {'processing': 'autoreject',
                           'suffix': 'epo'}
inverse_targets = []
noise_cov = 'ad-hoc'

# CMI Healthy Brain Network resting-state: ~5 min eyes-open/closed alternating
# block recorded as one continuous run. We treat it as fixed-length rest
# (task_is_rest) and pool 10 s windows, matching the TUAB/Cam-CAN rest arm.
task = "RestingState"
task_is_rest = True

sessions = []  # one run per subject, no ses- entity in this BIDS tree
data_type = "eeg"
ch_types = ["eeg"]

# 128 scalp electrodes of the EGI GSN-HydroCel-129 net. Cz is the online
# reference (flat until the average re-reference in compute_autoreject) and is
# dropped here; the remaining E1..E128 are the analysis montage.
analyze_channels = [f"E{i}" for i in range(1, 129)]

drop_channels = ["Cz"]

# EGI net — NOT 10-20. compute_autoreject reads this montage from the config
# (falls back to standard_1005 for the 10-20 cohorts).
eeg_template_montage = mne.channels.make_standard_montage("GSN-HydroCel-129")

l_freq = 0.1
h_freq = 49
raw_resample_sfreq = 200  # HBN is 500 Hz native

eeg_reference = []
eog_channels = []
find_breaks = False
n_proj_eog = dict(n_eeg=1)
reject = None
on_rename_missing_events = "warn"

# fixed-length epochs (continuous rest, no events) — match the 10 s window
epochs_tmin = 0
epochs_tmax = 10 - 1 / raw_resample_sfreq
baseline = None
rest_epochs_duration = 10.0
rest_epochs_overlap = 0.0

run_source_estimation = False
use_template_mri = "fsaverage"

# single resting condition (fixed-length rest events are labelled "rest")
conditions = ["rest"]

event_repeated = "drop"
l_trans_bandwidth = "auto"
h_trans_bandwidth = "auto"
random_state = 42
shortest_event = 1
log_level = "info"
mne_log_level = "error"
on_error = "continue"
