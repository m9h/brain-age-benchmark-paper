from pathlib import Path
import mne

study_name = "age-prediction-benchmark"

# DGX Spark paths. Raw BrainVision lives on the local SSD (NFS-unsafe for the
# many-small-file BIDS tree); derivatives go under the shared brain_age root.
n_jobs = 8
bids_root = Path("/mnt/t9/tdbrain/bids")
deriv_root = Path("/data/derivatives/brain_age/TDBRAIN_EEG")
subjects_dir = Path("/data/derivatives/brain_age/freesurfer")

source_info_path_update = {'processing': 'autoreject',
                           'suffix': 'epo'}
inverse_targets = []
noise_cov = 'ad-hoc'

# restEC = eyes-closed rest, the canonical resting brain-age condition.
# restEO (eyes-open) is the same montage/sfreq; swap `task` for the EO arm.
task = "restEC"

sessions = []  # one row per subject; we keep ses-1 only (see stage_tdbrain_ages)
data_type = "eeg"
ch_types = ["eeg"]

# 26 scalp electrodes (standard 10-20). The 7 trailing channels in the
# BrainVision header (VPVA/VNVB/HPHL/HNHR = EOG, Erbs/OrbOcc/Mass = ECG/EMG) are
# mislabeled as type EEG and must be dropped before analysis.
analyze_channels = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC3', 'FCz', 'FC4',
    'T7', 'C3', 'Cz', 'C4', 'T8', 'CP3', 'CPz', 'CP4',
    'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2']

drop_channels = ['VPVA', 'VNVB', 'HPHL', 'HNHR', 'Erbs', 'OrbOcc', 'Mass']

eeg_template_montage = mne.channels.make_standard_montage("standard_1005")

l_freq = 0.1
h_freq = 49
raw_resample_sfreq = 200  # TDBRAIN is 500 Hz native

eeg_reference = []
eog_channels = ["Fp1"]
find_breaks = False
n_proj_eog = dict(n_eeg=1)
reject = None
on_rename_missing_events = "warn"

# fixed-length epochs (continuous rest, no events) — match LEMON's 10 s window
epochs_tmin = 0
epochs_tmax = 10 - 1 / raw_resample_sfreq
baseline = None
rest_epochs_duration = 10.0
rest_epochs_overlap = 0.0

run_source_estimation = False
use_template_mri = "fsaverage"

# single resting condition per task file (no within-file event split)
conditions = ["rest"]

event_repeated = "drop"
l_trans_bandwidth = "auto"
h_trans_bandwidth = "auto"
random_state = 42
shortest_event = 1
log_level = "info"
mne_log_level = "error"
on_error = "continue"
