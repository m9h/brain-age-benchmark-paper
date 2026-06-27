"""MNE-BIDS-Pipeline config for the full-TUEG brain-age cohort.

Mirrors ``config_tuab_eeg.py`` (same TUH channel set, custom reference, 10-05
montage, 0.1-49 Hz / 200 Hz / 10 s rest epochs) but points at the local-SSD
BIDS tree built by ``convert_tueg_to_bids.py``. This is the *novel* all-comers
cohort (one resting recording per subject, no normal/abnormal split) — distinct
from the canonical TUAB-healthy arm in ``config_tuab_eeg.py``.
"""

from pathlib import Path
import mne

study_name = "age-prediction-benchmark"

bids_root = Path("/mnt/t9/TUEG-bids")
deriv_root = Path("/mnt/t9/TUEG-bids/derivatives")
subjects_dir = None

source_info_path_update = {"processing": "autoreject", "suffix": "epo"}

# Our converter (convert_tueg_to_bids.py) already strips TUH's "-REF"/"-LE"
# suffixes and writes clean 10-05 labels (Fp1, A1, T3, ...), so the template
# montage must use the plain standard_1005 names — NOT the "-REF"-suffixed
# variant used by config_tuab_eeg.py (whose BIDS keeps the -REF names).
eeg_template_montage = mne.channels.make_standard_montage("standard_1005")

inverse_targets = []

noise_cov = "ad-hoc"
eeg_reference = []  # TUH has a custom reference

analyze_channels = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1",
                    "O2", "F7", "F8", "T3", "T4", "T5", "T6", "A1", "A2",
                    "Fz", "Cz", "Pz"]

task = "rest"
# Rest recordings have no task events: tell MNE-BIDS-Pipeline to build
# fixed-length epochs (rest_epochs_duration/overlap) instead of matching
# annotation-derived condition names. Without this, _07_make_epochs takes the
# task-run branch and crashes on the empty `conditions` list.
task_is_rest = True
conditions = []
sessions = ["001"]

data_type = "eeg"
ch_types = ["eeg"]

l_freq = 0.1
h_freq = 49
raw_resample_sfreq = 200

find_breaks = False
spatial_filter = None
reject = None

on_rename_missing_events = "warn"

epochs_tmin = 0
epochs_tmax = 10 - 1 / raw_resample_sfreq
rest_epochs_duration = 10.0 - 1 / raw_resample_sfreq
rest_epochs_overlap = 0.0
baseline = None

run_source_estimation = False  # no FreeSurfer for the fb_covs path

event_repeated = "drop"
l_trans_bandwidth = "auto"
h_trans_bandwidth = "auto"

random_state = 42
shortest_event = 1

log_level = "info"
mne_log_level = "info"
on_error = "continue"
