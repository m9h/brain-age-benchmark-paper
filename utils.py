import importlib
from types import SimpleNamespace
import pandas as pd

def prepare_dataset(dataset):
    config_map = {'chbp': "config_chbp_eeg",
                  'lemon': "config_lemon_eeg",
                  'tuab': "config_tuab_eeg",
                  'tueg': "config_tueg_eeg",
                  'tueg-smoke': "config_tueg_smoke",
                  'camcan': "config_camcan_meg",
                  'hbn': "config_hbn_eeg"}
    if dataset not in config_map:
        raise ValueError(
            f"We don't know the dataset '{dataset}' you requested.")

    cfg_in = importlib.import_module(config_map[dataset])
    cfg_out = SimpleNamespace(
        bids_root=cfg_in.bids_root,
        deriv_root=cfg_in.deriv_root,
        task=cfg_in.task,
        analyze_channels=cfg_in.analyze_channels,
        data_type=cfg_in.data_type,
        subjects_dir=cfg_in.subjects_dir
    )
    # Montage for autoreject interpolation: 10-20 cohorts use standard_1005,
    # HBN uses the EGI GSN-HydroCel net (set in its config). Default keeps the
    # existing behaviour for cohorts that don't carry one.
    cfg_out.eeg_template_montage = getattr(
        cfg_in, 'eeg_template_montage', None)
    cfg_out.conditions = {  # use for epoching
        'lemon': ('eyes/closed', 'eyes/open'),
        'chbp': ('eyes/closed', 'eyes/open'),
        'tuab': ('rest',),
        'tueg': ('rest',),
        'tueg-smoke': ('rest',),
        'camcan': ('rest',),
        'hbn': ('rest',)
    }[dataset]
    cfg_out.feature_conditions = {  # use for selecting data for features
        'lemon': ('eyes',),
        'chbp': ('eyes',),
        'tuab': ('rest',),
        'tueg': ('rest',),
        'tueg-smoke': ('rest',),
        'camcan': ('rest',),
        'hbn': ('rest',)
    }[dataset]

    cfg_out.session = ''
    sessions = cfg_in.sessions
    if dataset in ('tuab', 'tueg', 'tueg-smoke', 'camcan'):
        cfg_out.session = 'ses-' + sessions[0]

    subjects_df = pd.read_csv(cfg_out.bids_root / "participants.tsv", sep='\t')
    subjects = sorted(sub for sub in subjects_df.participant_id if
                      (cfg_out.deriv_root / sub / cfg_out.session /
                       cfg_out.data_type).exists())

    return cfg_out, subjects
