import argparse
import os
import pathlib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import mne

from mne_bids import write_raw_bids, print_dir_tree, make_report, BIDSPath

lemon_info = pd.read_csv(
    "./META_File_IDs_Age_Gender_Education_Drug_Smoke_SKID_LEMON.csv")
lemon_info = lemon_info.set_index("ID")
eeg_subjects = pd.read_csv('./lemon_eeg_subjects.csv')
lemon_info = lemon_info.loc[eeg_subjects.subject]
lemon_info['gender'] = lemon_info['Gender_ 1=female_2=male'].map({1: 2, 2: 1})
lemon_info['age_guess'] = np.array(
  lemon_info['Age'].str.split('-').tolist(), dtype=int).mean(1)
subjects = list(lemon_info.index)

def convert_lemon_to_bids(lemon_data_dir, bids_save_dir, n_jobs=1, DEBUG=False):
    """Convert TUAB dataset to BIDS format.

    Parameters
    ----------
    lemon_data_dir : str
        Directory where the original LEMON dataset is saved, e.g.
        `/storage/store3/data/LEMON_RAW`.
    bids_save_dir : str
        Directory where to save the BIDS version of the dataset.
    n_jobs : None | int
        Number of jobs for parallelization.
    """
    subjects_ = subjects
    if DEBUG:
        subjects_ = subjects[:1]

    good_subjects = Parallel(n_jobs=n_jobs)(
        delayed(_convert_subject)(subject, lemon_data_dir, bids_save_dir)
        for subject in subjects_)
    bids_save_dir = pathlib.Path(bids_save_dir)
    subjects_ = [sub for sub in good_subjects if not isinstance(sub, tuple)]
    failed = [sub for sub in good_subjects if isinstance(sub, tuple)]
    if failed:
        _, bad_subjects, errs = zip(*failed)
        pd.DataFrame(dict(subjects=bad_subjects, error=errs)).to_csv(
            bids_save_dir / 'bids_conv_errors.csv')
        print(f"[lemon-bids] {len(failed)} subjects failed conversion")
    # update the participants file as LEMON has no official age data
    # (participant_id in the tsv carries the bare id, no 'sub-' prefix)
    participants = pd.read_csv(
        bids_save_dir / "participants.tsv", sep='\t')
    participants = participants.set_index("participant_id")
    bare_ids = [s.replace("sub-", "") for s in subjects_]
    age_by_bare = {s.replace("sub-", ""): lemon_info.loc[s, 'age_guess']
                   for s in subjects_}
    for bid in bare_ids:
        if bid in participants.index:
            participants.loc[bid, 'age'] = age_by_bare[bid]
    participants.to_csv(
        bids_save_dir / "participants.tsv", sep='\t')


def _convert_subject(subject, data_path, bids_save_dir):
    """Get the work done for one subject"""
    try:
        fname = pathlib.Path(data_path) / subject / "RSEEG" / f"{subject}.vhdr"    
        raw = mne.io.read_raw_brainvision(fname)

        raw.set_channel_types({"VEOG": "eog"})
        montage = mne.channels.make_standard_montage('standard_1005')
        raw.set_montage(montage)
        sub_id = subject.replace("sub-", "")
        # Newer MNE SubjectInfo only accepts a fixed key set (no participant_id
        # / age). sex is int 0/1/2; hand is int 1=right/2=left/3=ambi. Age has
        # no exact value in LEMON (only bins) so it is written to
        # participants.tsv from age_guess after conversion, not here.
        hand_map = {'right': 1, 'left': 2, 'ambidextrous': 3, 'both': 3}
        hand = hand_map.get(str(lemon_info.loc[subject, 'Handedness']).lower())
        subject_info = {
            'his_id': sub_id,
            'sex': int(lemon_info.loc[subject, 'gender']),
        }
        if hand is not None:
            subject_info['hand'] = hand
        raw.info['subject_info'] = subject_info
        # LEMON markers are BrainVision "Stimulus/S200" (eyes open) /
        # "Stimulus/S210" (eyes closed). events_from_annotations assigns its own
        # codes unless we pass an explicit mapping; everything else (S 1,
        # actiCAP comments) is dropped. Strip annotations afterwards so the
        # stricter write_raw_bids uses our events array rather than demanding
        # event_id cover every annotation description.
        custom_mapping = {"Stimulus/S200": 200, "Stimulus/S210": 210}
        events, _ = mne.events_from_annotations(raw, event_id=custom_mapping)
        event_id = {"eyes/open": 200, "eyes/closed": 210}
        raw.set_annotations(None)
        bids_path = BIDSPath(
            subject=sub_id, session=None, task='RSEEG',
            run=None,
            root=bids_save_dir, datatype='eeg', check=True)

        write_raw_bids(
            raw,
            bids_path,
            events=events,
            event_id=event_id,
            overwrite=True,
            allow_preload=False,
        )
    except Exception as err:
        print(err)
        return ("BAD", subject, err)
    return subject


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert LEMON to BIDS.')
    parser.add_argument(
        '--lemon_data_dir', type=str,
        default='/data/datasets/lemon/LEMON_RAW',
        help='Path to the original data.')
    parser.add_argument(
        '--bids_data_dir', type=str,
        default='/data/datasets/lemon/LEMON_EEG_BIDS',
        help='Path to where the converted data should be saved.')
    parser.add_argument(
        '--n_jobs', type=int, default=1,
        help='number of parallel processes to use (default: 1)')
    parser.add_argument(
        '--DEBUG', type=bool, default=False,
        help='activate debugging mode')
    args = parser.parse_args()

    convert_lemon_to_bids(
        args.lemon_data_dir, args.bids_data_dir, n_jobs=args.n_jobs,
        DEBUG=args.DEBUG)

    print_dir_tree(args.bids_data_dir)
    print(make_report(args.bids_data_dir))
