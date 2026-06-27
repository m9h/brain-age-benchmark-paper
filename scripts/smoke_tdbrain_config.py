"""Smoke-test config_tdbrain_eeg.py on the TD-BRAIN sample.

Validates the substantive preprocessing assumptions (channel drop, montage,
resample, bandpass, fixed-length epoching) end-to-end through FOOOF -> NEOBA on a
handful of sample subjects. Produces no meaningful MAE -- it is a plumbing check.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import mne

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config_tdbrain_eeg as cfg
import neoba

SAMPLE = Path("/mnt/t9/tdbrain/TD-BRAIN-SAMPLE/TD-BRAIN-SAMPLE")


def preprocess(vhdr):
    raw = mne.io.read_raw_brainvision(vhdr, preload=True, verbose="ERROR")
    raw.drop_channels([c for c in cfg.drop_channels if c in raw.ch_names])
    raw.pick(cfg.analyze_channels)
    raw.set_montage(cfg.eeg_template_montage, on_missing="warn", verbose="ERROR")
    raw.filter(cfg.l_freq, cfg.h_freq, verbose="ERROR")
    raw.resample(cfg.raw_resample_sfreq, verbose="ERROR")
    epochs = mne.make_fixed_length_epochs(
        raw, duration=cfg.rest_epochs_duration,
        overlap=cfg.rest_epochs_overlap, preload=True, verbose="ERROR")
    return epochs


def main():
    parts = pd.read_csv(SAMPLE / "participants.tsv", sep="\t")
    # one row per subject: earliest session
    parts = parts.sort_values(["participant_id", "sessID"])
    parts = parts.drop_duplicates("participant_id", keep="first")
    parts = parts[parts.age.apply(lambda v: str(v).replace(".", "").isdigit())]
    parts["age"] = parts.age.astype(float)
    # adults only (NEOBA aperiodic features are adult-validated)
    parts = parts[parts.age >= 18]

    feats_list, ages, ok = [], [], []
    groups_ref = None
    for _, row in parts.iterrows():
        sub = row.participant_id
        vhdr = (SAMPLE / sub / "ses-1" / "eeg" /
                f"{sub}_ses-1_task-{cfg.task}_eeg.vhdr")
        if not vhdr.exists():
            continue
        try:
            epochs = preprocess(vhdr)
            data = epochs.get_data()
            feats, groups, _ = neoba.extract_fooof_features(
                data, cfg.raw_resample_sfreq)
            feats_list.append(feats)
            ages.append(row.age)
            groups_ref = groups
            ok.append((sub, round(row.age, 1), data.shape, feats.shape))
        except Exception as e:  # smoke test: report, don't crash
            print(f"  FAIL {sub}: {type(e).__name__}: {e}")

    print(f"\n=== preprocessed {len(ok)} adult subjects (task={cfg.task}) ===")
    for sub, age, dshp, fshp in ok:
        print(f"  {sub}  age={age:5}  epochs={dshp}  feats={fshp}")

    if len(feats_list) < 3:
        print("\nToo few subjects for a model-fit plumbing check; stopping.")
        return

    X = np.array(feats_list)
    y = np.array(ages)
    print(f"\nX={X.shape}  y={y.shape}  groups bincount={np.bincount(groups_ref)}")
    model = neoba.make_neoba_model(groups_ref, cv=3)
    model.fit(X, y)
    pred = model.predict(X)
    print(f"NEOBA fit OK. in-sample MAE (meaningless, plumbing only) = "
          f"{np.abs(pred - y).mean():.2f}")
    print("\nSMOKE TEST PASSED: TDBRAIN raw -> config preprocess -> FOOOF -> NEOBA")


if __name__ == "__main__":
    main()
