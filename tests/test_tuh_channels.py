"""Test the TUH channel-rename fix that unblocks the parked A1/A2 montage error.

The TUH TCP montage carries old 10-20 names (A1/A2 earlobe refs, T3-T6) that the
hand-rolled ``rename_tuh_channels`` left untouched: A1/A2 are in the standard_1005
montage but NOT its 2D layout (so ``set_montage`` later crashes), and T3-T6 are
absent from 1005 (which uses T7/T8/P7/P8) so the channel intersect silently drops
them. The fix ports neuralfetch's 10-5 remap (A1->T9, A2->T10, T3->T7, ...).

``rename_tuh_channels`` is pure (uses only ``re``); we stub the heavy module-level
imports (braindecode/mne_bids) so it loads without torchaudio etc.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types

# Stub the heavy top-level imports so the pure function loads in a bare env.
for name in ("braindecode", "braindecode.datasets", "mne_bids"):
    sys.modules.setdefault(name, types.ModuleType(name))
for attr in ("TUHAbnormal",):
    setattr(sys.modules["braindecode.datasets"], attr, object)
for attr in ("write_raw_bids", "print_dir_tree", "make_report", "BIDSPath"):
    setattr(sys.modules["mne_bids"], attr, object)

_SCRIPT = os.path.join(os.path.dirname(__file__), "..", "convert_tuh_to_bids.py")


def _rename():
    spec = importlib.util.spec_from_file_location("_cttb", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.rename_tuh_channels


def test_a1_a2_mapped_into_10_5_layout():
    r = _rename()
    # A1/A2 in standard_1005 montage but not its 2D layout -> the parked crash;
    # remap to close 10-5 layout equivalents not already in the data.
    assert r("A1-REF") == "T9"
    assert r("A2-REF") == "T10"


def test_old_temporal_names_recovered():
    r = _rename()
    # 10-20 temporal names absent from standard_1005 -> were silently dropped.
    assert r("T3-REF") == "T7"
    assert r("T4-REF") == "T8"
    assert r("T5-REF") == "P7"
    assert r("T6-REF") == "P8"
    assert r("T1-REF") == "FT9"
    assert r("T2-REF") == "FT10"


def test_existing_case_rules_preserved():
    r = _rename()
    assert r("FP1-REF") == "Fp1"
    assert r("CZ-REF") == "Cz"
    # an unmapped standard channel passes through unchanged
    assert r("C3-REF") == "C3"
