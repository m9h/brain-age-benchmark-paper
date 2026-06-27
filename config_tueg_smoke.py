"""Smoke-test config: full TUEG pipeline on an 8-subject runless BIDS tree."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config_tueg_eeg import *  # noqa: E402,F401,F403

bids_root = Path("/mnt/t9/TUEG-bids-smoke3")
deriv_root = Path("/mnt/t9/TUEG-bids-smoke3/derivatives")

on_error = "abort"
