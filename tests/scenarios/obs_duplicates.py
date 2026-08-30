"""Deterministic Shiny scenario for observation-duplicate browser tests."""

import os
import sys
from pathlib import Path

import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[2] / "app"
os.chdir(APP_ROOT)
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from cards.obs_duplicates import instance
from proxy_data import proxy_data

frame = pd.DataFrame({
    "A": [1, 1, 1, 1, 9],
    "B": ["x", "x", "x", "y", "z"],
    "C": [10, 10, 11, 12, 99],
})

this = instance()
this._imports.set(proxy_data(_df=frame, _name="Observation duplicates test"))
app = this.application()
