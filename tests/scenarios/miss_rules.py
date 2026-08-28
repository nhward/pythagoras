"""Test-owned Shiny application for the missing-rules browser tests."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[2] / "app"
os.chdir(APP_ROOT)
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from cards.miss_rules import instance
from proxy_data import proxy_data

this = instance()
frame = pd.DataFrame({
    "y": [1, 0, 1, 0],
    "x1": [np.nan, np.nan, 12.0, 13.0],
    "x2": [None, None, "A", "B"],
    "id": [np.nan, 101, 102, 103],
    "part": ["Train", "Train", "Test", "Test"],
})
this._imports.set(proxy_data(_df=frame, _name="Test"))
app = this.application()
