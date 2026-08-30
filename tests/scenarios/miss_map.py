"""Deterministic Shiny scenario for missingness-map browser tests."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[2] / "app"
os.chdir(APP_ROOT)
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from cards.miss_map import instance
from proxy_data import proxy_data

frame = pd.DataFrame({
    "A": [np.nan, 2, np.nan, 4, 5, 6],
    "B": [np.nan, 2, np.nan, 4, np.nan, 6],
    "C": [np.nan, 2, 3, 4, 5, 6],
    "complete": range(6),
})

this = instance()
this._imports.set(proxy_data(_df=frame, _name="Missingness map test"))
app = this.application()
