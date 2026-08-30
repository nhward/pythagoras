"""Deterministic Shiny scenario for missingness-set browser tests."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[2] / "app"
os.chdir(APP_ROOT)
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from cards.miss_sets import instance
from proxy_data import proxy_data

frame = pd.DataFrame({
    "A": [np.nan, np.nan, np.nan, 4, 5, 6, 7, 8],
    "B": [np.nan, np.nan, 3, np.nan, 5, 6, 7, 8],
    "C": [1, 2, np.nan, 4, np.nan, 6, 7, 8],
    "complete": range(8),
})

this = instance()
this._imports.set(proxy_data(_df=frame, _name="Missingness sets test"))
app = this.application()
