"""Deterministic Shiny scenario for imputation-card browser tests."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[2] / "app"
os.chdir(APP_ROOT)
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from cards.miss_impute import instance
from proxy_data import proxy_data

frame = pd.DataFrame({
    "height": [1.50, 1.55, np.nan, 1.65, 1.70, 1.75, 1.80, 1.85, 1.90, 1.95],
    "weight": [50, 54, 58, 62, 66, np.nan, 74, 78, 82, 86],
    "group": pd.Categorical(["A", "A", "A", None, "B", "B", "B", "B", "A", "A"]),
})

this = instance()
this._imports.set(proxy_data(_df=frame, _name="Imputation test"))
app = this.application()
