"""Test-owned Shiny application for variable-modification browser tests."""

import os
import sys
from pathlib import Path

import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[2] / "app"
os.chdir(APP_ROOT)
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from cards.var_modify import instance
from proxy_data import proxy_data

this = instance()
dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
frame = pd.DataFrame(
    {
        "y32": pd.Series([1, 0, 1, 0], dtype="int32"),
        "y64": pd.Series([1, 0, 1, 0], dtype="int64"),
        "x32": pd.Series([10.0, 11.0, 12.0, 13.0], dtype="float32"),
        "x64": pd.Series([10.0, 11.0, 12.0, 13.0], dtype="float64"),
        "log": [True, False, True, True],
        "cat": pd.Series(["A", "B", "A", "B"], dtype="category"),
        "id": pd.Series([100, 101, 102, 103], dtype="Int64"),
        "part": ["Train", "Train", "Test", "Test"],
        "items": ["House;Car", "TV", "House;TV", None],
        "date_text": pd.Series(dates, dtype="string"),
        "date_DT": pd.to_datetime(dates),
        "date_D": pd.to_datetime(dates).date,
    }
)
this._imports.set(proxy_data(_df=frame, _name="Test"))
app = this.application()
