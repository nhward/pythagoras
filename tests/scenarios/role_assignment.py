"""Test-owned Shiny application for role-assignment browser tests."""

import os
import sys
from pathlib import Path

import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[2] / "app"
os.chdir(APP_ROOT)
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from cards.role_assignment import instance
from proxy_data import proxy_data

this = instance()
frame = pd.DataFrame(
    {
        "y": [1, 0, 1, 0],
        "x1": [10.0, 11.0, 12.0, 13.0],
        "x2": ["A", "B", "A", "B"],
        "id": [100, 101, 102, 103],
        "part": ["Train", "Train", "Test", "Test"],
    }
)
this._imports.set(proxy_data(_df=frame, _name="Test"))
app = this.application()
