"""Deterministic Shiny scenario for parallel-coordinates browser tests."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[2] / "app"
os.chdir(APP_ROOT)
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from cards.data_parallel import instance
from proxy_data import proxy_data
from roles import Role, RoleMap

count = 18
frame = pd.DataFrame({
    "amount": [np.nan] + np.linspace(10.0, 95.0, count - 1).tolist(),
    "score": np.linspace(-2.0, 2.0, count),
    "group": pd.Categorical(
        ["North", "South", "Central"] * 6,
        categories=["North", "Central", "South"],
        ordered=True,
    ),
    "when": pd.date_range("2025-01-01", periods=count, freq="D"),
    "flag": [True, False] * 9,
    "high_cardinality": [f"id-{index:02d}" for index in range(count)],
})
roles = RoleMap()
for column in frame.columns:
    roles.set_roles(column, [Role.PREDICTOR])
roles.set_roles("high_cardinality", [Role.IDENTIFIER])

this = instance()
this._imports.set(proxy_data(
    _df=frame,
    _roles=roles,
    _name="Parallel coordinates test",
))
app = this.application()
