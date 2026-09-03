"""Deterministic Shiny scenario for data-homogeneity browser tests."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[2] / "app"
os.chdir(APP_ROOT)
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from cards.data_homogeneity import instance
from proxy_data import proxy_data
from roles import Role, RoleMap

rng = np.random.default_rng(2025)
count = 60
frame = pd.DataFrame({
    "when": pd.date_range("2025-01-01", periods=count, freq="D"),
    "shifted": np.r_[rng.normal(0, 0.2, count // 2), rng.normal(4, 0.2, count // 2)],
    "category": ["A"] * 30 + ["B"] * 30,
    "stable": rng.normal(0, 1, count),
})
roles = RoleMap()
roles.set_roles("when", [Role.SEQUENCE])
for column in ("shifted", "category", "stable"):
    roles.set_roles(column, [Role.PREDICTOR])

this = instance()
this._imports.set(proxy_data(_df=frame, _roles=roles, _name="Homogeneity test"))
app = this.application()
