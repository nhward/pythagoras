"""Deterministic Shiny scenario for data-distribution browser tests."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[2] / "app"
os.chdir(APP_ROOT)
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from cards.var_transform import instance
from proxy_data import proxy_data
from roles import Role, RoleMap

rng = np.random.default_rng(2025)
frame = pd.DataFrame({
    "approximately_normal": rng.normal(10, 3, 120),
    "right_skewed": rng.exponential(2, 120),
    "different_scale": rng.normal(1000, 150, 120),
    "outcome": rng.exponential(5, 120),
    "category": pd.Categorical(np.tile(["A", "B", "C"], 40)),
})
roles = RoleMap()
for column in frame.columns:
    roles.set_roles(
        column,
        [Role.TARGET if column == "outcome" else Role.PREDICTOR],
    )

this = instance()
this._imports.set(proxy_data(_df=frame, _roles=roles, _name="Distribution test"))
app = this.application()
