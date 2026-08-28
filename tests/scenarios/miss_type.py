"""Deterministic Shiny scenario for missingness-type browser tests."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[2] / "app"
os.chdir(APP_ROOT)
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from cards.miss_type import instance
from module import Module
from proxy_data import proxy_data
from roles import Role, RoleMap

Module.N_JOBS = 1

rows = 40
group = np.tile([0, 1], rows // 2)
target = np.arange(rows, dtype=float)
target[group == 1] = np.nan
frame = pd.DataFrame(
    {
        "target": target,
        "group": group,
        "noise": np.random.default_rng(17).normal(size=rows),
    }
)
roles = RoleMap()
for variable in frame.columns:
    roles.set_roles(variable, [Role.PREDICTOR])

this = instance()
this._imports.set(proxy_data(_df=frame, _roles=roles, _name="Test"))
app = this.application()
