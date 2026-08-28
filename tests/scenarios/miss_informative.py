"""Deterministic Shiny scenario for informative-missingness browser tests."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[2] / "app"
os.chdir(APP_ROOT)
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from cards.miss_informative import instance
from module import Module
from proxy_data import proxy_data
from roles import Role, RoleMap

# Keep the browser scenario deterministic and avoid starting joblib workers.
Module.N_JOBS = 1

rows = 40
target = np.tile([0, 1], rows // 2)
rng = np.random.default_rng(17)
frame = pd.DataFrame(
    {
        # Median imputation makes x constant; its shadow retains the signal.
        "x": np.where(target == 1, np.nan, 0.0),
        "noise": rng.normal(size=rows),
        "target": target,
    }
)
roles = RoleMap()
roles.set_roles("x", [Role.PREDICTOR])
roles.set_roles("noise", [Role.PREDICTOR])
roles.set_roles("target", [Role.TARGET])

this = instance()
this._imports.set(proxy_data(_df=frame, _roles=roles, _name="Test"))
app = this.application()
