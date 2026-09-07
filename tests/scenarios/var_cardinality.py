"""Deterministic Shiny scenario for variable-cardinality browser tests."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[2] / "app"
os.chdir(APP_ROOT)
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from cards.var_cardinality import instance
from proxy_data import proxy_data
from roles import Role, RoleMap
from text_pandas import as_text

count = 80
frame = pd.DataFrame({
    "constant_decimal": np.repeat(3.5, count),
    "few_decimal": np.resize([0.1, 0.2, 0.3, 0.4], count),
    "count": np.resize(np.arange(40), count),
    "group": pd.Categorical(np.resize(["North", "South", "East", "West"], count)),
    "many_levels": pd.Categorical(
        [f"level-{index:02d}" for index in range(60)]
        + [f"level-{index:02d}" for index in range(20)]
    ),
    "record_code": pd.Series(
        [f"record-{index:03d}" for index in range(count)],
        dtype="string",
    ),
    "all_missing": np.full(count, np.nan),
    "metadata": [{"source": index % 2} for index in range(count)],
})
frame["free_text"] = as_text([
    f"Observation {index} contains a distinct explanatory sentence."
    for index in range(count)
])

roles = RoleMap()
for column in frame.columns:
    roles.set_roles(column, [Role.PREDICTOR])
roles.set_roles("record_code", [Role.IDENTIFIER])
roles.set_roles("free_text", [Role.SENSITIVE])

this = instance()
this._imports.set(proxy_data(
    _df=frame,
    _roles=roles,
    _name="Variable cardinality test",
))
app = this.application()
