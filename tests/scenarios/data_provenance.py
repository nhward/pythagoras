"""Seeded Shiny scenario for data-provenance browser tests."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[2] / "app"
os.chdir(APP_ROOT)
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from cards.data_provenance import instance
from cards.var_transform import _analyse_distribution, _apply_analysis
from proxy_data import proxy_data


frame = pd.DataFrame({
    "value": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
    "discard": [1, 1, 1, 1, 1, 1],
})
source = proxy_data(_df=frame, _name="Journey example")
cleaned = source.with_cleaned_data(
    frame.drop(columns="discard"),
    card="var_modify",
    operation="Remove unused variable",
    parameters={"variable": "discard"},
)
cleaned = cleaned.with_inactive_step(
    stage="Cleaning",
    card="obs_duplicates",
    operation="Remove exact duplicate observations",
)
analysis = _analyse_distribution(cleaned, ["Scale", "Center"])
prepared = _apply_analysis(cleaned, analysis)

this = instance()
this._imports.set(prepared)
app = this.application()
