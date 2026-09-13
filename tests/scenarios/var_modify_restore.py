"""Variable-modification card with proposed and committed bookmark state."""

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
this.restore_configuration_state({
    "inputs": {
        "Formats": "%Y-%m-%d",
        "Alternatives": "All",
        "MaxObs": 4,
    },
    "modifications": {
        "proposed": [
            {
                "source": "count",
                "source_type": "integer",
                "name": "count",
                "data_type": "decimal",
                "order": [],
            },
            {
                "source": "group",
                "source_type": "nominal",
                "name": "cohort",
                "data_type": "nominal",
                "order": [],
            },
        ],
        "committed": [
            {
                "source": "count",
                "source_type": "integer",
                "name": "count",
                "data_type": "decimal",
                "order": [],
            }
        ],
        "source_columns": ["count", "group"],
        "selected_source": "group",
    },
})
frame = pd.DataFrame({
    "count": pd.Series([1, 2, 3, 4], dtype="int64"),
    "group": pd.Series(["A", "B", "A", "B"], dtype="category"),
})
this._imports.set(proxy_data(_df=frame, _name="Restored"))
app = this.application()
