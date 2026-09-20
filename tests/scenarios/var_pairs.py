"""Mixed-type pairs with facets, restoration and empty input."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "app"
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from cards.var_pairs import instance
from cyclic_pandas import as_cyclic
from proxy_data import proxy_data
from roles import RoleMap
from shiny import reactive, render, ui

rng = np.random.default_rng(23)
n = 80
frame = pd.DataFrame({"amount": rng.normal(size=n), "score": rng.normal(size=n),
    "grade": pd.Categorical(np.tile(["low", "mid", "high", "mid"], 20), categories=["low", "mid", "high"], ordered=True),
    "class": pd.Categorical(np.tile(["yes", "no"], 40)),
    "hour": as_cyclic(np.arange(n) % 24, period=24),
    "site": pd.Categorical(np.tile(["A", "B"], 40))})
source = proxy_data(_df=frame, _roles=RoleMap.from_primitive({"predictor": list(frame.columns[:5]), "stratifier": ["site"]}))
this = instance()
this.restore_configuration_state({"inputs": {"Variables": list(frame.columns[:5]), "Facet": "site", "Layout": "checker"}})
this._imports.set(source)
original_server, original_footer = this.server, this._footer
this.footer = lambda: ui.TagList(original_footer(), ui.output_text("PassThrough"), ui.input_action_button("Empty", label="Empty source"))
def server(input, output, session):
    exported = original_server(input, output, session)
    @output
    @render.text
    def PassThrough():
        return f"unchanged={exported().equals(this.input_data())}"
    @reactive.effect
    @reactive.event(input.Empty)
    def empty():
        this._imports.set(proxy_data(_df=frame.iloc[:0], _roles=source.role_map))
    return exported
this.server = server
app = this.application()
