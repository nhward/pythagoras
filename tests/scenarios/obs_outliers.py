"""Outlier diagnostics with an identifier, target, weights and an empty-state action."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "app"
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from cards.obs_outliers import instance
from proxy_data import proxy_data
from roles import Role, RoleMap
from shiny import reactive, render, ui

rng = np.random.default_rng(2025)
x = rng.normal(size=100); z = rng.normal(size=100)
x[-1], z[-1] = 12, 15
frame = pd.DataFrame({"x": x, "z": z, "target": 2*x+rng.normal(size=100),
                      "id": [f"person-{i}" for i in range(100)], "weight": 1.})
roles = RoleMap()
for c in ["x", "z"]: roles.set_roles(c, [Role.PREDICTOR])
for c, role in [("id", Role.IDENTIFIER), ("weight", Role.WEIGHTING), ("target", Role.TARGET)]:
    roles.set_roles(c, [role])
source = proxy_data(_df=frame, _roles=roles)
this = instance()
this._imports.set(source)
original_server, original_footer = this.server, this._footer
this.footer = lambda: ui.TagList(original_footer(), ui.output_text("PassThrough"),
                                ui.input_action_button("EmptySource", label="Use constant predictors"))
def server(input, output, session):
    exported = original_server(input, output, session)
    @output
    @render.text
    def PassThrough():
        return f"unchanged={exported().equals(this.input_data())}"
    @reactive.effect
    @reactive.event(input.EmptySource)
    def empty():
        data = source.clone()
        data.frame["x"] = data.frame["z"] = 1.
        this._imports.set(data)
    return exported
this.server = server
app = this.application()
