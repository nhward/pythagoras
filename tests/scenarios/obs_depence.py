"""Longitudinal diagnostic with a pass-through check and an empty source."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "app"
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from cards.obs_dependence import instance
from proxy_data import proxy_data
from roles import RoleMap
from shiny import reactive, render, ui

frame = pd.DataFrame({"entity": np.repeat(["A", "B"], 100), "time": np.tile(np.arange(100), 2),
                      "value": np.tile(np.arange(100, dtype=float), 2)})
frame.loc[[10, 20, 110], "value"] = np.nan
source = proxy_data(_df=frame, _roles=RoleMap.from_primitive({"identifier": ["entity"], "sequence": ["time"], "predictor": ["value"]}))
this = instance()
this.restore_configuration_state({"inputs": {"Lags": 4, "Alpha": .01}})
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
