"""Weighting card with dynamic components, zeros and an exported-data probe."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "app"
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from cards.obs_weighting import instance
from proxy_data import proxy_data
from roles import Role
from shiny import reactive, render, ui

this = instance()
frame = pd.DataFrame({"importance": np.arange(20., dtype=float), "balance__weights": np.tile([1., 2.], 10)})
source = proxy_data(_df=frame)
source.role_map.set_roles("importance", [Role.WEIGHTING])
source.role_map.set_roles("balance__weights", [Role.WEIGHTING])
this._imports.set(source)
original_server, original_footer = this.server, this._footer
this.footer = lambda: ui.TagList(original_footer(), ui.output_text("Probe"),
    ui.input_action_button("None", label="No weights"), ui.input_action_button("Single", label="One weight"), ui.input_action_button("Multiple", label="Two weights"))


def server(input, output, session):
    exported = original_server(input, output, session)

    @reactive.effect
    @reactive.event(input["None"])
    def no_weights():
        changed = source.clone()
        for column in frame:
            changed.role_map.set_roles(column, [Role.NONE])
        this._imports.set(changed)

    @reactive.effect
    @reactive.event(input.Single)
    def single():
        changed = source.clone()
        changed.role_map.set_roles("balance__weights", [Role.NONE])
        this._imports.set(changed)

    @reactive.effect
    @reactive.event(input.Multiple)
    def multiple():
        this._imports.set(source.clone())

    @output
    @render.text
    def Probe():
        result = exported()
        return f"last={result.frame.importance.iloc[-1]:.3f} steps={len(result.pipeline_steps)}"

    return exported


this.server = server
app = this.application()
