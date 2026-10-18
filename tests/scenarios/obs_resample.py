"""Restored resampling plan, compact rendering, role changes and invalidation."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "app"
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from cards.obs_resample import instance
from evaluation_plan import build_plan
from proxy_data import proxy_data
from roles import RoleMap
from shiny import reactive, render, ui

frame = pd.DataFrame({"value": np.arange(60, dtype=float), "time": np.repeat(np.arange(20), 3),
    "entity": np.tile(["A", "B", "C"], 20), "site": pd.Categorical(np.tile(["North", "South", "West"], 20)), "class": pd.Categorical(np.tile(["yes", "no"], 30))})
source = proxy_data(_df=frame, _roles=RoleMap.from_primitive({"predictor": ["value"],
    "sequence": ["time"], "identifier": ["entity"], "stratifier": ["site"], "target": ["class"]}))
this = instance()
plan = build_plan(source, goal="future", window=5, horizon=2, step=2, gap=1)
this.restore_configuration_state({"inputs": {"Goal": "future", "Window": 5, "Horizon": 2, "Step": 2, "Gap": 1}, "evaluation_plan": plan.bookmark()})
this._imports.set(source)
original_server, original_footer = this.server, this._footer
this.footer = lambda: ui.TagList(original_footer(), ui.output_text("ExportState"), ui.output_text("ExportHistory"),
    ui.input_action_button("ChangeSource", label="Change source"), ui.input_action_button("Empty", label="Empty source"))
def server(input, output, session):
    exported = original_server(input, output, session)
    history = reactive.Value(())

    @reactive.effect
    def observe_exports():
        value = this.output_data()
        with reactive.isolate():
            history.set((*history(), value.evaluation_plan is not None))

    @output
    @render.text
    def ExportHistory():
        return str(history())

    @output
    @render.text
    def ExportState():
        value = this.output_data()
        p = value.evaluation_plan
        return f"plan={p is not None};ready={p.ready if p else False};rows={len(value.frame)}"
    @reactive.effect
    @reactive.event(input.ChangeSource)
    def change():
        changed = source.clean_frame.copy()
        changed.iloc[0, 0] = -1
        this._imports.set(proxy_data(_df=changed, _roles=source.role_map))
    @reactive.effect
    @reactive.event(input.Empty)
    def empty():
        this._imports.set(proxy_data(_df=frame.iloc[:0], _roles=source.role_map))
    return exported
this.server = server
app = this.application()
