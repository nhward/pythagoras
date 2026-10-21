"""Deterministic Shiny scenario for variable-correlation browser tests."""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[2] / "app"
os.chdir(APP_ROOT)
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from cards import var_correlation
from cards.var_correlation import instance
from shiny import reactive, render, ui

analysis_calls = 0
original_analysis = var_correlation._analyse_correlation

def counted_analysis(*args, **kwargs):
    global analysis_calls
    analysis_calls += 1
    return original_analysis(*args, **kwargs)

var_correlation._analyse_correlation = counted_analysis
from proxy_data import proxy_data
from roles import Role, RoleMap

rows = 80
rng = np.random.default_rng(1729)
x = np.linspace(-2, 2, rows)
frame = pd.DataFrame({
    "x": x,
    "positive": x + rng.normal(scale=0.05, size=rows),
    "negative": -x + rng.normal(scale=0.05, size=rows),
    "square": x**2 + rng.normal(scale=0.02, size=rows),
    "noise": rng.normal(size=rows),
    "target": 3 * x + rng.normal(scale=0.1, size=rows),
    "weight": np.linspace(0.5, 1.5, rows),
    "constant": np.ones(rows),
    "flag": np.resize([True, False], rows),
    "shadow__x": x,
    "label": np.resize(["A", "B"], rows),
})
frame.loc[::13, "positive"] = np.nan

roles = RoleMap()
for column in ("x", "positive", "negative", "square", "noise"):
    roles.set_roles(column, [Role.PREDICTOR])
roles.set_roles("target", [Role.TARGET])
roles.set_roles("weight", [Role.WEIGHTING])
for column in ("constant", "flag", "shadow__x", "label"):
    roles.set_roles(column, [Role.PREDICTOR])

this = instance()
this._imports.set(proxy_data(
    _df=frame,
    _roles=roles,
    _name="Variable correlation test",
))
original_server, original_footer = this.server, this._footer
this.footer = lambda: ui.TagList(original_footer(), ui.output_text('AnalysisCalls'),
                                ui.input_action_button(id='ReplaceData', label='Replace data'))

def server(input, output, session):
    result = original_server(input, output, session)

    @output
    @render.text
    def AnalysisCalls():
        reactive.invalidate_later(0.1)
        return str(analysis_calls)

    @reactive.effect
    @reactive.event(input.ReplaceData, ignore_init=True)
    def replace_data():
        updated = frame.copy()
        updated['x'] = updated['x'] ** 2
        this._imports.set(proxy_data(_df=updated, _roles=roles, _name='Updated data'))
    return result

this.server = server
app = this.application()
