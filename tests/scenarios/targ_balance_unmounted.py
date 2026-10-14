"""Numeric target passes through before any balancing controls are mounted."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'app'
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import pandas as pd
from cards.targ_balance import instance
from proxy_data import proxy_data
from roles import RoleMap
from shiny import reactive, render, ui

this = instance()
source = proxy_data(_df=pd.DataFrame({'x': [1., 2., 3.], 'target': [4., 5., 6.]}),
                    _roles=RoleMap.from_primitive({'predictor': ['x'], 'target': ['target']}))
this._imports.set(source)
# Deliberately omit Modes and all settings: input.Mode is never initialized.
this.settings = lambda: ui.TagList()
this.footer = lambda: ui.TagList(ui.output_text('ExportProbe'),
                                ui.input_action_button('ChangeSource', label='Change source'))
original_server = this.server

def server(input, output, session):
    exported = original_server(input, output, session)

    @output
    @render.text
    def ExportProbe():
        result = exported()
        return f'identity={result is this._imports()};rows={len(result.frame)};steps={len(result.pipeline_steps)}'

    @reactive.effect
    @reactive.event(input.ChangeSource)
    def change():
        this._imports.set(proxy_data(_df=source.frame.iloc[:2], _roles=source.role_map))

    return exported

this.server = server
app = this.application()
