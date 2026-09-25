"""Self-contained target balance card with an observable export."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'app'
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from cards.targ_balance import instance
from proxy_data import proxy_data
from roles import RoleMap
from shiny import reactive, render, ui

frame = pd.DataFrame({'x': np.arange(16, dtype=float), 'id': np.arange(16),
                      'target': pd.Categorical(['A'] * 12 + ['B'] * 4)})
source = proxy_data(_df=frame, _roles=RoleMap.from_primitive(
    {'predictor': ['x'], 'identifier': ['id'], 'target': ['target']}))
this = instance()
if globals().get('RESTORED'):
    this.restore_configuration_state({'inputs': {'Mode': 'resample', 'CountFraction': 100,
                                                'Up': 'random', 'Down': 'random'}})
this._imports.set(source)
original_server, original_footer = this.server, this._footer
this.footer = lambda: ui.TagList(original_footer(), ui.output_text('ExportProbe'),
                                ui.input_checkbox('NominalTarget', label='Nominal target', value=True))

def server(input, output, session):
    exported = original_server(input, output, session)
    @reactive.effect
    @reactive.event(input.NominalTarget)
    def change_target():
        if input.NominalTarget():
            this._imports.set(source)
        else:
            changed = source.clone()
            changed.frame['target'] = np.arange(len(frame), dtype=float)
            this._imports.set(changed)
    @output
    @render.text
    def ExportProbe():
        result = exported()
        return f'rows={len(result.frame)} weights={"Weights" in result.frame} steps={len(result.pipeline_steps)} unchanged={result is source}'
    return exported

this.server = server
app = this.application()
