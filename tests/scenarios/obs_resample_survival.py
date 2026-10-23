"""Restored survival configuration, including non-default event labels."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'app'
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from cards.obs_resample import instance
from proxy_data import proxy_data
from roles import RoleMap
from shiny import render, ui

this = instance()
this.restore_configuration_state({'inputs': {'Goal': 'observations', 'Task': 'survival',
    'Duration': 'followup', 'Event': 'status', 'EventValue': 'dead', 'CensoredValue': 'alive',
    'TimeUnit': 'months', 'TimeOrigin': 'Diagnosis', 'Folds': 3}})
frame = pd.DataFrame({'x': np.arange(30.), 'followup': np.arange(30.) + 1,
                      'status': pd.Categorical(['dead', 'alive'] * 15)})
this._imports.set(proxy_data(frame, _roles=RoleMap.from_primitive({'predictor': ['x'], 'stratifier': ['status']})))
original_server, original_footer = this.server, this._footer
this.footer = lambda: ui.TagList(original_footer(), ui.output_text('ExportState'))


def server(input, output, session):
    exported = original_server(input, output, session)
    @output
    @render.text
    def ExportState():
        value = this.output_data()
        plan = value.evaluation_plan
        return f'{value.modelling_context.task};{value.modelling_context.time_unit};ready={plan.ready if plan else False}'
    return exported


this.server = server
app = this.application()
