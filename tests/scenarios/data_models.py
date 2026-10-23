import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'app'
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from cards.data_models import instance
from proxy_data import proxy_data
from roles import RoleMap
from shiny import render, ui

this = instance()
source = proxy_data(pd.DataFrame({'x': [1., np.nan, 3.], 'cat': pd.Categorical(['a', 'b', 'a']), 'y': [1., 2., 3.]}),
                    _roles=RoleMap.from_primitive({'predictor': ['x', 'cat'], 'target': ['y']}))
this._imports.set(source)
original_server, original_footer = this.server, this._footer
this.footer = lambda: ui.TagList(original_footer(), ui.output_text('ExportState'))
def server(input, output, session):
    exported = original_server(input, output, session)
    @output
    @render.text
    def ExportState():
        value = exported()
        return f'unchanged={value.equals(source)};pipeline={value.has_pipeline}'
    return exported
this.server = server
app = this.application()
