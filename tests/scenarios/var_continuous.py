"""Continuous predictor curves with more than the default 15 variables."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'app'
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from cards.var_continuous import instance
from proxy_data import proxy_data
from roles import RoleMap
from shiny import render, ui

frame = pd.DataFrame({
    'gapped': np.r_[np.linspace(0, 1, 100), np.linspace(3, 4, 100)],
    'repeated': np.tile([0., 1.], 100),
    **{f'value_{i}': np.linspace(0, i + 1, 200) for i in range(15)},
})
source = proxy_data(_df=frame, _roles=RoleMap.from_primitive({'predictor': list(frame)}))
this = instance()
this._imports.set(source)
original_server, original_footer = this.server, this._footer
this.footer = lambda: ui.TagList(original_footer(), ui.output_text('PassThrough'))

def server(input, output, session):
    exported = original_server(input, output, session)
    @output
    @render.text
    def PassThrough():
        return f'unchanged={exported().equals(source)}'
    return exported

this.server = server
app = this.application()
