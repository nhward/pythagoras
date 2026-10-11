"""Mixed-structure overview with an immutable output probe."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'app'
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from cards.data_structure import instance
from proxy_data import proxy_data
from roles import RoleMap
from shiny import render, ui
from text_pandas import as_text

frame = pd.DataFrame({'id':np.repeat(['A','B'], 30),
    'time':list(pd.date_range('2025-01-01', periods=30))*2,
    'value':np.tile(np.sin(np.arange(30)/5),2),
    'text':as_text(['你好']*60)})
source = proxy_data(_df=frame, _roles=RoleMap.from_primitive({'identifier':['id'], 'sequence':['time'], 'predictor':['value']}))
this = instance()
this._imports.set(source)
original = this.server
footer = this._footer
this.footer = lambda: ui.TagList(footer(), ui.output_text('Probe'))

def server(input, output, session):
    exported = original(input, output, session)
    @output
    @render.text
    def Probe():
        return f'unchanged={exported() is source}'
    return exported

this.server = server
app = this.application()
