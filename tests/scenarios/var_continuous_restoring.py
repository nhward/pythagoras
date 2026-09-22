"""Restore a selection while an upstream decimal type arrives later."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'app'
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import pandas as pd
from cards.var_continuous import instance
from proxy_data import proxy_data
from roles import RoleMap
from shiny import reactive, ui

frame = pd.DataFrame({'gapped': [0., 1., 8., 9.], 'later': [1., 2., 3., 4.]})
source = proxy_data(_df=frame, _roles=RoleMap.from_primitive({'predictor': list(frame)}))
early = source.clone()
early.frame['later'] = [1, 2, 3, 4]
this = instance()
this.restore_configuration_state({'inputs': {'Variables': ['gapped', 'later'], 'Limit': 4}})
this._imports.set(early)
original_server, original_footer = this.server, this._footer
this.footer = lambda: ui.TagList(original_footer(), ui.input_action_button('FinishRestore', label='Finish upstream restore'))

def server(input, output, session):
    exported = original_server(input, output, session)
    @reactive.effect
    @reactive.event(input.FinishRestore)
    def finish():
        this._imports.set(source.clone())
    return exported

this.server = server
app = this.application()
