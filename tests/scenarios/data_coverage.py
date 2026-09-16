"""Deterministic coverage gaps, role filtering and bookmark restoration."""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2] / 'app'
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import pandas as pd
from cards.data_coverage import instance
from proxy_data import proxy_data
from roles import Role, RoleMap
from shiny import render, ui

frame = pd.DataFrame([('North','Control','A')]*40 + [('North','Treated','B')]*10 + [('South','Control','B')]*5 + [('South','Treated','A')]*45,
    columns=['Region','Treatment','Group'])
frame['weight'] = range(1,101)
roles = RoleMap()
for c,r in [('Region',Role.STRATIFIER),('Treatment',Role.TREATMENT),('Group',Role.SENSITIVE),('weight',Role.WEIGHTING)]:
    roles.set_roles(c,[r])
source = proxy_data(_df=frame,_roles=roles,_name='Coverage test')
this = instance()
this.restore_configuration_state({'inputs':{'Variables':['Region','Treatment','Group'],'Area':'expected'}})
this._imports.set(source)
original_server, original_footer = this.server, this._footer
this.footer = lambda: ui.TagList(original_footer(),ui.output_text('PassThrough'))
def server(input,output,session):
    result = original_server(input,output,session)
    @output
    @render.text
    def PassThrough():
        return f'unchanged={result().equals(source)}'
    return result
this.server = server
app = this.application()
