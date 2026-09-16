"""Code and nominal encoding, restored toggles and late Target-role changes."""
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]/'app'
os.chdir(ROOT);sys.path.insert(0,str(ROOT))

import pandas as pd
from cards.var_encode import instance
from proxy_data import proxy_data
from roles import Role, RoleMap
from shiny import reactive, render, ui

frame=pd.DataFrame({'code':pd.Series(['a','b','c']*6,dtype='string'),
    'nominal':pd.Categorical(['yes','no']*9), 'target':[float(i) for i in range(18)],
    'weight':[float(i+1) for i in range(18)]})
frame['target']=pd.Categorical(['low','middle','high']*6)
roles=RoleMap()
for c,r in [('code',Role.PREDICTOR),('nominal',Role.PREDICTOR),('target',Role.TARGET),('weight',Role.WEIGHTING)]:
    roles.set_roles(c,[r])
source=proxy_data(_df=frame,_roles=roles,_name='Code encoding test')
this=instance()
this.restore_configuration_state({'inputs':{'Encode':['nominal','code'],'EncodingType':'Code','CodeMethod':'smooth'}})
this._imports.set(source)
original_server,original_footer=this.server,this._footer
this.footer=lambda:ui.TagList(original_footer(),ui.output_text('Probe'),
    ui.input_action_button('NoTarget',label='Remove Target role'),
    ui.input_action_button('RestoreTarget',label='Restore Target role'))
def server(input,output,session):
    result=original_server(input,output,session)
    @reactive.effect
    @reactive.event(input.NoTarget)
    def remove():
        changed=source.clone();changed.role_map.clear_roles('target')
        this._imports.set(changed)
    @reactive.effect
    @reactive.event(input.RestoreTarget)
    def restore():
        this._imports.set(source.clone())
    @output
    @render.text
    def Probe():
        out=result()
        return f'steps={len(out.pipeline_steps)}; code_original={"code" in out.columns}; nominal_original={"nominal" in out.columns}; unchanged={out.equals(this.input_data())}'
    return result
this.server=server
app=this.application()
