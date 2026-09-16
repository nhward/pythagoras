"""Ordered encoding preview and independent learned pipeline toggle."""
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]/'app'
os.chdir(ROOT);sys.path.insert(0,str(ROOT))

import pandas as pd
from cards.var_encode import instance
from proxy_data import proxy_data
from roles import Role, RoleMap
from shiny import render, ui

frame=pd.DataFrame({'nominal':pd.Categorical(['a','b']*6),
    'code':pd.Series(['x','y','z']*4,dtype='string'),
    'ordered':pd.Categorical(['low','medium','high']*4,categories=['low','medium','high','very high'],ordered=True),
    'target':[float(i) for i in range(12)]})
roles=RoleMap()
for c in frame:roles.set_roles(c,[Role.TARGET if c=='target' else Role.PREDICTOR])
source=proxy_data(_df=frame,_roles=roles,_name='Ordered encoding example')
this=instance()
this.restore_configuration_state({'inputs':{'Encode':['ordered'],'EncodingType':'Ordered','OrderedMethod':'polynomial','OrderedDegree':2,'RemoveOriginal':False}})
this._imports.set(source)
original_server,original_footer=this.server,this._footer
this.footer=lambda:ui.TagList(original_footer(),ui.output_text('Probe'))
def server(input,output,session):
    result=original_server(input,output,session)
    @output
    @render.text
    def Probe():
        out=result()
        return f'steps={len(out.pipeline_steps)}; ordered_original={"ordered" in out.columns}; unchanged={out.equals(source)}'
    return result
this.server=server
app=this.application()
