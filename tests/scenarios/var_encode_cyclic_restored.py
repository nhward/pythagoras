"""Numeric and categorical cyclic predictors without a Target."""
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]/'app'
os.chdir(ROOT);sys.path.insert(0,str(ROOT))

import pandas as pd
from cards.var_encode import instance
from cyclic_pandas import as_cyclic
from proxy_data import proxy_data
from shiny import render, ui

frame=pd.DataFrame({'hour':as_cyclic(pd.Series([0.,6.,12.,18.,None]),period=24),
    'season':as_cyclic(pd.Series(pd.Categorical(['winter','spring','summer','autumn',None],
        categories=['winter','spring','summer','autumn'],ordered=True)))})
source=proxy_data(_df=frame,_name='Cyclic encoding example')
this=instance()
this.restore_configuration_state({'inputs':{'Encode':['cyclic'],'EncodingType':'Cyclic'}})
this._imports.set(source)
original_server,original_footer=this.server,this._footer
this.footer=lambda:ui.TagList(original_footer(),ui.output_text('Probe'))
def server(input,output,session):
    result=original_server(input,output,session)
    @output
    @render.text
    def Probe():
        out=result()
        return f'steps={len(out.pipeline_steps)}; original={"hour" in out.columns}; unchanged={out.equals(source)}'
    return result
this.server=server
app=this.application()
