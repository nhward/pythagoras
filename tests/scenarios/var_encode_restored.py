"""Standalone encoding-card browser scenario."""
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]/'app'
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
import pandas as pd
from cards.var_encode import instance
from proxy_data import proxy_data
from shiny import render,ui

frame=pd.DataFrame({'binary':pd.Categorical(['yes','no','yes','no',None,'yes']),
    'color':pd.Categorical(['red','blue','green','red','red','blue']), 'numeric':range(6)})
source=proxy_data(_df=frame,_name='Encoding test')
this=instance()
this.restore_configuration_state({'inputs':{'Encode':['nominal'],'RemoveOriginal':True}})
this._imports.set(source)
original_server,original_footer=this.server,this._footer
this.footer=lambda:ui.TagList(original_footer(),ui.output_text('Probe'))
def server(input,output,session):
    result=original_server(input,output,session)
    @output
    @render.text
    def Probe():
        out=result()
        return f'steps={len(out.pipeline_steps)}; original={"binary" in out.columns}; unchanged={out.equals(source)}'
    return result
this.server=server
app=this.application()
