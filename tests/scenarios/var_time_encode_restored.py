"""Date/time features without a Target."""
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]/'app'
os.chdir(ROOT);sys.path.insert(0,str(ROOT))

import pandas as pd
from cards.var_time_encode import instance
from proxy_data import proxy_data
from shiny import render, ui

frame=pd.DataFrame({'flag':pd.to_datetime(['2024-02-29 13:45:00','2024-03-01 12:00:00',None])})
source=proxy_data(_df=frame,_name='Time encoding example')
this=instance()
this.restore_configuration_state({'inputs':{'Encode':['time'],'Features':['month','hour_24']}})
this._imports.set(source)
original_server,original_footer=this.server,this._footer
this.footer=lambda:ui.TagList(original_footer(),ui.output_text('Probe'))
def server(input,output,session):
    result=original_server(input,output,session)
    @output
    @render.text
    def Probe():
        out=result()
        return f'steps={len(out.pipeline_steps)}; original={"flag" in out.columns}; unchanged={out.equals(source)}'
    return result
this.server=server
app=this.application()
