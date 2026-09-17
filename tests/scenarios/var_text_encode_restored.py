"""Date/time features without a Target."""
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]/'app'
os.chdir(ROOT);sys.path.insert(0,str(ROOT))

import pandas as pd
from text_pandas import as_text
from cards.var_text_encode import instance
from proxy_data import proxy_data
from shiny import render, ui

frame=pd.DataFrame({'flag':as_text(pd.Series(['This is excellent!','This is terrible.', 'A good product and good service.', 'A bad product and bad service.', '',None]))})
source=proxy_data(_df=frame,_name='Time encoding example')
this=instance()
this.restore_configuration_state({'inputs':{'Encode':['sentiment','characteristics'],'EncodingType':'Sentiment','RemoveOriginal':False}})
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
