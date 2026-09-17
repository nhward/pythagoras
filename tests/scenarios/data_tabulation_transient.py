"""Exercise missing and busy upstream sources without replacing the card."""
import os
import sys
import asyncio
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]/'app'
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
import pandas as pd
from shiny import reactive,render,ui
from cards.data_tabulation import instance
from proxy_data import proxy_data
this=instance()
original_server=this.server
original_footer=this._footer
this.footer=lambda:ui.TagList(original_footer(),ui.input_select(id='SourceState',label='Upstream state',
    choices=['ready','missing','pending'],selected='ready'),ui.output_text('SourceProbe'))
this._imports.set(proxy_data(pd.DataFrame({'value':[1,2,3]})))
def server(input,output,session):
    @reactive.extended_task
    async def compute():
        await asyncio.sleep(.3)
        return proxy_data(pd.DataFrame({'value':[101,102,103],'new_column':['a','b','c']}))
    @reactive.effect
    def start():
        if input.SourceState()=='pending':
            compute.cancel();compute.invoke()
    @reactive.calc
    def source():
        state=input.SourceState()
        if state=='missing':return None
        if state=='pending':return compute.result()
        return proxy_data(pd.DataFrame({'value':[1,2,3]}))
    this._upstream=lambda:source
    @output
    @render.text
    def SourceProbe():return input.SourceState()
    session.on_ended(compute.cancel)
    return original_server(input,output,session)
this.server=server
app=this.application()
