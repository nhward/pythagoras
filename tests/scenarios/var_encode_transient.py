"""Slow encoding and transient upstream data exercise output progress states."""
import asyncio
import os
import sys
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2] / 'app'
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import pandas as pd
from shiny import reactive, render, ui
from cards import var_encode
from proxy_data import proxy_data

analyze = var_encode._analyze_panels
def slow_analyze(*args, **kwargs):
    time.sleep(.5)
    return analyze(*args, **kwargs)
var_encode._analyze_panels = slow_analyze
this = var_encode.instance()
original_server, original_footer = this.server, this._footer
this.footer = lambda: ui.TagList(original_footer(), ui.input_select(
    'SourceState', label='Upstream state', choices=['ready', 'missing', 'pending'], selected='ready'),
    ui.output_text('SourceProbe'))
def dataset(extra=False):
    frame = pd.DataFrame({'flag': pd.Series([True, False, None], dtype='boolean')})
    if extra:
        frame['extra'] = [1., 2., 3.]
    return proxy_data(_df=frame)
this._imports.set(dataset())
def server(input, output, session):
    @reactive.extended_task
    async def compute():
        await asyncio.sleep(.5)
        return dataset(extra=True)
    @reactive.effect
    def start():
        if input.SourceState() == 'pending':
            compute.cancel()
            compute.invoke()
    @reactive.calc
    def source():
        state = input.SourceState()
        if state == 'missing':
            return None
        return compute.result() if state == 'pending' else dataset()
    this._upstream = lambda: source
    @output
    @render.text
    def SourceProbe():
        return input.SourceState()
    session.on_ended(compute.cancel)
    return original_server(input, output, session)
this.server = server
app = this.application()
