"""Declared incompatibility blocks an export that would discard a pipeline."""
import os
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2] / 'app'
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import pandas as pd
from card import Card
from proxy_data import proxy_data
from shiny import reactive, render, req, ui
from sklearn.preprocessing import FunctionTransformer

this = Card(file=__file__, mutable=True, pipeline_compatible=False)
this.front = lambda: ui.TagList(
    ui.input_checkbox(id='Learned', label='Learning step upstream', value=True),
    ui.output_text('Result'), ui.output_text('Calls'),
)
source = proxy_data(pd.DataFrame({'x': [1, 2]}))
step = FunctionTransformer()
learned = source.with_pipeline_step(step, name='upstream', preview_frame=step.fit_transform(source.frame))
replacement = proxy_data(pd.DataFrame({'replacement': [3, 4]}))


def server(input, output, session):
    calls = reactive.Value(0)

    @reactive.calc
    def upstream():
        enabled = input.Learned()
        req(enabled is not None)
        return learned if enabled else source
    this._upstream.set(upstream)

    @reactive.calc
    def export():
        this.input_data()
        with reactive.isolate():
            calls.set(calls() + 1)
        return replacement

    @output
    @render.text
    def Result():
        result = this.output_data()
        return f'columns={list(result.frame.columns)}; pipeline={result.has_pipeline}'

    @output
    @render.text
    def Calls():
        return str(calls())

    return export


this.server = server
app = this.application()
