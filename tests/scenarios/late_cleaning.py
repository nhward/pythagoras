"""Exercise late cleaning through the shared card publication path."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'app'
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import pandas as pd
from card import Card
from cards.miss_map import _transform_data
from proxy_data import proxy_data
from shiny import reactive, render, ui

this = Card(file=__file__, mutable=True)
this.front = lambda: ui.TagList(
    ui.input_checkbox(id='Learned', label='Learning step upstream', value=True),
    ui.output_text('Result'),
)
source = proxy_data(pd.DataFrame({'x': [1., None, 3.]}))
learned = _transform_data(source, remove_variables=True, remove_observations=False,
                          variable_threshold=0.9, observation_threshold=0.5)


def server(input, output, session):
    @reactive.calc
    def upstream():
        return learned if input.Learned() else source
    this._upstream.set(upstream)

    @reactive.calc
    def export():
        data = this.input_data()
        frame = data.frame.assign(shadow__x=data.frame['x'].isna().astype('int8'))
        return data.with_cleaned_data(frame, card='test_cleaner', operation='Add indicator')

    # A positioning error in an effect must not close the session either.
    @reactive.effect
    def consume():
        export()

    @output
    @render.text
    def Result():
        result = this.output_data()
        return f'columns={list(result.frame.columns)}; pipeline={result.has_pipeline}'
    return export


this.server = server
app = this.application()
