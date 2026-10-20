import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'app'))

import pandas as pd
from cards.var_nominal import instance
from proxy_data import proxy_data
from shiny import render, ui

this = instance()
this._imports.set(proxy_data(pd.DataFrame({
    'Day': pd.Categorical(['Monday'] * 30 + ['Mon.'] * 3 + ['Mondya'] + ['Tuesday'] * 20),
    'Colour': pd.Categorical(['Blue'] * 40 + ['Blu'] * 3 + ['Red'] * 11),
})))

original_server, original_footer = this.server, this._footer
this.footer = lambda: ui.TagList(original_footer(), ui.output_text('Preview'))
def server(input, output, session):
    result = original_server(input, output, session)
    @output
    @render.text
    def Preview():
        return 'Monday count: ' + str(int((result().frame['Day'] == 'Monday').sum()))
    return result
this.server = server
app = this.application()
