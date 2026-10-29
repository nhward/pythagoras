import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'app'
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import pandas as pd
from cards.tuning_parameters import instance
from MissingnessTransformer import MissingnessTransformer
from proxy_data import proxy_data
from shiny import render, ui
from sklearn.ensemble import RandomForestRegressor

this = instance()
frame = pd.DataFrame({'x':[1.,2.,3.]})
step = MissingnessTransformer(remove_observations=True)
source = proxy_data(frame).with_pipeline_step(step, name='miss_map', preview_frame=step.fit_transform(frame))
source = source.with_estimator(RandomForestRegressor(), label='Random forest', recipe_id='test')
this._imports.set(source)
original_server, original_footer = this.server, this._footer
this.footer = lambda: ui.TagList(original_footer(), ui.output_text('ExportState'))
def server(input, output, session):
    exported = original_server(input, output, session)
    @output
    @render.text
    def ExportState():
        spec = exported().tuning_spec
        return 'No specification' if spec is None else f'Committed combinations: {spec["combinations"]}'
    return exported
this.server = server
app = this.application()
