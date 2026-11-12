import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'app'
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from cards.data_models import instance
from proxy_data import proxy_data
from roles import RoleMap
from shiny import render, ui

this = instance()
source = proxy_data(pd.DataFrame({'x': [1., np.nan, 3.], 'cat': pd.Categorical(['a', 'b', 'a']), 'y': [1., 2., 3.]}),
                    _roles=RoleMap.from_primitive({'predictor': ['x', 'cat'], 'target': ['y']}))
if os.environ.get('PYTHAGORAS_TEST_MODEL_NUMERIC') == '1':
    source = proxy_data(pd.DataFrame({'x': np.arange(20.), 'y': 2*np.arange(20.)}),
                       _roles=RoleMap.from_primitive({'predictor': ['x'], 'target': ['y']}))
from evaluation_plan import build_plan

source = source.with_evaluation_plan(build_plan(source, strategy='kfold', objective='independent', folds=2), card='obs_resample')
this._imports.set(source)
original_server, original_footer = this.server, this._footer
this.footer = lambda: ui.TagList(original_footer(), ui.output_text('ExportState'), ui.output_text('ExportedMethod'), ui.output_text('ExportedCandidates'),
                               ui.download_link('Configuration', label='Configuration JSON'))
def server(input, output, session):
    exported = original_server(input, output, session)
    @output
    @render.text
    def ExportState():
        value = exported()
        return f'unchanged={value.equals(source)};pipeline={value.has_pipeline}'
    @output
    @render.text
    def ExportedMethod():
        value = exported()
        if not value.has_pipeline: return 'None'
        model = value.pipeline.named_steps.data_models.named_steps.estimator
        return type(model).__name__
    @output
    @render.text
    def ExportedCandidates():
        return ', '.join(c['label'] for c in exported().model_candidates) or 'None'
    @output
    @render.download(filename='model-configuration.json')
    def Configuration():
        yield json.dumps(this.configuration_state())
    return exported
this.server = server
app = this.application()
