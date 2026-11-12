"""Small deterministic regression search with an export probe."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'app'
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from cards.pipeline_tune import instance
from evaluation_plan import build_plan
from model_commit import prepare_predictors
from proxy_data import proxy_data
from roles import RoleMap
from shiny import render, ui
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from tuning import discover, fingerprint

x = np.arange(24, dtype=float)
source = proxy_data(pd.DataFrame({'x': x, 'y': 2*x + np.sin(x)}),
    _roles=RoleMap.from_primitive({'predictor': ['x'], 'target': ['y']}))
source = source.with_evaluation_plan(build_plan(source, strategy='kfold', objective='independent', folds=3), card='obs_resample')
source = source.with_estimator(Pipeline([
    ('predictors', FunctionTransformer(prepare_predictors, kw_args={'columns': ['x'], 'preparation': 'numeric'})),
    ('scale', StandardScaler()), ('estimator', Ridge(alpha=1.))]), label='Ridge', recipe_id='test-ridge')
source = source.with_tuning_spec({'fingerprint': fingerprint(discover(source.pipeline)),
    'grid': {'data_models__estimator__alpha': [.01, .1, 10.]}, 'combinations': 3})
this = instance()
this.restore_configuration_state({'inputs': {'Method': 'grid'}})
this._imports.set(source)
original_server, original_footer = this.server, this._footer
# Expose status messages in the test fixture even when hidden in the app footer.
this.footer = lambda: ui.TagList(original_footer(), ui.output_text('Message'), ui.output_text('ExportState'))


def server(input, output, session):
    exported = original_server(input, output, session)
    @output
    @render.text
    def ExportState():
        data = exported()
        if data is None: return 'waiting'
        accepted = [r for r in data.processing_records if r.card == 'pipeline_tune' and r.attempted]
        return f"accepted={bool(accepted)};alpha={data.pipeline.get_params()['data_models__estimator__alpha']}"
    return exported


this.server = server
app = this.application()
