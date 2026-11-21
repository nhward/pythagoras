"""Native wrapper defaults remain unset while their documented meanings display."""
import pandas as pd
from cards.cand_parameters import instance
from catboost_model import PreparedCatBoostRegressor
from proxy_data import proxy_data
from shiny import render, ui
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

this = instance()
source = proxy_data(pd.DataFrame({'x': [1., 2., 3.]})).with_model_candidates([
    {'id': 'catboost', 'label': 'CatBoost', 'estimator': Pipeline([('estimator', PreparedCatBoostRegressor())])},
    {'id': 'xgboost', 'label': 'XGBoost', 'estimator': Pipeline([('estimator', XGBRegressor())])},
])
this._imports.set(source)
original_server, original_footer = this.server, this._footer
this.footer = lambda: ui.TagList(original_footer(), ui.output_text('DefaultsProbe'))


def server(input, output, session):
    exported = original_server(input, output, session)

    @output
    @render.text
    def DefaultsProbe():
        candidates = exported().model_candidates
        return f"iterations={candidates[0]['estimator'].named_steps['estimator'].get_params().get('iterations')};depth={candidates[1]['estimator'].named_steps['estimator'].get_params().get('max_depth')}"
    return exported


this.server = server
app = this.application()
