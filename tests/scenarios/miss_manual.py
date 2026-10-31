"""Healthcare example with a prior pipeline step and multiple data types."""
import os
import sys
from pathlib import Path
import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[2] / 'app'
os.chdir(APP_ROOT)
sys.path.insert(0, str(APP_ROOT))
from cards import miss_manual as m
from proxy_data import proxy_data
from roles import Role, RoleMap
from shiny import render, ui
from sklearn.preprocessing import FunctionTransformer

frame = pd.DataFrame({
    'HEALTHCARE_COST': pd.Series([None, None, 20, 40] * 5, dtype='Int64'),
    'HEALTHCARE_BASIS': pd.Categorical(['FREE', 'FREE', 'PAID', 'PAID'] * 5),
    'LABEL': pd.Categorical([None, 'known', 'known', 'known'] * 5),
    'OUTCOME': [0, 0, 1, 1] * 5,
    'VISIT_DATE': pd.to_datetime([None, '2025-01-01', '2025-01-02', '2025-01-03'] * 5),
    'ORDERED': pd.Categorical([None, 'low', 'high', 'low'] * 5, categories=['low', 'high'], ordered=True),
})
roles = RoleMap()
roles.set_roles('HEALTHCARE_COST', [Role.PREDICTOR])
roles.set_roles('HEALTHCARE_BASIS', [Role.IDENTIFIER])
roles.set_roles('OUTCOME', [Role.TARGET])
source = proxy_data(_df=frame, _roles=roles, _name='Healthcare example')
source = source.with_pipeline_step(FunctionTransformer(), name='identity', preview_frame=frame.copy())
this = m.instance()
original_server = this.server

def server(input, output, session):
    result = original_server(input, output, session)
    @output
    @render.text
    def ExportProbe():
        data = result()
        return f"missing={data.frame.HEALTHCARE_COST.isna().sum()}; first={data.frame.HEALTHCARE_COST.iloc[0]}; clean_missing={data.clean_frame.HEALTHCARE_COST.isna().sum()}; steps={','.join(n for n, _ in data.pipeline_for_training().steps)}"
    return result

this.server = server
footer = this._footer
this.footer = lambda: ui.TagList(footer(), ui.output_text('ExportProbe'))
this._imports.set(source)
app = this.application()
