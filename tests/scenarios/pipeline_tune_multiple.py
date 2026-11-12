"""Two model branches, distinct grids and a final-method export probe."""
import runpy
from pathlib import Path

scenario = runpy.run_path(str(Path(__file__).with_name('pipeline_tune.py')))
from shiny import render, ui
from sklearn.base import clone
from sklearn.tree import DecisionTreeRegressor
from tuning import compile_spec, discover_source

this = scenario['this']
base = scenario['source']
ridge = base.pipeline.named_steps.data_models
tree = clone(ridge).set_params(estimator=DecisionTreeRegressor(max_depth=2))
source = base.with_model_candidates([
    {'id':'ridge', 'label':'Ridge Regression', 'estimator':ridge},
    {'id':'tree', 'label':'Decision Tree Regression', 'estimator':tree}])
rows = discover_source(source)
drafts = {}
for row in rows:
    if row.get('candidate_id') == 'ridge' and row['path'].endswith('__alpha'):
        values = [.1, 10.]
    elif row.get('candidate_id') == 'tree' and row['path'].endswith('__max_depth'):
        values = [2, 3]
    else:
        continue
    drafts[row['path']] = dict(row['definition'], values=values, owner=row['owner'], active=True)
source = source.with_tuning_spec(compile_spec(rows, drafts))
this._imports.set(source)
this.restore_configuration_state({'inputs':{'Method':'grid', 'Execution':'folds', 'Workers':2}})
this.footer = lambda: ui.TagList(scenario['original_footer'](), ui.output_text('Message'), ui.output_text('MultiState'))
def server(input, output, session):
    exported = scenario['original_server'](input, output, session)
    @output
    @render.text
    def MultiState():
        data = exported()
        if not data.has_pipeline:
            return 'No method accepted'
        estimator = data.pipeline.named_steps.data_models.named_steps.estimator
        return type(estimator).__name__ + ';depth=' + str(getattr(estimator, 'max_depth', None))
    return exported
this.server = server
app = this.application()
