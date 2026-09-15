"""Real importance weights, identifiers, and an observable membership pipeline."""
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
APP_ROOT = Path(__file__).resolve().parents[2] / 'app'
os.chdir(APP_ROOT)
sys.path.insert(0, str(APP_ROOT))
from cards.obs_clusters import instance
from proxy_data import proxy_data
from roles import Role, RoleMap
from shiny import reactive, render, ui

rng = np.random.default_rng(123)
frame = pd.DataFrame(np.vstack([rng.normal(-4, .2, (12, 2)), rng.normal(4, .2, (12, 2))]), columns=['x', 'y'])
frame['importance'] = np.resize([.5, 1., 2., 4.], len(frame))
frame.loc[0, 'importance'] = 0
frame['identifier'] = [f'case-{i + 1:02d}' for i in range(len(frame))]
frame['target'] = np.arange(len(frame)) * 1000.
roles = RoleMap()
for column in ['x', 'y']:
    roles.set_roles(column, [Role.PREDICTOR])
roles.set_roles('importance', [Role.WEIGHTING])
roles.set_roles('identifier', [Role.IDENTIFIER])
roles.set_roles('target', [Role.TARGET])
this = instance()
this.restore_configuration_state({'inputs': {'Projection': 'pca', 'Neighbours': 3}})
this._imports.set(proxy_data(_df=frame, _roles=roles, _cluster_count=2, _name='Two groups'))
original_server = this.server
original_footer = this._footer
this.footer = lambda: ui.TagList(original_footer(), ui.output_text('ExportProbe'), ui.output_text('PayloadProbe'), ui.input_action_button('ChangeK', label='Change upstream K'), ui.input_action_button('KOne', label='Set K to one'))

def server(input, output, session):
    selected = original_server(input, output, session)
    @reactive.effect
    @reactive.event(input.ChangeK)
    def change_k():
        this._imports.set(this._imports.get().with_cluster_count(3))

    @reactive.effect
    @reactive.event(input.KOne)
    def k_one():
        this._imports.set(this._imports.get().with_cluster_count(1))

    @output(suspend_when_hidden=False)
    @render.text
    def ExportProbe():
        data = selected()
        added = [col for col in data.columns if col not in frame.columns]
        if not added:
            return 'added=none'
        nominal = all(isinstance(data.frame[name].dtype, pd.CategoricalDtype) and not data.frame[name].cat.ordered for name in added)
        stratifiers = all(Role.STRATIFIER in data.role_map.roles_for(name) for name in added)
        methods = [record.parameters['method'] for record in data.processing_records if record.operation == 'Add cluster membership']
        return f"added={','.join(added)}; methods={','.join(methods)}; nominal={nominal}; stratifiers={stratifiers}; pipeline={','.join(data.pipeline_steps)}; clean_columns={','.join(data.clean_frame.columns)}"
    @output(suspend_when_hidden=False)
    @render.text
    def PayloadProbe():
        data = selected()
        added = [col for col in data.columns if col not in frame.columns]
        return json.dumps({
            'rows': len(data),
            'unchanged': data.frame[frame.columns].equals(frame),
            'weighting': sorted(data.role_map.columns_with_role(Role.WEIGHTING)),
            'levels': {col: list(data.frame[col].cat.categories) for col in added},
            'assigned': {col: int(data.frame[col].notna().sum()) for col in added},
            'recipes': {step.method: {'centre': step.centre, 'weighting': step.weighting}
                        for _, step in data.pipeline.steps} if data.has_pipeline else {},
        }, sort_keys=True)
    return selected
this.server = server
app = this.application()
