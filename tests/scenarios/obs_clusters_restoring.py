"""Fixed-K cluster membership scenario with an observable downstream payload."""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
APP_ROOT = Path(__file__).resolve().parents[2] / 'app'
os.chdir(APP_ROOT)
sys.path.insert(0, str(APP_ROOT))
from cards.obs_clusters import instance
from cards import obs_clusters as cluster_module
import threading
from proxy_data import proxy_data
from roles import Role
from shiny import reactive, render, ui

# Deliberately deliver one obsolete completion, as happens while bookmark
# settings/upstream data are still settling. A restored checkbox requests it.
original_analyze = cluster_module._analyze
stale_finished = threading.Event()
first_run = True
def restored_analysis(*args, **kwargs):
    global first_run
    result = original_analyze(*args, **kwargs)
    if first_run:
        first_run = False
        result.options = dict(result.options, projection='obsolete-restored-option')
        stale_finished.set()
    return result
cluster_module._analyze = restored_analysis

rng = np.random.default_rng(123)
frame = pd.DataFrame(np.vstack([rng.normal(-4, .2, (12, 2)), rng.normal(4, .2, (12, 2))]), columns=['x', 'y'])
this = instance()
this.restore_configuration_state({'inputs': {'Projection': 'pca', 'Neighbours': 3, 'IncludeMembership': ['Partition']}})
this._imports.set(proxy_data(_df=frame, _cluster_count=2, _name='Two groups'))
original_server = this.server
original_footer = this._footer
this.footer = lambda: ui.TagList(original_footer(), ui.output_text('ExportProbe'), ui.output_text('RestorationProbe'), ui.input_action_button('ChangeK', label='Change upstream K'), ui.input_action_button('KOne', label='Set K to one'))

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

    @output
    @render.text
    def RestorationProbe():
        reactive.invalidate_later(.05)
        return 'Obsolete completion delivered' if stale_finished.is_set() else 'Waiting'

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
    return selected
this.server = server
app = this.application()
