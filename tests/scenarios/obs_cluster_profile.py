"""Deterministic cluster profiling with real importance and two membership targets."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'app'
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from cards.obs_cluster_profile import instance
from proxy_data import proxy_data
from roles import Role, RoleMap
from shiny import reactive, render, ui

rng = np.random.default_rng(2025)
frame = pd.DataFrame({'x': np.r_[rng.normal(-3,.2,30),rng.normal(3,.2,30)], 'extra':rng.normal(size=60)})
frame['cluster_partition'] = pd.Categorical(['c1']*30 + ['c2']*30)
frame['cluster_density_2'] = pd.Categorical(['unallocated']*10 + ['c1']*20 + ['c2']*30)
frame['importance'] = np.resize([.5,1.,2.],60)
frame.loc[0,'importance'] = 0
frame['identifier'] = np.arange(60)
roles = RoleMap()
for col in ['x','extra']: roles.set_roles(col,[Role.PREDICTOR])
for col in ['cluster_partition','cluster_density_2']: roles.set_roles(col,[Role.STRATIFIER])
roles.set_roles('importance',[Role.WEIGHTING])
roles.set_roles('identifier',[Role.IDENTIFIER])
source = proxy_data(_df=frame, _roles=roles, _cluster_count=2)
this = instance()
this._imports.set(source)
original_server, original_footer = this.server, this._footer
this.footer = lambda: ui.TagList(original_footer(),ui.output_text('PassThrough'), ui.input_action_button('RemoveMemberships',label='Remove membership roles'), ui.input_action_button('RestoreMemberships',label='Restore membership roles'))
def server(input,output,session):
    selected = original_server(input,output,session)
    @reactive.effect
    @reactive.event(input.RemoveMemberships)
    def remove():
        changed = source.clone()
        for name in ['cluster_partition','cluster_density_2']:
            changed.role_map.set_roles(name,[Role.NONE])
        this._imports.set(changed)
    @reactive.effect
    @reactive.event(input.RestoreMemberships)
    def restore():
        this._imports.set(source.clone())
    @output
    @render.text
    def PassThrough():
        return f'unchanged={selected().equals(source)}'
    return selected
this.server = server
app = this.application()
