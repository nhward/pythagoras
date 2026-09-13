"""Small deterministic optimal-K scenario, including real observation importance."""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[2] / "app"
os.chdir(APP_ROOT)
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from cards.obs_k_clusters import instance
from proxy_data import proxy_data
from roles import Role, RoleMap
from shiny import render, ui


def scenario_data():
    rng = np.random.default_rng(1729)
    points = np.vstack([rng.normal(-4, .25, (16, 2)), rng.normal(4, .25, (16, 2))])
    frame = pd.DataFrame(points, columns=["x", "y"])
    frame["importance"] = np.resize([0.5, 1., 2., 4.], len(frame))
    frame.loc[0, "importance"] = 0
    frame["target"] = np.arange(len(frame)) * 1000.
    frame["id"] = np.arange(len(frame))
    roles = RoleMap()
    for col in ("x", "y"):
        roles.set_roles(col, [Role.PREDICTOR])
    roles.set_roles("importance", [Role.WEIGHTING])
    roles.set_roles("target", [Role.TARGET])
    roles.set_roles("id", [Role.IDENTIFIER])
    return proxy_data(_df=frame, _roles=roles, _name="Weighted two groups", _cluster_count=2)


this = instance()
this._imports.set(scenario_data())
original_server = this.server
original_footer = this._footer
this.footer = lambda: ui.TagList(original_footer(), ui.output_text("ExportProbe"))


def scenario_server(input, output, session):
    selected_data = original_server(input, output, session)

    @output
    @render.text
    def ExportProbe():
        data = selected_data()
        return (f"Export K={data.cluster_count}; rows={len(data)}; "
                f"weighting={','.join(sorted(data.role_map.columns_with_role(Role.WEIGHTING)))}; "
                f"unchanged={data.frame.equals(scenario_data().frame)}")

    return selected_data


this.server = scenario_server
app = this.application()
