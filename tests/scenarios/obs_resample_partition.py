"""Three explicitly assigned Partition levels, with live configuration capture."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "app"
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from cards.obs_clusters import _export_memberships
from cards.obs_resample import instance
from ClusterMembershipTransformer import ClusterMembershipTransformer
from proxy_data import proxy_data
from roles import RoleMap
from shiny import render, ui

frame = pd.DataFrame({"x": np.arange(60.), "partition": np.repeat(["train a", "train b", "validation", "holdout", "prediction"], 12),
    "site": pd.Categorical(np.tile(["North", "South", "West"], 20)),
    "target": pd.Categorical(np.tile(["yes", "no"], 30))})
this = instance()
this.restore_configuration_state({"inputs": {"Goal": "none"}, "partition_mapping": {
    "variable": "partition", "levels": {"str:train a": "development", "str:train b": "development"}}})
source = proxy_data(_df=frame, _name="Partition example", _roles=RoleMap.from_primitive({
    "predictor": ["x"], "stratifier": ["site"], "partition": ["partition"], "target": ["target"]}))
transformer = ClusterMembershipTransformer(("x",), n_clusters=2, output_column="cluster__membership").fit(source.frame)
this._imports.set(_export_memberships(source, {"Partition": transformer}))
original_server, original_footer = this.server, this._footer
this.footer = lambda: ui.TagList(original_footer(), ui.output_text("ExportState"), ui.output_text("MappingState"), ui.output_text("RecipeState"))
def server(input, output, session):
    exported = original_server(input, output, session)
    @output
    @render.text
    def ExportState():
        plan = exported().evaluation_plan
        return "No plan" if plan is None else f"ready={plan.ready};development={len(plan.development)};reserved={len(plan.reserved)}"
    @output
    @render.text
    def RecipeState():
        import json
        plan = exported().evaluation_plan
        return json.dumps(plan.recipe if plan else {}, sort_keys=True)

    @output
    @render.text
    def MappingState():
        import json
        return json.dumps(this.configuration_state().get("partition_mapping", {}), sort_keys=True)
    return exported
this.server = server
app = this.application()
