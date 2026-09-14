"""Test-owned Shiny application for role-assignment browser tests."""

import os
import sys
from pathlib import Path

import pandas as pd

APP_ROOT = Path(__file__).resolve().parents[2] / "app"
os.chdir(APP_ROOT)
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from cards.var_roles import instance
from proxy_data import proxy_data

this = instance()
frame = pd.DataFrame(
    {
        "y": [1, 0, 1, 0],
        "x1": [10.0, 11.0, 12.0, 13.0],
        "x2": ["A", "B", "A", "B"],
        "id": [100, 101, 102, 103],
        "part": ["Train", "Train", "Test", "Test"],
    }
)
this._imports.set(proxy_data(_df=frame, _name="Test"))
# Test-only upstream controls and an observable downstream payload.
from shiny import reactive, render, ui

original_footer = this._footer
original_server = this.server
this.footer = lambda: ui.TagList(
    original_footer(),
    ui.input_action_button("ChangeK", label="Upstream K=3"),
    ui.output_text("ExportProbe"),
)


def scenario_server(input, output, session):
    exported = original_server(input, output, session)

    @reactive.effect
    @reactive.event(input.ChangeK)
    def change_k():
        this._imports.set(this._imports.get().with_cluster_count(3))

    @output
    @render.text
    def ExportProbe():
        data = exported()
        return (f"K={data.cluster_count}; target={','.join(data.role_map.to_primitive()['target'])}; "
                f"columns={','.join(data.columns)}; steps={len(data.processing_records)}")

    return exported


this.server = scenario_server
app = this.application()
