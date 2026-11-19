"""Five reviewed method winners with an observable downstream export."""
import os
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2] / 'app'
os.chdir(APP_ROOT)
sys.path.insert(0, str(APP_ROOT))
from cards.cand_ensemble import instance
from ensemble_scenario import example_source
from shiny import render, ui


def application(task='regression', restored=False):
    source = example_source(task, count=5)
    this = instance()
    if restored:
        this.restore_configuration_state({'inputs': {'Models': ['method_1', 'method_3'],
            'Style': 'median', 'Metric': 'r2', 'Apply': ['Ensemble']}})
    this._imports.set(source)
    original_server, original_footer = this.server, this._footer
    this.footer = lambda: ui.TagList(original_footer(), ui.output_text('ExportProbe'))

    def server(input, output, session):
        exported = original_server(input, output, session)
        @output
        @render.text
        def ExportProbe():
            result = exported().ensemble_results
            return f'ensemble={len(result.members)};fitted={len(result.fitted)}' if result else 'incoming'
        return exported
    this.server = server
    return this.application()

app = application()
