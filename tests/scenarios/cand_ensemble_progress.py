"""Frequent upstream progress updates while a slower ensemble task is running."""
import time
from pathlib import Path
import os
import sys

APP_ROOT = Path(__file__).resolve().parents[2] / 'app'
os.chdir(APP_ROOT)
sys.path.insert(0, str(APP_ROOT))
from cards import cand_ensemble
from ensemble_scenario import example_source
from shiny import reactive, render, ui

counts = {'fits': 0, 'updates': 0}
original_evaluate = cand_ensemble.evaluate_ensemble


def slow_evaluate(*args, **kwargs):
    counts['fits'] += 1
    time.sleep(1.2)
    return original_evaluate(*args, **kwargs)


cand_ensemble.evaluate_ensemble = slow_evaluate
source = example_source()
this = cand_ensemble.instance()
this._imports.set(source)
original_server, original_footer = this.server, this._footer
this.footer = lambda: ui.TagList(original_footer(), ui.output_text('ProgressProbe'), ui.output_text('ExportProbe'))


def server(input, output, session):
    exported = original_server(input, output, session)

    @reactive.effect
    def progress():
        reactive.invalidate_later(.2)
        counts['updates'] += 1
        evidence = source.tuning_results
        evidence['state']['elapsed'] = counts['updates']
        this._imports.set(source.with_tuning_results(evidence))

    @output
    @render.text
    def ProgressProbe():
        reactive.invalidate_later(.2)
        return f"updates={counts['updates']};fits={counts['fits']}"

    @output
    @render.text
    def ExportProbe():
        result = exported().ensemble_results
        return f'ensemble={len(result.members)}' if result else 'incoming'
    return exported


this.server = server
app = this.application()
