"""A second upstream method appears after an initial single-method review."""
from pathlib import Path
import os
import sys
APP_ROOT = Path(__file__).resolve().parents[2] / 'app'
os.chdir(APP_ROOT)
sys.path.insert(0, str(APP_ROOT))
from cards import cand_ensemble
from ensemble_scenario import example_source
from shiny import reactive, render, ui

source = example_source()
evidence = source.tuning_results
first = next(iter(evidence['review']['representatives']))
evidence['review']['representatives'] = {first: evidence['review']['representatives'][first]}
this = cand_ensemble.instance()
this._imports.set(source.with_tuning_results(evidence))
calls = []
original_evaluate = cand_ensemble.evaluate_ensemble


def counted(*args, **kwargs):
    calls.append(1)
    return original_evaluate(*args, **kwargs)


cand_ensemble.evaluate_ensemble = counted
original_server, original_footer = this.server, this._footer
this.footer = lambda: ui.TagList(original_footer(), ui.input_action_button('SecondModel', label='Add second model'), ui.output_text('FitCalls'))


def server(input, output, session):
    exported = original_server(input, output, session)
    @reactive.effect
    @reactive.event(input.SecondModel)
    def second():
        this._imports.set(source)
    @output
    @render.text
    def FitCalls():
        reactive.invalidate_later(.2)
        return f'fits={len(calls)}'
    return exported


this.server = server
app = this.application()
