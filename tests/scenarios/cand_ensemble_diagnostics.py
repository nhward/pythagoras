"""Observable evaluation count for manual membership and residual-panel changes."""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name('cand_ensemble.py')))
from cards import cand_ensemble
from ensemble_scenario import example_source
from shiny import render, reactive, ui

calls = 0
original_evaluate = cand_ensemble.evaluate_ensemble


def evaluate(*args, **kwargs):
    global calls
    calls += 1
    return original_evaluate(*args, **kwargs)


cand_ensemble.evaluate_ensemble = evaluate
card = cand_ensemble.instance()
card._imports.set(example_source(count=5))
original_server, original_footer = card.server, card._footer
card.footer = lambda: ui.TagList(original_footer(), ui.output_text('EvaluationCalls'))


def server(input, output, session):
    exported = original_server(input, output, session)

    @output
    @render.text
    def EvaluationCalls():
        reactive.invalidate_later(.1)
        return str(calls)
    return exported


card.server = server
app = card.application()
