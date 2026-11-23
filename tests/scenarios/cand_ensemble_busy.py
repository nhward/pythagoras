"""One snapshot evaluation spans repeated upstream-readiness changes."""
import runpy
import time
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name('cand_ensemble.py')))
from cards import cand_ensemble
from ensemble_scenario import example_source
from shiny import reactive, render, ui

counts = {'evaluations': 0, 'waits': 0}
original_evaluate = cand_ensemble.evaluate_ensemble


def evaluate(*args, **kwargs):
    counts['evaluations'] += 1
    time.sleep(3)
    return original_evaluate(*args, **kwargs)


cand_ensemble.evaluate_ensemble = evaluate
card = cand_ensemble.instance()
card._imports.set(example_source())
original_server, original_footer = card.server, card._footer
card.footer = lambda: ui.TagList(original_footer(), ui.output_text('Activity'))


def server(input, output, session):
    exported = original_server(input, output, session)

    @reactive.effect
    def readiness_changes():
        reactive.invalidate_later(.05)
        if counts['evaluations'] and counts['waits'] < 40:
            counts['waits'] += 1
            card.waiting.set('Awaiting upstream data' if counts['waits'] % 2 else None)

    @output
    @render.text
    def Activity():
        reactive.invalidate_later(.05)
        return f"evaluations={counts['evaluations']};waits={counts['waits']}"

    return exported


card.server = server
app = card.application()
