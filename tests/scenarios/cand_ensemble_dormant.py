"""Training progress without reviewed exports must leave ensemble work dormant."""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name('cand_ensemble.py')))
from cards import cand_ensemble
from ensemble_scenario import example_source
from session_task import SessionExtendedTask
from shiny import reactive, render, ui

source = example_source()
card = cand_ensemble.instance()
counts = {'updates': 0, 'evaluations': 0, 'cancellations': 0}
original_evaluate = cand_ensemble.evaluate_ensemble
original_cancel = SessionExtendedTask.cancel


def evaluate(*args, **kwargs):
    counts['evaluations'] += 1
    return original_evaluate(*args, **kwargs)


def cancel(task):
    if task._func.__name__ == 'Evaluate':
        counts['cancellations'] += 1
    return original_cancel(task)


cand_ensemble.evaluate_ensemble = evaluate
SessionExtendedTask.cancel = cancel
original_server, original_footer = card.server, card._footer
card.footer = lambda: ui.TagList(original_footer(),
    ui.input_action_button('Review', label='Export reviewed models'), ui.output_text('Activity'))
evidence = source.tuning_results
evidence['review'] = {}
card._imports.set(source.with_tuning_results(evidence))


def server(input, output, session):
    exported = original_server(input, output, session)
    reviewed = reactive.Value(False)

    @reactive.effect
    @reactive.event(input.Review)
    def review():
        reviewed.set(True)

    @reactive.effect
    def progress():
        reactive.invalidate_later(.15)
        counts['updates'] += 1
        evidence = source.tuning_results
        evidence['state']['elapsed'] = counts['updates']
        if not reviewed():
            evidence['review'] = {}
        card._imports.set(source.with_tuning_results(evidence))

    @output
    @render.text
    def Activity():
        reactive.invalidate_later(.15)
        return ';'.join(f'{key}={value}' for key, value in counts.items())
    return exported


card.server = server
app = card.application()
