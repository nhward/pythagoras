"""Winning configurations change faster than one ensemble evaluation can finish."""
import os
import sys
import time
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2] / 'app'
os.chdir(APP_ROOT)
sys.path.insert(0, str(APP_ROOT))
from cards import cand_ensemble
from ensemble_scenario import example_source
from shiny import reactive, render, ui

source = example_source()
counts = {'fits': 0, 'updates': 0}
original = cand_ensemble.evaluate_ensemble


def slow(*args, **kwargs):
    counts['fits'] += 1
    time.sleep(1.2)
    return original(*args, **kwargs)


cand_ensemble.evaluate_ensemble = slow
this = cand_ensemble.instance()
this._imports.set(source)
original_server, original_footer = this.server, this._footer
this.footer = lambda: ui.TagList(original_footer(), ui.input_action_button('StopUpdates', label='Pause winner updates'),
    ui.output_text('ProgressProbe'), ui.output_text('ExportProbe'))


def server(input, output, session):
    exported = original_server(input, output, session)
    paused = reactive.Value(False)

    @reactive.effect
    @reactive.event(input.StopUpdates)
    def stop():
        paused.set(True)

    @reactive.effect
    def progress():
        if paused():
            return
        reactive.invalidate_later(.15)
        counts['updates'] += 1
        evidence = source.tuning_results
        # Simulate the review choosing a newly completed, differently configured
        # ridge trial. Scores are fixture metadata; the ensemble refits it for real.
        previous = next(t for t in evidence['state']['trials'] if t['model_id'] == 'method_0')
        trial = {**previous, 'id': 1000 + counts['updates'],
                 'params': {**previous['params'], 'cand_methods__estimator__alpha': counts['updates'] / 10}}
        evidence['state']['trials'].append(trial)
        evidence['pipelines'].choices[str(trial['id'])] = ('method_0', trial['params'])
        evidence['review']['representatives']['method_0'] = trial['id']
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
