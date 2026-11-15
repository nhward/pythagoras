"""Replay completed evaluations on a clock to exercise live publication timing."""
import runpy
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

scenario = runpy.run_path(str(Path(__file__).with_name('cand_train.py')))

import cards.cand_train as tuning
import cards.model_candidates as reviewing
from cards.model_candidates import instance
from module import Module
from pipeline_search import fresh, prepare, run
from shiny import App, reactive, render, ui

source, tune = scenario['source'], scenario['this']
problem = prepare(source, method='grid', execution='sequential', workers=1)
completed = run(problem, fresh(problem), trials=0, publish=lambda state: None, pause=lambda: False)


class ReplayWorker:
    """Fitting is real; only its delivery times are controlled for this test."""
    process = None

    def __init__(self):
        self.started = None
        self.running = False
        self.delivered = -1

    def start(self, problem, state, trials, seconds):
        self.started = time.monotonic()
        self.initial = deepcopy(state)
        self.running = True
        self.delivered = -1

    @property
    def elapsed(self):
        return time.monotonic() - self.started if self.started is not None else 0

    def poll(self):
        if not self.running:
            return None
        count = min(4, int(self.elapsed // 2))
        if count == self.delivered:
            return None
        self.delivered = count
        current = deepcopy(self.initial)
        current['trials'] = deepcopy(completed['trials'][:count])
        current['elapsed'] = self.elapsed
        current['status'] = 'exhausted' if count == 4 else 'running'
        if count == 4:
            self.running = False
        return current

    def pause(self):
        pass

    def close(self):
        self.running = False


tuning.SearchWorker = ReplayWorker
tune.restore_configuration_state({'inputs': {'Method': 'grid', 'Execution': 'sequential',
                                             'Workers': 1, 'RefreshSeconds': 30}})
review = instance()
review_calls = reactive.Value(0)
source_event = reactive.Value(0)
original_review = reviewing.review_frame


def counted_review(*args, **kwargs):
    with reactive.isolate():
        review_calls.set(review_calls() + 1)
    return original_review(*args, **kwargs)


reviewing.review_frame = counted_review
head = ui.head_content(
    [ui.include_js(script, method='inline') for script in dict.fromkeys(Module.script_list)],
    [ui.include_css(style, method='inline') for style in dict.fromkeys(Module.css_list)],
    [ui.tags.script(type='module', src=script) for script in dict.fromkeys(Module.mjs_list)],
)
app_ui = ui.page_fillable(head, ui.div(tune.call_ui(), review.call_ui(),
                                      id='cards-container', class_='cards-grid'),
                          ui.output_text('PublishedTrials'), ui.output_text('ReviewCalls'),
                          ui.output_text('SourceEvent'),
                          ui.input_action_button('RepeatSource', label='Repeat equivalent source'),
                          ui.input_action_button('ChangeSource', label='Change source data'))


def server(input, output, session):
    Module.ModSession = session
    exported = tune.call_server(input, output, session)
    review_export = review.call_server(input, output, session, upstream=reactive.Value(exported))

    @output
    @render.text
    def PublishedTrials():
        # Read both exports: stale buffered diagnostics must not leak downstream.
        data = review_export()
        results = data.tuning_results
        return str(len(results['state']['trials']) if results else 0)

    @output
    @render.text
    def ReviewCalls():
        return str(review_calls())

    @output
    @render.text
    def SourceEvent():
        review_export()
        return str(source_event())

    @reactive.effect
    @reactive.event(input.RepeatSource)
    def repeat_source():
        tune._imports.set(source.clone())
        source_event.set(input.RepeatSource())

    @reactive.effect
    @reactive.event(input.ChangeSource)
    def change_source():
        frame = source.frame.copy()
        frame['y'] += 100
        tune._imports.set(replace(source, _df=frame, _clean_df=None))

    tune.resume()
    review.resume()

    async def initialise():
        for card in (tune, review):
            await session.send_custom_message('init_card', {'id': card.ns('Card')})
    session.on_flushed(initialise, once=True)


app = App(app_ui, server, static_assets=Module.ROOT / 'www')
