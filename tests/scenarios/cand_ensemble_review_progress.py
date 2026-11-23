"""Continuous training publications pass through the real candidate-review card."""
import os
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2] / 'app'
os.chdir(APP_ROOT)
sys.path.insert(0, str(APP_ROOT))

from cards.cand_ensemble import instance as ensemble_card
from cards.cand_models import instance as review_card
from ensemble_scenario import example_source
from module import Module
from shiny import App, reactive, render, ui

# Exercise production settling; checkbox readiness must not rely on test bypasses.
Module.running_under_tests = staticmethod(lambda: False)
source = example_source()
review, ensemble = review_card(), ensemble_card()
review._imports.set(source)
app_ui = ui.page_fillable(ui.head_content(
    [ui.include_js(path, method='inline') for path in Module.script_list],
    [ui.include_css(path, method='inline') for path in Module.css_list]),
    ui.div(review.call_ui(), ensemble.call_ui(), class_='cards-grid'), ui.output_text('UpdateProbe'), ui.output_text('ExportProbe'), ui.output_text('ReadinessProbe'))


def server(input, output, session):
    Module.ModSession = session
    reviewed = review.call_server(input, output, session)
    exported = ensemble.call_server(input, output, session, upstream=reactive.Value(reviewed))
    review.resume()
    ensemble.resume()

    async def initialise():
        for card in (review, ensemble):
            await session.send_custom_message('init_card', {'id': card.ns('Card')})
    session.on_flushed(initialise, once=True)
    updates = reactive.Value(0)

    @reactive.effect
    def progress():
        reactive.invalidate_later(.2)
        with reactive.isolate():
            count = updates() + 1
        updates.set(count)
        evidence = source.tuning_results
        evidence['state']['elapsed'] = count
        review._imports.set(source.with_tuning_results(evidence))

    readiness = {'pending': 0}

    @reactive.effect
    def record_readiness():
        phase = ensemble.export_readiness.phase()
        if phase is not None:
            readiness['pending'] += 1

    @output
    @render.text
    def ReadinessProbe():
        updates()
        return str(readiness['pending'])

    @output
    @render.text
    def ExportProbe():
        result = exported().ensemble_results
        return f'ensemble={len(result.members)}' if result else 'incoming'

    @output
    @render.text
    def UpdateProbe():
        return str(updates())


app = App(app_ui, server, static_assets=APP_ROOT / 'www')
