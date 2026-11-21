"""Real training, review and ensemble cards sharing one session fit service."""
import runpy
from dataclasses import replace
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name('cand_ensemble.py')))
from cards.cand_train import instance as training_card
from cards.cand_models import instance as review_card
from cards.cand_ensemble import instance as ensemble_card
from ensemble_scenario import example_source
from module import Module
from shiny import App, reactive, render, ui

source = replace(example_source(), _tuning_results=None, _selected_candidate=None)
app_ui = ui.page_fillable(
    ui.head_content(
        *[ui.include_js(p, method='inline') for p in dict.fromkeys(Module.script_list)],
        *[ui.include_css(p, method='inline') for p in dict.fromkeys(Module.css_list)]),
    ui.output_ui('Workflow'), ui.output_text('ExportProbe'))


def server(input, output, session):
    Module.ModSession = session
    train, review, ensemble = training_card(), review_card(), ensemble_card()
    train.restore_configuration_state({'inputs': {'Method': 'grid', 'Execution': 'sequential', 'Workers': 1}})
    train._imports.set(source)
    trained = train.call_server(input, output, session)
    reviewed = review.call_server(input, output, session, upstream=reactive.Value(trained))
    exported = ensemble.call_server(input, output, session, upstream=reactive.Value(reviewed))
    for card in (train, review, ensemble):
        card.resume()

    @output
    @render.ui
    def Workflow():
        return ui.div(train.call_ui(), review.call_ui(), ensemble.call_ui(), class_='cards-grid')

    @output
    @render.text
    def ExportProbe():
        result = exported().ensemble_results
        return f'ensemble={len(result.members)}' if result else 'incoming'


app = App(app_ui, server, static_assets=Module.ROOT / 'www')
