"""Save and restore evaluated views without restarting training or evaluation."""
import json
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name('cand_ensemble.py')))
from cards import cand_ensemble
from ensemble_scenario import example_source
from module import Module
from shiny import App, reactive, render, ui

source = example_source()
calls = 0
original_evaluate = cand_ensemble.evaluate_ensemble


def evaluate(*args, **kwargs):
    global calls
    calls += 1
    return original_evaluate(*args, **kwargs)


cand_ensemble.evaluate_ensemble = evaluate
app_ui = ui.page_fillable(
    ui.head_content(
        *[ui.include_js(p, method='inline') for p in dict.fromkeys(Module.script_list)],
        *[ui.include_css(p, method='inline') for p in dict.fromkeys(Module.css_list)]),
    ui.input_action_button('Restore', label='Save and restore'),
    ui.input_action_button('RestoreLegacy', label='Restore legacy scores'),
    ui.input_action_button('RestoreMissing', label='Restore scores without diagnostics'),
    ui.output_ui('Workflow'), ui.output_text('Evaluations'))


def server(input, output, session):
    Module.ModSession = session
    current = reactive.Value(None)

    def build(saved=None):
        card = cand_ensemble.instance()
        if saved is not None:
            card.restore_configuration_state(saved)
        card._imports.set(source)
        card.call_server(input, output, session)
        card.resume()
        current.set(card)
        async def initialise():
            await session.send_custom_message('init_card', {'id': card.ns('Card')})
        session.on_flushed(initialise, once=True)

    build()

    def restore_card(version=None):
        card = current()
        saved = json.loads(json.dumps(card.configuration_state(), allow_nan=False))
        if version is not None:
            saved['evaluation']['version'] = version
            saved['evaluation'].pop('residuals', None)
        saved['inputs']['View'] = 'pairs'  # Old panel preference must not gate the combined chart.
        card.suspend()
        build(saved)

    @reactive.effect
    @reactive.event(input.Restore)
    def restore():
        restore_card()

    @reactive.effect
    @reactive.event(input.RestoreLegacy)
    def restore_legacy():
        restore_card(1)

    @reactive.effect
    @reactive.event(input.RestoreMissing)
    def restore_missing():
        restore_card(2)

    @output
    @render.ui
    def Workflow():
        return current().call_ui()

    @output
    @render.text
    def Evaluations():
        reactive.invalidate_later(.1)
        return f'evaluations={calls}'


app = App(app_ui, server, static_assets=Module.ROOT / 'www')
