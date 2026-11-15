"""Construct learned cards inside each browser session, as the full app does."""
import json
import runpy
from dataclasses import replace
from pathlib import Path

scenario = runpy.run_path(str(Path(__file__).with_name('cand_train.py')))
from cards.cand_train import instance as tuning_card
from cards.miss_impute import ImputationStep
from cards.miss_impute import instance as imputation_card
from module import Module
from shiny import App, reactive, render, ui

app_ui = ui.page_fillable(
    ui.head_content(*[ui.include_js(p, method='inline') for p in dict.fromkeys(Module.script_list)],
                    *[ui.include_css(p, method='inline') for p in dict.fromkeys(Module.css_list)],
                    *[ui.tags.script(type='module', src=p) for p in dict.fromkeys(Module.mjs_list)]),
    ui.input_text_area('Saved', label='Saved state', value=''),
    ui.input_action_button('Build', label='Build workflow'),
    ui.output_text('Names'), ui.output_text('Snapshot'),
    ui.output_ui('Workflow'),
)


def server(input, output, session):
    current = reactive.Value(None)
    namespaces = reactive.Value('')

    @reactive.effect
    @reactive.event(input.Build)
    def build():
        impute = imputation_card()
        source = replace(scenario['source'], _tuning_spec=None)
        source = source.with_pipeline_step(ImputationStep(eligible=('x',), predictors=('x',), method='iterative'),
            name=impute.namespace, preview_frame=source.frame, operation='Learned imputation')
        tune = tuning_card()
        footer = tune._footer
        tune.footer = lambda: ui.TagList(footer(), ui.output_text('Message'))
        saved = json.loads(input.Saved()) if input.Saved() else {'inputs': {
            'Method': 'grid', 'Execution': 'sequential', 'Workers': 1, 'Seed': 1729,
            'ModelValues': '{"cand_methods__estimator__alpha":[0.01,0.1,10]}', 'Trials': 1}}
        tune.restore_configuration_state(saved)
        tune._imports.set(source)
        namespaces.set(impute.namespace + ';' + tune.namespace)
        current.set(tune)
        tune.call_server(input, output, session)
        tune.resume()
        async def initialise():
            await session.send_custom_message('init_card', {'id': tune.ns('Card')})
        session.on_flushed(initialise, once=True)

    @output
    @render.ui
    def Workflow():
        tune = current()
        return tune.call_ui() if tune is not None else None

    @output
    @render.text
    def Names():
        return namespaces()

    @output
    @render.text
    def Snapshot():
        tune = current()
        return json.dumps(tune.configuration_state()) if tune is not None else ''


app = App(app_ui, server, static_assets=Module.ROOT / 'www')
