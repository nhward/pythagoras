"""Real GP method selection, scalar kernel commitment and pipeline training."""
import numpy as np
import pandas as pd
from cards.cand_methods import instance as method_card
from cards.cand_parameters import instance as parameter_card
from cards.cand_train import instance as training_card
from evaluation_plan import build_plan
from module import Module
from proxy_data import proxy_data
from roles import RoleMap
from shiny import App, reactive, render, ui

x = np.linspace(-1, 1, 24)
source = proxy_data(pd.DataFrame({'x': x, 'y': 2 * x + .2 * np.sin(3 * x)}),
    _roles=RoleMap.from_primitive({'predictor': ['x'], 'target': ['y']}))
source = source.with_evaluation_plan(build_plan(source, strategy='kfold', objective='independent', folds=3), card='obs_resample')
methods, parameters, training = method_card(), parameter_card(), training_card()
methods._imports.set(source)
training.restore_configuration_state({'inputs': {'Method': 'grid', 'Execution': 'sequential', 'Workers': 1}})
head = ui.head_content(
    [ui.include_js(script, method='inline') for script in dict.fromkeys(Module.script_list)],
    [ui.include_css(style, method='inline') for style in dict.fromkeys(Module.css_list)],
    [ui.tags.script(type='module', src=script) for script in dict.fromkeys(Module.mjs_list)])
app_ui = ui.page_fillable(head, ui.div(methods.call_ui(), parameters.call_ui(), training.call_ui(),
    id='cards-container', class_='cards-grid'), ui.output_text('KernelSpec'), ui.output_text('KernelTrials'))


def server(input, output, session):
    Module.ModSession = session
    method_export = methods.call_server(input, output, session)
    parameter_export = parameters.call_server(input, output, session, upstream=reactive.Value(method_export))
    training_export = training.call_server(input, output, session, upstream=reactive.Value(parameter_export))

    @output
    @render.text
    def KernelSpec():
        spec = parameter_export().tuning_spec
        return str(spec['combinations']) if spec else 'No commitment'

    @output
    @render.text
    def KernelTrials():
        results = training_export().tuning_results
        if results is None:
            return 'No results'
        return ';'.join(sorted({str(t['params'].get('cand_methods__estimator__kernel__metric', 'rbf'))
                               for t in results['state']['trials']}))

    for card in (methods, parameters, training):
        card.resume()
    async def initialize():
        for card in (methods, parameters, training):
            await session.send_custom_message('init_card', {'id': card.ns('Card')})
    session.on_flushed(initialize, once=True)


app = App(app_ui, server, static_assets=Module.ROOT / 'www')
