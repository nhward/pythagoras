"""Candidate assessment with grouped rows and observable pass-through exports."""
from dataclasses import replace

import cards.cand_assessment as assessment_card
from ensemble import evaluate_ensemble, representatives
from ensemble_scenario import example_source
from evaluation_plan import build_plan
from shiny import reactive, render, ui


def application(task='regression', hard=False, restored=False):
    source = example_source(task, count=2)
    frame = source.clean_frame.assign(stratum=['A', 'B'] * 20, sensitive=['G1'] * 20 + ['G2'] * 20)
    roles = source.role_map.to_primitive()
    roles.update(stratifier=['stratum'], sensitive=['sensitive'])
    from roles import RoleMap
    grouped = replace(source, _df=frame, _clean_df=frame, _roles=RoleMap.from_primitive(roles))
    grouped = grouped.with_evaluation_plan(build_plan(grouped, strategy='kfold', objective='independent', folds=5), card='obs_resample')
    source = grouped.with_tuning_results(source.tuning_results)
    if hard:
        evaluation = evaluate_ensemble(source, tuple(representatives(source)[1]), 'hard')
        source = source.with_ensemble(evaluation)
    this = assessment_card.instance()
    if restored:
        this.restore_configuration_state({'inputs': {'Color': 'stratum', 'Symbol': 'sensitive',
            'Class': '1', 'ChartType': 'confusion' if hard else 'auto', 'Subgroup': 'stratum=A · sensitive=G1'}})
    this._imports.set(source)
    original_server, original_footer = this.server, this._footer
    calls = 0
    original_assess = assessment_card.assess_candidate

    def assess(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original_assess(*args, **kwargs)
    assessment_card.assess_candidate = assess
    this.footer = lambda: ui.TagList(original_footer(), ui.output_text('ExportProbe'), ui.output_text('Calls'),
                                    ui.input_action_button('ClearWinner', label='Clear upstream winner'))

    def server(input, output, session):
        exported = original_server(input, output, session)

        @output
        @render.text
        def ExportProbe():
            return 'unchanged' if exported() is this._imports() else 'changed'

        @output
        @render.text
        def Calls():
            reactive.invalidate_later(.1)
            return f'collections={calls}'

        @reactive.effect
        @reactive.event(input.ClearWinner)
        def clear():
            incoming = this._imports()
            this._imports.set(replace(incoming, _tuning_results=None, _ensemble_results=None))
        return exported
    this.server = server
    return this.application()


app = application()
