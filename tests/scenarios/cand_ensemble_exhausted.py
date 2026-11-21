"""Restore exhausted training evidence; ensemble evaluation needs no new trials."""
import runpy
from pathlib import Path

scenario = runpy.run_path(str(Path(__file__).with_name('cand_train_ensemble.py')))
from cards.cand_train import instance as training_card
from cards.cand_models import instance as review_card
from cards.cand_ensemble import instance as ensemble_card
from prediction_evidence import portable_checkpoint
from module import Module
from pipeline_search import SearchWorker, fresh, prepare, run
from shiny import App, reactive, render, ui

source = scenario['source']
problem = prepare(source, method='grid')
checkpoint = run(problem, fresh(problem), trials=0, publish=lambda _: None, pause=lambda: False)
assert checkpoint['status'] == 'exhausted'


def no_training(*args, **kwargs):
    raise AssertionError('Restoring exhausted evidence must not resume training')


SearchWorker.start = no_training


def server(input, output, session):
    Module.ModSession = session
    train, review, ensemble = training_card(), review_card(), ensemble_card()
    train.restore_configuration_state({'inputs': {'Method': 'grid', 'Execution': 'sequential', 'Workers': 1},
        'search': {'checkpoint': portable_checkpoint(checkpoint), 'source_signature': source._tuning_evidence_signature()}})
    train._imports.set(source)
    trained = train.call_server(input, output, session)
    reviewed = review.call_server(input, output, session, upstream=reactive.Value(trained))
    ensemble.call_server(input, output, session, upstream=reactive.Value(reviewed))
    for card in (train, review, ensemble):
        card.resume()

    @output
    @render.ui
    def Workflow():
        return ui.div(train.call_ui(), review.call_ui(), ensemble.call_ui(), class_='cards-grid')


app = App(scenario['app_ui'], server, static_assets=Module.ROOT / 'www')
