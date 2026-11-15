"""Completed, repeated-CV Ridge and tree candidates for interactive review."""
import runpy
from pathlib import Path

scenario = runpy.run_path(str(Path(__file__).with_name('cand_train_multiple.py')))
from cards.model_candidates import instance
from evaluation_plan import build_plan
from model_review import completed_evidence
from pipeline_search import fresh, prepare, run
from shiny import render, ui

source = scenario['source']
source = source.with_evaluation_plan(build_plan(source, strategy='kfold', objective='independent', folds=3, repeats=2), card='obs_resample')
problem = prepare(source, method='grid')
state = run(problem, fresh(problem), trials=0, publish=lambda s: None, pause=lambda: False)
source = source.with_tuning_results(completed_evidence(problem, state))
this = instance()
this.restore_configuration_state({'inputs': {'EquivalenceTolerance': 1000, 'ComparisonLimit': 5}})
this._imports.set(source)
original_server, original_footer = this.server, this._footer
this.footer = lambda: ui.TagList(original_footer(), ui.output_text('ReviewState'))


def server(input, output, session):
    exported = original_server(input, output, session)
    @output
    @render.text
    def ReviewState():
        data = exported()
        records = [r for r in data.processing_records if r.card == 'model_candidates' and r.attempted]
        model = data.pipeline_for_training().get_params()['cand_methods__estimator'] if records else None
        return f'{type(model).__name__};evidence={len(data.tuning_results["state"]["trials"])}' if model is not None else 'No review accepted'
    return exported


this.server = server
app = this.application()
