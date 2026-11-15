"""Real regression evidence with deterministic trade-offs for candidate selection."""
import runpy
from pathlib import Path

scenario = runpy.run_path(str(Path(__file__).with_name('model_candidates.py')))
from cards.cand_models import instance
from shiny import render, ui

source = scenario['source']
results = source.tuning_results
# Keep real fitted-CV scores; make timing/risk differences deterministic for UI assertions.
for i, trial in enumerate(results['state']['trials']):
    for fold in trial['folds']:
        fold['evidence']['prediction_seconds'] = fold['evidence']['count'] / (100 + i * 100)
    trial['diagnostics']['p95_absolute_error'] = 20 - i
source = source.with_tuning_results(results)
this = instance()
this._imports.set(source)
original_server, original_footer = this.server, this._footer
this.footer = lambda: ui.TagList(original_footer(), ui.output_text('SelectionState'))


def server(input, output, session):
    exported = original_server(input, output, session)
    @output
    @render.text
    def SelectionState():
        data = exported()
        review = data.tuning_results.get('review', {})
        return f'overall={review.get("accepted_trial")};methods={review.get("representatives")};records={[r.card for r in data.processing_records if r.card == "cand_models" and r.attempted]}'
    return exported


this.server = server
app = this.application()
