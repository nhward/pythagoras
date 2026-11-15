"""A file-free completed archive reviewed under changed search settings."""
import runpy
from pathlib import Path

scenario = runpy.run_path(str(Path(__file__).with_name('cand_train_repeated.py')))
from pipeline_search import fresh, prepare, run
from prediction_evidence import portable_checkpoint
from shiny import render, ui

this, source = scenario['this'], scenario['source']
problem = prepare(source, method='grid', seed=1729)
state = run(problem, fresh(problem), trials=0, publish=lambda s: None, pause=lambda: False)
this.restore_configuration_state({'inputs': {'Method': 'grid', 'Seed': 1730, 'Execution': 'sequential', 'Workers': 1},
    'search': {'checkpoint': portable_checkpoint(state), 'source_signature': source._tuning_evidence_signature()}})
original_server, original_footer = this.server, this._footer
this.footer = lambda: ui.TagList(original_footer(), ui.output_text('HistoricalState'))


def server(input, output, session):
    exported = original_server(input, output, session)
    @output
    @render.text
    def HistoricalState():
        results = exported().tuning_results
        return f'{len(results["state"]["trials"])} completed · readonly={results.get("read_only", False)}' if results else 'No historical results'
    return exported


this.server = server
app = this.application()
