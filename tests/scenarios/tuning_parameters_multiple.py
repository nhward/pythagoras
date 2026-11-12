"""Method-specific parameter labels and conditional configuration counts."""
import runpy
from pathlib import Path

scenario = runpy.run_path(str(Path(__file__).with_name('pipeline_tune_multiple.py')))
from cards.tuning_parameters import instance
from shiny import render, ui

this = instance()
this._imports.set(scenario['source'])
original_server, original_footer = this.server, this._footer
this.footer = lambda: ui.TagList(original_footer(), ui.output_text('MultiSpec'))
def server(input, output, session):
    exported = original_server(input, output, session)
    @output
    @render.text
    def MultiSpec():
        spec = exported().tuning_spec
        return str(spec['combinations']) if spec else 'None'
    return exported
this.server = server
app = this.application()
