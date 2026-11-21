"""Construct the ensemble server before its UI, as workflow insertion does."""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name('cand_ensemble.py')))
from cards.cand_ensemble import instance
from ensemble_scenario import example_source
from module import Module
from shiny import App, render, ui

app_ui = ui.page_fillable(
    ui.head_content(
        *[ui.include_js(p, method='inline') for p in dict.fromkeys(Module.script_list)],
        *[ui.include_css(p, method='inline') for p in dict.fromkeys(Module.css_list)]),
    ui.output_ui('Workflow'), ui.output_text('ExportProbe'))


def server(input, output, session):
    Module.ModSession = session
    card = instance()
    card._imports.set(example_source(count=5))
    exported = card.call_server(input, output, session)
    card.resume()

    @output
    @render.ui
    def Workflow():
        return card.call_ui()

    @output
    @render.text
    def ExportProbe():
        result = exported().ensemble_results
        return f'ensemble={len(result.members)};fitted={len(result.fitted)}' if result else 'incoming'


app = App(app_ui, server, static_assets=Module.ROOT / 'www')
