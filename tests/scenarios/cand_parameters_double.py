"""Two parameter cards share data but keep row events and drafts independent."""
import runpy
from pathlib import Path

from cards.cand_parameters import instance
from module import Module
from shiny import App, ui

scenario = runpy.run_path(str(Path(__file__).with_name('cand_parameters.py')))
first, second = scenario['this'], instance()
second._imports.set(scenario['source'])
head = ui.head_content(
    [ui.include_js(script, method='inline') for script in dict.fromkeys(Module.script_list)],
    [ui.include_css(style, method='inline') for style in dict.fromkeys(Module.css_list)],
    [ui.tags.script(type='module', src=script) for script in dict.fromkeys(Module.mjs_list)])
app_ui = ui.page_fillable(head, ui.div(first.call_ui(), second.call_ui(), id='cards-container', class_='cards-grid'))


def server(input, output, session):
    Module.ModSession = session
    first.call_server(input, output, session)
    second.call_server(input, output, session)
    for card in (first, second):
        card.resume()
    async def initialize():
        for card in (first, second):
            await session.send_custom_message('init_card', {'id': card.ns('Card')})
    session.on_flushed(initialize, once=True)


app = App(app_ui, server, static_assets=Module.ROOT / 'www')
