"""Single-widget lifecycle reproduction; no Pythagoras application imports.

Run: .venv/bin/shiny run --launch-browser diagnostics/widget_cleanup/app.py
"""
import json

import plotly.graph_objects as go
import shinywidgets
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget

app_ui = ui.page_fluid(
    ui.h3('FigureWidget cleanup reproduction'),
    ui.input_action_button('Update', 'Update existing widget'),
    ui.input_action_button('Replace', 'Replace widget'),
    ui.input_action_button('Remove', 'Remove output'),
    ui.input_action_button('Close', 'Close model'),
    ui.div(output_widget('Chart'), id='chart-host'),
    ui.output_text('Probe'),
)


def server(input, output, session):
    current = {}
    state = reactive.Value({})

    @output
    @render_widget
    def Chart():
        revision = input.Replace()
        widget = go.FigureWidget(go.Scatter(x=[0, 1], y=[revision, revision + 1]))
        current['widget'] = widget
        state.set({'action': 'render', 'revision': revision, 'model_id': widget.model_id})
        return widget

    @reactive.effect
    @reactive.event(input.Update)
    def update():
        current['widget'].data[0].y = [10, 11]
        state.set({'action': 'update', 'model_id': current['widget'].model_id})

    @reactive.effect
    @reactive.event(input.Remove)
    def remove():
        ui.remove_ui('#chart-host')
        output.remove('Chart')
        state.set({'action': 'remove'})

    @reactive.effect
    @reactive.event(input.Close)
    def close():
        current['widget'].close()
        state.set({'action': 'close'})

    @output
    @render.text
    def Probe():
        return json.dumps(dict(state(), server_package=shinywidgets.__file__))


app = App(app_ui, server)
