"""Persistent, replaced and late Plotly widgets share card presentation rules."""
import pandas as pd
import plotly.graph_objects as go
from card import Card
from proxy_data import proxy_data
from shiny import render, ui
from shinywidgets import output_widget, render_widget
from widget_lifecycle import WidgetSlot, update_figure

card = Card(file=__file__, mutable=False, pipeline_compatible=True)
card._imports.set(proxy_data(pd.DataFrame({'x': [1, 2]})))
card.front = lambda: ui.TagList(
    output_widget('Persistent', height='130px'), output_widget('Replaced', height='130px'), ui.output_ui('Extra'))
card.footer = lambda: ui.TagList(ui.input_action_button('Update', label='Update charts'),
                               ui.input_action_button('Add', label='Add chart'))


def figure(right):
    result = go.Figure(go.Scatter(x=[1, 2], y=[3, 4]))
    result.update_layout(modebar={'orientation': 'h'}, margin={'l': 10, 'r': right, 't': 10, 'b': 20},
                         font={'size': 27, 'family': 'Arial', 'color': '#334155'})
    return result


def server(input, output, session):
    slot = WidgetSlot(session)

    @output
    @render_widget
    def Persistent():
        widget = slot.replace(go.FigureWidget(figure(60)))
        widget._config = {'displayModeBar': True, 'displaylogo': True}
        return widget

    @output
    @render_widget
    def Replaced():
        full = card.isFullScreen()
        input.Update()
        widget = go.FigureWidget(figure(0 if full else 50))
        widget._config = {'displayModeBar': True, 'displaylogo': True}
        return widget

    @card.reactable(triggers=[input.Update])
    async def update():
        widget = slot.current
        if widget is not None:
            config = {'displayModeBar': True, 'displaylogo': True}
            update_figure(widget, figure(0), config)
            await session.send_custom_message('update_plotly_config', {'id': card.ns('Persistent'), 'config': config})

    @output
    @render.ui
    def Extra():
        return output_widget('Late', height='130px') if input.Add() else None

    @output
    @render_widget
    def Late():
        return go.FigureWidget(figure(70))


card.server = server
app = card.application()
