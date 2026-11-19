"""A late-added output must not replace the manager of an existing chart."""
import plotly.graph_objects as go
from shiny import App, render, ui
from shinywidgets import output_widget, render_widget

early = output_widget('Early')
from widget_lifecycle import WidgetSlot  # noqa: E402

app_ui = ui.page_fluid(ui.input_action_button('Add', 'Add chart'),
    ui.input_action_button('Refresh', 'Refresh charts'), early, ui.output_ui('LateUI'))


def server(input, output, session):
    first, second = WidgetSlot(session), WidgetSlot(session)

    @output
    @render_widget
    def Early():
        revision = input.Refresh()
        return first.replace(go.FigureWidget(go.Scatter(x=[1, 2], y=[revision + 3, 4])))

    @output
    @render.ui
    def LateUI():
        return output_widget('Later') if input.Add() else None

    @output
    @render_widget
    def Later():
        revision = input.Refresh()
        return second.replace(go.FigureWidget(go.Scatter(x=[1, 2], y=[revision + 5, 6])))


app = App(app_ui, server)
