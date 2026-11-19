"""Exercise real Plotly widget updates and explicit replacement/close."""
import plotly.graph_objects as go
from shiny import App, reactive, ui
from shinywidgets import output_widget, render_widget
from widget_lifecycle import WidgetSlot, update_figure

app_ui = ui.page_fluid(ui.input_action_button('Replace', 'Replace'), ui.input_action_button('Update', 'Update'),
    ui.input_action_button('Close', 'Close'), output_widget('Chart'))

def server(input, output, session):
    slot = WidgetSlot(session)
    @output
    @render_widget
    def Chart():
        input.Replace()
        return slot.replace(go.FigureWidget(go.Scatter(x=[1, 2], y=[3, 4])))
    @reactive.effect
    @reactive.event(input.Update)
    def update():
        if slot.current:
            update_figure(slot.current, go.Figure(go.Scatter(x=[1, 2], y=[4, 5])), {})
    @reactive.effect
    @reactive.event(input.Close)
    def close():
        slot.close()
app = App(app_ui, server)
