"""Shared stylesheet behavior for forced, responsive and unclassed navset bars."""
from pathlib import Path

from shiny import App, ui

CSS = Path(__file__).resolve().parents[2] / 'app/www/pythagoras.css'


def bar(identifier, classes):
    return ui.div(ui.navset_bar(
        ui.nav_panel('First', 'First panel'),
        ui.nav_panel('Second tab with a long label', 'Second panel'),
        ui.nav_panel('VeryLongUnbrokenTabLabelForWrappingWithoutOverflow', 'Third panel'),
        id=identifier, title=None, padding=0,
        navbar_options=ui.navbar_options(collapsible=True, underline=True, class_=classes)),
        id=identifier + '-host', style='width: 320px; max-width: 100%;')


app_ui = ui.page_fluid(
    ui.head_content(ui.include_css(CSS)),
    bar('Forced', 'nominal-navbar force-collapse'),
    bar('Responsive', 'nominal-navbar'),
    bar('Unclassed', ''),
)
app = App(app_ui, lambda input, output, session: None)
