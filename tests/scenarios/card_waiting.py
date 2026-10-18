"""Shared card waiting status with a controllable upstream source."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "app"
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import pandas as pd
from card import Card
from proxy_data import proxy_data
from shiny import reactive, render, req, ui

this = Card(file=__file__)
this.front = lambda: ui.TagList(
    ui.input_radio_buttons("Mode", label="Upstream", choices=["pending", "ready", "error"], selected="pending"),
    ui.input_checkbox("Active", label="Active", value=True),
    ui.output_text("Result"),
)
this.footer = lambda: ui.TagList()

def server(input, output, session):
    source = proxy_data(_df=pd.DataFrame({"x": [1]}))

    @reactive.calc
    def upstream():
        mode = input.Mode()
        req(mode == "ready" or mode == "error")
        if mode == "error":
            raise ValueError("upstream failed")
        return source

    this._upstream.set(upstream)

    @reactive.effect
    def active():
        if input.Active():
            this.resume()
        else:
            this.suspend()

    @output
    @render.text
    def Result():
        return f"rows={len(this.input_data().frame)}"


this.server = server
app = this.application()
