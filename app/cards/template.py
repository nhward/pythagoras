"""Runnable starting point for a Pythagoras card; copy and rename this file.

DESIGN CHECKLIST
---------------
* Define the question, intended audience and meaning of each output.
* Choose MUTABLE below: this concerns exported proxy data, not interactive UI.
  Diagnostic cards pass through unchanged; mutable cards return a successor.
* Specify eligible roles AND semantic types. Do not accidentally analyze an
  Identifier, Target, Weighting, geometry or shadow column as a Predictor.
* Decide explicitly about observation weights, missing/nonfinite values,
  constants, duplicate indices, small samples and unsupported types.
* Bound expensive work. Document any sampling, fixed seed and excluded rows.
* Preserve identifiers for investigation; a displayed Row is a one-based
  position in the current input, not necessarily the original source file.
* Separate analysis from plotting so full-screen, tab and display-only changes
  do not retrain models. Pure helpers should be testable without Shiny.
* Explain methods, units, denominators and uncertainty. A ranking is not a
  probability. Make disagreement and unavailable results visible.
* Add markdown/<new_name>.qmd from markdown/template.qmd and render its HTML
  to app/www/markdown/<new_name>.html for the information button.
* Add tests/scenarios/<new_name>.py and tests/cards/test_<new_name>.py.
  Cover role filtering, numerical behavior, unchanged input, learned pipeline
  compatibility, restored inputs, empty data, delayed tasks and card flipping.

The example counts missing cells by variable. It intentionally considers ALL
columns and does not use weights. It defaults to immutable. The optional mutable
example removes completely empty rows BEFORE a learned pipeline has begun.
Replace that example with the intended operation; do not keep it accidentally.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Keep this before local imports: direct execution must find this checkout.
if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import shinywidgets
from card import Card
from module import Module
from proxy_data import proxy_data
from shiny import render, req, ui
from shiny.types import SilentException, SilentOperationInProgressException
from shinywidgets import render_widget

MUTABLE = False  # Change to True only when this card exports modified proxy data.
ACTION = "Remove completely empty rows"  # Example only; replace for a real card.


def _analyze(source: proxy_data) -> pd.DataFrame:
    """Pure analysis: no reactive reads, UI updates or mutation of source here."""
    frame = source.frame
    return pd.DataFrame({
        "Variable": [str(column) for column in frame.columns],
        "Missing cells": frame.isna().sum().to_numpy(),
    }).sort_values("Missing cells", ascending=False, kind="stable")


def _apply(source: proxy_data, *, enabled: bool, card_name: str) -> proxy_data:
    """Example mutable boundary. Never assign into source.frame or its roles."""
    if not enabled or source.has_pipeline:
        return source.with_inactive_step(
            stage="Cleaning", card=card_name, operation=ACTION,
        )
    # Keep indices, column types and metadata. Let proxy_data record the change.
    return source.with_cleaned_data(
        source.frame.dropna(how="all").copy(),
        card=card_name, operation=ACTION,
    )

    # LEARNED TRANSFORM ALTERNATIVE (replace the cleaning example above):
    # Use a sklearn-compatible estimator with fit/transform and cloneable init
    # parameters. Fit only a preview here; the stored pipeline must be refitted
    # within training folds, never reuse full-data learning for validation.
    # return source.with_pipeline_step(
    #     transformer, name=card_name, operation="Describe the learned operation",
    #     preview_frame=preview, added_roles=new_roles,
    #     removed_columns=removed_names,
    # )
    # Supply roles for added columns, avoid naming collisions, handle unseen
    # categories, and preserve row order. Do not replace the incoming pipeline.
    # For metadata-only changes, clone the proxy and use its supported setters.


def instance():
    # The file stem supplies the card identity and namespace. Copy under the
    # intended name; do not manually prefix UI IDs with a namespace.
    this = Card(file=__file__, mutable=MUTABLE)
    this.long_name = "Card template"
    this.description = "Developer example: missing cells by variable."

    this.front = lambda: ui.TagList(
        # Theme primary text and centering come from CSS, not a Plotly color.
        ui.span("Missing cells by variable", class_="text-primary text-center d-block"),
        shinywidgets.output_widget(
            id="Chart", fill=True, guide=this, title="Example chart",
            text="Counts missing cells in each variable. All columns are included; observations are equally weighted.",
            position="left",
        ),
    )
    this.back = lambda: ui.TagList(
        ui.span("All variables", class_="text-primary text-center d-block"),
        # Keep output containers stable when input is temporarily unavailable.
        ui.output_data_frame(
            id="Table", guide=this, title="Example table",
            text="Full counts, including variables not shown in the chart.",
            position="left",
        ),
    )

    if MUTABLE:
        this.footer=lambda: ui.TagList(
            ui.output_ui("Busy"), ui.output_text("Status"),
            ui.input_checkbox_group(
                id="Apply", label=None, choices=[ACTION], selected=[], inline=True,
                guide=this, title="Example mutable action",
                text="Applies to the incoming source. Uncheck to stop applying it. Place this cleaning example before learned transformations.",
                position="top",
            )
        )


    this.settings = lambda: ui.TagList(
        ui.input_slider(
            id="Top", label="Variables displayed", min=1, max=50, value=10, step=1,
            guide=this, text="Limits display only; every variable remains in the table and analysis.",
            position="left",
        ),
    )
    # Stable input IDs support automatic configuration restoration. For dynamic
    # choices use this.restored_configuration_input(...), preserve a valid saved
    # selection and wait for inputs to bind before treating it as invalid.
    # A Navset needs stable panel values; keep unavailable panels explainable.

    def server(input, output, session):
        busy = this.busy()
        analyze = this.record_code(_analyze)

        @this.reactable(calc=True)
        def Incoming():
            try:
                source = this.input_data()
            except SilentOperationInProgressException:
                # Translate upstream task progress to normal output clearing.
                # Let Busy own progress; don't leak a persistent output state.
                req(False)
            req(source is not None)
            return source

        @this.reactable(calc=True)
        @this.settle(seconds=2)
        def Enabled():
            return MUTABLE and ACTION in (input.Apply() or [])
        # Put expensive-analysis settings in a similarly settled Options calc.
        # Read actual controls in the settled function, not in its consumers.
        # Logarithmic sliders return exponents: convert with 10 ** input.Limit().
        # Keep display-only settings out of analysis options.

        @this.reactable(calc=True)
        def Export():
            source = Incoming()
            if not MUTABLE:
                return source
            # Always derive from the current INCOMING source, not the previous
            # exported result; otherwise reactive reruns can apply changes twice.
            return _apply(source, enabled=Enabled(), card_name=this.name)

        # Cheap analyses can be ordinary @this.reactable(calc=True) functions.
        # This extended-task example shows where expensive work belongs.
        # Decorator order matters: busy.track goes ABOVE this.extended_task.
        @busy.track("Calculating example summary…")
        @this.extended_task
        async def Calculate(source):
            # Background work must use arguments, never input.* or reactive reads.
            # Pyodide does not provide ordinary Python worker threads. A direct
            # fallback is adequate only for small work; large browser workloads
            # need a separate design and explicit performance testing.
            table = analyze(source) if Module.IS_SHINYLIVE else await asyncio.to_thread(analyze, source)
            return source, table

        @this.reactable()
        def Start():
            Calculate.cancel()
            Calculate.invoke(Export().clone())
            # Cancelling the coroutine does not stop an existing worker thread.
            # For long work use cooperative cancellation between bounded steps.

        @this.reactable(calc=True)
        def Results():
            try:
                source, table = Calculate.result()
            except SilentOperationInProgressException:
                req(False)  # Normal clear, never a fake successful result.
            req(source.equals(Export()))  # Reject results from an old input.
            # If analysis has options, return and compare those as well.
            return table

        @output
        @render_widget
        def Chart():
            full = bool(this.isFullScreen())
            try:
                table = Results().head(int(input.Top()))
                if table.empty:
                    figure = Card.empty_figure("No variables to display.")
                else:
                    figure = go.Figure(go.Bar(x=table["Variable"], y=table["Missing cells"]))
                    figure.update_layout(
                        title={
                            "text": "my title",
                            'font':{'size':17},
                            'x': 0.5,
                            'xanchor': "center"
                        },
                        template="plotly_white", 
                        xaxis_type="category",
                        xaxis_title="Variable", 
                        yaxis_title="Missing cells",
                        paper_bgcolor="rgba(0,0,0,0)", 
                        margin={"l": 45, "r": 15, "t": 10, "b": 15},
                        plot_bgcolor='#bbd6f8',
                        showlegend=True,
                        modebar={'orientation':'v'},
                        font={"size": 13 if this.FullScreen() else 10},
                    )
            except SilentException:
                figure = Card.empty_figure("Waiting for data or calculation.")
            # Use the framework's empty figure for unavailable plots. Do not
            # swallow real programming errors as if they were an empty dataset.
            widget = go.FigureWidget(figure)
            widget._config = {
                "displayModeBar": full, 
                "displaylogo": False, 
                "responsive": True
            }
            return widget

        @output
        @render.data_frame
        def Table():
            # Let the card own scrolling; avoid nested fixed-height containers.
            return render.DataTable(Results().round(4), width="100%", height=None)

        @output
        @render.ui
        def Busy():
            return busy.ui()

        @output
        @render.text
        def Status():
            if MUTABLE and Enabled() and Incoming().has_pipeline:
                return "Cannot remove rows after learning has begun; move this example before the learned pipeline."
            Results()
            return "Example cleaning enabled." if MUTABLE and Enabled() else "Data passed through unchanged."

        session.on_ended(Calculate.cancel)
        # Return a CALLABLE reactive source, not Export() or a DataFrame.
        # An immutable card's downstream data need not wait for diagnostics.
        return Export if MUTABLE else Incoming

    this.server = server
    return this


# Self-contained data avoids network dependencies and external file paths.
# From the repository: .venv/bin/python app/cards/template.py
if Module.running_directly(name=__name__):
    this = instance()
    frame = pd.DataFrame({"value": [1., None, 3.], "label": ["a", None, "b"]})
    this._imports.set(proxy_data(_df=frame, _name="Template example"))
    this.run()
