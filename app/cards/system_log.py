from __future__ import annotations

import os
import sys
from pathlib import Path

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    os.chdir(ROOT)
    root_string = str(ROOT)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)

import pandas as pd
from card import Card
from faicons import icon_svg as icon
from module import Module
from shiny import reactive, render, ui

LOG_COLUMNS = ["Time", "Level", "Logger", "Message", "Source", "Line", "Thread"]
LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
MESSAGE_COLUMN_STYLE = {
    "cols": "Message",
    "style": {
        "width": "50%",
        "min-width": "30rem",
        "white-space": "normal",
    },
}


def _log_frame(
    records: list[dict[str, object]],
    *,
    levels: list[str] | tuple[str, ...] | None = None,
    query: str = "",
    maximum: int = 1_000,
) -> pd.DataFrame:
    """Return the newest matching application records as a DataFrame."""
    frame = pd.DataFrame.from_records(records, columns=LOG_COLUMNS)
    if frame.empty:
        return frame

    frame["Source"] = frame["Source"].map(
        lambda source: Path(str(source)).name if pd.notna(source) else ""
    )

    selected_levels = set(levels or ())
    if selected_levels:
        frame = frame.loc[frame["Level"].isin(selected_levels)]
    else:
        frame = frame.iloc[0:0]

    search = query.strip()
    if search and not frame.empty:
        searchable = frame[["Level", "Logger", "Message", "Source", "Thread"]]
        matches = searchable.fillna("").astype(str).apply(
            lambda column: column.str.contains(search, case=False, regex=False)
        ).any(axis=1)
        frame = frame.loc[matches]

    maximum = max(1, int(maximum))
    frame = frame.tail(maximum).iloc[::-1].reset_index(drop=True)
    if not frame.empty:
        frame["Time"] = pd.to_datetime(frame["Time"]).dt.strftime(
            "%Y-%m-%d %H:%M:%S.%f"
        ).str[:-3]
    return frame


def instance():
    """Create the immutable application-log card."""
    this = Card(file=__file__, mutable=False)
    this.long_name = "System log"
    this.description = "This card lists recent application and card log records, with filtering and text search."

    def front():
        return ui.output_data_frame(
            id="LogTable",
            guide=this,
            title="Application log",
            text=(
                "Recent application records, newest first. Use the column "
                "filters for additional filtering."
            ),
            position="left",
        )

    this.front = front
   
    def footer():
        return ui.div(
            ui.output_text("Status"),
            ui.input_action_button(
                id="Refresh",
                label="Refresh",
                icon=icon("arrows-rotate", title="Refresh the log", a11y="sem"),
                width="250px",
                class_="btn rounded-pill btn-sm d-block mx-auto btn-primary",
                style="border: 0px; box-shadow: none;",
                guide=this,
                title="Refresh button",
                text="Refresh the application-log listing.",
                position="top",
            ),
            class_="text-center",
        )

    this.footer = footer

    def settings():
        return ui.TagList(
            ui.input_checkbox_group(
                id="Levels",
                label="Log levels",
                choices=LOG_LEVELS,
                selected=LOG_LEVELS,
                inline=True,
                guide=this,
                title="Log levels",
                text="Select the severity levels included in the table.",
                position="left",
            ),
            ui.input_text(
                id="Search",
                label="Search",
                placeholder="Logger, message, source, or thread",
                guide=this,
                title="Search log records",
                text="Case-insensitive plain-text search across the log fields.",
                position="left",
            ),
            ui.input_numeric(
                id="Maximum",
                label="Maximum records",
                value=1_000,
                min=1,
                max=5_000,
                step=100,
                guide=this,
                title="Maximum records",
                text="Limit the number of matching records shown, newest first.",
                position="left",
            ),
            ui.input_checkbox(
                id="AutoRefresh",
                label="Refresh automatically",
                value=False,
                guide=this,
                title="Automatic refresh",
                text="Refresh the table every five seconds while this card is active.",
                position="left",
            ),
        )

    this.settings = settings


    def server(input, output, session):
        @this.suspendable(calc=True)
        def LogFrame():
            input.Refresh()
            if input.AutoRefresh():
                reactive.invalidate_later(5)
            return _log_frame(
                Module.log_handler.snapshot(),
                levels=list(input.Levels() or ()),
                query=input.Search() or "",
                maximum=int(input.Maximum() or 1000),
            )

        @output
        @render.data_frame
        def LogTable():
            return render.DataGrid(
                LogFrame(),
                filters=True,
                summary=True,
                width="100%",
                styles=[MESSAGE_COLUMN_STYLE],
            )

        @output
        @render.text
        def Status():
            count = len(LogFrame())
            noun = "record" if count == 1 else "records"
            return f"Showing {count:,} {noun}"

    this.server = server
    return this


if Module.running_directly(name =__name__):
    this = instance()
    this.run()
