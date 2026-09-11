from __future__ import annotations

import os
import sys
from pathlib import Path

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    # Ensure local modules and packages are resolved from the app directory.
    os.chdir(ROOT)
    root_string = str(ROOT)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)

import json
import re
import tempfile
from collections.abc import Callable, Mapping
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from card import Card
from faicons import icon_svg as icon
from module import Module
from shiny import reactive, render, req, ui

BOOKMARK_SUFFIX = ".pythagoras.json"
BOOKMARK_DIRECTORY_ENV = "PYTHAGORAS_BOOKMARK_DIR"
TEST_BOOKMARK_DIRECTORY_ENV = "PYTHAGORAS_TEST_BOOKMARK_DIR"


def bookmark_directory() -> Path:
    """Return the directory used for immutable bookmarks in local mode."""
    configured = os.environ.get(BOOKMARK_DIRECTORY_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    if os.environ.get("SHINY_TESTMODE") == "1":
        test_directory = os.environ.get(TEST_BOOKMARK_DIRECTORY_ENV)
        if test_directory:
            return Path(test_directory).expanduser().resolve()
        return Path(tempfile.gettempdir()) / f"pythagoras-bookmarks-{os.getpid()}"
    return Path.home() / "Documents" / "Pythagoras" / "Bookmarks"


def data_source_name(configuration: Mapping[str, object]) -> str:
    """Derive a filesystem-safe bookmark stem from data-import state."""
    for group in configuration.get("layout", []):
        if not isinstance(group, Mapping):
            continue
        for card in group.get("cards", []):
            if not isinstance(card, Mapping) or card.get("module") != "data_import":
                continue
            state = card.get("state")
            if not isinstance(state, Mapping):
                continue
            inputs = state.get("inputs")
            if not isinstance(inputs, Mapping):
                continue
            committed = state.get("last_committed_tab")
            input_name = {
                "File based": "FName",
                "Dataset based": "DName",
                "Web based": "UName",
                "UC Irvine": "IName",
            }.get(committed)
            value = inputs.get(input_name) if input_name else None
            if isinstance(value, str) and value.strip():
                return value.strip()
    return "analysis"


def bookmark_filename(
    configuration: Mapping[str, object],
    created_at: datetime,
) -> str:
    """Return a portable filename based on data name and creation time."""
    slug = re.sub(r"[^A-Za-z0-9]+", "-", data_source_name(configuration))
    slug = slug.strip("-") or "analysis"
    timestamp = created_at.strftime("%Y%m%dT%H%M%S%z")
    return f"{slug}--{timestamp}{BOOKMARK_SUFFIX}"


def split_bookmark_filename(filename: str) -> tuple[str, str]:
    """Return the data-name and timestamp components of a bookmark filename."""
    if Path(filename).name != filename or not filename.endswith(BOOKMARK_SUFFIX):
        raise ValueError("Invalid bookmark filename")
    stem = filename[: -len(BOOKMARK_SUFFIX)]
    try:
        data_name, timestamp = stem.rsplit("--", 1)
    except ValueError as error:
        raise ValueError("Invalid bookmark filename") from error
    if not data_name or not timestamp:
        raise ValueError("Invalid bookmark filename")
    return data_name, timestamp


def bookmark_filename_from_parts(data_name: str, timestamp: str) -> str:
    """Reconstruct a validated bookmark filename from selector values."""
    filename = f"{data_name}--{timestamp}{BOOKMARK_SUFFIX}"
    parsed_data_name, parsed_timestamp = split_bookmark_filename(filename)
    if parsed_data_name != data_name or parsed_timestamp != timestamp:
        raise ValueError("Invalid bookmark selection")
    return filename


def format_bookmark_time(timestamp: str, created: float) -> str:
    """Format a bookmark timestamp using the host's locale conventions."""
    try:
        value = datetime.strptime(timestamp, "%Y%m%dT%H%M%S%z").astimezone()
    except ValueError:
        value = datetime.fromtimestamp(created).astimezone()
    return value.strftime("%c")


def with_bookmark_metadata(
    configuration: Mapping[str, object],
    *,
    filename: str,
    created_at: datetime,
) -> dict[str, object]:
    candidate = deepcopy(dict(configuration))
    candidate["bookmark"] = {
        "filename": filename,
        "created_at": created_at.astimezone().isoformat(),
    }
    return candidate


def filesystem_creation_time(path: Path) -> float:
    """Return birth time where available, with ctime as the Unix fallback."""
    status = path.stat()
    return float(getattr(status, "st_birthtime", status.st_ctime))


def list_local_bookmarks(directory: Path | None = None) -> list[dict[str, object]]:
    """List local bookmarks newest-first using filesystem creation time."""
    directory = Path(directory or bookmark_directory())
    if not directory.is_dir():
        return []
    records = []
    for path in directory.glob(f"*{BOOKMARK_SUFFIX}"):
        if not path.is_file():
            continue
        created = filesystem_creation_time(path)
        try:
            data_name, timestamp = split_bookmark_filename(path.name)
        except ValueError:
            records.append(
                {
                    "filename": path.name,
                    "data_name": None,
                    "timestamp": None,
                    "display_time": None,
                    "created": created,
                    "path": path,
                }
            )
            continue
        records.append(
            {
                "filename": path.name,
                "data_name": data_name,
                "timestamp": timestamp,
                "display_time": format_bookmark_time(timestamp, created),
                "created": created,
                "path": path,
            }
        )
    return sorted(records, key=lambda item: item["created"], reverse=True)


def load_local_bookmark(
    filename: str,
    *,
    directory: Path | None = None,
) -> dict[str, object]:
    """Read one safely named local bookmark."""
    if Path(filename).name != filename or not filename.endswith(BOOKMARK_SUFFIX):
        raise ValueError("Invalid bookmark filename")
    path = Path(directory or bookmark_directory()) / filename
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("A bookmark must contain a JSON object")
    return value


def latest_local_bookmark(
    *,
    directory: Path | None = None,
    validator: Callable[[Mapping[str, object]], None] | None = None,
) -> dict[str, object] | None:
    """Return the newest valid local bookmark, skipping invalid files."""
    for record in list_local_bookmarks(directory):
        try:
            configuration = load_local_bookmark(
                str(record["filename"]), directory=directory
            )
            if validator is not None:
                validator(configuration)
            return configuration
        except Exception:  # noqa: BLE001, S112
            continue
    return None


def save_local_bookmark(
    configuration: Mapping[str, object],
    *,
    directory: Path | None = None,
) -> Path:
    """Create a new bookmark without replacing an existing revision."""
    metadata = configuration.get("bookmark")
    if not isinstance(metadata, Mapping):
        raise ValueError("Bookmark metadata is missing")  # noqa: TRY004
    filename = metadata.get("filename")
    if not isinstance(filename, str):
        raise ValueError("Bookmark filename is missing")  # noqa: TRY004
    directory = Path(directory or bookmark_directory())
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    created = False
    try:
        with path.open("x", encoding="utf-8") as stream:
            created = True
            json.dump(configuration, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
    except Exception:
        if created:
            path.unlink(missing_ok=True)
        raise
    return path


def instance(
    *,
    configuration_provider: Callable[[], Mapping[str, object]] | None = None,
    configuration_validator: Callable[[Mapping[str, object]], None] | None = None,
):
    """Create the application-level bookmark manager card."""
    this = Card(
        file=__file__,
        mutable=False,
        allow_remove=False,
        allow_drag=False,
    )
    this.long_name = "Bookmarks"
    this.description = "Save and restore analysis bookmarks."
    catalogue_version = reactive.Value(0)

    def front():
        return ui.TagList(
            ui.tags.br(),
            ui.output_text("StorageMode"),
            ui.output_ui("BookmarkDataChooser"),
            ui.output_ui("BookmarkTimeChooser"),
            ui.output_ui("BookmarkImport"),
        )

    this.front = front

    def back():
        return ui.TagList(
            ui.p(
                "Bookmarks restore the saved data-import card state. Other "
                "card restoration will be added later."
            ),
            ui.output_text_verbatim("StorageLocation"),
        )

    this.back = back

    def footer():
        return ui.div(
            ui.input_action_button(
                id="SaveBookmark",
                label="Save bookmark",
                icon=icon("bookmark", title="Save bookmark", a11y="sem"),
                class_="btn btn-primary",
            ),
            ui.input_action_button(
                id="LoadBookmark",
                label="Load selected",
                icon=icon("folder-open", title="Load bookmark", a11y="sem"),
                class_="btn btn-secondary",
            ),
            ui.input_action_button(
                id="CloseManager",
                label="Close",
                class_="btn btn-light",
            ),
            class_="d-flex justify-content-center gap-2",
        )

    this.footer = footer

    def server(input, output, session):

        def validate(configuration):
            if configuration_validator is not None:
                configuration_validator(configuration)

        @output
        @render.text
        def StorageMode():
            mode = Module.runtime_mode(session)
            if mode == "local":
                return "Local bookmark files"
            return "Browser bookmark entries"

        @output
        @render.text
        def StorageLocation():
            if Module.runtime_mode(session) == "local":
                return str(bookmark_directory())
            return "Configurations are catalogued within this browser and are also downloaded as JSON files when saved."

        @reactive.calc
        def BookmarkRecords():
            catalogue_version()
            if Module.runtime_mode(session) == "local":
                return list_local_bookmarks()
            browser_records = input.BrowserBookmarks()
            if not isinstance(browser_records, (list, tuple)):
                return []
            return [
                {
                    "filename": str(record["filename"]),
                    "data_name": str(record["dataName"]),
                    "timestamp": str(record["timestamp"]),
                    "display_time": str(record["displayTime"]),
                    "created": float(record.get("createdAt", 0)),
                }
                for record in browser_records
                if isinstance(record, Mapping)
                and record.get("filename")
                and record.get("dataName")
                and record.get("timestamp")
                and record.get("displayTime")
            ]

        @output
        @render.ui
        def BookmarkDataChooser():
            names = list(
                dict.fromkeys(
                    str(record["data_name"])
                    for record in BookmarkRecords()
                    if record.get("data_name")
                )
            )
            return ui.input_select(
                id="SelectedDataName", label="Data name", choices={name: name for name in names}, selected=names[0] if names else None,
                guide=this, position="bottom", text="This dialogue selects the recent saved data names. They are in descending age order."
            )

        @output
        @render.ui
        def BookmarkTimeChooser():
            records = BookmarkRecords()
            selectable_records = [
                record
                for record in records
                if record.get("data_name")
                and record.get("timestamp")
                and record.get("display_time")
            ]
            if not selectable_records:
                choices = {}
            else:
                selected_data = input.SelectedDataName()
                available_names = {
                    str(record["data_name"]) for record in selectable_records
                }
                if selected_data not in available_names:
                    selected_data = str(selectable_records[0]["data_name"])
                choices = {
                    str(record["timestamp"]): str(record["display_time"])
                    for record in selectable_records
                    if record["data_name"] == selected_data
                }
            timestamps = list(choices)
            return ui.input_select(
                id="SelectedBookmarkTime", label="Saved at", choices=choices, selected=timestamps[0] if timestamps else None,
                guide=this, position="bottom", text="This selects the version of the bookmark. Versions are timestamps with the most recent at the top."
            )

        @output
        @render.ui
        def BookmarkImport():
            if Module.runtime_mode(session) == "local":
                return None
            return ui.TagList(
                ui.input_file(
                    id="ImportBookmark", label="Import a configuration file", accept=[".json"], multiple=False,
                    guide=this, position="left", text="Locates a previously exported bookmark that has been stored locally."
                ),
                ui.input_action_button(
                    id="ImportSelected", label="Import and load", class_="btn btn-outline-primary btn-sm",
                    guide=this, position="left", text="Imports and loads the located bookmark."
                ),
            )

        @reactive.effect
        @reactive.event(input.SaveBookmark)
        async def SaveBookmark():
            if configuration_provider is None:
                ui.notification_show(
                    "No configuration provider is available.", type="error"
                )
                return
            try:
                created_at = datetime.now().astimezone()
                snapshot = dict(configuration_provider())
                filename = bookmark_filename(snapshot, created_at)
                candidate = with_bookmark_metadata(
                    snapshot,
                    filename=filename,
                    created_at=created_at,
                )
                validate(candidate)
                if Module.runtime_mode(session) == "local":
                    path = save_local_bookmark(candidate)
                    catalogue_version.set(catalogue_version() + 1)
                    ui.notification_show(
                        f"Bookmark saved as {path.name}.", type="message"
                    )
                    return
                await session.send_custom_message(
                    "bookmark_save",
                    {
                        "filename": filename,
                        "createdAt": created_at.timestamp(),
                        "configuration": candidate,
                        "listInputId": session.ns("BrowserBookmarks"),
                        "operationInputId": session.ns("BrowserOperation"),
                    },
                )
            except Exception as error:  # noqa: BLE001
                ui.notification_show(
                    f"Bookmark was not saved: {error}",
                    type="error",
                    duration=None,
                )

        async def reload_configuration(configuration):
            validate(configuration)
            await session.send_custom_message(
                "bookmark_reload_configuration",
                {"configuration": configuration},
            )

        @reactive.effect
        @reactive.event(input.LoadBookmark)
        async def LoadBookmark():
            data_name = input.SelectedDataName()
            timestamp = input.SelectedBookmarkTime()
            req(data_name, timestamp)
            try:
                selected = bookmark_filename_from_parts(data_name, timestamp)
                if Module.runtime_mode(session) == "local":
                    await reload_configuration(load_local_bookmark(selected))
                else:
                    await session.send_custom_message(
                        "bookmark_load",
                        {
                            "filename": selected,
                            "operationInputId": session.ns("BrowserOperation"),
                        },
                    )
            except Exception as error:  # noqa: BLE001
                ui.notification_show(
                    f"Bookmark could not be loaded: {error}",
                    type="error",
                    duration=None,
                )

        @reactive.effect
        @reactive.event(input.ImportSelected)
        async def ImportSelected():
            files = input.ImportBookmark()
            req(files)
            try:
                configuration = json.loads(
                    Path(files[0]["datapath"]).read_text(encoding="utf-8")
                )
                if not isinstance(configuration, dict):
                    raise TypeError("A bookmark must contain a JSON object")
                validate(configuration)
                metadata = configuration.get("bookmark")
                imported_at = datetime.now().astimezone()
                if not isinstance(metadata, Mapping):
                    filename = bookmark_filename(configuration, imported_at)
                    configuration = with_bookmark_metadata(
                        configuration,
                        filename=filename,
                        created_at=imported_at,
                    )
                await session.send_custom_message(
                    "bookmark_import",
                    {
                        "filename": configuration["bookmark"]["filename"],
                        "createdAt": imported_at.timestamp(),
                        "configuration": configuration,
                        "operationInputId": session.ns("BrowserOperation"),
                    },
                )
            except Exception as error:  # noqa: BLE001
                ui.notification_show(
                    f"Bookmark could not be imported: {error}",
                    type="error",
                    duration=None,
                )

        @reactive.effect
        @reactive.event(input.BrowserOperation)
        def BrowserOperation():
            operation = input.BrowserOperation()
            if not isinstance(operation, Mapping):
                return
            if operation.get("ok"):
                ui.notification_show(str(operation.get("message")), type="message")
            else:
                ui.notification_show(str(operation.get("message")), type="error", duration=None)

        @reactive.effect
        @reactive.event(input.CloseManager)
        def CloseManager():
            ui.modal_remove()

    this.server = server
    return this


if Module.running_directly(name=__name__):
    this = instance()
    this.run()
