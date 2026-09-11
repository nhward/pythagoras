from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path

if __name__ == "__main__":
    ROOT=Path(__file__).resolve().parent.parent
    # Ensure local modules and packages are resolved from the app directory.
    os.chdir(ROOT)
    root_string=str(ROOT)
    if root_string not in sys.path:
        sys.path.insert(0, root_string)

import io
from collections.abc import Mapping
from contextlib import redirect_stdout
from urllib.parse import urlparse

import geopandas as gpd
import pandas as pd
import requests
import seaborn  # noqa: F401
import sklearn  # noqa: F401
import statsmodels  # noqa: F401
import vega_datasets  # noqa: F401
import xarray as xr
from card import Card
from faicons import icon_svg as icon
from module import Module
from proxy_data import proxy_data as Pxy
from shiny import reactive, render, req, ui
from shiny.types import SilentException
from ucimlrepo import fetch_ucirepo, list_available_datasets

import cards  # noqa: F401

# TODO: cleanup exception handling
# TODO: allow URL load to unzip zip files (unlikely to resolve multiple files except shp,shx,prj)

def capture_output(function, *args, **kwargs) -> str:
    buffer=io.StringIO()
    with redirect_stdout(buffer):
        function(*args, **kwargs)
    return buffer.getvalue()


class NativeFilePickerUnavailable(RuntimeError):
    """Raised when the local host has no supported graphical file picker."""


def native_file_picker_backend() -> tuple[str, str] | None:
    """Return the native picker name and executable available on this host."""
    if sys.platform == "darwin":
        executable = "/usr/bin/osascript"
        if os.path.isfile(executable) and os.access(executable, os.X_OK):
            return "osascript", executable
        return None

    if sys.platform == "win32":
        for executable_name in (
            "powershell.exe",
            "pwsh.exe",
            "powershell",
            "pwsh",
        ):
            executable = shutil.which(executable_name)
            if executable:
                return "powershell", executable
        return None

    if sys.platform.startswith("linux"):
        if not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
            return None
        for picker in ("zenity", "kdialog"):
            executable = shutil.which(picker)
            if executable:
                return picker, executable
    return None


def native_file_picker_available() -> bool:
    """Report whether this Python host can display a supported file picker."""
    return native_file_picker_backend() is not None


async def choose_local_file(initial_directory: Path | None = None) -> str:
    """Open the host's native picker asynchronously and return a full path."""
    backend = native_file_picker_backend()
    if backend is None:
        raise NativeFilePickerUnavailable(
            "no supported graphical file picker is available on this host"
        )
    picker, executable = backend
    directory = str((initial_directory or Path.home()).expanduser().resolve())
    environment = None
    if picker == "osascript":
        script = """
        on run argv
            set initialFolder to POSIX file (item 1 of argv)
            set selectedFile to choose file with prompt "Choose a data file" default location initialFolder
            return POSIX path of selectedFile
        end run
        """.strip()
        command = (executable, "-e", script, directory)
    elif picker == "powershell":
        script = r"""
        [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
        Add-Type -AssemblyName System.Windows.Forms
        $dialog = New-Object System.Windows.Forms.OpenFileDialog
        $dialog.Title = 'Choose a data file'
        $dialog.InitialDirectory = [Environment]::GetEnvironmentVariable('PYTHAGORAS_FILE_PICKER_DIRECTORY')
        $dialog.Multiselect = $false
        try {
            if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
                [Console]::Out.Write($dialog.FileName)
            }
        } finally {
            $dialog.Dispose()
        }
        """.strip()
        command = (
            executable,
            "-NoProfile",
            "-NonInteractive",
            "-STA",
            "-Command",
            script,
        )
        environment = os.environ.copy()
        environment["PYTHAGORAS_FILE_PICKER_DIRECTORY"] = directory
    elif picker == "zenity":
        command = (
            executable,
            "--file-selection",
            "--title=Choose a data file",
            f"--filename={directory}{os.sep}",
        )
    else:
        command = (
            executable,
            "--getopenfilename",
            directory,
            "--title",
            "Choose a data file",
        )
    process = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=environment,
    )
    stdout, stderr = await process.communicate()
    selected = stdout.decode(errors="replace").strip()
    error_message = stderr.decode(errors="replace").strip()
    if not selected and process.returncode == 0:
        return ""
    if not selected and process.returncode == 1:
        if picker in {"zenity", "kdialog"}:
            return ""
        if picker == "osascript" and (
            "(-128)" in error_message
            or "user canceled" in error_message.lower()
            or "user cancelled" in error_message.lower()
        ):
            return ""
    if process.returncode != 0:
        raise RuntimeError(error_message or f"{picker} exited with status {process.returncode}")
    return str(Path(selected).expanduser().resolve())


def instance():
    """
    Creates an instance of Card configured as "dataImport".
    """
    this=Card(file=__file__, mutable=True) # "mutable" means it can change the pxd - probably with a commit button
    this.long_name="Data import"
    this.description="This card facilitates the ingestion of data, be it numeric, categorical, textual, temporal or spatial."
    this.requires_import=False
    this._restored_configuration_state = {}

    def restore_configuration_state(state: Mapping[str, object]) -> None:
        """Attach validated, JSON-compatible state before the card UI is built."""
        if not isinstance(state, Mapping):
            raise TypeError("Data-import card state must be an object")
        inputs = state.get("inputs", {})
        if not isinstance(inputs, Mapping):
            raise TypeError("Data-import input state must be an object")
        this._restored_configuration_state = {
            "inputs": dict(inputs),
            "last_committed_tab": state.get("last_committed_tab"),
        }

    this.restore_configuration_state = restore_configuration_state

    def restored_input(name, default=None):
        inputs = this._restored_configuration_state.get("inputs", {})
        return inputs.get(name, default)

    def _load_sm(name):
        """
        Robust loader for statsmodels datasets.
        Strategy:
        1) Try sm.datasets.get_rdataset(name, cache=True).data   # Rdatasets (R datasets)
        2) Try to find a module in sm.datasets.__all__ matching `name`
            and call its load_pandas() or load() to get the DataFrame.
        Returns a pandas.DataFrame on success or raises ValueError.
        """
        import importlib

        import statsmodels.api as sm
        last_exc=None
        # 1) Rdatasets route: sm.datasets.get_rdataset(...).data
        try:
            rd=sm.datasets.get_rdataset(name, cache=True)
            if hasattr(rd, "data"):
                return rd.data
            # if it didn't have .data for some reason, try to return rd itself
            return rd
        except Exception as e:  # noqa: BLE001
            last_exc=e
        # 2) Try built-in statsmodels dataset modules (e.g., sm.datasets.co2)
        try:
            dataset_modules=getattr(sm.datasets, "__all__", []) or []
            # search for matching module name (case-insensitive)
            for mod_name in dataset_modules:
                if mod_name.lower() == name.lower():
                    mod=importlib.import_module(f"statsmodels.datasets.{mod_name}")
                    # common loader patterns
                    if hasattr(mod, "load_pandas"):
                        loaded=mod.load_pandas()
                        return getattr(loaded, "data", loaded)
                    if hasattr(mod, "load"):
                        loaded=mod.load()
                        return getattr(loaded, "data", loaded)
                    # some modules may expose a top-level variable `data` or similar
                    if hasattr(mod, "data"):
                        return mod.data
            # as a last attempt, check for attribute directly on sm.datasets
            mod_obj=getattr(sm.datasets, name, None)
            if mod_obj:
                if hasattr(mod_obj, "load_pandas"):
                    loaded=mod_obj.load_pandas()
                    return getattr(loaded, "data", loaded)
                if hasattr(mod_obj, "load"):
                    loaded=mod_obj.load()
                    return getattr(loaded, "data", loaded)
        except Exception as e:  # noqa: BLE001
            last_exc=e
        # Give a helpful message including the last exception
        raise ValueError(
            f"Statsmodels: could not load dataset '{name}'. "
            f"Tried get_rdataset() and internal sm.datasets modules. Last error: {last_exc}"
            )


    DATA_SOURCES={
        "seaborn": {
            "fetch": lambda: __import__("seaborn").get_dataset_names(),
            "load":  lambda name: __import__("seaborn").load_dataset(name)
        },
        "xarray": {
            # "fetch": lambda: sorted(getattr(__import__("xarray").tutorial, "DATASETS", {}).keys()),
            "fetch": lambda: ['air_temperature', 'rasm'],
            "load":  lambda name: __import__("xarray").tutorial.load_dataset(name)
        },
        "sklearn": {
            "fetch": lambda: ["iris", "digits", "wine", "breast_cancer"],
            "load":  lambda name: getattr(__import__("sklearn").datasets, f"load_{name}")(as_frame=True).frame
        },
        "vega_datasets": {
            "fetch": lambda: __import__("vega_datasets").data.list_datasets(),
            "load":  lambda name: (
                __import__("vega_datasets").data(name)()
                if callable(__import__("vega_datasets").data(name))
                else __import__("vega_datasets").data(name)
            )
        },
        "statsmodels": {
            "fetch": lambda: [
                m.split(".")[-1]
                for m in __import__("statsmodels.api").datasets.__all__
            ],
            # "load":  lambda name: __import__("statsmodels.api").datasets.get_rdataset(name, cache=True).data
            "load":  _load_sm
        },
    }


    def DatasetChoices():
        choices={}
        for pkg, source in DATA_SOURCES.items():
            result=source["fetch"]()
            inner={f"{pkg}::{ds}": ds for ds in result}  # inner dict: {value: label}
            choices[f"_______Package {pkg}_______"]=inner
        return choices

    def UciChoices():
        try:
            text=capture_output(list_available_datasets)
        except:  # noqa: E722
            text=""
        names=[]
        for line in text.splitlines():
            parts=line.rsplit(maxsplit=1)
            if len(parts) == 2 and parts[1].isdigit():
                names.append(parts[0].strip())
        return names

    def front():
        dataset_choices = DatasetChoices()
        restored_dataset = restored_input("Dataset")
        if restored_dataset and not any(restored_dataset in choices for choices in dataset_choices.values()):
            dataset_choices["_______Saved selection_______"] = {restored_dataset: restored_dataset}
        panels=[
            ui.nav_panel(
                "File based",
                ui.tags.br(),
                ui.output_ui(id="FilePickerUI"),
                ui.input_text(
                    id="LocalFilePath", label=None, value=restored_input("LocalFilePath", "")
                ),
                ui.input_text(id="FName", label="Short name", value=restored_input("FName", ""), guide=this, position="bottom", update_on="blur",
                    text='This is how you choose to name the dataset. Keep this name short. By default it is initially populated with the file name. Each of the importation styles has this field.')
            ),
            ui.nav_panel(
                "Dataset based",
                ui.tags.br(),
                ui.input_selectize(id="Dataset", label="Package dataset", multiple=False, width="80%", choices=dataset_choices,
                selected=restored_dataset,
                guide=this, text="Lists example datasets grouped by their supplying Python package. Selecting one creates a preview; downstream data changes only after Commit Import is clicked."),
                ui.input_text(id="DName", label="Short name", value=restored_input("DName", ""), guide=this, position="bottom",
                    text='This is how you choose to name the dataset. Keep this name short. By default, it is initially populated with the chosen dataset name. Each of the importation styles has this field.')
            ),
            ui.nav_panel(
                "Web based",
                ui.tags.br(),
                ui.input_text(id="Url", label="Url", width="100%", value=restored_input("Url", "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"),
                    placeholder="http://, https://, ftp://, ftps://, or file://", update_on="blur",
                    guide=this, position="bottom",
                    text='This text box is for the entry of a valid URL of a data file. Allowed protocols include http://, https://, ftp://, ftps://, and file://\n The field will shake if the URL is invalid.'),
                ui.input_text(id="UName", label="Short name", value=restored_input("UName", ""),
                    guide=this, position="bottom",
                    text='This is how you choose to name the dataset. Keep this name short. By default, it is initially populated with the base URL. Each of the importation styles has this field.')
            )
        ]
        if not Module.IS_SHINYLIVE:
            uci_choices = UciChoices()
            restored_uci = restored_input("UciDataset")
            if restored_uci and restored_uci not in uci_choices:
                uci_choices.append(restored_uci)
            panels.append(
                ui.nav_panel(
                    "UC Irvine",
                    ui.tags.a("Visit Official Site", href="https://archive.ics.uci.edu/datasets", target="_blank"),
                    ui.input_selectize(id="UciDataset", label="UCI dataset", multiple=False, width="80%", choices=uci_choices, selected=restored_uci,
                        guide=this, position="bottom",
                        text='The UCI Machine Learning Repository is a collection of databases that are used by the machine learning community for the analysis of machine learning algorithms.'),
                    ui.input_text(id="IName", label="Short name", value=restored_input("IName", ""),
                        guide=this, position="bottom",
                        text='This is how you choose to name the dataset. Keep this name short. By default, it is initially populated with the dataset ID. Each of the importation styles has this field.')
                )
            )
        restored_tab = restored_input("Navset", "File based")
        available_tabs = {
            "File based", "Dataset based", "Web based", "UC Irvine"
        }
        if Module.IS_SHINYLIVE:
            available_tabs.remove("UC Irvine")
        if restored_tab not in available_tabs:
            restored_tab = "File based"
        return ui.navset_bar(
            *panels,
            title=None, id="Navset", selected=restored_tab, padding=0, fillable=False
        )
    
    this.front=front

    def back():
        return ui.TagList(
            ui.card_header("Data Summary", class_="text-primary text-center"),
            ui.output_ui(id="Summary", 
                guide=this, title="Data Summary", position="top", priority=-10,
                text="This summary shows information specific to diferent classes of data. It is available even before the data has been committed.",
                style="font-size: 0.85rem; line-height: 1.1;")
        )
    
    this.back=back

    def footer():
        return ui.TagList(
            ui.input_action_button(
                id="Commit", label='Commit Import', icon=icon("gavel", title="Commit the import", a11y="sem"),
                disabled=True, width="250px", class_="btn rounded-pill btn-sm d-block mx-auto btn-primary", style="border: 0px; box-shadow: none;",
                guide=this, title="Commit import button", position="top",
                text="This button commits the file reading. It bounces momentarily when it is ready to be clicked."
            ),
            ui.output_ui(
                id="Check",
                guide=this, title="Card status", position="top",
                text="Reports whether the selected source was read successfully and is ready to commit. Errors here should be resolved before the provisional data is published downstream.",
            )
        )

    this.footer=footer

    def settings():
        return ui.TagList(
            ui.input_text(
                id="Separator", label="Between-column separator", value=restored_input("Separator", ","),
                guide=this, position="left",
                text='Since tab, semi-colon and comma characters can occur in the data, this string specifies the type of separation string to employ. "Auto" will make an automated assessment.'
            ),
            ui.input_numeric(
                id="Sheet", label="Worksheet position", value=restored_input("Sheet", 1), min=1,
                guide=this, position="left",
                text='When importing from a multi-worksheet spreadsheet, this number represents the particular worksheet to import.',
            )
        )

    this.settings=settings

    def server(input, output, session):

        #### Shiny variables ----
        CommittedData=reactive.Value(None)
        restored_inputs = this._restored_configuration_state.get("inputs", {})
        restored_committed_tab = this._restored_configuration_state.get(
            "last_committed_tab"
        )
        LastCommittedTab=reactive.Value(restored_committed_tab)

        @output
        @render.ui
        def FilePickerUI():
            if Module.runtime_mode(session) == "local":
                if not native_file_picker_available():
                    return ui.help_text(
                        "No supported native file picker is available on "
                        "this host."
                    )
                return ui.input_action_button(
                    id="NativeFilePicker",
                    label="Local File picker...",
                    icon=icon(
                        "folder-open",
                        title="Choose a local file",
                        a11y="sem",
                    ),
                    class_="btn btn-primary btn-sm mb-2",
                )
            return ui.input_file(
                id="ServerFile",
                label=None,
                button_label="File picker...",
                multiple=True,
                width="80%",
                guide=this,
                title="File path",
                position="bottom",
                text=(
                    "This button uploads a data file selected in the file "
                    "picker dialogue."
                ),
            )

        def uploaded_files():
            """Return the current upload, or a still-valid restored upload."""
            files = (
                None
                if Module.runtime_mode(session) == "local"
                else input.ServerFile()
            )
            if files:
                return files
            saved = restored_inputs.get("ServerFile")
            if not isinstance(saved, list) or not saved:
                return None
            file0 = saved[0]
            if not isinstance(file0, Mapping):
                return None
            path = file0.get("datapath")
            if not isinstance(path, str) or not os.path.isfile(path):
                return None
            expected_size = file0.get("size")
            if isinstance(expected_size, int):
                try:
                    if os.path.getsize(path) != expected_size:
                        return None
                except OSError:
                    return None
            return saved

        def serializable_upload(files):
            if not files:
                return None
            saved=[]
            for file_info in files:
                if not isinstance(file_info, Mapping):
                    continue
                try:
                    saved.append({
                        "name": str(file_info["name"]),
                        "size": int(file_info["size"]),
                        "type": str(file_info.get("type", "")),
                        "datapath": str(file_info["datapath"]),
                    })
                except (KeyError, TypeError, ValueError):
                    continue
            return saved or None

        def configuration_state():
            """Return this card's current JSON-compatible configuration state."""
            with reactive.isolate():
                files = (
                    restored_inputs.get("ServerFile")
                    if Module.runtime_mode(session) == "local"
                    else input.ServerFile() or restored_inputs.get("ServerFile")
                )
                values = {
                    "Navset": input.Navset(),
                    "ServerFile": serializable_upload(files),
                    "LocalFilePath": input.LocalFilePath(),
                    "FName": input.FName(),
                    "Dataset": input.Dataset(),
                    "DName": input.DName(),
                    "Url": input.Url(),
                    "UName": input.UName(),
                    "UciDataset": (
                        None if Module.IS_SHINYLIVE else input.UciDataset()
                    ),
                    "IName": (
                        None if Module.IS_SHINYLIVE else input.IName()
                    ),
                    "Separator": input.Separator(),
                    "Sheet": input.Sheet(),
                }
                committed_tab = LastCommittedTab()
            return {
                "inputs": values,
                "last_committed_tab": committed_tab,
            }

        this.configuration_state = configuration_state

        @this.suspendable(calc=True)
        def ExportedData():
            data = CommittedData()
            req(data is not None)
            return data

        @this.suspendable(calc=True)
        def TempFilePath():
            if Module.runtime_mode(session) == "local":
                local_path=input.LocalFilePath()
                if local_path and os.path.isfile(local_path):
                    return local_path
            files=uploaded_files()
            if not files:
                return None
            # file_info is a list of dicts (one per file); get the first one
            file0=files[0]
            file=file0["datapath"]
            if not os.path.isfile(file):
                return None
            return file            

        

        def dict_to_html(d):
            html="<ul style='margin:0;padding-left:1em;'>"
            for k, v in d.items():
                if isinstance(v, dict):
                    html += f"<li><b>{k.capitalize()}</b>: {dict_to_html(v)}</li>"
                else:
                    html += f"<li><b>{k.capitalize()}</b>: {v}</li>"
            html += "</ul>"
            return html

        def read_file(path, sep=",", sheet=None , **kwargs):
            """Dispatch a file source to its format-specific reader."""
            ext=os.path.splitext(path)[1].lower()
            if ext in [".csv", ".tsv"]:
                df=pd.read_csv(
                    path,
                    sep=sep or ("\t" if ext == ".tsv" else ","),
                    **kwargs
                )
                object_cols = df.select_dtypes(include="object").columns
                df[object_cols] = df[object_cols].astype("string")
                if "geometry" in df.columns:
                    try:
                        return gpd.GeoDataFrame(
                            df,
                            geometry=gpd.GeoSeries.from_wkt(df["geometry"]),
                        )
                    except Exception as e:
                        this.log.warning(e, exc_info=1)
                return df
            elif ext in [".xls", ".xlsx"]:
                df=pd.read_excel(path, sheet_name=sheet, **kwargs)
                if "geometry" in df.columns:
                    try:
                        return gpd.GeoDataFrame(
                            df,
                            geometry=gpd.GeoSeries.from_wkt(df["geometry"]),
                        )
                    except Exception as e:
                        this.log.warning(e, exc_info=1)
                return df
            elif ext in [".parquet"]:
                df=pd.read_parquet(path, **kwargs)
                if "geometry" in df.columns:
                    try:
                        return gpd.GeoDataFrame(
                            df,
                            geometry=gpd.GeoSeries.from_wkt(df["geometry"]),
                        )
                    except Exception as e:
                        this.log.warning(e, exc_info=1)
                return df
            elif ext in [".feather"]:
                return pd.read_feather(path, **kwargs)
            elif ext in [".json"]:
                return pd.read_json(path, **kwargs)
            elif ext in [".geojson", ".gpkg", ".gml", ".kml"]:
                return gpd.read_file(path, **kwargs)
            elif ext in [".nc", ".grib", ".h5", ".hdf5"]:
                d=xr.open_dataset(path, **kwargs)
                try:
                    return d.to_dataframe()
                except Exception as e:
                    this.log.warning(e, exc_info=1)
                return d
            raise ValueError(f"Unsupported file extension: {ext}")

        def load_data(import_tab):
            if import_tab == "File based":
                if TempFilePath() is None:
                    return None
                return read_file(
                    path=TempFilePath(),
                    sep=input.Separator(),
                    sheet=input.Sheet(),
                )
            elif import_tab == "Web based":
                if not input.Url():
                    return None
                return read_file(
                    path=input.Url(),
                    sep=input.Separator(),
                    sheet=input.Sheet(),
                )
            elif import_tab == "Dataset based":
                req(input.Dataset())
                package, name=input.Dataset().split("::", 1)
                source=DATA_SOURCES.get(package)
                if not source:
                    raise ValueError(f"No dataset source available for '{package}'")
                return source["load"](name)
            elif import_tab == "UC Irvine":
                if Module.runtime_mode(session) == "shinylive":
                    raise ValueError("UC Irvine imports are unavailable in Shinylive")
                req(input.UciDataset())
                uci=fetch_ucirepo(name=input.UciDataset())
                return uci.data.original
            raise ValueError(f"Unknown data-import tab {import_tab!r}")

        @this.suspendable(calc=True)
        @this.record_code
        def GetData():
            return load_data(input.Navset())

        @this.suspendable(calc=True)
        @this.record_code
        def GetPxyData():
            if GetData() is None:
                return None
            return Pxy.from_native(GetData())

        ###Generic summary for the type of dataset ----
        @output
        @render.ui
        @this.record_code
        def Summary():

            def pd_basic_html(df: pd.DataFrame, isFullScreen=False) -> str:
                """
                Return an HTML table similar to df.info()
                """
                if isFullScreen:
                    table_size=""
                else:
                    table_size=" table_sm"
                rows=[]
                t=type(df)
                rows.append({"Property" : "Data Class", "Value" : ".".join([t.__module__,f"<b>{t.__name__}</b>"])})
                rows.append({"Property" : "Columns", "Value" : df.shape[1]})
                rows.append({"Property" : "Distinct Data Types", "Value" : df.dtypes.nunique()})
                table1=pd.DataFrame(rows)
                # High-level info
                hl_html="<h4>Dataset info:</h4>" + table1.to_html(index=False, escape=False, border=False, 
                    classes="table table-hover table-striped" + table_size
                )
                # Memory usage
                mem_bytes=df.memory_usage(deep=True).sum()
                mem_mb=mem_bytes / (1024 ** 2)
                mem_str=f"<br><h4>Memory usage:</h4> {mem_mb:.2f} MB ({mem_bytes:,} bytes)</p>"
                return hl_html + mem_str

            def pd_info_html(df: pd.DataFrame, isFullScreen=False) -> str:
                """
                Return an HTML table similar to df.info()
                """
                if isFullScreen:
                    table_size=""
                else:
                    table_size=" table_sm"

                # Table rendering
                rows=[]
                for col in df.columns:
                    non_null=df[col].count()
                    nulls=len(df) - non_null
                    row={
                        "Column" : col,
                        "Non-Null Count" : f"{non_null} non-null",
                        "Null Count" : f"{nulls}",
                        "Dtype" : str(df[col].dtype)
                    }
                    rows.append(row)
                summary=pd.DataFrame(rows)
                table_html="<h4>Variables:</h4>" + summary.to_html(
                    index=False, 
                    escape=True, 
                    border=False, 
                    max_rows=100, 
                    classes="table table-hover table-striped" + table_size
                )
                # Category summary
                # cat_cols=df.select_dtypes(include=['object', 'str', 'category'])  Shinylive does not like this line 
                cat_cols=df.select_dtypes(include=['object', 'category'])
                if cat_cols.empty:
                    cat_html=""
                else:
                    df=pd.DataFrame(cat_cols.describe().T)
                    cat_html="<h4>Categories:</h4>" + df.to_html(
                        index=True, 
                        escape=True, 
                        border=False, 
                        classes="table table-hover table-striped" + table_size
                    )
                return table_html + cat_html

            def gpd_info_html(df: gpd.GeoDataFrame, isFullScreen=False):
                if isFullScreen:
                    table_size=""
                else:
                    table_size=" table_sm"

                #TODO: allow for multiple geometry columns
                rows=[]
                rows.append({"Property" : "CRS", "Value" : df.crs})
                rows.append({"Property" : "Columns", "Value" : df.geometry.name})
                rows.append({"Property" : "Types", "Value" : df.geometry.geom_type.value_counts().to_dict()})
                rows.append({"Property" : "Bounds", "Value" : df.total_bounds})
                summary=pd.DataFrame(rows)
                return "<h4>Geometry:</h4>" + summary.to_html(
                    index=False, 
                    escape=True, 
                    border=False, 
                    max_rows=100, 
                    classes="table table-hover table-striped" + table_size
                )

            def xr_info_html(df: xr.Dataset, isFullScreen=False):
                if isFullScreen:
                    table_size=""
                else:
                    table_size=" table_sm"
                t=type(df)
                rows=[]
                rows.append({"Property" : "Data Class",  "Value" : ".".join([t.__module__,f"<b>{t.__name__}</b>"])})
                rows.append({"Property" : "Dimensions",  "Value" : data.dims})
                rows.append({"Property" : "Coordinates", "Value" : list(data.coords)})
                rows.append({"Property" : "Variables",   "Value" : len(data.data_vars)})
                rows.append({"Property" : "Data Attributes",  "Value" : len(data.attrs.keys())})
                table1=pd.DataFrame(rows)


                rows=[]
                for var_name, da in data.data_vars.items():
                    rows.append({"Variable" : var_name, "Dimensions" : da.dims, "Shape" : da.shape, "Data Type" : da.dtype, "Variable Attributes" : dict_to_html(da.attrs)})
                table2=pd.DataFrame(rows)
                # Dataset-level attributes
                attr_html=""
                if data.attrs:
                    attr_html="<h4>Data attributes:</h4>" + dict_to_html(data.attrs)
                    # for k, v in data.attrs.items():
                    #     attr_html=attr_html +(f"{k.capitalize()}: {v}<br>")
                return table1.to_html(
                    index=False, 
                    escape=False, 
                    border=False, 
                    max_rows=100, 
                    classes="table table-hover table-striped" + table_size
                ) + "<h4>Variables:</h4>" + table2.to_html(
                    index=False, 
                    escape=False, 
                    border=False, 
                    max_rows=100, 
                    classes="table table-hover table-striped" + table_size
                ) + attr_html

            d=GetPxyData()
            req(d is not None)
            data=d.frame
            if data is None or len(data) == 0:
                return ui.span("No Data", class_="text-warning")
            output=[]
            if isinstance(data, pd.DataFrame):
                output.append(pd_basic_html(data, isFullScreen=this.isFullScreen()))
            if isinstance(data, gpd.GeoDataFrame):
                output.append(gpd_info_html(data, isFullScreen=this.isFullScreen()))
            if isinstance(data, pd.DataFrame):
                output.append(pd_info_html(data, isFullScreen=this.isFullScreen()))
            # if isinstance(data, xr.Dataset):
            #     output.append(xr_info_html(data, isFullScreen=this.isFullScreen()))
            if len(output) == 0:
                return ui.span(f"Data class '{type(data).__name__}'not expected", class_="text-warning")
            return ui.HTML("".join(output))


        def size_text(data, max_rows=None):
            """
            Return a compact string summarizing the dataset shape,
            handling pandas, GeoPandas, and xarray objects gracefully.
            """
            # --- pandas.DataFrame or geopandas.GeoDataFrame ---
            if hasattr(data, "shape") and not hasattr(data, "dims"):
                rows, cols=data.shape
                type_name=type(data).__name__
                if max_rows and rows == max_rows:
                    return f"({type_name}: Obs limited to first {rows}, Vars = {cols})"
                else:
                    return f"({type_name}: Obs = {rows}, Vars = {cols})"
            # --- xarray.Dataset ---
            elif hasattr(data, "dims"):
                # xarray.Dataset.dims is a mapping: {'time': 2920, 'lat': 25, 'lon': 53}
                dims=data.dims
                n_dims=len(dims)
                total_size=1
                for dim_size in dims.values():
                    total_size *= dim_size
                n_vars=len(data.data_vars)
                return f"(xarray.Dataset: Obs = {total_size}, Vars = {n_vars + n_dims})"
            # --- Fallback ---
            else:
                return f"({type(data).__name__}: shape unknown)"

        preserve_restored_name = {
            "Dataset": bool(restored_inputs.get("Dataset")),
            "Url": bool(restored_inputs.get("Url")),
            "UciDataset": bool(restored_inputs.get("UciDataset")),
        }

        @this.suspendable(triggers=[input.ServerFile])
        def ServerFile():
            req(input.ServerFile())
            files=input.ServerFile()
            file0=files[0]
            filename=file0["name"]
            stem, _=os.path.splitext(filename)  # Remove the extension
            ui.update_text(id="FName", value=stem)

        @this.suspendable(triggers=[input.NativeFilePicker])
        async def NativeFilePicker():
            if Module.runtime_mode(session) != "local":
                return
            try:
                filename=await choose_local_file(Path.home())
            except Exception as error:
                this.log.warning(
                    "Could not open the native file picker: %s",
                    error,
                    exc_info=True,
                )
                ui.notification_show(
                    f"The local file picker could not be opened: {error}",
                    type="error",
                    duration=8,
                )
                return
            if not filename:
                return
            stem=Path(filename).stem
            ui.update_text(id="LocalFilePath", value=filename)
            ui.update_text(id="FName", value=stem)


        @this.suspendable()
        def DatasetName():
            req(input.Dataset())
            if (
                preserve_restored_name["Dataset"]
                and input.Dataset() == restored_inputs.get("Dataset")
            ):
                preserve_restored_name["Dataset"] = False
                return
            preserve_restored_name["Dataset"] = False
            _, stem=input.Dataset().split("::", 1)
            ui.update_text(id="DName", value=stem)

        @this.suspendable()
        def UciDatasetName():
            req(input.UciDataset())
            if (
                preserve_restored_name["UciDataset"]
                and input.UciDataset() == restored_inputs.get("UciDataset")
            ):
                preserve_restored_name["UciDataset"] = False
                return
            preserve_restored_name["UciDataset"] = False
            ui.update_text(id="IName", value=input.UciDataset())

                
        @this.settle(seconds=2)
        @this.suspendable(calc=True)
        def Url():
            return input.Url()


        @this.suspendable(triggers=[Url])
        def Url2():
            req(Url())
            if (
                preserve_restored_name["Url"]
                and Url() == restored_inputs.get("Url")
            ):
                preserve_restored_name["Url"] = False
                return
            preserve_restored_name["Url"] = False
            path=urlparse(Url()).path  # Extract the path part of the URL
            filename=os.path.basename(path)  # Get the filename
            stem, _=os.path.splitext(filename)  # Remove the extension
            ui.update_text(id="UName", value=stem)


        def url_exists(url: str) -> bool:
            this.log.debug(f"Checking url: {url}")
            try:
                response=requests.head(url, allow_redirects=True, timeout=5)
                return response.status_code == 200
            except requests.RequestException:
                return False

        #### Check ----
        @output
        @render.ui
        async def Check():
            if input.Navset() == "Web based":
                butt_disabled=input.UName().strip() == "" 
                if Url().strip() == "":
                    message=ui.span("No URL supplied", class_="text-center text-warning")
                    butt_disabled=True
                else:
                    ok=url_exists(Url())
                    if ok is None:
                        butt_disabled=True
                        message=ui.span("No internet connectivity", class_="text-center text-warning")
                    elif not ok:
                        message=ui.span("The URL is not valid", class_="text-center text-danger")
                        await session.send_custom_message("animate", {"id" : session.ns("Url"), "animation" : "shakeX", "delay" : 500})
                        butt_disabled=True
                    else:
                        try:
                            d=GetPxyData()
                            d.name=Url()
                            text=size_text(GetData())
                            if CommittedData() == d:
                                message=ui.span("Web import successful ", text, class_="text-center text-success")
                            else:
                                message=ui.span("Web import ready ", text, class_="text-center text-primary")
                                await session.send_custom_message("animate", {"id" : session.ns("Commit"), "animation" : "bounce", "delay" : 500})
                        except SilentException:
                            message=""
                        except Exception as e:  # noqa: BLE001
                            message=ui.span(f"Error ({type(e).__name__}): {e}", class_="text-center text-danger")
                            butt_disabled=True
            elif input.Navset() == "File based":
                butt_disabled=input.FName().strip() == "" 
                try:
                    d=GetPxyData()
                    if d is None:
                        message=ui.span("No file supplied yet", class_="text-center text-warning")
                        butt_disabled=True
                    else:
                        d.name=input.FName()
                        text=size_text(GetData())
                        if CommittedData() == d:
                            message=ui.span("File import successful ", text, class_="text-center text-success")
                        else:
                            message=ui.span("File import ready ", text, class_="text-center text-primary")
                            await session.send_custom_message("animate", {"id" : session.ns("Commit"), "animation" : "bounce", "delay" : 500})
                except Exception as e:  # noqa: BLE001
                    message=ui.span(f"Error ({type(e).__name__}): {e}", class_="text-center text-danger")
                    butt_disabled=True
            elif input.Navset() == "Dataset based":
                butt_disabled=input.DName().strip() == "" 
                if input.Dataset() is None:
                    message=ui.span("No dataset selected", class_="text-center text-warning")
                    butt_disabled=True
                else:
                    try:
                        d=GetPxyData()
                        if d is None:
                            message=ui.span("No dataset chosen yet", class_="text-center text-warning")
                            butt_disabled=True
                        else:
                            d.name=input.DName()
                            text=size_text(GetData())
                            if CommittedData() == d:
                                message=ui.span("Dataset import successful ", text, class_="text-center text-success")
                            else:
                                message=ui.span("Dataset import ready ", text, class_="text-center text-primary")
                                await session.send_custom_message("animate", {"id" : session.ns("Commit"), "animation" : "bounce", "delay" : 500})
                    except SilentException:
                        message=""
                        butt_disabled=True

                    except Exception as e:  # noqa: BLE001
                        message=ui.span(f"Error ({type(e).__name__}): {e}", class_="text-center text-danger")
                        butt_disabled=True
            elif input.Navset() == "UC Irvine":
                butt_disabled=input.IName().strip() == "" 
                if input.UciDataset() is None:
                    message=ui.span("No dataset selected", class_="text-center text-warning")
                    butt_disabled=True
                else:
                    try:
                        d=GetPxyData()
                        if d is None:
                            message=ui.span("No dataset chosen yet", class_="text-center text-warning")
                            butt_disabled=True
                        else:
                            d.name=input.IName()
                            text=size_text(GetData())
                            if CommittedData() == d:
                                message=ui.span("Dataset import successful ", text, class_="text-center text-success")
                            else:
                                message=ui.span("Dataset import ready ", text, class_="text-center text-primary")
                                await session.send_custom_message("animate", {"id" : session.ns("Commit"), "animation" : "bounce", "delay" : 500})
                    except SilentException:
                        message=""
                        butt_disabled=True

                    except Exception as e:  # noqa: BLE001
                        message=ui.span(f"Error ({type(e).__name__}): {e}", class_="text-center text-danger")
                        butt_disabled=True
            ui.update_action_button(id="Commit", label="Commit Import", disabled=butt_disabled)
            return message


        #### Commit event ----
        def commit_import(import_tab):
            data=load_data(import_tab)
            if data is None:
                raise ValueError(f"No data is available for {import_tab!r}")
            pxd=Pxy.from_native(data)
            names = {
                "File based": input.FName,
                "Web based": input.UName,
                "Dataset based": input.DName,
                "UC Irvine": input.IName,
            }
            name = names[import_tab]().strip()
            if not name:
                raise ValueError(f"No short name is available for {import_tab!r}")
            pxd.name=name
            CommittedData.set(pxd.clone())
            LastCommittedTab.set(import_tab)

        @this.suspendable(triggers=[input.Commit])
        async def CommitEvent():
            commit_import(input.Navset())

        def restore_committed_import():
            """Try to recreate the last import, waiting silently for inputs."""
            if not restored_committed_tab:
                return True
            try:
                for input_id, expected in restored_inputs.items():
                    if input_id == "ServerFile" or expected is None:
                        continue
                    if (
                        input_id in {"UciDataset", "IName"}
                        and Module.runtime_mode(session) == "shinylive"
                    ):
                        continue
                    actual = input[input_id]()
                    if actual is None:
                        req(False)
                    if actual != expected:
                        raise ValueError(
                            f"{input_id} restored as {actual!r}, expected "
                            f"{expected!r}"
                        )
                if (
                    restored_committed_tab == "File based"
                    and TempFilePath() is None
                ):
                    raise FileNotFoundError(
                        "The previously uploaded temporary file is no longer "
                        "available"
                    )
                commit_import(restored_committed_tab)
            except SilentException:
                # Dynamically inserted UI is sent during one flush; its initial
                # values arrive from the browser later. Because the reads above
                # are reactive dependencies, this function will be retried.
                return False
            except Exception as error:
                this.log.warning(
                    "Could not restore the committed data import: %s",
                    error,
                    exc_info=True,
                )
                ui.notification_show(
                    "The previous data import could not be reconnected: "
                    f"{error}. Select the source and commit it afresh.",
                    type="warning",
                    duration=None,
                )
                return True
            this.log.info(
                "🔄 Restored committed %s data import",
                restored_committed_tab,
            )
            return True

        this._restore_committed_import = restore_committed_import

        restore_finished = False

        if restored_committed_tab:
            @reactive.effect
            def RestoreCommittedImport():
                nonlocal restore_finished
                if restore_finished:
                    return
                restore_finished = restore_committed_import()
        return ExportedData


    this.server=server

    return this
  
    
if Module.running_directly(name =__name__):
    this=instance()
    this.run()
