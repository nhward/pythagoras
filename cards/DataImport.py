from shiny import ui, render, reactive, req
from faicons import icon_svg as icon
from urllib.parse import urlparse
import geopandas as gpd
import xarray as xr
import os
import requests
import pandas as pd
from module import Module
from card import Card
from proxyData import ProxyData as pxd


__version__ = "0.1.0"

def instance():
    this = Card(name = "dataImport", mutable = True)
    this.long_name = "Data import"
    this.description = "This card facilitates the ingestion of data, be it numeric, categorical, textual, temporal or spatial."
    
    def settings():
        return ui.TagList(
            ui.input_text(
                id = "Separator", 
                label = "Between-column separator", 
                value = ",",
                guide = this,
                text = 'Since tab, semi-colon and comma separation are common, this string specifies the type of separation to be employed. "Auto" will make an automated asessment.',
                position = "left"),
            ui.input_numeric(
                id = "Sheet", 
                label = "Worksheet position", 
                value = 1, 
                min = 1, 
                guide = this,
                text = 'When importing from a multi-worksheet spreadsheet, this number represents the particular worksheet to import.',
                position = "left")
        )

    this.settings = settings

    def front():
        return ui.navset_bar(
            ui.nav_panel(
                "File based",
                ui.tags.br(),
                ui.input_file(id = "ServerFile", label = None, button_label = "Local File picker...", multiple = True, width = "80%", 
                    guide = this, title = "File path", position = "bottom",
                    text = 'This button starts a file-picker dialogue of data files on the client. The chosen file will be uploaded to the server.'),
                ui.input_text(id = "FName", label = "Short name", guide = this, position = "bottom",
                    text = 'This is how you choose to name the dataset. Keep this name short. By default it is initially populated with the file name. Each of the importation styles has this field.')
            ),
            ui.nav_panel(
                "Dataset based",
                ui.tags.br(),
                ui.input_selectize(id = "Dataset", label = "Package dataset", multiple = False, width = "80%", choices = {}, guide = this, text = ""),
                ui.input_text(id = "DName", label = "Short name", guide = this, position = "bottom",
                    text = "This is how you choose to name the dataset. Keep this name short. By default it is initially populated with the package's dataset name. Each of the importation styles has this field.")
            ),
            ui.nav_panel(
                "Web based",
                ui.tags.br(),
                ui.input_text(id = "Url", label = "Url", width = "100%", value  = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/110m_cultural/ne_110m_admin_0_countries.shp", placeholder = "http://, https://, ftp://, ftps://, or file://", 
                    guide = this, position = "bottom",
                    text = 'This text box is for the entry of a valid URL of a data file. Allowed protocols include http://, https://, ftp://, ftps://, and file://\n The field will shake if the URL is invalid.'),
                ui.input_text(id = "UName", label = "Short name", guide = this, position = "bottom",
                    text = 'This is how you choose to name the dataset. Keep this name short. By default it is initially populated with the base URL. Each of the importation styles has this field.')
            ),
            title = None,
            id = "Navset", 
            padding = 0, 
            fillable = True
        )
    
    this.front = front

    def back():
        return ui.TagList(
            ui.card_header("Data Summary", class_ = "text-primary text-center"),
            ui.output_ui(id = "Summary", 
                guide = this, title = "Data Summary", position = "top", priority = -10,
                text = "This summary shows information specific to diferent classes of data. It is available even before the data has been committed.",
                style = "font-size: 0.85rem; line-height: 1.1;")
        )
    
    this.back = back

    def footer():
        return ui.TagList(
            ui.input_action_button(
                id = "Commit", 
                label = 'Commit Import', 
                icon = icon("gavel", title = "Commit the import", a11y = "sem"),
                disabled = True, 
                width = "250px", 
                class_ = "btn rounded-pill btn-sm d-block mx-auto btn-primary",
                style = "border: 0px; box-shadow: none;",
                guide = this, 
                title = "Commit button",
                text = "This button commits the file reading. It bounces momentarily when it is ready to be clicked.",
                position = "top"
            ),
            ui.output_ui(
                id = "Check",
                guide = this, 
                title = "Card status",
                text = "This contains a line of colour coded information about the state of the card. Notice that it reveals when only a portion of the available data is being used. This limit can be changed in the settings.",
                position = "top")
        )

    this.footer = footer

    def server(input, output, session):

        #### Shiny variables ----
        CommittedData = reactive.Value(None)
        CommittedName = reactive.Value(None)

        @reactive.calc
        def TempFilePath():
            files = input.ServerFile()
            if not files:
                return None
            # file_info is a list of dicts (one per file); get the first one
            file0 = files[0]
            file = file0["datapath"]
            if not os.path.isfile(file):
                return None
            return file            

        
        # reactive debounce candidate
        

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
            last_exc = None
            # 1) Rdatasets route: sm.datasets.get_rdataset(...).data
            try:
                rd = sm.datasets.get_rdataset(name, cache=True)
                if hasattr(rd, "data"):
                    return rd.data
                # if it didn't have .data for some reason, try to return rd itself
                return rd
            except Exception as e:
                last_exc = e
            # 2) Try built-in statsmodels dataset modules (e.g., sm.datasets.co2)
            try:
                dataset_modules = getattr(sm.datasets, "__all__", []) or []
                # search for matching module name (case-insensitive)
                for mod_name in dataset_modules:
                    if mod_name.lower() == name.lower():
                        mod = importlib.import_module(f"statsmodels.datasets.{mod_name}")
                        # common loader patterns
                        if hasattr(mod, "load_pandas"):
                            loaded = mod.load_pandas()
                            return getattr(loaded, "data", loaded)
                        if hasattr(mod, "load"):
                            loaded = mod.load()
                            return getattr(loaded, "data", loaded)
                        # some modules may expose a top-level variable `data` or similar
                        if hasattr(mod, "data"):
                            return getattr(mod, "data")
                # as a last attempt, check for attribute directly on sm.datasets
                mod_obj = getattr(sm.datasets, name, None)
                if mod_obj:
                    if hasattr(mod_obj, "load_pandas"):
                        loaded = mod_obj.load_pandas()
                        return getattr(loaded, "data", loaded)
                    if hasattr(mod_obj, "load"):
                        loaded = mod_obj.load()
                        return getattr(loaded, "data", loaded)
            except Exception as e:
                last_exc = e
            # Give a helpful message including the last exception
            raise ValueError(
                f"Statsmodels: could not load dataset '{name}'. "
                f"Tried get_rdataset() and internal sm.datasets modules. Last error: {last_exc}"
            )


        DATA_SOURCES = {
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
                    (__import__("vega_datasets").data(name)()
                    if callable(__import__("vega_datasets").data(name))
                    else __import__("vega_datasets").data(name))
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

        def dict_to_html(d):
            html = "<ul style='margin:0;padding-left:1em;'>"
            for k, v in d.items():
                if isinstance(v, dict):
                    html += f"<li><b>{k.capitalize()}</b>: {dict_to_html(v)}</li>"
                else:
                    html += f"<li><b>{k.capitalize()}</b>: {v}</li>"
            html += "</ul>"
            return html

        @reactive.calc
        @this.record_code
        def GetData():

            def read_file(path, sep = ",", sheet = None , **kwargs):
                """
                Generic data importer.
                Dispatches to pandas, geopandas, or xarray depending on file extension.
                Parameters
                ----------
                path : str - Path to the input file.
                delimiter : str, optional - Explicit delimiter for CSV/TSV.
                sheet : the worksheet number.
                kwargs : passed to the underlying reader.
                """
                ext = os.path.splitext(path)[1].lower()
                # --- Tabular formats ---
                if ext in [".csv", ".tsv"]:
                    df = pd.read_csv(
                        path, 
                        sep = sep or ("\t" if ext == ".tsv" else ","),
                        **kwargs
                    )
                    # Promote to GeoDataFrame if "geometry" column exists
                    if "geometry" in df.columns:
                        try:
                            return gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df["geometry"]))
                        except Exception:
                            pass
                    return df
                elif ext in [".xls", ".xlsx"]:
                    df = pd.read_excel(
                        path, 
                        sheet_name = sheet,
                        **kwargs
                    )
                    if "geometry" in df.columns:
                        try:
                            return gpd.GeoDataFrame(df, geometry = gpd.GeoSeries.from_wkt(df["geometry"]))
                        except Exception:
                            pass
                    return df
                elif ext in [".parquet"]:
                    df = pd.read_parquet(
                        path, 
                        **kwargs
                    )
                    if "geometry" in df.columns:
                        try:
                            return gpd.GeoDataFrame(df, geometry = gpd.GeoSeries.from_wkt(df["geometry"]))
                        except Exception:
                            pass
                    return df
                elif ext in [".feather"]:
                    return pd.read_feather(path, **kwargs)
                elif ext in [".json"]:
                    return pd.read_json(path, **kwargs)
                # --- Spatial formats ---
                elif ext in [".geojson", ".gpkg", ".gml", ".kml"]:
                    return gpd.read_file(
                        path, 
                        **kwargs
                    )
                # --- Time-series / multidimensional ---
                elif ext in [".nc", ".grib", ".h5", ".hdf5"]:
                    d = xr.open_dataset(path, **kwargs)
                    try:
                        return d.to_dataframe()
                    except Exception:
                        pass
                    return d
                raise ValueError(f"Unsupported file extension: {ext}")
            
            if input.Navset() == "File based":
                if TempFilePath() is None:
                    return None
                d = read_file(path = TempFilePath(), sep = input.Separator(), sheet = input.Sheet())
            elif input.Navset() == "Web based":
                if Url() is None:
                    return None
                d =  read_file(path = Url(), sep = input.Separator(), sheet = input.Sheet())
            elif input.Navset() == "Dataset based":
                if input.Dataset() is None:
                    return None
                package, name = input.Dataset().split("::", 1)
                source = DATA_SOURCES.get(package)
                if not source:
                    raise ValueError(f"No dataset source available for '{package}'")
                d =  source["load"](name)
            return d

        @reactive.calc
        @this.record_code
        def GetData2():
            if GetData() is None:
                return None
            return pxd.from_native(GetData(), )

        ###Generic summary for the type of dataset ----
        @output
        @render.ui
        @this.record_code
        def Summary():

            def pd_basic_html(df: pd.DataFrame, isFullScreen = False) -> str:
                """
                Return an HTML table similar to df.info()
                """
                if isFullScreen:
                    table_size = ""
                else:
                    table_size = " table_sm"
                rows = []
                t = type(df)
                rows.append({"Property" : "Data Class", "Value" : ".".join([t.__module__,f"<b>{t.__name__}</b>"])})
                rows.append({"Property" : "Columns", "Value" : df.shape[1]})
                rows.append({"Property" : "Distinct Data Types", "Value" : df.dtypes.nunique()})
                table1 = pd.DataFrame(rows)
                # High-level info
                hl_html = "<h4>Dataset info:</h4>" + table1.to_html(index = False, escape = False, border = False, 
                    classes="table table-hover table-striped" + table_size
                )
                # Memory usage
                mem_bytes = df.memory_usage(deep=True).sum()
                mem_mb = mem_bytes / (1024 ** 2)
                mem_str = f"<br><h4>Memory usage:</h4> {mem_mb:.2f} MB ({mem_bytes:,} bytes)</p>"
                return hl_html + mem_str

            def pd_info_html(df: pd.DataFrame, isFullScreen = False) -> str:
                """
                Return an HTML table similar to df.info()
                """
                if isFullScreen:
                    table_size = ""
                else:
                    table_size = " table_sm"

                # Table rendering
                rows = []
                for col in df.columns:
                    non_null = df[col].count()
                    nulls = len(df) - non_null
                    row = {
                        "Column" : col,
                        "Non-Null Count" : f"{non_null} non-null",
                        "Null Count" : f"{nulls}",
                        "Dtype" : str(df[col].dtype)
                    }
                    rows.append(row)
                summary = pd.DataFrame(rows)
                table_html = "<h4>Variables:</h4>" + summary.to_html(
                    index = False, 
                    escape = True, 
                    border = False, 
                    max_rows = 100, 
                    classes = "table table-hover table-striped" + table_size
                )
                # Category summary
                cat_cols = df.select_dtypes(include=['object', 'category'])
                if cat_cols.empty:
                    cat_html = ""
                else:
                    df = pd.DataFrame(cat_cols.describe().T)
                    cat_html = "<h4>Categories:</h4>" + df.to_html(
                        index = True, 
                        escape = True, 
                        border = False, 
                        classes="table table-hover table-striped" + table_size
                    )
                return table_html + cat_html

            def gpd_info_html(df: gpd.GeoDataFrame, isFullScreen = False):
                if isFullScreen:
                    table_size = ""
                else:
                    table_size = " table_sm"

                #TODO: allow for multiple geometry columns
                rows = []
                rows.append({"Property" : "CRS", "Value" : df.crs})
                rows.append({"Property" : "Columns", "Value" : df.geometry.name})
                rows.append({"Property" : "Types", "Value" : df.geometry.geom_type.value_counts().to_dict()})
                rows.append({"Property" : "Bounds", "Value" : df.total_bounds})
                summary = pd.DataFrame(rows)
                return "<h4>Geometry:</h4>" + summary.to_html(
                    index = False, 
                    escape = True, 
                    border = False, 
                    max_rows = 100, 
                    classes = "table table-hover table-striped" + table_size
                )

            def xr_info_html(df: xr.Dataset, isFullScreen = False):
                if isFullScreen:
                    table_size = ""
                else:
                    table_size = " table_sm"
                t = type(df)
                rows = []
                rows.append({"Property" : "Data Class",  "Value" : ".".join([t.__module__,f"<b>{t.__name__}</b>"])})
                rows.append({"Property" : "Dimensions",  "Value" : data.dims})
                rows.append({"Property" : "Coordinates", "Value" : list(data.coords)})
                rows.append({"Property" : "Variables",   "Value" : len(data.data_vars)})
                rows.append({"Property" : "Data Attributes",  "Value" : len(data.attrs.keys())})
                table1 = pd.DataFrame(rows)


                rows = []
                for var_name, da in data.data_vars.items():
                    rows.append({"Variable" : var_name, "Dimensions" : da.dims, "Shape" : da.shape, "Data Type" : da.dtype, "Variable Attributes" : dict_to_html(da.attrs)})
                table2 = pd.DataFrame(rows)
                # Dataset-level attributes
                attr_html = ""
                if data.attrs:
                    attr_html = "<h4>Data attributes:</h4>" + dict_to_html(data.attrs)
                    # for k, v in data.attrs.items():
                    #     attr_html = attr_html +(f"{k.capitalize()}: {v}<br>")
                return table1.to_html(
                    index = False, 
                    escape = False, 
                    border = False, 
                    max_rows = 100, 
                    classes = "table table-hover table-striped" + table_size
                ) + "<h4>Variables:</h4>" + table2.to_html(
                    index = False, 
                    escape = False, 
                    border = False, 
                    max_rows = 100, 
                    classes = "table table-hover table-striped" + table_size
                ) + attr_html

            d = GetData2()
            req(d is not None)
            data = d.to_native()
            if data is None or len(data) == 0:
                return ui.span("No Data", class_ = "text-warning")
            output = []
            if isinstance(data, pd.DataFrame):
                output.append(pd_basic_html(data, isFullScreen = this.isFullScreen()))
            if isinstance(data, gpd.GeoDataFrame):
                output.append(gpd_info_html(data, isFullScreen = this.isFullScreen()))
            if isinstance(data, pd.DataFrame):
                output.append(pd_info_html(data, isFullScreen = this.isFullScreen()))
            # if isinstance(data, xr.Dataset):
            #     output.append(xr_info_html(data, isFullScreen = this.isFullScreen()))
            if len(output) == 0:
                return ui.span(f"Data class '{type(data).__name__}'not expected", class_ = "text-warning")
            return ui.HTML("".join(output))


        def size_text(data, max_rows=None):
            """
            Return a compact string summarizing the dataset shape,
            handling pandas, GeoPandas, and xarray objects gracefully.
            """
            # --- pandas.DataFrame or geopandas.GeoDataFrame ---
            if hasattr(data, "shape") and not hasattr(data, "dims"):
                rows, cols = data.shape
                type_name = type(data).__name__

                if max_rows and rows == max_rows:
                    return f"({type_name}: Obs limited to first {rows}, Vars = {cols})"
                else:
                    return f"({type_name}: Obs = {rows}, Vars = {cols})"
            # --- xarray.Dataset ---
            elif hasattr(data, "dims"):
                # xarray.Dataset.dims is a mapping: {'time': 2920, 'lat': 25, 'lon': 53}
                dims = data.dims
                n_dims = len(dims)
                total_size = 1
                for dim_size in dims.values():
                    total_size *= dim_size
                n_vars = len(data.data_vars)
                return f"(xarray.Dataset: Obs = {total_size}, Vars = {n_vars + n_dims})"
            # --- Fallback ---
            else:
                return f"({type(data).__name__}: shape unknown)"


        @this.suspendable(triggers = [input.ServerFile])
        def FName():
            req(input.ServerFile())
            files = input.ServerFile()
            file0 = files[0]
            filename = file0["name"]
            stem, _ = os.path.splitext(filename)  # Remove the extension
            ui.update_text(id = "FName", value = stem)


        @this.suspendable(triggers = [input.Dataset])
        def DName():
            req(input.Dataset())
            _, stem = input.Dataset().split("::", 1)
            ui.update_text(id = "DName", value = stem)
        
        
        #@this.debounce(2)
        @reactive.calc
        def Url():
            return input.Url()


        @this.suspendable(triggers = [Url])
        def UName():
            req(Url())
            path = urlparse(Url()).path  # Extract the path part of the URL
            filename = os.path.basename(path)  # Get the filename
            stem, _ = os.path.splitext(filename)  # Remove the extension
            ui.update_text(id = "UName", value = stem)


        def url_exists(url: str) -> bool:
            this.debug(f"Checking url: {url}")
            try:
                response = requests.head(url, allow_redirects=True, timeout=5)
                return response.status_code == 200
            except requests.RequestException:
                return False


        #### Check ----
        @output
        @render.ui
        async def Check():
            if input.Navset() == "Web based":
                butt_disabled = input.UName().strip() == "" 
                if Url().strip() == "":
                    message = ui.span("No URL supplied", class_ = "text-center text-warning")
                    butt_disabled = True
                else:
                    ok = url_exists(Url())
                    if ok is None:
                        butt_disabled = True
                        message = ui.span("No internet connectivity", class_ = "text-center text-warning")
                    elif not ok:
                        message = ui.span("The URL is not valid", class_ = "text-center text-danger")
                        await session.send_custom_message("animate", {"id" : session.ns("Url"), "animation" : "shakeX", "delay" : 500})
                        butt_disabled <- True
                    else:
                        try:
                            text = size_text(GetData())
                            if GetData2().__eq__(CommittedData()):
                                message = ui.span("Web import successful ", text, class_ = "text-center text-success")
                            else:
                                message = ui.span("Web import ready ", text, class_ = "text-center text-primary")
                                await session.send_custom_message("animate", {"id" : session.ns("Commit"), "animation" : "bounce", "delay" : 500})
                        except Exception as e:
                            message = ui.span(f"Error ({type(e).__name__}): {e}", class_ = "text-center text-danger")
                            butt_disabled = True
            elif input.Navset() == "File based":
                butt_disabled = input.FName().strip() == "" 
                try:
                    d = GetData2()
                    if d is None:
                        message = ui.span("No file supplied yet", class_ = "text-center text-warning")
                        butt_disabled = True
                    else:
                        text = size_text(GetData())
                        if d.__eq__(CommittedData()):
                            message = ui.span("File import successful ", text, class_ = "text-center text-success")
                        else:
                            message = ui.span("File import ready ", text, class_ = "text-center text-primary")
                            await session.send_custom_message("animate", {"id" : session.ns("Commit"), "animation" : "bounce", "delay" : 500})
                except Exception as e:
                    message = ui.span(f"Error ({type(e).__name__}): {e}", class_ = "text-center text-danger")
                    butt_disabled = True
            elif input.Navset() == "Dataset based":
                butt_disabled = input.DName().strip() == "" 
                if input.Dataset() is None:
                    message = ui.span("No dataset selected", class_ = "text-center text-warning")
                    butt_disabled = True
                else:
                    try:
                        d = GetData2()
                        if d is None:
                            message = ui.span("No dataset chosen yet", class_ = "text-center text-warning")
                            butt_disabled = True
                        else:
                            text = size_text(GetData())
                            if d.__eq__(CommittedData()):
                                message = ui.span("Dataset import successful ", text, class_ = "text-center text-success")
                            else:
                                message = ui.span("Dataset import ready ", text, class_ = "text-center text-primary")
                                await session.send_custom_message("animate", {"id" : session.ns("Commit"), "animation" : "bounce", "delay" : 500})

                    except Exception as e:
                        message = ui.span(f"Error ({type(e).__name__}): {e}", class_ = "text-center text-danger")
                        butt_disabled = True
            ui.update_action_button(id = "Commit", label = "Commit Import", disabled = butt_disabled)
            return message


        #### Commit event ----
        @this.suspendable(triggers = [input.Commit])
        def CommitEvent():
            if input.Navset() == "File based":
                CommittedName.set(input.FName())
            elif input.Navset() == "Web based":
                CommittedName.set(input.UName())
            elif input.Navset() == "Dataset based":
                CommittedName.set(input.DName())
            this._exports["name"].set(CommittedName())
            this.debug(f"Setting export name to value: {CommittedName()}")
            CommittedData.set(GetData2())
            this._exports["data"].set(CommittedData())

        @reactive.Effect
        def data_passthrough():
            pass


        @reactive.calc
        def getDatasetChoices():
            choices = {}
            for pkg, source in DATA_SOURCES.items():
                try:
                    result = source["fetch"]()
                    # inner dict: {value: label}
                    inner = {f"{pkg}::{ds}": ds for ds in result}
                    choices[f"_______Package {pkg}_______"] = inner
                except Exception as e:
                    this.warning(f"Failed to fetch datasets from {pkg}: {e}")
                    choices[pkg] = {}
            return choices


        #### Update dataset picker choices ----
        @this.suspendable(triggers = [getDatasetChoices])
        def ChoiceUpdate():
            ui.update_selectize(
                id = "Dataset",
                choices = getDatasetChoices()
            )


    this.server = server

    return this

if Module.running_under_tests():
    this = instance()
    app = Module.app(modules = {this.ns: this})
elif Module.running_directly(name =__name__):
    this = instance()
    Module.run(modules = {this.ns: this})