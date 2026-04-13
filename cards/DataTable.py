from shiny import ui, render, reactive, req
from faicons import icon_svg as icon
from module import Module
from card import Card
import pandas as pd
from typing import List, Dict
import geopandas as gpd
import shapely

# TODO: Suits pandas, polar or narwhals objects
# TODO: employ roles when/if available

__version__ = "0.1.0"

def instance():
    this = Card(name = "dataTable", mutable = False)
    this.long_name = "Data tabulation"
    this.description = "This card enables the data to be listed and searched."
    
    def front():
        return ui.output_ui(id = "DataTable") # Using dynamic data tables to avoid "sortable" problem of multiple tables
    
    this.front = front

    def settings():
        return ui.TagList(
            ui.input_slider(
                id = "Decimals", 
                label = "Number of decimal places to show", 
                min = -2,
                max = 10, 
                value = 2,
                guide = this,
                text = 'The numeric variables in the data table will be rounded to this number of decimal places.',
                position = "left"
            ),
            ui.input_checkbox(
                    id = "Bounded", 
                    label = "Each Geometry variable summarised as a bounding box", 
                    value = True,
                    guide = this,
                    text = 'Any geometery columns are individually summarised as bounding boxes.',
                    position = "left"
            ),
            ui.input_slider(
                id = "MaxObs", 
                label = "Maximum observations to chart", 
                min = 3,
                max = 7,
                value = 4,
                ticks = True,
                pre = "10^",
                guide = this,
                text = 'Limit to number of observations to chart to ensure responsiveness (logarithmic scale).',
                position = "left")
        )

    this.settings = settings

    def footer():
        return ui.download_button(
                id = "Export", 
                label = 'Export', 
                icon = icon("file-arrow-down", title = "Export the data", a11y = "sem"),
                width = "250px", 
                class_ = "btn rounded-pill btn-sm d-block mx-auto btn-primary",
                style = "border: 0px; box-shadow: none;",
                guide = this, 
                title = "Export button",
                text = "This button writes the data to a CSV file.",
                position = "top"
            )

    this.footer = footer

    
    def server(input, output, session):

        @reactive.calc
        def data_ready():
            return this._imports["data"].is_set()

        @reactive.calc()
        def GetData():
            req(data_ready())
            return this._imports["data"]()

        @this.debounce(2)
        @reactive.calc
        def Decimals():
            return input.Decimals()

        @this.record_code
        def _dtype_label_from_dtype(dtype) -> str:
            if pd.api.types.is_integer_dtype(dtype): 
                return "int"
            if pd.api.types.is_float_dtype(dtype):   
                return "float"
            if pd.api.types.is_bool_dtype(dtype):    
                return "bool"
            if pd.api.types.is_datetime64_any_dtype(dtype): 
                return "datetime"
            return "str"

        @reactive.calc
        @this.record_code
        def PreparedData():
            """
            Returns a Pandas DataFrame ready for Shiny DataTable:
            - Convert to native
            - Head/tail preview
            - Geometries reformatted
            - Numeric data rounded
            """
            df = GetData()
            if this.isFullScreen():
                df = df.sample(n = 10**input.MaxObs(), mode = "random", keep_geometry = True)
            else:
                df = df.sample(n = 8, mode = "headtail", keep_geometry = True)
            # materialize to native (richest form)
            df = df.to_native() if hasattr(df, "to_native") else df
            if hasattr(df, "to_pandas"):     # e.g., Polars
                df = df.to_pandas()
            long_geom = not input.Bounded()
            add_type_header = True,    # add "\n<type>" in headers
            include_crs_in_header = True,
            def _format_geometry_series_for_display(ser: gpd.GeoSeries, long: bool) -> pd.Series:
                if long:
                    return ser.apply(lambda g: shapely.to_wkt(g) if g is not None else None)
                def _short(g):
                    if g is None: 
                        return None
                    t = getattr(g, "geom_type", None)
                    if t == "Point":
                        return f"Point({g.x:.4f}, {g.y:.4f})"
                    try:
                        minx, miny, maxx, maxy = g.bounds
                        return f"{t} bound by {minx:.4f},{miny:.4f} to {maxx:.4f},{maxy:.4f}"
                    except Exception:
                        return str(g)
                return ser.apply(_short)
            # geometry → string (WKT or compact summary)
            is_geo = isinstance(df, gpd.GeoDataFrame)
            geom_cols: List[str] = []
            active_name = None
            crs_map: Dict[str, str] = {}
            if is_geo:
                geom_cols = [c for c in df.columns if getattr(df[c].dtype, "name", None) == "geometry"]
                active_name = df.geometry.name if getattr(df, "geometry", None) is not None else (geom_cols[0] if geom_cols else None)
                for c in geom_cols:
                    s = df[c]
                    if getattr(s, "crs", None) is not None:
                        crs_map[c] = s.crs.to_string()
                    elif getattr(df, "crs", None) is not None:
                        crs_map[c] = df.crs.to_string()
                    else:
                        crs_map[c] = ""
                    # stringify for grid
                    df[c] = _format_geometry_series_for_display(s, long=long_geom)
            # header second line (dtype / geometry info)
            if add_type_header:
                # dtype map (use original df, not stringified copy)
                dtype_map = (df.dtypes if not hasattr(df, "dtypes") else df.dtypes)
                if isinstance(dtype_map, pd.Series):
                    dtype_map = dtype_map.to_dict()
                else:
                    dtype_map = pd.Series(dtype_map).to_dict()  # normalize
                new_cols: List[str] = []
                for c in df.columns:
                    if is_geo and c in geom_cols:
                        parts = ["geometry"]
                        if c == active_name:
                            parts.append("active")
                        if include_crs_in_header and crs_map.get(c):
                            parts.append(crs_map[c])
                        tag = " ".join(parts)
                        new_cols.append(f"{c}\n{tag}")
                    else:
                        dt = dtype_map.get(c, object)
                        new_cols.append(f"{c}\n{_dtype_label_from_dtype(dt)}")
                df.columns = new_cols
            # rounding: numeric columns only
            if Decimals() is not None:
                try:
                    num_cols = df.select_dtypes(include=["float"]).columns
                    if len(num_cols) > 0:
                        df.loc[:, num_cols] = df.loc[:, num_cols].round(int(Decimals()))
                except Exception:
                    # be forgiving if any backend oddities slip through
                    pass
            return df

        @output
        @render.ui
        def DataTable():
            if this._imports["data"].is_set():
                return ui.output_data_frame(id = "DataTable2")
            else:
                return None

        @output
        @render.data_frame
        def DataTable2():
            full = this.isFullScreen()
            return render.DataTable(PreparedData(), summary=full, filters=full, width="100%", height="98%")

        @output
        @render.download(filename=f"{this.namespace}_data.csv")
        def Export():
            if data_ready():
                df = GetData()
                import io
                buf = io.StringIO()
                df.to_csv(buf, index=False, header = True)
                buf.seek(0)
                yield buf.read()


    this.server = server

    return this

if Module.running_under_tests():
    this = instance()
    app = Module.app(modules = {this.ns: this})
elif Module.running_directly(name =__name__):
    this = instance()
    Module.run(modules = {this.ns: this})
