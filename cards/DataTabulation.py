import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import geopandas as gpd
import pandas as pd  # needed for test / solo modes
import pyarrow  # noqa: F401
import shapely
from faicons import icon_svg as icon
from shiny import render, req, ui

from card import Card
from cyclic_pandas import is_cyclic
from geometry_pandas import is_geometry
from list_pandas import is_list
from module import Module
from proxyData import ProxyData
from text_pandas import is_text


def instance():
    """
    Creates an instance of Card configured as "dataTable".
    """
    this = Card(file=__file__, mutable=False) # "mutable" means it can change the pxd - probably with a commit button
    this.long_name = "Data tabulation"
    this.description = "This card enables the data to be listed and searched."
    
    def front():
        return ui.output_ui( # Using dynamic data tables to avoid "sortable" problem of multiple tables
            id = "DataTable",
            title = "A quick view", 
            guide = this,
            text = 'A top and bottom sample of the data.',
            position = "left"
        ) 
    
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

        @this.suspendable(calc = True)
        def incomingProxyData():
            req(this._imports.is_set())
            return this._imports.get()

        @this.throttle(2)
        @this.suspendable(calc = True)
        def Decimals():
            return input.Decimals()
        
        @this.throttle(2)
        @this.suspendable(calc = True)
        def MaxObs():
            return 10**input.MaxObs()


        @this.record_code
        def _dtype_label_from_dtype(dtype) -> str:
            if is_cyclic(dtype):
                return "cyc"
            if is_text(dtype):
                return "txt"
            if is_geometry(dtype):
                return "geo"
            if is_list(dtype):
                return "bkt"
            if isinstance(dtype, pd.StringDtype):
                return "cde"
            if isinstance(dtype, pd.CategoricalDtype):
                return "ord" if dtype.ordered else "nom"
            if pd.api.types.is_integer_dtype(dtype):
                return "int"
            if pd.api.types.is_float_dtype(dtype):
                return "dec"
            if pd.api.types.is_bool_dtype(dtype):
                return "log"
            if pd.api.types.is_datetime64_any_dtype(dtype):
                return "dte"
            if pd.api.types.is_object_dtype(dtype):
                return "obj"
            return str(dtype)

        @this.suspendable(calc = True)
        @this.record_code
        def PreparedData():
            df = incomingProxyData() #Returns ProxyData
            if this.isFullScreen():
                df = df.sample(n = MaxObs(), mode = "random", keep_geometry = True)
            else:
                df = df.sample(n = 10, mode = "headtail", keep_geometry = True)
            return df

        @this.suspendable(calc = True)
        @this.record_code
        def CleanDf():
            """
            Returns a Pandas DataFrame ready for Shiny DataTable:
            - Convert to native
            - Geometries reformatted
            - Numeric data rounded
            """
            px = PreparedData() #Returns ProxyData
            df = px.to_native() if hasattr(px, "to_native") else px
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
                    except Exception:  # noqa: BLE001
                        return str(g)
                return ser.apply(_short)
            # geometry → string (WKT or compact summary)
            is_geo = isinstance(df, gpd.GeoDataFrame)
            geom_cols: list[str] = []
            active_name = None
            crs_map: dict[str, str] = {}
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
                dtype_map = df.dtypes
                if isinstance(dtype_map, pd.Series):
                    dtype_map = dtype_map.to_dict()
                else:
                    dtype_map = pd.Series(dtype_map).to_dict()  # normalize
                new_cols: list[str] = []
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
                except Exception:  # noqa: BLE001, S110
                    # be forgiving if any backend oddities slip through
                    pass
            return df

        @output
        @render.ui
        def DataTable():
            req(PreparedData() is not None)
            return ui.output_data_frame(id = "DataTable2")

        @output
        @render.data_frame
        def DataTable2():
            req(PreparedData() is not None)
            full = this.isFullScreen()
            return render.DataTable(CleanDf(), summary=full, filters=full, width="100%", height="98%")

        @output
        @render.download(filename=f"{this.namespace}_data.csv")
        def Export():
            req(incomingProxyData())
            df = incomingProxyData()  # export raw not preparedData
            import io
            buf = io.StringIO()
            df.to_csv(buf, index=False, header = True)
            buf.seek(0)
            yield buf.read()


    this.server = server

    return this

if Module.running_under_tests():
    this = instance()
    df = pd.DataFrame(
        {
            "y": [1, 0, 1, 0],
            "x1": [10.0, 11.0, 12.0, 13.0],
            "x2": ["A", "B", "A", "B"],
            "id": [100, 101, 102, 103],
            "part": ["Train", "Train", "Test", "Test"],
        }
    )
    pxd = ProxyData(_df = df, _name = "Test")
    this._imports.set(pxd)
    app = this.application()
elif Module.running_directly(name =__name__):
    this = instance()
    df = pd.read_csv( Card.ROOT / "data" / "Ass2.csv")
    pxd = ProxyData(_df = df, _name = "Ass2")
    this._imports.set(pxd)
    this.run()