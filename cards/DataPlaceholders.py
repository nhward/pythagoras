from pathlib import Path
import sys
import pandas as pd  # needed for test / solo modes
from shinywidgets import render_widget
import shinywidgets
from typing import Tuple, Dict, List, Any
import plotly.graph_objects as go
import plotly.colors as pc
import numpy as np
import geopandas as gpd
from shiny import ui, reactive, req, render

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from module import Module  # noqa: E402
from card import Card  # noqa: E402
from proxyData import ProxyData as Pxy  # noqa: E402

# Converts missing value placeholders to Na/NaN/NaT
# Ideally this follows the correct conversion of strings to their real datatype esp. Datetime

#TODO: test code generation

def instance():
    this = Card(name = "dataPlaceholders", mutable = False)
    this.long_name = "Missing value placeholders"
    this.description = "This card detects any potential missing-value placeholders and allows their replacement with NA."

    def settings():
        return ui.TagList(
            ui.input_selectize(
                id = "NA_Strings", 
                label = "Missing string-value placeholders", 
                choices =  ["NA","-","--", "N/A", "Missing", "Not Applicable", "Not Available"],
                selected = ["NA","-","--", "N/A", "Missing", "Not Applicable", "Not Available"],
                multiple = True,
                options=({
                    "placeholder": "Enter string values",
                    "create": True,
                }),
                guide = this,
                text = 'This comma-delimited list supplies placeholders for missing string-values. Entire string-values that match any of these will be replaced with NA. The search is case insensitive.',
                position = "left"),
            ui.input_checkbox(
                id = "NA_CaseSensitive", 
                label = "Use a case-sensitive search", 
                value = False,
                guide = this,
                text = 'Whether "N/A" different to "N/a", "n/a", "n/A".',
                position = "left"),
            ui.input_selectize(
                id = "NA_Integers", 
                label = "Missing integer-value placeholders", 
                choices =  [-9999,-999,-99, -1],
                selected = [-9999,-999,-99, -1],
                multiple = True,
                options=({
                    "placeholder": "Enter integer values",
                    "create": True,
                }),
                guide = this,
                text = 'This comma-delimited list supplies placeholders for missing numeric-values. Entire values that match any of these will be replaced with NA <em>provided</em> they correspond with the lowest recorded values.',
                position = "left"),
            ui.input_selectize(
                id = "NA_Floats", 
                label = "Missing decimal-value placeholders", 
                choices =  [-9999.99,-999.99,-99.99, -99.00, -1.00],
                selected = [-9999.99,-999.99,-99.99, -99.00, -1.00],
                multiple = True,
                options=({
                    "placeholder": "Enter decimal values",
                    "create": True,
                }),
                guide = this,
                text = 'This comma-delimited list supplies placeholders for missing decimal-values. Entire values that match any of these will be replaced with NaN <em>provided</em> they correspond with the lowest recorded values.',
                position = "left"),
            ui.input_checkbox(
                id = "NA_Extrema", 
                label = "Only replace extreme numeric values (at minimum or maximum)", 
                value = True,
                guide = this,
                text = 'Only replace numbers <em>provided</em> they correspond with the lowest or highest recorded values.',
                position = "left"),
            ui.input_selectize(
                id = "NA_DateTime", 
                label = "Missing date/time-value placeholders", 
                choices = ["0000-00-00", "0001-01-01", "1900-01-01", "0"],
                selected = ["0000-00-00", "0001-01-01", "1900-01-01", "0"],
                multiple = True,
                options=({
                    "placeholder": "Enter date-literal values",
                    "create": True,
                }),
                guide = this,
                text = 'This comma-delimited list supplies placeholders for missing date/time-values. Date values that match any of these will be replaced with NaT.',
                position = "left"),
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
                position = "left"),
        )

    this.settings = settings

    def front():
        return ui.navset_bar(
            ui.nav_panel(
                "All variables",
                shinywidgets.output_widget(  ## needed as plotly workaround
                    id = "AllChart",
                    title = "All variables' placeholders chart",
                    guide = this,
                    text = 'A chart of the occurence of placeholders throughout the data. This shows all variable types (excluding any geometries).',
                    position = "left"
                    )
            ),
            ui.nav_panel(
                "Integer",
                shinywidgets.output_widget(
                    id = "IntegerChart",
                    title = "Integer variables' placeholders chart",
                    guide = this,
                    text = 'A chart of the occurence of placeholders throughout the data. This shows all integer types.',
                    position = "left"
                )
            ),
            ui.nav_panel(
                "Decimal",
                shinywidgets.output_widget(
                    id = "FloatChart",
                    title = "Decimal variables' placeholders chart",
                    guide = this,
                    text = 'A chart of the occurence of placeholders throughout the data. This shows all decimal (floating point) types.',
                    position = "left"
                )
            ),
            ui.nav_panel(
                "Character",
                shinywidgets.output_widget(
                    id = "CharacterChart",
                    title = "Character variables' placeholders chart",
                    guide = this,
                    text = 'A chart of the occurence of placeholders throughout the data. This shows all character types.',
                    position = "left"
                )
            ),
            ui.nav_panel(
                "Dates & Times",
                shinywidgets.output_widget(
                    id = "DateChart",
                    title = "Date/time variables' placeholders chart",
                    guide = this,
                    text = 'A chart of the occurence of placeholders throughout the data. This shows all datetime types.',
                    position = "left"
                )
            ),
            title = None,
            id = "Navset", 
            padding = 0, 
            fillable = True
        )
    
    this.front = front

    def back():
        return ui.TagList(
            ui.card_header("Placeholder Summary", class_ = "text-primary text-center"),
            ui.output_ui(
                id = "Summary", 
                guide = this, title = "Placeholder Summary", position = "top",
                text = "This summary shows the number of placeholders after any acknowledgements.",
                style = "font-size: 0.85rem; line-height: 1.1;"
            )
        )
    
    this.back = back

    def footer():
        return ui.input_checkbox_group(
            id = "Replace",
            label = None,
            choices=[],
            inline=True,
            guide = this, title = "Replace buttons", position = "top",
            text = "Buttons for replacing placeholders by converting them to missing values. There is a button for each type of placeholder found. The changes can be reversed.",
        )

    this.footer = footer

    def server(input, output, session):
    
        #@this.debounce(2)
        @reactive.calc
        def MaxObs():
            return 10**input.MaxObs()

        # isFront
        @reactive.calc
        def isFront():
            return input.FlipButton() % 2 == 0


        # isFullScreen
        @reactive.calc()
        def isFullScreen():
            if isinstance(input.Card_full_screen(), bool):
                return input.Card_full_screen()
            return False

        @reactive.calc
        def incommingData():
            req(this._imports.is_set())
            data = this._imports()
            # this.resume()   #TODO: review
            # await this.show(session)  #TODO: review
            return data

        Codes = reactive.value(None)
        Legend = reactive.value(None)

        @reactive.calc
        @this.record_code
        def PreparedData():
            d = incommingData()
            sample = d.sample(n = MaxObs(), mode = "random", keep_geometry = False)
            return sample


        @reactive.calc
        @this.record_code
        def Sentinels():
            return {
                "int":      input.NA_Integers(),
                "float":    input.NA_Floats(),
                "str":      input.NA_Strings(),
                "datetime": input.NA_DateTime(),
            }

        @reactive.calc
        @this.record_code
        def FixedData():
            sample = PreparedData()
            sentinels = [s.removeprefix("Replace ") for s in input.Replace()]
            fixed = ResolvePlaceholders(data = sample, sentinels=sentinels, extrema=input.NA_Extrema(), case_sensitive=input.NA_CaseSensitive())
            codes_df, legend = PlaceholderCodes(
                data = fixed,
                sentinels=Sentinels(),
                drop_geometry=True,
                extrema=input.NA_Extrema(),
                case_sensitive=input.NA_CaseSensitive(),
            )
            Codes.set(codes_df)
            Legend.set(legend)
            return Pxy.from_native(fixed)

        @reactive.effect
        def UpdateButtons():
            sample = PreparedData()
            codes_df, legend = PlaceholderCodes(
                data = sample,
                sentinels=Sentinels(),
                drop_geometry=True,
                extrema=input.NA_Extrema(),
                case_sensitive=input.NA_CaseSensitive(),
            )
            flat = pd.Series(codes_df.to_numpy().ravel())
            used_codes = pd.to_numeric(flat, errors="coerce").dropna().astype(int).unique().tolist()
            used_codes = sorted(k for k in used_codes if k not in (0, 1))
            reduced_labels = [str(legend.get(k, f"Code {k}")) for k in used_codes]
            choices = [f"Replace {lab}" for lab in reduced_labels]
            with reactive.isolate():
                previous = input.Replace() or []
            selected = [c for c in previous if c in choices]
            ui.update_checkbox_group(id="Replace", choices=choices, selected=selected)

        def empty_plotly(message="No data to display", subtext=None):
            txt = f"<b>{message}</b>" + (f"<br><span style='font-size:0.9em;color:#6c757d'>{subtext}</span>" if subtext else "")
            fig = go.Figure()
            fig.add_annotation(
                text=txt, x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, align="center",
                font=dict(size=18, color="#6c757d")
            )
            fig.update_layout(
                xaxis=dict(visible=False), 
                yaxis=dict(visible=False),
                plot_bgcolor="#f8f9fa", # make same as card
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=0, b=0),
                hovermode=False, 
                showlegend=False
            )
            fw = go.FigureWidget(fig)
            fw._config = (getattr(fw, "_config", {}) | {"displayModeBar": False, "displaylogo": False, "responsive": True})
            return fw

        def _select_cols(df: pd.DataFrame, bucket: str) -> list[str]:
            if bucket == "int":
                return df.select_dtypes(include=["int", "Int64", "integer"]).columns
            if bucket == "float":
                return df.select_dtypes(include=["float", "Float64", "floating"]).columns
            if bucket == "str":
                return df.select_dtypes(include=["string", "object"]).columns
            if bucket == "datetime":
                return df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns
            return df.columns

        @this.record_code
        def _placeholder_chart(codes_df: pd.DataFrame, legend: dict, *, fs: bool) -> go.FigureWidget:
            x = codes_df.index.astype(str).tolist()
            y = list(codes_df.columns)
            z = codes_df.to_numpy(dtype=float).T  # vars x obs
            non_nan = z[~np.isnan(z)]
            if not non_nan.size:
                return empty_plotly("No data to display")
            present_codes = sorted(set(non_nan.astype(int)))
            fig = go.Figure()
            palette = pc.qualitative.Set3
            for code in present_codes:
                label = legend.get(code, f"Code {code}")
                color = palette[code % len(palette)]
                hits = (z == code)
                z_mask = [[1.0 if v else None for v in row] for row in hits]
                fig.add_trace(go.Heatmap(
                    z=z_mask, zmin=0, zmax=1,
                    x=x, y=y,
                    colorscale=[[0, color], [1, color]],
                    text=np.where(hits, label, None).tolist() if fs and code > 1 else None,
                    hoverinfo="text" if fs else "none",
                    hovertemplate=("Variable: %{y}<br>Observation: %{x}<br>"
                                "Placeholder: %{text}<extra></extra>") if fs and code > 1 else None,
                    hoverongaps=False,
                    name=label,
                    legendgroup=f"placeholder({code})",
                    showlegend=fs,
                    showscale=False
                ))
            fig.update_layout(
                xaxis=dict(title="Observation", type="category"),
                yaxis=dict(title="Variables", type="category", autorange="reversed"),
                plot_bgcolor="#e5ecf6",
                margin=dict(l=2, r=10 if fs else 2, t=2, b=2),
                showlegend=fs,
                legend=dict(x=1.003, xanchor="left", y=0.0, yanchor="bottom", itemsizing="constant"),
                modebar=dict(orientation="v"),
            )
            fw = go.FigureWidget(fig)
            fw._config = (getattr(fw, "_config", {}) | {
                "displayModeBar": bool(fs),
                "displaylogo": False,
                "responsive": True,
            })
            return fw
    
        @output
        @render_widget
        @this.record_code
        def AllChart():
            FixedData()
            codes_df = Codes.get()
            legend = Legend.get()
            return _placeholder_chart(codes_df, legend, fs=isFullScreen())

        @output
        @render_widget
        @this.record_code
        def IntegerChart():
            data = FixedData()
            codes_df = Codes.get()
            legend = Legend.get()
            cols = _select_cols(data, "int")
            return empty_plotly("No integer data to display") if cols.__len__ == 0 else \
                _placeholder_chart(codes_df[cols], legend, fs=isFullScreen())

        @output
        @render_widget
        @this.record_code
        def FloatChart():
            data = FixedData()
            codes_df = Codes.get()
            legend = Legend.get()
            cols = _select_cols(data, "float")
            return empty_plotly("No decimal data to display") if cols.__len__ == 0 else \
                _placeholder_chart(codes_df[cols], legend, fs=isFullScreen())

        @output
        @render_widget
        @this.record_code
        def CharacterChart():
            data = FixedData()
            codes_df = Codes.get()
            legend = Legend.get()
            cols = _select_cols(data, "str")
            return empty_plotly("No character data to display") if cols.__len__ == 0 else \
                _placeholder_chart(codes_df[cols], legend, fs=isFullScreen())

        @output
        @render_widget
        @this.record_code
        def DateChart():
            data = FixedData()
            codes_df = Codes.get()
            legend = Legend.get()
            cols = _select_cols(data, "datetime")
            return empty_plotly("No datetime data to display") if cols.__len__ == 0 else \
                _placeholder_chart(codes_df[cols], legend, fs=isFullScreen())


        @reactive.event(input.Replace)
        @this.record_code
        def passthrough():
            full  = incommingData()
            sentinels = [s.removeprefix("Replace ") for s in input.Replace()]
            df = ResolvePlaceholders(data = full, sentinels=sentinels, extrema=input.NA_Extrema(), case_sensitive=input.NA_CaseSensitive(), drop_geometyry = False)
            pxy = pxd.ProxyData.from_native(df).with_roles(full.RoleMap)
            this._exports.set(pxy)


        @reactive.calc
        @this.record_code
        def build_summary_df():
            codes_df, legend = PlaceholderCodes(
                data = FixedData(),
                sentinels=Sentinels(),
                drop_geometry=True,
                extrema=input.NA_Extrema(),
                case_sensitive=input.NA_CaseSensitive(),
            )
            # counts per variable for all codes at once
            counts = (
                codes_df
                .apply(pd.Series.value_counts, dropna=True)  # rows=code, cols=var
                .fillna(0).astype(int)
                .sort_index()
                .T                                           # rows=var, cols=code
            )
            if counts.empty:
                return pd.DataFrame({"Note": ["No placeholders present"]})
            # Drop 1 (Not Missing) if present
            counts = counts[[c for c in counts.columns if int(c) != 1]]
            label_map = {int(k): str(v) for k, v in (legend or {}).items()}
            counts = counts.rename(columns=lambda c: label_map.get(int(c), f"Code {int(c)}"))
            return counts.rename_axis("Variable").reset_index()

        @output
        @render.ui
        def Summary():
            df = build_summary_df()   # your table of counts                
            def fmt(val):
                if isinstance(val, (int, float)) and val != 0:
                    return f'<td style="font-weight:700;">{val}</td>'
                return f"<td>{val}</td>"
            rows = []
            # header
            rows.append("<tr>" + "".join(f"<th>{c}</th>" for c in df.columns) + "</tr>")
            # body
            rows.append("<tbody>")
            for _, row in df.iterrows():
                cells = [fmt(v) for v in row]
                rows.append("<tr>" + "".join(cells) + "</tr>")
            html = "<table class='table table-sm table-bordered'>" + "".join(rows) + "</tbody></table>"
            return ui.HTML(html)

        @this.record_code
        def PlaceholderCodes(data, sentinels: Dict[str, List[Any]], float_eps: float = 1e-9, drop_geometry: bool = True,
            extrema = True, case_sensitive = False) -> Tuple[pd.DataFrame, Dict[str, Dict[int, str]]]:
            """
            Build a pandas DataFrame of integer codes for heatmap chart:
            - 0            => missing (NaN/NaT/None/"")
            - 1..k         => dtype-specific sentinel values (in order)
            - k+1          => everything else

            data : Something convertable to a Pandas dataframe 
            sentinels: "int", "float", "str", "datetime" e.g. {"int":[-999,-99,-9], "float":[-9999.0,-999.0], "str":["", "NA","N/A"], "datetime":["0000-00-00","0001-01-01","1900-01-01","0"]}
            float_eps: when two floats are equal
            drop_geometry: whether to reduce to tabular columns only
            extrema: whether to only identify extreme placeholders
            case_sensitive: how to match character placeholders 
            Returns: (codes_df, legend)
            """
            req(data is not None)
            if isinstance(data, pd.DataFrame):
                df = data
            elif isinstance(data, Pxy):
                df = data._df
            elif hasattr(data, "to_pandas"):      # polars -> pandas
                df = data.to_pandas()
            else:
                raise ValueError(f"Unknown dataset type supplied: {type(data)}")
                
            if isinstance(df, gpd.GeoDataFrame) and drop_geometry:
                # drop all geometry-typed columns
                geom_cols = [c for c in df.columns if getattr(df[c].dtype, "name", None) == "geometry"]
                df = df.drop(columns=geom_cols).copy()
            else:
                df = pd.DataFrame(df).copy()

            # Create a new DataFrame of same shape, all int8 with 1 (initially signifying not-missing)
            codes = pd.DataFrame(
                {col: pd.Series([1] * len(df), dtype="Int8") for col in df.columns},
                index=df.index
            )
            # change the codes entries where data is missing
            for col in df.columns:
                v = df[col]
                missing = v.isna()
                codes.loc[missing, col] = 0

            legend = {0: "Missing", 1: "Not Missing"}
            k = 2

            # Precompute the target columns per type once
            type_to_cols = {
                "int":      df.select_dtypes(include=["integer"]).columns,           # nullable Int* dtypes
                "float":    df.select_dtypes(include=["floating"]).columns,          # float/Float64
                "str":      df.select_dtypes(include=["string", "object"]).columns,
                "datetime": df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns,
            }

            for dtype_key, sent_list in (sentinels or {}).items():
                cols = type_to_cols.get(dtype_key)
                if not cols is not None or len(cols) == 0:
                    continue
                for sent in (sent_list or []):
                    for col in cols:
                        s = df[col]
                        # Build a mask with dtype-appropriate comparison
                        if dtype_key == "datetime":
                            # parse sentinel into datetime; invalid parses become NaT and won't match
                            dt = pd.to_datetime(sent, errors="coerce")
                            mask = (s == dt)
                        elif dtype_key == "float":
                            # safe float compare; avoid string/None crashing float()
                            try:
                                f = float(sent)
                                if not extrema or (min(s) == f or max(s) == f):
                                    mask = abs(s.astype("float64") - f) < float_eps
                                else:
                                    mask = pd.Series(False, index=s.index)
                            except Exception:
                                mask = pd.Series(False, index=s.index)
                        elif dtype_key == "int":
                            try:
                                i = int(sent)
                                if not extrema or (min(s) == i or max(s) == i):
                                    mask = (s == i) # works for pandas nullable Int* dtypes
                                else:
                                    mask = pd.Series(False, index=s.index)
                            except Exception:
                                mask = pd.Series(False, index=s.index)
                        elif dtype_key == "str":
                            if case_sensitive:
                                mask = (
                                    s.astype("string")
                                    .fillna("")
                                    .str
                                    .eq(sent))
                            else:
                                mask = (
                                    s.astype("string")
                                    .fillna("")
                                    .str.casefold()
                                    .eq(sent.casefold())
                                )
                        else:
                            mask = pd.Series(False, index=s.index)
                        # assign this sentinel's code
                        if mask.any():
                            codes.loc[mask, col] = k
                    legend[k] = f"{dtype_key}: {sent}"
                    k += 1
            return (codes, legend)

        @this.record_code
        def ResolvePlaceholders(data, sentinels: str = None, float_eps: float = 1e-9, extrema = True, case_sensitive = False):
            """
            Builds a Pandas data frame with the specified sentinels converted to the relevant missing value indictor.
            
            data : Something convertable to a Pandas dataframe 
            sentinels: "int", "float", "str", "datetime" e.g. {"int":[-999,-99,-9], "float":[-9999.0,-999.0], "str":["", "NA","N/A"], "datetime":["0000-00-00","0001-01-01","1900-01-01","0"]}
            float_eps: when two floats are equal
            drop_geometry: whether to reduce to tabular columns only
            extrema: whether to only identify extreme placeholders
            case_sensitive: how to match character placeholders 
            Returns: A modified Pandas dataframe copy
            """
            req(data is not None)
            if isinstance(data, pd.DataFrame):
                df = data
            elif isinstance(data, Pxy):
                df = data._df
            elif hasattr(data, "to_pandas"):      # polars -> pandas
                df = data.to_pandas()
            else:
                raise ValueError(f"Unknown dataset type supplied: {type(data)}")

            if sentinels is None or len(sentinels) == 0:
                return df
            df = pd.DataFrame(df).copy()
            type_to_cols = {
                "int":      df.select_dtypes(include=["integer"]).columns,           # nullable Int* dtypes
                "float":    df.select_dtypes(include=["floating"]).columns,          # float/Float64
                "str":      df.select_dtypes(include=["string", "object"]).columns,
                "datetime": df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns,
            }
            for sent in sentinels:
                vtype, placeholder = sent.split(sep = ": ", maxsplit = 2)
                for col in type_to_cols[vtype]:
                    s = df[col]
                    if vtype == "datetime":
                        mask = (s == placeholder.asType("datetime"))
                        df.loc[mask, col] = pd.NaT
                    elif vtype == "float":
                        try:
                            f = float(placeholder)
                            if not extrema or (min(s) == f or max(s) == f):
                                mask = abs(s.astype("float64") - f) < float_eps
                                this.debug(msg = f"Replaced {sum(mask)} {placeholder} {vtype} placeholders in {col}")
                                df.loc[mask,col] = pd.NA
                        except Exception:
                            pass
                    elif vtype == "int":
                        try:
                            i = int(sent)
                            if not extrema or (min(s) == i or max(s) == i):
                                mask = (s == i) # works for pandas nullable Int* dtypes
                                df.loc[mask,col] = pd.NA
                        except Exception:
                            pass
                    elif vtype == "str":
                        if case_sensitive:
                            mask = (
                                s.astype("string")
                                .fillna("")
                                .str
                                .eq(str(placeholder))
                            )
                        else:
                            mask = (
                                s.astype("string")
                                .fillna("")
                                .str.casefold()
                                .eq(str(placeholder).casefold())
                            )
                        this.debug(msg = f"Replaced {sum(mask)} {placeholder} {vtype} placeholders in {col}")
                        df.loc[mask,col] = pd.NA
            return df


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
    pxd = Pxy.from_native(df)
    pxd.name = "Test"
    this._imports.set(pxd)
    app = Module.app(modules = {this.ns: this})
elif Module.running_directly(name =__name__):
    this = instance()
    df = pd.read_csv( Card.ROOT / "data" / "Ass2.csv")
    pxd = Pxy.from_native(df)
    pxd.name = "Ass2"
    this._imports.set(pxd)
    Module.run(modules = {this.ns: this})
