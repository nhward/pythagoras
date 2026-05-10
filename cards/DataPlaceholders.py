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

def instance():
    """
    Creates an instance of dataPlaceholders class.
    """
    this = Card(name = "dataPlaceholders", mutable = True) # "mutable" means it can change the pxd - probably with a commit button
    this.long_name = "Missing value placeholders"
    this.description = "This card detects any potential missing-value placeholders and allows their replacement with NA."

    #############################
    # Define the user-interface #
    #############################

    def settings() -> ui.TagList:
        """
        These settings related ui elements appear in the sidebar of the card. 
        The optional 'guide', 'text', 'position' and 'priority' parameters of the ui elements allows for the Guide.
        """
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

    this.settings = settings ## The above "setting" function must be assigned to the instance 

    def front() -> ui.TagList:
        """
        These ui elements appear in the front of the card. 
        The optional 'guide', 'text', 'position' and 'priority' parameters of the ui elements allows for the Guide.
        """
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
    
    this.front = front   ## The above "front" function must be assigned to the instance

    def back() -> ui.TagList:
        """
        These ui elements appear in the back of the card. 
        The optional 'guide', 'text', 'position' and 'priority' parameters of the ui elements allows for the Guide.
        """
        return ui.TagList(
            ui.card_header("Placeholder Summary", class_ = "text-primary text-center"),
            ui.output_ui(
                id = "Summary", 
                guide = this, title = "Placeholder Summary", position = "top",
                text = "This summary shows the number of placeholders after any acknowledgements.",
                style = "font-size: 0.85rem; line-height: 1.1;"
            )
        )
    
    this.back = back  ## The above "back" function must be assigned to the instance

    def footer() -> ui.TagList:
        """
        These ui elements appear in the footer of the card (but only on the front). 
        The optional 'guide', 'text', 'position' and 'priority' parameters of the ui elements allows for the Guide.
        """
        return ui.input_checkbox_group(
            id = "Replace",
            label = None,
            choices=[],
            inline=True,
            guide = this, title = "Replace buttons", position = "top",
            text = "Buttons for replacing placeholders by converting them to missing values. There is a button for each type of placeholder found. The changes can be reversed.",
        )

    this.footer = footer  ## The above "footer" function must be assigned to the instance


    ########################
    # Define the behaviour #
    ########################

    def server(input, output, session):

        @this.suspendable(calc = True)
        def incomingProxyData():
            this._imports.get()
            req(this._imports.is_set())
            return this._imports.get()

        @this.throttle(2)
        @this.suspendable(calc = True)
        def MaxObs():
            return 10**input.MaxObs()


        @this.suspendable(calc = True)
        @this.record_code
        def PreparedData():
            d = incomingProxyData()
            sample = d.sample(n = MaxObs(), mode = "random", keep_geometry = False)
            return sample


        @this.suspendable(calc = True)
        @this.record_code
        def Sentinels():
            return {
                "int":      input.NA_Integers(),
                "float":    input.NA_Floats(),
                "str":      input.NA_Strings(),
                "datetime": input.NA_DateTime(),
            }


        @this.suspendable()
        def UpdateButtons():
            rawstate = RawCodes()
            flat = pd.Series(rawstate["codes"].to_numpy().ravel())
            used_codes = pd.to_numeric(flat, errors="coerce").dropna().astype(int).unique().tolist()
            used_codes = sorted(k for k in used_codes if k not in (0, 1))
            reduced_labels = [str(rawstate["legend"].get(k, f"Code {k}")) for k in used_codes]
            choices = [f"Replace {lab}" for lab in reduced_labels]
            with reactive.isolate():
                previous = input.Replace() or []
            selected = [c for c in previous if c in choices]
            ui.update_checkbox_group(id="Replace", choices=choices, selected=selected)


        def empty_plotly(message="No data to display", subtext=None):
            this.log.debug("Empty chart drawn")
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

        # @this.record_code
        # def _placeholder_chart(codes_df: pd.DataFrame, legend: dict, *, fs: bool) -> go.Figure:
        #     x = np.arange(codes_df.shape[0])   ##x = codes_df.index.astype(str).tolist()
        #     y = codes_df.columns.astype(str).tolist()
        #     z = codes_df.to_numpy(dtype=np.int16, copy=False).T
        #     non_nan = z[~np.isnan(z)]
        #     if not non_nan.size:
        #         return empty_plotly("No data to display")
        #     present_codes = sorted(np.unique(non_nan.astype(int)))
        #     palette = pc.qualitative.Set3
        #     zmin = min(present_codes)
        #     zmax = max(present_codes)
        #     this.log.debug("Chart drawn")
        #     # --- discrete colorscale ---
        #     if zmin == zmax:
        #         color = palette[zmin % len(palette)]
        #         colorscale = [[0, color], [1, color]]
        #     else:
        #         colorscale = []
        #         for code in present_codes:
        #             pos = (code - zmin) / (zmax - zmin)
        #             color = palette[code % len(palette)]
        #             colorscale.append([pos, color])
        #             colorscale.append([pos, color])
        #     fig = go.Figure()
        #     # ---------------------------
        #     # 1. Base heatmap (no hover)
        #     # ---------------------------
        #     fig.add_trace(go.Heatmap(
        #         z=z,
        #         x=x,
        #         y=y,
        #         zmin=zmin,
        #         zmax=zmax,
        #         colorscale=colorscale,
        #         showscale=False,
        #         hoverinfo="skip",   # 👈 disables all hover here
        #         hoverongaps=False,
        #     ))
        #     # ---------------------------
        #     # 2. Hover layer (codes > 1 only)
        #     # ---------------------------
        #     if fs:
        #         mask = (~np.isnan(z)) & (~np.isin(z, [0, 1]))
        #         if mask.any():
        #             z_hover = np.full(z.shape, None, dtype=object)
        #             z_hover[mask] = z[mask]
        #             text = np.empty(z.shape, dtype=object)
        #             text[:] = ""
        #             codes = z[mask].astype(int)
        #             text[mask] = [legend.get(c, f"Code {c}") for c in codes]
        #             fig.add_trace(go.Heatmap(
        #                 z=z_hover,
        #                 x=x,
        #                 y=y,
        #                 showscale=False,
        #                 colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
        #                 hovertemplate=(
        #                     "Variable: %{y}<br>"
        #                     "Observation: %{x}<br>"
        #                     "Placeholder: %{text}<extra></extra>"
        #                 ),
        #                 text=text,
        #                 hoverongaps=False,
        #                 showlegend=False,
        #             ))
        #         for code in present_codes:
        #             fig.add_trace(go.Scatter(
        #                 x=[None],
        #                 y=[None],
        #                 mode="markers",
        #                 marker=dict(
        #                     size=10,
        #                     color=palette[code % len(palette)],
        #                     symbol="square",
        #                 ),
        #                 name=legend.get(code, f"Code {code}"),
        #                 showlegend=True,
        #                 hoverinfo="skip",
        #             ))
        #     fig.update_layout(
        #         xaxis=dict(title="Observation", type="category"),
        #         yaxis=dict(title="Variables", type="category", autorange="reversed"),
        #         plot_bgcolor="#e5ecf6",
        #         margin=dict(l=2, r=35 if fs else 2, t=2, b=2),
        #         showlegend=fs,
        #         legend=dict(x=1.003, xanchor="left", y=0.0, yanchor="bottom", itemsizing="constant"),
        #         modebar=dict(orientation="v"),
        #     )
        #     fw = go.FigureWidget(fig)
        #     fw._config = (getattr(fw, "_config", {}) | {
        #         "displayModeBar": bool(fs),
        #         "displaylogo": False,
        #         "responsive": True,
        #     })
        #     return fw


        @this.record_code
        def _placeholder_chart(codes_df: pd.DataFrame, legend: dict, *, fs: bool) -> go.Figure:
            if codes_df.empty:
                return empty_plotly("No data to display")
            y = codes_df.columns.astype(str).tolist()
            # Keep codes compact. No NaNs should exist: 0 = missing, 1 = not missing.
            z = codes_df.to_numpy(dtype=np.int16, copy=False).T
            present_codes = np.unique(z).astype(int).tolist()
            if len(present_codes) == 0:
                return empty_plotly("No data to display")
            this.log.debug(f"Chart drawn: z shape={z.shape}, cells={z.size:,}")
            palette = pc.qualitative.Set3
            zmin = min(present_codes)
            zmax = max(present_codes)
            if zmin == zmax:
                color = palette[zmin % len(palette)]
                colorscale = [[0, color], [1, color]]
            else:
                colorscale = []
                for code in present_codes:
                    pos = (code - zmin) / (zmax - zmin)
                    color = palette[code % len(palette)]
                    colorscale.append([pos, color])
                    colorscale.append([pos, color])
            fig = go.Figure()
            fig.add_trace(go.Heatmap(
                z=z,
                zmin=zmin,
                zmax=zmax,
                y=y,
                colorscale=colorscale,
                showscale=False,
                hoverinfo="skip",
                hoverongaps=False,
                zsmooth=False,
            ))
            if fs:
                mask = z > 1

                if mask.any():
                    yy, xx = np.where(mask)
                    codes = z[yy, xx].astype(int)

                    hover_text = [
                        (
                            f"<b>{legend.get(c, f'Code {c}')}</b><br>"
                            f"Variable: {y[row]}<br>"
                            f"Observation: {col}"
                        )
                        for row, col, c in zip(yy, xx, codes)
                    ]

                    hover_colors = [palette[c % len(palette)] for c in codes]

                    fig.add_trace(go.Scatter(
                        x=xx,
                        y=[y[i] for i in yy],
                        mode="markers",
                        marker=dict(
                            size=10,
                            opacity=0,
                            color=hover_colors,
                        ),
                        text=hover_text,
                        hovertemplate="%{text}<extra></extra>",
                        hoverlabel=dict(
                            bgcolor=hover_colors,
                            bordercolor=hover_colors,
                            font=dict(color="black"),
                        ),
                        showlegend=False,
                        hoverinfo="text",
                    ))

                # Legend-only traces
                for code in present_codes:
                    fig.add_trace(go.Scatter(
                        x=[None],
                        y=[None],
                        mode="markers",
                        marker=dict(
                            size=10,
                            color=palette[code % len(palette)],
                            symbol="square",
                        ),
                        name=legend.get(code, f"Code {code}"),
                        showlegend=True,
                        hoverinfo="skip",
                    ))
            fig.update_layout(
                xaxis=dict(title="Observation"),
                yaxis=dict(title="Variables", type="category", autorange="reversed"),
                plot_bgcolor="#e5ecf6",
                margin=dict(l=2, r=35 if fs else 2, t=2, b=2),
                showlegend=fs,
                legend=dict(
                    x=1.003,
                    xanchor="left",
                    y=0.0,
                    yanchor="bottom",
                    itemsizing="constant",
                ),
                modebar=dict(orientation="v"),
            )
            fig._config = {
                "displayModeBar": bool(fs),
                "displaylogo": False,
                "responsive": True,
            }
            return fig

        @output
        @render_widget
        @this.record_code
        def AllChart():
            state = CorrectedState()
            codes_df = state["codes"]
            legend = state["legend"]
            chart = _placeholder_chart(codes_df, legend, fs=this.isFullScreen())
            return chart


        @output
        @render_widget
        @this.record_code
        def IntegerChart():
            state = CorrectedState()
            codes_df = state["codes"]
            legend = state["legend"]
            fixed = state["fixed"]
            cols = _select_cols(fixed, "int")
            if len(cols) == 0:
                return empty_plotly("No integer data to display")
            else:
                return _placeholder_chart(codes_df[cols], legend, fs=this.isFullScreen())


        @output
        @render_widget
        @this.record_code
        def FloatChart():
            state = CorrectedState()
            codes_df = state["codes"]
            legend = state["legend"]
            fixed = state["fixed"]
            cols = _select_cols(fixed.to_native(), "int")
            if len(cols) == 0:
                return empty_plotly("No decimal data to display")
            else:
                return _placeholder_chart(codes_df[cols], legend, fs=this.isFullScreen())


        @output
        @render_widget
        @this.record_code
        def CharacterChart():
            state = CorrectedState()
            codes_df = state["codes"]
            legend = state["legend"]
            fixed = state["fixed"]
            cols = _select_cols(fixed.to_native(), "str")
            if len(cols) == 0:
                return empty_plotly("No character data to display")
            else:
                return _placeholder_chart(codes_df[cols], legend, fs=this.isFullScreen())


        @output
        @render_widget
        @this.record_code
        def DateChart():
            state = CorrectedState()
            codes_df = state["codes"]
            legend = state["legend"]
            fixed = state["fixed"]
            cols = _select_cols(fixed.to_native(), "datetime")
            if len(cols) == 0:
                return empty_plotly("No datetime data to display")
            else:
                return _placeholder_chart(codes_df[cols], legend, fs=this.isFullScreen())


        @this.suspendable(calc = True)
        @this.record_code
        def TransformedData():
            full  = incomingProxyData()
            sentinels = [s.removeprefix("Replace ") for s in input.Replace()]
            df = ResolvePlaceholders(data = full, sentinels=sentinels, extrema=input.NA_Extrema(), case_sensitive=input.NA_CaseSensitive(), drop_geometry = False)
            pxy = Pxy(_df=df, _roles=full.role_map, _name=full.name)
            pxy.name = full.name
            return pxy

        @this.suspendable(triggers = [TransformedData])
        def export():
            this._exports.set(TransformedData())


        @this.suspendable(calc = True)
        @this.record_code
        def build_summary_df():
            state = CorrectedState()
            codes_df = state["codes"]
            legend = state["legend"]
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

        
        def PlaceholderCodes(data, sentinels: Dict[str, List[Any]], float_eps: float = 1e-9, drop_geometry: bool = True, 
        extrema: bool = True, case_sensitive: bool = False) -> Tuple[pd.DataFrame, Dict[int, str]]:
            req(data is not None)
            if isinstance(data, pd.DataFrame):
                df = data
            elif isinstance(data, Pxy):
                df = data.to_native()
            else:
                raise ValueError(f"Unknown dataset type supplied: {type(data)}")
            if isinstance(df, gpd.GeoDataFrame) and drop_geometry:
                geom_cols = [
                    c for c in df.columns
                    if getattr(df[c].dtype, "name", None) == "geometry"
                ]
                df = df.drop(columns=geom_cols)
            else:
                df = pd.DataFrame(df)
            # Fast dense int8 array: 1 = not missing, 0 = missing
            codes_arr = np.where(df.isna().to_numpy(), 0, 1).astype(np.int8, copy=False)
            col_pos = {col: i for i, col in enumerate(df.columns)}
            legend = {0: "Missing", 1: "Not Missing"}
            k = 2
            type_to_cols = {
                "int": df.select_dtypes(include=["integer"]).columns,
                "float": df.select_dtypes(include=["floating"]).columns,
                "str": df.select_dtypes(include=["string", "object"]).columns,
                "datetime": df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns,
            }
            # Precompute numeric extrema once
            int_extrema = {}
            float_extrema = {}
            if extrema:
                for col in type_to_cols["int"]:
                    s = df[col]
                    int_extrema[col] = (s.min(skipna=True), s.max(skipna=True))
                for col in type_to_cols["float"]:
                    s = df[col]
                    float_extrema[col] = (s.min(skipna=True), s.max(skipna=True))
            # Precompute string versions once
            str_cache = {}
            if len(type_to_cols["str"]):
                for col in type_to_cols["str"]:
                    s = df[col].astype("string").fillna("")
                    if not case_sensitive:
                        s = s.str.casefold()
                    str_cache[col] = s
            for dtype_key, sent_list in (sentinels or {}).items():
                cols = type_to_cols.get(dtype_key)
                if cols is None or len(cols) == 0:
                    continue
                for sent in sent_list or []:
                    if dtype_key == "datetime":
                        dt = pd.to_datetime(sent, errors="coerce")
                        if pd.isna(dt):
                            legend[k] = f"{dtype_key}: {sent}"
                            k += 1
                            continue
                        for col in cols:
                            mask = (df[col] == dt).to_numpy()
                            if mask.any():
                                codes_arr[mask, col_pos[col]] = k
                    elif dtype_key == "float":
                        try:
                            f = float(sent)
                        except Exception:
                            legend[k] = f"{dtype_key}: {sent}"
                            k += 1
                            continue
                        for col in cols:
                            if extrema:
                                mn, mx = float_extrema[col]
                                if not (np.isclose(mn, f, atol=float_eps) or np.isclose(mx, f, atol=float_eps)):
                                    continue
                            values = df[col].to_numpy(dtype="float64", na_value=np.nan)
                            mask = np.abs(values - f) < float_eps
                            if mask.any():
                                codes_arr[mask, col_pos[col]] = k
                    elif dtype_key == "int":
                        try:
                            i = int(sent)
                        except Exception:
                            legend[k] = f"{dtype_key}: {sent}"
                            k += 1
                            continue
                        for col in cols:
                            if extrema:
                                mn, mx = int_extrema[col]
                                if not (mn == i or mx == i):
                                    continue
                            mask = (df[col] == i).to_numpy(dtype=bool, na_value=False)
                            if mask.any():
                                codes_arr[mask, col_pos[col]] = k
                    elif dtype_key == "str":
                        target = sent if case_sensitive else str(sent).casefold()
                        for col in cols:
                            mask = str_cache[col].eq(target).to_numpy(dtype=bool, na_value=False)
                            if mask.any():
                                codes_arr[mask, col_pos[col]] = k
                    legend[k] = f"{dtype_key}: {sent}"
                    k += 1
            codes = pd.DataFrame(codes_arr, index=df.index, columns=df.columns)
            return codes, legend


        @this.record_code
        def ResolvePlaceholders(data, sentinels: str = None, float_eps: float = 1e-9, extrema: bool = True, case_sensitive: bool = False, drop_geometry: bool = True): #TODO: get drop_geometry to do something
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
                                this.log.debug(msg = f"Replaced {sum(mask)} {placeholder} {vtype} placeholders in {col}")
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
                        this.log.debug(msg = f"Replaced {sum(mask)} {placeholder} {vtype} placeholders in {col}")
                        df.loc[mask,col] = pd.NA
            return df

        @this.suspendable(calc=True)
        def RawCodes():
            sample = PreparedData()
            codes_df, legend = PlaceholderCodes(
                data=sample,
                sentinels=Sentinels(),
                drop_geometry=True,
                extrema=input.NA_Extrema(),
                case_sensitive=input.NA_CaseSensitive()
            )
            return {
                "codes": codes_df,
                "legend": legend,
            }
      
        
        @this.suspendable(calc=True)
        def CorrectedState():
            sample = PreparedData()
            sentinels = [s.removeprefix("Replace ") for s in input.Replace()]
            fixed = ResolvePlaceholders(
                data=sample,
                sentinels=sentinels,
                extrema=input.NA_Extrema(),
                case_sensitive=input.NA_CaseSensitive(),
            )
            codes_df, legend = PlaceholderCodes(
                data=fixed,
                sentinels=Sentinels(),
                drop_geometry=True,
                extrema=input.NA_Extrema(),
                case_sensitive=input.NA_CaseSensitive(),
            )
            return {
                "fixed": Pxy.from_native(fixed),
                "codes": codes_df,
                "legend": legend,
            }

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
    pxd = Pxy(_df = df, _name = "Test")
    this._imports.set(pxd)
    app = this.application()
elif Module.running_directly(name =__name__):
    this = instance()
    df = pd.read_csv( Card.ROOT / "data" / "Ass2.csv")
    pxd = Pxy(_df = df, _name = "Ass2")
    this._imports.set(pxd)
    this.run()
