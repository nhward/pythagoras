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

import geopandas as gpd
import numpy as np
import pandas as pd  # needed for test / solo modes
import plotly.colors as pc
import plotly.graph_objects as go
import shinywidgets
from card import Card
from cyclic_pandas import is_cyclic
from list_pandas import is_list
from module import Module
from proxy_data import proxy_data as Pxy
from shiny import reactive, render, req, ui
from shinywidgets import render_widget
from text_pandas import is_text

# Converts missing value placeholders to Na/NaN/NaT
# Ideally this follows the correct conversion of strings to their real datatype esp. Datetime

SPECIAL_PLACEHOLDER_COLOURS = {
    0: "#6c757d",  # Missing
    1: "#6fa5f8",  # Not Missing
}


def _placeholder_colour_map(present_codes: list[int]) -> dict[int, str]:
    """Assign semantic base colours and presence-based placeholder colours."""
    colours = {
        code: SPECIAL_PLACEHOLDER_COLOURS[code]
        for code in present_codes
        if code in SPECIAL_PLACEHOLDER_COLOURS
    }
    palette = pc.qualitative.Set3
    placeholder_codes = [code for code in present_codes if code > 1]
    colours.update({
        code: palette[position % len(palette)]
        for position, code in enumerate(placeholder_codes)
    })
    return colours


def _placeholder_kind(series: pd.Series) -> str | None:
    """Return the semantic placeholder-matching family for a Series."""
    dtype = series.dtype
    if getattr(dtype, "name", None) == "geometry":
        return None
    if is_list(dtype):
        return "list"
    if is_cyclic(dtype):
        return "str" if dtype.is_categorical else "float"
    if isinstance(dtype, pd.CategoricalDtype):
        return "str"
    if is_text(dtype):
        return "str"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "datetime"
    if pd.api.types.is_integer_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
        return "int"
    if pd.api.types.is_float_dtype(dtype):
        return "float"
    if pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
        return "str"
    return None


def _columns_by_placeholder_kind(df: pd.DataFrame) -> dict[str, list[str]]:
    columns = {"int": [], "float": [], "str": [], "datetime": [], "list": []}
    for column in df.columns:
        kind = _placeholder_kind(df[column])
        if kind is not None:
            columns[kind].append(column)
    return columns


def _normalise_placeholder_text(value: object, *, case_sensitive: bool) -> str:
    text = str(value)
    return text if case_sensitive else text.casefold()


def _scalar_placeholder_mask(
    series: pd.Series,
    kind: str,
    placeholder: object,
    *,
    float_eps: float,
    extrema: bool,
    case_sensitive: bool,
) -> np.ndarray:
    """Match one placeholder against a scalar-valued Series."""
    if kind == "str":
        values = series.astype("string").fillna("")
        target = _normalise_placeholder_text(
            placeholder,
            case_sensitive=case_sensitive,
        )
        if not case_sensitive:
            values = values.str.casefold()
        return values.eq(target).to_numpy(dtype=bool, na_value=False)
    if kind == "datetime":
        target = pd.to_datetime(placeholder, errors="coerce")
        if pd.isna(target):
            return np.zeros(len(series), dtype=bool)
        return series.eq(target).to_numpy(dtype=bool, na_value=False)
    if kind == "int":
        try:
            target = int(placeholder)
        except (TypeError, ValueError):
            return np.zeros(len(series), dtype=bool)
        if extrema:
            minimum, maximum = series.min(skipna=True), series.max(skipna=True)
            if target != minimum and target != maximum:
                return np.zeros(len(series), dtype=bool)
        return series.eq(target).to_numpy(dtype=bool, na_value=False)
    if kind == "float":
        try:
            target = float(placeholder)
        except (TypeError, ValueError):
            return np.zeros(len(series), dtype=bool)
        values = series.to_numpy(dtype="float64", na_value=np.nan)
        finite = values[np.isfinite(values)]
        if extrema and finite.size:  # noqa: SIM102
            if not (
                np.isclose(finite.min(), target, atol=float_eps)
                or np.isclose(finite.max(), target, atol=float_eps)
            ):
                return np.zeros(len(series), dtype=bool)
        return np.isfinite(values) & (np.abs(values - target) < float_eps)
    return np.zeros(len(series), dtype=bool)


def _apply_list_placeholder_codes(
    df: pd.DataFrame,
    columns: list[str],
    lookup: dict[str, int],
    codes_arr: np.ndarray,
    col_pos: dict[str, int],
    *,
    case_sensitive: bool,
) -> None:
    """Scan every list element once and assign the first matching code per cell."""
    if not lookup:
        return
    for column in columns:
        output = codes_arr[:, col_pos[column]]
        for row, value in enumerate(df[column].array):
            if value is pd.NA:
                continue
            for item in value:
                if not isinstance(item, str):
                    continue
                key = item if case_sensitive else item.casefold()
                code = lookup.get(key)
                if code is not None:
                    output[row] = code
                    break


def _rebuild_custom_series(series: pd.Series, values: list[object]) -> pd.Series:
    array = type(series.array)._from_sequence(values, dtype=series.dtype)
    return pd.Series(array, index=series.index, name=series.name)


def _replace_scalar_matches(series: pd.Series, mask: np.ndarray) -> pd.Series:
    if not mask.any():
        return series
    if is_cyclic(series.dtype) or is_text(series.dtype):
        values = list(series.array)
        for position in np.flatnonzero(mask):
            values[int(position)] = pd.NA
        return _rebuild_custom_series(series, values)
    result = series.copy()
    result.loc[mask] = pd.NaT if pd.api.types.is_datetime64_any_dtype(series.dtype) else pd.NA
    return result


def _remove_list_placeholders(
    series: pd.Series,
    placeholders: set[str],
    *,
    case_sensitive: bool,
) -> pd.Series:
    """Remove matching string elements in one pass, preserving ListDtype."""
    if not placeholders:
        return series
    changed = False
    values: list[object] = []
    for value in series.array:
        if value is pd.NA:
            values.append(pd.NA)
            continue
        retained = [
            item
            for item in value
            if not (
                isinstance(item, str)
                and (item if case_sensitive else item.casefold()) in placeholders
            )
        ]
        matched = len(retained) != len(value)
        if matched:
            changed = True
            values.append(retained if retained else pd.NA)
        else:
            values.append(value)
    if not changed:
        return series
    # TODO: Implement __setitem__/_putmask in the custom extension arrays so pandas-native masked assignment can replace this reconstruction step.
    return _rebuild_custom_series(series, values)

def instance():
    """
    Creates an instance of Card configured as MissPlaceholders.
    """
    this = Card(file=__file__, mutable=True) # "mutable" means it can change the pxd - probably with a commit button
    this.long_name = "Missing value placeholders"
    this.description = "This card detects any potential missing-value placeholders and allows their replacement with NA."

    #############################
    # Define the user-interface #
    #############################

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

    def footer():
        """
        These ui elements appear in the footer of the card (but only on the front). 
        The optional 'guide', 'text', 'position' and 'priority' parameters of the ui elements allows for the Guide.
        """
        return ui.div(
            ui.output_ui(id="Message"),
            ui.input_checkbox_group(
                id = "Replace",
                label = None,
                choices=[],
                inline=True,
                guide = this, title = "Replace buttons", position = "top",
                text = "Buttons for replacing placeholders by converting them to missing values. There is a button for each type of placeholder found. The changes can be reversed."
            ),
            class_ = "vertically-scrollable-footer")

    this.footer = footer  ## The above "footer" function must be assigned to the instance

    def settings() -> ui.TagList:
        """
        These settings related ui elements appear in the sidebar of the card. 
        The optional 'guide', 'text', 'position' and 'priority' parameters of the ui elements allows for the Guide.
        """
        return ui.TagList(
            ui.input_selectize(
                id="NA_Strings", label = "Missing string-value placeholders", 
                choices=["NA","-","--", "N/A", "Missing", "Not Applicable", "Not Available"],
                selected=["NA","-","--", "N/A", "Missing", "Not Applicable", "Not Available"],
                multiple=True, options=({"placeholder": "Enter string values", "create": True}),
                guide=this, position="left",
                text = 'This comma-delimited list supplies placeholders for missing string-values. Entire string-values that match any of these will be replaced with NA. The search is case insensitive.',
            ),
            ui.input_checkbox(
                id = "NA_CaseSensitive", label = "Use a case-sensitive search", value = False,
                guide = this, text = 'Whether "N/A" is different to "N/a", "n/a", "n/A".', position = "left"
            ),
            ui.input_selectize(
                id = "NA_Integers", label = "Missing integer-value placeholders", choices =  [-9999,-999,-99, -1], selected = [-9999,-999,-99, -1],
                multiple = True, options=({"placeholder": "Enter integer values", "create": True}),
                guide = this, position = "left",
                text = 'This comma-delimited list supplies placeholders for missing numeric-values. Entire values that match any of these will be replaced with NA <em>provided</em> they correspond with the lowest recorded values.',
            ),
            ui.input_selectize(
                id = "NA_Floats", label = "Missing decimal-value placeholders",  choices =  [-9999.99,-999.99,-99.99, -99.00, -1.00],
                selected = [-9999.99,-999.99,-99.99, -99.00, -1.00], multiple = True, options=({"placeholder": "Enter decimal values", "create": True}),
                guide = this, position = "left",
                text = 'This comma-delimited list supplies placeholders for missing decimal-values. Entire values that match any of these will be replaced with NaN <em>provided</em> they correspond with the lowest recorded values.',
            ),
            ui.input_checkbox(
                id = "NA_Extrema", label = "Only replace extreme numeric values (at minimum or maximum)",  value = True,
                guide = this, position = "left", text = """
                Only replace numbers <em>provided</em> they correspond with the lowest or highest recorded values. When set ON, a numeric placeholder (e.g. -99) 
                is only flagged if it sits at the edge of the observed distribution for that variable (i.e., equals the current minimum or maximum).
                When OFF, any occurrence of the placeholder is matched regardless of position.
                <br>Example: If a column’s observed range is −1000 … 1200<br>
                -99 is inside the range → not flagged when `Replace extrema only` is ON.
                <br>Example: If the range is −99 … 1200:<br>
                -99 equals the minimum → is flagged when `Replace extrema only` is ON.  
                """,
            ),
            ui.input_selectize(
                id = "NA_DateTime", label = "Missing date/time-value placeholders", choices = ["0000-00-00", "0001-01-01", "1900-01-01", "0"],
                selected = ["0000-00-00", "0001-01-01", "1900-01-01", "0"], multiple = True, options=({"placeholder": "Enter date-literal values", "create": True}),
                guide = this, position = "left",
                text = 'This comma-delimited list supplies placeholders for missing date/time-values. Date values that match any of these will be replaced with NaT.',
            ),
            ui.input_slider(
                id = "MaxObs", label = "Maximum observations to analyse", min = 3, max = 7, value = 3, ticks = True, pre = "10^",
                guide = this, text = 'Limit to number of observations to analyse to ensure responsiveness (logarithmic scale).', position = "left"
            ),
        )

    this.settings = settings ## The above "setting" function must be assigned to the instance 

    ########################
    # Define the behaviour #
    ########################

    def server(input, output, session):

        @this.suspendable(calc = True)
        def incomingproxy_data():
            this._imports.get()
            req(this._imports.is_set())
            return this._imports.get()

        @this.settle(seconds=2)
        @this.suspendable(calc = True)
        def Replace():
            return input.Replace()

        @this.settle(seconds=2)
        @this.suspendable(calc = True)
        def MaxObs():
            return 10**input.MaxObs()

        @this.suspendable(calc = True)
        @this.record_code
        def PreparedData():
            sample = incomingproxy_data().sample(n = MaxObs(), mode = "random", keep_geometry = False)
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


        @this.suspendable(calc=True)
        def Choices():
            rawstate = RawCodes()
            flat = pd.Series(rawstate["codes"].to_numpy().ravel())
            used_codes = pd.to_numeric(flat, errors="coerce").dropna().astype(int).unique().tolist()
            used_codes = sorted(k for k in used_codes if k not in (0, 1))
            reduced_labels = [str(rawstate["legend"].get(k, f"Code {k}")) for k in used_codes]
            return [f"Replace {lab}" for lab in reduced_labels]

        @output
        @render.ui
        def Message():
            if len(Choices())==0:
                return ui.span("Placeholders not detected", class_="text-success")

        @this.suspendable()
        def UpdateButtons():
            choices = Choices()
            with reactive.isolate():
                previous = Replace()
            selected = [c for c in previous if c in choices]
            ui.update_checkbox_group(id="Replace", choices=choices, selected=selected, )



        def empty_plotly(message="No data to display", subtext=None):
            this.log.debug("Empty chart drawn")
            txt = f"<b>{message}</b>" + (f"<br><span style='font-size:0.9em;color:#6c757d'>{subtext}</span>" if subtext else "")
            fig = go.Figure()
            fig.add_annotation(
                text=txt, x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, align="center",
                font={"size": 18, "color": "#6c757d"}
            )
            fig.update_layout(
                xaxis={"visible": False}, 
                yaxis={"visible": False},
                plot_bgcolor="#f8f9fa", # make same as card
                paper_bgcolor="rgba(0,0,0,0)",
                margin={"l": 0, "r": 0, "t": 0, "b": 0},
                hovermode=False, 
                showlegend=False
            )
            fw = go.FigureWidget(fig)
            fw._config = (getattr(fw, "_config", {}) | {"displayModeBar": False, "displaylogo": False, "responsive": True})
            return fw

        @this.record_code
        def _select_cols(df: pd.DataFrame | Pxy, bucket: str) -> list[str]:
            native = df.to_native() if isinstance(df, Pxy) else df
            columns = _columns_by_placeholder_kind(native)
            if bucket in columns:
                return columns[bucket]
            # TODO: Add a dedicated basket/list chart tab if collection-valued variables become common enough to justify another front panel.
            return native.columns.tolist()


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
            code_colours = _placeholder_colour_map(present_codes)
            zmin = min(present_codes)
            zmax = max(present_codes)
            if zmin == zmax:
                color = code_colours[zmin]
                colorscale = [[0, color], [1, color]]
            else:
                colorscale = []
                for code in present_codes:
                    pos = (code - zmin) / (zmax - zmin)
                    color = code_colours[code]
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

                    hover_colors = [code_colours[c] for c in codes]

                    fig.add_trace(go.Scatter(
                        x=xx,
                        y=[y[i] for i in yy],
                        mode="markers",
                        marker={
                            "size": 10,
                            "opacity": 0,
                            "color": hover_colors,
                        },
                        text=hover_text,
                        hovertemplate="%{text}<extra></extra>",
                        hoverlabel={
                            "bgcolor": hover_colors,
                            "bordercolor": hover_colors,
                            "font": {"color": "black"},
                        },
                        showlegend=False,
                        hoverinfo="text",
                    ))

                # Legend-only traces
                for code in present_codes:
                    fig.add_trace(go.Scatter(
                        x=[None],
                        y=[None],
                        mode="markers",
                        marker={
                            "size": 10,
                            "color": code_colours[code],
                            "symbol": "square",
                        },
                        name=legend.get(code, f"Code {code}"),
                        showlegend=True,
                        hoverinfo="skip",
                    ))
            fig.update_layout(
                xaxis={"title": "Observation"},
                yaxis={"title": "Variables", "type": "category", "autorange": "reversed"},
                plot_bgcolor="#e5ecf6",
                margin={"l": 2, "r": 2, "t": 2, "b": 2},
                showlegend=fs,
                legend={
                    "x": 1.0,
                    "xanchor": "left",
                    "y": 0.0,
                    "yanchor": "bottom",
                    "itemsizing": "constant",
                    "font": {"size": 16}
                },
                modebar={"orientation": "v"}
            )
            fw = go.FigureWidget(fig)
            fw._config = (getattr(fw, "_config", {}) | {
                "displayModeBar": bool(fs),
                "displaylogo": False,
                "responsive": True
            })
            return fw

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
            cols = _select_cols(fixed.to_native(), "float")
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
            full  = incomingproxy_data()
            sentinels = [s.removeprefix("Replace ") for s in Replace()]
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

        
        def PlaceholderCodes(data, sentinels: dict[str, list[any]], float_eps: float = 1e-9, drop_geometry: bool = True, 
        extrema: bool = True, case_sensitive: bool = False) -> tuple[pd.DataFrame, dict[int, str]]:
            req(data is not None)
            if isinstance(data, pd.DataFrame):
                df = data
            elif isinstance(data, Pxy):
                df = data.to_native()
            else:
                raise TypeError(f"Unknown dataset type supplied: {type(data)}")
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
            type_to_cols = _columns_by_placeholder_kind(df)
            list_string_codes: dict[str, int] = {}
            for dtype_key, sent_list in (sentinels or {}).items():
                cols = type_to_cols.get(dtype_key)
                if cols is None or len(cols) == 0:
                    cols = []
                for sent in sent_list or []:
                    for col in cols:
                        mask = _scalar_placeholder_mask(
                            df[col],
                            dtype_key,
                            sent,
                            float_eps=float_eps,
                            extrema=extrema,
                            case_sensitive=case_sensitive,
                        )
                        if mask.any():
                            codes_arr[mask, col_pos[col]] = k
                    if dtype_key == "str":
                        target = _normalise_placeholder_text(
                            sent,
                            case_sensitive=case_sensitive,
                        )
                        list_string_codes[target] = k
                    legend[k] = f"{dtype_key}: {sent}"
                    k += 1
            _apply_list_placeholder_codes(
                df,
                type_to_cols["list"],
                list_string_codes,
                codes_arr,
                col_pos,
                case_sensitive=case_sensitive,
            )
            codes = pd.DataFrame(codes_arr, index=df.index, columns=df.columns)
            return codes, legend


        @this.record_code
        def ResolvePlaceholders(data, sentinels: str | None, float_eps: float = 1e-9, extrema: bool = True, case_sensitive: bool = False, drop_geometry: bool = True): #TODO: get drop_geometry to do something
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
            type_to_cols = _columns_by_placeholder_kind(df)
            parsed: dict[str, list[str]] = {
                "int": [], "float": [], "str": [], "datetime": []
            }
            for sent in sentinels:
                vtype, placeholder = sent.split(sep=": ", maxsplit=1)
                if vtype in parsed:
                    parsed[vtype].append(placeholder)
            for vtype, placeholders in parsed.items():
                for placeholder in placeholders:
                    for col in type_to_cols[vtype]:
                        mask = _scalar_placeholder_mask(
                            df[col],
                            vtype,
                            placeholder,
                            float_eps=float_eps,
                            extrema=extrema,
                            case_sensitive=case_sensitive,
                        )
                        if mask.any():
                            this.log.debug(
                                f"Replaced {int(mask.sum())} {placeholder} "
                                f"{vtype} placeholders in {col}"
                            )
                            df[col] = _replace_scalar_matches(df[col], mask)

            list_placeholders = {
                _normalise_placeholder_text(
                    placeholder,
                    case_sensitive=case_sensitive,
                )
                for placeholder in parsed["str"]
            }
            for col in type_to_cols["list"]:
                df[col] = _remove_list_placeholders(
                    df[col],
                    list_placeholders,
                    case_sensitive=case_sensitive,
                )
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
            sentinels = [s.removeprefix("Replace ") for s in Replace()]
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


if Module.running_directly(name =__name__):
    this = instance()
    df = pd.read_csv( Card.ROOT / "data" / "Ass2.csv")
    pxd = Pxy(_df = df, _name = "Ass2")
    this._imports.set(pxd)
    this.run()
