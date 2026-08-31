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

import difflib
from collections.abc import Hashable, Sequence

import numpy as np
import pandas as pd
from card import Card
from cyclic_pandas import as_cyclic, is_cyclic_like
from faicons import icon_svg as icon
from geometry_pandas import as_geometry, is_geometry_like
from list_pandas import as_list, is_list, is_list_like
from module import Module
from proxy_data import proxy_data as pxd
from shiny import reactive, render, req, ui
from text_pandas import as_text, is_text_like

#TODO: Improve Script output
#TODO: Expose more setting used in _Like functions

def instance():
    """
    Creates an instance of Card configured as "varModify".
    Functions:
    - rename a variable,
    - convert a variable data-type to another viable data-type
    - reorder levels of an ordered categorical variable,
    - expand delimiter-separated basket-like data into lists.
    """

    this = Card(file=__file__, mutable=True)
    this.long_name = "Modification"
    this.description = "This card allows the basic modification of variables such as name, data-type, and cyclic-order ."

    TYPE_CHOICES = {
        "float64": "decimal",
        "int64": "integer",
        "float32": "decimal",
        "int32": "integer",
        "integer": "integer",
        "boolean": "boolean",
        "date": "date",
        "datetime": "datetime",
        "time": "time",
        "text": "text",
        "category": "nominal",
        "ordered": "ordered",
        "cyclic": "cyclic",
        "basket": "basket",
        "str": "code",
        "object": "object"
    }

    DATE_FORMATS = [
        "%Y-%m-%d",  # 2026-05-16
        "%d/%m/%Y",  # 16/05/2026
        "%d-%m-%Y",  # 16-05-2026
        "%Y/%m/%d",  # 2026/05/16
        "%d %b %Y",  # 16 May 2026
        "%d %B %Y",  # 16 May 2026
        "%Y-%m-%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
    ]

    def _dtype_choice(dtype) -> str:
        """Return the card's friendly type name for a pandas dtype."""
        name = getattr(dtype, "name", str(dtype))
        if name == "cyclic":
            return "cyclic"
        if name == "text":
            return "text"
        if name == "basket":
            return "basket"
        if name == "geometry":
            return "geometry"
        if isinstance(dtype, pd.CategoricalDtype):
            return "ordered" if dtype.ordered else "nominal"
        if pd.api.types.is_bool_dtype(dtype):
            return "boolean"
        if pd.api.types.is_integer_dtype(dtype):
            return "integer"
        if pd.api.types.is_float_dtype(dtype):
            return "decimal"
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return "datetime"
        if isinstance(dtype, pd.StringDtype):
            return "code"
        if pd.api.types.is_object_dtype(dtype):
            return "code"
        return TYPE_CHOICES.get(str(dtype), str(dtype))

    def front():
        return ui.div(
            ui.output_data_frame(
                id="Table",
                title="Table of variables",
                text="Choose a variable from this table and alter the name, data type, or cyclic order.",
                guide=this,
                position="bottom",
            ),
            id = "TableDiv"
        )

    this.front = front

    def back() -> ui.TagList:
        return ui.TagList(
            ui.card_header("Committed Differences", class_="text-primary text-center"),
            this.guidedDiv(
                ui.output_ui(id="DFDiff"),
                id="X-DFDiff",
                class_="html-fill-container html-fill-item",
                guide=this,
                title="Change report",
                text="This report lists structural changes to the dataset by this card.",
                position="top",
            ),
        )

    this.back = back

    def footer() -> ui.TagList:
        return ui.TagList(
            this.guidedDiv(
                ui.input_text(id="NewName", label="New Name", value=None, 
                guide = this, title = "Change the current variable's name by using this field", position="top"),
                ui.input_selectize(id="NewDataType", label="New Data Type", choices=list(set(TYPE_CHOICES)), selected=False, options={"dropdownParent": "body"}, 
                guide = this, title = "Change the current variable's data-type by using this field", position="top"
                ),
                ui.input_selectize(id="NewOrder", label="New order", choices=[], selected=None, multiple=True, remove_button=False, width = "100%",
                options={
                    "dropdownParent": "body",
                    "plugins": ["drag_drop"],
                    "onDelete": ui.js_eval("function(values) { return false; }"),
                },
                guide = this, title = "Change the current variable's order by using this field - provided the data-type is 'ordered' or 'cyclic'", position="top"),
                id="Row",
                title="New attributes for the choosen variable.",
                text="The fields shown here can be modified. The available choices are controlled by the 'Alternatives' settings.",
                guide=this,
                position="top",
                class_="across-row",
            ),
            ui.div(
                ui.input_action_button(
                    id="Commit",
                    label="Commit modification",
                    icon=icon("gavel", title="Commit modification", a11y="sem"),
                    # disabled=True,
                    width="220px",
                    class_="btn rounded-pill btn-sm btn-primary",
                    style="border: 0px; box-shadow: none;",
                    guide=this,
                    title="Commit button",
                    text="This button modifies the data and feeds the next card with these changes.",
                    position="top",
                ),
                ui.input_action_button(
                    id="Reset",
                    label=None,
                    icon=icon(
                        "arrow-rotate-left", title="Reset modifications", a11y="sem"
                    ),
                    class_="btn rounded-pill btn-sm btn-primary",
                    style="border: 0px; box-shadow: none;",
                    guide=this,
                    title="Reset button",
                    text="This button reverts all modifications and lets you start again.",
                    position="top",
                ),
                id="Reset-Commit",
                class_="btn-group mx-auto text-center gap-2",
            ),
        )

    this.footer = footer

    def settings() -> ui.TagList:
        return ui.TagList(
            ui.input_text(
                id="Formats", label="Possible date formats", width="100%", value=", ".join(map(str, DATE_FORMATS)),
                guide=this, text="Comma-delimited date formats to try - in this order. %Y means year; %m means month; %d means day.", position="left",
            ),
            ui.input_radio_buttons(
                id="Alternatives", label="Alternative types", choices=["Sensible", "Related", "All"], selected="Sensible",
                guide=this, position="left", text="""How the data-type choices are offered based on the nature of the variable; <br>
                <b>All:</b> all possible choices are offered, <br><b>Related:</b> the data-type related choices are offered, <br>
                <b>Sensible:</b> the choices based on data-type, values and cardinality""",
            ),
            ui.input_slider(
                id = "MaxObs", label = "Maximum observations to analyse", min = 3, max = 7, value = 4, ticks = True, pre = "10^",
                guide = this, text = 'Limit to number of observations to analyse to ensure responsiveness (logarithmic scale).', position = "left")
        )

    this.settings = settings

    def server(input, output, session):

        @this.suspendable(calc=True)
        def incomingproxy_data():
            req(this._imports.is_set())
            return this._imports.get()

        @this.settle(seconds=2)
        @this.suspendable(calc = True)
        def MaxObs():
            return 10**input.MaxObs()

        @this.suspendable(calc=True)
        def PreparedData() -> pxd:
            samp = incomingproxy_data().sample(n=MaxObs(), mode="random", keep_geometry=True)
            return samp

        @this.suspendable()
        def PxdChange():
            this._exports.set(incomingproxy_data())

        @this.suspendable(calc = True)
        def Schema():
            def first_role(column: str) -> str:
                roles = px.role_map.get_roles(column)
                return next(iter(roles)).value if roles else ""

            px = PreparedData()
            req(px is not None)

            def levels(series):
                if _dtype_choice(series.dtype) == "ordered":
                    if not isinstance(series.dtype, pd.CategoricalDtype):
                        series = series.astype("category")
                    order = series.cat.categories.tolist()
                else:
                    order = []
                return ",".join(order)
            
            df = pd.DataFrame(
                {
                    "Orig\nname": px.columns,
                    "New\nname": px.columns,
                    "Orig\nd-type": [_dtype_choice(px.data[c].dtype) for c in px.columns],
                    "New\nd-type": [_dtype_choice(px.data[c].dtype) for c in px.columns],
                    "Orig\norder": [levels(px.data[c]) for c in px.columns],
                    "New\norder": [levels(px.data[c]) for c in px.columns],
                    "Role": [first_role(c) for c in px.columns],
                }
            )
            df.set_index("Orig\nname")
            return df                


        @output
        @render.data_frame
        def Table():
            schema = Schema()
            req(schema is not None)
            return render.DataGrid(
                schema,
                selection_mode="row",
            )

        selection_scheduled = False

        @this.suspendable()
        def schedule_initial_selection():
            nonlocal selection_scheduled
            if selection_scheduled:
                return
            data = Table.data()  # creates dependence on Table
            # Suspends until the browser-side grid has initialized.
            Table.cell_selection()
            if len(data) == 0:
                return
            selection_scheduled = True

            async def apply_initial_selection():
                await Table.update_cell_selection(
                    {
                        "type": "row",
                        "rows": [0],
                    }
                )

            # Runs after the flush in which client readiness was observed.
            session.on_flushed(apply_initial_selection, once=True)

        @this.suspendable(calc=True)
        def selected_row():
            return Table.data_view(selected=True)

        previous_row = reactive.value(None)

        @this.suspendable(calc=True)
        def allowed_d_types():
            row = selected_row()
            req(row is not None, not row.empty)
            origType = row["Orig\nd-type"].iloc[0]
            if input.Alternatives() == "All":
                return list(set(TYPE_CHOICES.values()))
            elif input.Alternatives() == "Related":
                if origType == "cyclic":
                    possible = ["text", "nominal", "ordered", "cyclic"]
                elif origType == "list":
                    possible = ["nominal", "text"]
                elif origType == "geometry":
                    possible = ["geometry", "text", "decimal"]
                if origType == "ordered":
                    possible = ["text", "nominal", "ordered", "cyclic", "integer"]
                elif origType in ["date", "datetime", "time"]:
                    possible = ["date", "datetime", "time"]
                elif origType == "text":  # noqa: SIM114
                    possible = ["text", "nominal", "ordered", "cyclic", "code"]
                elif origType == "nominal":
                    possible = ["text", "nominal", "ordered", "cyclic", "code"]
                elif origType == "decimal":  # noqa: SIM114
                    possible = ["decimal", "integer", "nominal", "ordered"]
                elif origType == "integer":
                    possible = ["decimal", "integer", "nominal", "ordered"]
                elif origType in["bool"]:
                    possible = ["integer", "nominal"]
                elif origType in["code"]:
                    possible = ["text", "nominal", "ordered", "cyclic", "code"]
                return possible
            else:
                sensible = []
                px = PreparedData()
                req(px)
                origName = row["Orig\nname"].iloc[0]
                df = px.data
                req(not df.empty)
                series = df[origName]
                if is_cyclic_like(series):
                    sensible.append("cyclic")
                if is_list_like(series):
                    sensible.append("list")
                if is_geometry_like(series):
                    sensible.append("geometry")
                if is_date_like(series, formats = DATE_FORMATS):
                    sensible.append("date")
                if is_text_like(series, uniqueness_threshold = 0.90):
                    sensible.append("text")
                if is_nominal_like(series, high_cardinality = 20):
                    sensible.append("nominal")
                if is_ordered_like(series, high_cardinality = 15):
                    sensible.append("ordered")
                if is_integer_like(series):
                    sensible.append("integer")
                if is_numeric_like(series):
                    sensible.append("decimal")
                if len(sensible) == 0 and pd.api.types.is_string_dtype(series):
                    sensible.append("code")
                return sensible

        @this.suspendable()
        async def RowChange():
            row = selected_row()
            req(row is not None, not row.empty)
            req(row["Orig\nname"].iloc[0] != previous_row.get())
            previous_row.set(row["Orig\nname"].iloc[0])
            ui.update_text(id="NewName", value=row["New\nname"].iloc[0])
            at = allowed_d_types()
            if len(at) == 1:
                selected = at[0]
            else:
                selected = row["New\nd-type"].iloc[0]
            ui.update_selectize(id="NewDataType", choices=at, selected=selected)
            if row["New\nd-type"].iloc[0] not in at:
                await session.send_custom_message(
                    "animate",
                    {"id": session.ns("NewDataType"), "animation": "shakeX", "delay": 0, "duration": 500, "lock": "TableDiv"},
                )
            order = row["New\norder"].iloc[0].split(",")
            ui.update_selectize(id="NewOrder", choices=order, selected=order)
            
        @this.suspendable(triggers=[input.NewName])
        async def validate_new_name():
            row = selected_row()
            req(row is not None, not row.empty)
            origName = row["Orig\nname"].iloc[0]
            df = Table.data().copy()
            if input.NewName() != origName:
                if input.NewName() in df["Orig\nname"].values:
                    await session.send_custom_message(
                        "animate",
                        {"id": session.ns("NewName"), "animation": "shakeX", "delay": 500},
                    )
                else:
                    df.loc[df["Orig\nname"] == origName, "New\nname"] = input.NewName()
                    await Table.update_data(df)

        @this.suspendable(triggers = [input.NewDataType])
        async def TypeChange():
            row = selected_row()
            req(row is not None, not row.empty)
            origName = row["Orig\nname"].iloc[0]
            if len(input.NewDataType()) > 0:
                df = Table.data().copy()
                df.loc[df["Orig\nname"]==origName, "New\nd-type"] = input.NewDataType()
                await Table.update_data(df)
            if input.NewDataType() in ["ordered", "cyclic"]:
                px = PreparedData()
                req(px)
                var = px.data[row["Orig\nname"].iloc[0]]
                if not isinstance(var, pd.CategoricalDtype):
                    var = var.astype("category")
                order = var.cat.categories.tolist()
                ui.update_selectize(id = "NewOrder", choices = order, selected = order)
            else:
                ui.update_selectize(id = "NewOrder", choices = None, selected = None)

        @this.suspendable()
        def AltChange():
            input.Alternatives() # create dependency that is not rejected for being the same row
            row = selected_row()
            req(row is not None, not row.empty)
            value = row["New\nd-type"].iloc[0]
            if value in allowed_d_types():
                ui.update_selectize(id="NewDataType", choices=allowed_d_types(), selected=value)
            else:
                ui.update_selectize(id="NewDataType", choices=allowed_d_types())
                
        @this.suspendable(triggers=[input.NewOrder])
        async def validate_new_order():
            row = selected_row()
            req(row is not None, not row.empty)
            origName = row["Orig\nname"].iloc[0]
            if len(input.NewOrder()) > 0:
                df = Table.data().copy()
                df.loc[df["Orig\nname"]==origName, "New\norder"] = ",".join(input.NewOrder())
                await Table.update_data(df)


        def _dataframe_structure_text(df: pd.DataFrame) -> list[str]:
            """
            Small, stable text representation for a structural diff.
            """
            req(not this.isFront())
            lines = [f"rows: {len(df)}", f"columns: {len(df.columns)}"]
            for col in df.columns:
                s = df[col]
                if is_list(s):
                    lines.append(
                        f"{col}: dtype={s.dtype}, missing={int(s.isna().sum())}, unique=NA"
                    )
                else:
                    lines.append(
                        f"{col}: dtype={s.dtype}, missing={int(s.isna().sum())}, unique={s.nunique(dropna=True)}"
                    )
            return lines

        @output

        @render.ui
        def DFDiff():
            old_lines = _dataframe_structure_text(incomingproxy_data().to_native())
            new_lines = _dataframe_structure_text(this._exports.get().to_native())
            diff = "\n".join(
                difflib.unified_diff(
                    old_lines,
                    new_lines,
                    fromfile="incoming",
                    tofile="modified",
                    lineterm="",
                )
            )
            if not diff or len(diff) == 0:
                diff = "No committed modifications."
            return ui.tags.pre(diff, style="white-space: pre-wrap; font-size: 0.85rem;")

        @this.suspendable(triggers=[input.Commit])
        @this.record_code
        def CommitEvent():
            # ui.update_action_button(id="Commit", disabled=True)
            data = incomingproxy_data()
            rm = data._copy_roles()
            table_data = Table.data()
            #Transform the data as per Table control
            d = data.to_native().copy()
            for name, newName, origType, newType, origOrder, newOrder in table_data[["Orig\nname","New\nname","Orig\nd-type","New\nd-type","Orig\norder","New\norder"]].itertuples(index=False, name=None):
                if origType != newType or origOrder != newOrder:
                    d[name] = convert_series(series = d[name], new_type = newType, order=newOrder, formats = input.Formats)
                if name != newName:
                    d.rename(columns={name: newName}, inplace=True)
                    rm.rename_column(from_=name, to_=newName)
            data2 = pxd(_df=d, _roles=rm, _name=data.name)
            #set the output data
            this._exports.set(data2)

        @this.suspendable(triggers=[input.Reset])
        async def Reset():
            #reset the Table's data
            df = Table.data().copy()
            df["New\nname"] = df["Orig\nname"]
            df["New\nd-type"] = df["Orig\nd-type"]
            df["New\norder"] = df["Orig\norder"]
            await Table.update_data(df)
            #reinstate the pass-through of the actual data
            this._exports.set(incomingproxy_data())
            row = selected_row()
            req(row is not None, not row.empty)
            ui.update_text(id = "NewName", value = row["Orig\nname"].iloc[0])
            ui.update_selectize(id = "NewDataType", selected = row["Orig\nd-type"].iloc[0])
            await Table.update_data(df)


        def _unique_non_na(series: pd.Series) -> pd.Series:
            """
            Return unique non-missing values as a Series.
            """
            return pd.Series(series.dropna().unique())


        def is_numeric_like(
            series: pd.Series,
            *,
            threshold: float = 0.95,
            limit: int | None = None,
            allow_bool: bool = True,
            allow_infinite: bool = True,
            blank_as_missing: bool = True,
        ) -> bool:
            """Return whether predominantly all inspected values parse as numeric."""
            if not isinstance(series, pd.Series):
                raise TypeError("series must be a pandas Series")
            if not 0 <= threshold <= 1:
                raise ValueError("threshold must be between 0 and 1")
            if limit is not None:
                if isinstance(limit, bool) or not isinstance(limit, int):
                    raise TypeError("limit must be an integer or None")
                if limit <= 0:
                    raise ValueError("limit must be greater than zero")
            if pd.api.types.is_bool_dtype(series.dtype):
                return allow_bool
            values = series
            if blank_as_missing and (
                pd.api.types.is_object_dtype(values.dtype)
                or pd.api.types.is_string_dtype(values.dtype)
            ):
                values = values.replace(r"^\s*$", pd.NA, regex=True)
            values = values.dropna()
            if values.empty:
                return False
            if limit is not None and len(values) > limit:
                positions = np.linspace(
                    0,
                    len(values) - 1,
                    num=limit,
                    dtype=int,
                )
                values = values.iloc[positions]
            if pd.api.types.is_numeric_dtype(series.dtype):
                if allow_infinite:
                    return True
                return bool(np.isfinite(values.to_numpy()).all())
            try:
                parsed = pd.to_numeric(values, errors="coerce")
            except (TypeError, ValueError):
                return False
            valid = parsed.notna()
            if not allow_infinite:
                valid &= np.isfinite(parsed)
            return bool(valid.mean() >= threshold)


        def is_integer_like(
            series: pd.Series,
            *,
            threshold: float = 0.95,
            limit: int | None = None,
            allow_bool: bool = False,
            allow_infinite: bool = False,
            blank_as_missing: bool = True,
            atol: float = 0.0,
        ) -> bool:
            """Return whether predominantly all inspected values represent integers.

            Parameters
            ----------
            series:
                Series to inspect.
            threshold:
                Required proportion of inspected, non-missing values that must be
                safely interpretable as integers. Must lie in ``[0, 1]``.
            limit:
                Maximum number of non-missing values to inspect. ``None`` examines
                every non-missing value.
            allow_bool:
                Whether Boolean values should count as integers.
            allow_infinite:
                Normally ``False``, because infinity cannot be represented as an
                integer. Included for API symmetry, although enabling it usually
                makes little semantic sense.
            blank_as_missing:
                Whether blank and whitespace-only strings should be excluded.
            atol:
                Absolute tolerance permitted when comparing numeric values with the
                nearest integer. The default of zero requires exact integrality.

            Returns
            -------
            bool
                ``True`` when the proportion of integer-like values meets
                ``threshold``.
            """
            if not isinstance(series, pd.Series):
                raise TypeError("series must be a pandas Series")
            if not 0 <= threshold <= 1:
                raise ValueError("threshold must be between 0 and 1")
            if limit is not None:
                if isinstance(limit, bool) or not isinstance(limit, int):
                    raise TypeError("limit must be an integer or None")
                if limit <= 0:
                    raise ValueError("limit must be greater than zero")
            if atol < 0:
                raise ValueError("atol must be non-negative")
            if pd.api.types.is_bool_dtype(series.dtype):
                return allow_bool
            if pd.api.types.is_integer_dtype(series.dtype):
                return True
            values = series
            if blank_as_missing:
                values = values.map(
                    lambda value: (
                        pd.NA
                        if isinstance(value, str) and not value.strip()
                        else value
                    )
                )
            values = values.dropna()
            if values.empty:
                return False
            if limit is not None and len(values) > limit:
                positions = np.linspace(
                    0,
                    len(values) - 1,
                    num=limit,
                    dtype=int,
                )
                values = values.iloc[positions]
            try:
                parsed = pd.to_numeric(values, errors="coerce")
            except (TypeError, ValueError):
                return False
            numeric = parsed.notna()
            if not allow_infinite:
                numeric &= np.isfinite(parsed)
            nearest = parsed.round()
            if atol == 0:
                integral = parsed.eq(nearest)
            else:
                integral = pd.Series(
                    np.isclose(
                        parsed,
                        nearest,
                        rtol=0,
                        atol=atol,
                        equal_nan=False,
                    ),
                    index=parsed.index,
                )
            valid = numeric & integral
            return bool(valid.mean() >= threshold)


        def is_date_like(
            series: pd.Series,
            *,
            formats: Sequence[str] = (),
            threshold: float = 0.95,
            limit: int | None = None,
            format_policy: str = "single",
            allow_mixed: bool = False,
            dayfirst: bool = False,
            yearfirst: bool = False,
            blank_as_missing: bool = True,
            accept_datetime_dtype: bool = True,
        ) -> bool:
            """Return whether predominantly all inspected values represent dates.

            Parameters
            ----------
            series:
                Series to inspect.
            formats:
                Explicit ``strftime`` formats to try, such as ``"%Y-%m-%d"``.
            threshold:
                Required proportion of inspected non-missing values that must parse
                successfully. Must lie in ``[0, 1]``.
            limit:
                Maximum number of non-missing values to inspect. ``None`` examines
                all values. Limited samples are distributed across the column.
            format_policy:
                ``"single"`` requires one format to meet ``threshold``.
                ``"combined"`` permits different values to match different formats.
            allow_mixed:
                If no configured-format test succeeds, try pandas' mixed-format
                parser.
            dayfirst:
                Passed to mixed-format parsing.
            yearfirst:
                Passed to mixed-format parsing.
            blank_as_missing:
                Whether blank and whitespace-only strings should be excluded.
            accept_datetime_dtype:
                Whether an existing pandas datetime dtype should return ``True``.

            Returns
            -------
            bool
                Whether the Series satisfies the configured date-parsing policy.
            """
            if not isinstance(series, pd.Series):
                raise TypeError("series must be a pandas Series")
            if not 0 <= threshold <= 1:
                raise ValueError("threshold must be between 0 and 1")
            if limit is not None:
                if isinstance(limit, bool) or not isinstance(limit, int):
                    raise TypeError("limit must be an integer or None")
                if limit <= 0:
                    raise ValueError("limit must be greater than zero")
            if format_policy not in {"single", "combined"}:
                raise ValueError(
                    "format_policy must be either 'single' or 'combined'"
                )
            cleaned_formats = tuple(
                fmt.strip()
                for fmt in formats
                if isinstance(fmt, str) and fmt.strip()
            )
            if len(cleaned_formats) != len(set(cleaned_formats)):
                cleaned_formats = tuple(dict.fromkeys(cleaned_formats))
            # Existing datetime data are unambiguously date-like.
            if pd.api.types.is_datetime64_any_dtype(series.dtype):
                return accept_datetime_dtype
            # Timedeltas are not calendar dates.
            if pd.api.types.is_timedelta64_dtype(series.dtype):
                return False
            # Avoid interpreting True/False or numeric measurements/identifiers as
            # timestamps. Numeric epoch parsing should be a separate explicit policy.
            if (
                pd.api.types.is_bool_dtype(series.dtype)
                or pd.api.types.is_numeric_dtype(series.dtype)
            ):
                return False
            values = series
            if blank_as_missing:
                values = values.map(
                    lambda value: (
                        pd.NA
                        if isinstance(value, str) and not value.strip()
                        else value
                    )
                )
            values = values.dropna()
            if values.empty:
                return False
            if limit is not None and len(values) > limit:
                positions = np.linspace(
                    0,
                    len(values) - 1,
                    num=limit,
                    dtype=int,
                )
                values = values.iloc[positions]
            text = values.astype("string").str.strip()
            if format_policy == "single":
                for fmt in cleaned_formats:
                    parsed = pd.to_datetime(
                        text,
                        format=fmt,
                        errors="coerce",
                    )
                    if parsed.notna().mean() >= threshold:
                        return True
            else:
                # A value succeeds if it matches at least one configured format.
                valid = pd.Series(False, index=text.index)
                for fmt in cleaned_formats:
                    parsed = pd.to_datetime(
                        text,
                        format=fmt,
                        errors="coerce",
                    )
                    valid |= parsed.notna()
                    if valid.mean() >= threshold:
                        return True
            if not allow_mixed:
                return False
            parsed = pd.to_datetime(
                text,
                format="mixed",
                errors="coerce",
                dayfirst=dayfirst,
                yearfirst=yearfirst,
            )
            return bool(parsed.notna().mean() >= threshold)


        def is_nominal_like(
            series: pd.Series,
            *,
            high_cardinality: int = 20,
            max_unique_ratio: float = 0.20,
            allow_ordered_categorical: bool = False,
            allow_numeric: bool = True,
            blank_as_missing: bool = True,
        ) -> bool:
            """Return whether a Series probably represents a nominal variable.

            Explicit unordered categoricals and Boolean columns are treated as
            nominal. Other eligible columns must satisfy both an absolute cardinality
            limit and a relative uniqueness limit.

            Parameters
            ----------
            series:
                Series to inspect.
            high_cardinality:
                Maximum number of distinct values allowed for inferred nominal data.
            max_unique_ratio:
                Maximum ratio of distinct values to non-missing observations.
                Must lie in ``[0, 1]``.
            allow_ordered_categorical:
                Whether an ordered categorical may be classified as nominal.
                Normally ``False`` because ordered categoricals are ordinal.
            allow_numeric:
                Whether low-cardinality numeric columns may be inferred as nominal.
                This is useful for numeric category codes, but can misclassify small
                integer measurements.
            blank_as_missing:
                Whether blank and whitespace-only strings should be excluded.

            Returns
            -------
            bool
                Whether the Series is explicitly or probably nominal.
            """
            if not isinstance(series, pd.Series):
                raise TypeError("series must be a pandas Series")
            if isinstance(high_cardinality, bool) or not isinstance(
                high_cardinality, int
            ):
                raise TypeError("high_cardinality must be an integer")
            if high_cardinality < 1:
                raise ValueError("high_cardinality must be greater than zero")
            if not 0 <= max_unique_ratio <= 1:
                raise ValueError("max_unique_ratio must be between 0 and 1")
            dtype = series.dtype
            # An explicit categorical dtype is stronger evidence than the observed
            # values, even if the Series is empty or entirely missing.
            if isinstance(dtype, pd.CategoricalDtype):
                return bool(not dtype.ordered or allow_ordered_categorical)
            # Boolean is a two-level nominal variable for most inference purposes.
            if pd.api.types.is_bool_dtype(dtype):
                return True
            # Calendar/time values are not nominal merely because the observed sample
            # has few distinct values.
            if (
                pd.api.types.is_datetime64_any_dtype(dtype)
                or pd.api.types.is_timedelta64_dtype(dtype)
                or isinstance(dtype, (pd.PeriodDtype, pd.IntervalDtype))
            ):
                return False
            if pd.api.types.is_numeric_dtype(dtype) and not allow_numeric:
                return False
            values = series
            if blank_as_missing and (
                pd.api.types.is_object_dtype(dtype)
                or pd.api.types.is_string_dtype(dtype)
            ):
                values = values.map(
                    lambda value: (
                        pd.NA
                        if isinstance(value, str) and not value.strip()
                        else value
                    )
                )
            values = values.dropna()
            if values.empty:
                return False
            # nunique() can fail for object columns containing unhashable values such
            # as lists or dictionaries. Those require separate structural inference.
            try:
                cardinality = values.nunique(dropna=True)
            except TypeError:
                return False
            unique_ratio = cardinality / len(values)
            return bool(
                cardinality <= high_cardinality
                and unique_ratio <= max_unique_ratio
            )


        def is_ordered_like(
            series: pd.Series,
            *,
            high_cardinality: int = 20,
            min_cardinality: int = 3,
            max_unique_ratio: float = 0.20,
            known_orders: Sequence[Sequence[Hashable]] = (),
            allow_candidate_inference: bool = True,
            allow_numeric_codes: bool = False,
            blank_as_missing: bool = True,
            require_complete_order: bool = False,
        ) -> bool:
            """Return whether a Series has evidence of representing ordinal data.

            Strong evidence consists of an ordered ``CategoricalDtype`` or values
            matching a configured order. A low-cardinality fallback can be enabled,
            but it establishes only that ordering is plausible—not that an order is
            known.

            Parameters
            ----------
            series:
                Series to inspect.
            high_cardinality:
                Maximum number of distinct values for candidate ordinal inference.
            min_cardinality:
                Minimum number of distinct values for candidate ordinal inference.
            max_unique_ratio:
                Maximum ratio of distinct values to non-missing observations for the
                low-cardinality candidate heuristic.
            known_orders:
                Recognized ordered level sequences, for example
                ``[("low", "medium", "high")]``.
            allow_candidate_inference:
                Whether an otherwise unrecognized low-cardinality column may count as
                ordered-like. This is weak evidence and defaults to ``False``.
            allow_numeric_codes:
                Whether numeric columns may be matched to configured orders or treated
                as ordinal candidates.
            blank_as_missing:
                Whether blank and whitespace-only strings should be excluded.
            require_complete_order:
                When matching ``known_orders``, require every configured level to be
                observed. If ``False``, an observed subset is sufficient.

            Returns
            -------
            bool
                Whether the Series is explicitly ordered or satisfies an enabled
                ordinal inference rule.
            """
            if not isinstance(series, pd.Series):
                raise TypeError("series must be a pandas Series")
            if isinstance(high_cardinality, bool) or not isinstance(
                high_cardinality, int
            ):
                raise TypeError("high_cardinality must be an integer")
            if isinstance(min_cardinality, bool) or not isinstance(
                min_cardinality, int
            ):
                raise TypeError("min_cardinality must be an integer")
            if min_cardinality < 2:
                raise ValueError("min_cardinality must be at least 2")
            if high_cardinality < min_cardinality:
                raise ValueError(
                    "high_cardinality must be at least min_cardinality"
                )
            if not 0 <= max_unique_ratio <= 1:
                raise ValueError("max_unique_ratio must be between 0 and 1")
            dtype = series.dtype
            # Explicit dtype metadata is authoritative, even for an empty or
            # entirely missing Series.
            if isinstance(dtype, pd.CategoricalDtype) and dtype.ordered:
                return True
            # Boolean values have two levels but usually lack an ordinal scale.
            if pd.api.types.is_bool_dtype(dtype):
                return False
            # Calendar and elapsed-time data are ordered quantities, not ordinal
            # categorical variables.
            if (
                pd.api.types.is_datetime64_any_dtype(dtype)
                or pd.api.types.is_timedelta64_dtype(dtype)
                or isinstance(dtype, (pd.PeriodDtype, pd.IntervalDtype))
            ):
                return False
            if pd.api.types.is_numeric_dtype(dtype) and not allow_numeric_codes:
                return False
            values = series
            if blank_as_missing and (
                pd.api.types.is_object_dtype(dtype)
                or pd.api.types.is_string_dtype(dtype)
            ):
                values = values.map(
                    lambda value: (
                        pd.NA
                        if isinstance(value, str) and not value.strip()
                        else value
                    )
                )
            values = values.dropna()
            if values.empty:
                return False
            try:
                observed = set(values.unique())
            except TypeError:
                # Lists, dictionaries, and other unhashable scalars are not suitable
                # ordinal levels for this implementation.
                return False
            # Recognized scales provide actual evidence of an order.
            for order in known_orders:
                levels = tuple(order)
                if len(levels) < 2:
                    raise ValueError(
                        "each entry in known_orders must contain at least two levels"
                    )
                try:
                    level_set = set(levels)
                except TypeError as exc:
                    raise TypeError(
                        "known order levels must be hashable"
                    ) from exc
                if len(level_set) != len(levels):
                    raise ValueError(
                        "known order levels must not contain duplicates"
                    )
                if require_complete_order:
                    matches = observed == level_set
                else:
                    matches = observed.issubset(level_set)
                if matches:
                    return True
            if not allow_candidate_inference:
                return False
            cardinality = len(observed)
            unique_ratio = cardinality / len(values)
            return bool(
                min_cardinality <= cardinality <= high_cardinality
                and unique_ratio <= max_unique_ratio
            )


        @this.record_code
        def convert_series(series: pd.Series, new_type: str, *, order: [str] | None, formats: list[str]) -> pd.Series:
            """
            Convert one pandas Series according to the selected target type.
            """
            if new_type == "decimal":
                return pd.to_numeric(series, errors="coerce").astype("Float64")
            if new_type == "integer":
                return pd.to_numeric(series, errors="coerce").round().astype("Int64")
            if new_type == "date":
                for fmt in formats:
                    parsed = pd.to_datetime(series, format=fmt, errors="coerce")
                    if parsed.notna().sum() > 0:
                        return parsed
                return pd.to_datetime(series, errors="coerce")
            if new_type == "text":
                return as_text(series)
            if new_type == "nominal":
                return series.astype("category")
            if new_type == "ordered":
                return pd.Categorical(series.astype(str), categories=order.split(","), ordered=True)
            if new_type == "cyclic":
                if pd.api.types.is_numeric_dtype(series):
                    return as_cyclic(series)
                if not isinstance(series.dtype, pd.CategoricalDtype):
                    series = pd.Categorical(series.astype(str), categories=order.split(","), ordered=True)
                if series.dtype.ordered:
                    return as_cyclic(series)
                return as_cyclic(series)
            if new_type == "list":
                return as_list(series)
            if new_type == "geometry":
                return as_geometry(series)
            if new_type == "code":
                return series.astype("string")
            raise ValueError(f"Unsupported conversion type: {new_type}")


    this.server = server

    return this


if Module.running_directly(name=__name__):
    this = instance()
    df = pd.read_csv(Card.ROOT / "data" / "Assmnt.csv")
    px = pxd(_df=df, _name="Ass2")
    this._imports.set(px)
    this.run()
