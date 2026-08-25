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

from collections import Counter

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shinywidgets
from card import Card
from cyclic_pandas import is_cyclic
from geometry_pandas import is_geometry
from list_pandas import is_list
from module import Module
from proxy_data import proxy_data
from roles import Role
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.stats import skew
from shiny import render, req, ui
from shinywidgets import render_widget

#TODO: expose correlation extra-weighting as a setting
#TODO: employ observations weights when available (How?)

def instance():
    """Create the immutable variable-dissimilarity card."""
    this = Card(file=__file__, mutable=False)
    this.long_name = "Variable Dissimilarity"
    this.description = (
        "This card explores the characteristics of each variable and builds "
        "a dissimilarity matrix visualised as a hierarchy chart."
    )

    def front():
        return ui.TagList(
            ui.div(
                ui.span(
                    "Variable Dissimilarity chart",
                    class_="text-primary text-center d-block",
                ),
                shinywidgets.output_widget(
                    id="Chart",
                    fill=True,
                    guide=this,
                    title="Chart of variable dissimilarity",
                    text=(
                        "A hierarchy derived from the variable dissimilarity "
                        "matrix. Nearby leaves describe variables with similar "
                        "names, summaries, missingness, or values."
                    ),
                    position="left",
                ),
                id="X-Chart",
                class_="html-fill-container html-fill-item",
                style="width:100%; height:100%;",
            )
        )

    this.front = front

    def back():
        return ui.TagList(
            ui.span(
                "Variable Dissimilarity table",
                class_="text-primary text-center d-block",
            ),
            ui.output_ui( # Using dynamic data tables to avoid "sortable" problem of multiple tables
                id = "Table",
                guide=this,
                title="Table of variable dissimilarity",
                text=(
                    "A symmetric matrix in which zero means identical and "
                    "larger values mean more dissimilar variables."
                ),
                position="left"
            ),
        )

    this.back = back

    def settings():
        return ui.TagList(
            ui.input_checkbox(
                id="Robust",
                label="Employ robust statistics for central tendency and spread",
                value=True,
                guide=this,
                text="Robust uses the median and MAD; otherwise mean and standard deviation are used.",
                position="left",
            ),
            ui.input_slider(
                id="Qgram",
                label="The size of q-grams",
                min=1,
                max=5,
                value=2,
                guide=this,
                text="Q-grams compare variable names and dtype names. Values of one or two are usually appropriate.",
                position="left",
            ),
            ui.input_radio_buttons(
                id="Which",
                label="Hierarchical clustering technique",
                choices=["Agglomerative", "Divisive"],
                selected="Agglomerative",
                guide=this,
                text="Choose bottom-up agglomerative or top-down divisive clustering.",
                position="left",
            ),
            ui.input_radio_buttons(
                id="Style",
                label="Hierarchy chart layout",
                choices={
                    "rectangular": "Rectangular",
                    "radial": "Radial",
                },
                selected="radial",
                guide=this,
                text="Rectangular and radial chart layouts of the same hierarchy information.",
                position="left",
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
                position = "left"
            ),
        )

    this.settings = settings

    def server(input, output, session):

        @this.throttle(2)
        @this.suspendable(calc = True)
        def MaxObs():
            return 10**input.MaxObs()

        @this.suspendable(calc=True)
        def incomingproxy_data():
            req(this._imports.is_set())
            return this._imports.get()

        @this.throttle(2)
        @this.suspendable(calc=True)
        def Qgram():
            return max(1, int(input.Qgram()))

        @this.suspendable(calc=True)
        @this.record_code
        def PreparedData():
            samp = incomingproxy_data().sample(n=MaxObs(), mode="random", keep_geometry=True)
            return samp

        @this.suspendable(calc=True)
        @this.record_code
        def CleanDf():
            pxd = PreparedData()
            predictors = pxd.role_map.columns_with_role(Role.PREDICTOR)
            frame = pxd.to_native()
            if not isinstance(frame, pd.DataFrame):
                raise TypeError("Variable dissimilarity requires tabular data")
            keep = [
                column for column in frame.columns
                if not str(column).startswith(Card.SHADOW_PREFIX) and column in predictors
            ]
            frame = pd.DataFrame(frame.loc[:, keep])
            geometry_columns = [
                column for column in frame.columns
                if getattr(frame[column].dtype, "name", None) == "geometry"
            ]
            excluded = set(geometry_columns)
            excluded.update(
                column for column in frame.columns
                if str(column).startswith(Card.SHADOW_PREFIX)
            )
            return pd.DataFrame(frame.drop(columns=list(excluded), errors="ignore")).copy()

        @this.record_code
        def _safe_scale(values: np.ndarray) -> np.ndarray:
            values = np.asarray(values, dtype=float)
            finite = np.isfinite(values)
            if not finite.any():
                return np.full(values.shape, np.nan, dtype=float)
            scale = np.max(np.abs(values[finite]))
            if scale == 0:
                result = np.zeros(values.shape, dtype=float)
                result[~finite] = np.nan
                return result
            return values / scale

        @this.record_code
        def _numeric_stats(frame: pd.DataFrame, *, robust: bool) -> pd.DataFrame:
            n_rows = len(frame)
            rows = []
            for column in frame.columns:
                series = frame[column]
                dtype = series.dtype
                # nunique() can fail for object columns containing unhashable values such
                # as lists or dictionaries. Those require separate structural inference.
                try:
                    cardinality = series.nunique(dropna=True) / n_rows if n_rows else np.nan
                except TypeError:
                    cardinality = None
                centre = spread = asymmetry = np.nan
                if is_list(dtype) or is_geometry(dtype):
                    # Neither exact-value frequencies nor numeric moments provide a
                    # useful summary for collection-valued or spatial variables.
                    pass
                elif is_cyclic(dtype):
                    observed = series.dropna()
                    if len(observed):
                        if dtype.is_categorical:
                            positions = observed.map(
                                {value: index for index, value in enumerate(dtype.categories)}
                            ).to_numpy(dtype=float)
                            period = len(dtype.categories)
                        else:
                            positions = pd.to_numeric(
                                observed, errors="coerce"
                            ).to_numpy(dtype=float)
                            period = float(dtype.period)
                        angles = 2 * np.pi * positions / period
                        resultant = np.hypot(
                            np.mean(np.cos(angles)),
                            np.mean(np.sin(angles)),
                        )
                        spread = float(1 - resultant)
                elif pd.api.types.is_datetime64_any_dtype(dtype):
                    values = series.astype("int64").to_numpy(dtype=float)
                    values[series.isna().to_numpy()] = np.nan
                    finite = values[np.isfinite(values)]
                    if finite.size:
                        centre = float(np.median(finite) if robust else np.mean(finite))
                        if robust:
                            spread = float(1.4826 * np.median(np.abs(finite - np.median(finite))))
                        elif finite.size > 1:
                            spread = float(np.std(finite, ddof=1))
                        if finite.size > 2 and not np.allclose(finite, finite[0]):
                            asymmetry = float(skew(finite, bias=True, nan_policy="omit"))
                elif pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
                    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
                    finite = values[np.isfinite(values)]
                    if finite.size:
                        centre = float(np.median(finite) if robust else np.mean(finite))
                        if robust:
                            spread = float(1.4826 * np.median(np.abs(finite - np.median(finite))))
                        elif finite.size > 1:
                            spread = float(np.std(finite, ddof=1))
                        if finite.size > 2 and not np.allclose(finite, finite[0]):
                            asymmetry = float(skew(finite, bias=True, nan_policy="omit"))
                else:
                    counts = series.value_counts(dropna=True)
                    if counts.sum():
                        spread = float(1 - counts.iloc[0] / counts.sum())
                rows.append((cardinality, centre, spread, asymmetry))
            stats = pd.DataFrame(
                rows,
                index=frame.columns,
                columns=["Cardinality", "Centrality", "Spread", "Skew"],
                dtype=float,
            )
            stats["Centrality"] = _safe_scale(stats["Centrality"].to_numpy())
            numeric_centre = np.asarray([row[1] for row in rows], dtype=float)
            numeric_spread = stats["Spread"].to_numpy(copy=True)
            is_numeric = np.isfinite(numeric_centre)
            numeric_spread[is_numeric] /= np.maximum(np.abs(numeric_centre[is_numeric]), 1.0)
            stats["Spread"] = numeric_spread
            stats["Skew"] = _safe_scale(stats["Skew"].to_numpy())
            return stats

        @this.record_code
        def _cosine_rows(values: np.ndarray) -> np.ndarray:
            values = np.asarray(values, dtype=float)
            count = values.shape[0]
            result = np.full((count, count), np.nan, dtype=float)
            np.fill_diagonal(result, 0.0)
            for i in range(count):
                for j in range(i + 1, count):
                    valid = np.isfinite(values[i]) & np.isfinite(values[j])
                    if not valid.any():
                        continue
                    left, right = values[i, valid], values[j, valid]
                    denominator = np.linalg.norm(left) * np.linalg.norm(right)
                    if denominator == 0:
                        distance = 0.0 if np.allclose(left, right) else np.nan
                    else:
                        similarity = np.dot(left, right) / denominator
                        distance = 1 - float(np.clip(similarity, -1, 1))
                    result[i, j] = result[j, i] = distance
            return result

        @this.record_code
        def _qgrams(value: object, q: int) -> Counter:
            text = str(value).casefold()
            if len(text) < q:
                return Counter([text]) if text else Counter()
            return Counter(text[i:i + q] for i in range(len(text) - q + 1))

        @this.record_code
        def _string_cosine(values: list[object], q: int) -> np.ndarray:
            grams = [_qgrams(value, q) for value in values]
            count = len(values)
            result = np.zeros((count, count), dtype=float)
            for i in range(count):
                for j in range(i + 1, count):
                    keys = grams[i].keys() | grams[j].keys()
                    dot = sum(grams[i][key] * grams[j][key] for key in keys)
                    left = np.sqrt(sum(value * value for value in grams[i].values()))
                    right = np.sqrt(sum(value * value for value in grams[j].values()))
                    distance = 1 - dot / (left * right) if left and right else float(grams[i] != grams[j])
                    result[i, j] = result[j, i] = distance
            return result

        @this.record_code
        def _value_correlation_distance(frame: pd.DataFrame) -> np.ndarray:
            count = frame.shape[1]
            result = np.full((count, count), np.nan, dtype=float)
            np.fill_diagonal(result, 0.0)
            numeric = frame.select_dtypes(include=["number"]).select_dtypes(exclude=["bool"])
            if numeric.empty:
                return result
            if input.Robust():
                correlation = numeric.corr(method="spearman", min_periods=2).abs()
            else:
                correlation = numeric.corr(method="pearson", min_periods=2).abs()
            positions = {column: position for position, column in enumerate(frame.columns)}
            for left in correlation.columns:
                for right in correlation.columns:
                    value = correlation.loc[left, right]
                    if pd.notna(value):
                        result[positions[left], positions[right]] = 1 - float(value)
            return result

        @this.record_code
        def _missingness_distance(frame: pd.DataFrame, *, min_events: int = 1) -> np.ndarray:
            """
            Calculate Jaccard distances between columns' missingness patterns.

            Missing values are treated as events. Jointly observed values provide no
            evidence of similar missingness and therefore do not affect the distance.

            Distance:
                0 = the variables are missing on exactly the same rows
                1 = their missing rows do not overlap

            Comparisons involving fewer than ``min_events`` distinct missing rows
            return NaN because there is insufficient missingness evidence. The
            diagonal is always zero.
            """
            if isinstance(min_events, bool) or not isinstance(min_events, int):
                raise TypeError("min_events must be an integer")
            if min_events < 1:
                raise ValueError("min_events must be at least 1")
            # Use int64 because int8 matrix multiplication can overflow.
            missing = frame.isna().to_numpy(dtype=np.int64)
            # Number of rows where both columns are missing.
            intersection = missing.T @ missing
            # Number of rows where either column is missing.
            missing_counts = missing.sum(axis=0)
            union = (
                missing_counts[:, None]
                + missing_counts[None, :]
                - intersection
            )
            similarity = np.full(intersection.shape, np.nan, dtype=float)
            sufficiently_observed = union >= min_events
            np.divide(
                intersection,
                union,
                out=similarity,
                where=sufficiently_observed,
            )
            distance = 1.0 - similarity
            np.fill_diagonal(distance, 0.0)
            return distance

        @this.suspendable(calc=True)
        @this.record_code
        def DissimilarityMatrix():
            pxd = PreparedData()
            predictors = pxd.role_map.columns_with_role(Role.PREDICTOR)
            frame = pxd.to_native()
            if not isinstance(frame, pd.DataFrame):
                raise TypeError("Variable dissimilarity requires tabular data")
            keep = [
                column for column in frame.columns
                if not str(column).startswith(Card.SHADOW_PREFIX) and column in predictors
            ]
            frame = pd.DataFrame(frame.loc[:, keep]).copy()
            names = frame.columns.astype(str).tolist()
            count = len(names)
            if count == 0:
                return pd.DataFrame(dtype=float)
            stats = _numeric_stats(frame, robust=bool(input.Robust()))
            matrices = [
                _cosine_rows(stats.to_numpy()),
                _string_cosine(names, Qgram()),
                _value_correlation_distance(frame),
                _missingness_distance(frame)
            ]
            weights = np.asarray([1, 1, 5, 1], dtype=float)  # Herein lies the BIG assumption (heavy weighting to correlation)
            total = np.zeros((count, count), dtype=float)
            available = np.zeros((count, count), dtype=float)
            for matrix, weight in zip(matrices, weights):
                valid = np.isfinite(matrix)
                total[valid] += weight * matrix[valid]
                available[valid] += weight
            result = np.divide(
                total,
                available,
                out=np.zeros_like(total),
                where=available > 0,
            )
            result = np.clip((result + result.T) / 2, 0, 2)
            np.fill_diagonal(result, 0.0)
            return pd.DataFrame(result, index=names, columns=names)

        @this.record_code
        def _divisive_linkage(distance: np.ndarray) -> np.ndarray:
            """Return a SciPy linkage representation of a DIANA-style hierarchy."""
            distance = np.asarray(distance, dtype=float)
            count = len(distance)
            rows = []

            def split(members: list[int]) -> tuple[list[int], list[int]]:
                if len(members) == 2:
                    return [members[0]], [members[1]]
                sub = distance[np.ix_(members, members)]
                seed = members[int(np.argmax(sub.sum(axis=1) / (len(members) - 1)))]
                splinter = [seed]
                remainder = [member for member in members if member != seed]
                while len(remainder) > 1:
                    gains = []
                    for member in remainder:
                        others = [item for item in remainder if item != member]
                        to_remainder = np.mean([distance[member, item] for item in others])
                        to_splinter = np.mean([distance[member, item] for item in splinter])
                        gains.append((to_remainder - to_splinter, member))
                    gain, member = max(gains)
                    if gain <= 0:
                        break
                    remainder.remove(member)
                    splinter.append(member)
                return remainder, splinter

            def build(members: list[int]) -> tuple[int, float, int]:
                if len(members) == 1:
                    return members[0], 0.0, 1
                left_members, right_members = split(members)
                left_id, left_height, left_count = build(left_members)
                right_id, right_height, right_count = build(right_members)
                between = distance[np.ix_(left_members, right_members)]
                height = max(float(np.max(between)), left_height, right_height)
                rows.append([left_id, right_id, height, left_count + right_count])
                return count + len(rows) - 1, height, left_count + right_count

            build(list(range(count)))
            return np.asarray(rows, dtype=float)

        @this.record_code
        def _hierarchy(distance_frame: pd.DataFrame, technique: str) -> np.ndarray:
            if len(distance_frame) < 2:
                return np.empty((0, 4), dtype=float)
            values = distance_frame.to_numpy(dtype=float)
            finite = values[np.triu_indices_from(values, k=1)]
            replacement = float(np.nanmean(finite)) if np.isfinite(finite).any() else 1.0
            values = np.nan_to_num(values, nan=replacement, posinf=replacement, neginf=0.0)
            values = np.maximum((values + values.T) / 2, 0)
            np.fill_diagonal(values, 0)
            if technique == "Divisive":
                return _divisive_linkage(values)
            return linkage(squareform(values, checks=False), method="average")

        @this.record_code
        def _hierarchy_figure(
            distance_frame: pd.DataFrame,
            *,
            technique: str,
            style: str,
        ) -> go.Figure:
            names = distance_frame.columns.astype(str).tolist() if distance_frame is not None else None
            if not names:
                figure = go.Figure()
                figure.add_annotation(text="No variables to compare", x=0.5, y=0.5, showarrow=False)
                return figure
            if len(names) == 1:
                figure = go.Figure(go.Scatter(x=[0], y=[0], text=names, mode="markers+text"))
                figure.update_layout(xaxis_visible=False, yaxis_visible=False)
                return figure
            hierarchy = _hierarchy(distance_frame, technique)
            tree = dendrogram(hierarchy, labels=names, no_plot=True)
            radial = style == "radial"
            figure = go.Figure()
            if radial:
                maximum = max(max(values) for values in tree["dcoord"]) or 1.0
                leaf_count = len(names)
                for xs, ys in zip(tree["icoord"], tree["dcoord"]):
                    dense_x, dense_y = [], []
                    for index in range(3):
                        dense_x.extend(np.linspace(xs[index], xs[index + 1], 12, endpoint=False))
                        dense_y.extend(np.linspace(ys[index], ys[index + 1], 12, endpoint=False))
                    dense_x.append(xs[-1])
                    dense_y.append(ys[-1])
                    theta = (np.asarray(dense_x) - 5) / (10 * leaf_count) * 2 * np.pi
                    radius = maximum - np.asarray(dense_y)
                    figure.add_trace(go.Scatterpolar(
                        theta=np.degrees(theta), r=radius, mode="lines",
                        line={"color": "#2c3e50"}, hoverinfo="skip", showlegend=False,
                    ))
                leaf_theta = np.arange(leaf_count) / leaf_count * 360
                figure.add_trace(go.Scatterpolar(
                    theta=leaf_theta,
                    r=np.repeat(maximum, leaf_count),
                    text=tree["ivl"],
                    mode="markers+text",
                    textposition="top center",
                    marker={"size": 7, "color": "#2c3e50"},
                    hovertemplate="%{text}<extra></extra>",
                    showlegend=False,
                ))
                figure.update_layout(
                    polar={
                        "radialaxis": {"visible": False},
                        "angularaxis": {"visible": False},
                    }
                )
            else:
                for xs, ys in zip(tree["icoord"], tree["dcoord"]):
                    figure.add_trace(go.Scatter(
                        x=xs, y=ys, mode="lines",
                        line={"color": "#2c3e50"}, hoverinfo="skip", showlegend=False,
                    ))
                ticks = [5 + 10 * index for index in range(len(tree["ivl"]))]
                figure.update_xaxes(tickmode="array", tickvals=ticks, ticktext=tree["ivl"])
                figure.update_yaxes(title="Dissimilarity")
            figure.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#e5ecf6",
                margin={"l": 20, "r": 20, "t": 15, "b": 45},
                showlegend=False,
            )
            return figure

        @output
        @render_widget
        def Chart():
            matrix = DissimilarityMatrix()
            fig = _hierarchy_figure(
                matrix,
                technique=input.Which(),
                style=input.Style(),
            )
            fig.update_layout(
                plot_bgcolor="#e5ecf6",
                margin={"l": 20, "r": 20, "t": 20, "b": 20},
                modebar={"orientation": "v"},
                modebar_remove=[
                    "pan2d",
                    "zoom2d",
                    "select2d",
                    "lasso2d",
                    "zoomIn2d",
                    "zoomOut2d",
                    "autoScale2d",
                    "toggleHover",
                    "toggleSpikelines",
                    "hoverClosestCartesian",
                    "hoverCompareCartesian",
                ]
            )
            fw = go.FigureWidget(fig)
            fw._config = (getattr(fw, "_config", {}) | {
                "displayModeBar": bool(this.isFullScreen()),
                "displaylogo": False,
                "responsive": True
            })
            return fw


        @output
        @render.ui
        def Table():
            req(PreparedData() is not None)
            return ui.output_data_frame(id = "Table2")

        @output
        @render.data_frame
        def Table2():
            req(DissimilarityMatrix() is not None)
            matrix = DissimilarityMatrix().round(3)
            table = matrix.copy()
            table.insert(0, "Variable", table.index)
            table.reset_index(drop=True)
            return render.DataTable(table, width="100%")

    this.server = server
    return this


if Module.running_under_tests():
    this = instance()
    df = pd.DataFrame({
        "y": [1, 0, 1, 0],
        "x1": [10.0, 11.0, 12.0, 13.0],
        "x2": ["A", "B", "A", "B"],
        "id": [100, 101, 102, 103],
        "part": ["Train", "Train", "Test", "Test"],
    })
    this._imports.set(proxy_data(_df=df, _name="Test"))
    app = this.application()
elif Module.running_directly(name=__name__):
    this = instance()
    df = pd.read_csv(Card.ROOT / "data" / "Ass2.csv")
    this._imports.set(proxy_data(_df=df, _name="Ass2"))
    this.run()
