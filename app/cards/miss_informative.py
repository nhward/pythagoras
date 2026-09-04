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

import asyncio
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shinywidgets
from card import Card
from module import Module
from proxy_data import proxy_data
from roles import Role
from shiny import reactive, render, req, ui
from shinywidgets import render_widget
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.multiclass import type_of_target


@dataclass
class ForestAnalysis:
    """Cross-validated Random Forest performance and permutation importance."""

    target: str | None
    task: str | None
    score_name: str
    score: float
    score_sd: float
    folds: int
    observations: int
    importance: pd.DataFrame
    message: str | None = None


IMPORTANCE_COLUMNS = [
    "Variable",
    "Variable Type",
    "Source Variable",
    "Missing Proportion",
    "Importance",
    "Importance SD",
    "Positive Fraction",
    "Interpretation",
]

def _empty_analysis(target: str | None, message: str) -> ForestAnalysis:
    return ForestAnalysis(
        target=target,
        task=None,
        score_name="Score",
        score=np.nan,
        score_sd=np.nan,
        folds=0,
        observations=0,
        importance=pd.DataFrame(columns=IMPORTANCE_COLUMNS),
        message=message,
    )


def _feature_frame(
    frame: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Create stable numeric/categorical columns for fold-local preprocessing."""
    converted: dict[str, pd.Series] = {}
    numeric: list[str] = []
    categorical: list[str] = []
    for column in columns:
        series = frame[column]
        if getattr(series.dtype, "name", None) == "geometry":
            continue
        name = str(column)
        if pd.api.types.is_datetime64_any_dtype(series.dtype):
            values = series.astype("int64", copy=False).astype(float)
            values[series.isna()] = np.nan
            converted[name] = values
            numeric.append(name)
        elif (
            pd.api.types.is_numeric_dtype(series.dtype)
            and not pd.api.types.is_bool_dtype(series.dtype)
        ):
            converted[name] = pd.to_numeric(series, errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            numeric.append(name)
        else:
            try:
                values = series.astype("string").astype(object)
                values.loc[series.isna()] = np.nan
            except (TypeError, ValueError):
                continue
            converted[name] = values
            categorical.append(name)
    return pd.DataFrame(converted, index=frame.index), numeric, categorical


def _forest_pipeline(
    *,
    task: str,
    numeric: list[str],
    categorical: list[str],
    random_state: int,
) -> Pipeline:
    transformers = []
    if numeric:
        transformers.append((
            "numeric",
            SimpleImputer(strategy="median", keep_empty_features=True),
            numeric,
        ))
    if categorical:
        transformers.append((
            "categorical",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent", keep_empty_features=True),
                ),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical,
        ))
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    if task == "classification":
        forest = RandomForestClassifier(random_state=random_state, n_jobs=Module.N_JOBS)
    else:
        forest = RandomForestRegressor(random_state=random_state, n_jobs=Module.N_JOBS)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("forest", forest),
    ])


def _fit_forest_importance(
    frame: pd.DataFrame,
    *,
    target: str | None,
    predictors: list[str],
    missing_variables: list[str],
    sample_weight: pd.Series | None = None,
    cv_folds: int = 5,
    minimum_balanced_accuracy: float = 0.55,
    random_state: int = 2025,
) -> ForestAnalysis:
    """Estimate held-out permutation importance across cross-validation folds."""
    if target is None:
        return _empty_analysis(None, "The Target role is not assigned")
    if target not in frame.columns:
        return _empty_analysis(target, f"The target {target!r} is not in the data")
    predictors = [
        column for column in frame.columns
        if column in set(predictors) and column != target
    ]
    if not predictors:
        return _empty_analysis(target, "No predictor variables are assigned")
    if not missing_variables:
        return _empty_analysis(target, "No predictors exceed the minimum missing-value proportion")
    working = frame.copy()
    missing_proportions = {
        column: float(working[column].isna().mean())
        for column in missing_variables
    }
    shadow_sources: dict[str, str] = {}
    for column in missing_variables:
        shadow = f"{Card.SHADOW_PREFIX}{column}"
        working[shadow] = working[column].isna().astype(np.int8)
        shadow_sources[shadow] = column
        if shadow not in predictors:
            predictors.append(shadow)
    observed_target = working[target].notna()
    working = working.loc[observed_target].copy()
    if len(working) < 4:
        return _empty_analysis(target, "Too few observations have a recorded target")
    y = working[target]
    try:
        target_kind = type_of_target(y)
    except (TypeError, ValueError) as error:
        return _empty_analysis(target, f"Unsupported target type: {error}")
    if target_kind in {"binary", "multiclass"}:
        task = "classification"
        counts = y.value_counts(dropna=False)
        if len(counts) < 2 or int(counts.min()) < 2:
            return _empty_analysis(
                target,
                "Classification requires at least two observations in every class",
            )
        folds = min(max(2, int(cv_folds)), int(counts.min()))
        splitter = StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=random_state,
        )
        splits = splitter.split(working, y)
        scoring = "balanced_accuracy"
        score_name = "CV Balanced Accuracy"
    elif target_kind == "continuous":
        task = "regression"
        folds = min(max(2, int(cv_folds)), len(working) // 2)
        if folds < 2:
            return _empty_analysis(target, "Too few observations for regression")
        splitter = KFold(
            n_splits=folds,
            shuffle=True,
            random_state=random_state,
        )
        splits = splitter.split(working)
        scoring = "r2"
        score_name = "CV R-squared"
    else:
        return _empty_analysis(target, f"Unsupported target type: {target_kind}")
    features, numeric, categorical = _feature_frame(working, predictors)
    if features.shape[1] == 0:
        return _empty_analysis(target, "No predictors can be represented for modeling")
    y = y.loc[features.index]
    weights = None
    if sample_weight is not None:
        weights = pd.to_numeric(
            sample_weight.reindex(frame.index).loc[observed_target],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)
        fill = float(weights.dropna().median()) if weights.notna().any() else 1.0
        weights = weights.fillna(fill).clip(lower=0)
        if not bool(weights.gt(0).any()):
            weights = None
    fold_scores: list[float] = []
    fold_importances: list[np.ndarray] = []
    for fold, (train, test) in enumerate(splits):
        model = _forest_pipeline(
            task=task,
            numeric=numeric,
            categorical=categorical,
            random_state=random_state + fold,
        )
        train_weight = weights.iloc[train].to_numpy() if weights is not None else None
        test_weight = weights.iloc[test].to_numpy() if weights is not None else None
        fit_options = (
            {"forest__sample_weight": train_weight}
            if train_weight is not None
            else {}
        )
        model.fit(features.iloc[train], y.iloc[train], **fit_options)
        prediction = model.predict(features.iloc[test])
        if task == "classification":
            score = balanced_accuracy_score(
                y.iloc[test], prediction, sample_weight=test_weight
            )
        else:
            score = r2_score(y.iloc[test], prediction, sample_weight=test_weight)
        fold_scores.append(float(score))
        importance = permutation_importance(
            model,
            features.iloc[test],
            y.iloc[test],
            scoring=scoring,
            n_repeats=5,
            n_jobs=Module.N_JOBS,
            random_state=random_state + fold,
            sample_weight=test_weight,
        )
        fold_importances.append(np.asarray(importance.importances, dtype=float))
    importance_values = np.concatenate(fold_importances, axis=1)
    importance_mean = np.mean(importance_values, axis=1)
    importance_sd = np.std(importance_values, axis=1, ddof=1)
    positive_fraction = np.mean(importance_values > 0, axis=1)
    adequate_model = (
        float(np.mean(fold_scores)) >= float(minimum_balanced_accuracy)
        if task == "classification"
        else float(np.mean(fold_scores)) > 0
    )
    rows = []
    for index, variable in enumerate(features.columns):
        source = shadow_sources.get(variable)
        is_shadow = source is not None
        informative = bool(
            is_shadow
            and adequate_model
            and importance_mean[index] > 0
            and positive_fraction[index] >= 0.8
        )
        rows.append({
            "Variable": variable,
            "Variable Type": "Shadow" if is_shadow else "Predictor",
            "Source Variable": source,
            "Missing Proportion": missing_proportions.get(
                source if is_shadow else variable,
                np.nan,
            ),
            "Importance": float(importance_mean[index]),
            "Importance SD": float(importance_sd[index]),
            "Positive Fraction": float(positive_fraction[index]),
            "Interpretation": "Informative" if informative else "Uninformative",
        })
    table = pd.DataFrame(rows, columns=IMPORTANCE_COLUMNS).sort_values(
        ["Importance", "Variable"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    return ForestAnalysis(
        target=target,
        task=task,
        score_name=score_name,
        score=float(np.mean(fold_scores)),
        score_sd=float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0,
        folds=folds,
        observations=len(working),
        importance=table,
    )


def _importance_figure(
    analysis: ForestAnalysis,
    *,
    maximum_variables: int = 25,
) -> go.Figure:
    if analysis.message:
        return Card.empty_figure(analysis.message)
    table = analysis.importance.copy()
    table["Importance"] = pd.to_numeric(table["Importance"], errors="coerce")
    table = table.loc[np.isfinite(table["Importance"])].copy()
    if table.empty:
        return Card.empty_figure("Variable importance could not be estimated")
    table = table.head(maximum_variables).sort_values(
        "Importance", ascending=True
    )
    importance_sd = pd.to_numeric(
        table["Importance SD"], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    positive_fraction = pd.to_numeric(
        table["Positive Fraction"], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    missing_proportion = pd.to_numeric(
        table["Missing Proportion"], errors="coerce"
    ).replace([np.inf, -np.inf], np.nan)
    positive_labels = [
        f"{value:.0%}" if pd.notna(value) else "Unavailable"
        for value in positive_fraction
    ]
    missing_labels = [
        f"{value:.1%}" if pd.notna(value) else "Not applicable"
        for value in missing_proportion
    ]
    colours = np.where(
        table["Variable Type"].eq("Shadow"),
        "#ffc107",
        "#0d6efd",
    )
    labels = table["Variable"].where(
        table["Variable Type"].ne("Shadow"),
        "Missing: " + table["Source Variable"].astype(str),
    )
    figure = go.Figure(go.Bar(
        x=table["Importance"],
        y=labels,
        orientation="h",
        marker={"color": colours},
        error_x={
            "type": "data",
            "array": importance_sd,
            "visible": True,
        },
        customdata=list(zip(
            table["Variable Type"].astype(str),
            positive_labels,
            missing_labels,
            strict=True,
        )),
        hovertemplate=(
            "%{y}<br>Importance: %{x:.4f}"
            "<br>Type: %{customdata[0]}"
            "<br>Positive fraction: %{customdata[1]}"
            "<br>Missing proportion: %{customdata[2]}<extra></extra>"
        ),
    ))
    figure.add_vline(x=0, line_color="#6c757d", line_width=1)
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#bbd6f8",
        margin={"l": 20, "r": 25, "t": 20, "b": 45},
        xaxis_title="Held-out permutation importance (score decrease)",
        yaxis_title=None,
        showlegend=False,
    )
    return figure


def _add_shadow_variables(
    data: proxy_data,
    columns: list[str] | tuple[str, ...],
) -> proxy_data:
    """Add selected missingness indicators through ProxyData's mutation API."""
    selected = list(dict.fromkeys(map(str, columns)))
    if not selected:
        return data.clone()

    result = data.clone()
    frame = result.frame.copy()
    shadow_columns = []
    for column in selected:
        if column not in frame.columns:
            raise KeyError(f"Cannot create a shadow for unknown variable {column!r}")
        shadow = f"{Card.SHADOW_PREFIX}{column}"
        frame[shadow] = frame[column].isna().to_numpy(dtype=np.int8)
        result.role_map.set_roles(shadow, [Role.PREDICTOR])
        shadow_columns.append(shadow)

    return data.with_cleaned_data(
        frame,
        card="miss_informative",
        operation="Add informative-missingness shadow variables",
        parameters={
            "source_variables": selected,
            "shadow_variables": shadow_columns,
        },
        role_map=result.role_map,
    )

def instance():
    """Create the mutable missingness-type card."""
    this = Card(file=__file__, mutable=True)
    this.long_name = "Informative Missingness"
    this.description = "This card uses a Random Forest model to assess the informative missingness of each predictor's missing values."

    def front():
        return ui.TagList(
            ui.span("Cross-validated variable importance", class_="text-primary text-center d-block"),
            shinywidgets.output_widget(
                id="Importance", fill=True, guide=this, position="left",
                title="Variable importance chart", text="Variable importance of variables of a Random Forest model predicting the target."
            )
        )

    this.front = front

    def back():
        return ui.TagList(
            ui.span("Cross-validated shadow-variable importance", class_="text-primary text-center d-block"),
            ui.output_ui(
                id="Table", guide=this, title="Variable importance table", position="left",
                text="""
                    Shadow rows describe whether each predictor's missingness 
                    helps predict the target. The 'Interpretation' column is the conclusion.
                    The columns are
                    <ul>
                    <li>Variable: The variable name</li>
                    <li>Variable type: Standard predictor or Shadow predictor</li>
                    <li>Source Variable: The underlying variable name</li>
                    <li>Missing proportion: Fraction missing values</li>
                    <li>Importance: Mean held-out permutation importance across cross-validation 
                    folds</li>
                    <li>Importance standard deviation: Spread based on permuations</li>
                    <li>Positive Fraction: The proportion of permutations where the importance is greater than zero</li>
                    <li>Interpretation: Informative or not</li>
                    </ul>
                    """
            ),
        )

    this.back = back

    def footer():
        return ui.div(
            ui.output_ui(id="Busy"),
            ui.output_ui(id="Significance"),
            ui.input_checkbox_group(id="Shadow", label="Permanently add shadow variables", inline=True, choices = [],
            guide=this, position="top",
            text="Permanently add a boolean shadow variable for any named predictors. These should have the interpretation \"Informative\"."),
            class_ = "vertically-scrollable-footer"
       )

    this.footer = footer

    def settings():
        return ui.TagList(
            ui.input_slider(
                id="CVFolds", label="Cross-validation folds", min=2, max=10, value=5, step=1,
                guide=this, text="Number of held-out folds. For classification this is reduced automatically when the minority class is small.", position="left",
            ),
            ui.input_slider(
                id="MinMissProp", label="Minimum missing proportion", min=0, max=0.5, value=0.05, step=0.01,
                guide=this, text="For a predictor to be considered to have missing values, its missing proportion must exceed this value.", position="left",
            ),
            ui.input_slider(
                id="MinBalancedAccuracy", label="Minimum balanced accuracy", min=0.50, max=0.90, value=0.55, step=0.01,
                guide=this, text="Minimum cross-validated balanced accuracy for a patterned classification.", position="left",
            ),
            ui.input_slider(
                id="MaxObs", label="Maximum observations to analyse", min=3, max=7, value=4, ticks=True, pre="10^",
                guide=this, text="Limit to number of observations to analyse to ensure responsiveness (logarithmic scale).", position="left",
            ),
        )

    this.settings = settings

    def server(input, output, session):
        busy = this.busy()

        @this.suspendable(calc=True)
        def incomingproxy_data():
            req(this._imports.is_set())
            return this._imports.get()

        @this.settle(seconds=2)
        @this.suspendable(calc = True)
        def MaxObs():
            return 10**input.MaxObs()

        @this.settle(seconds=2)
        @this.suspendable(calc = True)
        def Shadow():
            return input.Shadow()

        @this.settle(seconds=2)
        @this.suspendable(calc=True)
        def CVFolds():
            return input.CVFolds()

        @this.settle(seconds=2)
        @this.suspendable(calc=True)
        def MinMissProp():
            return input.MinMissProp()

        @this.settle(seconds=2)
        @this.suspendable(calc=True)
        def MinBalancedAccuracy():
            return input.MinBalancedAccuracy()

        @this.suspendable(calc=True)
        @this.record_code
        def PreparedData():
            samp =  incomingproxy_data().sample(n=MaxObs(), mode="random", keep_geometry=True)
            return samp

        @this.suspendable(calc=True)
        def Target():
            pxd = PreparedData()
            return next(
               iter(pxd.role_map.columns_with_role(Role.TARGET)),
                None,
            )

        @this.suspendable(calc=True)
        def MissingVariables():
            minimum_missing_proportion = float(MinMissProp())
            proxy = PreparedData()
            frame = proxy.frame
            predictors = proxy.role_map.columns_with_role(Role.PREDICTOR)
            return [
                column
                for column in frame.columns
                if (
                    column in predictors
                    and frame[column].isna().mean() > minimum_missing_proportion
                )
            ]

        @this.suspendable(calc=True)
        def PredictorVariables():
            proxy = PreparedData()
            frame = proxy.frame
            predictors = proxy.role_map.columns_with_role(Role.PREDICTOR)
            return [column for column in frame.columns if column in predictors]

        def _weighting_column(proxy: proxy_data) -> str | None:
            columns = sorted(proxy.role_map.columns_with_role(Role.WEIGHTING))
            return columns[0] if columns else None

        @reactive.effect
        def UpdateShadowChoices():
            choices = MissingVariables()
            with reactive.isolate():
                selected = [
                    column for column in (input.Shadow() or [])
                    if column in choices
                ]
            ui.update_checkbox_group(id="Shadow", choices=choices, selected=selected)
        @busy.track("Analysing informative missingness…")
        @reactive.extended_task
        async def CalculateAnalysis(
            frame,
            target,
            predictors,
            missing_variables,
            sample_weight,
            cv_folds,
            minimum_balanced_accuracy,
        ):
            return await asyncio.to_thread(
                _fit_forest_importance,
                frame,
                target=target,
                predictors=predictors,
                missing_variables=missing_variables,
                sample_weight=sample_weight,
                cv_folds=cv_folds,
                minimum_balanced_accuracy=minimum_balanced_accuracy,
            )

        @output
        @render.ui
        def Busy():
            return busy.ui()

        @this.suspendable()
        def StartAnalysis():
            proxy = PreparedData()
            frame = proxy.frame.copy()
            weighting = _weighting_column(proxy)
            sample_weight = frame[weighting].copy() if weighting else None
            CalculateAnalysis.invoke(
                frame,
                Target(),
                PredictorVariables(),
                MissingVariables(),
                sample_weight,
                int(CVFolds()),
                float(MinBalancedAccuracy()),
            )

        @this.suspendable(calc=True)
        @this.record_code
        def Analysis():
            return CalculateAnalysis.result()

        @this.suspendable(calc=True)
        @this.record_code
        def TransformedData():
            selected = Shadow() or []
            if selected:
                this.log.info(f"Adding shadow to predictors: {selected}")
            return _add_shadow_variables(incomingproxy_data(), selected)

        @this.suspendable(triggers=[TransformedData])
        def export():
            this._exports.set(TransformedData())

        @output
        @render_widget
        def Importance():
            figure = _importance_figure(Analysis())
            figure.update_layout(
                modebar={"orientation": "v"},
                modebar_remove=[
                    "select2d", "lasso2d", "toggleHover", "toggleSpikelines",
                    "hoverClosestCartesian", "hoverCompareCartesian",
                ],
            )
            widget = go.FigureWidget(figure)
            widget._config = getattr(widget, "_config", {}) | {
                "displayModeBar": bool(this.isFullScreen()),
                "displaylogo": False,
                "responsive": True,
            }
            return widget

        @output
        @render.ui
        def Table():
            return ui.output_data_frame(id="Table2")

        @output
        @render.data_frame
        def Table2():
            table = Analysis().importance.copy()
            table = table.loc[
                table["Variable Type"].eq("Shadow")
            ].reset_index(drop=True)
            numeric = [
                "Missing Proportion",
                "Importance",
                "Importance SD",
                "Positive Fraction",
            ]
            table[numeric] = table[numeric].round(3)
            styles = [
                {
                    "rows": [index],
                    "class": "miss-informative-row",
                }
                for index, row in table.iterrows()
                if row["Interpretation"] == "Informative"
            ]
            table = table.drop(columns=["Source Variable", "Variable Type"])
            return render.DataTable(
                table,
                width="100%",
                height="98%",
                styles=styles,
            )

        @output
        @render.ui
        def Significance():
            analysis = Analysis()
            if analysis.message:  
                return None
            #     return ui.span(analysis.message, class_="text-secondary") # this is displayed in in the chart area
            shadow = analysis.importance[
                analysis.importance["Variable Type"].eq("Shadow")
            ]
            informative = shadow.loc[
                shadow["Interpretation"].eq("Informative"), "Source Variable"
            ].tolist()
            # performance = (
            #     f"{analysis.score_name}: {analysis.score:.3f} ± "
            #     f"{analysis.score_sd:.3f} ({analysis.folds} folds). "
            # )
            if informative:
                variables = ", ".join(map(str, informative))
                return ui.span(
                    f"Potentially informative missingness: {variables}.",
                    class_="text-warning-emphasis",
                )
            return ui.span(
                "No missingness indicator has stable positive importance.",
                class_="text-success",
            )


    this.server = server
    return this


if Module.running_directly(name=__name__):
    this = instance()
    df = pd.read_csv(Card.ROOT / "data" / "Ass2.csv")
    pxd = proxy_data(_df=df, _name="Ass2")
    pxd.role_map.set_roles("DEATH_RATE", [Role.TARGET])
    this._imports.set(pxd)
    this.run()
