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

import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shinywidgets
from card import Card
from mlxtend.frequent_patterns import apriori, association_rules
from module import Module
from proxy_data import proxy_data
from shiny import render, req, ui
from shinywidgets import render_widget


def instance():
    """Create the immutable missingness-association-rules card."""
    this = Card(file=__file__, mutable=False)
    this.long_name = "Missingness rules"
    this.description = "This card visualises Association Rules that describe patterns of missingness."

    def front():
        return ui.TagList(
            ui.span(
                "Missingness association network",
                class_="text-primary text-center d-block",
            ),
            shinywidgets.output_widget(
                id="Network",
                fill=True,
                guide=this,
                title="Missingness association network",
                text=(
                    "Variables are square nodes and association rules are "
                    "circular nodes. A path from variables through a rule to "
                    "other variables reads: if the former are missing, the "
                    "latter are also likely to be missing."
                ),
                position="left",
            )
        )

    this.front = front

    def back():
        return ui.TagList(
            ui.span(
                "Missingness association rules",
                class_="text-primary text-center d-block",
            ),
            ui.output_ui(
                id="Table",
                guide=this,
                title="Missingness association rules",
                text=(
                    "Each row reads: if the LHS variables are missing, the RHS "
                    "variables are missing with the reported confidence."
                ),
                position="left",
            ),
        )

    this.back = back

    def footer():
        return ui.div(
            ui.output_ui(id="Check"),
            class_="html-fill-container html-fill-item text-center",
        )

    this.footer = footer

    def settings():
        return ui.TagList(
            ui.input_slider(
                id="MinSupport", label="Minimum permitted rule support", min=0.1, max=0.95, value=0.1, step=0.01,
                guide=this, text="Rules whose support is below this threshold are dropped.", position="left",
            ),
            ui.input_slider(
                id="MinLift", label="Minimum permitted rule lift", min=0.1, max=5, value=2, step=0.1,
                guide=this, text="Rules whose lift is below this threshold are dropped.", position="left",
            ),
            ui.input_slider(
                id="MaxLength", label="Maximum variables per rule", min=2, max=15, value=10, step=1,
                guide=this, text="Limits the total number of LHS and RHS variables in a rule.", position="left",
            ),
            ui.input_checkbox(
                id="RemoveRedundant", label="Prune redundant rules", value=False,
                guide=this, text="Hide specialised rules when a simpler rule with the same conclusion has at least as much confidence.", position="left",
            ),
            ui.input_slider(
                id="MaxObs", label="Maximum observations to analyse", min=3, max=7, value=4, ticks=True, pre="10^",
                guide=this, text = 'Limit to number of observations to analyse to ensure responsiveness (logarithmic scale).', position="left",
            ),
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

        @this.settle(seconds=2)
        @this.suspendable(calc=True)
        def MinSupport():
            return float(input.MinSupport())

        @this.settle(seconds=2)
        @this.suspendable(calc=True)
        def MinLift():
            return float(input.MinLift())

        @this.settle(seconds=2)
        @this.suspendable(calc=True)
        def MaxLength():
            return max(2, int(input.MaxLength()))

        @this.suspendable(calc=True)
        @this.record_code
        def PreparedData():
            samp = incomingproxy_data().sample(n=MaxObs(), mode="random", keep_geometry=True)
            return samp

        @this.suspendable(calc=True)
        @this.record_code
        def MissingVariables():
            frame = PreparedData().to_native()
            return [column for column in frame.columns if frame[column].isna().any()]

        @this.suspendable(calc=True)
        @this.record_code
        def MissingTransactions():
            frame = PreparedData().to_native()
            variables = MissingVariables()
            if not variables:
                return pd.DataFrame(index=frame.index, dtype=bool)
            transactions = frame.loc[:, variables].isna()
            return transactions.loc[transactions.any(axis=1)].astype(bool)

        @this.record_code
        def _remove_redundant(rules: pd.DataFrame) -> pd.DataFrame:
            if rules.empty:
                return rules.copy()
            keep = np.ones(len(rules), dtype=bool)
            antecedents = rules["antecedents"].tolist()
            consequents = rules["consequents"].tolist()
            confidence = rules["confidence"].to_numpy(dtype=float)
            support = rules["support"].to_numpy(dtype=float)
            for candidate in range(len(rules)):
                for simpler in range(len(rules)):
                    if candidate == simpler:
                        continue
                    same_conclusion = consequents[simpler] == consequents[candidate]
                    simpler_condition = (
                        antecedents[simpler] < antecedents[candidate]
                    )
                    no_weaker = (
                        confidence[simpler] >= confidence[candidate]
                        and support[simpler] >= support[candidate]
                    )
                    if same_conclusion and simpler_condition and no_weaker:
                        keep[candidate] = False
                        break
            return rules.loc[keep].reset_index(drop=True)

        @this.suspendable(calc=True)
        @this.record_code
        def Rules():
            transactions = MissingTransactions()
            columns = [
                "antecedents", "consequents", "antecedent support",
                "consequent support", "support", "confidence", "lift",
                "leverage", "conviction",
            ]
            if transactions.empty or transactions.shape[1] < 2:
                return pd.DataFrame(columns=columns)
            frequent = apriori(
                transactions,
                min_support=MinSupport(),
                use_colnames=True,
                max_len=min(MaxLength(), transactions.shape[1]),
                low_memory=transactions.shape[1] > 30,
            )
            if frequent.empty or not frequent["itemsets"].map(lambda x: len(x) > 1).any():
                return pd.DataFrame(columns=columns)
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="invalid value encountered in divide",
                    category=RuntimeWarning,
                    module=r"mlxtend\.frequent_patterns\.association_rules",
                )
                rules = association_rules(
                    frequent,
                    metric="lift",
                    min_threshold=MinLift(),
                    num_itemsets=len(transactions),
                )
            if rules.empty:
                return pd.DataFrame(columns=columns)
            length = rules["antecedents"].map(len) + rules["consequents"].map(len)
            rules = rules.loc[length <= MaxLength()].copy()
            if bool(input.RemoveRedundant()):
                rules = _remove_redundant(rules)
            return rules.sort_values(
                ["lift", "confidence", "support"],
                ascending=[False, False, False],
            ).reset_index(drop=True)

        @this.record_code
        def _itemset_label(items) -> str:
            return ", ".join(sorted(map(str, items), key=str.casefold))

        @this.suspendable(calc=True)
        @this.record_code
        def RulesTable():
            rules = Rules()
            if rules.empty:
                return pd.DataFrame(columns=[
                    "LHS", "RHS", "Support", "Confidence", "Lift",
                    "Leverage", "Conviction", "Count",
                ])
            transactions = MissingTransactions()
            table = pd.DataFrame({
                "LHS": rules["antecedents"].map(_itemset_label),
                "RHS": rules["consequents"].map(_itemset_label),
                "Support": rules["support"],
                "Confidence": rules["confidence"],
                "Lift": rules["lift"],
                "Leverage": rules["leverage"],
                "Conviction": rules["conviction"],
                "Count": np.rint(rules["support"] * len(transactions)).astype(int),
            })
            numeric = [
                "Support", "Confidence", "Lift", "Leverage", "Conviction"
            ]
            table.loc[:, numeric] = table.loc[:, numeric].round(3)
            return table

        @this.record_code
        def _network_figure(rules: pd.DataFrame, *, limit: int = 50) -> go.Figure:
            if rules.empty:
                return Card.empty_figure("No significant rules to display")
            selected = rules.sort_values(
                ["confidence", "support"], ascending=[False, False]
            ).head(limit).reset_index(drop=True)
            variables = sorted(
                set().union(*selected["antecedents"], *selected["consequents"]),
                key=lambda value: str(value).casefold(),
            )
            variable_angles = np.linspace(0, 2 * np.pi, len(variables), endpoint=False)
            variable_position = {
                variable: (np.cos(angle), np.sin(angle))
                for variable, angle in zip(variables, variable_angles)
            }
            rule_angles = np.linspace(0, 2 * np.pi, len(selected), endpoint=False)
            rule_position = {
                index: (0.42 * np.cos(angle), 0.42 * np.sin(angle))
                for index, angle in enumerate(rule_angles)
            }
            edge_x, edge_y = [], []
            for index, rule in selected.iterrows():
                rule_x, rule_y = rule_position[index]
                for variable in rule["antecedents"] | rule["consequents"]:
                    variable_x, variable_y = variable_position[variable]
                    edge_x.extend([variable_x, rule_x, None])
                    edge_y.extend([variable_y, rule_y, None])
            figure = go.Figure()
            figure.add_trace(go.Scatter(
                x=edge_x,
                y=edge_y,
                mode="lines",
                line={"color": "rgba(80,80,80,0.35)", "width": 1.5},
                hoverinfo="skip",
                showlegend=False,
            ))
            figure.add_trace(go.Scatter(
                x=[variable_position[value][0] for value in variables],
                y=[variable_position[value][1] for value in variables],
                text=[str(value) for value in variables],
                mode="markers+text",
                textposition="top center",
                marker={
                    "symbol": "square",
                    "size": 18,
                    "color": "#8dd3c7",
                    "line": {"color": "#2c3e50", "width": 2},
                },
                name="Variables",
                hovertemplate="Variable: %{text}<extra></extra>",
            ))
            hover = [
                (
                    f"<b>Rule {index + 1}</b><br>"
                    f"{_itemset_label(rule['antecedents'])} → "
                    f"{_itemset_label(rule['consequents'])}<br>"
                    f"Support: {rule['support']:.3f}<br>"
                    f"Confidence: {rule['confidence']:.3f}<br>"
                    f"Lift: {rule['lift']:.3f}"
                )
                for index, rule in selected.iterrows()
            ]
            figure.add_trace(go.Scatter(
                x=[rule_position[index][0] for index in range(len(selected))],
                y=[rule_position[index][1] for index in range(len(selected))],
                text=hover,
                mode="markers",
                marker={
                    "symbol": "circle",
                    "size": 12 + 22 * selected["support"].to_numpy(dtype=float),
                    "color": selected["confidence"],
                    "colorscale": "Blues",
                    "cmin": 0,
                    "cmax": 5,
                    "showscale": True,
                    "colorbar": {"title": "Confidence", "thickness": 12},
                    "line": {"color": "#2c3e50", "width": 1},
                },
                name="Rules",
                hovertemplate="%{text}<extra></extra>",
            ))
            figure.update_layout(
                template="plotly_white",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#e5ecf6",
                margin={"l": 15, "r": 15, "t": 15, "b": 15},
                xaxis={"visible": False, "range": [-1.3, 1.3]},
                yaxis={
                    "visible": False,
                    "range": [-1.3, 1.3],
                    "scaleanchor": "x",
                    "scaleratio": 1,
                },
                showlegend=False,
                hovermode="closest",
            )
            return figure

        @output
        @render_widget
        def Network():
            figure = _network_figure(Rules(), limit=50)
            figure.update_layout(
                modebar={"orientation": "v"},
                modebar_remove=[
                    "select2d", "lasso2d", "toggleHover",
                    "toggleSpikelines", "hoverClosestCartesian",
                    "hoverCompareCartesian",
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
            return render.DataTable(RulesTable(), width="100%", height="98%")

        @output
        @render.ui
        def Check():
            missing_count = len(MissingVariables())
            rule_count = len(Rules())
            if missing_count == 0:
                return ui.span(
                    "The data does not contain missing values.",
                    class_="text-success",
                )
            if rule_count == 0:
                return ui.TagList(
                    ui.span(
                        "No significant rules explain the patterns of missingness.",
                        class_="text-info",
                    ),
                    ui.br(),
                    ui.span("Consider changing the parameters.", class_="text-info"),
                )
            variable_word = "variable has" if missing_count == 1 else "variables have"
            rule_word = "rule" if rule_count == 1 else "rules"
            return ui.span(
                f"{missing_count} {variable_word} missing values that generate "
                f"{rule_count} {rule_word}.",
                class_="text-primary",
            )


    this.server = server
    return this


if Module.running_directly(name=__name__):
    this = instance()
    df = pd.read_csv(Card.ROOT / "data" / "Ass2.csv")
    this._imports.set(proxy_data(_df=df, _name="Ass2"))
    this.run()
