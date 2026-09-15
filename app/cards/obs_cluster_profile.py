"""Explain fixed cluster labels with a cross-validated, interpretable surrogate tree."""
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
import re
from dataclasses import dataclass, field
from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shinywidgets
from card import Card
from module import Module
from plotly.colors import qualitative
from proxy_data import proxy_data
from roles import Role
from shiny import reactive, render, req, ui
from shinywidgets import render_widget
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from cards.miss_informative import _feature_frame

MEMBERSHIP = re.compile(r"^cluster_(?:partition|mixture|density)(?:_\d+)?$")


def _targets(data):
    return [c for c in data.columns if MEMBERSHIP.fullmatch(str(c))
            and Role.STRATIFIER in data.role_map.roles_for(c)]


@dataclass
class Profile:
    target: str | None
    message: str = ""
    model: object = None
    features: list = field(default_factory=list)
    predictors: list = field(default_factory=list)
    observations: int = 0
    eligible: int = 0
    folds: int = 0
    weighted: bool = False
    accuracy: float = np.nan
    balanced: float = np.nan
    baseline: float = np.nan
    baseline_balanced: float = np.nan
    fold_sd: float = np.nan
    rules: pd.DataFrame = field(default_factory=pd.DataFrame)
    confusion: pd.DataFrame = field(default_factory=pd.DataFrame)
    recall: pd.DataFrame = field(default_factory=pd.DataFrame)


def _pipeline(numeric, categorical, depth, leaf):
    transforms = []
    if numeric:
        transforms.append(('numeric', SimpleImputer(strategy='median', keep_empty_features=True), numeric))
    if categorical:
        transforms.append(('categorical', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
            ('encoder', OneHotEncoder(handle_unknown='ignore')),
        ]), categorical))
    return Pipeline([
        ('prepare', ColumnTransformer(transforms, verbose_feature_names_out=False)),
        ('tree', DecisionTreeClassifier(max_depth=depth, min_samples_leaf=leaf, random_state=2025)),
    ])


def _analyze(data, target=None, *, depth=3, leaf=.02, folds=5, limit=5000,
             use_weights=True, include_unallocated=True):
    result = Profile(target)
    try:
        if target not in _targets(data):
            raise ValueError('Assign a cluster_partition, cluster_mixture or cluster_density column (optional numeric suffix) the Stratifier role, then select it.')
        frame = data.frame.reset_index(drop=True)
        weights = np.ones(len(frame))
        weighting = data.role_map.columns_with_role(Role.WEIGHTING)
        if use_weights and weighting:
            if len(weighting) != 1:
                raise ValueError('Assign exactly one observation weighting column.')
            series = frame[next(iter(weighting))]
            if not pd.api.types.is_numeric_dtype(series.dtype) or pd.api.types.is_bool_dtype(series.dtype) or pd.api.types.is_complex_dtype(series.dtype):
                raise ValueError('Observation importance must be numeric.')
            weights = series.to_numpy(dtype=float, na_value=np.nan)
            if not np.isfinite(weights).all() or (weights < 0).any() or not (weights > 0).any():
                raise ValueError('Observation importance must be finite, nonnegative, and include positive values.')
            result.weighted = True
        observed = frame[target].notna().to_numpy() & (weights > 0)
        labels = frame[target].astype('string')
        if not include_unallocated:
            observed &= labels.ne('unallocated').fillna(False).to_numpy(dtype=bool)
        positions = np.flatnonzero(observed)
        result.eligible = len(positions)
        if len(positions) < 4:
            raise ValueError('At least four labeled, positive-importance observations are required.')
        if len(positions) > limit:
            positions, _ = train_test_split(positions, train_size=int(limit), stratify=labels.iloc[positions], random_state=2025)
            positions.sort()
        y = labels.iloc[positions].astype(str).to_numpy()
        counts = pd.Series(y).value_counts()
        if len(counts) < 2 or counts.min() < 2:
            raise ValueError('At least two clusters with two analyzed observations in every cluster are required; increase the sample cap or review rare labels.')
        weights = weights[positions]
        weights = weights / weights.mean()
        columns = [c for c in frame if Role.PREDICTOR in data.role_map.roles_for(c)
                   and c != target and c not in weighting and not MEMBERSHIP.fullmatch(str(c))
                   and not str(c).startswith(Card.SHADOW_PREFIX)
                   and not pd.api.types.is_complex_dtype(frame[c].dtype)
                   and getattr(frame[c].dtype, 'name', None) != 'geometry'
                   and not frame[c].map(lambda v: isinstance(v, (list, dict, set, tuple, np.ndarray))).any()]
        x, numeric, categorical = _feature_frame(frame.iloc[positions].reset_index(drop=True), columns)
        # No imputation or category vocabulary is learned outside a training fold.
        if x.shape[1] == 0:
            raise ValueError('No eligible Predictor-role variables are available for profiling.')
        result.predictors = list(x.columns)
        result.observations = len(x)
        result.folds = min(max(2, int(folds)), int(counts.min()))
        predictions = np.empty(len(y), dtype=object)
        null_predictions = np.empty(len(y), dtype=object)
        scores = []
        for train, test in StratifiedKFold(result.folds, shuffle=True, random_state=2025).split(x, y):
            model = _pipeline(numeric, categorical, int(depth), float(leaf))
            model.fit(x.iloc[train], y[train], tree__sample_weight=weights[train])
            predictions[test] = model.predict(x.iloc[test])
            null = DummyClassifier(strategy='most_frequent').fit(x.iloc[train], y[train], sample_weight=weights[train])
            null_predictions[test] = null.predict(x.iloc[test])
            scores.append(balanced_accuracy_score(y[test], predictions[test], sample_weight=weights[test]))
        result.accuracy = accuracy_score(y, predictions, sample_weight=weights)
        result.balanced = balanced_accuracy_score(y, predictions, sample_weight=weights)
        result.baseline = accuracy_score(y, null_predictions, sample_weight=weights)
        result.baseline_balanced = balanced_accuracy_score(y, null_predictions, sample_weight=weights)
        result.fold_sd = float(np.std(scores, ddof=1))
        fitted = _pipeline(numeric, categorical, int(depth), float(leaf)).fit(x, y, tree__sample_weight=weights)
        result.model = fitted.named_steps['tree']
        result.features = list(fitted.named_steps['prepare'].get_feature_names_out())
        classes = list(result.model.classes_)
        matrix = confusion_matrix(y, predictions, labels=classes, sample_weight=weights)
        result.confusion = pd.DataFrame(matrix, columns=[f'Predicted {c}' for c in classes])
        result.confusion.insert(0, 'Actual', classes)
        result.recall = pd.DataFrame({'Cluster': classes, 'Rows': [int((y == c).sum()) for c in classes],
            'Importance mass': matrix.sum(axis=1), 'CV Recall': matrix.diagonal() / matrix.sum(axis=1)})
        result.rules = _rules(result)
    except (ValueError, TypeError) as error:
        result.message = str(error)
    return result


def _rules(result):
    tree = result.model.tree_
    rows = []
    def visit(node, path):
        left, right = tree.children_left[node], tree.children_right[node]
        if left == right:
            values = tree.value[node][0]
            best = int(values.argmax())
            rows.append({'Leaf': node, 'Rule': ' AND '.join(path) or 'All observations',
                'Predicted cluster': str(result.model.classes_[best]), 'Rows': int(tree.n_node_samples[node]),
                'Importance mass': float(tree.weighted_n_node_samples[node]),
                'Training purity': float(values[best] / values.sum())})
        else:
            feature = result.features[tree.feature[node]]
            threshold = tree.threshold[node]
            visit(left, path + [f'{feature} ≤ {threshold:.6g}'])
            visit(right, path + [f'{feature} > {threshold:.6g}'])
    visit(0, [])
    return pd.DataFrame(rows)


def _figure(result):
    if result.message or result.model is None:
        return Card.empty_figure(result.message or 'No fitted tree')
    tree = result.model.tree_
    positions = {}
    leaves = 0
    def visit(node, depth):
        nonlocal leaves
        left, right = tree.children_left[node], tree.children_right[node]
        if left == right:
            x = leaves
            leaves += 1
        else:
            x = (visit(left, depth + 1) + visit(right, depth + 1)) / 2
        positions[node] = (x, -depth)
        return x
    visit(0, 0)
    fig = go.Figure()
    xs, ys = [], []
    for node, (x, y) in positions.items():
        left, right = tree.children_left[node], tree.children_right[node]
        for child in (left, right) if left != right else ():
            cx, cy = positions[child]
            xs.extend([x, cx, None]); ys.extend([y, cy, None])
        values = tree.value[node][0]
        best = int(values.argmax())
        label = escape(str(result.model.classes_[best]))
        purity = values[best] / values.sum()
        split = '' if left == right else f'{escape(result.features[tree.feature[node]])} ≤ {tree.threshold[node]:.4g}<br>'
        fig.add_annotation(
            x=x, 
            y=y, 
            text=f'{split}{label} ({purity:.0%})<br>n={tree.n_node_samples[node]}',
            showarrow=False, 
            bgcolor=qualitative.Pastel[best % len(qualitative.Pastel)],
            bordercolor='#64748b', 
            borderpad=5, 
            font={'size':11},
            hovertext=f'Training purity: {purity:.1%}<br>Importance mass: {tree.weighted_n_node_samples[node]:.2f}', 
            captureevents=True
        )
    fig.add_scatter(x=xs, y=ys, mode='lines', line={'color':'#64748b'}, hoverinfo='skip')
    fig.update_layout(
        template='plotly_white', 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='#e5ecf6',
        showlegend=False, 
        modebar={"orientation": "v"},
        margin={'l':25,'r':25,'t':30,'b':25},
        xaxis={'visible':False,'range':[-1, max(leaves, 1)]},
        yaxis={'visible':False,'range':[-result.model.get_depth()-.6,.6]})
    return fig


def instance():
    this = Card(file=__file__)
    this.defer_configuration_input('Membership')
    this.long_name = 'Cluster profiling'
    this.description = 'Explain cluster memberships using an interpretable tree and held-out fidelity scores.'
    this.front = lambda: ui.navset_bar(
        ui.nav_panel('No memberships', shinywidgets.output_widget('EmptyTree', fill=True), value='__empty__'),
        id='Membership', title=None, padding=0, fillable=True)
        
    this.back = lambda: ui.navset_tab(
        ui.nav_panel('Model accuracy', ui.output_data_frame('Comparison')),
        ui.nav_panel('Rules', ui.output_text('RulesTarget'), ui.output_data_frame('Rules')),
        ui.nav_panel('Cluster accuracy', ui.output_text('RecallTarget'), ui.output_data_frame('Recall')),
        ui.nav_panel('Confusion', ui.output_text('ConfusionTarget'), ui.output_data_frame('Confusion')), id='Details')
    this.footer = lambda: ui.TagList(ui.output_ui('Busy'), ui.output_text('Accuracy'))
    def settings():
        def slider(id, label, low, high, value, step, text):
            return ui.input_slider(id, label=label, min=low, max=high, value=value, step=step, guide=this, position='left', text=text)
        return ui.TagList(
            slider('Depth', 'Maximum tree depth', 1, 6, 3, 1, 'Limits explanation complexity. Deeper trees may reproduce labels more closely but are harder to read and can overfit.'),
            slider('Leaf', 'Minimum leaf fraction', .01, .25, .02, .01, 'Minimum fraction of training rows in each leaf. This uses row counts, not importance mass.'),
            slider('Folds', 'Cross-validation folds', 2, 10, 5, 1, 'Stratified folds, capped by the smallest cluster count. Accuracy uses out-of-fold predictions; balanced accuracy averages recall across clusters. Fold SD describes variability, not a confidence interval. Existing cluster labels are fixed, so this does not validate the upstream clustering pipeline.'),
            ui.input_checkbox('UseWeights', label='Use assigned observation weighting', value=True, guide=this, position='left', text='Use finite nonnegative importance weights for tree fitting and held-out metrics. Zero-weight rows are excluded; positive weights are normalized to mean one. Imputation and sampling use row counts. Disable to give all rows equal importance.'),
            ui.input_checkbox('Unallocated', label='Include unallocated', value=True, guide=this, position='left', text='Treat DBSCAN unallocated as a class to explain. Disable to profile allocated clusters only. Missing labels are always omitted. At least two observed classes are needed.'),
            slider('Limit', 'Maximum observations to analyze', 100, 10000, 5000, 100, 'Reproducible stratified sampling above this cap. Rare clusters need at least two sampled rows. Raising it improves coverage but can substantially increase fitting time.')
        )

    this.settings = settings
    def server(input, output, session):
        busy = this.busy()
        restored = this.restored_configuration_input('Membership', None)
        restore_pending = True
        current_tabs = ('__empty__',)
        registered = set()

        @this.suspendable(calc=True)
        def Targets():
            return tuple(_targets(this.input_data()))

        @this.suspendable(calc=True)
        def SelectedTarget():
            targets = Targets()
            selected = input.Membership()
            return selected if selected in targets else (targets[0] if targets else None)

        def register_tree(target):
            output_id = 'Tree_' + target
            @output(id=output_id)
            @render_widget
            def tree():
                results = Profiles()
                req(target in results, cancel_output=True)
                widget = go.FigureWidget(_figure(results[target]))
                widget._config = {'displayModeBar':bool(this.isFullScreen()), 'displaylogo':False}
                return widget

        @this.suspendable()
        def UpdateTabs():
            nonlocal current_tabs, restore_pending
            targets = Targets()
            desired = targets or ('__empty__',)
            with reactive.isolate():
                selected = input.Membership()
            selected = restored if restore_pending and restored in targets else selected
            if selected not in desired:
                selected = desired[0]
            # Keep a panel present while replacing the empty state or last tab.
            for target in desired:
                if target in current_tabs:
                    continue
                if target == '__empty__':
                    panel = ui.nav_panel('No memberships', shinywidgets.output_widget('EmptyTree', fill=True), value=target)
                else:
                    if target not in registered:
                        register_tree(target)
                        registered.add(target)
                    panel = ui.nav_panel(target.removeprefix("cluster_").title(),
                        shinywidgets.output_widget(
                            id='Tree_' + target, fill=True, 
                            guide=this, title=f'{target} decision tree', position='left',
                            text='Each tab explains one Stratifier-role cluster membership using Predictor-role variables. All membership columns are excluded from predictors. Left branches satisfy ≤; right branches mean >. Nodes show predicted cluster, training purity, and row count. One-hot category splits ≤0.5 mean absent. The footer reports held-out accuracy for this tab; the flip side compares all trees. These associations explain existing labels, not causes.'
                        ),
                        value=target
                    )
                ui.insert_nav_panel('Membership', panel, select=(target == selected))
            ui.update_navset('Membership', selected=selected)
            for target in current_tabs:
                if target not in desired:
                    ui.remove_nav_panel('Membership', target=target)
            current_tabs = desired
            if targets:
                restore_pending = False

        @this.suspendable(calc=True)
        def Options():
            return {"depth": input.Depth(), "leaf": input.Leaf(), "folds": input.Folds(), "limit": input.Limit(),
                "use_weights": input.UseWeights(), "include_unallocated": input.Unallocated()}

        @busy.track('Profiling cluster memberships…')
        @this.extended_task
        async def Calculate(data, options):
            def calculate_all():
                return {target: _analyze(data, target=target, **options) for target in _targets(data)}
            results = calculate_all() if Module.IS_SHINYLIVE else await asyncio.to_thread(calculate_all)
            return data, options, results

        @this.suspendable()
        def Start():
            Calculate.cancel()
            Calculate.invoke(this.input_data().clone(), Options())

        @this.suspendable(calc=True)
        def Profiles():
            data, options, results = Calculate.result()
            req(data.equals(this.input_data()) and options == Options(), cancel_output=True)
            return results

        @this.suspendable(calc=True)
        def Analysis():
            results = Profiles()
            return results.get(SelectedTarget(), Profile(None, message='Assign a cluster-named column the Stratifier role.'))

        @output
        @render_widget
        def EmptyTree():
            widget = go.FigureWidget(Card.empty_figure('No eligible cluster membership columns.'))
            widget._config = {'displayModeBar':False, 'displaylogo':False}
            return widget

        @output
        @render.ui
        def Busy():
            return busy.ui()

        @output
        @render.data_frame
        def Comparison():
            rows = [{'Membership':target.removeprefix("cluster_").title(), 'CV Accuracy':r.accuracy, 'CV Balanced Accuracy':r.balanced,
                     'Majority baseline':r.baseline, 'Balanced baseline':r.baseline_balanced,
                     'Fold SD':r.fold_sd, 'Folds':r.folds, 'Rows':r.observations,
                     'Leaves':r.model.get_n_leaves() if r.model is not None else 0,
                     'Status':r.message or 'Fidelity to existing labels'} for target,r in Profiles().items()]
            return render.DataTable(pd.DataFrame(rows).round(4), height='auto', width='100%')

        for output_id in ('RulesTarget', 'RecallTarget', 'ConfusionTarget'):
            def register_title(output_id):
                @output(id=output_id)
                @render.text
                def title():
                    return SelectedTarget() or 'No membership selected'
            register_title(output_id)

        @output
        @render.text
        def Accuracy():
            r = Analysis()
            if r.message:
                return r.message
            return (f'{r.target.removeprefix("cluster_").title()}: CV accuracy {r.accuracy:.1%}; balanced accuracy {r.balanced:.1%} '
                f'(fold SD {r.fold_sd:.1%}). Majority baseline {r.baseline:.1%}; balanced {r.baseline_balanced:.1%}. '
                f'{r.folds} folds; {r.observations} of {r.eligible} eligible rows; {len(r.predictors)} predictors. '
                f'{"Observation importance applied. " if r.weighted else "Equal observation importance. "}'
                'Fidelity to existing labels, not cluster validity.')
        @output
        @render.data_frame
        def Rules():
            return render.DataTable(Analysis().rules.round(4), height='auto', width='100%')
        @output
        @render.data_frame
        def Recall():
            return render.DataTable(Analysis().recall.round(4), height='auto', width='100%')
        @output
        @render.data_frame
        def Confusion():
            return render.DataTable(Analysis().confusion.round(4), height='auto', width='100%')
        
        session.on_ended(Calculate.cancel)
        
        return this.input_data
    this.server = server
    return this


if Module.running_directly(name=__name__):
    this = instance()
    df = pd.read_csv(Card.ROOT / "data" / "Ass2.csv")
    this._imports.set(proxy_data(_df=df, _name="Ass2"))
    this.run()
