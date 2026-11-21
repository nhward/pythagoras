"""Held-out prediction assessment of the active reviewed candidate or ensemble."""
import threading
import uuid
from contextlib import nullcontext
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from code_recording import recordable
from ensemble import ClassificationEnsemble, RegressionEnsemble
from evaluation_plan import evaluation_frame, role_columns
from fit_responses import collect_responses, combine_responses
from fit_service import fit_key
from pipeline_search import prepare
from roles import Role
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)


class AssessmentWaiting(ValueError):
    """No current upstream selection is available."""


def selection(source):
    ensemble = source.ensemble_results
    if ensemble is not None:
        return ensemble, None, None
    terminal = source.pipeline.steps[-1][1] if source.has_pipeline else None
    if isinstance(terminal, (ClassificationEnsemble, RegressionEnsemble)):
        raise AssessmentWaiting('Waiting for the current committed ensemble evaluation.')
    results = source.tuning_results
    if not results or results.get('read_only'):
        raise AssessmentWaiting('Waiting for an upstream weighted best model or committed ensemble.')
    winner = results.get('review', {}).get('winner_trial')
    trial = next((t for t in results['state']['trials'] if t['id'] == winner and t['state'] == 'complete'), None)
    if trial is None or str(winner) not in results['pipelines']:
        raise AssessmentWaiting('Waiting for an upstream weighted best model.')
    return None, results, trial


def assessment_signature(source):
    ensemble, results, trial = selection(source)
    groups = evaluation_frame(source, grouping_columns(source))[grouping_columns(source)]
    if ensemble is not None:
        return joblib.hash(('ensemble', ensemble.input_signature, groups))
    return joblib.hash((source.clean_frame, source.role_map.to_primitive(), source.modelling_context,
                        source.evaluation_plan, results['state']['specification']['seed'], trial['id'],
                        results['pipelines'][str(trial['id'])], groups))


def grouping_columns(source):
    eligible = set(role_columns(source, Role.STRATIFIER)) | set(role_columns(source, Role.SENSITIVE))
    return [c for c in dict.fromkeys((*source.clean_frame.columns, *source.frame.columns)) if c in eligible]


@dataclass
class Assessment:
    label: str
    task: str
    events: pd.DataFrame
    classes: np.ndarray
    scores: object
    score_kind: str
    groups: pd.DataFrame
    splits: int
    new_fits: int
    reused_fits: int
    notes: tuple


def _aligned_scores(response, classes):
    """Return scores in the common class order; absent classes stay unavailable."""
    for name, kind in (('predict_proba', 'Probability'), ('decision_function', 'Decision score')):
        if name not in response.values:
            continue
        values = np.asarray(response.values[name], dtype=float)
        if values.ndim == 1 and len(response.classes) == 2:
            values = np.column_stack((-values, values))
        if values.ndim != 2 or values.shape[1] != len(response.classes):
            continue
        aligned = np.full((len(values), len(classes)), np.nan)
        for i, label in enumerate(response.classes):
            aligned[:, np.flatnonzero(classes == label)[0]] = values[:, i]
        return aligned, kind
    return None, ''


@recordable
def assess_candidate(source, *, fit_service=None, cancelled=None):
    """Reuse split fits/responses, fitting missing individual-model artifacts only."""
    cancelled = cancelled or threading.Event()
    ensemble, results, trial = selection(source)
    runtime = getattr(fit_service, 'runtime', None)
    memory = getattr(fit_service, 'entries', None)
    owner = 'assessment:' + uuid.uuid4().hex
    frame = source.clean_frame
    target, = source.role_map.columns_with_role(Role.TARGET)
    if ensemble is not None:
        splits, task = ensemble.splits, ensemble.task
        label = f'Ensemble ({ensemble.definition.style}; {len(ensemble.members)} members)'
        problem = None
    else:
        selected = source.with_selected_model(trial['id'])
        problem = prepare(selected, seed=results['state']['specification']['seed'])
        splits, task = problem.splits, problem.specification['task']
        label = f"{trial.get('method', trial['model_id'])} · trial {trial['id'] + 1}"
    used = sorted({int(row) for a, b in splits for row in (*a, *b)})
    classes = np.unique(frame[target].iloc[used]) if task == 'classification' else np.array([])
    events, scores, kinds, notes = [], [], [], []
    active_keys = set()
    new_fits = reused_fits = 0
    try:
        for fold, (train, validation) in enumerate(splits):
            if cancelled.is_set():
                raise InterruptedError('Assessment cancelled.')
            validation = np.asarray(validation, dtype=int)
            batch = frame.iloc[validation]
            if ensemble is not None:
                model = ensemble.fitted[fold]
                responses = []
                for member, (_, fitted) in enumerate(model.estimators_):
                    key_index = fold * len(model.estimators_) + member
                    key = ensemble.cache_keys[key_index] if key_index < len(ensemble.cache_keys) else None
                    if key is not None:
                        active_keys.add(key)
                    cached = runtime.get(key, owner) if runtime and key else memory.get(key) if memory is not None and key else None
                    responses.append(cached[1] if cached is not None else collect_responses(fitted, batch, task))
                response = combine_responses(responses, model.style, task, getattr(model, 'classes_', None))
                reused_fits += len(responses)
                if task == 'classification' and model.style == 'hard':
                    votes = np.asarray([r.values['predict'] for r in responses])
                    score = np.column_stack([(votes == c).mean(axis=0) for c in classes])
                    kind = 'Vote share'
                else:
                    score, kind = _aligned_scores(response, classes) if task == 'classification' else (None, '')
            else:
                pipeline = problem.pipeline
                key = fit_key(problem.fit_context, pipeline, train, validation)
                active_keys.add(key)
                cached = runtime.get(key, owner) if runtime else memory.get(key) if memory is not None else None
                if cached is not None:
                    model, response = cached
                    reused_fits += 1
                else:
                    with runtime.lease('assessment', problem.specification['execution'].get('estimated_fit_bytes', 0), cancelled) if runtime else nullcontext():
                        if cancelled.is_set():
                            raise InterruptedError('Assessment cancelled.')
                        model = clone(pipeline).fit(frame.iloc[train].copy(), frame[target].iloc[train].copy())
                        response = collect_responses(model, batch, task)
                        new_fits += 1
                        if runtime:
                            error = runtime.put(key, model, response, owner)
                            if error:
                                notes.append(error)
                            runtime.remember(key, model, response)
                        elif memory is not None:
                            memory[key] = model, response
                score, kind = _aligned_scores(response, classes) if task == 'classification' else (None, '')
                # Pairwise OVO decision columns are not per-class ranking scores.
                if kind == 'Decision score' and any(k.endswith('decision_function_shape') and v == 'ovo'
                        for k, v in pipeline.get_params(deep=True).items()):
                    score, kind = None, ''
            if 'predict' not in response.values:
                raise ValueError(response.errors.get('predict', 'Prediction labels unavailable.'))
            predicted = np.asarray(response.values['predict'])
            actual = batch[target].to_numpy()
            if predicted.shape != actual.shape:
                raise ValueError('Predictions do not preserve validation-row alignment.')
            if task == 'regression' and not (np.isfinite(actual.astype(float)).all() and np.isfinite(predicted.astype(float)).all()):
                raise ValueError('Assessment requires finite observed and predicted regression values.')
            events.append(pd.DataFrame({'row': validation, 'split': fold, 'actual': actual, 'predicted': predicted}))
            scores.append(score)
            kinds.append(kind)
        score_kind = kinds[0] if kinds and len(set(kinds)) == 1 else ''
        aligned = np.concatenate(scores) if score_kind and all(s is not None for s in scores) else None
        if task == 'classification' and aligned is None:
            notes.append('Ranking scores unavailable; showing a confusion matrix.')
        groups = evaluation_frame(source, grouping_columns(source))[grouping_columns(source)].copy()
        if cancelled.is_set():
            raise InterruptedError('Assessment cancelled.')
        if memory is not None:
            for key in list(memory):
                if key not in active_keys:
                    del memory[key]
        return Assessment(label, task, pd.concat(events, ignore_index=True), classes, aligned, score_kind,
                          groups, len(splits), new_fits, reused_fits, tuple(dict.fromkeys(notes)))
    finally:
        if runtime:
            runtime.release(owner)


def group_values(result, column):
    """At most 12 categorical levels or five disclosed quantile bins, plus missing."""
    if not column or column not in result.groups:
        return pd.Series('All', index=result.events.index)
    values = result.groups[column].reset_index(drop=True)
    if pd.api.types.is_numeric_dtype(values.dtype):
        values = values.replace([np.inf, -np.inf], np.nan)
    if pd.api.types.is_numeric_dtype(values.dtype) and not pd.api.types.is_bool_dtype(values.dtype) and values.nunique() > 5:
        values = pd.qcut(values, q=5, duplicates='drop')
        labels = values.astype('string')
    else:
        labels = values.astype('string')
        levels = labels.value_counts().index[:12]
        labels = labels.where(labels.isna() | labels.isin(levels), '<Other>')
    labels = labels.fillna('<Missing>')
    return labels.iloc[result.events.row.to_numpy()].reset_index(drop=True)


def group_masks(result, colour='', symbol=''):
    yield 'Overall', np.ones(len(result.events), dtype=bool)
    columns = list(dict.fromkeys(c for c in (colour, symbol) if c and c in result.groups))
    if not columns:
        return
    labels = [group_values(result, c) for c in columns]
    combinations = pd.DataFrame({c: values for c, values in zip(columns, labels)})
    for values in combinations.drop_duplicates().itertuples(index=False, name=None):
        mask = np.ones(len(result.events), dtype=bool)
        for c, value in zip(columns, values):
            mask &= combinations[c].to_numpy() == value
        yield ' · '.join(f'{c}={v}' for c, v in zip(columns, values)), mask


def roc_splits(result, mask, class_index):
    """Use split-specific ranking: score scales need not agree between fits."""
    curves, aucs = [], []
    if result.scores is None:
        return curves, aucs
    observed = result.events.actual.to_numpy() == result.classes[class_index]
    values = result.scores[:, class_index]
    for fold in sorted(result.events.split.unique()):
        subset = mask & (result.events.split.to_numpy() == fold)
        if not subset.any() or not np.isfinite(values[subset]).all() or len(np.unique(observed[subset])) < 2:
            continue
        fpr, tpr, _ = roc_curve(observed[subset], values[subset])
        curves.append((fpr, tpr))
        aucs.append(roc_auc_score(observed[subset], values[subset]))
    return curves, aucs


@recordable
def assessment_table(result, colour='', symbol='', class_index=0):
    rows = []
    for label, mask in group_masks(result, colour, symbol):
        events = result.events.loc[mask]
        row = {'Group': label, 'Observations': events.row.nunique(), 'Predictions': len(events), 'Splits': events.split.nunique()}
        actual, predicted = events.actual.to_numpy(), events.predicted.to_numpy()
        if result.task == 'regression':
            error = predicted.astype(float) - actual.astype(float)
            lower, upper = np.quantile(error, [.025, .975])
            row.update({'RMSE': float(np.sqrt(np.mean(error ** 2))), 'MAE': mean_absolute_error(actual, predicted),
                        'R²': r2_score(actual, predicted) if len(events) > 1 and np.var(actual.astype(float)) > 0 else np.nan,
                        'Mean error (bias)': error.mean(), 'Error variance': error.var(),
                        'Error P2.5': lower, 'Error P97.5': upper})
        else:
            curves, aucs = roc_splits(result, mask, class_index)
            row.update({'Class versus rest': str(result.classes[class_index]), 'Accuracy': accuracy_score(actual, predicted),
                        'Balanced accuracy': np.mean([np.mean(predicted[actual == c] == c) for c in np.unique(actual)]),
                        'AUC': np.mean(aucs) if aucs else np.nan,
                        'AUC split SD': np.std(aucs, ddof=1) if len(aucs) > 1 else np.nan,
                        'ROC splits': len(curves), 'Score': result.score_kind or 'Labels only'})
            if result.score_kind == 'Probability':
                score = result.scores[mask, class_index]
                error = score - (actual == result.classes[class_index])
                finite = np.isfinite(error)
                row.update({'Probability errors': int(finite.sum()),
                            'Mean probability error': error[finite].mean() if finite.any() else np.nan,
                            'Probability error variance': error[finite].var() if finite.any() else np.nan,
                            'Brier (selected class)': np.mean(error[finite] ** 2) if finite.any() else np.nan})
            if len(curves) < events.split.nunique():
                row['Note'] = 'Some splits lack both classes or finite ranking scores.'
        rows.append(row)
    return pd.DataFrame(rows)


@recordable
def assessment_figure(result, colour='', symbol='', class_index=0, chart='auto', subgroup='Overall', full_screen=False):
    import plotly.graph_objects as go
    from plotly.colors import qualitative
    from sklearn.metrics import confusion_matrix

    palette = qualitative.Safe
    markers = ('circle', 'square', 'diamond', 'cross', 'x', 'triangle-up', 'triangle-down')
    figure = go.Figure()
    if result.task == 'regression':
        colour_values, symbol_values = group_values(result, colour), group_values(result, symbol)
        colours = {v: palette[i % len(palette)] for i, v in enumerate(colour_values.unique())}
        symbols = {v: markers[i % len(markers)] for i, v in enumerate(symbol_values.unique())}
        # Deterministic display sampling only; references and statistics use all events.
        displayed = np.linspace(0, len(result.events) - 1, min(5000, len(result.events)), dtype=int)
        for name, mask in group_masks(result, colour, symbol):
            if name == 'Overall' and (colour or symbol):
                continue
            positions = displayed[mask[displayed]]
            if not len(positions):
                continue
            points = result.events.iloc[positions]
            first = positions[0]
            figure.add_trace(go.Scattergl(x=points.actual, y=points.predicted, mode='markers', name=name,
                marker={'color': colours[colour_values.iloc[first]], 'symbol': symbols[symbol_values.iloc[first]], 'size': 6, 'opacity': .6},
                customdata=np.column_stack((points.row + 1, points.split + 1)),
                hovertemplate='Actual %{x:.5g}<br>Predicted %{y:.5g}<br>Row %{customdata[0]} · split %{customdata[1]}<extra>%{fullData.name}</extra>'))
        actual = result.events.actual.to_numpy(dtype=float)
        predicted = result.events.predicted.to_numpy(dtype=float)
        low, high = np.quantile(predicted - actual, [.025, .975])
        minimum, maximum = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
        padding = max((maximum - minimum) * .05, 1e-9)
        bounds = [minimum - padding, maximum + padding]
        for name, offset, dash in (('1:1', 0., 'solid'), ('Error P2.5', low, 'dash'), ('Error P97.5', high, 'dash')):
            figure.add_trace(go.Scatter(x=bounds, y=np.asarray(bounds) + offset, mode='lines', name=name,
                line={'color': '#333333' if offset == 0 else '#b85c00', 'dash': dash}, hovertemplate=name + '<extra></extra>'))
        figure.update_xaxes(title='Actual', range=bounds, constrain='domain')
        figure.update_yaxes(title='Predicted', range=bounds, scaleanchor='x', scaleratio=1, constrain='domain')
        if len(result.events) > 5000:
            figure.add_annotation(text='5,000 displayed predictions; statistics use all predictions', x=.5, y=1.08, xref='paper', yref='paper', showarrow=False)
    elif chart == 'confusion' or result.scores is None:
        mask = dict(group_masks(result, colour, symbol)).get(subgroup, np.ones(len(result.events), dtype=bool))
        events = result.events.loc[mask]
        counts = confusion_matrix(events.actual, events.predicted, labels=result.classes)
        totals = counts.sum(axis=1, keepdims=True)
        percentages = np.divide(counts * 100., totals, out=np.zeros(counts.shape, dtype=float), where=totals != 0)
        labels = [str(c) for c in result.classes]
        figure.add_trace(go.Heatmap(z=percentages, x=labels, y=labels, customdata=counts,
            text=np.where(totals != 0, np.char.add(np.round(percentages, 1).astype(str), '%'), '—'), texttemplate='%{text}',
            zmin=0, zmax=100, colorscale='Blues', colorbar={'title': '% of actual'},
            hovertemplate='Actual %{y}<br>Predicted %{x}<br>%{z:.1f}% · %{customdata} predictions<extra></extra>'))
        figure.update_xaxes(title='Predicted class', type='category', constrain='domain')
        figure.update_yaxes(title='Actual class', type='category', autorange='reversed', scaleanchor='x', scaleratio=1, constrain='domain')
    else:
        grid = np.linspace(0, 1, 101)
        colour_values, symbol_values = group_values(result, colour), group_values(result, symbol)
        colours = {v: palette[i % len(palette)] for i, v in enumerate(colour_values.unique())}
        symbols = {v: markers[i % len(markers)] for i, v in enumerate(symbol_values.unique())}
        figure.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Chance', line={'color': '#888888', 'dash': 'dash'}))
        found = False
        for label, mask in group_masks(result, colour, symbol):
            curves, aucs = roc_splits(result, mask, class_index)
            if not curves:
                continue
            found = True
            if label == 'Overall':
                for fpr, tpr in curves:
                    figure.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', showlegend=False, hoverinfo='skip',
                        line={'color': 'rgba(100,100,100,.2)', 'width': 1}))
            interpolated = np.asarray([np.interp(grid, fpr, tpr) for fpr, tpr in curves])
            interpolated[:, 0], interpolated[:, -1] = 0, 1
            sd = f' ± {np.std(aucs, ddof=1):.3f}' if len(aucs) > 1 else ''
            first = np.flatnonzero(mask)[0]
            figure.add_trace(go.Scatter(x=grid, y=interpolated.mean(axis=0), mode='lines+markers' if symbol else 'lines',
                marker={'symbol': 'circle' if label == 'Overall' else symbols[symbol_values.iloc[first]], 'size': 4},
                line={'color': '#222222' if label == 'Overall' else colours[colour_values.iloc[first]], 'width': 2},
                name=f'{label} · AUC {np.mean(aucs):.3f}{sd} ({len(curves)} splits)'))
        if not found:
            figure.add_annotation(text='ROC requires both classes and finite scores within a split.', x=.5, y=.5, xref='paper', yref='paper', showarrow=False)
        figure.update_xaxes(title='False positive rate', range=[0, 1], constrain='domain')
        figure.update_yaxes(title='True positive rate', range=[0, 1], scaleanchor='x', scaleratio=1, constrain='domain')
    figure.update_layout(
        template='plotly_white', 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='#bbd6f8',
        margin={'l': 15, 'r': 15, 't': 5, 'b': 5}, 
        font={'size': 13 if full_screen else 10},
        showlegend=full_screen,
        legend={'orientation': 'h', 'y': -0.15},
        modebar={'orientation':'v'},
        uirevision=f'{result.task}:{chart}:{class_index}:{colour}:{symbol}:{subgroup}')
    return figure
