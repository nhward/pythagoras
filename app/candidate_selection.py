"""Weighted candidate decisions from all completed evidence, without fitting."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from model_review import review_frame
from pipeline_search import METRICS, displayed, mean_score
from plotly.colors import qualitative

AXES = ('Score', 'Observed rows/s', 'P95 absolute error')


def criteria(frame):
    return ('Score', 'Observed rows/s', 'Error rate' if 'Error rate' in frame else 'P95 absolute error')


def weights_from_handles(handles):
    values = np.asarray(handles, dtype=float)
    if values.shape != (2,) or not np.isfinite(values).all() or not 0 <= values[0] <= values[1] <= 100:
        raise ValueError('Choose two ordered weight boundaries between 0 and 100.')
    a, b = values / 100
    return np.array([a, b - a, 1 - b])


def candidate_frame(results, tolerance=None, probability=.95):
    if results is None:
        return pd.DataFrame(), 'Complete pipeline tuning upstream to compare models.'
    frame, note = review_frame(results, tolerance, limit=None, probability=probability)
    if frame.empty:
        return frame, note or 'No completed configurations are available.'
    if results['state']['specification']['task'] == 'classification':
        return frame, note
    ratios = {}
    for trial in results['state']['trials']:
        rmse = displayed(mean_score(trial, 'neg_root_mean_squared_error'), 'neg_root_mean_squared_error') if all('neg_root_mean_squared_error' in fold for fold in trial['folds']) else None
        mae = displayed(mean_score(trial, 'neg_mean_absolute_error'), 'neg_mean_absolute_error') if all('neg_mean_absolute_error' in fold for fold in trial['folds']) else None
        ratios[trial['id'] + 1] = rmse / mae if rmse is not None and mae is not None and mae > 0 else np.nan
    frame['RMSE/MAE'] = frame.Trial.map(ratios)
    return frame, note


def pareto_frontier(frame, metric):
    """All three axes; missing coordinates do not fabricate frontier points."""
    if frame.empty:
        return np.zeros(0, dtype=bool)
    values = frame[list(criteria(frame))].to_numpy(dtype=float, copy=True)
    if not metric.startswith('neg_'):
        values[:, 0] *= -1
    values[:, 1] *= -1
    valid = np.isfinite(values).all(axis=1) & (frame['Observed rows/s'].to_numpy() > 0) & (frame[criteria(frame)[2]].to_numpy() >= 0)
    return np.array([valid[i] and not np.any(valid & np.all(values <= row, axis=1) & np.any(values < row, axis=1))
                     for i, row in enumerate(values)])


def weighted_winner(frame, metric, weights, floor):
    """Log WPM: minimize error, maximize throughput, minimize tail error.

    A positive explicit floor applies to zero/near-zero loss. R² uses 1-R².
    Missing inactive criteria are allowed. Exact utility ties prefer the full
    Pareto frontier, then the primary score and stable trial identity.
    """
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (3,) or not np.isfinite(weights).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError('Criterion weights must be nonnegative and sum to one.')
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError('Choose a positive minimum error for weighted products.')
    if frame.empty:
        return None
    values = frame[list(criteria(frame))].to_numpy(dtype=float, copy=True)
    if metric == 'r2':  # noqa: SIM114
        values[:, 0] = 1 - values[:, 0]
    elif metric in {'accuracy', 'balanced_accuracy', 'roc_auc'}:
        values[:, 0] = 1 - values[:, 0]
    elif metric not in {'neg_root_mean_squared_error', 'neg_mean_absolute_error', 'neg_log_loss'}:
        raise ValueError('Choose a supported regression or classification metric.')
    valid = np.ones(len(frame), dtype=bool)
    utility = np.zeros(len(frame))
    for i, weight in enumerate(weights):
        if weight == 0:
            continue
        column = values[:, i]
        available = np.isfinite(column) & (column > 0 if i == 1 else column >= 0)
        valid &= available
        safe = np.where(available, column, 1.)
        utility += weight * (np.log(safe) if i == 1 else -np.log(np.maximum(safe, floor)))
    indices = np.flatnonzero(valid)
    if not len(indices):
        return None
    best = indices[utility[indices] == np.max(utility[indices])]
    frontier = pareto_frontier(frame, metric)
    direction = 1 if metric.startswith('neg_') else -1
    index = min(best, key=lambda i: (not frontier[i], direction * frame.iloc[i].Score, int(frame.iloc[i].Trial)))
    return int(frame.iloc[index].Trial)


def decisions(frame, metric, weights, floor):
    return {'overall': weighted_winner(frame, metric, weights, floor),
            'methods': {str(key): weighted_winner(group, metric, weights, floor)
                        for key, group in frame.groupby('Model', sort=False)}}


def pareto_figure(frame, metric, winner, title, colours):
    frame = frame.reset_index(drop=True)
    task = 'classification' if 'Error rate' in frame else 'regression'
    error_label = 'Error rate' if task == 'classification' else 'P95 error'
    figure = go.Figure()
    frontier = pareto_frontier(frame, metric)
    visible = np.isfinite(frame[list(criteria(frame))].to_numpy(dtype=float)).all(axis=1)
    for method, group in frame.groupby('Model', sort=False):
        colour = qualitative.Plotly[colours.get(method, 0) % len(qualitative.Plotly)]
        for efficient in (False, True):
            rows = group.loc[visible[group.index] & (frontier[group.index] == efficient)]
            if rows.empty:
                continue
            figure.add_trace(go.Scatter3d(x=rows[AXES[0]], y=rows[AXES[1]], z=rows[criteria(frame)[2]], mode='markers',
                name=str(rows.iloc[0].Method), legendgroup=str(method), showlegend=efficient or not frontier[group.index].any(),
                marker={'color': colour, 'size': 6 if efficient else 4, 'opacity': 1 if efficient else .65,
                        'line': {'color': '#334155', 'width': 1}},
                customdata=rows.Trial, meta={'trials': rows.Trial.astype(int).tolist()},
                hovertemplate='Trial %{customdata}<br>Score %{x:.5g}<br>Rows/s %{y:.5g}<br>' + error_label + ' %{z:.5g}<extra>%{fullData.name}</extra>'))
    selected = frame.loc[frame.Trial == winner]
    if not selected.empty and np.isfinite(selected[list(criteria(frame))].to_numpy(dtype=float)).all():
        row = selected.iloc[0]
        figure.add_trace(go.Scatter3d(x=[row[AXES[0]]], y=[row[AXES[1]]], z=[row[criteria(frame)[2]]], mode='markers',
            name=f'Weighted winner · Trial {winner}', marker={'size': 11, 'symbol': 'diamond', 'color': '#111111',
            'line': {'width': 3, 'color': '#ffffff'}}, meta={'trials': [winner]},
            hovertemplate=f'Trial {winner} · weighted winner<extra></extra>'))
    figure.update_layout(title={'text': title, 'font': {'size': 16}}, margin={'l': 0, 'r': 0, 't': 35, 'b': 0},
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#bbd6f8',
        autosize=True, uirevision='cand-models-camera', scene={'xaxis': {'title': {'text': METRICS[task][metric].split(' · ')[0], 'font': {'size': 11}}, 'tickfont': {'size': 10}, 'nticks': 3},
            'yaxis': {'title': {'text': 'Rows/s', 'font': {'size': 11}}, 'tickfont': {'size': 10}, 'nticks': 3},
            'zaxis': {'title': {'text': error_label, 'font': {'size': 11}}, 'tickfont': {'size': 10}, 'nticks': 3},
            'aspectmode': 'cube'}, legend={'orientation': 'h', 'y': -.05})
    figure.update_scenes(bgcolor='#bbd6f8', **{
        axis: {'backgroundcolor': '#bbd6f8', 'showbackground': True, 'gridcolor': '#ffffff', 'zerolinecolor': '#ffffff'}
        for axis in ('xaxis', 'yaxis', 'zaxis')})
    return figure


def transparent_candidates(frame, results, catalogue):
    """Restrict methods by explicit catalogue identity, never by display names."""
    if frame.empty:
        return frame.copy()
    entries = catalogue['estimators']
    by_id = {entry['id']: entry for entry in entries}
    recipes = {row['id']: row['estimator'] for section in ('profiles', 'specialists')
               for row in catalogue.get(section, [])}
    allowed = set()
    pipelines = results.get('pipelines', {})
    templates = getattr(pipelines, 'templates', {})
    for method in frame.Model.unique():
        entry = by_id.get(recipes.get(method, method))
        pipeline = templates.get(method)
        if pipeline is None:
            trial = int(frame.loc[frame.Model == method, 'Trial'].iloc[0]) - 1
            try:
                pipeline = pipelines[str(trial)]
            except (ValueError, KeyError, TypeError):
                pipeline = None
        if pipeline is not None:
            # Prefer the actual terminal class over a user-assigned method ID.
            model = pipeline
            while hasattr(model, 'steps'):
                model = model.steps[-1][1]
            implementation = type(model).__module__ + '.' + type(model).__name__
            matches = [row for row in entries if implementation in (row['id'], row.get('implementation'))]
        else:
            # Historical read-only evidence can still carry an exact recipe ID.
            matches = [entry] if entry is not None else []
        # Conflicting aliases or absent metadata cannot certify the method.
        if matches and all(row.get('taxonomy', {}).get('transparency', {}).get('status') == 'reviewed'
                           and row['taxonomy']['transparency'].get('values') == ['usually_transparent']
                           for row in matches):
            allowed.add(method)
    return frame.loc[frame.Model.isin(allowed)].reset_index(drop=True)
