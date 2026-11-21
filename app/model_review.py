"""Review completed CV trials without fitting, complexity scores or ensembling."""
import re
from collections.abc import Mapping
from copy import deepcopy

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from model_commit import configured_pipeline
from pipeline_comparison import comparison_data, paired_posterior
from pipeline_search import METRICS, displayed, mean_score
from plotly.colors import qualitative
from sklearn.base import clone


class CandidatePipelines(Mapping):
    """One unfitted template per method; trial pipelines materialize on demand."""

    def __init__(self, templates, choices):
        self.templates = templates
        self.choices = choices

    def __getitem__(self, key):
        method, parameters = self.choices[key]
        if self.templates.get(method) is None:
            raise ValueError('This historical pipeline is unavailable in the current runtime.')
        return configured_pipeline(self.templates[method], {k: v for k, v in parameters.items() if k != '__method__'})

    def __iter__(self):
        return iter(self.choices)

    def __len__(self):
        return len(self.choices)


def completed_evidence(problem, state):
    """Completed scores/summaries and shared unfitted method templates only."""
    trials = [deepcopy(t) for t in state['trials'] if t['state'] == 'complete']
    templates, choices = {}, {}
    for trial in trials:
        key = trial['params'].get('__method__')
        branch = problem.branches[key] if problem.branches else problem
        method = trial['model_id']
        if method not in templates:
            templates[method] = clone(branch.pipeline)
        # Prepared templates already include fixed seeds/thread limits. Share
        # each trial's scalar choices with its history instead of duplicating them.
        choices[str(trial['id'])] = (method, trial['params'])
    return {'version': 2, 'state': {'specification': deepcopy(problem.specification), 'trials': trials},
            'pipelines': CandidatePipelines(templates, choices)}


def archived_evidence(source, state):
    """Read-only historical review when data match but native fitting is unavailable."""
    trials = [deepcopy(t) for t in state['trials'] if t['state'] == 'complete']
    templates, choices = {}, {}
    candidates = {c['id'] for c in source.model_candidates}
    recipe_ids = {r.parameters.get('recipe_id') for r in source.processing_records if r.card == 'cand_methods'}
    spec = state['specification']
    for trial in trials:
        method = trial['model_id']
        if method not in templates:
            try:
                pipeline = source.pipeline_for_training(method if method in candidates else None)
                if pipeline is None or method not in (candidates or recipe_ids):
                    raise ValueError('Method unavailable')
                branch = spec.get('branches', {}).get(method, spec)
                templates[method] = pipeline.set_params(**branch.get('fixed', {}))
            except (ValueError, TypeError, KeyError, AttributeError):
                templates[method] = None
        choices[str(trial['id'])] = (method, trial['params'])
    return {'version': 2, 'read_only': True,
            'state': {'specification': deepcopy(spec), 'trials': trials},
            'pipelines': CandidatePipelines(templates, choices)}


def interpretability(pipeline):
    model = pipeline.steps[-1][1]
    if hasattr(model, 'steps'):
        model = model.steps[-1][1]
    module, name = type(model).__module__, type(model).__name__
    if '.linear_model' in module or module.startswith(('skglm', 'glmnet')):
        return 'Coefficients describe additive effects on the model scale; preprocessing affects their interpretation.'
    if name in {'DecisionTreeRegressor', 'DecisionTreeClassifier', 'ExtraTreeRegressor', 'ExtraTreeClassifier'}:
        return 'Individual predictions can be traced through a single tree; inspect the fitted paths.'
    if 'Cubist' in name:
        return 'Rules with local linear predictions; committees and corrections can make explanations less direct.'
    if 'GaussianProcess' in name:
        return 'Kernel-based predictions; no single global coefficient explanation.'
    if name.startswith('Dummy'):
        return 'A constant or simple reference prediction with an explicit rule.'
    return 'Inspect method-specific explanation tools; this description does not certify transparency or causal interpretation.'


def diagnostics(trial, spec):
    summary = trial.get('diagnostics', {})
    folds = [f.get('evidence', {}) for f in trial['folds']]
    complete = bool(folds) and all('count' in e for e in folds)
    seconds = sum(e.get('prediction_seconds', 0) for e in folds) if complete else 0
    note = summary.get('error') or ('Compact out-of-fold error summaries; observed warm validation-batch timing, including pipeline transforms.'
                                  if summary.get('version') == 2 else 'Prediction summaries are unavailable; rerun tuning to collect them.')
    result = {'Observed rows/s': sum(e['count'] for e in folds) / seconds if complete and seconds > 0 else np.nan,
              'Evidence': note}
    if spec['task'] == 'regression':
        result['P95 absolute error'] = summary.get('p95_absolute_error', np.nan)
    else:
        result['Error rate'] = summary.get('error_rate', np.nan)
        result['Log loss'] = summary.get('log_loss') if summary.get('log_loss') is not None else np.nan
    return result


def review_frame(results, tolerance=None, limit=5, probability=.95):
    if results is None:
        return pd.DataFrame(), 'Complete pipeline tuning upstream to review candidate models.'
    if limit is not None and (type(limit) is not int or not 1 <= limit <= 20):
        return pd.DataFrame(), 'Choose 1–20 trials per method.'
    if not isinstance(probability, (int, float)) or not .5 < probability < 1:
        return pd.DataFrame(), 'Choose a decision probability between 0.5 and 1.'
    if tolerance is not None and (not np.isfinite(tolerance) or tolerance <= 0):
        return pd.DataFrame(), 'Use a positive practical-equivalence tolerance, or leave it blank.'
    state, rows = results['state'], []
    spec = state['specification']
    comparable, ratios, note = comparison_data(state)
    eligible = {t['id'] for t in comparable}
    groups = {}
    for trial in state['trials']:
        groups.setdefault(trial['model_id'], []).append(trial)
    for key, trials in groups.items():
        trials.sort(key=lambda t: (-mean_score(t, spec['metric']), t['id']))
        best = trials[0]
        # The description depends on the method's estimator class, not trial
        # parameters. Avoid cloning a full pipeline for every displayed trial.
        try:
            description = interpretability(results['pipelines'][str(best['id'])])
        except (ValueError, KeyError, TypeError):
            description = 'Historical model unavailable here; review saved diagnostics only.'
        for trial in trials[:limit]:
            row = {'Trial': trial['id'] + 1, 'Method': trial.get('method', key), 'Model': key,
                   'Score': displayed(mean_score(trial, spec['metric']), spec['metric']),
                   'Loss vs method best': mean_score(best, spec['metric']) - mean_score(trial, spec['metric']),
                   'P(equivalent)': np.nan, 'Assessment': 'Inconclusive',
                   'Interpretability': description}
            if trial['id'] == best['id']:
                row['Assessment'] = 'Best score (reference)'
            elif tolerance is not None and trial['id'] in eligible and best['id'] in eligible:
                differences = np.asarray([f[spec['metric']] for f in best['folds']]) - np.asarray([f[spec['metric']] for f in trial['folds']])
                try:
                    posterior = paired_posterior(differences, ratios, tolerance, probability)
                    row['P(equivalent)'] = posterior['P(equivalent)']
                    row['Assessment'] = {'Practically equivalent': 'Practically equivalent', 'A better': 'Meaningfully worse'}.get(posterior['Assessment'], 'Inconclusive')
                except ValueError:
                    row['Assessment'] = 'Inconclusive (interval unavailable)'
            row.update(diagnostics(trial, spec))
            rows.append(row)
    messages = []
    if note:
        messages.append('Equivalence unavailable for these evaluation splits.')
    if tolerance is None:
        messages.append('Set a tolerance to assess equivalence.')
    if results.get('read_only'):
        messages.append('Historical review is read-only.')
    return pd.DataFrame(rows), ' '.join(messages)


def representatives(frame, choices):
    if frame.empty:
        return frame
    ids = [choices.get(key, int(group.iloc[0]['Trial'])) for key, group in frame.groupby('Model', sort=False)]
    # Invalid/stale choices revert to the best visible row of that method.
    return pd.DataFrame([group.loc[group.Trial == selected].iloc[0] if selected in group.Trial.values else group.iloc[0]
                         for (_, group), selected in zip(frame.groupby('Model', sort=False), ids)]).reset_index(drop=True)


def axis_options(task):
    return ['Score', 'P95 absolute error', 'Observed rows/s'] if task == 'regression' else ['Score', 'Error rate', 'Log loss', 'Observed rows/s']


def pareto_mask(frame, axes, metric):
    if frame.empty:
        return np.zeros(0, dtype=bool)
    values = frame[list(axes)].to_numpy(dtype=float, copy=True)
    for i, axis in enumerate(axes):
        if axis == 'Observed rows/s' or (axis == 'Score' and not metric.startswith('neg_')):
            values[:, i] *= -1
    valid = np.isfinite(values).all(axis=1)
    return np.array([valid[i] and not any(valid[j] and np.all(values[j] <= values[i]) and np.any(values[j] < values[i])
                                        for j in range(len(values)) if j != i) for i in range(len(values))])


def within_figure(frame, method, tolerance):
    figure = go.Figure()
    subset = frame.loc[frame.Model == method] if not frame.empty else frame
    colors = {'Best score (reference)': '#636efa', 'Practically equivalent': '#2ca02c', 'Meaningfully worse': '#d62728'}
    if tolerance is not None and np.isfinite(tolerance) and tolerance > 0:
        figure.add_vrect(x0=-tolerance, x1=tolerance, fillcolor='green', opacity=.12, line_width=0)
    if not subset.empty:
        labels = ['Reference' if str(r.Assessment).startswith('Best') else
                  f'{r["P(equivalent)"]:.0%} within tolerance' if pd.notna(r['P(equivalent)']) else r.Assessment
                  for _, r in subset.iterrows()]
        figure.add_trace(go.Scatter(x=subset['Loss vs method best'], y=[f'Trial {n}' for n in subset.Trial],
            mode='markers+text', text=labels, textposition='middle right',
            marker={'color': [colors.get(a, '#7f7f7f') for a in subset.Assessment], 'size': 10},
            meta={'trials': subset.Trial.tolist()}, customdata=subset.Assessment, hovertemplate='%{y}<br>Loss: %{x:.6g}<br>%{customdata}<extra></extra>'))
    if not subset.empty:
        bound = tolerance if tolerance is not None and np.isfinite(tolerance) and tolerance > 0 else 0.
        low, high = min(0., -bound), max(float(subset['Loss vs method best'].max()), bound)
        span = max(high - low, 1e-9)
        figure.update_xaxes(range=[low - .05 * span, high + .65 * span])
    figure.update_xaxes(title_text='Performance loss vs method best (positive = worse)')
    figure.update_yaxes(autorange='reversed')
    return style(figure, 'Configurations within a method')


def tradeoff_figure(frame, axes, metric):
    figure = go.Figure()
    frontier = pareto_mask(frame, axes, metric)
    for i, (_, row) in enumerate(frame.iterrows()):
        figure.add_trace(go.Scatter(x=[row[axes[0]]], y=[row[axes[1]]], mode='markers', name=row.Method,
            marker={'color': qualitative.Plotly[i % len(qualitative.Plotly)], 'size': 12,
                    'symbol': 'diamond' if frontier[i] else 'circle'},
            meta={'trials': [int(row.Trial)]}, text=[f'{row.Method} · Trial {row.Trial} · {"Pareto frontier" if frontier[i] else "Trade-off / incomplete evidence"}'],
            hovertemplate='%{text}<br>%{x:.6g}, %{y:.6g}<extra></extra>'))
    metric_label = next(label for metrics in METRICS.values() for key, label in metrics.items() if key == metric)
    figure.update_xaxes(title_text=f'Score ({metric_label})' if axes[0] == 'Score' else axes[0])
    figure.update_yaxes(title_text=f'Score ({metric_label})' if axes[1] == 'Score' else axes[1])
    return style(figure, 'Method representatives · diamonds: Pareto frontier')


def error_figure(results, frame):
    figure = go.Figure()
    spec = results['state']['specification']
    selected = [t for t in results['state']['trials'] if t['id'] + 1 in frame.Trial.values]
    probability_loss = bool(selected) and all(t.get('diagnostics', {}).get('loss_kind') == 'log_loss' for t in selected)
    for i, (_, row) in enumerate(frame.iterrows()):
        trial = next(t for t in selected if t['id'] == row.Trial - 1)
        summary = trial.get('diagnostics', {})
        if summary.get('version') != 2:
            continue
        if spec['task'] == 'classification' and not probability_loss:
            if 'error_rate' not in summary:
                continue
            thresholds = np.linspace(0, 1, 101)
            rates = np.where(thresholds < 1, summary['error_rate'], 0.) * 100
        else:
            thresholds = np.linspace(0., summary['maximum'], len(summary['exceedance']))
            rates = np.asarray(summary['exceedance']) * 100
        figure.add_trace(go.Scatter(x=thresholds, y=rates, mode='lines', name=row.Method,
            line={'color': qualitative.Plotly[i % len(qualitative.Plotly)]},
            hovertemplate='Threshold: %{x:.6g}<br>Exceeding: %{y:.3g}%<extra>%{fullData.name}</extra>'))
    figure.update_xaxes(title_text='Absolute error threshold' if spec['task'] == 'regression' else 'Log-loss threshold' if probability_loss else 'Classification error threshold (0=correct, 1=incorrect)')
    figure.update_yaxes(title_text='Observations exceeding threshold (%)', range=[0, 100])
    return style(figure, 'Error exceedance · mean of repeat curves')


def style(figure, title):
    figure.update_layout(title={'text': title, 'font': {'size': 13}}, template='plotly_white',
                         margin={'l': 50, 'r': 45, 't': 40, 'b': 40}, legend={'orientation': 'h'})
    return figure


def parameter_label(path, trial):
    """Readable display label; search/estimator parameter keys remain unchanged."""
    root, *children = path.split('__')
    if root == 'cand_methods' or not children:
        method = trial.get('method', trial['model_id'])
        method = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', method)
        title = re.sub(r'\s+(Regressor|Regression|Classifier|Classification)$', '', method)
        suffix = '-'.join(children) if children else root
    else:
        title = root.replace('_', ' ').capitalize()
        suffix = '-'.join(children)
    return f"{title}: {suffix.replace('_', '-')}"


def parameter_differences(results, selected, reference):
    trials = {t['id'] + 1: t for t in results['state']['trials']}
    try:
        a = results['pipelines'][str(selected - 1)].get_params()
        b = results['pipelines'][str(reference - 1)].get_params()
    except (ValueError, KeyError):
        # Read-only browser review need not import the original native estimator.
        a, b = trials[selected]['params'], trials[reference]['params']
    paths = set(trials[selected]['params']) | set(trials[reference]['params'])
    rows = [{'Parameter': parameter_label(p, trials[selected]), f'Trial {reference}': repr(b.get(p, 'Inactive / not varied')),
             f'Trial {selected}': repr(a.get(p, 'Inactive / not varied'))}
            for p in sorted(paths - {'__method__'}) if repr(a.get(p)) != repr(b.get(p))]
    return pd.DataFrame(rows)
