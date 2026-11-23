"""Matched held-out residual diagnostics, with portable aggregate-only views."""
from dataclasses import dataclass

import numpy as np
import plotly.graph_objects as go
from code_recording import recordable
from pipeline_search import METRICS
from plotly.subplots import make_subplots

BINS = 32


@dataclass(frozen=True)
class ResidualDiagnostics:
    summary: dict
    samples: object = None  # class × sampled validation event × model; session-only
    rows: object = None
    folds: object = None

    def __deepcopy__(self, memo):
        return self


def _finite_number(value):
    return float(value) if np.isfinite(value) else None


@recordable
def build_residual_diagnostics(members, blocks, task, classes=()):
    """Blocks share validation row/fold identities across every selected model."""
    if not blocks:
        return None
    rows = np.concatenate([block[0] for block in blocks])
    folds = np.concatenate([np.full(len(block[0]), i) for i, block in enumerate(blocks)])
    values = np.concatenate([block[1] for block in blocks], axis=1)
    models = [{'key': key, 'label': label} for key, label in (*members, ('ensemble', 'Ensemble'))]
    count = len(models)
    views = []
    for index, residuals in enumerate(values):
        finite = np.isfinite(residuals)
        maximum = np.max(np.abs(residuals[finite])) if finite.any() else 0.
        limit = float(max(maximum * 1.05, 1e-9))
        edges = np.linspace(-limit, limit, BINS + 1)
        statistics, pairs = [], []
        for column in range(count):
            mask = finite[:, column]
            data = residuals[mask, column]
            statistics.append({'events': int(mask.sum()), 'observations': len(np.unique(rows[mask])),
                'bias': float(data.mean()) if len(data) else None,
                'sd': float(data.std()) if len(data) else None,
                'histogram': np.histogram(data, bins=edges)[0].tolist()})
        for i in range(count):
            for j in range(i):
                mask = finite[:, i] & finite[:, j]
                x, y = residuals[mask, j], residuals[mask, i]
                correlation = None
                if len(x) > 1 and np.ptp(x) > 0 and np.ptp(y) > 0:
                    correlation = _finite_number(np.corrcoef(x, y)[0, 1])
                density = np.histogram2d(x, y, bins=(edges, edges))[0].astype(int)
                a, b = np.nonzero(density)
                pairs.append({'i': i, 'j': j, 'correlation': correlation, 'events': int(mask.sum()),
                    'observations': len(np.unique(rows[mask])),
                    'bins': [[int(u), int(v), int(density[u, v])] for u, v in zip(a, b)]})
        label = 'Residual (predicted − actual)' if task == 'regression' else f'Probability error · class {classes[index]} (predicted − observed)'
        views.append({'label': label, 'limit': limit, 'events': len(rows),
                      'observations': len(np.unique(rows)), 'statistics': statistics, 'pairs': pairs})
    # One deterministic event sample is shared by all pairs/classes. Bound the
    # total SVG points, rather than silently omitting any manually selected pair.
    sample_size = min(len(rows), 2000, max(1, 20000 // max(1, count * (count - 1) // 2)))
    sample = np.linspace(0, len(rows) - 1, sample_size, dtype=int)
    summary = {'version': 1, 'models': models, 'classes': [str(c) for c in classes], 'views': views,
               'note': ('Probability residuals require predict_proba; unavailable models remain visible as unavailable.'
                        if task == 'classification' else '')}
    return ResidualDiagnostics(summary, values[:, sample], rows[sample], folds[sample])


@recordable
def residual_block(validation, observed, responses, combined, task, classes):
    """Construct aligned residuals without calling a model or fitting anything."""
    all_responses = [response for _, response in responses] + [combined]
    count = len(all_responses)
    values = np.full((1 if task == 'regression' else len(classes), len(validation), count), np.nan)
    for column, response in enumerate(all_responses):
        if task == 'regression':
            predicted = response.values.get('predict')
            if predicted is not None and np.asarray(predicted).shape == (len(validation),):
                values[0, :, column] = np.asarray(predicted, dtype=float) - np.asarray(observed, dtype=float)
        elif 'predict_proba' in response.values and response.classes is not None:
            probabilities = np.asarray(response.values['predict_proba'])
            if probabilities.shape != (len(validation), len(response.classes)):
                continue
            for index, label in enumerate(classes):
                positions = np.flatnonzero(np.asarray(response.classes) == label)
                predicted = probabilities[:, positions[0]] if len(positions) else np.zeros(len(validation))
                values[index, :, column] = predicted - (np.asarray(observed) == label)
    return np.asarray(validation, dtype=int), values


def restore_residual_diagnostics(summary, expected_keys):
    if summary is None:
        return None
    try:
        if summary['version'] != 1 or [m['key'] for m in summary['models']] != list(expected_keys):
            raise ValueError('Invalid saved residual diagnostics.')
        count = len(expected_keys)
        if len(summary['views']) != max(1, len(summary['classes'])):
            raise ValueError('Invalid saved residual diagnostics.')
        for view in summary['views']:
            if not np.isfinite(view['limit']) or view['limit'] <= 0 or len(view['statistics']) != count:
                raise ValueError('Invalid saved residual diagnostics.')
            if {(p['i'], p['j']) for p in view['pairs']} != {(i, j) for i in range(count) for j in range(i)}:
                raise ValueError('Invalid saved residual diagnostics.')
            for stat in view['statistics']:
                if len(stat['histogram']) != BINS or any(n < 0 for n in stat['histogram']):
                    raise ValueError('Invalid saved residual diagnostics.')
            for pair in view['pairs']:
                r = pair['correlation']
                if r is not None and (not np.isfinite(r) or abs(r) > 1.0000001):
                    raise ValueError('Invalid saved residual diagnostics.')
                if any(not (0 <= a < BINS and 0 <= b < BINS and n > 0) for a, b, n in pair['bins']):
                    raise ValueError('Invalid saved residual diagnostics.')
        return ResidualDiagnostics(summary)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError('Invalid saved residual diagnostics.') from exc


def _view(result, class_index):
    diagnostics = result.residuals
    if diagnostics is None:
        return None, None
    views = diagnostics.summary['views']
    return diagnostics, views[min(max(int(class_index), 0), len(views) - 1)]


@recordable
def performance_figure(result, metric, full_screen=False, class_index=0, pair_axes=None):
    """Lower correlation triangle, upper residual pairs and row-aligned scores."""
    table = result.table(metric)
    keys, names = table.Key.tolist(), table.Model.tolist()
    n = len(keys)
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, column_widths=[0.7, 0.3],
                        horizontal_spacing=.015, subplot_titles=('Residual correlations', 'Resampled performance'))
    fig.add_trace(go.Bar(x=table.Score, y=keys, orientation='h', name='Resampled performance',
        marker_color=['#e69f00' if key == 'ensemble' else '#337ab7' for key in keys],
        customdata=[[name, _finite_number(sd), int(splits)] for name, sd, splits in zip(names, table.SD, table.Splits)],
        hovertemplate='%{customdata[0]}<br>Score %{x:.5g}<br>Fold SD %{customdata[1]:.5g}<br>%{customdata[2]} splits<extra></extra>'), row=1, col=2)
    diagnostics, view = _view(result, class_index)
    representation, limit = 'unavailable', None
    if view is not None:
        original = [m['key'] for m in diagnostics.summary['models']]
        indices = [original.index(key) for key in keys]
        pairs = {(p['i'], p['j']): p for p in view['pairs']}
        sample_index = min(max(int(class_index), 0), len(diagnostics.summary['views']) - 1)
        samples = diagnostics.samples[sample_index] if diagnostics.samples is not None else None
        representation = 'points' if samples is not None else 'density'
        limit = view['limit']
        if samples is not None:
            finite = samples[np.isfinite(samples)]
            limit = float(max(np.max(np.abs(finite)) * 1.05, 1e-9)) if len(finite) else 1e-9
        centers = np.linspace(-view['limit'], view['limit'], BINS, endpoint=False) + view['limit'] / BINS
        z = [[None] * n for _ in range(n)]
        background = [[0 if j <= i else None for j in range(n)] for i in range(n)]
        hover = [[''] * n for _ in range(n)]
        labels_x, labels_y, labels, label_colors = [], [], [], []
        for i, a in enumerate(indices):
            for j, b in enumerate(indices):
                if j > i:
                    continue
                if a == b:
                    stat = view['statistics'][a]
                    r = 1. if stat['events'] > 1 and stat['sd'] > 0 else None
                    events, observations = stat['events'], stat['observations']
                    detail = (f"<br>Bias {stat['bias']:.5g}<br>SD {stat['sd']:.5g}" if stat['bias'] is not None else '')
                else:
                    pair = pairs[max(a, b), min(a, b)]
                    r, events, observations = pair['correlation'], pair['events'], pair['observations']
                    detail = ''
                z[i][j] = r
                hover[i][j] = (f'{names[i]} / {names[j]}<br>Pearson r: {r:.3f}' if r is not None else
                               f'{names[i]} / {names[j]}<br>Correlation unavailable (missing or constant residuals)')
                hover[i][j] += f'<br>{events:,} paired validation predictions<br>{observations:,} unique observations' + detail
                labels_x.append(keys[j])
                labels_y.append(keys[i])
                labels.append(f'{r:.2f}' if r is not None else 'N/A')
                label_colors.append('white' if r is not None and abs(r) > .55 else '#222222')
        fig.add_trace(go.Heatmap(x=keys, y=keys, z=background, colorscale=[[0, '#dddddd'], [1, '#dddddd']],
            showscale=False, text=hover, hoverongaps=False, hovertemplate='%{text}<extra></extra>',
            meta={'role': 'correlation-background'}, name='Correlation background'), row=1, col=1)
        fig.add_trace(go.Heatmap(x=keys, y=keys, z=z, zmin=-1, zmax=1, zmid=0, colorscale='RdBu_r',
            text=hover, hovertemplate='%{text}<extra></extra>', hoverongaps=False, showscale=full_screen,
            meta={'role': 'correlation'}, name='Residual correlation',
            colorbar={'orientation': 'h', 'yref': 'container', 'y': .02, 'yanchor': 'bottom',
                      'x': .38, 'len': .6, 'thickness': 7, 'tickfont': {'size': 9},
                      'title': {'text': 'Pearson r', 'side': 'top', 'font': {'size': 9}}}), row=1, col=1)
        fig.add_trace(go.Scatter(x=labels_x, y=labels_y, text=labels, mode='text', hoverinfo='skip',
            textfont={'size': 11 if full_screen else 9, 'color': label_colors},
            meta={'role': 'correlation-values'}, name='Correlation values'), row=1, col=1)
        left, right = fig.layout.xaxis.domain
        pair_axes = {} if pair_axes is None else pair_axes
        for i, a in enumerate(indices):
            for j in range(i + 1, n):
                b = indices[j]
                pair_id = tuple(sorted((keys[i], keys[j])))
                if pair_id not in pair_axes:
                    pair_axes[pair_id] = max(pair_axes.values(), default=2) + 1
                axis_index = pair_axes[pair_id]
                pair = pairs[max(a, b), min(a, b)]
                xref, yref = f'x{axis_index}', f'y{axis_index}'
                cell_meta = {'role': 'residual-pair', 'row': i, 'column': j, 'x_key': keys[j], 'y_key': keys[i]}
                if samples is not None:
                    mask = np.isfinite(samples[:, a]) & np.isfinite(samples[:, b])
                    trace = go.Scatter(x=samples[mask, b], y=samples[mask, a], mode='markers',
                        marker={'size': 3, 'opacity': .4, 'color': '#337ab7'},
                        customdata=np.column_stack((diagnostics.rows[mask] + 1, diagnostics.folds[mask] + 1)),
                        hovertemplate=names[j] + ' residual %{x:.5g}<br>' + names[i] + ' residual %{y:.5g}<br>Row %{customdata[0]} · split %{customdata[1]}<extra></extra>')
                else:
                    density = np.zeros((BINS, BINS))
                    for x, y, count in pair['bins']:
                        density[y, x] = count
                    if a < b:
                        density = density.T
                    trace = go.Heatmap(x=centers, y=centers, z=[[int(v) if v else None for v in line] for line in density],
                        colorscale='Blues', showscale=False,
                        hovertemplate=names[j] + ' residual %{x:.5g}<br>' + names[i] + ' residual %{y:.5g}<br>%{z} validation predictions<extra></extra>')
                trace.update(xaxis=xref, yaxis=yref, meta=cell_meta, name=f'{names[j]} / {names[i]}',
                             legendgroup='pair:' + '|'.join(sorted((keys[i], keys[j]))))
                fig.add_trace(trace)
                common = {'range': [-limit, limit], 'autorange': False, 'showticklabels': False,
                          'showgrid': False, 'zeroline': True, 'zerolinecolor': '#aaaaaa', 'zerolinewidth': 1}
                gap = .003
                fig.update_layout({f'xaxis{axis_index}': {**common, 'anchor': yref,
                    'domain': [left + (right-left)*j/n + gap, left + (right-left)*(j+1)/n - gap]},
                    f'yaxis{axis_index}': {**common, 'anchor': xref,
                    'domain': [1-(i+1)/n + gap, 1-i/n - gap]}})
                fig.add_shape(type='line', x0=-limit, y0=-limit, x1=limit, y1=limit,
                    xref=xref, yref=yref, line={'color': '#888888', 'width': 1, 'dash': 'dot'})
                if not pair['events']:
                    fig.add_annotation(text='Unavailable', x=0, y=0, xref=xref, yref=yref, showarrow=False,
                                       font={'size': 9})
    #     note = f"{view['observations']:,} observations · {view['events']:,} validation predictions · {view['label']}"
    #     note += ' · sampled pairs' if samples is not None else ' · saved pair density'
    # else:
    #     note = 'Saved scores have no residual diagnostics; seeking current diagnostics.'
    ticks = names if full_screen else [name if len(name) <= 14 else name[:12] + '…' for name in names]
    fig.update_xaxes(tickmode='array', tickvals=keys, ticktext=ticks, tickangle=-45,
                     categoryorder='array', categoryarray=keys, range=[-.5, n-.5], autorange=False, row=1, col=1)
    fig.update_xaxes(title_text=METRICS[result.task][metric], row=1, col=2)
    for col in (1, 2):
        fig.update_yaxes(categoryorder='array', categoryarray=keys, range=[n-.5, -.5], autorange=False,
                         tickmode='array', tickvals=keys, ticktext=ticks, automargin=False, row=1, col=col)
    fig.update_layout(autosize=True, width=None, height=None,
        showlegend=False, modebar={'orientation': 'v'},
        meta={'model_keys': keys, 'pair_count': n*(n-1)//2, 'representation': representation, 'residual_limit': limit},
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#bbd6f8',
        margin={'l': 150 if full_screen else 110, 'r': 35 if full_screen else 5,
                't': 60 if full_screen else 35, 'b': 100 if full_screen else 60, 'autoexpand': False},
        font={'size': 13 if full_screen else 10})
    # fig.add_annotation(text=note, x=0, y=1.16, xanchor='left', xref='paper', yref='paper',
    #                    showarrow=False, font={'size': 10})
    return fig
