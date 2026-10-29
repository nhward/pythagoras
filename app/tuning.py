"""Parameter discovery and finite, independent tuning specifications."""
import hashlib
import inspect
import json
import math
import re
from copy import deepcopy
from decimal import Decimal, InvalidOperation

import numpy as np
from sklearn.base import BaseEstimator

# Public, application-owned suggestions, not sklearn private constraints.
STANDARD_METADATA = {
    'sklearn.impute._base.SimpleImputer': {
        'strategy': {'kind': 'categorical', 'choices': ['mean', 'median', 'most_frequent', 'constant'], 'importance': 90}},
    'sklearn.preprocessing._data.StandardScaler': {
        'with_mean': {'kind': 'boolean'}, 'with_std': {'kind': 'boolean'}},
}
for _name in ('RandomForestRegressor', 'RandomForestClassifier', 'ExtraTreesRegressor', 'ExtraTreesClassifier'):
    STANDARD_METADATA[f'sklearn.ensemble._forest.{_name}'] = {
        'n_estimators': {'kind':'integer', 'suggested':{'values':[100,300,500]}, 'importance':100},
        'max_depth': {'kind':'integer', 'suggested':{'values':[None,3,6,12]}, 'importance':95},
        'min_samples_leaf': {'kind':'integer', 'suggested':{'values':[1,2,5,10]}, 'importance':90},
    }
for _name in ('HistGradientBoostingRegressor', 'HistGradientBoostingClassifier'):
    STANDARD_METADATA[f'sklearn.ensemble._hist_gradient_boosting.gradient_boosting.{_name}'] = {
        'learning_rate': {'kind':'float', 'suggested':{'values':[0.01,0.05,0.1]}, 'importance':100},
        'max_iter': {'kind':'integer', 'suggested':{'values':[100,200,500]}, 'importance':95},
        'max_leaf_nodes': {'kind':'integer', 'suggested':{'values':[15,31,63]}, 'importance':90},
    }
STANDARD_METADATA['sklearn.linear_model._ridge.Ridge'] = {
    'alpha': {'kind':'float', 'suggested':{'values':[0.01,0.1,1.,10.,100.]}, 'importance':100}}
FIXED = {'handle_unknown', 'columns', 'target', 'target_columns', 'predictors', 'eligible', 'schemas',
         'rules', 'batches', 'variables', 'remove_original', 'features', 'steps', 'transformers',
         'remainder', 'func', 'inverse_func', 'kw_args', 'inv_kw_args', 'memory',
         'n_jobs', 'verbose', 'random_state', 'seed', 'weight_column', 'weight_columns',
         'weighting', 'use_weights', 'use_weighting', 'target_type', 'output_column', 'embedding_path', 'embedding_digest', 'embedding_label', 'methods'}


def text_value(value):
    if callable(value):
        return f'{value.__module__}.{getattr(value, "__qualname__", type(value).__qualname__)}'
    return re.sub(r' at 0x[0-9a-f]+', '', repr(value))


def discover(pipeline):
    if pipeline is None:
        return []
    params = pipeline.get_params(deep=True)
    rows = []
    for path, value in params.items():
        if '__' not in path or isinstance(value, BaseEstimator):
            continue
        owner_path, name = path.rsplit('__', 1)
        owner = params.get(owner_path)
        if not isinstance(owner, BaseEstimator):
            continue
        # Composite topology is represented by its actual child estimators.
        if name in ('steps', 'transformers', 'estimators'):
            continue
        cls = type(owner)
        identity = f'{cls.__module__}.{cls.__qualname__}'
        metadata = deepcopy(STANDARD_METADATA.get(identity, {}).get(name, {}))
        metadata.update(deepcopy(getattr(cls, 'tuning_metadata', {}).get(name, {})))
        kind = metadata.get('kind', '')
        if not kind:
            if isinstance(value, (bool, np.bool_)): kind = 'boolean'
            elif isinstance(value, (int, np.integer)): kind = 'integer'
            elif isinstance(value, (float, np.floating)): kind = 'float'
            elif isinstance(value, str): kind = 'categorical'
        try:
            parameter = inspect.signature(cls.__init__).parameters.get(name)
            default = parameter.default if parameter else inspect.Parameter.empty
        except (ValueError, TypeError):
            default = inspect.Parameter.empty
        fixed = metadata.get('fixed', name in FIXED or callable(value))
        definition = {'kind': kind, 'mode': 'values', 'values': []}
        if kind == 'boolean': definition['values'] = [False, True]
        if 'choices' in metadata: definition['values'] = metadata['choices']
        definition.update(metadata.get('suggested', {}))
        rows.append({'path': path, 'name': metadata.get('label', name.replace('_', ' ').capitalize()),
                     'source': owner_path.replace('__', ' › '), 'owner': identity,
                     'current': text_value(value),
                     'default': 'No constructor default' if default is inspect.Parameter.empty else text_value(default),
                     'fixed': bool(fixed), 'metadata': metadata, 'definition': definition,
                     'importance': metadata.get('importance', 0)})
    # Keep source steps in pipeline order; importance orders their parameters.
    sources = list(dict.fromkeys(row['source'] for row in rows))
    return sorted(rows, key=lambda row: (sources.index(row['source']), -row['importance'], row['name']))


def fingerprint(rows):
    return hashlib.sha256(json.dumps([(r['path'], r['owner'], r['current']) for r in rows]).encode()).hexdigest()


def search_values(definition, metadata=None):
    metadata = metadata or {}
    kind = definition.get('kind')
    if kind not in ('integer', 'float', 'categorical', 'boolean'):
        raise ValueError('Choose a parameter type first.')
    if definition.get('mode') == 'range':
        if kind not in ('integer', 'float'):
            raise ValueError('Ranges require a numeric parameter type.')
        try:
            start, stop, step = [Decimal(str(definition[key])) for key in ('min', 'max', 'step')]
            if not all(v.is_finite() for v in (start, stop, step)) or step <= 0 or stop < start:
                raise ValueError('Use finite endpoints, maximum ≥ minimum, and a positive step.')
            if kind == 'integer' and any(v != v.to_integral_value() for v in (start, stop, step)):
                raise ValueError('Integer ranges require integer endpoints and step.')
            count = int((stop - start) // step) + 1
            if count > 10000: raise ValueError('Limit each parameter to 10,000 values.')
            values = [(int if kind == 'integer' else float)(start + i * step) for i in range(count)]
        except (InvalidOperation, KeyError) as exc:
            raise ValueError('Enter minimum, maximum and step.') from exc
    else:
        values = definition.get('values', [])
    if not isinstance(values, list) or not values or len(values) > 10000:
        raise ValueError('Supply a nonempty list of at most 10,000 values.')
    for value in values:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('Choices must not contain infinity or NaN.')
        if kind == 'boolean' and not isinstance(value, bool): raise ValueError('Boolean choices must be true or false.')
        if kind in ('integer', 'float') and value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('Numeric choices must be finite numbers or null.')
            if kind == 'integer' and not isinstance(value, int): raise ValueError('Integer choices must be integers or null.')
        if not isinstance(value, (str, int, float, bool, type(None))):
            raise TypeError('Choices must be scalar JSON values.')
        bounds = metadata.get('bounds')
        if bounds and value is not None and (not isinstance(value, (int, float)) or not bounds[0] <= value <= bounds[1]):
            raise ValueError(f'Values must be between {bounds[0]} and {bounds[1]}.')
        if 'choices' in metadata and value not in metadata['choices']:
            raise ValueError('A value is outside the supported choices.')
    unique = []
    for value in values:
        if not any(type(value) is type(v) and value == v for v in unique): unique.append(value)
    return unique


def compile_spec(rows, drafts):
    grid = {}
    known = {r['path']: r for r in rows}
    for path, draft in drafts.items():
        if not draft.get('active'): continue
        if path not in known: raise ValueError(f'Parameter no longer present: {path}')
        row = known[path]
        if draft.get('owner') != row['owner']: raise ValueError(f'Parameter implementation changed: {path}')
        if row['fixed']: raise ValueError(f'{row["name"]} is fixed configuration.')
        grid[path] = search_values(draft, row['metadata'])
    return {'fingerprint': fingerprint(rows), 'grid': grid,
            'combinations': math.prod(len(values) for values in grid.values())}
