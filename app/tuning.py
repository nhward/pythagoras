"""Parameter discovery and finite tuning spaces with declared conditional branches."""
import hashlib
import inspect
import json
import math
import re
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.gaussian_process.kernels import Kernel
from sklearn.pipeline import Pipeline

PURPOSES = {'recommended': 'Recommended tuning', 'advanced': 'Advanced tuning',
            'unreviewed': 'Unreviewed', 'operational': 'Operational configuration',
            'recipe': 'Recipe parameters'}
VISIBILITY = {'recommended': 'Recommended', 'advanced': 'Recommended + advanced',
              'unreviewed': 'All tunable (including unreviewed)', 'everything': 'Everything (including operational and recipe)'}

# Public, reviewed application metadata. Parameter existence/type/choices alone
# never establish whether searching it is useful.
with (Path(__file__).parent / 'data' / 'tuning_parameters.json').open() as _stream:
    _registry = json.load(_stream)
if _registry.get('schema_version') != 1:
    raise ValueError('Unsupported tuning metadata version')
STANDARD_METADATA = _registry['estimators']
FIXED = set(_registry['fixed_parameters'])
for _parameters in STANDARD_METADATA.values():
    for _metadata in _parameters.values():
        if _metadata.get('purpose') not in PURPOSES or not _metadata.get('reason') or not _metadata.get('reference'):
            raise ValueError('Reviewed tuning metadata requires purpose, reason and reference')


def visible(row, level, *, active=False):
    if active or level == 'everything': return True
    if row['fixed']: return False
    return row['purpose'] in {'recommended': {'recommended'},
        'advanced': {'recommended', 'advanced'},
        'unreviewed': {'recommended', 'advanced', 'unreviewed'}}.get(level, {'recommended'})


def text_value(value):
    if callable(value):
        return f'{value.__module__}.{getattr(value, "__qualname__", type(value).__qualname__)}'
    return re.sub(r' at 0x[0-9a-f]+', '', repr(value))


def discover(pipeline):
    if pipeline is None:
        return []
    params = pipeline.get_params(deep=True)
    rows = []
    # Some optional wrappers omit unset constructor/kwargs parameters. Expose
    # only reviewed names; CatBoost also does not inherit BaseEstimator.
    reviewed_wrapper_owners = {}
    for owner_path, owner in list(params.items()):
        identities = [f'{cls.__module__}.{cls.__qualname__}' for cls in type(owner).__mro__]
        identity = next((name for name in identities if name in {
            'catboost.core.CatBoostRegressor', 'catboost.core.CatBoostClassifier',
            'xgboost.sklearn.XGBRegressor', 'xgboost.sklearn.XGBClassifier'}), None)
        if identity:
            reviewed_wrapper_owners[owner_path] = identity
            for name in STANDARD_METADATA.get(identity, {}):
                params.setdefault(f'{owner_path}__{name}', None)
    for path, value in params.items():
        # Distinguish internal recipe preparation from explicit upstream steps.
        recipe_parameter = (isinstance(params.get('data_models'), Pipeline)
                and path.startswith('data_models__')
                and not path.startswith('data_models__estimator__'))
        if '__' not in path or isinstance(value, (BaseEstimator, Kernel)):
            continue
        owner_path, name = path.rsplit('__', 1)
        owner = params.get(owner_path)
        groups = getattr(owner, 'parameter_groups', {})
        if name in getattr(owner, 'parameter_aliases', {}) or any(name in group['parameters'] for group in groups.values()):
            continue  # Legacy flat aliases are accepted, not duplicated in discovery.
        parent_path, _, group_name = owner_path.rpartition('__')
        parent = params.get(parent_path)
        group = getattr(parent, 'parameter_groups', {}).get(group_name)
        legacy_name = None
        if group:
            legacy_name = next((old for old, nested in group['parameters'].items() if nested == name), None)
            if legacy_name is None:
                continue
        if not isinstance(owner, (BaseEstimator, Kernel)) and owner_path not in reviewed_wrapper_owners:
            continue
        # Composite topology is represented by its actual child estimators.
        if name in ('steps', 'transformers', 'estimators'):
            continue
        cls = type(owner)
        identity = f'{cls.__module__}.{cls.__qualname__}'
        if group:
            cls = type(parent)
            identity = f'{cls.__module__}.{cls.__qualname__}'
        metadata_identity = reviewed_wrapper_owners.get(owner_path, identity)
        metadata_name = legacy_name or name
        metadata = deepcopy(STANDARD_METADATA.get(metadata_identity, {}).get(metadata_name, {}))
        published = getattr(cls, 'tuning_metadata', {}).get(metadata_name, {})
        metadata.update(deepcopy(published))
        kind = metadata.get('kind', '')
        if not kind:
            if isinstance(value, (bool, np.bool_)): kind = 'boolean'
            elif isinstance(value, (int, np.integer)): kind = 'integer'
            elif isinstance(value, (float, np.floating)): kind = 'float'
            elif isinstance(value, str): kind = 'categorical'
        try:
            parameter = inspect.signature(cls.__init__).parameters.get(metadata_name)
            default = parameter.default if parameter else inspect.Parameter.empty
        except (ValueError, TypeError):
            default = inspect.Parameter.empty
        purpose = metadata.get('purpose', 'recommended' if published else 'unreviewed')
        fixed = name in FIXED or callable(value) or metadata.get('fixed', False) or purpose == 'operational'
        if fixed: purpose = 'operational'
        reason = metadata.get('reason') or ('Fixed pipeline or execution configuration; not a model-selection parameter.' if fixed else
            'Published tuning option for this transformer.' if published else 'Not yet reviewed for tuning relevance; missing search values do not imply low importance.')
        if recipe_parameter:
            purpose = 'recipe'
            fixed = True
            reason = ('Internal model-recipe preparation, such as normalization, imputation or string encoding. '
                      'The upstream pipeline is expected to manage this preparation; excluded from automatic tuning.')
        definition = {'kind': kind, 'mode': 'values', 'values': []}
        if kind == 'boolean' and not fixed: definition['values'] = [False, True]
        if 'choices' in metadata: definition['values'] = metadata['choices']
        definition.update(metadata.get('suggested', {}))
        selectors = (group.get('selectors', {group.get('selector'): group.get('values')}) if group else {})
        terms = [{'selector': f'{parent_path}__{selector}', 'values': values,
                  'current': list(getattr(parent, selector)) if isinstance(getattr(parent, selector), tuple)
                  else getattr(parent, selector)} for selector, values in selectors.items()]
        condition = ({'any': terms} if len(terms) > 1 else terms[0]) if terms else None
        if condition:
            reason += ' Applies when ' + ' or '.join(
                f'{selector} includes {", ".join(map(str, values))}' for selector, values in selectors.items()) + '.'
        rows.append({'path': path, 'name': metadata.get('label', metadata_name.replace('_', ' ').capitalize()).replace('Neighbours', 'Neighbors'),
                     'source': owner_path.replace('__', ' › '), 'owner': identity,
                     'current': text_value(value),
                     'default': 'No constructor default' if default is inspect.Parameter.empty else text_value(default),
                     'fixed': bool(fixed), 'metadata': metadata, 'definition': definition,
                     'purpose': purpose, 'reason': reason, 'reference': metadata.get('reference', ''),
                     'condition': condition,
                     'legacy_path': (f'{parent_path}__' + next((old for old, new in getattr(parent, 'parameter_aliases', {}).items()
                                                             if new == legacy_name), legacy_name)) if group else None,
                     'legacy_paths': [f'{parent_path}__{legacy_name}'] if group else [],
                     'importance': metadata.get('importance', 0)})
    # Keep source steps in pipeline order; importance orders their parameters.
    sources = list(dict.fromkeys(row['source'] for row in rows))
    return sorted(rows, key=lambda row: (sources.index(row['source']), list(PURPOSES).index(row['purpose']), -row['importance'], row['name']))


def candidate_prefix(recipe_id):
    return 'candidate_' + hashlib.sha256(recipe_id.encode()).hexdigest()[:16] + '__'


def discover_source(source):
    candidates = source.model_candidates
    if len(candidates) <= 1 or source.selected_model_candidate is not None:
        rows = discover(source.pipeline)
        selected = next((c for c in candidates if c['id'] == source.selected_model_candidate), candidates[0] if candidates else None)
        label = selected['label'] if selected else next(
            (r.method for r in reversed(source.processing_records) if r.card == 'data_models'), '')
        for row in rows:
            if row['source'].startswith('data_models') and label:
                row['display_source'] = label + row['source'].removeprefix('data_models').removeprefix(' › estimator')
        return rows
    rows = [r for r in discover(source.pipeline) if not r['path'].startswith('data_models__')]
    for c in candidates:
        pipeline = source.pipeline_for_training(c['id'])
        for row in discover(pipeline):
            if not row['path'].startswith('data_models__'):
                continue
            row['candidate_id'] = c['id']
            row['parameter_path'] = row['path']
            row['path'] = candidate_prefix(c['id']) + row['path']
            if row.get('condition'):
                for term in condition_terms(row['condition']):
                    term['selector'] = candidate_prefix(c['id']) + term['selector']
            if row.get('legacy_path'):
                row['legacy_path'] = candidate_prefix(c['id']) + row['legacy_path']
            row['legacy_paths'] = [candidate_prefix(c['id']) + p for p in row.get('legacy_paths', [])]
            row['display_source'] = c['label'] + row['source'].removeprefix('data_models').removeprefix(' › estimator')
            rows.append(row)
    return rows


def fingerprint(rows):
    return hashlib.sha256(json.dumps([(r['path'], r['owner'], r['current'], r.get('condition')) for r in rows]).encode()).hexdigest()


def applicable(condition, parameters):
    if not condition:
        return True
    if 'any' in condition:
        return any(applicable(term, parameters) for term in condition['any'])
    value = parameters.get(condition['selector'], condition['current'])
    values = value if isinstance(value, (list, tuple)) else [value]
    return any(v in condition['values'] for v in values)


def condition_terms(condition):
    return condition.get('any', [condition])


def conditional_grids(grid, conditions):
    """Expand selector choices only; retain compact independent child grids."""
    from sklearn.model_selection import ParameterGrid
    selectors = {t['selector'] for c in conditions.values() for t in condition_terms(c)}
    choices = {k: v for k, v in grid.items() if k in selectors}
    branches = []
    for selected in ParameterGrid(choices):
        branch = {k: ([selected[k]] if k in selected else v) for k, v in grid.items()
                  if applicable(conditions.get(k), selected)}
        if branch not in branches:
            branches.append(branch)
    return branches


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


def migrate_drafts(rows, drafts):
    aliases = {row['legacy_path']: row['path'] for row in rows if row.get('legacy_path')}
    aliases.update({path: row['path'] for row in rows for path in row.get('legacy_paths', [])})
    migrated = {aliases.get(path, path): draft for path, draft in drafts.items()}
    # An explicitly saved canonical definition takes precedence over its alias.
    migrated.update({path: draft for path, draft in drafts.items() if path not in aliases})
    return migrated


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
    candidate_ids = list(dict.fromkeys(r['candidate_id'] for r in rows if r.get('candidate_id')))
    from sklearn.model_selection import ParameterGrid
    def count(candidate=None):
        selected = {p: v for p, v in grid.items() if not known[p].get('candidate_id') or known[p]['candidate_id'] == candidate}
        conditions = {p: known[p]['condition'] for p in selected if known[p].get('condition')}
        return len(ParameterGrid(conditional_grids(selected, conditions)))
    return {'fingerprint': fingerprint(rows), 'grid': grid,
            'combinations': sum(count(c) for c in candidate_ids) if candidate_ids else count()}
