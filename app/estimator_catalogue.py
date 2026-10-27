"""Query the static capability catalogue without importing estimator libraries."""
from __future__ import annotations

import json
from copy import deepcopy
from functools import lru_cache
from importlib.metadata import distributions
from pathlib import Path
from model_taxonomy import normalized_outputs, has_prediction_spread

DEFAULT_PATH = Path(__file__).parent / 'data/estimator_capabilities/catalogue.json'


@lru_cache(maxsize=4)
def _read_catalogue(path, modified_ns, size):
    catalogue = json.loads(Path(path).read_text())
    if catalogue.get('schema_version') not in {'1.0.0', '1.1.0', '1.2.0', '1.3.0', '1.4.0'}:
        raise ValueError('Unsupported estimator catalogue schema')
    for section in ('estimators', 'profiles', 'specialists'):
        entries = catalogue.get(section, [])
        if len({entry['id'] for entry in entries}) != len(entries):
            raise ValueError(f'Duplicate identifiers in {section}')
    identifiers = {entry['id'] for entry in catalogue['estimators']}
    if any(p['estimator'] not in identifiers for p in catalogue['profiles']):
        raise ValueError('Profile references an absent estimator')
    links = {entry['id']: entry['superseded_by']['method']
             for entry in catalogue['estimators'] if entry.get('superseded_by')}
    for start in links:
        seen, current = set(), start
        while current in links:
            if current in seen:
                raise ValueError('Cyclic estimator supersession')
            seen.add(current)
            current = links[current]
            if current not in identifiers:
                raise ValueError('Supersession references an absent estimator')
    return catalogue


def load_catalogue(path=DEFAULT_PATH):
    """Cache parsing by file revision, returning an isolated mutable snapshot."""
    path = Path(path).resolve()
    stat = path.stat()
    return deepcopy(_read_catalogue(str(path), stat.st_mtime_ns, stat.st_size))


def installed_versions():
    """Read distribution metadata without importing any estimator modules."""
    return {d.metadata['Name'].lower().replace('_', '-'): d.version
            for d in distributions() if d.metadata['Name']}


def find_profiles(catalogue, *, task=None, required=(), validated_only=True,
                  package_versions=None):
    """Find joint capability profiles, never merge different configurations.

    `package_versions` is an optional deployment-version mapping. Without it,
    matches refer to the recorded catalogue environment, not the current host.
    Unknown capability values do not satisfy a request.
    """
    estimators = {entry['id']: entry for entry in catalogue['estimators']}
    results = []
    for profile in catalogue['profiles']:
        if task and profile['task'] != task:
            continue
        if not profile['version_match']:
            continue
        if validated_only and profile['validation'] != 'passed':
            continue
        if not all(profile['capabilities'].get(name) is True for name in required):
            continue
        entry = estimators.get(profile['estimator'])
        if entry is None:
            continue
        if package_versions is not None:
            expected = profile['inspected_dependencies'] | {entry['package']: entry['package_version']}
            if any(package_versions.get(package) != version for package, version in expected.items()):
                continue
        results.append(profile)
    return results


TASK_ALIASES = {'regression': 'regressor', 'classification': 'classifier'}


def assess_profiles(catalogue, *, task, required=(), structure=(), outputs=(),
                    available_requirements=(), package_versions=None):
    """Return explanatory rows, including excluded and unverified recipes.

    Compatibility, evidence, integration and installation are independent axes.
    No code in JSON is imported or executed. An empty versions mapping means no
    packages are installed; None means deployment availability was not checked.
    """
    entries = {e['id']: e for e in catalogue['estimators']}
    result = []
    for specialist, profiles in ((False, catalogue['profiles']), (True, catalogue.get('specialists', []))):
        for profile in profiles:
            actual = TASK_ALIASES.get(profile['task'], profile['task'])
            if actual != TASK_ALIASES.get(task, task):
                continue
            reasons, unknown = [], []
            for capability in required:
                value = profile['capabilities'].get(capability)
                if value is False:
                    reasons.append(f'{capability}: outside this recipe')
                elif value is not True:
                    unknown.append(f'{capability}: unreviewed')
            for value in structure:
                if value not in profile.get('structure', []):
                    unknown.append(f'{value} structure: unreviewed')
            for value in normalized_outputs(outputs):
                supported = has_prediction_spread(profile) if value == 'prediction_spread' else value in normalized_outputs(profile.get('outputs', []))
                if not supported:
                    unknown.append(f'{value} output: unreviewed')
            missing = sorted(set(profile.get('requirements', [])) - set(available_requirements))
            reasons.extend(f'Requires {name}' for name in missing)
            package = profile['package'] if specialist else entries[profile['estimator']]['package']
            expected = (profile.get('documentation_version') if specialist
                        else entries[profile['estimator']]['package_version'])
            installed = None if package_versions is None else package_versions.get(package)
            availability = ('unchecked' if package_versions is None else 'not_installed' if installed is None
                            else 'version_unreviewed' if expected is None or installed != expected else 'version_matches')
            if not specialist and package_versions is not None and availability == 'version_matches' and any(package_versions.get(k) != v for k, v in profile['inspected_dependencies'].items()):
                availability = 'version_unreviewed'
            verified = not specialist and profile['validation'] == 'passed' and profile['version_match']
            purpose = entries.get(profile['estimator'], {}).get('purpose', 'model_candidate')
            compatibility = 'incompatible' if reasons else 'unknown' if unknown else 'compatible'
            result.append({'id': profile['id'], 'estimator': profile['estimator'], 'task': profile['task'],
                           'purpose': purpose,
                           'superseded_by': deepcopy(entries.get(profile['estimator'], {}).get('superseded_by')),
                           'compatibility': compatibility, 'reasons': reasons + unknown,
                           'evidence': 'probe_passed' if verified else 'documentation_only' if specialist else profile['validation'],
                           'integration': profile.get('integration', 'tabular_recipe'),
                           'availability': availability,
                           'selectable': purpose != 'benchmark_only' and compatibility == 'compatible' and verified and availability == 'version_matches',
                           'input_assessment': input_assessment(profile, entries.get(profile['estimator'], {}), required),
                           'profile': deepcopy(profile)})
    return result


def input_assessment(profile, estimator, required=()):
    """Separate estimator input conditions from the preparation used in a probe.

    Default tags are used only for the numeric estimator behind impute_onehot;
    configured native categorical profiles take precedence over default tags.
    """
    preparation = profile.get('construction', {}).get('preparation')
    capabilities = profile.get('capabilities', {})
    required = set(required)
    needs = []
    conditions = []
    unknown = False
    if preparation == 'impute_onehot':
        tags = estimator.get('default_tags', {}).get('input_tags', {})
        conditions.append('Numeric representation of categorical predictors')
        if 'categorical_predictors' in required:
            needs.append('Represent categorical predictors numerically')
        if tags.get('allow_nan') is False:
            conditions.append('No missing predictor values')
            if required & {'missing_numeric_predictors', 'missing_categorical_predictors'}:
                needs.append('Resolve missing predictor values')
        elif tags.get('allow_nan') is not True:
            conditions.append('Native missing-value support unreviewed')
            unknown = True
    elif preparation in {'numeric', 'categorical_complete', 'pandas_categories', 'catboost_strings'}:
        conditions.append({'numeric': 'Numeric predictors',
                           'categorical_complete': 'Numeric or categorical predictors',
                           'pandas_categories': 'Categorical predictors declared as pandas categories',
                           'catboost_strings': 'Categorical predictors supplied as strings with categorical columns identified'}[preparation])
        if preparation == 'numeric' and 'categorical_predictors' in required:
            needs.append('Represent categorical predictors numerically')
        if preparation in {'pandas_categories', 'catboost_strings'} and 'categorical_predictors' in required:
            needs.append('Check categorical representation and estimator configuration')
        for kind in ('numeric', 'categorical'):
            key = f'missing_{kind}_predictors'
            support = capabilities.get(key)
            # CatBoost's tested recipe fills categorical missing values itself.
            if preparation == 'catboost_strings' and kind == 'categorical':
                support = False
            if support is False:
                conditions.append(f'No missing {kind} predictor values')
                if key in required:
                    needs.append(f'Resolve missing {kind} predictor values')
            elif support is None and key in required:
                unknown = True
    else:
        return {'requirements': 'Not reviewed; see documented conditions',
                'readiness': 'Unreviewed', 'actions': []}
    return {'requirements': '; '.join(conditions),
            'readiness': 'Requires preparation' if needs else 'Unreviewed' if unknown else 'Meets reviewed input requirements',
            'actions': needs}


def requirements_from_proxy(source, *, use_weights=False):
    """Summarise current model inputs; a preview is not a fold-fit guarantee."""
    import pandas as pd
    context = source.modelling_context
    frame = source.frame
    required = set()
    unsupported = []
    numeric_count = 0
    for column in source.model_predictors:
        values = frame[column]
        numeric = pd.api.types.is_numeric_dtype(values.dtype) and not pd.api.types.is_bool_dtype(values.dtype)
        numeric_count += int(numeric)
        categorical = (isinstance(values.dtype, pd.CategoricalDtype)
                       or pd.api.types.is_bool_dtype(values.dtype)
                       or pd.api.types.is_string_dtype(values.dropna()))
        if not numeric and not categorical:
            unsupported.append(column)
            continue
        kind = 'numeric' if numeric else 'categorical'
        required.add(f'{kind}_predictors')
        if values.isna().any():
            required.add(f'missing_{kind}_predictors')
    if use_weights:
        required.add('sample_weight_fit')
    if context.task == 'survival':
        required.add(f'{context.censoring}_censoring')
    count = len(source.model_predictors)
    predictor_kind = ('' if not count else 'numeric' if numeric_count == count
                      else 'non-numeric' if not numeric_count else 'mixed')
    plan = source.evaluation_plan
    return {'task': context.resolved_task(source), 'required': sorted(required),
            'unsupported_columns': unsupported, 'rows': len(frame), 'predictors': count, 'predictor_kind': predictor_kind,
            'evaluation_ready': bool(plan and plan.ready),
            'prediction_objective': plan.recipe.get('objective') if plan else None,
            'qualification': 'Preview requirements only; fold-fitted preparation and evaluation need separate checks.'}


def coverage(catalogue):
    """Expose reviewed breadth without suggesting exhaustive ecosystem coverage."""
    tasks = sorted({p['task'] for p in catalogue['profiles'] + catalogue.get('specialists', [])})
    return {task: {'probed_recipes': sum(p['task'] == task and p['validation'] == 'passed'
                                       for p in catalogue['profiles']),
                   'documented_recipes': sum(p['task'] == task for p in catalogue.get('specialists', [])),
                   'exhaustive': False} for task in tasks}
