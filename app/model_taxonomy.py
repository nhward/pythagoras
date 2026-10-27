"""Static, evidence-labelled model descriptions; no estimator imports or fitting."""
import json
from copy import deepcopy
from pathlib import Path

DIRECTORY = Path(__file__).parent / 'data/estimator_capabilities'
FIELDS = ('core_mechanism', 'composition', 'fitting_criterion', 'learning_strategy', 'transparency')
LABELS = {
    'core_mechanism': 'Core mechanism', 
    'composition': 'Composition',
    'fitting_criterion': 'Fitting criterion',
    'learning_strategy': 'Learning strategy',
    'transparency': 'Transparency', 
    'outputs': 'Prediction outputs'
}
VALUE_LABELS = {
    'generalized_linear': 'Gen. linear', 
    'single': 'Single model',
    'not_applicable': 'Not applicable', 
    'depends_on_configuration': 'Depends on configuration',
    'unreviewed': 'Unreviewed', 'deferred': 'Stage 2',
    'usually_transparent': 'Usually transparent', 
    'usually_opaque': 'Usually opaque'
}


def attribute(values=(), *, status='reviewed', note=''):
    return {'values': list(values), 'status': status, 'note': note}


def normalized_outputs(outputs):
    return list(dict.fromkeys('class_label' if value == 'class_prediction' else value for value in outputs))


SPREAD_OUTPUTS = frozenset({'prediction_interval', 'predictive_standard_deviation', 'predictive_covariance', 'predictive_distribution', 'predictive_samples', 'predictive_quantiles'})
POINT_OUTPUTS = frozenset({'point_prediction', 'conditional_mean', 'fitted_mean', 'conditional_quantile'})


def has_prediction_spread(profile):
    """Only documented prediction-specific outputs, not ensemble or Bayesian labels."""
    return bool(SPREAD_OUTPUTS.intersection(profile.get('outputs', [])))


def load_registry(path=DIRECTORY / 'taxonomy.json'):
    registry = json.loads(Path(path).read_text())
    methods = {}
    for group in registry['groups']:
        for identifier in group['methods']:
            if identifier in methods:
                raise ValueError(f'Duplicate taxonomy method: {identifier}')
            methods[identifier] = group
    return registry, methods


def describe(identifier, kinds, methods):
    """An explicit allowlist, never classification by an estimator's name."""
    non_predictive = bool(kinds) and set(kinds) <= {'transformer', 'sampler'}
    status = 'not_applicable' if non_predictive else 'unreviewed'
    note = ('Predictive-model taxonomy does not classify standalone preparation steps.' if non_predictive
            else 'No reviewed classification for this method; absence is not a negative claim.')
    result = {key: attribute(status=status, note=note) for key in FIELDS}
    result['capacity_growth'] = attribute(status='deferred', note='Reserved for stage 2; not assessed.')
    result['sources'] = []
    result['reviewed_on'] = None
    if identifier in methods:
        group = methods[identifier]
        result.update(deepcopy(group['attributes']))
        result['sources'] = list(group['sources'])
        result['reviewed_on'] = group['reviewed_on']
    return result


def annotate_catalogue(catalogue, registry_path=DIRECTORY / 'taxonomy.json'):
    """Refresh descriptions only, preserving inspection and runtime-probe evidence."""
    registry, methods = load_registry(registry_path)
    entries = {entry['id']: entry for entry in catalogue['estimators']}
    for entry in entries.values():
        entry['taxonomy'] = describe(entry['id'], entry['kinds'], methods)
        # A method's API is not a promise for every configuration. Outputs live on recipes.
        entry['outputs'] = []
        entry['output_review'] = {
            'status': 'not_applicable' if entry['taxonomy']['core_mechanism']['status'] == 'not_applicable' else 'depends_on_configuration',
            'note': 'Consult an individual recipe; API method presence alone does not establish output support.',
            'sources': [],
        }
    for profile in catalogue['profiles'] + catalogue.get('specialists', []):
        entry = entries.get(profile['estimator'], {})
        taxonomy = describe(profile['estimator'], entry.get('kinds', [profile['task']]), methods)
        override = registry['recipes'].get(profile['id'], {})
        stale = bool(override) and (
            override['configuration'] != profile.get('construction', profile.get('parameters', {}))
            or override['package_version'] != profile.get('reviewed_package_version', profile.get('documentation_version')))
        if stale:
            override = {}
        taxonomy.update(deepcopy(override.get('attributes', {})))
        taxonomy['sources'] = list(dict.fromkeys(taxonomy['sources'] + override.get('sources', [])))
        if override:
            taxonomy['reviewed_on'] = registry['reviewed_on']
        profile['taxonomy'] = taxonomy
        profile['outputs'] = normalized_outputs(override.get('outputs', profile.get('outputs', [])))
        profile['output_review'] = {
            'status': 'reviewed' if profile['outputs'] else 'not_applicable' if profile['task'] in {'transformer', 'sampler'} else 'unreviewed',
            'note': override.get('output_note', 'Recorded recipe outputs; not an exhaustive API inventory. Probability estimates do not establish calibration. Runtime evidence remains separate.'),
            'sources': list(dict.fromkeys(profile.get('sources', []) + override.get('sources', []))),
        }
        if stale:
            profile['outputs'] = []
            profile['output_review'].update(status='unreviewed', note='Recipe configuration or reviewed version changed; review outputs and taxonomy refinements again.')
    catalogue['semantics']['taxonomy'] = registry['policy']
    return catalogue


def display_value(value):
    return VALUE_LABELS.get(value, value.replace('_', ' ').capitalize())


def description_label(profile, field):
    if field == 'outputs':
        values = normalized_outputs(profile.get('outputs', []))
        status = profile.get('output_review', {}).get('status', 'unreviewed')
        if profile.get('task') in {'regression', 'regressor'} and POINT_OUTPUTS.intersection(values):
            return 'Point prediction & spread' if has_prediction_spread(profile) else 'Point prediction'
    else:
        item = profile.get('taxonomy', {}).get(field, {})
        values, status = item.get('values', []), item.get('status', 'unreviewed')
    label = ', '.join(display_value(value) for value in values)
    if status == 'depends_on_configuration':
        return (label + ' — ' if label else '') + 'Depends on configuration'
    return label or display_value(status)


def group_description(rows, field):
    """Compare all displayed recipes, not just the selected summary recipe."""
    labels = {description_label(row['profile'], field) for row in rows}
    return next(iter(labels)) if len(labels) == 1 else 'Varies by recipe'
