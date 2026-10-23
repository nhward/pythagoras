"""Build a versioned, evidence-labelled estimator inventory; never fits models.

Optional packages can be supplied with --extra-path without installing them in
Pythagoras. Curated achievable configurations live in profiles.json.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import inspect
import json
import platform
import sys
import warnings
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'app/data/estimator_capabilities'
PACKAGES = {
    'sklearn': ('scikit-learn', None),
    'imblearn': ('imbalanced-learn', ['ensemble', 'over_sampling', 'under_sampling', 'combine', 'pipeline', 'FunctionSampler']),
    'feature_engine': ('feature-engine', ['imputation', 'encoding', 'discretisation', 'selection', 'creation', 'transformation', 'outliers', 'wrappers', 'datetime', 'timeseries.forecasting', 'preprocessing', 'pipeline']),
    'mlxtend': ('mlxtend', ['classifier', 'regressor', 'feature_selection', 'preprocessing']),
    'skrub': ('skrub', ['']),
    'xgboost': ('xgboost', ['']),
    'lightgbm': ('lightgbm', ['']),
    'catboost': ('catboost', ['']),
}


def safe_signature(obj):
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return None


def discover(root, modules, issues):
    if root == 'sklearn':
        from sklearn.utils.discovery import all_estimators
        return {f'{cls.__module__.split("._")[0]}.{name}': cls for name, cls in all_estimators()}
    found = {}
    for suffix in modules:
        module_name = root + ('.' + suffix if suffix else '')
        try:
            module = importlib.import_module(module_name)
        except Exception as error:  # noqa: BLE001
            # FunctionSampler is exported as a class, not a submodule.
            if suffix == 'FunctionSampler':
                found['imblearn.FunctionSampler'] = importlib.import_module(root).FunctionSampler
                continue
            issues.append({'module': module_name, 'error': f'{type(error).__name__}: {error}'})
            continue
        for name, cls in inspect.getmembers(module, inspect.isclass):
            if (not name.startswith('_') and cls.__module__.startswith(root + '.')
                    and callable(getattr(cls, 'fit', None))
                    and callable(getattr(cls, 'get_params', None))):
                found[f'{module_name}.{name}'] = cls
    return found


def describe(identifier, cls, distribution, version):
    from sklearn.utils import get_tags
    constructor = inspect.signature(cls)
    required = [name for name, p in constructor.parameters.items()
                if p.default is inspect.Parameter.empty
                and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
    methods = {name: safe_signature(getattr(cls, name)) for name in
               ('fit', 'predict', 'predict_proba', 'decision_function', 'transform',
                'fit_resample', 'partial_fit', 'score') if callable(getattr(cls, name, None))}
    tags, error = None, None
    if not required:
        try:
            model = cls()
            # Do not present sklearn fallback tags as a third-party declaration.
            if callable(getattr(model, '__sklearn_tags__', None)):
                raw = get_tags(model)
                tags = asdict(raw) if is_dataclass(raw) else None
            else:
                error = 'No modern __sklearn_tags__ implementation; capability unknown.'
        except Exception as exc:  # noqa: BLE001
            error = f'{type(exc).__name__}: {exc}'
    else:
        error = 'Constructor requires arguments; tags depend on supplied configuration.'
    kinds = []
    kind = (tags or {}).get('estimator_type') or getattr(cls, '_estimator_type', None)
    if kind:
        kinds.append(kind)
    for method, role in [('transform', 'transformer'), ('fit_resample', 'sampler')]:
        if method in methods and role not in kinds:
            kinds.append(role)
    # Legacy mlxtend estimators do not consistently implement sklearn tags.
    if not kinds and identifier.startswith('mlxtend.classifier.'):
        kinds.append('classifier')
    if not kinds and identifier.startswith('mlxtend.regressor.'):
        kinds.append('regressor')
    fit = inspect.signature(cls.fit)
    explicit_weight = 'sample_weight' in fit.parameters
    return {
        'id': identifier, 'implementation': f'{cls.__module__}.{cls.__name__}',
        'package': distribution, 'package_version': version, 'kinds': kinds,
        'constructor': {'signature': str(constructor), 'required_arguments': required},
        'default_tags': tags, 'tag_error': error,
        'tag_evidence': 'installed_instance_introspection' if tags is not None else 'unavailable',
        'methods': methods,
        'weighting': {
            'fit_sample_weight': 'explicit_parameter' if explicit_weight else 'not_explicit_unknown',
            'class_weight_explicit_parameter': 'class_weight' in constructor.parameters,
            'importance_weight_interpretation': 'Requires reviewed profile; a signature is not proof that weights affect fitting.',
            'rebalance_weight_interpretation': 'Per-row rebalance weights may use sample_weight where verified; class weights and resampling are separate mechanisms.',
            'prediction_weights': 'Not inferred from score(sample_weight). Training and evaluation weights are separate.',
        },
        'incremental_api_present': 'partial_fit' in methods,
        'achievable_profile_ids': [],
        'unreviewed_capabilities': ['Capabilities outside attached reviewed profiles are unknown, not unsupported.'],
    }


def build(extra_paths=()):
    for path in reversed(extra_paths):
        sys.path.insert(0, str(Path(path).resolve()))
    profiles = json.loads((DIRECTORY / 'profiles.json').read_text())
    entries, packages, issues = {}, {}, []
    for root, (distribution, modules) in PACKAGES.items():
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = {'version': None, 'status': 'not_installed'}
            continue
        packages[distribution] = {'version': version, 'status': 'inspected', 'module_scope': modules or ['all_estimators']}
        try:
            with warnings.catch_warnings(record=True):
                classes = discover(root, modules, issues)
        except Exception as exc:  # noqa: BLE001
            issues.append({'module': root, 'error': str(exc)})
            packages[distribution]['status'] = 'import_failed'
            continue
        for identifier, cls in classes.items():
            try:
                with warnings.catch_warnings(record=True):
                    entries[identifier] = describe(identifier, cls, distribution, version)
            except Exception as exc:  # noqa: BLE001
                issues.append({'module': identifier, 'error': str(exc)})
    # Resolve profile public aliases even when discovery chooses a different alias.
    for profile in profiles:
        identifier = profile['estimator']
        if identifier not in entries:
            root = identifier.split('.')[0]
            distribution = PACKAGES[root][0]
            version = packages[distribution]['version']
            if version is not None:
                try:
                    module, name = identifier.rsplit('.', 1)
                    cls = getattr(importlib.import_module(module), name)
                    entries[identifier] = describe(identifier, cls, distribution, version)
                except Exception as exc:  # noqa: BLE001
                    issues.append({'module': identifier, 'error': str(exc)})
        entry = entries.get(identifier)
        profile['inspected_package_version'] = entry['package_version'] if entry else None
        inspected = {}
        for dependency in profile['reviewed_dependencies']:
            try:
                inspected[dependency] = importlib.metadata.version(dependency)
            except importlib.metadata.PackageNotFoundError:
                inspected[dependency] = None
        profile['inspected_dependencies'] = inspected
        profile['version_match'] = bool(entry and entry['package_version'] == profile['reviewed_package_version']
                                       and inspected == profile['reviewed_dependencies'])
        profile['validation'] = 'not_run'
        if entry:
            entry['achievable_profile_ids'].append(profile['id'])
    return {
        '$schema': './schema.json', 'schema_version': '1.1.0', 'catalogue_version': '1.1.0',
        'generated_at': datetime.now(UTC).isoformat(), 'python_version': platform.python_version(),
        'scope': 'Public estimator classes in the explicitly listed modules; not every sklearn-compatible package or every possible configuration.',
        'packages': packages, 'discovery_issues': issues,
        'semantics': {
            'unknown': 'No verified claim. Never interpret as false or infer compatibility from absence.',
            'default_tags': 'Unfitted default-instance declarations only, not the achievable capability envelope.',
            'profiles': 'Joint configurations: capabilities within one profile are achievable together under its conditions. Do not combine guarantees from different profiles.',
            'versions': 'Only version_match profiles may be selected automatically. Rebuild and revalidate after dependency updates.',
            'big_data': {'many_rows': 'Planning scenario: >=100000 rows', 'many_features': 'Planning scenario: >=1000 dense features or >=100000 sparse features', 'out_of_core': 'Training data exceeds available RAM', 'qualification': 'Heuristics, not hardware-independent limits or measured performance. See per-profile assumptions.'},
            'imbalance': 'Mechanisms, not a claim of better-than-most performance. Validate on stratified/group/time-appropriate held-out data and a task-specific metric.',
            'weights': 'Importance weights represent observation relevance; rebalance weights change class contribution. Both may enter sample_weight, but avoid inadvertently multiplying equivalent balancing schemes. Evaluation weights are a separate choice.',
        },
        'estimators': sorted(entries.values(), key=lambda item: item['id']),
        'profiles': profiles,
        'specialists': json.loads((DIRECTORY / 'specialists.json').read_text()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extra-path', action='append', default=[])
    parser.add_argument('--output', type=Path, default=DIRECTORY / 'catalogue.json')
    args = parser.parse_args()
    catalogue = build(args.extra_path)
    import jsonschema
    jsonschema.validate(catalogue, json.loads((DIRECTORY / 'schema.json').read_text()))
    args.output.write_text(json.dumps(catalogue, indent=2, allow_nan=False) + '\n')
    print(f"Wrote {len(catalogue['estimators'])} estimators and {len(catalogue['profiles'])} profiles to {args.output}")
    if catalogue['discovery_issues']:
        print(json.dumps(catalogue['discovery_issues'], indent=2))


if __name__ == '__main__':
    main()
