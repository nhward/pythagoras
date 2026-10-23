"""Exercise reviewed recipes on small synthetic datasets; not a benchmark.

Does not execute code strings from JSON. Only trusted reviewed class paths and
parameters are imported. Reports outcomes in the generated catalogue.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import sys
import time
import traceback
import warnings
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / 'app/data/estimator_capabilities'


def construct(profile, X):
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    module, cls = profile['estimator'].rsplit('.', 1)
    constructor = getattr(importlib.import_module(module), cls)
    params = dict(profile['construction']['parameters'])
    # Resource-only changes keep CI fast; report exactly what was exercised.
    import inspect
    signature = inspect.signature(constructor).parameters
    overrides = {}
    for name, value in [('n_estimators', 8), ('iterations', 8), ('max_iter', 20)]:
        if name in signature and name not in params:
            overrides[name] = value
    if module == 'catboost':
        overrides = {'iterations': 8}
    # XGB hides n_estimators in **kwargs.
    if module == 'xgboost':
        overrides['n_estimators'] = 8
    params.update(overrides)
    model = constructor(**params)
    preparation = profile['construction']['preparation']
    X = X.copy()
    cat_cols = list(X.select_dtypes(include=['category', 'object', 'string', 'bool']).columns)
    fit_params = dict(profile['recipe']['fit_parameters'])
    if preparation == 'catboost_strings':
        for col in cat_cols:
            # A token absent from observed values prevents conflating missing
            # values with a genuine category named like the sentinel.
            values = X[col].astype('string')
            token = '__missing__'
            observed = set(values.dropna())
            while token in observed:
                token += '_'
            X[col] = values.fillna(token).astype(str)
        fit_params['cat_features'] = cat_cols
    elif preparation == 'pandas_categories':
        for col in cat_cols:
            X[col] = X[col].astype('category')
    elif preparation == 'categorical_complete':
        for col in cat_cols:
            X[col] = X[col].astype(object)
    elif preparation == 'impute_onehot':
        numeric = [col for col in X if col not in cat_cols]
        # The synthetic data use np.nan consistently for missing categories.
        for col in cat_cols:
            X[col] = X[col].astype(object).where(X[col].notna(), np.nan)
        preprocessor = ColumnTransformer([
            ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')),
                                  ('scale', StandardScaler())]), numeric),
            ('categorical', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                                      ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols),
        ])
        model = Pipeline([('prepare', preprocessor), ('model', model)])
    return model, X, fit_params, overrides


def probe(profile):
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from sklearn.base import clone

    rng = np.random.default_rng(1729)
    n = 90
    X = pd.DataFrame({'value': rng.normal(size=n), 'other': rng.uniform(size=n)})
    task = profile['task']
    y = (X.value.to_numpy() > 0).astype(int) if task == 'classifier' else X.value.to_numpy() * 2 + rng.normal(scale=.1, size=n)
    if profile['capabilities']['categorical_predictors']:
        X['group'] = pd.Categorical(np.resize(['north', 'south', 'west'], n))
    if profile['capabilities']['missing_numeric_predictors']:
        X.loc[::11, 'value'] = np.nan
    if 'group' in X and profile['capabilities']['missing_categorical_predictors']:
        X.loc[::13, 'group'] = np.nan
    model, prepared, fit_params, overrides = construct(profile, X)
    weight_arg = profile['recipe']['weight_argument']
    if weight_arg:
        fit_params[weight_arg] = np.linspace(.2, 2, n)
    operations = []
    start = time.monotonic()
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter('always')
        if task == 'transformer':
            transformed = model.fit_transform(prepared, y)
            assert transformed.shape[0] == n
            operations.append('fit_transform_row_count')
        else:
            model.fit(prepared, y, **fit_params)
            prediction = np.asarray(model.predict(prepared))
            assert len(prediction) == n and np.isfinite(prediction).all()
            operations.append('fit_predict_finite')
            if weight_arg:
                operations.append('fit_accepts_nonuniform_sample_weights')
            if profile['capabilities']['categorical_predictors']:
                unseen = prepared.iloc[:3].copy()
                if profile['construction']['preparation'] == 'pandas_categories':
                    unseen['group'] = pd.Categorical(['unseen', None, 'north'])
                elif profile['construction']['preparation'] == 'catboost_strings':
                    unseen['group'] = ['unseen', '__missing__', 'north']
                else:
                    unseen['group'] = ['unseen', np.nan, 'north']
                assert np.isfinite(model.predict(unseen)).all()
                operations.append('predict_unseen_category')
            if profile['capabilities']['sparse_input'] is True:
                sparse_model = clone(model)
                sparse_X = sparse.csr_matrix(prepared.to_numpy())
                sparse_model.fit(sparse_X, y, **fit_params)
                assert np.isfinite(sparse_model.predict(sparse_X)).all()
                operations.append('sparse_fit_predict')
            if profile['capabilities']['incremental_training']:
                incremental = clone(model)
                for start_row in range(0, n, 30):
                    kw = {'classes': np.unique(y)} if task == 'classifier' else {}
                    if weight_arg:
                        kw['sample_weight'] = fit_params[weight_arg][start_row:start_row+30]
                    incremental.partial_fit(prepared.iloc[start_row:start_row+30], y[start_row:start_row+30], **kw)
                assert np.isfinite(incremental.predict(prepared)).all()
                operations.append('partial_fit_three_batches')
    return {
        'checked_at': datetime.now(UTC).isoformat(),
        'operations': operations, 'resource_parameter_overrides': overrides,
        'warnings': sorted({str(w.message) for w in emitted}),
        'seconds': round(time.monotonic() - start, 4),
        'limits': 'Small synthetic compatibility check, not predictive-quality/scalability benchmarking or proof of every parameter combination. Accepting weights alone does not establish their statistical semantics.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extra-path', action='append', default=[])
    parser.add_argument('--catalogue', type=Path, default=DIRECTORY / 'catalogue.json')
    args = parser.parse_args()
    for path in reversed(args.extra_path):
        sys.path.insert(0, str(Path(path).resolve()))
    catalogue = json.loads(args.catalogue.read_text())
    failures = []
    from threadpoolctl import threadpool_limits
    entries = {entry['id']: entry for entry in catalogue['estimators']}
    for profile in catalogue['profiles']:
        entry = entries.get(profile['estimator'])
        actual = {}
        expected = dict(profile['reviewed_dependencies'])
        if entry:
            expected[entry['package']] = profile['reviewed_package_version']
        for package in expected:
            try:
                actual[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                actual[package] = None
        profile['inspected_dependencies'] = {package: actual[package]
                                              for package in profile['reviewed_dependencies']}
        profile['inspected_package_version'] = actual.get(entry['package']) if entry else None
        profile['version_match'] = bool(entry and actual == expected)
        profile.pop('validation_details', None)
        if profile['inspected_package_version'] is None:
            profile['validation'] = 'unavailable'
            continue
        if not profile['version_match']:
            profile['validation'] = 'version_mismatch'
            continue
        try:
            with threadpool_limits(limits=1):
                profile['validation_details'] = probe(profile)
            profile['validation'] = 'passed'
            print('PASS', profile['id'], flush=True)
        except Exception as error:  # noqa: BLE001
            profile['validation'] = 'failed'
            profile['validation_details'] = {'error': f'{type(error).__name__}: {error}', 'traceback': traceback.format_exc()}
            failures.append(profile['id'])
            print('FAIL', profile['id'], profile['validation_details']['error'], flush=True)
    import jsonschema
    jsonschema.validate(catalogue, json.loads((DIRECTORY / 'schema.json').read_text()))
    args.catalogue.write_text(json.dumps(catalogue, indent=2, allow_nan=False) + '\n')
    print(f'{len(failures)} failed profiles', flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
