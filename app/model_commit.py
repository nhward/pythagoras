"""Construct unfitted catalogue recipes without importing unsupported adapters."""
from importlib import import_module

import numpy as np
import pandas as pd
from evaluation_plan import level_key
from sklearn.base import clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def configured_pipeline(template, parameters):
    """Apply recorded trial choices and preserve cloneable native defaults.

    CatBoost.set_params retains explicit None values, while its constructor drops
    them. Reconstruct those unfitted estimators with the same parameters so later
    sklearn clones and fit-cache identities agree with constructor semantics.
    """
    result = clone(template).set_params(**parameters)
    for path, estimator in result.get_params(deep=True).items():
        if not any(cls.__module__ == 'catboost.core' for cls in type(estimator).__mro__):
            continue
        settings = estimator.get_params(deep=False)
        if any(value is None for value in settings.values()):
            result.set_params(**{path: type(estimator)(**settings)})
    return result


def prepare_predictors(frame, *, columns, preparation, selected_schema=False,
                       encoded_schema=None, encoded_columns=None):
    if selected_schema:
        columns = [c for c in columns if c in frame]
        if not columns:
            raise ValueError('The fitted selection retained no model predictors')
    result = frame.loc[:, columns].copy()
    categorical = result.select_dtypes(include=['category', 'object', 'string', 'bool']).columns
    for column in categorical:
        if preparation == 'catboost_strings':
            # All actual values get a typed prefix, so none can collide with the missing token.
            result[column] = result[column].astype(object).map(
                lambda value: '__pythagoras_missing__' if pd.isna(value) else level_key(value))
        elif preparation == 'pandas_categories':
            result[column] = result[column].astype('category')
        elif preparation in ('categorical_complete', 'impute_onehot'):
            result[column] = result[column].astype(object).where(result[column].notna(), np.nan)
    return np.asarray(result, dtype=float) if preparation == 'numeric_array' else result


def commit_problem(row):
    if row.get('resampling_exclusion'):
        return row['resampling_exclusion']
    if not row.get('selectable'):
        return 'This recipe is not validated and installed for the current requirements.'
    construction = row['profile']['construction']
    if construction.get('wrapper') not in (None, 'sklearn.pipeline.Pipeline', 'sklearn.compose.TransformedTargetRegressor') or construction.get('preparation') not in (
        'none', 'numeric', 'numeric_complete', 'numeric_array', 'pandas_categories', 'categorical_complete', 'impute_onehot', 'catboost_strings'
    ):
        return 'This recipe needs a construction adapter before it can be committed.'
    if construction.get('preparation') == 'catboost_strings' and row['estimator'] not in {
            'catboost.CatBoostRegressor', 'catboost.CatBoostClassifier'}:
        return 'Categorical CatBoost preparation requires a supported CatBoost estimator.'
    if row['profile'].get('recipe', {}).get('fit_parameters'):
        return 'This recipe needs a fit-parameter adapter before it can be committed.'
    return ''


def build_model(row, predictors):
    selected_schema = False
    if hasattr(predictors, 'model_predictors'):
        source = predictors
        predictors = list(source.model_predictors)
        for _, step in (() if source.pipeline is None else source.pipeline.steps):
            if getattr(step, 'selection_step', False):
                selected_schema = True
                predictors.extend(step.predictors)
        predictors = list(dict.fromkeys(predictors))
    problem = commit_problem(row)
    if problem:
        raise ValueError(problem)
    profile = row['profile']
    module, name = row['estimator'].rsplit('.', 1)
    preparation = profile['construction']['preparation']
    if preparation == 'catboost_strings':
        from catboost_model import PreparedCatBoostClassifier, PreparedCatBoostRegressor
        factory = PreparedCatBoostClassifier if name == 'CatBoostClassifier' else PreparedCatBoostRegressor
    else:
        factory = getattr(import_module(module), name)
    from kernel_search import kernel_parameters
    estimator = factory(**kernel_parameters(row['estimator'], profile['construction']['parameters']))
    if profile['construction'].get('wrapper') == 'sklearn.compose.TransformedTargetRegressor':
        estimator = TransformedTargetRegressor(regressor=estimator)
    steps = [('predictors', FunctionTransformer(prepare_predictors,
        kw_args={'columns': list(predictors), 'preparation': preparation, 'selected_schema': selected_schema}))]
    if preparation == 'impute_onehot':
        steps.append(('prepare', ColumnTransformer([
            ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')),
                                  ('scale', StandardScaler())]), make_column_selector(dtype_include=np.number)),
            ('categorical', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]),
                make_column_selector(dtype_exclude=np.number)),
        ])))
        if row['estimator'] == 'cubist.Cubist':
            # Cubist assigns feature names even to array inputs during fit.
            # Preserve names at prediction too, rather than suppressing validation.
            steps[-1][1].set_output(transform='pandas')
    steps.append(('estimator', estimator))
    return Pipeline(steps)


def with_selection_schema(estimator, candidates):
    """Clone a catalogue model and let its fitted selector determine predictors."""
    from sklearn.base import clone
    result = clone(estimator)
    if not isinstance(result, Pipeline):
        return result
    preparation = result.named_steps.get('predictors')
    if isinstance(preparation, FunctionTransformer) and preparation.func is prepare_predictors:
        options = dict(preparation.kw_args)
        options['columns'] = list(dict.fromkeys([*options['columns'], *candidates]))
        if options.get('encoded_columns') is not None:
            options['encoded_columns'] = list(dict.fromkeys([*options['encoded_columns'], *candidates]))
        options['selected_schema'] = True
        preparation.set_params(kw_args=options)
    return result


def with_encoding_schema(estimator, source):
    """Record approved encoder lineage without fitting on the preview data."""
    if not isinstance(estimator, Pipeline):
        return estimator
    selector = estimator.named_steps.get('predictors')
    if not isinstance(selector, FunctionTransformer) or selector.func is not prepare_predictors:
        return estimator
    schema = {}
    predictors = set(source.model_predictors)
    for _, step in (() if source.pipeline is None else source.pipeline.steps):
        if getattr(step, 'selection_step', False):
            predictors.update(step.predictors)
    for record in source.processing_records:
        groups = record.parameters.get('predictor_output_groups')
        if groups is not None:
            schema[record.card] = {
                'groups': dict(groups),
                'enabled': [column for column, names in groups.items()
                            if not names or any(name in predictors for name in names)],
                'removed': [column for column in groups if column not in source.frame],
                'excluded': [name for names in groups.values() for name in names if name not in predictors]}
    if schema:
        selector.set_params(kw_args=dict(selector.kw_args, encoded_schema=schema,
            encoded_columns=selector.kw_args.get('encoded_columns', selector.kw_args['columns'])))
    return estimator


def bind_encoded_predictors(estimator, steps, frame):
    """Replace preview indicators with exact outputs of the fitted fold encoders."""
    if not isinstance(estimator, Pipeline):
        return
    selector = estimator.named_steps.get('predictors')
    if not isinstance(selector, FunctionTransformer) or selector.func is not prepare_predictors:
        return
    options = dict(selector.kw_args)
    schema = options.get('encoded_schema', {})
    if not schema:
        return
    columns = list(options['encoded_columns'])
    fitted = dict(steps)
    for step, definition in schema.items():
        groups = definition['groups']
        actual = fitted[step].predictor_output_groups_
        preview = {name for names in groups.values() for name in names}
        columns = [name for name in columns if name not in preview and name not in definition['removed']]
        excluded = set(definition['excluded'])
        columns.extend(name for source in definition['enabled'] for name in actual.get(source, [])
                       if name in frame and name not in excluded)
    # Follow the fold's DataFrame order, retaining missing fixed inputs so that
    # ordinary schema errors are still reported rather than silently ignored.
    wanted = set(columns)
    options['columns'] = [name for name in frame if name in wanted] + [name for name in columns if name not in frame]
    selector.set_params(kw_args=options)
