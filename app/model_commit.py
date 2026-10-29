"""Construct unfitted catalogue recipes without importing unsupported adapters."""
from importlib import import_module

import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def prepare_predictors(frame, *, columns, preparation):
    result = frame.loc[:, columns].copy()
    categorical = result.select_dtypes(include=['category', 'object', 'string', 'bool']).columns
    for column in categorical:
        if preparation == 'pandas_categories':
            result[column] = result[column].astype('category')
        elif preparation in ('categorical_complete', 'impute_onehot'):
            result[column] = result[column].astype(object).where(result[column].notna(), np.nan)
    return result


def commit_problem(row):
    if not row.get('selectable'):
        return 'This recipe is not validated and installed for the current requirements.'
    construction = row['profile']['construction']
    if construction.get('wrapper') not in (None, 'sklearn.pipeline.Pipeline') or construction.get('preparation') not in (
        'none', 'numeric', 'numeric_complete', 'pandas_categories', 'categorical_complete', 'impute_onehot'
    ):
        return 'This recipe needs a construction adapter before it can be committed.'
    if row['profile'].get('recipe', {}).get('fit_parameters'):
        return 'This recipe needs a fit-parameter adapter before it can be committed.'
    return ''


def build_model(row, predictors):
    problem = commit_problem(row)
    if problem:
        raise ValueError(problem)
    profile = row['profile']
    module, name = row['estimator'].rsplit('.', 1)
    estimator = getattr(import_module(module), name)(**profile['construction']['parameters'])
    preparation = profile['construction']['preparation']
    steps = [('predictors', FunctionTransformer(prepare_predictors,
        kw_args={'columns': list(predictors), 'preparation': preparation}))]
    if preparation == 'impute_onehot':
        steps.append(('prepare', ColumnTransformer([
            ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')),
                                  ('scale', StandardScaler())]), make_column_selector(dtype_include=np.number)),
            ('categorical', Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]),
                make_column_selector(dtype_exclude=np.number)),
        ])))
    steps.append(('estimator', estimator))
    return Pipeline(steps)
