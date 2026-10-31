"""Fixed, human-reviewed missing-value rules; fit never discovers branches."""
from copy import deepcopy
from typing import ClassVar

import numpy as np
import pandas as pd
from cyclic_pandas import is_cyclic
from geometry_pandas import is_geometry
from list_pandas import is_list
from code_recording import recordable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


@recordable
def manual_kind(series):
    if is_cyclic(series.dtype) or is_geometry(series.dtype) or is_list(series.dtype):
        return 'unsupported'
    if isinstance(series.dtype, pd.CategoricalDtype):
        if series.cat.ordered:
            return 'unsupported'
        levels = series.cat.categories
        if not all(isinstance(v, (str, int, float, bool, np.integer, np.floating, np.bool_)) and
                   (not isinstance(v, (float, np.floating)) or np.isfinite(v)) for v in levels):
            return 'unsupported'
        return 'categorical'
    if pd.api.types.is_bool_dtype(series.dtype):
        return 'boolean'
    if pd.api.types.is_integer_dtype(series.dtype):
        return 'integer'
    if pd.api.types.is_float_dtype(series.dtype):
        return 'number'
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return 'datetime'
    if (pd.api.types.is_object_dtype(series.dtype) or pd.api.types.is_string_dtype(series.dtype)) and all(isinstance(v, str) for v in series.dropna()):
        return 'categorical'
    return 'unsupported'


@recordable
def manual_levels(series):
    """Observed labels in the current preview, preserving categorical order."""
    if isinstance(series.dtype, pd.CategoricalDtype):
        return list(series.cat.remove_unused_categories().cat.categories)
    return list(series.dropna().unique())


@recordable
def manual_value(series, value, extra_level='', levels=None):
    """Validate without rounding, truncation, overflow, or implicit categories."""
    kind = manual_kind(series)
    if kind in ('integer', 'number'):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError('Enter a finite number.')
        if kind == 'integer':
            # Parse through a decimal representation to preserve large integers.
            from decimal import Decimal
            exact = Decimal(str(value))
            if exact != exact.to_integral_value():
                raise ValueError('Enter a whole number.')
            number = int(exact)
            dtype = getattr(series.dtype, 'numpy_dtype', series.dtype)
            bounds = np.iinfo(dtype)
            if not bounds.min <= number <= bounds.max:
                raise ValueError(f'Value is outside the range of {series.dtype}.')
        else:
            dtype = getattr(series.dtype, 'numpy_dtype', series.dtype)
            if abs(number) > np.finfo(dtype).max:
                raise ValueError(f'Value is outside the range of {series.dtype}.')
        return number
    if kind == 'boolean':
        if value not in ('true', 'false', True, False):
            raise ValueError('Choose True or False.')
        return value is True or value == 'true'
    if kind == 'categorical':
        if levels is None:
            levels = manual_levels(series)
        if value not in levels and (not extra_level or value != extra_level):
            raise ValueError('Choose an existing category or the configured additional level.')
        return value
    if kind == 'datetime':
        parsed = pd.Timestamp(value)
        if pd.isna(parsed):
            raise ValueError('Enter a date and time.')
        tz = getattr(series.dtype, 'tz', None)
        if tz is not None:
            parsed = parsed.tz_localize(tz) if parsed.tzinfo is None else parsed.tz_convert(tz)
        elif parsed.tzinfo is not None:
            raise ValueError('Use a timezone-free date and time for this variable.')
        return parsed.isoformat()
    raise ValueError('This data type is not supported for manual imputation.')


@recordable
def manual_feature(frame, feature):
    """Encode one split feature; missing and unseen categories remain NaN."""
    series = frame[feature['column']]
    kind = feature['kind']
    if kind == 'categorical':
        known = series.isin(feature['levels']) & series.notna()
        values = np.where(known, series.eq(feature['level']).fillna(False).to_numpy(dtype=bool).astype(float), np.nan)
    elif kind == 'datetime':
        values = np.array([np.nan if pd.isna(v) else pd.Timestamp(v).value / 86400000000000 for v in series], dtype=float)
    else:
        values = series.to_numpy(dtype=float, na_value=np.nan)
    values = np.asarray(values, dtype=np.float32)
    values[~np.isfinite(values)] = np.nan
    return values


@recordable
def manual_mask(frame, conditions):
    mask = np.ones(len(frame), dtype=bool)
    for condition in conditions:
        values = manual_feature(frame, condition['feature'])
        observed_left = np.ones(len(values), dtype=bool) if condition.get('missing_split') else values <= condition['threshold']
        left = np.where(np.isnan(values), condition['missing_left'], observed_left)
        mask &= left if condition['left'] else ~left
    return mask


@recordable
class ManualImputationTransformer(TransformerMixin, BaseEstimator):
    """Cloneable fixed rules, evaluated simultaneously against incoming values."""
    tuning_metadata: ClassVar[dict[str, dict[str, object]]] = {'rules': {'fixed': True}}

    def __init__(self, rules=()):
        self.rules = rules

    def _validate(self, X, rules):
        if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
            raise ValueError('Manual imputation requires a DataFrame with unique columns.')
        for rule in rules:
            columns = [rule['column']] + [c['feature']['column'] for c in rule['conditions']]
            if any(c not in X for c in columns):
                raise ValueError(f'Manual imputation requires columns {columns}.')
            series = X[rule['column']]
            if manual_kind(series) != rule['kind']:
                raise ValueError(f"Data type changed for {rule['column']}.")
            manual_value(series, rule['value'], rule.get('extra_level', ''), rule.get('levels'))
            for condition in rule['conditions']:
                feature = condition['feature']
                if feature['column'] == rule['column']:
                    raise ValueError('A rule cannot depend on its own destination.')
                if manual_kind(X[feature['column']]) != feature['kind']:
                    raise ValueError(f"Split data type changed for {feature['column']}.")
                if not condition.get('missing_split') and not np.isfinite(condition['threshold']):
                    raise ValueError('Split thresholds must be finite.')

    def fit(self, X, y=None):
        self._validate(X, self.rules)
        self.rules_ = deepcopy(self.rules)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, 'rules_')
        self._validate(X, self.rules_)
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError('Manual imputation columns must match the fitted schema.')
        result = X.copy()
        assigned = {}
        for rule in self.rules_:
            column = rule['column']
            mask = manual_mask(X, rule['conditions']) & X[column].isna().to_numpy()
            previous = assigned.setdefault(column, np.zeros(len(X), dtype=bool))
            if np.any(mask & previous):
                raise ValueError(f'Overlapping manual imputation rules for {column}.')
            previous |= mask
            series = result[column].copy()
            value = manual_value(X[column], rule['value'], rule.get('extra_level', ''), rule.get('levels'))
            if rule['kind'] == 'datetime':
                value = pd.Timestamp(value)
            if isinstance(series.dtype, pd.CategoricalDtype) and value not in series.cat.categories:
                series = series.cat.add_categories([value])
            series.iloc[np.flatnonzero(mask)] = value
            result[column] = series
        return result

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'feature_names_in_')
        if input_features is not None and list(input_features) != list(self.feature_names_in_):
            raise ValueError('input_features must match the fitted schema.')
        return self.feature_names_in_.copy()
