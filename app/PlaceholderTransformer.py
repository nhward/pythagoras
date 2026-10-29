"""Fixed, user-approved placeholder replacements for sklearn pipelines."""
from copy import deepcopy
from typing import ClassVar

import numpy as np
import pandas as pd
from code_recording import recordable
from cyclic_pandas import is_cyclic
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from text_pandas import is_text
from var_types import key_from_dtype


@recordable
def _columns_by_placeholder_kind(df: pd.DataFrame) -> dict[str, list[str]]:
    columns = {"int": [], "dec": [], "str": [], "dte": [], "bkt": []}
    for column in df.columns:
        kind = key_from_dtype(df[column].dtype)
        if kind in ["geo", "cpx", "dur", "obj", "unk", "log"]:
            continue
        if kind == "cyc":
            kind="str" if df[column].dtype.is_categorical else "dec"
        if kind in ["cde", "nom", "ord", "txt"]:
            kind="str"
        columns[kind].append(column)
    return columns


@recordable
def _normalise_placeholder_text(value: object, *, case_sensitive: bool) -> str:
    text = str(value)
    return text if case_sensitive else text.casefold()


@recordable
def _scalar_placeholder_mask(
    series: pd.Series,
    kind: str,
    placeholder: object,
    *,
    float_eps: float,
    extrema: bool,
    case_sensitive: bool,
) -> np.ndarray:
    """Match one placeholder against a scalar-valued Series."""
    if kind == "str":
        values = series.astype("string").fillna("")
        target = _normalise_placeholder_text(
            placeholder,
            case_sensitive=case_sensitive,
        )
        if not case_sensitive:
            values = values.str.casefold()
        return values.eq(target).to_numpy(dtype=bool, na_value=False)
    if kind == "dte":
        target = pd.to_datetime(placeholder, errors="coerce")
        if pd.isna(target):
            return np.zeros(len(series), dtype=bool)
        return series.eq(target).to_numpy(dtype=bool, na_value=False)
    if kind == "int":
        try:
            target = int(placeholder)
        except (TypeError, ValueError):
            return np.zeros(len(series), dtype=bool)
        if extrema:
            minimum, maximum = series.min(skipna=True), series.max(skipna=True)
            if target != minimum and target != maximum:
                return np.zeros(len(series), dtype=bool)
        return series.eq(target).to_numpy(dtype=bool, na_value=False)
    if kind == "dec":
        try:
            target = float(placeholder)
        except (TypeError, ValueError):
            return np.zeros(len(series), dtype=bool)
        values = series.to_numpy(dtype="float64", na_value=np.nan)
        finite = values[np.isfinite(values)]
        if extrema and finite.size:  # noqa: SIM102
            if not (
                np.isclose(finite.min(), target, atol=float_eps)
                or np.isclose(finite.max(), target, atol=float_eps)
            ):
                return np.zeros(len(series), dtype=bool)
        return np.isfinite(values) & (np.abs(values - target) < float_eps)
    return np.zeros(len(series), dtype=bool)


@recordable
def _apply_list_placeholder_codes(
    df: pd.DataFrame,
    columns: list[str],
    lookup: dict[str, int],
    codes_arr: np.ndarray,
    col_pos: dict[str, int],
    *,
    case_sensitive: bool,
) -> None:
    """Scan every list element once and assign the first matching code per cell."""
    if not lookup:
        return
    for column in columns:
        output = codes_arr[:, col_pos[column]]
        for row, value in enumerate(df[column].array):
            if value is pd.NA:
                continue
            for item in value:
                if not isinstance(item, str):
                    continue
                key = item if case_sensitive else item.casefold()
                code = lookup.get(key)
                if code is not None:
                    output[row] = code
                    break


@recordable
def _rebuild_custom_series(series: pd.Series, values: list[object]) -> pd.Series:
    array = type(series.array)._from_sequence(values, dtype=series.dtype)
    return pd.Series(array, index=series.index, name=series.name)


@recordable
def _replace_scalar_matches(series: pd.Series, mask: np.ndarray) -> pd.Series:
    if not mask.any():
        return series
    if is_cyclic(series.dtype) or is_text(series.dtype):
        values = list(series.array)
        for position in np.flatnonzero(mask):
            values[int(position)] = pd.NA
        return _rebuild_custom_series(series, values)
    result = series.copy()
    result.loc[mask] = pd.NaT if pd.api.types.is_datetime64_any_dtype(series.dtype) else pd.NA
    return result


@recordable
def _remove_list_placeholders(
    series: pd.Series,
    placeholders: set[str],
    *,
    case_sensitive: bool,
) -> pd.Series:
    """Remove matching string elements in one pass, preserving ListDtype."""
    if not placeholders:
        return series
    changed = False
    values: list[object] = []
    for value in series.array:
        if value is pd.NA:
            values.append(pd.NA)
            continue
        retained = [
            item
            for item in value
            if not (
                isinstance(item, str)
                and (item if case_sensitive else item.casefold()) in placeholders
            )
        ]
        matched = len(retained) != len(value)
        if matched:
            changed = True
            values.append(retained if retained else pd.NA)
        else:
            values.append(value)
    if not changed:
        return series
    # TODO: Implement __setitem__/_putmask in the custom extension arrays so pandas-native masked assignment can replace this reconstruction step.
    return _rebuild_custom_series(series, values)


@recordable
def placeholder_rules(frame, sentinels, *, extrema=True, case_sensitive=False, float_eps=1e-9):
    """Resolve reviewed choices to fixed column rules, outside estimator fitting.

    Extrema eligibility uses the unchanged review frame, so candidate order does
    not affect approval. It is never recalculated on training or prediction data.
    """
    columns = _columns_by_placeholder_kind(frame)
    rules = []
    for sentinel in sentinels:
        kind, value = sentinel.split(": ", 1)
        if kind not in {"str", "int", "dec", "dte"}:
            raise ValueError(f"Unknown placeholder kind: {kind}")
        for column in columns[kind]:
            if extrema and kind in {"int", "dec"}:
                mask = _scalar_placeholder_mask(frame[column], kind, value,
                    float_eps=float_eps, extrema=True, case_sensitive=case_sensitive)
                if not mask.any():
                    continue
            rules.append((column, kind, value))
        if kind == "str":
            rules.extend((column, "bkt", value) for column in columns["bkt"])
    return rules


@recordable
class PlaceholderTransformer(TransformerMixin, BaseEstimator):
    """Apply fixed (column, kind, value) rules; fit only validates the schema."""

    tuning_metadata: ClassVar[dict[str, dict[str, object]]] = {'rules': {'fixed': True},
     'float_eps': {'fixed': True},
     'case_sensitive': {'fixed': True}}

    def __init__(self, rules=(), *, case_sensitive=False, float_eps=1e-9, target_columns=()):
        self.target_columns = target_columns
        self.rules = rules
        self.case_sensitive = case_sensitive
        self.float_eps = float_eps

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
            raise ValueError("Placeholder replacement requires a DataFrame with unique columns")
        if not np.isfinite(self.float_eps) or self.float_eps <= 0:
            raise ValueError("float_eps must be positive and finite")
        for column, kind, value in self.rules:
            if (column not in X and column not in self.target_columns) or kind not in {"str", "int", "dec", "dte", "bkt"}:
                raise ValueError(f"Invalid placeholder rule: {(column, kind, value)!r}")
        self.target_columns_ = tuple(self.target_columns)
        self.rules_ = deepcopy(self.rules)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self, "rules_")
        if not isinstance(X, pd.DataFrame) or list(X.columns) != list(self.feature_names_in_):
            raise ValueError("Placeholder replacement requires the fitted columns in order")
        return self._replace(X, [rule for rule in self.rules_ if rule[0] in X])

    def _replace(self, X, rules):
        result = X.copy()
        for column in dict.fromkeys(rule[0] for rule in rules):
            column_rules = [rule for rule in rules if rule[0] == column]
            if column_rules[0][1] == "bkt":
                values = {_normalise_placeholder_text(value, case_sensitive=self.case_sensitive)
                          for _, _, value in column_rules}
                result[column] = _remove_list_placeholders(X[column], values,
                    case_sensitive=self.case_sensitive)
            else:
                mask = np.zeros(len(X), dtype=bool)
                for _, kind, value in column_rules:
                    mask |= _scalar_placeholder_mask(X[column], kind, value,
                        float_eps=self.float_eps, extrema=False, case_sensitive=self.case_sensitive)
                result[column] = _replace_scalar_matches(X[column], mask)
        return result

    def transform_x_y(self, X, y):
        """Replay the same target rules through the project's paired pipeline API.

        Array target columns follow target_columns order. Pandas target names,
        indexes, and dtypes are retained; rows are never dropped here.
        """
        frame = self.transform(X)
        if y is None:
            return frame, None
        if len(y) != len(X):
            raise ValueError("X and y must have the same number of rows")
        rules = [rule for rule in self.rules_ if rule[0] in self.target_columns_]
        if not rules:
            return frame, y
        if isinstance(y, pd.DataFrame):
            if not y.columns.is_unique or not set(y.columns).issubset(self.target_columns_):
                raise ValueError("Target DataFrame columns must match target_columns")
            return frame, self._replace(y, [rule for rule in rules if rule[0] in y])
        if isinstance(y, pd.Series):
            column = y.name if y.name in self.target_columns_ else None
            if column is None and len(self.target_columns_) == 1:
                column = self.target_columns_[0]
            if column is None:
                raise ValueError("Name the target Series when multiple target columns are configured")
            target = self._replace(y.to_frame(name=column),
                [rule for rule in rules if rule[0] == column]).iloc[:, 0]
            target.name = y.name
            return frame, target
        values = np.asarray(y)
        if values.ndim not in (1, 2):
            raise ValueError("Targets must be a one- or two-dimensional array")
        width = 1 if values.ndim == 1 else values.shape[1]
        if width != len(self.target_columns_):
            raise ValueError("Target array width must match target_columns")
        target = self._replace(pd.DataFrame(values, columns=self.target_columns_), rules).to_numpy()
        return frame, target.ravel() if values.ndim == 1 else target

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_in_")
        if input_features is not None and list(input_features) != list(self.feature_names_in_):
            raise ValueError("input_features must match the fitted schema")
        return self.feature_names_in_.copy()
