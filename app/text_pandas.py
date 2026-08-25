"""A small pandas extension dtype for free text data.
It uses only pandas' public extension APIs.

Public entry points
-------------------
``TextArray``
    One-dimensional storage class for :class:`TextDtype`.

``TextDtype``
    Dtype class for free prose, distinct from ordinary string codes.

``TextEvidence``
    Class describing evidence that a Series is plausibly free text.

``as_text(data)``
    Converts values to a Series with ``TextDtype``.

``is_text(obj)``
    Tests a Series, array, or dtype for an explicit ``TextDtype``.

``text_evidence(series)``
    Returns evidence that a Series could plausibly contain free text.

``is_text_like(series)``
    Returns whether a Series could plausibly contain free text.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.extensions import (
    ExtensionArray,
    ExtensionDtype,
    register_series_accessor,
    take,
)


@dataclass(frozen=True)
class TextEvidence:
    """Evidence returned by :func:`text_evidence`."""
    matches: bool
    confidence: str
    reason: str
    success_ratio: float
    cardinality: int = 0
    uniqueness_ratio: float = 0.0
    median_length: float = 0.0
    whitespace_ratio: float = 0.0


class TextDtype(ExtensionDtype):
    """Dtype for free text rather than short strings used as codes."""
    name = "text"
    type = str
    kind = "O"
    na_value = pd.NA

    @classmethod
    def construct_array_type(cls) -> type[TextArray]:
        return TextArray

    def __repr__(self) -> str:
        return "TextDtype()"


class TextArray(ExtensionArray):
    """One-dimensional storage class for :class:`TextDtype`.

    Every non-missing scalar is a Python string and missing observations are
    stored as ``pd.NA``.
    """
    def __init__(self, values: np.ndarray, *, copy=False):
        data = np.empty(len(values), dtype=object)
        for position, value in enumerate(values):
            data[position] = _normalize_text_scalar(value, coerce=False)
        self._data = data.copy() if copy else data
        self._dtype = TextDtype()

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        if dtype is not None and not isinstance(dtype, TextDtype):
            raise TypeError("constructing a TextArray requires a TextDtype")
        values = list(scalars)
        data = np.empty(len(values), dtype=object)
        data[:] = [_normalize_text_scalar(value, coerce=False) for value in values]
        return cls(data, copy=copy)

    @classmethod
    def _from_factorized(cls, values, original):
        return cls._from_sequence(values, dtype=original.dtype)

    @classmethod
    def _concat_same_type(cls, to_concat):
        if not to_concat:
            raise ValueError("cannot concatenate an empty sequence")
        return cls(np.concatenate([array._data for array in to_concat]))

    @property
    def dtype(self) -> TextDtype:
        return self._dtype

    @property
    def nbytes(self) -> int:
        return self._data.nbytes + sum(
            value.__sizeof__() for value in self._data if value is not pd.NA
        )

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, item):
        result = self._data[item]
        return type(self)(result) if isinstance(result, np.ndarray) else result

    def __eq__(self, other):
        if isinstance(other, TextArray):
            if len(other) != len(self):
                return np.zeros(len(self), dtype=bool)
            return np.array([
                _text_scalar_equal(left, right)
                for left, right in zip(self._data, other._data)
            ])
        try:
            normalized = _normalize_text_scalar(other, coerce=False)
        except TypeError:
            return np.zeros(len(self), dtype=bool)
        return np.array([
            _text_scalar_equal(value, normalized) for value in self._data
        ])

    def isna(self) -> np.ndarray:
        return np.fromiter(
            (value is pd.NA for value in self._data),
            dtype=bool,
            count=len(self),
        )

    def take(self, indices, *, allow_fill=False, fill_value=None):
        if fill_value is None or fill_value is pd.NA:
            sentinel = pd.NA
        else:
            sentinel = _normalize_text_scalar(fill_value, coerce=False)
        result = take(
            self._data,
            indices,
            allow_fill=allow_fill,
            fill_value=sentinel,
        )
        return type(self)(result)

    def copy(self) -> TextArray:
        return type(self)(self._data.copy())

    def to_numpy(self, dtype=None, copy=False, na_value=pd.NA):
        values = self._data.copy()
        if na_value is not pd.NA:
            values[self.isna()] = na_value
        return values.astype(dtype) if dtype is not None else values

    def __array__(self, dtype=None, copy=None):
        return self.to_numpy(dtype=dtype, copy=bool(copy))


def _is_missing_scalar(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(result, (bool, np.bool_)) and bool(result)


def _normalize_text_scalar(value: object, *, coerce: bool) -> str | Any:
    if _is_missing_scalar(value):
        return pd.NA
    if isinstance(value, str):
        return value
    if coerce:
        return str(value)
    raise TypeError(f"{type(value).__name__} is not a text scalar")


def _text_scalar_equal(left: object, right: object) -> bool:
    if left is pd.NA or right is pd.NA:
        return left is pd.NA and right is pd.NA
    return bool(left == right)


def as_text(data, *, errors: str = "raise") -> pd.Series:
    """Return ``data`` as a Series with ``TextDtype``.

    Missing values are preserved. ``errors="raise"`` requires every other
    value to already be a string. ``errors="coerce"`` applies ``str(value)``.
    The input Series index and name are preserved.
    """
    if errors not in {"raise", "coerce"}:
        raise ValueError("errors must be 'raise' or 'coerce'")
    source = data if isinstance(data, pd.Series) else pd.Series(data)
    if is_text(source):
        return source
    values = [
        _normalize_text_scalar(value, coerce=errors == "coerce")
        for value in source
    ]
    array = TextArray._from_sequence(values, dtype=TextDtype())
    return pd.Series(array, index=source.index, name=source.name)


def is_text(obj: object) -> bool:
    """Return whether ``obj`` is a text Series, array, or dtype."""
    return isinstance(getattr(obj, "dtype", obj), TextDtype)


def text_evidence(
    series: pd.Series,
    *,
    high_cardinality: int = 100,
    uniqueness_threshold: float = 0.80,
    min_median_length: float = 20,
    whitespace_threshold: float = 0.50,
    string_threshold: float = 0.95,
    limit: int | None = None,
    blank_as_missing: bool = True,
) -> TextEvidence:
    """Return evidence that ``series`` probably contains free text.

    Free text is inferred from string content, high cardinality, typical
    length, and internal whitespace. An explicit ``TextDtype`` is definitive.
    Ordinary pandas strings that fail this test remain suitable as codes.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")
    if isinstance(high_cardinality, bool) or not isinstance(high_cardinality, int):
        raise TypeError("high_cardinality must be an integer")
    if high_cardinality < 1:
        raise ValueError("high_cardinality must be greater than zero")
    for name, value in {
        "uniqueness_threshold": uniqueness_threshold,
        "whitespace_threshold": whitespace_threshold,
        "string_threshold": string_threshold,
    }.items():
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    if min_median_length < 0:
        raise ValueError("min_median_length must be non-negative")
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an integer or None")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
    if is_text(series):
        return TextEvidence(
            True, "certain", "Series has an explicit TextDtype", 1.0,
        )
    dtype = series.dtype
    eligible = (
        pd.api.types.is_string_dtype(dtype)
        or pd.api.types.is_object_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
    )
    if not eligible:
        return TextEvidence(
            False, "certain", "Series dtype is not string-like", 0.0,
        )
    values = series.dropna()
    if values.empty:
        return TextEvidence(
            False, "unknown", "No non-missing values", 0.0,
        )
    if limit is not None and len(values) > limit:
        positions = np.linspace(0, len(values) - 1, num=limit, dtype=int)
        values = values.iloc[positions]
    strings = values.map(lambda value: isinstance(value, str))
    success_ratio = float(strings.mean())
    if success_ratio < string_threshold:
        return TextEvidence(
            False, "unknown",
            f"Only {success_ratio:.0%} of inspected values are strings",
            success_ratio,
        )
    text = values[strings].astype("string").str.strip()
    if blank_as_missing:
        text = text.mask(text.eq("")).dropna()
    if text.empty:
        return TextEvidence(
            False, "unknown", "No nonblank string values", success_ratio,
        )
    cardinality = int(text.nunique(dropna=True))
    uniqueness_ratio = cardinality / len(text)
    lengths = text.str.len()
    median_length = float(lengths.median())
    whitespace_ratio = float(
        text.str.contains(r"\s", regex=True, na=False).mean()
    )
    cardinality_matches = (
        cardinality > high_cardinality
        or uniqueness_ratio >= uniqueness_threshold
    )
    length_matches = median_length >= min_median_length
    whitespace_matches = whitespace_ratio >= whitespace_threshold
    matches = cardinality_matches and length_matches and whitespace_matches
    failed = []
    if not cardinality_matches:
        failed.append("cardinality")
    if not length_matches:
        failed.append("median length")
    if not whitespace_matches:
        failed.append("whitespace")
    reason = (
        "Values have high cardinality, prose-like length, and internal whitespace"
        if matches
        else f"Free-text evidence failed: {', '.join(failed)}"
    )
    return TextEvidence(
        matches,
        "strong" if matches else "unknown",
        reason,
        success_ratio,
        cardinality,
        uniqueness_ratio,
        median_length,
        whitespace_ratio,
    )


def is_text_like(series: pd.Series, **kwargs) -> bool:
    """Return whether a Series could plausibly contain free text."""
    return text_evidence(series, **kwargs).matches


@register_series_accessor("text")
class TextAccessor:
    """Free-text operations available as ``series.text``."""
    def __init__(self, series: pd.Series):
        if not is_text(series):
            raise AttributeError("the .text accessor requires TextDtype")
        self._series = series

    def lengths(self) -> pd.Series:
        """Return nullable character counts."""
        result = [
            pd.NA if value is pd.NA else len(value)
            for value in self._series.array
        ]
        return pd.Series(
            pd.array(result, dtype="Int64"),
            index=self._series.index,
            name=self._series.name,
        )

    def word_counts(self) -> pd.Series:
        """Return nullable whitespace-delimited word counts."""
        result = [
            pd.NA if value is pd.NA else len(value.split())
            for value in self._series.array
        ]
        return pd.Series(
            pd.array(result, dtype="Int64"),
            index=self._series.index,
            name=self._series.name,
        )

    def to_string(self) -> pd.Series:
        """Return an ordinary pandas ``StringDtype`` Series."""
        return pd.Series(
            pd.array(self._series.array.to_numpy(), dtype="string"),
            index=self._series.index,
            name=self._series.name,
        )


__all__ = [
    "TextAccessor",
    "TextArray",
    "TextDtype",
    "TextEvidence",
    "as_text",
    "is_text",
    "is_text_like",
    "text_evidence",
]
