"""A small pandas extension dtype for list-valued and basket data.
It uses only pandas' public extension APIs.

Public entry points
-------------------
``ListArray``
    One-dimensional storage class for :class:`ListDtype`.

``ListDtype``
    Dtype class for list-valued observations.

``ListLikeEvidence``
    Class describing evidence that a Series is plausibly list-valued.

``as_list(data)``
    Converts collections, JSON arrays, or delimited strings to ``ListDtype``.

``is_list(obj)``
    Tests a Series, array, or dtype for an explicit ``ListDtype``.

``list_like_evidence(series)``
    Returns evidence that a Series could plausibly represent lists.

``is_list_like(series)``
    Returns whether a Series could plausibly represent lists.
"""

from __future__ import annotations

import csv
import io
import json
import re
from collections.abc import Mapping, Sequence
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

DEFAULT_DELIMITERS: tuple[str, ...] = (",", "|", ";", "\t", "%")


@dataclass(frozen=True)
class ListLikeEvidence:
    """Evidence returned by :func:`list_like_evidence`."""
    matches: bool
    confidence: str
    representation: str | None
    delimiter: str | None
    success_ratio: float
    reason: str


class ListDtype(ExtensionDtype):
    """Dtype for list-valued observations.

    ``item_type`` is descriptive metadata. It is inferred when possible but
    values are not coerced to that type by this reference implementation.
    """
    name = "basket"
    type = list
    kind = "O"
    na_value = pd.NA
    _metadata = ("item_type",)

    def __init__(self, item_type: type | tuple[type, ...] | None = None) -> None:
        if item_type is not None and not (
            isinstance(item_type, type)
            or (
                isinstance(item_type, tuple)
                and item_type
                and all(isinstance(value, type) for value in item_type)
            )
        ):
            raise TypeError("item_type must be a type, a tuple of types, or None")
        self.item_type = item_type

    @classmethod
    def construct_array_type(cls) -> type[ListArray]:
        return ListArray

    def __repr__(self) -> str:
        if self.item_type is None:
            return "ListDtype()"
        if isinstance(self.item_type, tuple):
            names = ", ".join(value.__name__ for value in self.item_type)
        else:
            names = self.item_type.__name__
        return f"ListDtype(item_type={names})"


class ListArray(ExtensionArray):
    """One-dimensional storage class for :class:`ListDtype`.

    Each non-missing scalar is exposed as a Python list. Missing observations
    are stored as ``pd.NA``. Sets are sorted by representation during
    conversion so their otherwise arbitrary iteration order is not retained.
    """
    def __init__(self, values: np.ndarray, dtype: ListDtype, *, copy=False):
        if not isinstance(dtype, ListDtype):
            raise TypeError("dtype must be a ListDtype")
        data = np.empty(len(values), dtype=object)
        for position, value in enumerate(values):
            data[position] = value
        self._data = data.copy() if copy else data
        self._dtype = dtype

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        values = list(scalars)
        normalized = [_normalize_list_scalar(value) for value in values]
        if dtype is None:
            dtype = ListDtype(_infer_item_type(normalized))
        if not isinstance(dtype, ListDtype):
            raise TypeError("constructing a ListArray requires a ListDtype")
        data = np.empty(len(normalized), dtype=object)
        data[:] = normalized
        return cls(data, dtype, copy=copy)

    @classmethod
    def _from_factorized(cls, values, original):
        return cls._from_sequence(values, dtype=original.dtype)

    @classmethod
    def _concat_same_type(cls, to_concat):
        if not to_concat:
            raise ValueError("cannot concatenate an empty sequence")
        dtype = to_concat[0].dtype
        if any(array.dtype != dtype for array in to_concat[1:]):
            dtype = ListDtype()
        return cls(np.concatenate([array._data for array in to_concat]), dtype)

    @property
    def dtype(self) -> ListDtype:
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
        if isinstance(result, np.ndarray):
            return type(self)(result, self.dtype)
        return result

    def __eq__(self, other):
        if isinstance(other, ListArray):
            if len(other) != len(self):
                return np.zeros(len(self), dtype=bool)
            return np.array([
                _list_scalar_equal(left, right)
                for left, right in zip(self._data, other._data)
            ])
        try:
            normalized = _normalize_list_scalar(other)
        except TypeError:
            return np.zeros(len(self), dtype=bool)
        return np.array([
            _list_scalar_equal(value, normalized) for value in self._data
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
            sentinel = _normalize_list_scalar(fill_value)
        result = take(
            self._data,
            indices,
            allow_fill=allow_fill,
            fill_value=sentinel,
        )
        return type(self)(result, self.dtype)

    def copy(self) -> ListArray:
        copied = np.empty(len(self), dtype=object)
        copied[:] = [
            value.copy() if value is not pd.NA else pd.NA
            for value in self._data
        ]
        return type(self)(copied, self.dtype)

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
    if isinstance(value, (list, tuple, set, frozenset, np.ndarray, Mapping)):
        return False
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(result, (bool, np.bool_)) and bool(result)


def _normalize_list_scalar(value: object) -> list[Any] | Any:
    if _is_missing_scalar(value):
        return pd.NA
    if isinstance(value, list):
        return value.copy()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=repr)
    if isinstance(value, np.ndarray) and value.ndim == 1:
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not a supported list value")


def _list_scalar_equal(left: object, right: object) -> bool:
    if left is pd.NA or right is pd.NA:
        return left is pd.NA and right is pd.NA
    try:
        result = left == right
    except (TypeError, ValueError):
        return False
    return bool(result) if isinstance(result, (bool, np.bool_)) else False


def _infer_item_type(values: Sequence[object]) -> type | tuple[type, ...] | None:
    types = {
        type(item)
        for value in values
        if value is not pd.NA
        for item in value
        if not _is_missing_scalar(item)
    }
    if not types:
        return None
    ordered = tuple(sorted(types, key=lambda value: (value.__module__, value.__qualname__)))
    return ordered[0] if len(ordered) == 1 else ordered


def is_list(obj) -> bool:
    """Return whether ``obj`` is a list Series, array, or dtype."""
    return isinstance(getattr(obj, "dtype", obj), ListDtype)


def _sample_nonmissing(
    series: pd.Series,
    *,
    limit: int | None,
    blank_as_missing: bool,
) -> pd.Series:
    keep = series.map(lambda value: not _is_missing_scalar(value))
    values = series[keep]
    if blank_as_missing:
        values = values[
            values.map(lambda value: not (isinstance(value, str) and not value.strip()))
        ]
    if limit is not None and len(values) > limit:
        positions = np.linspace(0, len(values) - 1, num=limit, dtype=int)
        values = values.iloc[positions]
    return values


def _arrow_list_evidence(
    series: pd.Series,
) -> ListLikeEvidence | None:
    """Recognise PyArrow list and large-list extension dtypes."""
    dtype = series.dtype
    # This is already our own ListDtype, not an Arrow dtype.
    if isinstance(dtype, ListDtype):
        return None
    arrow_dtype_class = getattr(pd, "ArrowDtype", None)
    if (
        arrow_dtype_class is None
        or not isinstance(dtype, arrow_dtype_class)
    ):
        return None
    pyarrow_dtype = dtype.pyarrow_dtype
    try:
        import pyarrow as pa
    except ImportError:
        return None
    is_arrow_list = (
        pa.types.is_list(pyarrow_dtype)
        or pa.types.is_large_list(pyarrow_dtype)
        or pa.types.is_fixed_size_list(pyarrow_dtype)
    )
    if not is_arrow_list:
        return None
    return ListLikeEvidence(
        matches=True,
        confidence="certain",
        representation="typed-list",
        delimiter=None,
        success_ratio=1.0,
        reason=f"Series has a typed Arrow {pyarrow_dtype} dtype",
    )

def _is_supported_collection(
    value: object,
    *,
    allow_tuple: bool,
    allow_set: bool,
    allow_array: bool,
    reject_mapping: bool,
) -> bool:
    if isinstance(value, (str, bytes, bytearray)):
        return False
    if reject_mapping and isinstance(value, Mapping):
        return False
    accepted: tuple[type, ...] = (list,)
    if allow_tuple:
        accepted += (tuple,)
    if allow_set:
        accepted += (set, frozenset)
    return isinstance(value, accepted) or bool(
        allow_array and isinstance(value, np.ndarray) and value.ndim == 1
    )


def _collection_evidence(
    values: pd.Series,
    *,
    threshold: float,
    min_items: int,
    min_multivalue_ratio: float,
    allow_tuple: bool,
    allow_set: bool,
    allow_array: bool,
    reject_mapping: bool,
) -> ListLikeEvidence:
    valid = values.map(lambda value: _is_supported_collection(
        value,
        allow_tuple=allow_tuple,
        allow_set=allow_set,
        allow_array=allow_array,
        reject_mapping=reject_mapping,
    ))
    success_ratio = float(valid.mean())
    if success_ratio < threshold:
        return ListLikeEvidence(
            False, "unknown", None, None, success_ratio,
            "Too few values are supported collection objects",
        )
    collections = values[valid]
    multivalue_ratio = float((collections.map(len) >= min_items).mean())
    if multivalue_ratio < min_multivalue_ratio:
        return ListLikeEvidence(
            False, "weak", "collection", None, success_ratio,
            "Collections are predominantly empty or single-valued",
        )
    element_types = sorted({
        type(item).__name__
        for collection in collections
        for item in collection
        if not _is_missing_scalar(item)
    })
    description = ", ".join(element_types) if element_types else "unknown"
    return ListLikeEvidence(
        True, "strong", "collection", None, success_ratio,
        f"{success_ratio:.0%} are collections; element classes: {description}",
    )


def _json_array_evidence(
    values: pd.Series,
    *,
    threshold: float,
    min_items: int,
    min_multivalue_ratio: float,
) -> ListLikeEvidence:
    parsed = []
    for value in values:
        try:
            result = json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            result = None
        parsed.append(result if isinstance(result, list) else None)
    valid = np.array([value is not None for value in parsed], dtype=bool)
    success_ratio = float(valid.mean())
    if success_ratio < threshold:
        return ListLikeEvidence(
            False, "unknown", None, None, success_ratio,
            "Too few strings are valid JSON arrays",
        )
    lengths = [len(value) for value in parsed if value is not None]
    multivalue_ratio = float(np.mean([length >= min_items for length in lengths]))
    if multivalue_ratio < min_multivalue_ratio:
        return ListLikeEvidence(
            False, "weak", "json-array", None, success_ratio,
            "JSON arrays are predominantly empty or single-valued",
        )
    return ListLikeEvidence(
        True, "strong", "json-array", None, success_ratio,
        f"{success_ratio:.0%} of values are valid JSON arrays",
    )


def _parse_delimited(
    value: str,
    delimiter: str,
    *,
    strip_items: bool,
) -> list[str] | None:
    if len(delimiter) == 1:
        try:
            row = next(csv.reader(
                io.StringIO(value),
                delimiter=delimiter,
                skipinitialspace=strip_items,
            ))
        except (csv.Error, StopIteration):
            return None
    else:
        row = value.split(delimiter)
    if strip_items:
        row = [item.strip() for item in row]
    return None if any(not item for item in row) else row


def _prose_like_ratio(values: pd.Series, *, delimiter: str) -> float:
    def appears_prose(value: str) -> bool:
        words = re.findall(r"\b[\w'-]+\b", value)
        sentence = bool(re.search(r"[.!?](?:\s|$)", value))
        comma_prose = delimiter == "," and ", " in value and len(words) >= 6
        return len(words) >= 10 or sentence or comma_prose
    return float(values.map(appears_prose).mean())


def _delimited_string_evidence(
    values: pd.Series,
    *,
    delimiters: Sequence[str],
    threshold: float,
    min_items: int,
    min_multivalue_ratio: float,
    min_delimiter_occurrences: int,
    max_prose_ratio: float,
    strip_items: bool,
) -> ListLikeEvidence:
    candidates = []
    for delimiter in delimiters:
        parsed = [
            _parse_delimited(value, delimiter, strip_items=strip_items)
            for value in values
        ]
        parseable = np.array([items is not None for items in parsed], dtype=bool)
        multivalue = np.array([
            items is not None
            and len(items) >= min_items
            and value.count(delimiter) >= min_delimiter_occurrences
            for value, items in zip(values, parsed)
        ], dtype=bool)
        success_ratio = float(parseable.mean())
        multivalue_ratio = float(multivalue.mean())
        prose_ratio = _prose_like_ratio(values, delimiter=delimiter)
        matches = (
            success_ratio >= threshold
            and multivalue_ratio >= min_multivalue_ratio
            and prose_ratio <= max_prose_ratio
        )
        candidates.append((
            matches, multivalue_ratio, success_ratio, -prose_ratio,
            delimiter, prose_ratio,
        ))
    matches, multivalue_ratio, success_ratio, _, delimiter, prose_ratio = max(candidates)
    return ListLikeEvidence(
        matches,
        "plausible" if matches else "unknown",
        "delimited-string" if matches else None,
        delimiter if matches else None,
        success_ratio,
        (
            f"Best delimiter is {delimiter!r}: {multivalue_ratio:.0%} "
            f"multi-item rows and {prose_ratio:.0%} prose-like rows"
        ),
    )


def list_like_evidence(
    series: pd.Series,
    *,
    threshold: float = 0.95,
    limit: int | None = 1_000,
    delimiters: Sequence[str] = DEFAULT_DELIMITERS,
    min_items: int = 2,
    min_multivalue_ratio: float = 0.50,
    min_delimiter_occurrences: int = 1,
    max_prose_ratio: float = 0.50,
    allow_tuple: bool = True,
    allow_set: bool = True,
    allow_array: bool = True,
    allow_json: bool = True,
    strip_items: bool = True,
    blank_as_missing: bool = True,
    reject_mapping: bool = True,
) -> ListLikeEvidence:
    """Return evidence that ``series`` plausibly contains list observations.

    Typed list arrays and actual collections provide strong evidence. JSON
    arrays provide strong serialized evidence. Consistently delimited strings
    provide plausible evidence because ordinary prose can be ambiguous.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")
    for name, value in {
        "threshold": threshold,
        "min_multivalue_ratio": min_multivalue_ratio,
        "max_prose_ratio": max_prose_ratio,
    }.items():
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an integer or None")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
    if isinstance(min_items, bool) or not isinstance(min_items, int):
        raise TypeError("min_items must be an integer")
    if min_items < 2:
        raise ValueError("min_items must be at least 2")
    if (
        isinstance(min_delimiter_occurrences, bool)
        or not isinstance(min_delimiter_occurrences, int)
        or min_delimiter_occurrences < 1
    ):
        raise ValueError("min_delimiter_occurrences must be a positive integer")
    delimiters = tuple(dict.fromkeys(
        delimiter for delimiter in delimiters
        if isinstance(delimiter, str) and delimiter
    ))
    if not delimiters:
        raise ValueError("at least one nonempty delimiter is required")
    if is_list(series):
        return ListLikeEvidence(
            True, "certain", "list-dtype", None, 1.0,
            "Series has an explicit ListDtype",
        )
    arrow = _arrow_list_evidence(series)
    if arrow is not None:
        return arrow
    values = _sample_nonmissing(
        series, limit=limit, blank_as_missing=blank_as_missing,
    )
    if values.empty:
        return ListLikeEvidence(
            False, "unknown", None, None, 0.0,
            "No non-missing values and no typed list dtype",
        )
    collections = _collection_evidence(
        values,
        threshold=threshold,
        min_items=min_items,
        min_multivalue_ratio=min_multivalue_ratio,
        allow_tuple=allow_tuple,
        allow_set=allow_set,
        allow_array=allow_array,
        reject_mapping=reject_mapping,
    )
    if collections.matches:
        return collections
    if not values.map(lambda value: isinstance(value, str)).all():
        return ListLikeEvidence(
            False, "unknown", None, None, collections.success_ratio,
            "Values mix strings, collections, or unsupported objects",
        )
    if allow_json:
        json_result = _json_array_evidence(
            values,
            threshold=threshold,
            min_items=min_items,
            min_multivalue_ratio=min_multivalue_ratio,
        )
        if json_result.matches:
            return json_result
    return _delimited_string_evidence(
        values,
        delimiters=delimiters,
        threshold=threshold,
        min_items=min_items,
        min_multivalue_ratio=min_multivalue_ratio,
        min_delimiter_occurrences=min_delimiter_occurrences,
        max_prose_ratio=max_prose_ratio,
        strip_items=strip_items,
    )


def is_list_like(series: pd.Series, **kwargs) -> bool:
    """Return whether a Series could plausibly represent lists."""
    return list_like_evidence(series, **kwargs).matches


def as_list(
    data,
    *,
    delimiter: str | None = None,
    delimiters: Sequence[str] = DEFAULT_DELIMITERS,
    allow_json: bool = True,
    strip_items: bool = True,
    infer: bool = True,
) -> pd.Series:
    """Return ``data`` as a Series with ``ListDtype``.

    Collection values are normalized to Python lists. JSON-array strings are
    decoded. Other strings are split with ``delimiter``; when it is omitted,
    :func:`list_like_evidence` selects a plausible delimiter. A single value
    without the selected delimiter becomes a one-item list.
    """
    source = data if isinstance(data, pd.Series) else pd.Series(data)
    if is_list(source):
        return source
    arrow = _arrow_list_evidence(source)
    if arrow is not None:
        values = source.astype(object).tolist()
        array = ListArray._from_sequence(values)
        return pd.Series(array, index=source.index, name=source.name)
    nonmissing = _sample_nonmissing(source, limit=None, blank_as_missing=True)
    all_collections = (
        not nonmissing.empty
        and nonmissing.map(lambda value: _is_supported_collection(
            value,
            allow_tuple=True,
            allow_set=True,
            allow_array=True,
            reject_mapping=True,
        )).all()
    )
    representation = "collection" if all_collections else None
    if representation is None and delimiter is None and infer:
        evidence = list_like_evidence(
            source,
            delimiters=delimiters,
            allow_json=allow_json,
            strip_items=strip_items,
        )
        if not evidence.matches:
            raise ValueError(f"data are not plausibly list-like: {evidence.reason}")
        representation = evidence.representation
        delimiter = evidence.delimiter
    elif representation is None and delimiter is not None:
        representation = "delimited-string"
    elif representation is None and not infer:
        raise TypeError("non-collection input requires a delimiter or infer=True")

    converted = []
    for value in source:
        if _is_missing_scalar(value) or (
            isinstance(value, str) and not value.strip()
        ):
            converted.append(pd.NA)
        elif _is_supported_collection(
            value,
            allow_tuple=True,
            allow_set=True,
            allow_array=True,
            reject_mapping=True,
        ):
            converted.append(_normalize_list_scalar(value))
        elif not isinstance(value, str):
            raise TypeError(f"cannot convert {type(value).__name__} to a list value")
        elif representation == "json-array":
            parsed = json.loads(value)
            if not isinstance(parsed, list):
                raise ValueError(f"{value!r} is not a JSON array")
            converted.append(parsed)
        elif representation == "delimited-string" and delimiter is not None:
            parsed = _parse_delimited(value, delimiter, strip_items=strip_items)
            if parsed is None:
                raise ValueError(f"{value!r} cannot be parsed with {delimiter!r}")
            converted.append(parsed)
        else:
            raise ValueError("could not determine the list representation")
    array = ListArray._from_sequence(converted)
    return pd.Series(array, index=source.index, name=source.name)


@register_series_accessor("basket")
class ListAccessor:
    """List and basket operations available as ``series.basket``.

    Pandas reserves ``Series.list`` for its own list accessor, so this module
    deliberately uses the domain-oriented name ``basket``.
    """
    def __init__(self, series: pd.Series):
        if not is_list(series):
            raise AttributeError("the .basket accessor requires ListDtype")
        self._series = series

    def lengths(self) -> pd.Series:
        """Return nullable list lengths."""
        result = [pd.NA if value is pd.NA else len(value) for value in self._series.array]
        return pd.Series(
            pd.array(result, dtype="Int64"),
            index=self._series.index,
            name=self._series.name,
        )

    def contains(self, item: object) -> pd.Series:
        """Return whether each list contains ``item``."""
        result = [
            pd.NA if value is pd.NA else item in value
            for value in self._series.array
        ]
        return pd.Series(
            pd.array(result, dtype="boolean"),
            index=self._series.index,
            name=self._series.name,
        )

    def explode(self, *, ignore_index: bool = False) -> pd.Series:
        """Expand list elements into rows."""
        return self._series.astype(object).explode(ignore_index=ignore_index)

    def indicators(
        self,
        *,
        dtype: str | type = "boolean",
        sparse: bool = False,
    ) -> pd.DataFrame:
        """Expand lists into binary indicator columns."""
        long = self.explode()
        long = long[long.notna()]
        if long.empty:
            return pd.DataFrame(index=self._series.index)
        result = pd.crosstab(long.index, long).gt(0)
        result = result.reindex(self._series.index, fill_value=False)
        if sparse:
            return result.astype(pd.SparseDtype(bool, fill_value=False))
        return result.astype(dtype)


__all__ = [
    "DEFAULT_DELIMITERS",
    "ListAccessor",
    "ListArray",
    "ListDtype",
    "ListLikeEvidence",
    "as_list",
    "is_list",
    "is_list_like",
    "list_like_evidence",
]
