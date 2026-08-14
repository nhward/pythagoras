"""A small pandas extension dtype for categorical and numeric cyclic data.
It uses only pandas' public extension APIs.

Public entry points
-------------------
``DEFAULT_CYCLES``
    A dictionary of common cycles

``CyclicArray``
    One-dimensional storage class for :class:`CyclicDtype`

``CyclicDtype``
    Dtype class for either a finite categorical cycle or a numeric cycle.

``CyclicEvidence``
    Class for cyclic evidence

``as_cyclic(ordered_categorical)``
    Builds a categorical cycle from the declared category order.

``as_cyclic(numeric_data, period=...)``
    Builds a numeric cycle.  The period is deliberately never inferred.

``is_cyclic(obj)``
    Tests a Series, array, or dtype.

``cyclic_evidence``
    Returns evidence that ``series`` could plausibly represent a cycle.

``is_cyclic_like``
    Returns whether a Series could plausibly represent a cycle.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import time
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.extensions import (
    ExtensionArray,
    ExtensionDtype,
    register_series_accessor,
    take,
)

DEFAULT_CYCLES: dict[str, tuple[str, ...]] = {
    "month": (
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    ),
    "month_abbreviation": (
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec",
    ),
    "weekday": (
        "monday", "tuesday", "wednesday", "thursday",
        "friday", "saturday", "sunday",
    ),
    "weekday_abbreviation": (
        "mon", "tue", "wed", "thu", "fri", "sat", "sun",
    ),
    "season": ("spring", "summer", "autumn", "winter"),
    "season_fall": ("spring", "summer", "fall", "winter"),
    "compass_4": ("north", "east", "south", "west"),
    "compass_4_abbreviation": ("n", "e", "s", "w"),
    "compass_8": ("n", "ne", "e", "se", "s", "sw", "w", "nw"),
    "moon_phase": (
        "new moon", "waxing crescent", "first quarter", "waxing gibbous",
        "full moon", "waning gibbous", "last quarter", "waning crescent",
    ),
    "tide_phase": ("low", "rising", "high", "falling"),
    "zodiac": (
        "aries", "taurus", "gemini", "cancer", "leo", "virgo",
        "libra", "scorpio", "sagittarius", "capricorn", "aquarius", "pisces",
    ),
}


@dataclass(frozen=True)
class CyclicEvidence:
    """Evidence returned by :func:`cyclic_evidence`."""
    matches: bool
    confidence: str
    reason: str
    period: float | int | None = None
    cycle: str | None = None


_NUMERIC_NAME_HINTS: tuple[
    tuple[re.Pattern[str], tuple[float | int, ...], str], ...
] = (
    (re.compile(r"(?:^|_)(?:month|month_number|month_num)(?:_|$)"), (12,), "month"),
    (re.compile(r"(?:^|_)(?:weekday|day_of_week|dow)(?:_|$)"), (7,), "weekday"),
    (re.compile(r"(?:^|_)(?:hour|hour_of_day)(?:_|$)"), (12, 24), "hour"),
    (re.compile(r"(?:^|_)(?:minute|minute_of_hour)(?:_|$)"), (60,), "minute"),
    (re.compile(r"(?:^|_)(?:second|second_of_minute)(?:_|$)"), (60,), "second"),
    (re.compile(r"(?:^|_)(?:season|season_number)(?:_|$)"), (4,), "season"),
    (
        re.compile(r"(?:^|_)(?:radian|radians|angle_rad)(?:_|$)"),
        (2 * np.pi,),
        "angle_radians",
    ),
    (
        re.compile(
            r"(?:^|_)(?:angle|bearing|azimuth|heading|direction|degree|degrees)(?:_|$)"
        ),
        (360,),
        "angle_degrees",
    ),
)


class CyclicDtype(ExtensionDtype):
    """Dtype for either a finite categorical cycle or a numeric cycle.

    Exactly one of ``categories`` and ``period`` must be supplied.  Category
    position zero and numeric zero are the implicit origins.
    """

    name = "cyclic"
    type = object
    kind = "O"
    na_value = pd.NA
    _metadata = ("categories", "period")

    def __init__(
        self,
        *,
        categories: Sequence[Any] | None = None,
        period: Real | None = None,
    ) -> None:
        if (categories is None) == (period is None):
            raise TypeError("specify exactly one of 'categories' or 'period'")

        if categories is not None:
            categories = tuple(categories)
            if len(categories) < 2:
                raise ValueError("a categorical cycle needs at least two categories")
            index = pd.Index(categories)
            if index.hasnans:
                raise ValueError("cyclic categories cannot contain missing values")
            if not index.is_unique:
                raise ValueError("cyclic categories must be unique")
            self.categories = categories
            self.period = None
        else:
            if not isinstance(period, Real) or not np.isfinite(period) or period <= 0:
                raise ValueError("period must be a finite positive number")
            self.categories = None
            self.period = period

    @property
    def is_categorical(self) -> bool:
        return self.categories is not None

    @property
    def cardinality(self) -> int | None:
        return len(self.categories) if self.categories is not None else None

    @classmethod
    def construct_array_type(cls) -> type[CyclicArray]:
        return CyclicArray

    def __repr__(self) -> str:
        if self.is_categorical:
            body = ", ".join(repr(x) for x in self.categories)
            return f"CyclicDtype(categories=[{body}])"
        return f"CyclicDtype(period={self.period!r})"


class CyclicArray(ExtensionArray):
    """One-dimensional storage for :class:`CyclicDtype`.

    Categorical data are stored as integer codes (``-1`` means missing).
    Numeric data are normalized to ``[0, period)`` and stored as floats.
    """

    def __init__(self, values: np.ndarray, dtype: CyclicDtype, *, copy=False):
        if not isinstance(dtype, CyclicDtype):
            raise TypeError("dtype must be a CyclicDtype")
        expected = np.int64 if dtype.is_categorical else np.float64
        self._data = np.array(values, dtype=expected, copy=copy)
        if self._data.ndim != 1:
            raise ValueError("CyclicArray must be one-dimensional")
        self._dtype = dtype

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        if not isinstance(dtype, CyclicDtype):
            raise TypeError("constructing a CyclicArray requires a CyclicDtype")

        if dtype.is_categorical:
            categorical = pd.Categorical(scalars, categories=dtype.categories)
            codes = categorical.codes.astype(np.int64, copy=copy)
            # pd.Categorical silently turns unknown labels into missing values.
            supplied_missing = pd.isna(np.asarray(scalars, dtype=object))
            unknown = (codes == -1) & ~supplied_missing
            if unknown.any():
                bad = np.asarray(scalars, dtype=object)[unknown][0]
                raise ValueError(f"{bad!r} is not in the cyclic categories")
            return cls(codes, dtype, copy=False)

        numeric = pd.to_numeric(np.asarray(scalars), errors="raise").astype(float)
        normalized = np.mod(numeric, dtype.period)
        return cls(normalized, dtype, copy=copy)

    @classmethod
    def _from_factorized(cls, values, original):
        return cls._from_sequence(values, dtype=original.dtype)

    @classmethod
    def _concat_same_type(cls, to_concat):
        if not to_concat:
            raise ValueError("cannot concatenate an empty sequence")
        dtype = to_concat[0].dtype
        if any(array.dtype != dtype for array in to_concat[1:]):
            raise TypeError("all cyclic arrays must have the same dtype")
        return cls(np.concatenate([array._data for array in to_concat]), dtype)

    @property
    def dtype(self) -> CyclicDtype:
        return self._dtype

    @property
    def nbytes(self) -> int:
        return self._data.nbytes

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, item):
        result = self._data[item]
        if isinstance(result, np.ndarray):
            return type(self)(result, self.dtype)
        if self.dtype.is_categorical:
            return pd.NA if result == -1 else self.dtype.categories[int(result)]
        return pd.NA if np.isnan(result) else result.item()

    def __eq__(self, other):
        if isinstance(other, CyclicArray):
            if self.dtype != other.dtype:
                return np.zeros(len(self), dtype=bool)
            return self._data == other._data
        try:
            encoded = type(self)._from_sequence(
                np.repeat(other, len(self)), dtype=self.dtype
            )
        except (TypeError, ValueError):
            return np.zeros(len(self), dtype=bool)
        return self._data == encoded._data

    def isna(self) -> np.ndarray:
        if self.dtype.is_categorical:
            return self._data == -1
        return np.isnan(self._data)

    def take(self, indices, *, allow_fill=False, fill_value=None):
        sentinel = -1 if self.dtype.is_categorical else np.nan
        if fill_value is not None and fill_value is not pd.NA:
            encoded = type(self)._from_sequence([fill_value], dtype=self.dtype)
            sentinel = encoded._data[0]
        result = take(
            self._data,
            indices,
            allow_fill=allow_fill,
            fill_value=sentinel,
        )
        return type(self)(result, self.dtype)

    def copy(self) -> CyclicArray:
        return type(self)(self._data.copy(), self.dtype)

    def to_numpy(self, dtype=None, copy=False, na_value=pd.NA):
        values = np.asarray(list(self), dtype=object)
        if na_value is not pd.NA:
            values[pd.isna(values)] = na_value
        if dtype is not None:
            values = values.astype(dtype)
        elif copy:
            values = values.copy()
        return values

    def __array__(self, dtype=None, copy=None):
        return self.to_numpy(dtype=dtype, copy=bool(copy), na_value=np.nan)


def as_cyclic(data, *, period: Real | None = None) -> pd.Series:
    """Return ``data`` as a cyclic Series.

    Ordered categorical input supplies its complete category order.  Numeric
    input requires an explicit ``period``; it is never inferred from values.
    The input Series' index and name are preserved.
    """
    source = data if isinstance(data, pd.Series) else pd.Series(data)
    if isinstance(source.dtype, pd.CategoricalDtype):
        if period is not None:
            raise TypeError("period is not used for categorical cyclic data")
        if not source.dtype.ordered:
            raise TypeError("categorical input must have ordered=True")
        dtype = CyclicDtype(categories=source.cat.categories.tolist())
    else:
        if not pd.api.types.is_numeric_dtype(source.dtype):
            raise TypeError("input must be numeric or an ordered categorical")
        if period is None:
            raise TypeError("numeric cyclic data require an explicit period")
        dtype = CyclicDtype(period=period)
    array = CyclicArray._from_sequence(source.array, dtype=dtype)
    return pd.Series(array, index=source.index, name=source.name)


def is_cyclic(obj) -> bool:
    """Return whether ``obj`` is a cyclic Series, array, or dtype."""
    return isinstance(getattr(obj, "dtype", obj), CyclicDtype)


def _normalize_cycle_label(value: object) -> str:
    text = str(value).strip().casefold()
    text = re.sub(r"[_-]+", " ", text)
    return re.sub(r"\s+", " ", text)


def _normalize_name(name: object) -> str:
    if name is None:
        return ""
    text = str(name).strip().casefold()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _match_cycle_vocabulary(
    values: pd.Series,
    *,
    known_cycles: Mapping[str, Sequence[object]],
    min_coverage: float,
    allow_subset: bool,
) -> tuple[str, int, float] | None:
    if values.empty or not values.map(lambda value: isinstance(value, str)).all():
        return None

    observed = {
        label
        for value in values
        if (label := _normalize_cycle_label(value))
    }
    if len(observed) < 2:
        return None

    best: tuple[str, int, float] | None = None
    for name, cycle in known_cycles.items():
        normalized = tuple(_normalize_cycle_label(value) for value in cycle)
        cycle_set = set(normalized)
        if not normalized or "" in cycle_set:
            raise ValueError(f"Known cycle {name!r} contains a blank label")
        if len(cycle_set) != len(normalized):
            raise ValueError(
                f"Known cycle {name!r} contains duplicate normalized labels"
            )
        if not observed.issubset(cycle_set):
            continue

        coverage = len(observed) / len(cycle_set)
        if (allow_subset or observed == cycle_set) and coverage >= min_coverage:
            candidate = (name, len(normalized), coverage)
            if best is None or coverage > best[2]:
                best = candidate
    return best


def _fits_cyclic_range(
    values: pd.Series,
    period: float,
    *,
    atol: float = 1e-12,
) -> bool:
    minimum = float(values.min())
    maximum = float(values.max())

    # Conventional zero-based representation: [0, period).
    if minimum >= -atol and maximum < period:
        return True

    integral = bool(
        np.isclose(values, np.round(values), rtol=0, atol=0).all()
    )
    # Conventional one-based representation: [1, period].
    if integral and minimum >= 1 and maximum <= period + atol:
        return True

    # Angles are also commonly represented symmetrically.
    half_period = period / 2
    return bool(
        minimum >= -half_period - atol
        and maximum <= half_period + atol
    )


def _numeric_cycle_evidence(
    series: pd.Series,
    values: pd.Series,
    *,
    conventional_periods: Sequence[int],
    min_coverage: float,
    min_observations: int,
    numeric_threshold: float,
    use_name_hints: bool,
) -> CyclicEvidence | None:
    if len(values) < min_observations:
        return None
    parsed = pd.to_numeric(values, errors="coerce")
    if parsed.notna().mean() < numeric_threshold:
        return None
    numeric = parsed.dropna().astype(float)
    if numeric.empty or not np.isfinite(numeric).all():
        return None
    name = _normalize_name(series.name)
    if use_name_hints and name:
        for pattern, hinted_periods, cycle_name in _NUMERIC_NAME_HINTS:
            if not pattern.search(name):
                continue
            for period in hinted_periods:
                if _fits_cyclic_range(numeric, period):
                    return CyclicEvidence(
                        True,
                        "strong",
                        (
                            f"Column name suggests {cycle_name!r} and values "
                            f"fit a period of {period:g}"
                        ),
                        period=period,
                        cycle=cycle_name,
                    )
    # Without a name hint, require substantial coverage of an integer cycle.
    if not np.isclose(numeric, np.round(numeric), rtol=0, atol=0).all():
        return None
    observed = set(np.round(numeric).astype(np.int64).tolist())
    for period in conventional_periods:
        zero_based = set(range(period))
        one_based = set(range(1, period + 1))
        if observed.issubset(zero_based):
            allowed, encoding = zero_based, "zero-based"
        elif observed.issubset(one_based):
            allowed, encoding = one_based, "one-based"
        else:
            continue
        coverage = len(observed) / len(allowed)
        if coverage >= min_coverage:
            return CyclicEvidence(
                True,
                "plausible",
                (
                    f"Integer values cover {coverage:.0%} of a conventional "
                    f"{period}-position {encoding} cycle"
                ),
                period=period,
                cycle="unidentified_numeric",
            )
    return None


def cyclic_evidence(
    series: pd.Series,
    *,
    known_cycles: Mapping[str, Sequence[object]] = DEFAULT_CYCLES,
    conventional_periods: Sequence[int] = (4, 7, 12, 24, 60, 360),
    min_vocabulary_coverage: float = 0.50,
    min_numeric_coverage: float = 0.80,
    numeric_threshold: float = 0.95,
    min_observations: int = 3,
    use_name_hints: bool = True,
    allow_unordered_vocabulary: bool = True,
    allow_ordered_cardinality: bool = True,
    allow_numeric_range: bool = True,
) -> CyclicEvidence:
    """Return evidence that ``series`` could plausibly represent a cycle.

    An explicit :class:`CyclicDtype` is definitive. Recognised vocabularies
    and time-of-day objects are strong evidence. Conventional cardinality or
    numeric range/coverage is reported only as plausible evidence.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")
    for name, value in {
        "min_vocabulary_coverage": min_vocabulary_coverage,
        "min_numeric_coverage": min_numeric_coverage,
        "numeric_threshold": numeric_threshold,
    }.items():
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1")
    if isinstance(min_observations, bool) or not isinstance(min_observations, int):
        raise TypeError("min_observations must be an integer")
    if min_observations < 1:
        raise ValueError("min_observations must be positive")
    periods = tuple(conventional_periods)
    if any(
        isinstance(period, bool)
        or not isinstance(period, int)
        or period < 2
        for period in periods
    ):
        raise ValueError(
            "conventional_periods must contain integers of at least 2"
        )
    dtype = series.dtype
    if isinstance(dtype, CyclicDtype):
        period = dtype.cardinality if dtype.is_categorical else dtype.period
        return CyclicEvidence(
            True,
            "certain",
            "Series has an explicit CyclicDtype",
            period=period,
            cycle="declared",
        )
    # A timestamp is linear. Components such as month/hour may be cyclic.
    if (
        pd.api.types.is_datetime64_any_dtype(dtype)
        or pd.api.types.is_timedelta64_dtype(dtype)
        or isinstance(dtype, pd.PeriodDtype)
    ):
        return CyclicEvidence(
            False,
            "certain",
            "Datetime, timedelta, and Period values are not themselves cycles",
        )
    values = series.dropna()
    if values.empty:
        return CyclicEvidence(
            False,
            "unknown",
            "No non-missing values and no explicit cyclic dtype",
        )
    if values.map(lambda value: isinstance(value, time)).all():
        return CyclicEvidence(
            True,
            "strong",
            "All values are time-of-day objects",
            period=24 * 60 * 60,
            cycle="time_of_day",
        )
    categorical = isinstance(dtype, pd.CategoricalDtype)
    domain = pd.Series(dtype.categories, dtype=object) if categorical else values
    ordered = bool(dtype.ordered) if categorical else False
    vocabulary = _match_cycle_vocabulary(
        domain,
        known_cycles=known_cycles,
        min_coverage=min_vocabulary_coverage,
        allow_subset=allow_unordered_vocabulary or ordered,
    )
    if vocabulary is not None:
        cycle_name, period, coverage = vocabulary
        return CyclicEvidence(
            True,
            "strong",
            (
                f"Values match {coverage:.0%} of the recognised "
                f"{cycle_name!r} cycle"
            ),
            period=period,
            cycle=cycle_name,
        )
    if categorical and ordered and allow_ordered_cardinality:
        cardinality = len(dtype.categories)
        if cardinality in periods:
            return CyclicEvidence(
                True,
                "plausible",
                (
                    "Ordered categorical has a conventional cyclic "
                    f"cardinality of {cardinality}"
                ),
                period=cardinality,
                cycle="unidentified_categorical",
            )
    if allow_numeric_range:
        numeric = _numeric_cycle_evidence(
            series,
            values,
            conventional_periods=periods,
            min_coverage=min_numeric_coverage,
            min_observations=min_observations,
            numeric_threshold=numeric_threshold,
            use_name_hints=use_name_hints,
        )
        if numeric is not None:
            return numeric
    return CyclicEvidence(
        False,
        "unknown",
        "No declared type, recognised vocabulary, or plausible numeric cycle",
    )


def is_cyclic_like(series: pd.Series, **kwargs) -> bool:
    """Return whether a Series could plausibly represent a cycle."""
    return cyclic_evidence(series, **kwargs).matches


@register_series_accessor("cyclic")
class CyclicAccessor:
    """Circular operations available as ``series.cyclic``."""
    def __init__(self, series: pd.Series):
        if not is_cyclic(series):
            raise AttributeError("the .cyclic accessor requires CyclicDtype")
        self._series = series

    @property
    def period(self):
        dtype = self._series.dtype
        return dtype.cardinality if dtype.is_categorical else dtype.period

    def codes(self) -> pd.Series:
        """Return zero-based positions; missing values become nullable NA."""

        data = self._series.array._data.copy()
        if self._series.dtype.is_categorical:
            return pd.Series(
                pd.array(data, dtype="Int64"),
                index=self._series.index,
                name=self._series.name,
            ).mask(data == -1)
        return pd.Series(data, index=self._series.index, name=self._series.name)

    def shift(self, steps=1) -> pd.Series:
        """Move values around the cycle by ``steps``."""

        raw = self._series.array._data.copy()
        missing = self._series.array.isna()
        raw[~missing] = np.mod(raw[~missing] + steps, self.period)
        result = CyclicArray(raw, self._series.dtype)
        return pd.Series(result, index=self._series.index, name=self._series.name)

    def successor(self) -> pd.Series:
        return self.shift(1)

    def predecessor(self) -> pd.Series:
        return self.shift(-1)

    def distance(self, other, *, signed=False) -> pd.Series:
        """Shortest circular distance from each value to ``other``.

        With ``signed=True``, results lie in ``[-period/2, period/2)``.
        Otherwise their absolute values are returned.
        """

        target = CyclicArray._from_sequence(
            np.repeat(other, len(self._series)), dtype=self._series.dtype
        )._data
        source = self._series.array._data
        delta = np.mod(target - source + self.period / 2, self.period) - self.period / 2
        delta = delta.astype(float)
        delta[self._series.array.isna()] = np.nan
        if not signed:
            delta = np.abs(delta)
        return pd.Series(delta, index=self._series.index, name=self._series.name)


__all__ = [
    "DEFAULT_CYCLES",
    "CyclicArray",
    "CyclicDtype",
    "CyclicEvidence",
    "as_cyclic",
    "cyclic_evidence",
    "is_cyclic",
    "is_cyclic_like",
]
