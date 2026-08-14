from datetime import time

import numpy as np
import pandas as pd
import pytest

from cyclic_pandas import (
    DEFAULT_CYCLES,
    CyclicArray,
    CyclicDtype,
    CyclicEvidence,
    as_cyclic,
    cyclic_evidence,
    is_cyclic,
    is_cyclic_like,
)

WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


@pytest.fixture
def weekday_source():
    return pd.Series(
        pd.Categorical(
            ["Sun", "Mon", None, "Wed"],
            categories=WEEKDAYS,
            ordered=True,
        ),
        index=[10, 11, 12, 13],
        name="weekday",
    )


@pytest.fixture
def weekday_cycle(weekday_source):
    return as_cyclic(weekday_source)


@pytest.fixture
def degree_cycle():
    return as_cyclic(
        pd.Series([-1, 0, 361, None], index=list("abcd"), name="bearing"),
        period=360,
    )


class TestCyclicDtype:
    def test_categorical_metadata(self):
        dtype = CyclicDtype(categories=WEEKDAYS)
        assert dtype.is_categorical
        assert dtype.categories == WEEKDAYS
        assert dtype.cardinality == 7
        assert dtype.period is None
        assert dtype.name == "cyclic"
        assert dtype.type is object
        assert dtype.na_value is pd.NA
        assert dtype.construct_array_type() is CyclicArray

    def test_numeric_metadata(self):
        dtype = CyclicDtype(period=360)
        assert not dtype.is_categorical
        assert dtype.categories is None
        assert dtype.cardinality is None
        assert dtype.period == 360

    def test_repr_includes_parameters(self):
        assert repr(CyclicDtype(period=360)) == "CyclicDtype(period=360)"
        assert repr(CyclicDtype(categories=["a", "b"])) == (
            "CyclicDtype(categories=['a', 'b'])"
        )

    def test_equality_and_hash_include_metadata(self):
        left = CyclicDtype(categories=["a", "b"])
        same = CyclicDtype(categories=["a", "b"])
        rotated = CyclicDtype(categories=["b", "a"])
        assert left == same
        assert hash(left) == hash(same)
        assert left != rotated
        assert CyclicDtype(period=12) != CyclicDtype(period=24)

    @pytest.mark.parametrize(
        ("kwargs", "error", "message"),
        [
            ({}, TypeError, "exactly one"),
            ({"categories": ["a", "b"], "period": 2}, TypeError, "exactly one"),
            ({"categories": ["a"]}, ValueError, "at least two"),
            ({"categories": ["a", "a"]}, ValueError, "unique"),
            ({"categories": ["a", None]}, ValueError, "missing"),
            ({"period": 0}, ValueError, "positive"),
            ({"period": -1}, ValueError, "positive"),
            ({"period": np.nan}, ValueError, "positive"),
            ({"period": np.inf}, ValueError, "positive"),
            ({"period": "360"}, ValueError, "positive"),
        ],
    )
    def test_validation(self, kwargs, error, message):
        with pytest.raises(error, match=message):
            CyclicDtype(**kwargs)


class TestConstruction:
    def test_ordered_categorical_uses_declared_complete_domain(
        self, weekday_source, weekday_cycle
    ):
        assert is_cyclic(weekday_cycle)
        assert weekday_cycle.dtype.categories == WEEKDAYS
        assert weekday_cycle.tolist()[:2] == ["Sun", "Mon"]
        assert pd.isna(weekday_cycle.iloc[2])
        assert weekday_cycle.iloc[3] == "Wed"

    def test_preserves_index_and_name(self, weekday_source, weekday_cycle):
        assert weekday_cycle.index.equals(weekday_source.index)
        assert weekday_cycle.name == weekday_source.name

    def test_numeric_requires_explicit_period(self):
        with pytest.raises(TypeError, match="explicit period"):
            as_cyclic(pd.Series([359, 0, 1]))

    def test_numeric_normalizes_to_half_open_interval(self, degree_cycle):
        values = degree_cycle.tolist()
        assert values[:3] == [359.0, 0.0, 1.0]
        assert pd.isna(values[3])
        assert degree_cycle.dtype.period == 360

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(-720, 0.0), (-360, 0.0), (-1, 359.0), (360, 0.0), (721, 1.0)],
    )
    def test_numeric_normalization_multiple_wraps(self, value, expected):
        result = as_cyclic(pd.Series([value]), period=360)
        assert result.iloc[0] == expected

    def test_unordered_categorical_is_rejected(self):
        source = pd.Series(pd.Categorical(["a", "b"], ordered=False))
        with pytest.raises(TypeError, match="ordered=True"):
            as_cyclic(source)

    @pytest.mark.parametrize(
        "source",
        [pd.Series(["a", "b"]), pd.Series([object(), object()])],
    )
    def test_non_numeric_non_categorical_is_rejected(self, source):
        with pytest.raises(TypeError, match="numeric or an ordered categorical"):
            as_cyclic(source, period=2)

    def test_unknown_categorical_value_is_rejected(self):
        dtype = CyclicDtype(categories=["a", "b"])
        with pytest.raises(ValueError, match="not in the cyclic categories"):
            CyclicArray._from_sequence(["a", "c"], dtype=dtype)


class TestExtensionArray:
    def test_scalar_and_slice_getitem(self, weekday_cycle):
        assert weekday_cycle.array[0] == "Sun"
        assert pd.isna(weekday_cycle.array[2])
        sliced = weekday_cycle.array[1:]
        assert isinstance(sliced, CyclicArray)
        assert sliced.dtype == weekday_cycle.dtype
        assert sliced.tolist()[0] == "Mon"

    def test_isna(self, weekday_cycle, degree_cycle):
        assert weekday_cycle.array.isna().tolist() == [False, False, True, False]
        assert degree_cycle.array.isna().tolist() == [False, False, False, True]

    def test_take_without_fill(self, weekday_cycle):
        result = weekday_cycle.array.take([3, 0])
        assert isinstance(result, CyclicArray)
        assert result.dtype == weekday_cycle.dtype
        assert result.tolist() == ["Wed", "Sun"]

    def test_take_with_default_missing_fill(self, weekday_cycle):
        result = weekday_cycle.array.take([0, -1], allow_fill=True)
        assert result[0] == "Sun"
        assert pd.isna(result[1])

    def test_take_with_explicit_fill(self, weekday_cycle):
        result = weekday_cycle.array.take(
            [0, -1], allow_fill=True, fill_value="Fri"
        )
        assert result.tolist() == ["Sun", "Fri"]

    def test_copy_has_independent_backing_storage(self, degree_cycle):
        copied = degree_cycle.array.copy()
        assert copied is not degree_cycle.array
        assert not np.shares_memory(copied._data, degree_cycle.array._data)
        np.testing.assert_array_equal(copied._data, degree_cycle.array._data)

    def test_to_numpy_replaces_missing_value(self, weekday_cycle):
        result = weekday_cycle.array.to_numpy(na_value=None)
        assert result.tolist() == ["Sun", "Mon", None, "Wed"]
        assert result.dtype == object

    def test_scalar_equality(self, weekday_cycle):
        assert (weekday_cycle == "Mon").tolist() == [False, True, False, False]
        assert (weekday_cycle == "not-a-category").tolist() == [False] * 4

    def test_array_equality_with_different_dtype_is_false(self):
        left = as_cyclic(
            pd.Series(pd.Categorical(["a"], categories=["a", "b"], ordered=True))
        )
        right = as_cyclic(
            pd.Series(pd.Categorical(["a"], categories=["a", "c"], ordered=True))
        )
        assert (left.array == right.array).tolist() == [False]

    def test_concat_preserves_dtype(self, weekday_cycle):
        result = pd.concat([weekday_cycle, weekday_cycle], ignore_index=True)
        assert is_cyclic(result)
        assert result.dtype == weekday_cycle.dtype
        assert len(result) == 8

    def test_concat_same_type_rejects_different_metadata(self):
        first = as_cyclic(pd.Series([0, 1]), period=12).array
        second = as_cyclic(pd.Series([0, 1]), period=24).array
        with pytest.raises(TypeError, match="same dtype"):
            CyclicArray._concat_same_type([first, second])

    def test_constructor_rejects_non_1d_data(self):
        with pytest.raises(ValueError, match="one-dimensional"):
            CyclicArray(
                np.array([[1.0, 2.0]], dtype=np.float64),
                CyclicDtype(period=12),
            )


class TestAccessor:
    def test_period(self, weekday_cycle, degree_cycle):
        assert weekday_cycle.cyclic.period == 7
        assert degree_cycle.cyclic.period == 360

    def test_categorical_codes_are_nullable(self, weekday_cycle):
        result = weekday_cycle.cyclic.codes()
        assert str(result.dtype) == "Int64"
        assert result.tolist()[:2] == [6, 0]
        assert pd.isna(result.iloc[2])
        assert result.iloc[3] == 2
        assert result.index.equals(weekday_cycle.index)
        assert result.name == weekday_cycle.name

    def test_numeric_codes_return_normalized_values(self, degree_cycle):
        result = degree_cycle.cyclic.codes()
        assert result.tolist()[:3] == [359.0, 0.0, 1.0]
        assert pd.isna(result.iloc[3])

    @pytest.mark.parametrize(
        ("steps", "expected"),
        [
            (1, ["Mon", "Tue", pd.NA, "Thu"]),
            (-1, ["Sat", "Sun", pd.NA, "Tue"]),
            (8, ["Mon", "Tue", pd.NA, "Thu"]),
        ],
    )
    def test_categorical_shift_wraps(self, weekday_cycle, steps, expected):
        result = weekday_cycle.cyclic.shift(steps).tolist()
        assert result[:2] == expected[:2]
        assert pd.isna(result[2])
        assert result[3] == expected[3]

    def test_numeric_shift_wraps_and_preserves_metadata(self, degree_cycle):
        result = degree_cycle.cyclic.shift(2)
        assert result.tolist()[:3] == [1.0, 2.0, 3.0]
        assert pd.isna(result.iloc[3])
        assert result.index.equals(degree_cycle.index)
        assert result.name == degree_cycle.name
        assert result.dtype == degree_cycle.dtype

    def test_successor_and_predecessor(self, weekday_cycle):
        assert weekday_cycle.cyclic.successor().tolist()[:2] == ["Mon", "Tue"]
        assert weekday_cycle.cyclic.predecessor().tolist()[:2] == ["Sat", "Sun"]

    def test_unsigned_shortest_distance(self, weekday_cycle):
        result = weekday_cycle.cyclic.distance("Tue")
        assert result.tolist()[:2] == [2.0, 1.0]
        assert pd.isna(result.iloc[2])
        assert result.iloc[3] == 1.0

    def test_signed_shortest_distance(self, degree_cycle):
        result = degree_cycle.cyclic.distance(1, signed=True)
        assert result.tolist()[:3] == [2.0, 1.0, 0.0]
        assert pd.isna(result.iloc[3])

    def test_distance_rejects_invalid_category(self, weekday_cycle):
        with pytest.raises(ValueError, match="not in the cyclic categories"):
            weekday_cycle.cyclic.distance("Funday")

    def test_accessor_rejects_non_cyclic_series(self):
        with pytest.raises(AttributeError, match="requires CyclicDtype"):
            pd.Series([1, 2, 3]).cyclic  # noqa: B018


class TestEvidence:
    def test_explicit_dtype_is_certain(self, weekday_cycle):
        result = cyclic_evidence(weekday_cycle)
        assert result == CyclicEvidence(
            True,
            "certain",
            "Series has an explicit CyclicDtype",
            period=7,
            cycle="declared",
        )

    @pytest.mark.parametrize(
        ("values", "cycle", "period"),
        [
            (["January", "February", "March", "April", "May", "June"], "month", 12),
            (["MON", "tue", "Wed", "Thu"], "weekday_abbreviation", 7),
            (["North", "East", "South", "West"], "compass_4", 4),
            (["spring", "summer", "autumn", "winter"], "season", 4),
        ],
    )
    def test_recognized_vocabulary(self, values, cycle, period):
        result = cyclic_evidence(pd.Series(values))
        assert result.matches
        assert result.confidence == "strong"
        assert result.cycle == cycle
        assert result.period == period

    def test_label_normalization_handles_hyphen_and_whitespace(self):
        custom = {"phases": ("phase one", "phase two", "phase three")}
        result = cyclic_evidence(
            pd.Series([" Phase-One ", "phase_two", "phase three"]),
            known_cycles=custom,
        )
        assert result.matches
        assert result.cycle == "phases"

    def test_vocabulary_below_coverage_is_rejected(self):
        result = cyclic_evidence(pd.Series(["Mon", "Tue", "Mon"]))
        assert not result.matches

    def test_full_vocabulary_can_be_required(self):
        partial = pd.Series(["Mon", "Tue", "Wed", "Thu"])
        result = cyclic_evidence(partial, allow_unordered_vocabulary=False)
        assert not result.matches

    def test_ordered_categorical_conventional_cardinality_is_plausible(self):
        source = pd.Series(pd.Categorical(
            ["a", "b"], categories=["a", "b", "c", "d"], ordered=True
        ))
        result = cyclic_evidence(source)
        assert result.matches
        assert result.confidence == "plausible"
        assert result.period == 4
        assert result.cycle == "unidentified_categorical"

    def test_ordered_cardinality_inference_can_be_disabled(self):
        source = pd.Series(pd.Categorical(
            ["a", "b"], categories=["a", "b", "c", "d"], ordered=True
        ))
        assert not cyclic_evidence(
            source, allow_ordered_cardinality=False
        ).matches

    @pytest.mark.parametrize(
        ("name", "values", "cycle", "period"),
        [
            ("month_number", [1, 2, 11, 12], "month", 12),
            ("day-of-week", [0, 1, 5, 6], "weekday", 7),
            ("hour_of_day", [0, 6, 12, 23], "hour", 24),
            ("wind_direction_degrees", [359, 0, 1], "angle_degrees", 360),
            ("heading_radians", [-np.pi, 0, np.pi], "angle_radians", 2 * np.pi),
        ],
    )
    def test_numeric_name_hints(self, name, values, cycle, period):
        result = cyclic_evidence(pd.Series(values, name=name))
        assert result.matches
        assert result.confidence == "strong"
        assert result.cycle == cycle
        assert result.period == pytest.approx(period)

    def test_name_hints_can_be_disabled(self):
        result = cyclic_evidence(
            pd.Series([359, 0, 1], name="wind_direction_degrees"),
            use_name_hints=False,
        )
        assert not result.matches

    def test_unnamed_nearly_complete_integer_cycle_is_plausible(self):
        result = cyclic_evidence(
            pd.Series([0, 1, 2, 3, 4, 5]),
            conventional_periods=(7,),
            min_numeric_coverage=0.80,
        )
        assert result.matches
        assert result.confidence == "plausible"
        assert result.period == 7

    def test_numeric_range_inference_can_be_disabled(self):
        source = pd.Series([1, 2, 11, 12], name="month_number")
        assert not cyclic_evidence(source, allow_numeric_range=False).matches

    def test_time_of_day_objects_are_cyclic(self):
        result = cyclic_evidence(pd.Series([time(23, 30), time(0, 15)]))
        assert result.matches
        assert result.confidence == "strong"
        assert result.cycle == "time_of_day"
        assert result.period == 24 * 60 * 60

    @pytest.mark.parametrize(
        "series",
        [
            pd.Series(pd.date_range("2025-01-01", periods=3)),
            pd.Series(pd.to_timedelta([1, 2, 3], unit="h")),
            pd.Series(pd.period_range("2025-01", periods=3, freq="M")),
        ],
    )
    def test_linear_time_types_are_not_themselves_cyclic(self, series):
        result = cyclic_evidence(series)
        assert not result.matches
        assert result.confidence == "certain"

    @pytest.mark.parametrize("series", [pd.Series(dtype=object), pd.Series([None])])
    def test_empty_or_all_missing_is_unknown(self, series):
        result = cyclic_evidence(series)
        assert not result.matches
        assert result.confidence == "unknown"

    def test_custom_vocabulary(self):
        cycles = {**DEFAULT_CYCLES, "traffic_light": ("red", "amber", "green")}
        result = cyclic_evidence(
            pd.Series(["red", "amber", "green"]), known_cycles=cycles
        )
        assert result.matches
        assert result.cycle == "traffic_light"

    @pytest.mark.parametrize(
        "cycles",
        [
            {"invalid": ("", "valid")},
            {"invalid": ("phase-one", "phase one")},
        ],
    )
    def test_invalid_custom_vocabulary_is_rejected(self, cycles):
        with pytest.raises(ValueError, match="blank|duplicate"):
            cyclic_evidence(
                pd.Series(["valid", "other"]),
                known_cycles=cycles,
                min_vocabulary_coverage=0,
            )

    @pytest.mark.parametrize(
        ("kwargs", "error", "message"),
        [
            ({"min_vocabulary_coverage": -0.1}, ValueError, "between 0 and 1"),
            ({"min_numeric_coverage": 1.1}, ValueError, "between 0 and 1"),
            ({"numeric_threshold": 2}, ValueError, "between 0 and 1"),
            ({"min_observations": 0}, ValueError, "positive"),
            ({"min_observations": True}, TypeError, "integer"),
            ({"conventional_periods": (1, 7)}, ValueError, "at least 2"),
            ({"conventional_periods": (4.0, 7)}, ValueError, "integers"),
        ],
    )
    def test_evidence_configuration_validation(self, kwargs, error, message):
        with pytest.raises(error, match=message):
            cyclic_evidence(pd.Series([1, 2, 3]), **kwargs)

    def test_requires_series(self):
        with pytest.raises(TypeError, match="pandas Series"):
            cyclic_evidence(["Mon", "Tue", "Wed"])

    def test_boolean_wrapper_matches_evidence(self, weekday_cycle):
        assert is_cyclic_like(weekday_cycle)
        assert not is_cyclic_like(pd.Series([100, 200, 300]))


class TestPublicPredicates:
    def test_is_cyclic_accepts_series_array_and_dtype(self, weekday_cycle):
        assert is_cyclic(weekday_cycle)
        assert is_cyclic(weekday_cycle.array)
        assert is_cyclic(weekday_cycle.dtype)

    @pytest.mark.parametrize(
        "value",
        [pd.Series([1, 2]), np.array([1, 2]), np.dtype("float64"), object()],
    )
    def test_is_cyclic_rejects_other_objects(self, value):
        assert not is_cyclic(value)
