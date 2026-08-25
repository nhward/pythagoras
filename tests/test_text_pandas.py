import sys
from pathlib import Path

path = Path(__file__).resolve().parent.parent / "app"
if path not in sys.path:
    sys.path.insert(0, path)

import numpy as np
import pandas as pd
import pytest
import text_pandas as tp
from text_pandas import (
    TextArray,
    TextDtype,
    TextEvidence,
    as_text,
    is_text,
    is_text_like,
    text_evidence,
)

PROSE = [
    "The first response contains enough words to look like genuine free text.",
    "A second distinct response also has spaces and a comfortably long length.",
    "This final observation is another unique sentence written by a respondent.",
]


@pytest.fixture
def text_source():
    return pd.Series(
        [PROSE[0], None, PROSE[1]],
        index=[10, 20, 30],
        name="comments",
        dtype=object,
    )


@pytest.fixture
def text_series(text_source):
    return as_text(text_source)


class TestTextDtype:
    def test_metadata(self):
        dtype = TextDtype()
        assert dtype.name == "text"
        assert dtype.type is str
        assert dtype.kind == "O"
        assert dtype.na_value is pd.NA
        assert dtype.construct_array_type() is TextArray

    def test_repr(self):
        assert repr(TextDtype()) == "TextDtype()"

    def test_instances_compare_and_hash_equally(self):
        assert TextDtype() == TextDtype()
        assert hash(TextDtype()) == hash(TextDtype())

    def test_pandas_accepts_explicit_dtype_instance(self):
        dtype = TextDtype()
        assert pd.api.types.pandas_dtype(dtype) is dtype


class TestConstruction:
    def test_strings_and_missing_values(self, text_series):
        assert text_series.iloc[0] == PROSE[0]
        assert text_series.iloc[1] is pd.NA
        assert text_series.iloc[2] == PROSE[1]
        assert is_text(text_series)

    def test_preserves_index_and_name(self, text_source, text_series):
        assert text_series.index.equals(text_source.index)
        assert text_series.name == text_source.name

    @pytest.mark.parametrize("missing", [None, pd.NA, np.nan, pd.NaT])
    def test_common_missing_scalars_are_preserved(self, missing):
        result = as_text(["present", missing])
        assert result.iloc[0] == "present"
        assert result.iloc[1] is pd.NA

    @pytest.mark.parametrize("value", [1, 1.5, True, object(), ["nested"]])
    def test_non_strings_raise_by_default(self, value):
        with pytest.raises(TypeError, match="not a text scalar"):
            as_text([value])

    def test_coerce_converts_nonmissing_values_to_strings(self):
        result = as_text(pd.Series([1, 2.5, True, None]), errors="coerce")
        assert result.tolist()[:3] == ["1", "2.5", "True"]
        assert result.iloc[3] is pd.NA

    @pytest.mark.parametrize("errors", ["ignore", "convert", "", None])
    def test_invalid_errors_option(self, errors):
        with pytest.raises(ValueError, match="errors"):
            as_text(["value"], errors=errors)

    def test_existing_text_series_is_returned_unchanged(self, text_series):
        assert as_text(text_series) is text_series

    def test_text_array_sequence_rejects_other_dtype(self):
        with pytest.raises(TypeError, match="TextDtype"):
            TextArray._from_sequence(["a"], dtype=np.dtype("object"))


class TestExtensionArray:
    def test_scalar_and_slice_getitem(self, text_series):
        assert text_series.array[0] == PROSE[0]
        sliced = text_series.array[1:]
        assert isinstance(sliced, TextArray)
        assert isinstance(sliced.dtype, TextDtype)
        assert sliced[0] is pd.NA

    def test_isna(self, text_series):
        assert text_series.array.isna().tolist() == [False, True, False]

    def test_take_without_fill(self, text_series):
        result = text_series.array.take([2, 0])
        assert isinstance(result, TextArray)
        assert result.tolist() == [PROSE[1], PROSE[0]]

    def test_take_with_default_missing_fill(self, text_series):
        result = text_series.array.take([0, -1], allow_fill=True)
        assert result[0] == PROSE[0]
        assert result[1] is pd.NA

    def test_take_with_explicit_string_fill(self, text_series):
        result = text_series.array.take(
            [0, -1], allow_fill=True, fill_value="replacement"
        )
        assert result.tolist() == [PROSE[0], "replacement"]

    def test_take_rejects_non_string_fill(self, text_series):
        with pytest.raises(TypeError, match="not a text scalar"):
            text_series.array.take([0, -1], allow_fill=True, fill_value=123)

    def test_copy_has_independent_storage(self, text_series):
        copied = text_series.array.copy()
        assert copied is not text_series.array
        assert not np.shares_memory(copied._data, text_series.array._data)
        assert copied.tolist() == text_series.tolist()

    def test_to_numpy_replaces_missing_value(self, text_series):
        result = text_series.array.to_numpy(na_value=None)
        assert result.dtype == object
        assert result.tolist() == [PROSE[0], None, PROSE[1]]

    def test_numpy_array_protocol(self, text_series):
        result = np.asarray(text_series.array)
        assert result.dtype == object
        assert result[0] == PROSE[0]
        assert result[1] is pd.NA

    def test_scalar_equality(self, text_series):
        assert (text_series.array == PROSE[0]).tolist() == [True, False, False]

    def test_missing_scalar_equality(self, text_series):
        assert (text_series.array == pd.NA).tolist() == [False, True, False]

    def test_invalid_scalar_equality_is_all_false(self, text_series):
        assert (text_series.array == 123).tolist() == [False, False, False]

    def test_array_equality(self):
        left = as_text(["a", None, "c"]).array
        right = as_text(["a", None, "x"]).array
        assert (left == right).tolist() == [True, True, False]

    def test_different_length_array_equality_is_all_false(self):
        left = as_text(["a", "b"]).array
        right = as_text(["a"]).array
        assert (left == right).tolist() == [False, False]

    def test_concat_preserves_dtype(self, text_series):
        result = pd.concat([text_series, text_series], ignore_index=True)
        assert is_text(result)
        assert len(result) == 6

    def test_concat_empty_sequence_is_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            TextArray._concat_same_type([])

    def test_nbytes_is_positive(self, text_series):
        assert text_series.array.nbytes > text_series.array._data.nbytes


class TestPredicates:
    def test_is_text_accepts_series_array_and_dtype(self, text_series):
        assert is_text(text_series)
        assert is_text(text_series.array)
        assert is_text(text_series.dtype)

    @pytest.mark.parametrize(
        "value",
        [pd.Series(["ordinary"]), pd.StringDtype(), np.dtype("object"), str],
    )
    def test_is_text_rejects_other_string_representations(self, value):
        assert not is_text(value)

    def test_boolean_wrapper_matches_evidence(self):
        source = pd.Series(PROSE)
        assert is_text_like(source) == text_evidence(source).matches


class TestEvidence:
    def test_explicit_dtype_is_certain(self, text_series):
        assert text_evidence(text_series) == TextEvidence(
            True,
            "certain",
            "Series has an explicit TextDtype",
            1.0,
        )

    def test_prose_is_strong_evidence(self):
        result = text_evidence(pd.Series(PROSE))
        assert result.matches
        assert result.confidence == "strong"
        assert result.success_ratio == 1.0
        assert result.cardinality == 3
        assert result.uniqueness_ratio == 1.0
        assert result.median_length >= 20
        assert result.whitespace_ratio == 1.0
        assert "high cardinality" in result.reason

    @pytest.mark.parametrize(
        ("values", "failed_measure"),
        [
            (["same long phrase repeated"] * 5, "cardinality"),
            (["a b", "c d", "e f"], "median length"),
            (["a" * 30, "b" * 30, "c" * 30], "whitespace"),
        ],
    )
    def test_each_required_signal_can_fail(self, values, failed_measure):
        result = text_evidence(pd.Series(values))
        assert not result.matches
        assert failed_measure in result.reason

    def test_short_unique_codes_are_not_text(self):
        result = text_evidence(pd.Series(["NZ01", "NZ02", "NZ03", "NZ04"]))
        assert not result.matches
        assert result.uniqueness_ratio == 1.0
        assert "median length" in result.reason
        assert "whitespace" in result.reason

    def test_numeric_dtype_is_certainly_not_text(self):
        result = text_evidence(pd.Series([1, 2, 3]))
        assert result == TextEvidence(
            False, "certain", "Series dtype is not string-like", 0.0
        )

    def test_empty_series_has_unknown_evidence(self):
        result = text_evidence(pd.Series([], dtype=object))
        assert not result.matches
        assert result.confidence == "unknown"
        assert result.reason == "No non-missing values"

    def test_all_missing_series_has_unknown_evidence(self):
        result = text_evidence(pd.Series([None, pd.NA], dtype=object))
        assert not result.matches
        assert result.reason == "No non-missing values"

    def test_blank_strings_are_missing_by_default(self):
        result = text_evidence(pd.Series(["", "  ", "\t"]))
        assert not result.matches
        assert result.reason == "No nonblank string values"

    def test_blank_strings_can_be_retained(self):
        result = text_evidence(
            pd.Series(["", "  ", "\t"]), blank_as_missing=False,
            min_median_length=0, whitespace_threshold=0,
            uniqueness_threshold=0.30,
        )
        assert result.matches
        assert result.cardinality == 1

    def test_string_threshold_controls_mixed_object_data(self):
        source = pd.Series([PROSE[0], PROSE[1], 42], dtype=object)
        assert not text_evidence(source, string_threshold=0.95).matches
        result = text_evidence(source, string_threshold=2 / 3)
        assert result.matches
        assert result.success_ratio == pytest.approx(2 / 3)

    def test_threshold_parameters_control_classification(self):
        source = pd.Series(["a b", "c d", "e f"])
        assert not is_text_like(source)
        assert is_text_like(
            source,
            min_median_length=3,
            uniqueness_threshold=1.0,
            whitespace_threshold=1.0,
        )

    def test_limit_samples_across_whole_series(self):
        source = pd.Series([PROSE[0], "middle code", PROSE[1]])
        result = text_evidence(source, limit=2)
        assert result.matches
        assert result.cardinality == 2

    def test_categorical_strings_are_eligible(self):
        source = pd.Series(pd.Categorical(PROSE, ordered=False))
        assert text_evidence(source).matches

    def test_string_dtype_is_eligible(self):
        source = pd.Series(PROSE, dtype="string")
        assert text_evidence(source).matches

    @pytest.mark.parametrize(
        ("kwargs", "error", "message"),
        [
            ({"high_cardinality": True}, TypeError, "integer"),
            ({"high_cardinality": 1.5}, TypeError, "integer"),
            ({"high_cardinality": 0}, ValueError, "greater than zero"),
            ({"uniqueness_threshold": -0.1}, ValueError, "between 0 and 1"),
            ({"uniqueness_threshold": 1.1}, ValueError, "between 0 and 1"),
            ({"whitespace_threshold": -0.1}, ValueError, "between 0 and 1"),
            ({"string_threshold": 1.1}, ValueError, "between 0 and 1"),
            ({"min_median_length": -1}, ValueError, "non-negative"),
            ({"limit": True}, TypeError, "integer or None"),
            ({"limit": 1.5}, TypeError, "integer or None"),
            ({"limit": 0}, ValueError, "greater than zero"),
        ],
    )
    def test_configuration_validation(self, kwargs, error, message):
        with pytest.raises(error, match=message):
            text_evidence(pd.Series(PROSE), **kwargs)

    def test_non_series_is_rejected(self):
        with pytest.raises(TypeError, match="pandas Series"):
            text_evidence(PROSE)


class TestAccessor:
    def test_accessor_rejects_ordinary_string_series(self):
        with pytest.raises(AttributeError, match="requires TextDtype"):
            tp.TextAccessor(pd.Series(["ordinary"]))

    def test_lengths_are_nullable_and_preserve_metadata(self, text_series):
        result = text_series.text.lengths()
        assert str(result.dtype) == "Int64"
        assert result.iloc[0] == len(PROSE[0])
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == len(PROSE[1])
        assert result.index.equals(text_series.index)
        assert result.name == text_series.name

    def test_word_counts_are_nullable_and_preserve_metadata(self, text_series):
        result = text_series.text.word_counts()
        assert str(result.dtype) == "Int64"
        assert result.iloc[0] == len(PROSE[0].split())
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == len(PROSE[1].split())
        assert result.index.equals(text_series.index)
        assert result.name == text_series.name

    def test_to_string_returns_pandas_string_dtype(self, text_series):
        result = text_series.text.to_string()
        assert isinstance(result.dtype, pd.StringDtype)
        assert not is_text(result)
        assert result.iloc[0] == PROSE[0]
        assert pd.isna(result.iloc[1])
        assert result.index.equals(text_series.index)
        assert result.name == text_series.name


def test_public_exports_are_complete():
    assert set(tp.__all__) == {
        "TextAccessor",
        "TextArray",
        "TextDtype",
        "TextEvidence",
        "as_text",
        "is_text",
        "is_text_like",
        "text_evidence",
    }
