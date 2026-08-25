import sys
from pathlib import Path

path = Path(__file__).resolve().parent.parent / "app"
if path not in sys.path:
    sys.path.insert(0, path)

import list_pandas as lp
import numpy as np
import pandas as pd
import pytest
from list_pandas import (
    DEFAULT_DELIMITERS,
    ListArray,
    ListDtype,
    ListLikeEvidence,
    as_list,
    is_list,
    is_list_like,
    list_like_evidence,
)


@pytest.fixture
def basket_source():
    return pd.Series(
        [["apples", "milk"], ("bread", "cheese"), None, {"jam", "biscuits"}],
        index=[10, 20, 30, 40],
        name="basket",
        dtype=object,
    )


@pytest.fixture
def basket(basket_source):
    return as_list(basket_source)


class TestListDtype:
    def test_metadata(self):
        dtype = ListDtype()
        assert dtype.name == "basket"
        assert dtype.type is list
        assert dtype.kind == "O"
        assert dtype.na_value is pd.NA
        assert dtype.item_type is None
        assert dtype.construct_array_type() is ListArray

    def test_item_type_metadata_and_repr(self):
        assert repr(ListDtype(str)) == "ListDtype(item_type=str)"
        assert repr(ListDtype((int, str))) == "ListDtype(item_type=int, str)"
        assert repr(ListDtype()) == "ListDtype()"

    def test_equality_and_hash_include_metadata(self):
        assert ListDtype(str) == ListDtype(str)
        assert hash(ListDtype(str)) == hash(ListDtype(str))
        assert ListDtype(str) != ListDtype(int)
        assert ListDtype() != ListDtype(str)

    @pytest.mark.parametrize("item_type", ["str", (), (str, "int"), 1, [str]])
    def test_invalid_item_type(self, item_type):
        with pytest.raises(TypeError, match="item_type"):
            ListDtype(item_type)

    def test_pandas_accepts_explicit_dtype_instance(self):
        dtype = ListDtype(str)
        assert pd.api.types.pandas_dtype(dtype) is dtype

    def test_name_does_not_impersonate_arrow_list_dtype(self):
        assert not str(ListDtype()).startswith("list")


class TestConstruction:
    def test_collections_are_normalized(self, basket):
        assert basket.iloc[0] == ["apples", "milk"]
        assert basket.iloc[1] == ["bread", "cheese"]
        assert basket.iloc[2] is pd.NA
        assert basket.iloc[3] == ["biscuits", "jam"]

    def test_preserves_index_and_name(self, basket_source, basket):
        assert basket.index.equals(basket_source.index)
        assert basket.name == basket_source.name

    def test_infers_homogeneous_item_type(self, basket):
        assert basket.dtype.item_type is str

    def test_infers_multiple_item_types_deterministically(self):
        result = as_list([[1, "one"], [2, "two"]])
        assert result.dtype.item_type == (int, str)

    def test_empty_and_missing_lists_have_unknown_item_type(self):
        result = as_list([[], None])
        assert result.dtype.item_type is None
        assert result.iloc[0] == []
        assert result.iloc[1] is pd.NA

    def test_one_dimensional_numpy_arrays_are_supported(self):
        result = as_list(pd.Series([np.array([1, 2]), np.array([3, 4])]))
        assert result.tolist() == [[1, 2], [3, 4]]

    @pytest.mark.parametrize("value", [{"a": 1}, np.array([[1, 2], [3, 4]]), 42])
    def test_unsupported_collection_values_are_rejected(self, value):
        with pytest.raises((TypeError, ValueError)):
            as_list(pd.Series([value], dtype=object))

    def test_existing_list_series_is_returned_unchanged(self, basket):
        assert as_list(basket) is basket


class TestExtensionArray:
    def test_scalar_and_slice_getitem(self, basket):
        assert basket.array[0] == ["apples", "milk"]
        sliced = basket.array[1:]
        assert isinstance(sliced, ListArray)
        assert sliced.dtype == basket.dtype
        assert sliced[0] == ["bread", "cheese"]

    def test_isna(self, basket):
        assert basket.array.isna().tolist() == [False, False, True, False]

    def test_take_without_fill(self, basket):
        result = basket.array.take([3, 0])
        assert isinstance(result, ListArray)
        assert result.tolist() == [["biscuits", "jam"], ["apples", "milk"]]

    def test_take_with_missing_fill(self, basket):
        result = basket.array.take([0, -1], allow_fill=True)
        assert result[0] == ["apples", "milk"]
        assert result[1] is pd.NA

    def test_take_with_explicit_list_fill(self, basket):
        result = basket.array.take([0, -1], allow_fill=True, fill_value=("other",))
        assert result[1] == ["other"]

    def test_copy_duplicates_inner_lists(self, basket):
        copied = basket.array.copy()
        copied[0].append("eggs")
        assert copied[0] == ["apples", "milk", "eggs"]
        assert basket.array[0] == ["apples", "milk"]

    def test_to_numpy_replaces_missing_value(self, basket):
        result = basket.array.to_numpy(na_value=None)
        assert result.dtype == object
        assert result[2] is None
        assert result[0] == ["apples", "milk"]

    def test_scalar_equality(self, basket):
        assert (basket.array == ["apples", "milk"]).tolist() == [True, False, False, False]

    def test_array_equality(self):
        left = as_list([[1, 2], None, [3]]).array
        right = as_list([(1, 2), None, [4]]).array
        assert (left == right).tolist() == [True, True, False]

    def test_concat_preserves_common_dtype(self, basket):
        result = pd.concat([basket, basket], ignore_index=True)
        assert is_list(result)
        assert result.dtype == basket.dtype
        assert len(result) == 8

    def test_concat_different_item_types_uses_unspecified_item_type(self):
        left = as_list([[1, 2]]).array
        right = as_list([["a", "b"]]).array
        result = ListArray._concat_same_type([left, right])
        assert result.dtype == ListDtype()
        assert result.tolist() == [[1, 2], ["a", "b"]]

    def test_concat_empty_sequence_is_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            ListArray._concat_same_type([])

    def test_constructor_requires_list_dtype(self):
        with pytest.raises(TypeError, match="ListDtype"):
            ListArray(np.array([], dtype=object), np.dtype("object"))


class TestPredicatesAndEvidence:
    def test_is_list_accepts_series_array_and_dtype(self, basket):
        assert is_list(basket)
        assert is_list(basket.array)
        assert is_list(basket.dtype)

    @pytest.mark.parametrize("value", [pd.Series([[1, 2]], dtype=object), np.dtype("object"), list])
    def test_is_list_rejects_untyped_values(self, value):
        assert not is_list(value)

    def test_explicit_dtype_is_certain(self, basket):
        assert list_like_evidence(basket) == ListLikeEvidence(
            True, "certain", "list-dtype", None, 1.0,
            "Series has an explicit ListDtype",
        )

    @pytest.mark.parametrize("values", [
        [[1, 2], [3, 4]],
        [(1, 2), (3, 4)],
        [{1, 2}, {3, 4}],
        [np.array([1, 2]), np.array([3, 4])],
    ])
    def test_supported_collections_are_strong_evidence(self, values):
        result = list_like_evidence(pd.Series(values, dtype=object))
        assert result.matches
        assert result.confidence == "strong"
        assert result.representation == "collection"
        assert result.success_ratio == 1.0

    def test_json_arrays_are_strong_evidence(self):
        result = list_like_evidence(pd.Series(['["a", "b"]', '["c", "d"]']))
        assert result.matches
        assert result.confidence == "strong"
        assert result.representation == "json-array"
        assert result.delimiter is None

    @pytest.mark.parametrize("delimiter", DEFAULT_DELIMITERS)
    def test_default_delimiters_are_detected(self, delimiter):
        result = list_like_evidence(pd.Series([
            delimiter.join(("apples", "milk")),
            delimiter.join(("bread", "cheese")),
        ]))
        assert result.matches
        assert result.confidence == "plausible"
        assert result.representation == "delimited-string"
        assert result.delimiter == delimiter

    def test_csv_quoting_is_respected(self):
        result = as_list(pd.Series(['"Smith, John",bread', 'milk,"Doe, Jane"']), delimiter=",")
        assert result.tolist() == [["Smith, John", "bread"], ["milk", "Doe, Jane"]]

    def test_prose_is_not_list_like(self):
        series = pd.Series([
            "This is an ordinary sentence, with a dependent clause.",
            "Another prose sentence, containing several ordinary words.",
        ])
        assert not is_list_like(series)

    def test_empty_or_missing_series_has_no_evidence(self):
        result = list_like_evidence(pd.Series([None, pd.NA, " "]))
        assert not result.matches
        assert result.success_ratio == 0.0
        assert "No non-missing" in result.reason

    def test_singleton_collections_are_weak_by_default(self):
        result = list_like_evidence(pd.Series([[1], [2]], dtype=object))
        assert not result.matches

    def test_threshold_controls_dirty_collection_series(self):
        series = pd.Series([[1, 2], [3, 4], "invalid"], dtype=object)
        assert not is_list_like(series, threshold=0.95)
        assert is_list_like(series, threshold=2 / 3)

    def test_collection_options_can_disable_tuple_set_and_array(self):
        assert not is_list_like(pd.Series([(1, 2), (3, 4)]), allow_tuple=False)
        assert not is_list_like(pd.Series([{1, 2}, {3, 4}]), allow_set=False)
        arrays = pd.Series([np.array([1, 2]), np.array([3, 4])])
        assert not is_list_like(arrays, allow_array=False)

    def test_boolean_wrapper_matches_evidence(self):
        series = pd.Series(["a|b", "c|d"])
        assert is_list_like(series) == list_like_evidence(series).matches

    @pytest.mark.parametrize(("kwargs", "error", "message"), [
        ({"threshold": -0.1}, ValueError, "threshold"),
        ({"threshold": 1.1}, ValueError, "threshold"),
        ({"min_multivalue_ratio": 1.1}, ValueError, "min_multivalue_ratio"),
        ({"max_prose_ratio": -0.1}, ValueError, "max_prose_ratio"),
        ({"limit": 0}, ValueError, "greater than zero"),
        ({"limit": True}, TypeError, "integer or None"),
        ({"min_items": 1}, ValueError, "at least 2"),
        ({"min_items": 2.5}, TypeError, "integer"),
        ({"min_delimiter_occurrences": 0}, ValueError, "positive integer"),
        ({"delimiters": ("",)}, ValueError, "nonempty delimiter"),
    ])
    def test_evidence_configuration_validation(self, kwargs, error, message):
        with pytest.raises(error, match=message):
            list_like_evidence(pd.Series(["a,b", "c,d"]), **kwargs)

    def test_non_series_is_rejected(self):
        with pytest.raises(TypeError, match="pandas Series"):
            list_like_evidence([[1, 2]])


class TestAsList:
    def test_infers_delimiter_and_preserves_metadata(self):
        source = pd.Series(["apples|milk", "bread|cheese"], index=[2, 4], name="goods")
        result = as_list(source)
        assert result.tolist() == [["apples", "milk"], ["bread", "cheese"]]
        assert result.index.equals(source.index)
        assert result.name == "goods"

    def test_explicit_multicharacter_delimiter(self):
        result = as_list(["a::b", "c::d"], delimiter="::")
        assert result.tolist() == [["a", "b"], ["c", "d"]]

    def test_json_is_decoded(self):
        result = as_list(['["a", "b"]', '["c", "d"]'])
        assert result.tolist() == [["a", "b"], ["c", "d"]]

    def test_blank_strings_and_none_become_missing(self):
        result = as_list(pd.Series(["a,b", " ", None]), delimiter=",")
        assert result.iloc[0] == ["a", "b"]
        assert result.iloc[1] is pd.NA
        assert result.iloc[2] is pd.NA

    def test_single_value_with_explicit_delimiter_becomes_singleton(self):
        result = as_list(["apples", "bread"], delimiter=",")
        assert result.tolist() == [["apples"], ["bread"]]

    def test_inference_rejects_ordinary_strings(self):
        with pytest.raises(ValueError, match="not plausibly list-like"):
            as_list(["apples", "bread"])

    def test_infer_false_needs_collections_or_delimiter(self):
        with pytest.raises(TypeError, match="delimiter or infer=True"):
            as_list(["a,b", "c,d"], infer=False)

    def test_empty_item_is_rejected(self):
        with pytest.raises(ValueError, match="cannot be parsed"):
            as_list(["a,,b"], delimiter=",")


class TestAccessor:
    def test_accessor_rejects_object_series(self):
        with pytest.raises(AttributeError, match="requires ListDtype"):
            lp.ListAccessor(pd.Series([[1, 2]], dtype=object))

    def test_lengths_are_nullable_and_preserve_metadata(self, basket):
        result = basket.basket.lengths()
        assert str(result.dtype) == "Int64"
        assert result.tolist()[:2] == [2, 2]
        assert pd.isna(result.iloc[2])
        assert result.iloc[3] == 2
        assert result.index.equals(basket.index)
        assert result.name == basket.name

    def test_contains_is_nullable_and_preserves_metadata(self, basket):
        result = basket.basket.contains("milk")
        assert str(result.dtype) == "boolean"
        assert result.iloc[0] == True
        assert result.iloc[1] == False
        assert pd.isna(result.iloc[2])
        assert result.index.equals(basket.index)
        assert result.name == basket.name

    def test_explode_preserves_index_by_default(self, basket):
        result = basket.basket.explode()
        assert result.index.tolist() == [10, 10, 20, 20, 30, 40, 40]
        assert result.name == basket.name

    def test_explode_can_ignore_index(self, basket):
        result = basket.basket.explode(ignore_index=True)
        assert isinstance(result.index, pd.RangeIndex)
        assert len(result) == 7

    def test_indicators(self, basket):
        result = basket.basket.indicators()
        assert result.index.equals(basket.index)
        assert set(result.columns) == {"apples", "milk", "bread", "cheese", "biscuits", "jam"}
        assert result.loc[10, "apples"]
        assert not result.loc[10, "bread"]
        assert not result.loc[30].any()
        assert all(str(dtype) == "boolean" for dtype in result.dtypes)

    def test_sparse_indicators(self, basket):
        result = basket.basket.indicators(sparse=True)
        assert all(isinstance(dtype, pd.SparseDtype) for dtype in result.dtypes)
        assert result.loc[20, "bread"]

    def test_empty_indicators_retain_source_index(self):
        source = as_list([[], None])
        result = source.basket.indicators()
        assert result.empty
        assert result.index.equals(source.index)


def test_public_exports_are_complete():
    assert set(lp.__all__) == {
        "DEFAULT_DELIMITERS", "ListAccessor", "ListArray", "ListDtype",
        "ListLikeEvidence", "as_list", "is_list", "is_list_like",
        "list_like_evidence",
    }
