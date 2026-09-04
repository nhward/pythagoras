import sys
from pathlib import Path

path = Path(__file__).resolve().parent.parent / "app"
if path not in sys.path:
    sys.path.insert(0, path)

import itertools

import pandas as pd
import proxy_data as pxd
import pytest
from cleaning_record import CleaningRecord
from roles import Role
from sklearn.preprocessing import StandardScaler


@pytest.mark.unit
def test_frame_is_the_preferred_dataframe_accessor():
    native = pd.DataFrame({"value": [1, 2]})
    data = pxd.proxy_data.from_native(native)

    assert data.frame is data.to_native()
    assert data.frame is data.data


@pytest.mark.unit
def test_with_cleaned_data_returns_successor_and_records_operation():
    source = pxd.proxy_data(
        pd.DataFrame({"keep": [1, 2, 3], "drop": [4, 5, 6]}),
        _name="example",
    )

    result = source.with_cleaned_data(
        source.frame.loc[:, ["keep"]],
        card="example_card",
        operation="Drop a variable",
        parameters={"variable": "drop"},
    )

    assert source.frame.columns.tolist() == ["keep", "drop"]
    assert result.frame.columns.tolist() == ["keep"]
    assert result.name == "example"
    assert result.role_map.columns_with_role(Role.PREDICTOR) == {"keep"}
    assert source.cleaning_records == ()
    assert result.cleaning_records == (
        CleaningRecord(
            card="example_card",
            operation="Drop a variable",
            parameters={"variable": "drop"},
            input_shape=(3, 2),
            output_shape=(3, 1),
        ),
    )


@pytest.mark.unit
def test_cleaning_records_are_preserved_and_appended_in_order():
    source = pxd.proxy_data(pd.DataFrame({"value": [1, 1, 2]}))
    first = source.with_cleaned_data(
        source.frame.drop_duplicates(),
        card="duplicates",
        operation="Remove duplicates",
    )
    second = first.with_cleaned_data(
        first.frame.rename(columns={"value": "measurement"}),
        card="modify",
        operation="Rename variable",
    )

    assert [record.card for record in second.cleaning_records] == [
        "duplicates",
        "modify",
    ]
    assert second.clone().cleaning_records == second.cleaning_records
    assert second.sample(mode="all").cleaning_records == second.cleaning_records


@pytest.mark.unit
def test_proxy_equality_includes_cleaning_records_by_default():
    frame = pd.DataFrame({"value": [1, 2]})
    plain = pxd.proxy_data(frame)
    recorded = plain.with_cleaned_data(
        frame,
        card="modify",
        operation="Confirm values",
    )

    assert plain != recorded
    assert plain.equals(recorded, check_cleaning_records=False)


@pytest.mark.unit
def test_pipeline_keeps_clean_source_and_unfitted_training_blueprint():
    source = pxd.proxy_data(pd.DataFrame({"value": [1.0, 2.0, 3.0, 100.0]}))
    preview_scaler = StandardScaler().set_output(transform="pandas")
    preview = preview_scaler.fit_transform(source.frame)

    result = source.with_pipeline_step(
        preview_scaler,
        name="scale",
        preview_frame=preview,
    )

    pd.testing.assert_frame_equal(result.clean_frame, source.frame)
    pd.testing.assert_frame_equal(result.frame, preview)
    assert result.pipeline_steps == ("scale",)
    assert not hasattr(result.pipeline.named_steps["scale"], "mean_")

    train = result.clean_frame.iloc[:3]
    test = result.clean_frame.iloc[3:]
    fitted = result.pipeline_for_training().fit(train)
    assert fitted.transform(train)["value"].mean() == pytest.approx(0.0)
    assert fitted.transform(test)["value"].iloc[0] > 100


@pytest.mark.unit
def test_pipeline_steps_chain_with_unique_names_and_survive_clone():
    source = pxd.proxy_data(pd.DataFrame({"value": [1.0, 2.0, 3.0]}))
    scaler = StandardScaler().set_output(transform="pandas")
    first_preview = scaler.fit_transform(source.frame)
    first = source.with_pipeline_step(
        scaler, name="scale", preview_frame=first_preview,
    )
    second_scaler = StandardScaler(with_std=False).set_output(transform="pandas")
    second_preview = second_scaler.fit_transform(first.frame)
    second = first.with_pipeline_step(
        second_scaler, name="scale", preview_frame=second_preview,
    )

    assert second.pipeline_steps == ("scale", "scale_2")
    assert second.clone().pipeline_steps == second.pipeline_steps
    pd.testing.assert_frame_equal(second.clone().clean_frame, source.frame)


@pytest.mark.unit
def test_cleaning_and_direct_replacement_are_rejected_after_pipeline_starts():
    source = pxd.proxy_data(pd.DataFrame({"value": [1.0, 2.0, 3.0]}))
    scaler = StandardScaler().set_output(transform="pandas")
    transformed = source.with_pipeline_step(
        scaler,
        name="scale",
        preview_frame=scaler.fit_transform(source.frame),
    )

    with pytest.raises(RuntimeError, match="Cleaning operations cannot run"):
        transformed.with_cleaned_data(
            transformed.frame,
            card="late_cleaner",
            operation="Invalid late cleaning",
        )
    with pytest.raises(RuntimeError, match="Cannot replace data directly"):
        transformed.data = transformed.frame.copy()


def _is_strictly_increasing(seq):
    """Adjacent-pairs order check (backend-agnostic)."""
    return all(a < b for a, b in itertools.pairwise(seq))

@pytest.mark.unit
def test_sample_random_preserves_order_pandas():
    import pandas as pd
    native = pd.DataFrame({"i": range(10)})
    d = pxd.proxy_data.from_native(native)
    out = d.sample(n=5, mode="random").to_native()
    # Order check via column values (works even if index is re-based)
    assert _is_strictly_increasing(out["i"].tolist())


@pytest.mark.unit
def test_sample_random_preserves_order_polars():
    import polars as pl
    native = pl.DataFrame({"i": list(range(10))})
    d = pxd.proxy_data.from_native(native)
    out = d.sample(n=5, mode="random").to_native()
    vals = out["i"].to_list()
    assert _is_strictly_increasing(vals)


@pytest.mark.unit
def test_sample_headtail_order_and_length_pandas():
    import pandas as pd
    native = pd.DataFrame({"i": range(10)})
    d = pxd.proxy_data.from_native(native)
    out = d.sample(n=6, mode="headtail").to_native()
    vals = out["i"].tolist()
    assert _is_strictly_increasing(vals)
    assert len(out) == 6


@pytest.mark.unit
def test_sample_headtail_order_and_length_polars():
    import polars as pl
    native = pl.DataFrame({"i": list(range(10))})
    d = pxd.proxy_data.from_native(native)
    out = d.sample(n=6, mode="headtail").to_native()
    vals = out["i"].to_list()
    assert _is_strictly_increasing(vals)
    assert len(out) == 6


@pytest.mark.unit
def test_sample_n_greater_than_len_returns_all_pandas():
    import pandas as pd
    native = pd.DataFrame({"i": range(5)})
    d = pxd.proxy_data.from_native(native)
    out = d.sample(n=999, mode="random").to_native()
    assert len(out) == 5
    assert _is_strictly_increasing(out["i"].tolist())


@pytest.mark.unit
def test_sample_n_greater_than_len_returns_all_polars():
    import polars as pl
    native = pl.DataFrame({"i": list(range(5))})
    d = pxd.proxy_data.from_native(native)
    out = d.sample(n=999, mode="random").to_native()
    vals = out["i"].to_list()
    assert len(vals) == 5
    assert _is_strictly_increasing(vals)


@pytest.mark.unit
def test_geopandas_geometry_preserved_on_sample():
    import geopandas as gpd
    import shapely.geometry as sgeom
    native = gpd.GeoDataFrame({
            "id": [0, 1, 2, 3], 
            "location1": [sgeom.Point(x, x) for x in range(4)], 
            "location2": [sgeom.Point(x, x) for x in range(4)]
        },
        geometry="location1",
        crs="EPSG:4326",
    )
    d = pxd.proxy_data.from_native(native)
    samp = d.sample(n=3, mode="random", keep_geometry = True).to_native()
    assert isinstance(samp, gpd.GeoDataFrame)
    assert samp.geometry.name == "location1"
    assert samp.crs.to_string() == "EPSG:4326"
    assert _is_strictly_increasing(samp["id"].tolist())

@pytest.mark.unit
def test_geopandas_geometry_preserved_on_full():
    import geopandas as gpd
    import shapely.geometry as sgeom
    native = gpd.GeoDataFrame(
        {"i": [0, 1, 2, 3], "geom": [sgeom.Point(x, x) for x in range(4)]},
        geometry="geom",
        crs="EPSG:4326",
    )
    d = pxd.proxy_data.from_native(native)
    full = d.sample(n=500, mode="random", keep_geometry = True).to_native()
    assert isinstance(full, gpd.GeoDataFrame)
    assert full.geometry.name == "geom"
    assert full.crs.to_string() == "EPSG:4326"
    assert _is_strictly_increasing(full["i"].tolist())


def make_sample_proxy() -> pxd.proxy_data:
    """
    Helper to build a proxy_data with different dtypes and roles.

    Columns:
      y    : int,  Role.TARGET
      x1   : float, Role.PREDICTOR
      x2   : float, Role.PREDICTOR
      id   : int, Role.IDENTIFIER
      txt  : object/str, Role.SENSITIVE
    """
    df = pd.DataFrame(
        {
            "y": [1, 2, 3],
            "x1": [0.1, 0.2, 0.3],
            "x2": [10.0, 20.0, 30.0],
            "id": [101, 102, 103],
            "txt": ["a", "b", "c"],
        }
    )
    roles = {
        "target": ["y"],
        "predictor": ["x1", "x2"],
        "identifier": ["id"],
        "sensitive" : ["txt"]
    }
    return pxd.proxy_data(df, _roles=roles)


# -------------------------------------------------------------------
# select_dtypes tests
# -------------------------------------------------------------------

@pytest.mark.unit
def test_select_dtypes_include_only_float():
    p = make_sample_proxy()
    p_sel = p.select_dtypes(include=["float"])
    cols = list(p_sel._df.columns)
    assert cols == ["x1", "x2"]


@pytest.mark.unit
def test_select_dtypes_include_int_and_object():
    p = make_sample_proxy()
    p_sel = p.select_dtypes(include=["int", "object"])
    cols = list(p_sel._df.columns)
    # ints: y, id; object: txt
    assert cols == ["y", "id", "txt"]


@pytest.mark.unit
def test_select_dtypes_exclude_object():
    p = make_sample_proxy()
    p_sel = p.select_dtypes(exclude=["object"])
    cols = list(p_sel._df.columns)
    # txt should be dropped
    assert "txt" not in cols
    assert set(cols) == {"y", "x1", "x2", "id"}


@pytest.mark.unit
def test_select_dtypes_include_and_exclude():
    p = make_sample_proxy()
    # keep only numeric, but exclude floats
    p_sel = p.select_dtypes(include=["number"], exclude=["float"])
    cols = list(p_sel._df.columns)
    # Should leave only integer columns y and id
    assert set(cols) == {"y", "id"}


# -------------------------------------------------------------------
# select_drole tests
# -------------------------------------------------------------------
@pytest.mark.unit
def test_select_drole_single_role():
    p = make_sample_proxy()
    p_sel = p.select_drole(include=Role.PREDICTOR)
    cols = list(p_sel._df.columns)
    # Only predictor columns
    assert set(cols) == {"x1", "x2"}


@pytest.mark.unit
def test_select_drole_multiple_roles():
    p = make_sample_proxy()
    # target + predictors
    p_sel = p.select_drole(include=[Role.TARGET, Role.PREDICTOR])
    cols = list(p_sel._df.columns)
    assert set(cols) == {"y", "x1", "x2"}


@pytest.mark.unit
def test_select_drole_exclude_role():
    p = make_sample_proxy()
    # exclude identifiers
    p_sel = p.select_drole(exclude=Role.IDENTIFIER)
    cols = set(p_sel._df.columns)
    assert "id" not in cols
    # everything else remains
    assert cols == {"y", "x1", "x2", "txt"}


@pytest.mark.unit
def test_select_drole_include_and_exclude():
    p = make_sample_proxy()
    # keep only predictors, but drop any that are sensitive (none in this set)
    p_sel = p.select_drole(include=Role.PREDICTOR,
                           exclude=Role.SENSITIVE)
    cols = set(p_sel._df.columns)
    assert cols == {"x1", "x2"}


@pytest.mark.unit
def test_select_drole_columns_with_no_roles():
    # Add one column with no role metadata
    p = make_sample_proxy()
    p._df["extra"] = [0, 0, 0]
    # roles mapping doesn't mention "extra" → treated as having empty role set
    # 1) No include/exclude → extra should be kept
    p_all = p.select_drole(include=None, exclude=None)
    assert "extra" in p_all._df.columns
    # 2) Include only TARGET → extra should be dropped
    p_target = p.select_drole(include=Role.TARGET)
    assert "extra" not in p_target._df.columns


@pytest.mark.unit
def test_select_drole_overlapping_include_exclude_raises():
    p = make_sample_proxy()
    with pytest.raises(ValueError):
        p.select_drole(include=[Role.TARGET, Role.PREDICTOR],
                       exclude=[Role.PREDICTOR])
