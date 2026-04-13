# import sys
# import pathlib
# # --- Ensure 'src' is importable if you use a src/ layout ---
# ROOT = pathlib.Path(__file__).resolve().parents[1]
# SRC = ROOT / "src"
# if SRC.exists() and str(SRC) not in sys.path:
#     sys.path.insert(0, str(SRC))

import pytest  # noqa: F401
import pandas as pd
import proxyData as pxd
from roles import Role

def _is_strictly_increasing(seq):
    """Adjacent-pairs order check (backend-agnostic)."""
    return all(a < b for a, b in zip(seq, seq[1:]))


def test_sample_random_preserves_order_pandas():
    import pandas as pd
    native = pd.DataFrame({"i": range(10)})
    d = pxd.ProxyData.from_native(native)
    out = d.sample(n=5, mode="random").to_native()
    # Order check via column values (works even if index is re-based)
    assert _is_strictly_increasing(out["i"].tolist())


def test_sample_random_preserves_order_polars():
    import polars as pl
    native = pl.DataFrame({"i": list(range(10))})
    d = pxd.ProxyData.from_native(native)
    out = d.sample(n=5, mode="random").to_native()
    vals = out["i"].to_list()
    assert _is_strictly_increasing(vals)


#@pytest.mark.skipif(pd is None, reason="pandas not installed")
def test_sample_headtail_order_and_length_pandas():
    import pandas as pd
    native = pd.DataFrame({"i": range(10)})
    d = pxd.ProxyData.from_native(native)
    out = d.sample(n=6, mode="headtail").to_native()
    vals = out["i"].tolist()
    assert _is_strictly_increasing(vals)
    assert len(out) == 6


#@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_sample_headtail_order_and_length_polars():
    import polars as pl
    native = pl.DataFrame({"i": list(range(10))})
    d = pxd.ProxyData.from_native(native)
    out = d.sample(n=6, mode="headtail").to_native()
    vals = out["i"].to_list()
    assert _is_strictly_increasing(vals)
    assert len(out) == 6


def test_sample_n_greater_than_len_returns_all_pandas():
    import pandas as pd
    native = pd.DataFrame({"i": range(5)})
    d = pxd.ProxyData.from_native(native)
    out = d.sample(n=999, mode="random").to_native()
    assert len(out) == 5
    assert _is_strictly_increasing(out["i"].tolist())


def test_sample_n_greater_than_len_returns_all_polars():
    import polars as pl
    native = pl.DataFrame({"i": list(range(5))})
    d = pxd.ProxyData.from_native(native)
    out = d.sample(n=999, mode="random").to_native()
    vals = out["i"].to_list()
    assert len(vals) == 5
    assert _is_strictly_increasing(vals)


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
    d = pxd.ProxyData.from_native(native)
    samp = d.sample(n=3, mode="random", keep_geometry = True).to_native()
    assert isinstance(samp, gpd.GeoDataFrame)
    assert samp.geometry.name == "location1"
    assert samp.crs.to_string() == "EPSG:4326"
    assert _is_strictly_increasing(samp["id"].tolist())

def test_geopandas_geometry_preserved_on_full():
    import geopandas as gpd
    import shapely.geometry as sgeom
    native = gpd.GeoDataFrame(
        {"i": [0, 1, 2, 3], "geom": [sgeom.Point(x, x) for x in range(4)]},
        geometry="geom",
        crs="EPSG:4326",
    )
    d = pxd.ProxyData.from_native(native)
    full = d.sample(n=500, mode="random", keep_geometry = True).to_native()
    assert isinstance(full, gpd.GeoDataFrame)
    assert full.geometry.name == "geom"
    assert full.crs.to_string() == "EPSG:4326"
    assert _is_strictly_increasing(full["i"].tolist())


def make_sample_proxy() -> pxd.ProxyData:
    """
    Helper to build a ProxyData with different dtypes and roles.

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
        "y": {Role.TARGET},
        "x1": {Role.PREDICTOR},
        "x2": {Role.PREDICTOR},
        "id": {Role.IDENTIFIER},
        "txt": {Role.SENSITIVE},
    }
    return pxd.ProxyData(df, _roles=roles)


# -------------------------------------------------------------------
# select_dtypes tests
# -------------------------------------------------------------------

def test_select_dtypes_include_only_float():
    p = make_sample_proxy()
    p_sel = p.select_dtypes(include=["float"])
    cols = list(p_sel._df.columns)
    assert cols == ["x1", "x2"]


def test_select_dtypes_include_int_and_object():
    p = make_sample_proxy()
    p_sel = p.select_dtypes(include=["int", "object"])
    cols = list(p_sel._df.columns)
    # ints: y, id; object: txt
    assert cols == ["y", "id", "txt"]


def test_select_dtypes_exclude_object():
    p = make_sample_proxy()
    p_sel = p.select_dtypes(exclude=["object"])
    cols = list(p_sel._df.columns)
    # txt should be dropped
    assert "txt" not in cols
    assert set(cols) == {"y", "x1", "x2", "id"}


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

def test_select_drole_single_role():
    p = make_sample_proxy()
    p_sel = p.select_drole(include=Role.PREDICTOR)
    cols = list(p_sel._df.columns)
    # Only predictor columns
    assert set(cols) == {"x1", "x2"}


def test_select_drole_multiple_roles():
    p = make_sample_proxy()
    # target + predictors
    p_sel = p.select_drole(include=[Role.TARGET, Role.PREDICTOR])
    cols = list(p_sel._df.columns)
    assert set(cols) == {"y", "x1", "x2"}


def test_select_drole_exclude_role():
    p = make_sample_proxy()
    # exclude identifiers
    p_sel = p.select_drole(exclude=Role.IDENTIFIER)
    cols = set(p_sel._df.columns)
    assert "id" not in cols
    # everything else remains
    assert cols == {"y", "x1", "x2", "txt"}


def test_select_drole_include_and_exclude():
    p = make_sample_proxy()
    # keep only predictors, but drop any that are sensitive (none in this set)
    p_sel = p.select_drole(include=Role.PREDICTOR,
                           exclude=Role.SENSITIVE)
    cols = set(p_sel._df.columns)
    assert cols == {"x1", "x2"}


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


def test_select_drole_overlapping_include_exclude_raises():
    p = make_sample_proxy()
    with pytest.raises(ValueError):
        p.select_drole(include=[Role.TARGET, Role.PREDICTOR],
                       exclude=[Role.PREDICTOR])
