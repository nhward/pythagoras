import json
import sys
from pathlib import Path

path = Path(__file__).resolve().parent.parent / "app"
if path not in sys.path:
    sys.path.insert(0, path)

import geometry_pandas as gp
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from geometry_pandas import (
    GeometryEvidence,
    as_geometry,
    geometry_evidence,
    is_geometry,
    is_geometry_like,
)
from shapely.geometry import LineString, Point, mapping


@pytest.fixture
def points():
    return [Point(174.763336, -36.848461), Point(172.636225, -43.532054)]


@pytest.fixture
def point_series(points):
    return pd.Series(points, index=[10, 20], name="location", dtype=object)


@pytest.fixture
def geo_series(points):
    return gpd.GeoSeries(
        points,
        index=[10, 20],
        name="location",
        crs="EPSG:4326",
    )


@pytest.fixture
def wkt_series():
    return pd.Series(
        ["POINT (174.763336 -36.848461)", "POINT (172.636225 -43.532054)"],
        index=[10, 20],
        name="location",
    )


@pytest.fixture
def coordinate_series():
    return pd.Series(
        [(174.763336, -36.848461), (172.636225, -43.532054)],
        index=[10, 20],
        name="lon_lat",
    )


class TestPublicPredicates:
    def test_is_geometry_accepts_geoseries_array_and_dtype(self, geo_series):
        assert is_geometry(geo_series)
        assert is_geometry(geo_series.array)
        assert is_geometry(geo_series.dtype)

    @pytest.mark.parametrize(
        "value",
        [
            pd.Series([Point(1, 2)], dtype=object),
            pd.Series(["POINT (1 2)"]),
            np.dtype("object"),
            object(),
        ],
    )
    def test_is_geometry_rejects_non_geometry_dtype(self, value):
        assert not is_geometry(value)

    def test_boolean_wrapper_matches_evidence(self, point_series):
        assert is_geometry_like(point_series)
        assert not is_geometry_like(pd.Series(["ordinary", "text"]))


class TestParsers:
    def test_shapely_parser(self):
        point = Point(1, 2)
        assert gp._parse_shapely_geometry(point) is point
        assert gp._parse_shapely_geometry("POINT (1 2)") is None

    @pytest.mark.parametrize(
        ("value", "geometry_type"),
        [
            ("POINT (1 2)", "Point"),
            (" point z (1 2 3) ", "Point"),
            ("LINESTRING (0 0, 1 1)", "LineString"),
            ("POLYGON ((0 0, 1 0, 1 1, 0 0))", "Polygon"),
            ("POINT EMPTY", "Point"),
        ],
    )
    def test_wkt_parser(self, value, geometry_type):
        result = gp._parse_wkt_geometry(value)
        assert result is not None
        assert result.geom_type == geometry_type

    @pytest.mark.parametrize(
        "value",
        ["not geometry", "POINT broken", "", 123, None],
    )
    def test_wkt_parser_rejects_invalid_values(self, value):
        assert gp._parse_wkt_geometry(value) is None

    def test_wkb_parser_accepts_binary_hex_and_prefixed_hex(self):
        point = Point(1, 2)
        binary = point.wkb
        hexadecimal = point.wkb_hex
        for value in (binary, bytearray(binary), memoryview(binary), hexadecimal, "0x" + hexadecimal):
            assert gp._parse_wkb_geometry(value).equals(point)

    @pytest.mark.parametrize("value", [b"bad", "XYZ", "0xXYZ", 123, None])
    def test_wkb_parser_rejects_invalid_values(self, value):
        assert gp._parse_wkb_geometry(value) is None

    def test_geojson_parser_accepts_mapping_and_json_string(self):
        candidate = mapping(Point(1, 2))
        assert gp._parse_geojson_geometry(candidate).equals(Point(1, 2))
        assert gp._parse_geojson_geometry(json.dumps(candidate)).equals(Point(1, 2))

    @pytest.mark.parametrize(
        "value",
        [
            {"type": "Feature", "geometry": mapping(Point(1, 2)), "properties": {}},
            {"type": "Point"},
            "not json",
            "{broken}",
            None,
        ],
    )
    def test_geojson_parser_rejects_invalid_or_non_geometry_values(self, value):
        assert gp._parse_geojson_geometry(value) is None


class TestCoordinateRecognition:
    @pytest.mark.parametrize(
        ("value", "order"),
        [
            ((174.763336, -36.848461), "lon-lat"),
            ((-36.848461, 174.763336), "lat-lon"),
            ([174.763336, -36.848461], "either"),
            (np.array([174.763336, -36.848461]), "either"),
            ((174.763336, -36.848461, 15.5), "lon-lat"),
        ],
    )
    def test_valid_coordinate_pairs(self, value, order):
        assert gp._is_numeric_coordinate_pair(value, order=order)

    @pytest.mark.parametrize(
        "value",
        [
            (181, 91),
            (np.inf, 1),
            (np.nan, 1),
            (True, 1),
            (1,),
            (1, 2, 3, 4),
            [[1, 2], [3, 4]],
            {"x": 1, "y": 2},
            "174,-36",
        ],
    )
    def test_invalid_coordinate_pairs(self, value):
        assert not gp._is_numeric_coordinate_pair(value)

    def test_z_coordinate_can_be_disabled(self):
        assert not gp._is_numeric_coordinate_pair((1, 2, 3), allow_z=False)

    def test_longitude_360_is_opt_in(self):
        value = (250, -36)
        assert not gp._is_numeric_coordinate_pair(value, order="lon-lat")
        assert gp._is_numeric_coordinate_pair(
            value, order="lon-lat", allow_longitude_360=True
        )

    def test_decimal_precision_can_apply_to_either_coordinate(self):
        assert gp._is_numeric_coordinate_pair(
            (174.763336, -36.8),
            order="lon-lat",
            min_decimal_places=5,
        )

    def test_decimal_precision_can_be_required_for_both_coordinates(self):
        assert not gp._is_numeric_coordinate_pair(
            (174.763336, -36.8),
            order="lon-lat",
            min_decimal_places=5,
            require_both_precise=True,
        )

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(1, 0), (1.2, 1), ("1.23000", 2), ("1.23456", 5), (np.nan, None), (True, None)],
    )
    def test_apparent_decimal_places(self, value, expected):
        assert gp._apparent_decimal_places(value) == expected

    @pytest.mark.parametrize(
        ("kwargs", "error", "message"),
        [
            ({"order": "xy"}, ValueError, "order"),
            ({"min_decimal_places": -1}, ValueError, "non-negative"),
            ({"min_decimal_places": 1.5}, TypeError, "integer or None"),
            ({"min_decimal_places": True}, TypeError, "integer or None"),
        ],
    )
    def test_coordinate_configuration_validation(self, kwargs, error, message):
        with pytest.raises(error, match=message):
            gp._is_numeric_coordinate_pair((1, 2), **kwargs)


class TestEvidence:
    def test_geometry_dtype_is_certain(self, geo_series):
        result = geometry_evidence(geo_series)
        assert result == GeometryEvidence(
            True,
            "certain",
            "geometry-dtype",
            1.0,
            "Series has GeoPandas GeometryDtype",
            ("Point",),
        )

    @pytest.mark.parametrize(
        ("series", "representation", "geometry_types"),
        [
            (
                pd.Series([Point(1, 2), LineString([(0, 0), (1, 1)])], dtype=object),
                "shapely",
                ("LineString", "Point"),
            ),
            (pd.Series(["POINT (1 2)", "POINT (3 4)"]), "wkt", ("Point",)),
            (pd.Series([Point(1, 2).wkb, Point(3, 4).wkb]), "wkb", ("Point",)),
            (
                pd.Series([mapping(Point(1, 2)), mapping(Point(3, 4))]),
                "geojson",
                ("Point",),
            ),
        ],
    )
    def test_strong_representation_evidence(self, series, representation, geometry_types):
        result = geometry_evidence(series)
        assert result.matches
        assert result.confidence == "strong"
        assert result.representation == representation
        assert result.success_ratio == 1.0
        assert result.geometry_types == geometry_types

    def test_threshold_controls_dirty_series(self):
        series = pd.Series(["POINT (1 2)", "POINT (3 4)", "invalid"])
        assert not geometry_evidence(series, threshold=0.95).matches
        result = geometry_evidence(series, threshold=2 / 3)
        assert result.matches
        assert result.success_ratio == pytest.approx(2 / 3)

    @pytest.mark.parametrize(
        ("option", "representation", "series"),
        [
            ("allow_wkt", "wkt", pd.Series(["POINT (1 2)", "POINT (3 4)"])),
            ("allow_wkb", "wkb", pd.Series([Point(1, 2).wkb, Point(3, 4).wkb])),
            (
                "allow_geojson",
                "geojson",
                pd.Series([mapping(Point(1, 2)), mapping(Point(3, 4))]),
            ),
        ],
    )
    def test_representation_detection_can_be_disabled(self, option, representation, series):
        enabled = geometry_evidence(series)
        assert enabled.representation == representation
        disabled = geometry_evidence(series, **{option: False})
        assert not disabled.matches

    def test_coordinate_pairs_are_disabled_by_default(self, coordinate_series):
        assert not geometry_evidence(coordinate_series).matches

    def test_coordinate_pairs_with_name_hint_are_plausible(self, coordinate_series):
        result = geometry_evidence(
            coordinate_series,
            allow_coordinate_pairs=True,
            coordinate_order="lon-lat",
            min_decimal_places=5,
        )
        assert result.matches
        assert result.confidence == "plausible"
        assert result.representation == "coordinate-pair"
        assert result.geometry_types == ("Point",)

    def test_coordinate_name_hint_can_be_required(self, coordinate_series):
        unnamed = coordinate_series.rename("measurements")
        result = geometry_evidence(unnamed, allow_coordinate_pairs=True)
        assert not result.matches
        assert result.confidence == "unknown"
        assert result.success_ratio == 1.0

    def test_coordinate_name_hint_can_be_disabled(self, coordinate_series):
        unnamed = coordinate_series.rename("measurements")
        result = geometry_evidence(
            unnamed,
            allow_coordinate_pairs=True,
            require_coordinate_name_hint=False,
        )
        assert result.matches

    def test_blank_and_missing_values_are_ignored(self):
        series = pd.Series(["POINT (1 2)", None, pd.NA, "  "])
        result = geometry_evidence(series)
        assert result.matches
        assert result.success_ratio == 1.0

    @pytest.mark.parametrize("series", [pd.Series(dtype=object), pd.Series([None, pd.NA])])
    def test_empty_or_all_missing_is_unknown(self, series):
        result = geometry_evidence(series)
        assert not result.matches
        assert result.confidence == "unknown"
        assert result.success_ratio == 0.0

    def test_limit_samples_across_series(self):
        series = pd.Series(["POINT (1 2)", "invalid", "POINT (3 4)"])
        result = geometry_evidence(series, limit=2)
        assert result.matches
        assert result.success_ratio == 1.0

    @pytest.mark.parametrize(
        ("kwargs", "error", "message"),
        [
            ({"threshold": -0.1}, ValueError, "between 0 and 1"),
            ({"threshold": 1.1}, ValueError, "between 0 and 1"),
            ({"limit": 0}, ValueError, "greater than zero"),
            ({"limit": True}, TypeError, "integer or None"),
            ({"limit": 1.5}, TypeError, "integer or None"),
        ],
    )
    def test_configuration_validation(self, kwargs, error, message):
        with pytest.raises(error, match=message):
            geometry_evidence(pd.Series(["POINT (1 2)"]), **kwargs)

    def test_requires_series(self):
        with pytest.raises(TypeError, match="pandas Series"):
            geometry_evidence(["POINT (1 2)"])


class TestAsGeometry:
    def test_shapely_conversion_preserves_index_name_and_crs(self, point_series):
        result = as_geometry(point_series, crs="EPSG:4326")
        assert isinstance(result, gpd.GeoSeries)
        assert is_geometry(result)
        assert result.index.equals(point_series.index)
        assert result.name == point_series.name
        assert result.crs.to_epsg() == 4326
        assert result.geom_type.tolist() == ["Point", "Point"]

    def test_wkt_is_inferred(self, wkt_series):
        result = as_geometry(wkt_series, crs="EPSG:4326")
        assert result.x.tolist() == pytest.approx([174.763336, 172.636225])
        assert result.y.tolist() == pytest.approx([-36.848461, -43.532054])

    def test_explicit_wkt_conversion_can_skip_inference(self, wkt_series):
        result = as_geometry(
            wkt_series,
            representation="wkt",
            infer=False,
        )
        assert is_geometry(result)

    def test_wkb_conversion(self, points):
        result = as_geometry(
            pd.Series([point.wkb for point in points]),
            representation="wkb",
        )
        assert result.geom_type.tolist() == ["Point", "Point"]

    def test_geojson_conversion(self, points):
        result = as_geometry(
            pd.Series([mapping(point) for point in points]),
            representation="geojson",
        )
        assert result.geom_type.tolist() == ["Point", "Point"]

    def test_coordinate_pair_conversion(self, coordinate_series):
        result = as_geometry(
            coordinate_series,
            crs="EPSG:4326",
            representation="coordinate-pair",
            coordinate_order="lon-lat",
        )
        assert result.x.tolist() == pytest.approx([174.763336, 172.636225])
        assert result.y.tolist() == pytest.approx([-36.848461, -43.532054])

    def test_lat_lon_coordinate_order_is_reversed_for_shapely(self):
        source = pd.Series([(-36.848461, 174.763336)], name="lat_lon")
        result = as_geometry(
            source,
            representation="coordinate-pair",
            coordinate_order="lat-lon",
        )
        assert result.iloc[0].x == pytest.approx(174.763336)
        assert result.iloc[0].y == pytest.approx(-36.848461)

    def test_three_dimensional_point(self):
        result = as_geometry(
            pd.Series([(174.763336, -36.848461, 12.5)]),
            representation="coordinate-pair",
            coordinate_order="lon-lat",
        )
        assert result.iloc[0].has_z
        assert result.iloc[0].z == pytest.approx(12.5)

    def test_missing_and_blank_values_become_missing_geometry(self):
        result = as_geometry(
            pd.Series(["POINT (1 2)", None, pd.NA, "  "]),
            representation="wkt",
        )
        assert result.isna().tolist() == [False, True, True, True]

    def test_existing_geoseries_is_returned_with_metadata(self, geo_series):
        result = as_geometry(geo_series)
        assert isinstance(result, gpd.GeoSeries)
        assert result.index.equals(geo_series.index)
        assert result.name == geo_series.name
        assert result.crs == geo_series.crs

    def test_existing_geoseries_accepts_equivalent_crs(self, geo_series):
        result = as_geometry(geo_series, crs="EPSG:4326")
        assert result.crs.to_epsg() == 4326

    def test_existing_geoseries_rejects_conflicting_crs(self, geo_series):
        with pytest.raises(ValueError, match="different CRS"):
            as_geometry(geo_series, crs="EPSG:3857")

    def test_existing_unset_crs_can_be_set(self, points):
        source = gpd.GeoSeries(points)
        result = as_geometry(source, crs="EPSG:4326")
        assert result.crs.to_epsg() == 4326

    def test_invalid_representation_is_rejected(self, wkt_series):
        with pytest.raises(ValueError, match="representation must be"):
            as_geometry(wkt_series, representation="svg")

    def test_representation_required_when_inference_disabled(self, wkt_series):
        with pytest.raises(TypeError, match="representation is required"):
            as_geometry(wkt_series, infer=False)

    def test_failed_inference_reports_reason(self):
        with pytest.raises(ValueError, match="not plausibly geometry-like"):
            as_geometry(pd.Series(["ordinary", "text"]))

    def test_invalid_value_for_explicit_representation_is_rejected(self):
        with pytest.raises(ValueError, match="not valid WKT"):
            as_geometry(
                pd.Series(["POINT (1 2)", "invalid"]),
                representation="wkt",
            )

    def test_invalid_coordinate_is_rejected(self):
        with pytest.raises(ValueError, match="not a plausible coordinate pair"):
            as_geometry(
                pd.Series([(181, 91)]),
                representation="coordinate-pair",
                coordinate_order="lon-lat",
            )
