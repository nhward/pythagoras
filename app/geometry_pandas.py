"""Pandas helpers for detecting and constructing geometry-valued Series.
GeoPandas supplies the extension dtype; this module supplies inference and
conversion while keeping GeoPandas and Shapely optional until required.

Public entry points
-------------------
``GeometryEvidence``
    Class describing evidence that a Series is plausibly geometry-valued.

``as_geometry(data)``
    Converts Shapely, WKT, WKB, GeoJSON, or coordinate-pair values to a
    GeoPandas ``GeoSeries``.

``is_geometry(obj)``
    Tests a Series, array, or dtype for GeoPandas ``GeometryDtype``.

``geometry_evidence(series)``
    Returns evidence that a Series could plausibly contain geometries.

``is_geometry_like(series)``
    Returns whether a Series could plausibly contain geometries.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GeometryEvidence:
    """Evidence returned by :func:`geometry_evidence`."""
    matches: bool
    confidence: str
    representation: str | None
    success_ratio: float
    reason: str
    geometry_types: tuple[str, ...] = ()


_WKT_PREFIX = re.compile(
    r"""
    ^\s*(
        POINT | LINESTRING | POLYGON | MULTIPOINT | MULTILINESTRING |
        MULTIPOLYGON | GEOMETRYCOLLECTION | CIRCULARSTRING |
        COMPOUNDCURVE | CURVEPOLYGON | MULTICURVE | MULTISURFACE |
        TRIANGLE | TIN | POLYHEDRALSURFACE
    )(?:\s+Z|\s+M|\s+ZM)?\s*(?:EMPTY|\()
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)
_HEX_WKB = re.compile(r"^(?:[0-9A-Fa-f]{2})+$")
_GEOJSON_GEOMETRY_TYPES = {
    "Point", "MultiPoint", "LineString", "MultiLineString", "Polygon",
    "MultiPolygon", "GeometryCollection",
}
_COORDINATE_NAME_HINT = re.compile(
    r"""
    (coordinate|coordinates|coord|location|position|point|geometry|geom|
     lon[_\s-]*lat|lng[_\s-]*lat|lat[_\s-]*lon|xy|xyz)
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)


def _require_geopandas():
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError(
            "GeoPandas is required to construct a geometry Series"
        ) from exc
    return gpd


def _require_shapely():
    try:
        import shapely
    except ImportError as exc:
        raise ImportError(
            "Shapely is required to parse or construct geometries"
        ) from exc
    return shapely


def is_geometry(obj: object) -> bool:
    """Return whether ``obj`` has GeoPandas ``GeometryDtype``."""
    dtype = getattr(obj, "dtype", obj)
    try:
        from geopandas.array import GeometryDtype
    except ImportError:
        return type(dtype).__name__ == "GeometryDtype" and str(dtype) == "geometry"
    return isinstance(dtype, GeometryDtype)


def _is_missing_scalar(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, (Mapping, list, tuple, np.ndarray)):
        return False
    try:
        result = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return isinstance(result, (bool, np.bool_)) and bool(result)


def _sample_geometry_values(
    series: pd.Series,
    *,
    limit: int | None,
    blank_as_missing: bool,
) -> pd.Series:
    values = series[series.map(lambda value: not _is_missing_scalar(value))]
    if blank_as_missing:
        values = values[
            values.map(lambda value: not (isinstance(value, str) and not value.strip()))
        ]
    if limit is not None and len(values) > limit:
        positions = np.linspace(0, len(values) - 1, num=limit, dtype=int)
        values = values.iloc[positions]
    return values


def _parse_shapely_geometry(value: object):
    try:
        from shapely.geometry.base import BaseGeometry
    except ImportError:
        return None
    return value if isinstance(value, BaseGeometry) else None


def _parse_wkt_geometry(value: object):
    if not isinstance(value, str) or not _WKT_PREFIX.match(value.strip()):
        return None
    try:
        from shapely import wkt
        return wkt.loads(value.strip())
    except Exception:  # noqa: BLE001
        return None


def _parse_wkb_geometry(value: object):
    try:
        from shapely import wkb
    except ImportError:
        return None
    try:
        if isinstance(value, (bytes, bytearray, memoryview)):
            return wkb.loads(bytes(value))
        if isinstance(value, str):
            text = value.strip()
            if text.lower().startswith("0x"):
                text = text[2:]
            if _HEX_WKB.fullmatch(text):
                return wkb.loads(text, hex=True)
    except Exception:  # noqa: BLE001
        return None
    return None


def _parse_geojson_geometry(value: object):
    candidate = value
    if isinstance(value, str):
        text = value.strip()
        if not text.startswith("{"):
            return None
        try:
            candidate = json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    if not isinstance(candidate, Mapping):
        return None
    if candidate.get("type") not in _GEOJSON_GEOMETRY_TYPES:
        return None
    try:
        from shapely.geometry import shape
        return shape(candidate)
    except Exception:  # noqa: BLE001
        return None


def _evaluate_geometry_parser(
    values: pd.Series,
    *,
    representation: str,
    confidence: str,
    parser: Callable[[object], object | None],
    threshold: float,
) -> GeometryEvidence:
    parsed = [parser(value) for value in values]
    valid = [geometry is not None for geometry in parsed]
    success_ratio = float(np.mean(valid))
    geometry_types = tuple(sorted({
        geometry.geom_type for geometry in parsed if geometry is not None
    }))
    return GeometryEvidence(
        success_ratio >= threshold,
        confidence if success_ratio >= threshold else "weak",
        representation if success_ratio >= threshold else None,
        success_ratio,
        (
            f"{success_ratio:.0%} of inspected values are valid "
            f"{representation.upper()} geometries"
        ),
        geometry_types,
    )


def _apparent_decimal_places(value: object) -> int | None:
    """Return apparent significant decimal places for a numeric scalar."""
    if isinstance(value, (bool, np.bool_)):
        return None
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    if not decimal.is_finite():
        return None
    return max(0, -decimal.normalize().as_tuple().exponent)


def _coordinate_elements(value: object) -> list[object] | None:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        return None
    if isinstance(value, np.ndarray):
        return value.tolist() if value.ndim == 1 else None
    if isinstance(value, (list, tuple)):
        return list(value)
    return None


def _is_numeric_coordinate_pair(
    value: object,
    *,
    order: str = "either",
    allow_longitude_360: bool = False,
    min_decimal_places: int | None = None,
    require_both_precise: bool = False,
    allow_z: bool = True,
) -> bool:
    """Return whether a value plausibly represents a geographic coordinate."""
    if order not in {"lon-lat", "lat-lon", "either"}:
        raise ValueError(
            "order must be 'lon-lat', 'lat-lon', or 'either'"
        )
    if min_decimal_places is not None:
        if (
            isinstance(min_decimal_places, bool)
            or not isinstance(min_decimal_places, int)
        ):
            raise TypeError(
                "min_decimal_places must be an integer or None"
            )
        if min_decimal_places < 0:
            raise ValueError(
                "min_decimal_places must be non-negative"
            )
    elements = _coordinate_elements(value)
    allowed_lengths = {2, 3} if allow_z else {2}
    if elements is None or len(elements) not in allowed_lengths:
        return False
    if any(
        isinstance(element, (bool, np.bool_))
        for element in elements
    ):
        return False
    try:
        numbers = np.asarray(
            elements,
            dtype=np.float64,
        )
    except (TypeError, ValueError, OverflowError):
        return False
    # Reject nested sequences and multidimensional arrays.
    if numbers.ndim != 1:
        return False
    if numbers.size not in allowed_lengths:
        return False
    if not np.isfinite(numbers).all():
        return False
    first = float(numbers[0])
    second = float(numbers[1])
    def latitude(number: float) -> bool:
        return -90.0 <= number <= 90.0
    def longitude(number: float) -> bool:
        return (
            -180.0 <= number <= 180.0
            or (
                allow_longitude_360
                and 0.0 <= number <= 360.0
            )
        )
    if order == "lon-lat":
        valid_range = longitude(first) and latitude(second)
    elif order == "lat-lon":
        valid_range = latitude(first) and longitude(second)
    else:
        valid_range = (
            longitude(first) and latitude(second)
        ) or (
            latitude(first) and longitude(second)
        )
    if not valid_range:
        return False
    if min_decimal_places is None:
        return True
    precisions = (
        _apparent_decimal_places(elements[0]),
        _apparent_decimal_places(elements[1]),
    )
    precise = [
        places is not None
        and places >= min_decimal_places
        for places in precisions
    ]
    return (
        all(precise)
        if require_both_precise
        else any(precise)
    )


def _coordinate_pair_evidence(
    series: pd.Series,
    values: pd.Series,
    *,
    threshold: float,
    require_name_hint: bool,
    coordinate_order: str,
    allow_longitude_360: bool,
    min_decimal_places: int | None,
    require_both_precise: bool,
    allow_z: bool,
) -> GeometryEvidence:
    valid = values.map(lambda value: _is_numeric_coordinate_pair(
        value,
        order=coordinate_order,
        allow_longitude_360=allow_longitude_360,
        min_decimal_places=min_decimal_places,
        require_both_precise=require_both_precise,
        allow_z=allow_z,
    ))
    success_ratio = float(valid.mean())
    name = "" if series.name is None else str(series.name)
    name_matches = bool(_COORDINATE_NAME_HINT.search(name))
    matches = success_ratio >= threshold and (name_matches or not require_name_hint)
    if success_ratio < threshold:
        reason = f"Only {success_ratio:.0%} are numeric coordinate pairs"
    elif require_name_hint and not name_matches:
        reason = "Coordinate-like values lack a spatial column-name hint"
    else:
        reason = f"{success_ratio:.0%} of values are numeric coordinate pairs"
    return GeometryEvidence(
        matches,
        "plausible" if matches else "weak",
        "coordinate-pair" if matches else None,
        success_ratio,
        reason,
        ("Point",) if matches else (),
    )


def geometry_evidence(
    series: pd.Series,
    *,
    threshold: float = 0.95,
    limit: int | None = 1_000,
    blank_as_missing: bool = True,
    allow_wkt: bool = True,
    allow_wkb: bool = True,
    allow_geojson: bool = True,
    allow_coordinate_pairs: bool = False,
    require_coordinate_name_hint: bool = True,
    coordinate_order: str = "either",
    allow_longitude_360: bool = False,
    min_decimal_places: int | None = None,
    require_both_precise: bool = False,
    allow_z: bool = True,
) -> GeometryEvidence:
    """Return evidence that ``series`` plausibly contains geometries.

    Geometry dtype is certain evidence. Shapely, WKT, WKB, and GeoJSON are
    strong evidence. Numeric coordinate pairs are only plausible evidence and
    are disabled by default because arbitrary numeric pairs are ambiguous.
    """
    if not isinstance(series, pd.Series):
        raise TypeError("series must be a pandas Series")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if limit is not None:
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an integer or None")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
    if is_geometry(series):
        geometry_types = ()
        try:
            geometry_types = tuple(sorted(series.geom_type.dropna().unique()))
        except AttributeError:
            pass
        return GeometryEvidence(
            True, "certain", "geometry-dtype", 1.0,
            "Series has GeoPandas GeometryDtype", geometry_types,
        )
    values = _sample_geometry_values(
        series, limit=limit, blank_as_missing=blank_as_missing,
    )
    if values.empty:
        return GeometryEvidence(
            False, "unknown", None, 0.0,
            "No non-missing values and no geometry dtype",
        )
    tests: list[tuple[str, str, Callable[[object], object | None]]] = [
        ("shapely", "strong", _parse_shapely_geometry),
    ]
    if allow_wkt:
        tests.append(("wkt", "strong", _parse_wkt_geometry))
    if allow_wkb:
        tests.append(("wkb", "strong", _parse_wkb_geometry))
    if allow_geojson:
        tests.append(("geojson", "strong", _parse_geojson_geometry))
    results = [
        _evaluate_geometry_parser(
            values,
            representation=representation,
            confidence=confidence,
            parser=parser,
            threshold=threshold,
        )
        for representation, confidence, parser in tests
    ]
    best = max(results, key=lambda result: result.success_ratio)
    if best.matches:
        return best
    if allow_coordinate_pairs:
        coordinates = _coordinate_pair_evidence(
            series,
            values,
            threshold=threshold,
            require_name_hint=require_coordinate_name_hint,
            coordinate_order=coordinate_order,
            allow_longitude_360=allow_longitude_360,
            min_decimal_places=min_decimal_places,
            require_both_precise=require_both_precise,
            allow_z=allow_z,
        )
        if coordinates.matches:
            return coordinates
        if coordinates.success_ratio > best.success_ratio:
            best = coordinates
    return GeometryEvidence(
        False,
        "unknown",
        None,
        best.success_ratio,
        (
            "No supported geometry representation met the "
            f"{threshold:.0%} threshold; best result: {best.reason}"
        ),
        best.geometry_types,
    )


def is_geometry_like(series: pd.Series, **kwargs) -> bool:
    """Return whether a Series could plausibly contain geometries."""
    return geometry_evidence(series, **kwargs).matches


def _coordinate_to_point(
    value: object,
    *,
    order: str,
    allow_longitude_360: bool,
    min_decimal_places: int | None,
    require_both_precise: bool,
    allow_z: bool,
):
    if not _is_numeric_coordinate_pair(
        value,
        order=order,
        allow_longitude_360=allow_longitude_360,
        min_decimal_places=min_decimal_places,
        require_both_precise=require_both_precise,
        allow_z=allow_z,
    ):
        raise ValueError(f"{value!r} is not a plausible coordinate pair")
    elements = _coordinate_elements(value)
    assert elements is not None
    numbers = [float(element) for element in elements]
    if order == "lat-lon":
        numbers[0], numbers[1] = numbers[1], numbers[0]
    elif order == "either":
        first, second = numbers[:2]
        # Only swap when the first value cannot be latitude but the second can.
        if abs(first) <= 90 and abs(second) > 90:
            numbers[0], numbers[1] = second, first
    from shapely.geometry import Point
    return Point(numbers)


def as_geometry(
    data,
    *,
    crs=None,
    representation: str | None = None,
    infer: bool = True,
    threshold: float = 0.95,
    allow_coordinate_pairs: bool = False,
    require_coordinate_name_hint: bool = True,
    coordinate_order: str = "either",
    allow_longitude_360: bool = False,
    min_decimal_places: int | None = None,
    require_both_precise: bool = False,
    allow_z: bool = True,
):
    """Return ``data`` as a GeoPandas ``GeoSeries``.

    ``representation`` may be ``"shapely"``, ``"wkt"``, ``"wkb"``,
    ``"geojson"``, or ``"coordinate-pair"``. If omitted, the representation
    is inferred. Coordinate-pair inference is disabled by default and its
    coordinate order should preferably be stated explicitly.
    """
    gpd = _require_geopandas()
    _require_shapely()
    source = data if isinstance(data, pd.Series) else pd.Series(data)
    if is_geometry(source):
        result = gpd.GeoSeries(source, index=source.index, name=source.name)
        if crs is not None:
            try:
                result = result.set_crs(crs, allow_override=False)
            except ValueError as exc:
                raise ValueError(
                    "source already has a different CRS; use .to_crs() to transform it"
                ) from exc
        return result
    valid_representations = {
        "shapely", "wkt", "wkb", "geojson", "coordinate-pair",
    }
    if representation is not None and representation not in valid_representations:
        raise ValueError(
            f"representation must be one of {sorted(valid_representations)!r}"
        )
    if representation is None and infer:
        evidence = geometry_evidence(
            source,
            threshold=threshold,
            allow_coordinate_pairs=allow_coordinate_pairs,
            require_coordinate_name_hint=require_coordinate_name_hint,
            coordinate_order=coordinate_order,
            allow_longitude_360=allow_longitude_360,
            min_decimal_places=min_decimal_places,
            require_both_precise=require_both_precise,
            allow_z=allow_z,
        )
        if not evidence.matches:
            raise ValueError(f"data are not plausibly geometry-like: {evidence.reason}")
        representation = evidence.representation
    elif representation is None:
        raise TypeError("representation is required when infer=False")
    parsers = {
        "shapely": _parse_shapely_geometry,
        "wkt": _parse_wkt_geometry,
        "wkb": _parse_wkb_geometry,
        "geojson": _parse_geojson_geometry,
    }
    converted = []
    for value in source:
        if _is_missing_scalar(value) or (
            isinstance(value, str) and not value.strip()
        ):
            converted.append(None)
        elif representation == "coordinate-pair":
            converted.append(_coordinate_to_point(
                value,
                order=coordinate_order,
                allow_longitude_360=allow_longitude_360,
                min_decimal_places=min_decimal_places,
                require_both_precise=require_both_precise,
                allow_z=allow_z,
            ))
        else:
            geometry = parsers[representation](value)
            if geometry is None:
                raise ValueError(
                    f"{value!r} is not valid {representation.upper()} geometry"
                )
            converted.append(geometry)
    return gpd.GeoSeries(
        converted,
        index=source.index,
        name=source.name,
        crs=crs,
    )


__all__ = [
    "GeometryEvidence",
    "as_geometry",
    "geometry_evidence",
    "is_geometry",
    "is_geometry_like",
]
