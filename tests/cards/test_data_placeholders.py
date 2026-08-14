from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from playwright.sync_api import Page, expect
from shapely.geometry import Point
from shiny import reactive
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from proxyData import ProxyData

app = create_app_fixture(app="../../cards/DataPlaceholders.py", scope="function")
_HELPER_CARDS = {}


@pytest.fixture(scope="session")
def browser_context_args():
    return {"viewport": {"width": 1600, "height": 1000}}


@pytest.fixture
def card_module():
    return importlib.import_module("cards.DataPlaceholders")


class FakeInputs:
    def __init__(
        self,
        *,
        max_obs=3,
        integers=None,
        floats=None,
        strings=None,
        datetimes=None,
        replace=None,
        extrema=True,
        case_sensitive=False,
    ):
        self._max_obs = max_obs
        self._integers = [-999] if integers is None else integers
        self._floats = [-99.99] if floats is None else floats
        self._strings = ["NA"] if strings is None else strings
        self._datetimes = ["1900-01-01"] if datetimes is None else datetimes
        self._replace = [] if replace is None else replace
        self._extrema = extrema
        self._case_sensitive = case_sensitive

    def MaxObs(self):
        return self._max_obs

    def NA_Integers(self):
        return self._integers

    def NA_Floats(self):
        return self._floats

    def NA_Strings(self):
        return self._strings

    def NA_DateTime(self):
        return self._datetimes

    def Replace(self):
        return self._replace

    def NA_Extrema(self):
        return self._extrema

    def NA_CaseSensitive(self):
        return self._case_sensitive


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "integer": pd.Series([-999, 1, pd.NA], dtype="Int64"),
        "decimal": [-99.99, 2.5, np.nan],
        "character": pd.Series(["NA", "present", pd.NA], dtype="string"),
        "date": pd.to_datetime(["1900-01-01", "2025-01-01", None]),
    })


def recorded_helpers(card_module, *, frame=None, inputs=None, fullscreen=False):
    """Expose nested server helpers through inert decorators."""
    card = _HELPER_CARDS.get(card_module)
    if card is None:
        card = card_module.instance()
        _HELPER_CARDS[card_module] = card
    functions = {}

    def capture(function):
        functions[function.__name__] = function
        return function

    card.record_code = capture
    card.suspendable = lambda *args, **kwargs: capture
    card.throttle = lambda *args, **kwargs: capture
    card.isFullScreen = lambda: fullscreen
    source = sample_frame() if frame is None else frame
    proxy = source if isinstance(source, ProxyData) else ProxyData(_df=source, _name="Test")
    with reactive.isolate():
        card._imports.set(proxy)
    card.server(inputs or FakeInputs(), capture, SimpleNamespace())
    raw_codes = functions["RawCodes"]
    closure = dict(zip(
        raw_codes.__code__.co_freevars,
        (cell.cell_contents for cell in (raw_codes.__closure__ or ())),
    ))
    functions["PlaceholderCodes"] = closure["PlaceholderCodes"]
    return card, functions


def get_card(page: Page):
    return page.locator(".card").first


def get_namespace(page: Page) -> str:
    card_id = get_card(page).get_attribute("id")
    assert card_id is not None
    return card_id.partition("-")[0]


def namespaced_id(page: Page, local_id: str) -> str:
    return f"{get_namespace(page)}-{local_id}"


def by_id(page: Page, local_id: str):
    return page.locator(f"#{namespaced_id(page, local_id)}")


def set_shiny_input(page: Page, local_id: str, value):
    page.wait_for_function("() => !!window.Shiny?.setInputValue")
    page.evaluate(
        """
        ([inputId, inputValue]) => window.Shiny.setInputValue(
            inputId, inputValue, {priority: "event"}
        )
        """,
        [namespaced_id(page, local_id), value],
    )


class TestInstance:
    @pytest.mark.unit
    def test_metadata_and_regions(self, card_module):
        card = card_module.this
        assert card.name == "dataPlaceholders"
        assert card.long_name == "Missing value placeholders"
        assert "replacement with NA" in card.description
        assert card.mutable
        assert card.hasSidebar()
        assert card.hasFlipSide()
        assert card.hasFooter()

    @pytest.mark.unit
    def test_settings_contain_expected_controls(self, card_module):
        html = str(card_module.this.settings)
        for control in (
            "NA_Strings", "NA_CaseSensitive", "NA_Integers", "NA_Floats",
            "NA_Extrema", "NA_DateTime", "MaxObs",
        ):
            assert f'id="{control}"' in html

    @pytest.mark.unit
    def test_front_contains_all_tabs_and_charts(self, card_module):
        html = str(card_module.this.front.tagify())
        for label in ("All variables", "Integer", "Decimal", "Character", "Dates &amp; Times"):
            assert label in html
        for output_id in ("AllChart", "IntegerChart", "FloatChart", "CharacterChart", "DateChart"):
            assert f'id="{output_id}"' in html

    @pytest.mark.unit
    def test_back_and_footer_outputs(self, card_module):
        assert 'id="Summary"' in str(card_module.this.back)
        assert "Placeholder Summary" in str(card_module.this.back)
        assert 'id="Replace"' in str(card_module.this.footer)

    @pytest.mark.unit
    def test_test_mode_seeds_expected_data(self, card_module):
        with reactive.isolate():
            proxy = card_module.this._imports.get()
        assert proxy.to_native().shape == (4, 5)
        assert proxy.to_native().columns.tolist() == ["y", "x1", "x2", "id", "part"]


class TestServerInputs:
    @pytest.mark.unit
    def test_max_observations_is_logarithmic(self, card_module):
        _, functions = recorded_helpers(card_module, inputs=FakeInputs(max_obs=5))
        assert functions["MaxObs"]() == 100_000

    @pytest.mark.unit
    def test_sentinels_preserve_input_groups(self, card_module):
        inputs = FakeInputs(
            integers=[-1], floats=[-1.5], strings=["missing"], datetimes=["2000-01-01"]
        )
        _, functions = recorded_helpers(card_module, inputs=inputs)
        assert functions["Sentinels"]() == {
            "int": [-1],
            "float": [-1.5],
            "str": ["missing"],
            "datetime": ["2000-01-01"],
        }

    @pytest.mark.unit
    def test_prepared_data_respects_sample_limit(self, card_module):
        frame = pd.DataFrame({"value": range(1500)})
        _, functions = recorded_helpers(
            card_module, frame=frame, inputs=FakeInputs(max_obs=3)
        )
        with reactive.isolate():
            result = functions["PreparedData"]()
        assert isinstance(result, ProxyData)
        assert result.shape == (1000, 1)


class TestPlaceholderCodes:
    @pytest.fixture
    def helpers(self, card_module):
        return recorded_helpers(card_module)[1]

    @pytest.mark.unit
    def test_codes_missing_and_each_sentinel(self, helpers):
        codes, legend = helpers["PlaceholderCodes"](
            sample_frame(),
            {"int": [-999], "float": [-99.99], "str": ["NA"], "datetime": ["1900-01-01"]},
        )
        assert codes.iloc[2].eq(0).all()
        assert codes.iloc[0].tolist() == [2, 3, 4, 5]
        assert codes.iloc[1].eq(1).all()
        assert legend == {
            0: "Missing", 1: "Not Missing", 2: "int: -999",
            3: "float: -99.99", 4: "str: NA", 5: "datetime: 1900-01-01",
        }

    @pytest.mark.unit
    def test_extrema_filter_can_exclude_internal_numeric_value(self, helpers):
        frame = pd.DataFrame({"value": pd.Series([-100, -99, 3], dtype="Int64")})
        strict, _ = helpers["PlaceholderCodes"](frame, {"int": [-99]}, extrema=True)
        permissive, _ = helpers["PlaceholderCodes"](frame, {"int": [-99]}, extrema=False)
        assert strict["value"].tolist() == [1, 1, 1]
        assert permissive["value"].tolist() == [1, 2, 1]

    @pytest.mark.unit
    def test_case_sensitive_string_matching(self, helpers):
        frame = pd.DataFrame({"value": pd.Series(["NA", "na", "ok"], dtype="string")})
        insensitive, _ = helpers["PlaceholderCodes"](
            frame, {"str": ["NA"]}, case_sensitive=False
        )
        sensitive, _ = helpers["PlaceholderCodes"](
            frame, {"str": ["NA"]}, case_sensitive=True
        )
        assert insensitive["value"].tolist() == [2, 2, 1]
        assert sensitive["value"].tolist() == [2, 1, 1]

    @pytest.mark.unit
    def test_geometry_is_dropped_without_losing_other_columns(self, helpers):
        frame = gpd.GeoDataFrame(
            {"value": ["NA", "ok"], "geometry": [Point(1, 2), Point(3, 4)]},
            geometry="geometry",
            crs="EPSG:4326",
        )
        codes, _ = helpers["PlaceholderCodes"](
            frame, {"str": ["NA"]}, drop_geometry=True
        )
        assert codes.columns.tolist() == ["value"]

    @pytest.mark.unit
    def test_proxy_data_is_accepted(self, helpers):
        proxy = ProxyData(_df=sample_frame(), _name="Sentinels")
        codes, _ = helpers["PlaceholderCodes"](proxy, {"str": ["NA"]})
        assert codes.shape == sample_frame().shape

    @pytest.mark.unit
    def test_unknown_data_type_is_rejected(self, helpers):
        with pytest.raises(TypeError, match="Unknown dataset type"):
            helpers["PlaceholderCodes"]([1, 2, 3], {"int": [-1]})


class TestResolutionAndCharts:
    @pytest.fixture
    def helpers(self, card_module):
        return recorded_helpers(card_module)[1]

    @pytest.mark.unit
    def test_resolve_string_placeholder_without_mutating_input(self, helpers):
        frame = pd.DataFrame({"value": ["NA", "na", "present"]})
        result = helpers["ResolvePlaceholders"](
            frame, ["str: NA"], case_sensitive=False
        )
        assert result["value"].isna().tolist() == [True, True, False]
        assert frame["value"].tolist() == ["NA", "na", "present"]

    @pytest.mark.unit
    def test_resolve_float_placeholder(self, helpers):
        frame = pd.DataFrame({"value": [-99.99, 1.5, 2.5]})
        result = helpers["ResolvePlaceholders"](frame, ["float: -99.99"])
        assert result["value"].isna().tolist() == [True, False, False]

    @pytest.mark.unit
    def test_empty_sentinel_selection_returns_data_unchanged(self, helpers):
        frame = pd.DataFrame({"value": [1, 2]})
        assert helpers["ResolvePlaceholders"](frame, []) is frame

    @pytest.mark.unit
    def test_placeholder_chart_contains_transposed_heatmap(
        self, helpers, card_module, monkeypatch
    ):
        monkeypatch.setattr(card_module.go, "FigureWidget", lambda figure: figure)
        codes = pd.DataFrame({"a": [1, 2, 0], "b": [1, 1, 3]})
        figure = helpers["_placeholder_chart"](
            codes, {0: "Missing", 1: "Not Missing", 2: "str: NA", 3: "int: -1"}, fs=False
        )
        assert len(figure.data) == 1
        assert figure.data[0].type == "heatmap"
        assert np.asarray(figure.data[0].z).shape == (2, 3)
        assert figure.layout.showlegend is False
        assert figure._config["displayModeBar"] is False

    @pytest.mark.unit
    def test_fullscreen_chart_adds_hover_and_legend_traces(
        self, helpers, card_module, monkeypatch
    ):
        monkeypatch.setattr(card_module.go, "FigureWidget", lambda figure: figure)
        codes = pd.DataFrame({"a": [1, 2, 0]})
        figure = helpers["_placeholder_chart"](
            codes, {0: "Missing", 1: "Not Missing", 2: "str: NA"}, fs=True
        )
        assert [trace.type for trace in figure.data] == ["heatmap", "scatter", "scatter", "scatter", "scatter"]
        assert figure.layout.showlegend is True
        assert figure._config["displayModeBar"] is True


class TestWebKitUI:
    @pytest.mark.ui
    def test_card_tabs_and_empty_replace_group_render(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        expect(get_card(page)).to_be_visible()
        expect(page.get_by_text("Missing value placeholders", exact=True)).to_be_visible()
        for label in ("All variables", "Integer", "Decimal", "Character", "Dates & Times"):
            expect(page.get_by_role("tab", name=label, exact=True)).to_be_visible()
        expect(by_id(page, "Replace").locator('input[type="checkbox"]')).to_have_count(0)

    @pytest.mark.ui
    @pytest.mark.parametrize(
        ("tab", "output_id"),
        [
            ("Integer", "IntegerChart"),
            ("Decimal", "FloatChart"),
            ("Character", "CharacterChart"),
            ("Dates & Times", "DateChart"),
            ("All variables", "AllChart"),
        ],
    )
    def test_chart_tabs_can_be_selected(self, page: Page, app: ShinyAppProc, tab, output_id):
        page.goto(app.url)
        page.get_by_role("tab", name=tab, exact=True).click()
        expect(by_id(page, output_id)).to_be_visible()

    @pytest.mark.ui
    def test_declaring_a_present_string_sentinel_adds_replace_choice(
        self, page: Page, app: ShinyAppProc
    ):
        page.goto(app.url)
        set_shiny_input(page, "NA_Strings", ["NA", "A"])
        expect(by_id(page, "Replace")).to_contain_text("Replace str: A", timeout=10_000)

    @pytest.mark.ui
    def test_flip_displays_placeholder_summary(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        by_id(page, "FlipButton").click(force=True)
        expect(by_id(page, "Summary")).to_be_visible()
        expect(by_id(page, "Summary")).to_contain_text("Variable")

    @pytest.mark.ui
    def test_fullscreen_toggle_keeps_card_visible(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        by_id(page, "ExpandButton").click(force=True)
        expect(get_card(page)).to_be_visible()
