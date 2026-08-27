import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

path = Path(__file__).resolve().parent.parent.parent / 'app'
os.chdir(path)
if str(path) not in sys.path:
    sys.path.insert(0, str(path))

import geopandas as gpd
import pandas as pd
import pytest
from cyclic_pandas import as_cyclic
from geometry_pandas import as_geometry
from list_pandas import as_list
from playwright.sync_api import Page, expect
from proxy_data import proxy_data
from roles import Role, RoleMap
from shapely.geometry import LineString, Point
from shiny import reactive
from shiny.playwright import controller
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc
from text_pandas import as_text

app = create_app_fixture(app="../../app/cards/data_tabulation.py", scope="function")
_HELPER_CARDS = {}


@pytest.fixture(scope="session")
def browser_context_args():
    return {"viewport": {"width": 1600, "height": 1000}}


@pytest.fixture
def card_module():
    return importlib.import_module("cards.data_tabulation")


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
    page.evaluate(
        """
        ([inputId, value]) => {
            if (!window.Shiny || !window.Shiny.setInputValue) {
                throw new Error("Shiny.setInputValue not available");
            }
            window.Shiny.setInputValue(inputId, value, {priority: "event"});
        }
        """,
        [namespaced_id(page, local_id), value],
    )


def recorded_helpers(card_module, *, data=None, decimals=2, bounded=True, fullscreen=False):
    """Expose helpers registered inside ``server`` using inert decorators."""
    card = _HELPER_CARDS.get(card_module)
    if card is None:
        card = card_module.instance()
        _HELPER_CARDS[card_module] = card
    functions = {}

    def record_code(function):
        functions[function.__name__] = function
        return function

    card.record_code = record_code
    card.suspendable = lambda **kwargs: lambda function: function
    card.throttle = lambda *args, **kwargs: lambda function: function
    card.isFullScreen = lambda: fullscreen
    if data is not None:
        with reactive.isolate():
            card._imports.set(
                data if isinstance(data, proxy_data) else proxy_data(_df=data, _name="Test")
            )
    inputs = SimpleNamespace(
        Decimals=lambda: decimals,
        Bounded=lambda: bounded,
        MaxObs=lambda: 3,
    )
    card.server(inputs, lambda function: function, None)
    return card, functions


class TestInstance:
    @pytest.mark.unit
    def test_metadata(self, card_module):
        card = card_module.this
        assert card.name == "data_tabulation"
        assert card.long_name == "Data tabulation"
        assert "listed and searched" in card.description
        assert not card.mutable

    @pytest.mark.unit
    def test_ui_regions_are_present(self, card_module):
        card = card_module.this
        assert card.front is not None
        assert card.settings is not None
        assert card.footer is not None
        assert card.back is not None
        assert card.hasSidebar()
        assert card.hasFooter()
        assert card.hasFlipSide()
        assert 'id="StructTable"' in str(card.back)

    @pytest.mark.unit
    def test_test_mode_seeds_expected_data(self, card_module):
        with reactive.isolate():
            assert card_module.this._imports.is_set()
            frame = card_module.this._imports.get().to_native()
        assert frame.shape == (4, 5)
        assert frame.columns.tolist() == ["y", "x1", "x2", "id", "part"]
        assert frame["x1"].tolist() == [10.0, 11.0, 12.0, 13.0]

    @pytest.mark.unit
    def test_settings_have_expected_controls_and_defaults(self, card_module):
        html = str(card_module.this.settings)
        assert "Decimals" in html
        assert "Bounded" in html
        assert "MaxObs" in html
        assert "Number of decimal places to show" in html
        assert "bounding box" in html
        assert "Maximum observations" in html

    @pytest.mark.unit
    def test_footer_contains_csv_export(self, card_module):
        html = str(card_module.this.footer)
        assert "Export" in html
        assert "download" in html.lower()


class TestDtypeLabels:
    @pytest.fixture
    def label(self, card_module):
        _, functions = recorded_helpers(card_module)
        return functions["_dtype_label_from_dtype"]

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("dtype", "expected"),
        [
            (pd.Series([1, 2], dtype="int64").dtype, "int"),
            (pd.Series([1.0, 2.0]).dtype, "dec"),
            (pd.Series([True, False]).dtype, "log"),
            (pd.Series(pd.to_datetime(["2025-01-01"])).dtype, "dte"),
            (pd.CategoricalDtype(["a", "b"], ordered=False), "nom"),
            (pd.CategoricalDtype(["low", "high"], ordered=True), "ord"),
            (pd.Series([object()], dtype=object).dtype, "obj"),
            (pd.Series(["A", "B", "C"], dtype="string").dtype, "cde"),
        ],
    )
    def test_standard_dtype_labels(self, label, dtype, expected):
        assert label(dtype) == expected

    @pytest.mark.unit
    def test_cyclic_dtype_label(self, label):
        dtype = as_cyclic(pd.Series([0, 1]), period=12).dtype
        assert label(dtype) == "cyc"

    @pytest.mark.unit
    def test_text_dtype_label(self, label):
        assert label(as_text(["long free text"]).dtype) == "txt"

    @pytest.mark.unit
    def test_geometry_dtype_label(self, label):
        assert label(as_geometry([Point(1, 2)]).dtype) == "geo"

    @pytest.mark.unit
    def test_list_dtype_label(self, label):
        assert label(as_list([["bread", "milk"]]).dtype) == "bkt"


class TestCleanDf:
    @pytest.mark.unit
    def test_headtail_preview_and_type_headers(self, card_module):
        frame = pd.DataFrame({
            "integer": range(12),
            "decimal": [value + 0.1234 for value in range(12)],
            "category": pd.Categorical(["a", "b"] * 6),
        })
        _, functions = recorded_helpers(card_module, data=frame, decimals=2)
        with reactive.isolate():
            result = functions["CleanDf"]()
        assert result.shape == (10, 3)
        assert result.index.tolist() == [0, 1, 2, 3, 4, 7, 8, 9, 10, 11]
        assert result.columns.tolist() == [
            "integer\nint", "decimal\ndec", "category\nnom"
        ]
        assert result["decimal\ndec"].iloc[0] == pytest.approx(0.12)

    @pytest.mark.unit
    def test_none_decimals_skips_rounding(self, card_module):
        frame = pd.DataFrame({"value": [1.23456]})
        _, functions = recorded_helpers(card_module, data=frame, decimals=None)
        with reactive.isolate():
            result = functions["CleanDf"]()
        assert result.iloc[0, 0] == pytest.approx(1.23456)

    @pytest.mark.unit
    def test_point_geometry_has_compact_bounded_display(self, card_module):
        frame = gpd.GeoDataFrame(
            {"name": ["A"], "geometry": [Point(174.763336, -36.848461)]},
            geometry="geometry",
            crs="EPSG:4326",
        )
        _, functions = recorded_helpers(card_module, data=frame, bounded=True)
        with reactive.isolate():
            result = functions["CleanDf"]()
        assert "geometry\ngeometry active EPSG:4326" in result.columns
        assert result.iloc[0]["geometry\ngeometry active EPSG:4326"] == (
            "Point(174.7633, -36.8485)"
        )

    @pytest.mark.unit
    def test_nonpoint_geometry_has_bounding_box_display(self, card_module):
        frame = gpd.GeoDataFrame(
            {"geometry": [LineString([(1, 2), (3, 5)])]},
            geometry="geometry",
            crs="EPSG:4326",
        )
        _, functions = recorded_helpers(card_module, data=frame, bounded=True)
        with reactive.isolate():
            result = functions["CleanDf"]()
        assert result.iloc[0, 0] == "LineString bound by 1.0000,2.0000 to 3.0000,5.0000"

    @pytest.mark.unit
    def test_unbounded_geometry_uses_wkt(self, card_module):
        frame = gpd.GeoDataFrame(
            {"geometry": [Point(1, 2)]}, geometry="geometry", crs="EPSG:4326"
        )
        _, functions = recorded_helpers(card_module, data=frame, bounded=False)
        with reactive.isolate():
            result = functions["CleanDf"]()
        assert result.iloc[0, 0] == "POINT (1 2)"

    @pytest.mark.unit
    def test_fullscreen_uses_maximum_observation_sample(self, card_module):
        frame = pd.DataFrame({"value": range(1500)})
        _, functions = recorded_helpers(card_module, data=frame, fullscreen=True)
        with reactive.isolate():
            result = functions["CleanDf"]()
        assert len(result) == 1000
        assert result.index.is_monotonic_increasing


class TestStructureData:
    @pytest.mark.unit
    def test_structure_has_one_row_per_variable_with_roles_and_counts(
        self, card_module
    ):
        frame = pd.DataFrame({
            "amount": [1.0, 2.0, None, 4.0],
            "group": pd.Categorical(["A", "A", "B", None]),
            "when": pd.to_datetime([
                "2025-01-01", "2025-01-02", None, "2025-01-04"
            ]),
        })
        roles = RoleMap()
        roles.set_roles("amount", [Role.PREDICTOR])
        roles.set_roles("group", [Role.STRATIFIER])
        roles.set_roles("when", [Role.SEQUENCE])
        px = proxy_data(_df=frame, _roles=roles, _name="Test")
        _, functions = recorded_helpers(card_module, data=px)

        with reactive.isolate():
            result = functions["StructureData"]()

        assert result["Variable"].tolist() == ["amount", "group", "when"]
        assert result.columns.tolist() == [
            "Variable", "Data type", "Storage type", "Role", "Complete",
            "Missing", "Missing %", "Unique", "Summary",
        ]
        amount = result.set_index("Variable").loc["amount"]
        assert amount["Role"] == "predictor"
        assert amount["Complete"] == 3
        assert amount["Missing"] == 1
        assert amount["Missing %"] == pytest.approx(25.0)
        assert amount["Unique"] == 3
        assert "median 2" in amount["Summary"]
        assert "mode: A (2)" in result.set_index("Variable").loc["group", "Summary"]
        assert "2025-01-01" in result.set_index("Variable").loc["when", "Summary"]

    @pytest.mark.unit
    def test_custom_dtype_summaries_are_conservative(self, card_module):
        _, functions = recorded_helpers(card_module)
        basket = as_list([["bread", "milk"], ["bread"], None])
        geometry = as_geometry([Point(1, 2), LineString([(0, 0), (1, 1)]), None])
        cycle = as_cyclic(pd.Series([0, 1, 2]), period=12)
        text = as_text(["short", "a little longer", None])

        assert functions["_safe_unique_count"](basket) is None
        assert functions["_safe_unique_count"](geometry) is None
        assert "median 1.5" in functions["_column_summary"](basket)
        assert "LineString, Point" in functions["_column_summary"](geometry)
        assert functions["_column_summary"](cycle) == "cycle period: 12"
        assert "characters" in functions["_column_summary"](text)


class TestWebKitUI:
    @pytest.mark.ui
    def test_card_and_data_grid_render(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        expect(get_card(page)).to_be_visible()
        expect(page.get_by_text("Data tabulation")).to_be_visible()
        grid = controller.OutputDataFrame(page, namespaced_id(page, "DataTable2"))
        grid.expect_ncol(5)
        grid.expect_nrow(4)

    @pytest.mark.ui
    def test_seeded_column_labels_include_type_codes(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        grid = controller.OutputDataFrame(page, namespaced_id(page, "DataTable2"))
        grid.expect_column_labels([
            "y\nint", "x1\ndec", "x2\ncde", "id\nint", "part\ncde"
        ])

    @pytest.mark.ui
    def test_seeded_values_are_visible(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        grid = controller.OutputDataFrame(page, namespaced_id(page, "DataTable2"))
        grid.expect_cell("10", row=0, col=1)
        grid.expect_cell("A", row=0, col=2)
        grid.expect_cell("Test", row=3, col=4)

    @pytest.mark.ui
    def test_flip_side_displays_structure_table(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        by_id(page, "FlipButton").click(force=True)
        grid = controller.OutputDataFrame(page, namespaced_id(page, "Structure"))
        grid.expect_nrow(5)
        grid.expect_ncol(9)
        grid.expect_column_labels([
            "Variable", "Data type", "Storage type", "Role", "Complete",
            "Missing", "Missing %", "Unique", "Summary",
        ])

    @pytest.mark.ui
    def test_settings_controls_exist(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        expect(by_id(page, "Decimals")).to_be_attached()
        expect(by_id(page, "Bounded")).to_be_attached()
        expect(by_id(page, "MaxObs")).to_be_attached()

    @pytest.mark.ui
    def test_changing_decimals_keeps_grid_available(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        set_shiny_input(page, "Decimals", 0)
        grid = controller.OutputDataFrame(page, namespaced_id(page, "DataTable2"))
        grid.expect_nrow(4)

    @pytest.mark.ui
    def test_fullscreen_toggle_keeps_grid_working(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        by_id(page, "ExpandButton").click(force=True)
        expect(get_card(page)).to_be_visible()
        grid = controller.OutputDataFrame(page, namespaced_id(page, "DataTable2"))
        grid.expect_nrow(4)

    @pytest.mark.ui
    def test_export_control_is_available(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        export = by_id(page, "Export")
        expect(export).to_be_visible()
        with page.expect_download() as download_info:
            export.click()
        assert download_info.value.suggested_filename == "data_tabulation_data.csv"
