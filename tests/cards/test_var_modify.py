import importlib

import pandas as pd
import pytest
from playwright.sync_api import Page, expect
from shiny import reactive
from shiny.playwright import controller
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

from cyclic_pandas import is_cyclic
from geometry_pandas import is_geometry
from list_pandas import is_list
from proxyData import ProxyData
from roles import Role
from text_pandas import is_text

app = create_app_fixture(app="../../cards/VarModify.py", scope="function")
_HELPER_CARDS = {}


@pytest.fixture(scope="session")
def browser_context_args():
    return {"viewport": {"width": 1600, "height": 1000}}


@pytest.fixture
def card_module():
    return importlib.import_module("cards.VarModify")


class FakeInputs:
    def __init__(self, *, max_obs=1000, alternatives="Sensible"):
        self.max_obs = max_obs
        self.alternatives = alternatives
        self.Formats = lambda: "%Y-%m-%d, %d/%m/%Y"

    def MaxObs(self):
        return self.max_obs

    def Alternatives(self):
        return self.alternatives

    def NewName(self):
        return ""

    def NewDataType(self):
        return ""

    def NewOrder(self):
        return []

    def Commit(self):
        return 0

    def Reset(self):
        return 0


class FakeSession:
    def ns(self, value):
        return f"varModify-{value}"

    def on_flushed(self, function, *, once=False):
        return None

    async def send_custom_message(self, name, payload):
        return None


def closure_values(function) -> dict[str, object]:
    closure = function.__closure__ or ()
    return dict(zip(function.__code__.co_freevars, (cell.cell_contents for cell in closure)))


def recorded_helpers(card_module, *, frame=None, max_obs=1000):
    card = _HELPER_CARDS.get(card_module)
    if card is None:
        card = card_module.instance()
        _HELPER_CARDS[card_module] = card
    functions = {}

    def capture(function):
        functions[function.__name__] = function
        return function

    card.suspendable = lambda **kwargs: capture
    card.throttle = lambda *args, **kwargs: capture
    card.isFullScreen = lambda: False
    source = frame if frame is not None else pd.DataFrame({
        "integer": pd.Series([1, 2, 3], dtype="int64"),
        "decimal": pd.Series([1.25, 2.5, 3.75], dtype="float64"),
        "code": pd.Series(["A", "B", "C"], dtype="string"),
        "nominal": pd.Series(pd.Categorical(["a", "b", "a"])),
        "ordered": pd.Series(pd.Categorical(
            ["low", "high", "medium"],
            categories=["low", "medium", "high"],
            ordered=True,
        )),
        "when": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
    })
    proxy = source if isinstance(source, ProxyData) else ProxyData(_df=source, _name="Test")
    with reactive.isolate():
        card._imports.set(proxy)
    inputs = FakeInputs(max_obs=max_obs)
    card.server(inputs, lambda function: function, FakeSession())
    inference = closure_values(functions["allowed_d_types"])
    conversion = closure_values(functions["CommitEvent"])
    return card, inputs, functions, inference, conversion


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


class TestInstance:
    @pytest.mark.unit
    def test_metadata(self, card_module):
        card = card_module.this
        assert card.name == "varModify"
        assert card.long_name == "Modification"
        assert "modification of variables" in card.description
        assert card.mutable

    @pytest.mark.unit
    def test_expected_ui_regions(self, card_module):
        card = card_module.this
        assert card.front is not None
        assert card.back is not None
        assert card.footer is not None
        assert card.settings is not None
        assert card.hasFlipSide()
        assert card.hasSidebar()
        assert card.hasFooter()

    @pytest.mark.unit
    def test_front_back_footer_and_settings_controls(self, card_module):
        assert 'id="Table"' in str(card_module.this.front)
        assert 'id="DFDiff"' in str(card_module.this.back)
        footer = str(card_module.this.footer)
        for control in ("NewName", "NewDataType", "NewOrder", "Commit", "Reset"):
            assert f'id="{control}"' in footer
        settings = str(card_module.this.settings)
        for control in ("Formats", "Alternatives", "MaxObs"):
            assert f'id="{control}"' in settings

    @pytest.mark.unit
    def test_seeded_test_data_have_expected_columns_and_dtypes(self, card_module):
        with reactive.isolate():
            frame = card_module.this._imports.get().to_native()
        assert frame.shape == (4, 12)
        assert frame.columns.tolist() == [
            "y32", "y64", "x32", "x64", "log", "cat", "id", "part",
            "items", "date_text", "date_DT", "date_D",
        ]
        assert str(frame["y32"].dtype) == "int32"
        assert str(frame["x32"].dtype) == "float32"
        assert isinstance(frame["cat"].dtype, pd.CategoricalDtype)
        assert pd.api.types.is_datetime64_any_dtype(frame["date_DT"])


class TestSchema:
    @pytest.mark.unit
    def test_max_observations_is_direct_count(self, card_module):
        _, _, functions, _, _ = recorded_helpers(card_module, max_obs=1234)
        assert functions["MaxObs"]() == 1234

    @pytest.mark.unit
    def test_prepared_data_respects_sample_limit(self, card_module):
        frame = pd.DataFrame({"value": range(1500)})
        _, _, functions, _, _ = recorded_helpers(
            card_module, frame=frame, max_obs=1000
        )
        with reactive.isolate():
            result = functions["PreparedData"]()
        assert isinstance(result, ProxyData)
        assert result.shape == (1000, 1)

    @pytest.mark.unit
    def test_schema_columns_and_type_labels(self, card_module):
        _, _, functions, _, _ = recorded_helpers(card_module)
        with reactive.isolate():
            result = functions["Schema"]()
        assert result.columns.tolist() == [
            "Orig\nname", "New\nname", "Orig\nd-type", "New\nd-type",
            "Orig\norder", "New\norder", "Role",
        ]
        types = result.set_index("Orig\nname")["Orig\nd-type"].to_dict()
        assert types == {
            "integer": "integer",
            "decimal": "decimal",
            "code": "code",
            "nominal": "nominal",
            "ordered": "ordered",
            "when": "datetime",
        }

    @pytest.mark.unit
    def test_schema_preserves_ordered_category_levels(self, card_module):
        _, _, functions, _, _ = recorded_helpers(card_module)
        with reactive.isolate():
            result = functions["Schema"]()
        row = result.set_index("Orig\nname").loc["ordered"]
        assert row["Orig\norder"] == "low,medium,high"
        assert row["New\norder"] == "low,medium,high"

    @pytest.mark.unit
    def test_schema_reports_first_role(self, card_module):
        frame = pd.DataFrame({"value": [1, 2, 3]})
        proxy = ProxyData(_df=frame, _name="Roles")
        proxy.role_map.set_roles("value", [Role.TARGET])
        _, _, functions, _, _ = recorded_helpers(card_module, frame=proxy)
        with reactive.isolate():
            result = functions["Schema"]()
        assert result.loc[0, "Role"] == "target"


class TestInferenceHelpers:
    @pytest.fixture
    def helpers(self, card_module):
        _, _, _, inference, _ = recorded_helpers(card_module)
        return inference

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("values", "expected"),
        [(["1", "2.5", "3"], True), (["one", "two"], False), ([1, 2, 3], True)],
    )
    def test_numeric_like(self, helpers, values, expected):
        assert helpers["is_numeric_like"](pd.Series(values)) is expected

    @pytest.mark.unit
    def test_numeric_like_threshold_controls_dirty_values(self, helpers):
        series = pd.Series(["1", "2", "bad"])
        assert not helpers["is_numeric_like"](series)
        assert helpers["is_numeric_like"](series, threshold=2 / 3)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("values", "expected"),
        [(["1", "2.0", "-3"], True), (["1", "2.2", "3"], False), ([1, 2, 3], True)],
    )
    def test_integer_like(self, helpers, values, expected):
        assert helpers["is_integer_like"](pd.Series(values)) is expected

    @pytest.mark.unit
    def test_date_like_uses_configured_formats(self, helpers):
        function = helpers["is_date_like"]
        assert function(pd.Series(["2025-01-01", "2025-01-02"]), formats=["%Y-%m-%d"])
        assert not function(pd.Series([1, 2]), formats=["%Y-%m-%d"])

    @pytest.mark.unit
    def test_nominal_like_requires_low_absolute_and_relative_cardinality(self, helpers):
        function = helpers["is_nominal_like"]
        assert function(pd.Series(["a", "b"] * 10))
        assert not function(pd.Series([f"code-{i}" for i in range(20)]))

    @pytest.mark.unit
    def test_ordered_like_accepts_declared_order(self, helpers):
        series = pd.Series(pd.Categorical(
            ["low", "medium", "high"],
            categories=["low", "medium", "high"],
            ordered=True,
        ))
        assert helpers["is_ordered_like"](series)

    @pytest.mark.unit
    def test_ordered_like_can_match_known_order(self, helpers):
        series = pd.Series(["low", "high", "medium", "low"])
        assert helpers["is_ordered_like"](
            series, known_orders=[("low", "medium", "high")]
        )


class TestConversions:
    @pytest.fixture
    def convert(self, card_module):
        _, _, _, _, conversion = recorded_helpers(card_module)
        return conversion["_convert_series"]

    @pytest.mark.unit
    def test_decimal_conversion(self, convert):
        result = convert(pd.Series(["1.5", "bad", None]), "decimal", order=None, formats=[])
        assert str(result.dtype) == "Float64"
        assert result.iloc[0] == 1.5
        assert pd.isna(result.iloc[1])

    @pytest.mark.unit
    def test_integer_conversion_rounds_and_is_nullable(self, convert):
        result = convert(pd.Series([1.2, 2.8, None]), "integer", order=None, formats=[])
        assert str(result.dtype) == "Int64"
        assert result.tolist()[:2] == [1, 3]
        assert pd.isna(result.iloc[2])

    @pytest.mark.unit
    def test_date_conversion(self, convert):
        result = convert(
            pd.Series(["31/12/2025", "01/01/2026"]),
            "date", order=None, formats=["%d/%m/%Y"],
        )
        assert pd.api.types.is_datetime64_any_dtype(result)
        assert result.dt.day.tolist() == [31, 1]

    @pytest.mark.unit
    def test_nominal_and_ordered_conversion(self, convert):
        nominal = convert(pd.Series(["a", "b", "a"]), "nominal", order=None, formats=[])
        assert isinstance(nominal.dtype, pd.CategoricalDtype)
        ordered = convert(
            pd.Series(["medium", "low", "high"]),
            "ordered", order="low,medium,high", formats=[],
        )
        assert ordered.ordered
        assert ordered.categories.tolist() == ["low", "medium", "high"]

    @pytest.mark.unit
    def test_code_and_text_conversion(self, convert):
        code = convert(pd.Series(["A", "B"]), "code", order=None, formats=[])
        assert isinstance(code.dtype, pd.StringDtype)
        text = convert(pd.Series(["some prose", "more prose"]), "text", order=None, formats=[])
        assert is_text(text)

    @pytest.mark.unit
    def test_list_geometry_and_cyclic_conversion(self, convert):
        basket = convert(pd.Series(["a,b", "c,d"]), "list", order=None, formats=[])
        assert is_list(basket)
        geometry = convert(
            pd.Series(["POINT (1 2)", "POINT (3 4)"]),
            "geometry", order=None, formats=[],
        )
        assert is_geometry(geometry)
        categorical = pd.Series(pd.Categorical(
            ["Mon", "Tue", "Wed"],
            categories=["Mon", "Tue", "Wed"],
            ordered=True,
        ))
        assert is_cyclic(convert(
            categorical, "cyclic", order="Mon,Tue,Wed", formats=[]
        ))

    @pytest.mark.unit
    def test_unknown_conversion_is_rejected(self, convert):
        with pytest.raises(ValueError, match="Unsupported conversion type"):
            convert(pd.Series([1]), "unknown", order=None, formats=[])


class TestWebKit:
    @pytest.mark.ui
    def test_card_and_schema_grid_render(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        expect(get_card(page)).to_be_visible()
        expect(page.get_by_text("Modification", exact=True)).to_be_visible()
        grid = controller.OutputDataFrame(page, namespaced_id(page, "Table"))
        grid.expect_nrow(12)
        grid.expect_ncol(7)

    @pytest.mark.ui
    def test_grid_has_expected_schema_headers(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        grid = controller.OutputDataFrame(page, namespaced_id(page, "Table"))
        grid.expect_column_labels([
            "Orig\nname", "New\nname", "Orig\nd-type", "New\nd-type",
            "Orig\norder", "New\norder", "Role",
        ])

    @pytest.mark.ui
    def test_first_variable_is_selected_and_controls_are_populated(
        self, page: Page, app: ShinyAppProc
    ):
        page.goto(app.url)
        grid = controller.OutputDataFrame(page, namespaced_id(page, "Table"))
        grid.expect_selected_rows([0])
        expect(by_id(page, "NewName")).to_have_value("y32")

    @pytest.mark.ui
    def test_commit_starts_disabled(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        expect(by_id(page, "Commit")).to_be_disabled()

    @pytest.mark.ui
    def test_renaming_first_variable_enables_commit(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        grid = controller.OutputDataFrame(page, namespaced_id(page, "Table"))
        grid.expect_selected_rows([0])
        name = controller.InputText(page, namespaced_id(page, "NewName"))
        name.set("outcome")
        expect(by_id(page, "Commit")).to_be_enabled()
        grid.expect_cell("outcome", row=0, col=1)

    @pytest.mark.ui
    def test_reset_restores_pending_rename(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        grid = controller.OutputDataFrame(page, namespaced_id(page, "Table"))
        grid.expect_selected_rows([0])
        controller.InputText(page, namespaced_id(page, "NewName")).set("outcome")
        expect(by_id(page, "Commit")).to_be_enabled()
        by_id(page, "Reset").click()
        grid.expect_cell("y32", row=0, col=1)
        expect(by_id(page, "Commit")).to_be_disabled()

    @pytest.mark.ui
    def test_commit_and_diff_report_rename(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        grid = controller.OutputDataFrame(page, namespaced_id(page, "Table"))
        grid.expect_selected_rows([0])
        controller.InputText(page, namespaced_id(page, "NewName")).set("outcome")
        by_id(page, "Commit").click()
        get_card(page).hover()
        by_id(page, "FlipButton").click(force=True)
        diff = by_id(page, "DFDiff")
        expect(diff).to_be_visible()
        expect(diff).to_contain_text("outcome")

    @pytest.mark.ui
    def test_settings_controls_are_attached(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        expect(by_id(page, "Formats")).to_be_attached()
        expect(by_id(page, "Alternatives")).to_be_attached()
        expect(by_id(page, "MaxObs")).to_be_attached()

    @pytest.mark.ui
    def test_expand_and_restore_keep_grid_available(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        get_card(page).hover()
        by_id(page, "ExpandButton").click(force=True)
        restore = by_id(page, "ContractButton")
        expect(restore).to_be_visible()
        restore.click(force=True)
        controller.OutputDataFrame(page, namespaced_id(page, "Table")).expect_nrow(12)
