from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from playwright.sync_api import Page, expect
from shiny import reactive
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from proxyData import ProxyData
from roles import Role, RoleMap

app = create_app_fixture(app="../../cards/VarDissimilar.py", scope="function")
_HELPER_CARDS = {}


@pytest.fixture(scope="session")
def browser_context_args():
    return {"viewport": {"width": 1600, "height": 1000}}


@pytest.fixture
def card_module():
    return importlib.import_module("cards.VarDissimilar")


class FakeInputs:
    def __init__(
        self,
        *,
        robust=True,
        qgram=2,
        technique="Agglomerative",
        style="radial",
        max_obs=4,
    ):
        self.robust = robust
        self.qgram = qgram
        self.technique = technique
        self.style = style
        self.max_obs = max_obs

    def Robust(self):
        return self.robust

    def Qgram(self):
        return self.qgram

    def Which(self):
        return self.technique

    def Style(self):
        return self.style

    def MaxObs(self):
        return self.max_obs


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "height": [150.0, 160.0, 170.0, np.nan, 190.0, 200.0],
        "height_copy": [300.0, 320.0, 340.0, np.nan, 380.0, 400.0],
        "group": ["A", "A", "B", "B", None, "C"],
        "code": ["a1", "a2", "b1", "b2", "c1", "c2"],
    })


def recorded_helpers(card_module, *, frame=None, inputs=None, role_map=None, fullscreen=False):
    """Expose helpers declared inside the card server using inert decorators."""
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
    proxy = source if isinstance(source, ProxyData) else ProxyData(
        _df=source,
        _roles=role_map or RoleMap(),
        _name="Test",
    )
    with reactive.isolate():
        card._imports.set(proxy)
    card.server(inputs or FakeInputs(), capture, SimpleNamespace())
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
        assert card.name == "VarDissimilar"
        assert card.long_name == "Variable Dissimilarity"
        assert "dissimilarity matrix" in card.description
        assert not card.mutable
        assert card.hasSidebar()
        assert card.hasFlipSide()
        assert not card.hasFooter()

    @pytest.mark.unit
    def test_front_and_back_have_expected_outputs(self, card_module):
        front = str(card_module.this.front.tagify())
        back = str(card_module.this.back.tagify())
        assert 'id="Chart"' in front
        assert "Variable Dissimilarity chart" in front
        assert 'id="Table"' in back
        assert "Variable Dissimilarity table" in back

    @pytest.mark.unit
    def test_settings_have_all_controls(self, card_module):
        html = str(card_module.this.settings)
        for control in ("Robust", "Qgram", "Which", "Style", "MaxObs"):
            assert f'id="{control}"' in html
        for choice in ("Agglomerative", "Divisive", "Rectangular", "Radial"):
            assert choice in html

    @pytest.mark.unit
    def test_test_mode_seeds_expected_frame(self, card_module):
        with reactive.isolate():
            proxy = card_module.this._imports.get()
        assert proxy.to_native().shape == (4, 5)
        assert proxy.to_native().columns.tolist() == ["y", "x1", "x2", "id", "part"]


class TestPreparationAndStatistics:
    @pytest.mark.unit
    def test_max_observations_is_logarithmic(self, card_module):
        _, functions = recorded_helpers(card_module, inputs=FakeInputs(max_obs=5))
        assert functions["MaxObs"]() == 100_000

    @pytest.mark.unit
    def test_qgram_is_at_least_one(self, card_module):
        _, functions = recorded_helpers(card_module, inputs=FakeInputs(qgram=0))
        assert functions["Qgram"]() == 1

    @pytest.mark.unit
    def test_prepared_data_samples_and_keeps_only_predictors(self, card_module):
        frame = pd.DataFrame({
            "predictor": range(1500),
            "target": range(1500),
            "shadow__predictor": range(1500),
        })
        roles = RoleMap()
        roles.set_roles("predictor", [Role.PREDICTOR])
        roles.set_roles("target", [Role.TARGET])
        roles.set_roles("shadow__predictor", [Role.PREDICTOR])
        _, functions = recorded_helpers(
            card_module,
            frame=frame,
            role_map=roles,
            inputs=FakeInputs(max_obs=3),
        )
        with reactive.isolate():
            result = functions["CleanDf"]()
        assert result.shape == (1000, 1)
        assert result.columns.tolist() == ["predictor"]

    @pytest.mark.unit
    def test_safe_scale_preserves_nan_and_scales_largest_magnitude(self, card_module):
        _, functions = recorded_helpers(card_module)
        result = functions["_safe_scale"](np.array([-2.0, 1.0, np.nan]))
        np.testing.assert_allclose(result[:2], [-1.0, 0.5])
        assert np.isnan(result[2])

    @pytest.mark.unit
    def test_robust_statistics_reduce_outlier_influence(self, card_module):
        frame = pd.DataFrame({"value": [1.0, 2.0, 100.0]})
        _, functions = recorded_helpers(card_module, frame=frame)
        robust = functions["_numeric_stats"](frame, robust=True)
        ordinary = functions["_numeric_stats"](frame, robust=False)
        assert robust.loc["value", "Cardinality"] == 1.0
        assert robust.loc["value", "Centrality"] == 1.0
        assert robust.loc["value", "Spread"] < ordinary.loc["value", "Spread"]

    @pytest.mark.unit
    def test_categorical_spread_is_one_minus_modal_proportion(self, card_module):
        frame = pd.DataFrame({"group": ["A", "A", "A", "B"]})
        _, functions = recorded_helpers(card_module, frame=frame)
        result = functions["_numeric_stats"](frame, robust=True)
        assert result.loc["group", "Spread"] == pytest.approx(0.25)
        assert np.isnan(result.loc["group", "Centrality"])


class TestComponentDistances:
    @pytest.fixture
    def functions(self, card_module):
        return recorded_helpers(card_module)[1]

    @pytest.mark.unit
    def test_qgrams_are_case_insensitive_and_count_repetitions(self, functions):
        assert functions["_qgrams"]("ABaba", 2) == {
            "ab": 2,
            "ba": 2,
        }

    @pytest.mark.unit
    def test_string_cosine_is_symmetric_with_zero_diagonal(self, functions):
        result = functions["_string_cosine"](["height", "height2", "group"], 2)
        np.testing.assert_allclose(result, result.T)
        np.testing.assert_allclose(np.diag(result), 0)
        assert result[0, 1] < result[0, 2]

    @pytest.mark.unit
    def test_cosine_rows_uses_only_common_finite_features(self, functions):
        values = np.array([[1.0, np.nan, 0.0], [1.0, 4.0, 0.0]])
        result = functions["_cosine_rows"](values)
        assert result[0, 1] == pytest.approx(0.0)

    @pytest.mark.unit
    def test_robust_value_distance_uses_spearman(self, card_module):
        frame = pd.DataFrame({"x": [1, 2, 3, 4], "y": [1, 4, 9, 16]})
        _, robust = recorded_helpers(card_module, frame=frame, inputs=FakeInputs(robust=True))
        _, ordinary = recorded_helpers(card_module, frame=frame, inputs=FakeInputs(robust=False))
        robust_distance = robust["_value_correlation_distance"](frame)
        ordinary_distance = ordinary["_value_correlation_distance"](frame)
        assert robust_distance[0, 1] == pytest.approx(0.0)
        assert ordinary_distance[0, 1] > 0

    @pytest.mark.unit
    def test_missingness_distance_ignores_shared_observed_rows(self, functions):
        frame = pd.DataFrame({
            "same_a": [np.nan, 1, np.nan, 1],
            "same_b": [np.nan, 2, np.nan, 2],
            "opposite": [1, np.nan, 1, np.nan],
            "complete_a": [1, 2, 3, 4],
            "complete_b": [5, 6, 7, 8],
        })
        result = functions["_missingness_distance"](frame)
        assert result[0, 1] == pytest.approx(0.0)
        assert result[0, 2] == pytest.approx(1.0)
        assert result[0, 3] == pytest.approx(1.0)
        assert np.isnan(result[3, 4])
        np.testing.assert_allclose(np.diag(result), 0)

    @pytest.mark.unit
    @pytest.mark.parametrize("value", [True, 1.5, "2"])
    def test_missingness_distance_rejects_noninteger_min_events(self, functions, value):
        with pytest.raises(TypeError, match="min_events must be an integer"):
            functions["_missingness_distance"](sample_frame(), min_events=value)

    @pytest.mark.unit
    def test_missingness_distance_rejects_nonpositive_min_events(self, functions):
        with pytest.raises(ValueError, match="at least 1"):
            functions["_missingness_distance"](sample_frame(), min_events=0)

    @pytest.mark.unit
    def test_min_events_marks_sparse_comparison_unavailable(self, functions):
        frame = pd.DataFrame({"a": [np.nan, 1, 2], "b": [np.nan, 3, 4]})
        result = functions["_missingness_distance"](frame, min_events=2)
        assert np.isnan(result[0, 1])


class TestCombinedDistanceAndHierarchy:
    @pytest.mark.unit
    def test_dissimilarity_matrix_is_labelled_symmetric_and_bounded(self, card_module):
        _, functions = recorded_helpers(card_module)
        with reactive.isolate():
            result = functions["DissimilarityMatrix"]()
        assert result.columns.tolist() == sample_frame().columns.tolist()
        assert result.index.tolist() == sample_frame().columns.tolist()
        np.testing.assert_allclose(result, result.T)
        np.testing.assert_allclose(np.diag(result), 0)
        assert result.to_numpy().min() >= 0
        assert result.to_numpy().max() <= 2
        assert result.loc["height", "height_copy"] < result.loc["height", "group"]

    @pytest.mark.unit
    def test_empty_predictor_set_returns_empty_matrix(self, card_module):
        frame = pd.DataFrame({"target": [1, 2, 3]})
        roles = RoleMap()
        roles.set_roles("target", [Role.TARGET])
        _, functions = recorded_helpers(card_module, frame=frame, role_map=roles)
        with reactive.isolate():
            result = functions["DissimilarityMatrix"]()
        assert result.empty

    @pytest.mark.unit
    @pytest.mark.parametrize("technique", ["Agglomerative", "Divisive"])
    def test_hierarchy_has_valid_linkage_shape(self, card_module, technique):
        _, functions = recorded_helpers(card_module)
        matrix = pd.DataFrame(
            [[0.0, 0.2, 0.8], [0.2, 0.0, 0.7], [0.8, 0.7, 0.0]],
            index=["a", "b", "c"],
            columns=["a", "b", "c"],
        )
        result = functions["_hierarchy"](matrix, technique)
        assert result.shape == (2, 4)
        assert (result[:, 2] >= 0).all()
        assert result[-1, 3] == 3

    @pytest.mark.unit
    def test_rectangular_and_radial_figures_use_expected_trace_types(self, card_module):
        _, functions = recorded_helpers(card_module)
        matrix = pd.DataFrame(
            [[0.0, 0.2, 0.8], [0.2, 0.0, 0.7], [0.8, 0.7, 0.0]],
            index=["a", "b", "c"],
            columns=["a", "b", "c"],
        )
        rectangular = functions["_hierarchy_figure"](
            matrix, technique="Agglomerative", style="rectangular"
        )
        radial = functions["_hierarchy_figure"](
            matrix, technique="Divisive", style="radial"
        )
        assert {trace.type for trace in rectangular.data} == {"scatter"}
        assert {trace.type for trace in radial.data} == {"scatterpolar"}

    @pytest.mark.unit
    def test_empty_and_single_variable_figures_are_informative(self, card_module):
        _, functions = recorded_helpers(card_module)
        empty = functions["_hierarchy_figure"](
            pd.DataFrame(), technique="Agglomerative", style="rectangular"
        )
        single = functions["_hierarchy_figure"](
            pd.DataFrame([[0.0]], index=["only"], columns=["only"]),
            technique="Agglomerative",
            style="rectangular",
        )
        assert empty.layout.annotations[0].text == "No variables to compare"
        assert single.data[0].text[0] == "only"


class TestWebKitUI:
    @pytest.mark.ui
    def test_card_chart_and_settings_render(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        expect(get_card(page)).to_be_visible()
        expect(page.get_by_text("Variable Dissimilarity", exact=True)).to_be_visible()
        expect(by_id(page, "Chart")).to_be_visible()
        for control in ("Robust", "Qgram", "Which", "Style", "MaxObs"):
            expect(by_id(page, control)).to_be_attached()

    @pytest.mark.ui
    def test_switching_to_rectangular_agglomerative_chart_keeps_widget_visible(
        self, page: Page, app: ShinyAppProc
    ):
        page.goto(app.url)
        set_shiny_input(page, "Style", "rectangular")
        set_shiny_input(page, "Which", "Agglomerative")
        expect(by_id(page, "Chart")).to_be_visible()
        expect(by_id(page, "Chart").locator(".plotly")).to_be_attached(timeout=10_000)

    @pytest.mark.ui
    def test_switching_to_divisive_radial_chart_keeps_widget_visible(
        self, page: Page, app: ShinyAppProc
    ):
        page.goto(app.url)
        set_shiny_input(page, "Style", "radial")
        set_shiny_input(page, "Which", "Divisive")
        expect(by_id(page, "Chart")).to_be_visible()
        expect(by_id(page, "Chart").locator(".plotly")).to_be_attached(timeout=10_000)

    @pytest.mark.ui
    def test_flip_displays_labelled_distance_table(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        by_id(page, "FlipButton").click(force=True)
        expect(by_id(page, "Table")).to_be_visible()
        expect(by_id(page, "Table2")).to_be_visible(timeout=10_000)
        table = by_id(page, "Table2")
        expect(table).to_contain_text("Variable")
        expect(table).to_contain_text("x1")
        expect(table).to_contain_text("x2")

    @pytest.mark.ui
    def test_fullscreen_toggle_keeps_chart_available(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        by_id(page, "ExpandButton").click(force=True)
        expect(get_card(page)).to_be_visible()
        expect(by_id(page, "Chart")).to_be_visible()
