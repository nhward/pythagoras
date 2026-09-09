from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from playwright.sync_api import Page, expect
from proxy_data import proxy_data
from roles import Role, RoleMap
from shiny import reactive
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

path = Path(__file__).resolve().parents[2] / "app"
os.chdir(path)
if str(path) not in sys.path:
    sys.path.insert(0, str(path))

app = create_app_fixture(app="../scenarios/var_correlation.py", scope="function")


@pytest.fixture(scope="session")
def browser_context_args():
    return {"viewport": {"width": 1600, "height": 1000}}


@pytest.fixture(scope="module")
def card_module():
    return importlib.import_module("cards.var_correlation")


def correlation_data(rows: int = 120) -> proxy_data:
    rng = np.random.default_rng(1729)
    x = np.linspace(-2, 2, rows)
    frame = pd.DataFrame({
        "x": x,
        "positive": x + rng.normal(scale=0.05, size=rows),
        "negative": -x + rng.normal(scale=0.05, size=rows),
        "square": x**2 + rng.normal(scale=0.02, size=rows),
        "noise": rng.normal(size=rows),
        "treatment": np.resize([0.0, 1.0], rows),
        "target": 3 * x + rng.normal(scale=0.1, size=rows),
        "weight": np.linspace(0.5, 1.5, rows),
        "constant": np.ones(rows),
        "flag": np.resize([True, False], rows),
        "shadow__x": x,
        "label": np.resize(["A", "B"], rows),
    })
    frame.loc[::13, "positive"] = np.nan
    roles = RoleMap()
    for column in ("x", "positive", "negative", "square", "noise"):
        roles.set_roles(column, [Role.PREDICTOR])
    roles.set_roles("treatment", [Role.TREATMENT])
    roles.set_roles("target", [Role.TARGET])
    roles.set_roles("weight", [Role.WEIGHTING])
    for column in ("constant", "flag", "shadow__x", "label"):
        roles.set_roles(column, [Role.PREDICTOR])
    return proxy_data(
        _df=frame,
        _roles=roles,
        _name="Variable correlation unit test",
    )


@pytest.fixture
def card(card_module):
    card = card_module.instance()
    with reactive.isolate():
        card._imports.set(correlation_data())
    return card


def analyze(card_module, method: str, *, include_target: bool = False):
    return card_module._analyse_correlation(
        correlation_data(),
        method=method,
        include_target=include_target,
        maximum_observations=1_000,
    )


def signed_analysis(card_module):
    matrix = pd.DataFrame(
        [[1.0, -0.8, 0.5], [-0.8, 1.0, -0.3], [0.5, -0.3, 1.0]],
        index=["a", "b", "c"],
        columns=["a", "b", "c"],
    )
    return card_module.CorrelationAnalysis(
        method=card_module.METHODS["pearson"],
        matrix=matrix,
        source_observations=20,
        analyzed_observations=20,
        sampled=False,
        weighted=False,
    )


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


class TestCardDefinition:
    @pytest.mark.unit
    def test_metadata_regions_and_immutability(self, card):
        assert card.name == "var_correlation"
        assert card.long_name == "Variable correlation"
        assert "directional measures" in card.description
        assert card.mutable is False
        assert card.hasSidebar()
        assert card.hasFlipSide()
        assert card.hasFooter()

    @pytest.mark.unit
    def test_front_has_all_methods_and_back_has_values_table(self, card):
        front = str(card.front.tagify())
        back = str(card.back.tagify())
        for method in (
            "pearson", "spearman", "kendall", "mutual_information",
            "distance_correlation", "pps", "latent", "xi",
        ):
            assert f'id="Chart_{method}"' in front
        assert 'id="Table"' in back
        assert 'id="TableTitle"' in back

    @pytest.mark.unit
    def test_settings_have_expected_controls(self, card):
        settings = str(card.settings)
        for control in (
            "Style", "Threshold", "IncludeTarget", "Absolute", "Ordering",
            "MaxObs",
        ):
            assert f'id="{control}"' in settings
        assert "Show absolute values" in settings


class TestSelectionAndAnalysis:
    @pytest.mark.unit
    def test_eligible_columns_follow_roles_and_exclusions(self, card_module):
        data = correlation_data()
        without_target = card_module._eligible_columns(data, False)
        with_target = card_module._eligible_columns(data, True)

        assert without_target == [
            "x", "positive", "negative", "square", "noise", "treatment",
        ]
        assert with_target == without_target + ["target"]
        for excluded in (
            "weight", "constant", "flag", "shadow__x", "label",
        ):
            assert excluded not in with_target

    @pytest.mark.unit
    def test_sampling_is_deterministic_and_method_caps_are_respected(
        self, card_module
    ):
        rows = 2_500
        frame = pd.DataFrame({"x": np.arange(rows), "y": np.arange(rows)})
        data = proxy_data.from_native(frame)
        columns = ["x", "y"]

        first, _, sampled = card_module._analysis_frame(
            data, columns, "distance_correlation", 10_000,
        )
        second, _, _ = card_module._analysis_frame(
            data, columns, "distance_correlation", 10_000,
        )

        assert sampled is True
        assert len(first) == card_module.DISTANCE_CORRELATION_LIMIT
        assert first.index.tolist() == second.index.tolist()
        assert first.index.is_monotonic_increasing
        assert card_module._sample_limit("pps", 50_000) == card_module.PPS_LIMIT

    @pytest.mark.unit
    @pytest.mark.parametrize("method", ["pearson", "spearman", "kendall"])
    def test_standard_correlations_are_labelled_signed_and_symmetric(
        self, card_module, method
    ):
        result = analyze(card_module, method)

        assert result.method.signed
        assert not result.method.directional
        assert result.matrix.index.tolist() == result.matrix.columns.tolist()
        np.testing.assert_allclose(result.matrix, result.matrix.T, equal_nan=True)
        assert result.matrix.loc["x", "positive"] > 0.95
        assert result.matrix.loc["x", "negative"] < -0.95

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("method", "associated"),
        [
            ("mutual_information", "square"),
            ("distance_correlation", "square"),
            ("latent", "positive"),
        ],
    )
    def test_nonlinear_matrices_are_symmetric_and_bounded(
        self, card_module, method, associated
    ):
        result = analyze(card_module, method)
        values = result.matrix.to_numpy(dtype=float)

        np.testing.assert_allclose(values, values.T, equal_nan=True)
        assert np.nanmin(values) >= (-1 if result.method.signed else 0)
        assert np.nanmax(values) <= 1
        assert (
            abs(result.matrix.loc["x", associated])
            > abs(result.matrix.loc["x", "noise"])
        )

    @pytest.mark.unit
    @pytest.mark.parametrize("method", ["pps", "xi"])
    def test_directional_methods_preserve_both_directions(
        self, card_module, method
    ):
        result = analyze(card_module, method)

        assert result.method.directional
        assert result.matrix.loc["x", "square"] > 0.75
        assert result.matrix.loc["square", "x"] < 0.25
        assert not np.allclose(result.matrix, result.matrix.T, equal_nan=True)

    @pytest.mark.unit
    def test_weighting_and_target_are_reported_without_modifying_source(
        self, card_module
    ):
        data = correlation_data()
        before = data.frame.copy(deep=True)

        weighted = card_module._analyse_correlation(
            data,
            method="pearson",
            include_target=True,
            maximum_observations=1_000,
        )
        unweighted = card_module._analyse_correlation(
            data,
            method="kendall",
            include_target=False,
            maximum_observations=1_000,
        )

        assert weighted.weighted is True
        assert "target" in weighted.matrix.columns
        assert unweighted.weighted is False
        assert "target" not in unweighted.matrix.columns
        assert_frame_equal(data.frame, before)


class TestTablesAndFigures:
    @pytest.mark.unit
    def test_values_table_uses_unordered_or_ordered_pairs(self, card_module):
        symmetric = signed_analysis(card_module)
        symmetric_table = card_module._values_table(symmetric, 0.0)
        assert len(symmetric_table) == 3
        assert set(zip(symmetric_table["Source"], symmetric_table["Destination"])) == {
            ("a", "b"), ("a", "c"), ("b", "c"),
        }

        directional = card_module.CorrelationAnalysis(
            method=card_module.METHODS["pps"],
            matrix=symmetric.matrix.abs(),
            source_observations=20,
            analyzed_observations=20,
            sampled=False,
            weighted=False,
        )
        directional_table = card_module._values_table(directional, 0.0)
        pairs = set(zip(
            directional_table["Source"], directional_table["Destination"]
        ))
        assert len(directional_table) == 6
        assert ("a", "b") in pairs and ("b", "a") in pairs

    @pytest.mark.unit
    def test_display_matrix_applies_threshold_and_absolute_values(self, card_module):
        analysis = signed_analysis(card_module)
        signed = card_module._display_matrix(
            analysis, threshold=0.4, absolute=False, ordering="original"
        )
        absolute = card_module._display_matrix(
            analysis, threshold=0.4, absolute=True, ordering="original"
        )

        assert signed.loc["a", "b"] == pytest.approx(-0.8)
        assert signed.loc["b", "c"] == pytest.approx(0.0)
        assert absolute.loc["a", "b"] == pytest.approx(0.8)
        assert absolute.to_numpy().min() >= 0

    @pytest.mark.unit
    @pytest.mark.parametrize("absolute", [False, True])
    def test_heatmap_colourbar_is_fullscreen_only(
        self, card_module, absolute
    ):
        analysis = signed_analysis(card_module)
        compact = card_module._heatmap_figure(
            analysis,
            threshold=0,
            absolute=absolute,
            ordering="original",
            full_screen=False,
        )
        expanded = card_module._heatmap_figure(
            analysis,
            threshold=0,
            absolute=absolute,
            ordering="original",
            full_screen=True,
        )

        assert compact.data[0].showscale is False
        assert expanded.data[0].showscale is True
        expected_title = "Absolute value" if absolute else "Value"
        assert expanded.data[0].colorbar.title.text == expected_title

    @pytest.mark.unit
    def test_chord_absolute_values_and_fullscreen_legend_contract(
        self, card_module
    ):
        analysis = signed_analysis(card_module)
        compact = card_module._chord_figure(
            analysis,
            threshold=0,
            absolute=False,
            ordering="original",
            full_screen=False,
        )
        signed = card_module._chord_figure(
            analysis,
            threshold=0,
            absolute=False,
            ordering="original",
            full_screen=True,
        )
        absolute = card_module._chord_figure(
            analysis,
            threshold=0,
            absolute=True,
            ordering="original",
            full_screen=True,
        )

        assert compact.layout.showlegend is False
        assert [trace.name for trace in signed.data if trace.showlegend] == [
            "Positive association", "Negative association",
        ]
        assert absolute.layout.showlegend is False
        assert not any(trace.showlegend for trace in absolute.data)
        signed_ribbons = signed.layout.shapes[len(analysis.matrix):]
        absolute_ribbons = absolute.layout.shapes[len(analysis.matrix):]
        assert any("214,39,40" in shape.fillcolor for shape in signed_ribbons)
        assert not any("214,39,40" in shape.fillcolor for shape in absolute_ribbons)
        assert any(": -" in text for text in signed.data[0].text)
        assert not any(": -" in text for text in absolute.data[0].text)


class TestWebKitUI:
    @pytest.mark.ui
    def test_card_chart_tabs_settings_and_summary_render(
        self, page: Page, app: ShinyAppProc
    ):
        page.goto(app.url)

        plot = by_id(page, "Chart_pearson").locator(".js-plotly-plot")
        expect(plot).to_be_attached(timeout=20_000)
        expect(by_id(page, "Check")).to_contain_text(
            "5 variables", timeout=20_000,
        )
        expect(by_id(page, "Check")).to_contain_text(
            "observation weights applied"
        )
        for control in (
            "Style", "Threshold", "IncludeTarget", "Absolute", "Ordering",
            "MaxObs",
        ):
            expect(by_id(page, control)).to_be_attached()
        for label in (
            "Pearson", "Spearman", "Kendall", "Mutual info", "Distance",
            "PPS", "Latent", "Xi",
        ):
            expect(page.get_by_text(label, exact=True)).to_be_attached()

    @pytest.mark.ui
    def test_fullscreen_chord_legend_tracks_absolute_setting(
        self, page: Page, app: ShinyAppProc
    ):
        page.goto(app.url)
        plot = by_id(page, "Chart_pearson").locator(".js-plotly-plot")
        expect(plot).to_be_attached(timeout=20_000)

        by_id(page, "ExpandButton").click(force=True)
        page.wait_for_function(
            """
            selector => {
                const plot = document.querySelector(selector);
                return plot?.layout?.showlegend === true
                    && plot.data.filter(trace => trace.showlegend).length === 2;
            }
            """,
            arg=f"#{namespaced_id(page, 'Chart_pearson')} .js-plotly-plot",
            timeout=20_000,
        )
        assert plot.evaluate(
            "element => element.data.filter(trace => trace.showlegend).map(trace => trace.name)"
        ) == ["Positive association", "Negative association"]

        set_shiny_input(page, "Absolute", True)
        page.wait_for_function(
            """
            selector => {
                const plot = document.querySelector(selector);
                return plot?.layout?.showlegend === false
                    && plot.data.every(trace => !trace.showlegend);
            }
            """,
            arg=f"#{namespaced_id(page, 'Chart_pearson')} .js-plotly-plot",
            timeout=20_000,
        )

    @pytest.mark.ui
    def test_heatmap_colourbar_is_visible_only_in_fullscreen(
        self, page: Page, app: ShinyAppProc
    ):
        page.goto(app.url)
        set_shiny_input(page, "Style", "heatmap")
        plot = by_id(page, "Chart_pearson").locator(".js-plotly-plot")
        expect(plot).to_be_attached(timeout=20_000)
        page.wait_for_function(
            """
            selector => {
                const plot = document.querySelector(selector);
                return plot?.data?.[0]?.type === "heatmap"
                    && plot.data[0].showscale === false;
            }
            """,
            arg=f"#{namespaced_id(page, 'Chart_pearson')} .js-plotly-plot",
            timeout=20_000,
        )

        by_id(page, "ExpandButton").click(force=True)
        page.wait_for_function(
            """
            selector => document.querySelector(selector)?.data?.[0]?.showscale === true
            """,
            arg=f"#{namespaced_id(page, 'Chart_pearson')} .js-plotly-plot",
            timeout=20_000,
        )

    @pytest.mark.ui
    def test_directional_method_flip_displays_values_table(
        self, page: Page, app: ShinyAppProc
    ):
        page.goto(app.url)
        page.get_by_role("tab", name="Xi", exact=True).click()
        expect(by_id(page, "Check")).to_contain_text(
            "directional matrix", timeout=20_000,
        )
        expect(by_id(page, "Chart_xi").locator(".js-plotly-plot")).to_be_attached(
            timeout=20_000,
        )

        set_shiny_input(page, "Threshold", 0)
        by_id(page, "FlipButton").click(force=True)
        expect(by_id(page, "TableTitle")).to_contain_text(
            "directional values", timeout=20_000,
        )
        table = by_id(page, "Table2")
        expect(table).to_be_visible(timeout=20_000)
        for heading in ("Source", "Destination", "Value"):
            expect(table).to_contain_text(heading)
        expect(table).to_contain_text("square")
