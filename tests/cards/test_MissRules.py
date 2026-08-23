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

app = create_app_fixture(app="../../cards/MissRules.py", scope="function")
_HELPER_CARDS = {}


@pytest.fixture(scope="session")
def browser_context_args():
    return {"viewport": {"width": 1600, "height": 1000}}


@pytest.fixture
def card_module():
    return importlib.import_module("cards.MissRules")


class FakeInputs:
    def __init__(
        self,
        *,
        min_support=0.1,
        min_lift=1.5,
        max_length=10,
        remove_redundant=False,
        max_obs=4,
    ):
        self.min_support = min_support
        self.min_lift = min_lift
        self.max_length = max_length
        self.remove_redundant = remove_redundant
        self.max_obs = max_obs

    def MinSupport(self):
        return self.min_support

    def MinLift(self):
        return self.min_lift

    def MaxLength(self):
        return self.max_length

    def RemoveRedundant(self):
        return self.remove_redundant

    def MaxObs(self):
        return self.max_obs


def rule_frame() -> pd.DataFrame:
    """Missing A and B coincide; C is missing on a disjoint set of rows."""
    return pd.DataFrame({
        "A": [np.nan] * 4 + list(range(4, 12)),
        "B": [np.nan] * 4 + list(range(104, 112)),
        "C": list(range(4)) + [np.nan] * 4 + list(range(8, 12)),
        "complete": range(12),
    })


def recorded_helpers(card_module, *, frame=None, inputs=None, fullscreen=False):
    """Expose nested server helpers using inert test decorators."""
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
    source = rule_frame() if frame is None else frame
    proxy = source if isinstance(source, ProxyData) else ProxyData(
        _df=source,
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
        assert card.name == "MissRules"
        assert card.long_name == "Missingness rules"
        assert "Association Rules" in card.description
        assert not card.mutable
        assert card.hasSidebar()
        assert card.hasFlipSide()
        assert card.hasFooter()

    @pytest.mark.unit
    def test_front_back_and_footer_outputs(self, card_module):
        front = str(card_module.this.front.tagify())
        back = str(card_module.this.back.tagify())
        footer = str(card_module.this.footer.tagify())
        assert 'id="Network"' in front
        assert "Missingness association network" in front
        assert 'id="Table"' in back
        assert "Missingness association rules" in back
        assert 'id="Check"' in footer

    @pytest.mark.unit
    def test_settings_contain_current_controls(self, card_module):
        html = str(card_module.this.settings)
        for control in (
            "MinSupport", "MinLift", "MaxLength", "RemoveRedundant", "MaxObs"
        ):
            assert f'id="{control}"' in html
        assert "Minimum permitted rule lift" in html
        assert "10^" in html

    @pytest.mark.unit
    def test_test_mode_seeds_missing_values(self, card_module):
        with reactive.isolate():
            proxy = card_module.this._imports.get()
        frame = proxy.to_native()
        assert frame.shape == (4, 5)
        assert frame.columns.tolist() == ["y", "x1", "x2", "id", "part"]
        assert frame[["x1", "x2", "id"]].isna().any().all()


class TestPreparation:
    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("slider", "expected"),
        [(3, 1_000), (4, 10_000), (5, 100_000)],
    )
    def test_max_observations_is_logarithmic(
        self, card_module, slider, expected
    ):
        _, functions = recorded_helpers(
            card_module,
            inputs=FakeInputs(max_obs=slider),
        )
        assert functions["MaxObs"]() == expected

    @pytest.mark.unit
    def test_threshold_inputs_are_normalised(self, card_module):
        inputs = FakeInputs(min_support=0.25, min_lift=2.4, max_length=1)
        _, functions = recorded_helpers(card_module, inputs=inputs)
        assert functions["MinSupport"]() == pytest.approx(0.25)
        assert functions["MinLift"]() == pytest.approx(2.4)
        assert functions["MaxLength"]() == 2

    @pytest.mark.unit
    def test_prepared_data_returns_proxy_and_respects_sample_limit(self, card_module):
        frame = pd.DataFrame({"value": range(1500)})
        _, functions = recorded_helpers(
            card_module,
            frame=frame,
            inputs=FakeInputs(max_obs=3),
        )
        with reactive.isolate():
            result = functions["PreparedData"]()
        assert isinstance(result, ProxyData)
        assert result.shape == (1000, 1)
        assert result.to_native().index.is_monotonic_increasing

    @pytest.mark.unit
    def test_missing_variables_returns_only_columns_with_na(self, card_module):
        _, functions = recorded_helpers(card_module)
        with reactive.isolate():
            assert functions["MissingVariables"]() == ["A", "B", "C"]

    @pytest.mark.unit
    def test_transactions_drop_complete_rows_and_observed_columns(self, card_module):
        _, functions = recorded_helpers(card_module)
        with reactive.isolate():
            result = functions["MissingTransactions"]()
        assert result.shape == (8, 3)
        assert result.columns.tolist() == ["A", "B", "C"]
        assert result.dtypes.eq(bool).all()
        assert result.any(axis=1).all()
        assert result[["A", "B"]].iloc[:4].all(axis=None)
        assert result["C"].iloc[4:].all()

    @pytest.mark.unit
    def test_no_missing_values_produce_empty_transactions(self, card_module):
        frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        _, functions = recorded_helpers(card_module, frame=frame)
        with reactive.isolate():
            result = functions["MissingTransactions"]()
        assert result.empty
        assert result.index.equals(frame.index)


class TestRules:
    @pytest.mark.unit
    def test_rules_use_lift_threshold_and_are_sorted(self, card_module):
        _, functions = recorded_helpers(
            card_module,
            inputs=FakeInputs(min_lift=1.5),
        )
        with reactive.isolate():
            rules = functions["Rules"]()
        assert len(rules) == 2
        assert rules["lift"].ge(1.5).all()
        assert rules["lift"].is_monotonic_decreasing
        assert set(rules["antecedents"]) == {frozenset({"A"}), frozenset({"B"})}
        assert set(rules["consequents"]) == {frozenset({"A"}), frozenset({"B"})}
        assert rules["support"].eq(0.5).all()
        assert rules["confidence"].eq(1.0).all()
        assert rules["lift"].eq(2.0).all()

    @pytest.mark.unit
    def test_higher_lift_threshold_removes_rules(self, card_module):
        _, functions = recorded_helpers(
            card_module,
            inputs=FakeInputs(min_lift=2.1),
        )
        with reactive.isolate():
            assert functions["Rules"]().empty

    @pytest.mark.unit
    def test_one_missing_variable_cannot_form_rule(self, card_module):
        frame = pd.DataFrame({"a": [np.nan, 1], "complete": [1, 2]})
        _, functions = recorded_helpers(card_module, frame=frame)
        with reactive.isolate():
            rules = functions["Rules"]()
        assert rules.empty
        assert "antecedents" in rules.columns
        assert "lift" in rules.columns

    @pytest.mark.unit
    def test_maximum_length_limits_generated_rules(self, card_module):
        frame = pd.DataFrame({
            "A": [np.nan, np.nan, 1, 1],
            "B": [np.nan, np.nan, 1, 1],
            "C": [np.nan, np.nan, 1, 1],
            "D": [1, 1, np.nan, np.nan],
        })
        _, functions = recorded_helpers(
            card_module,
            frame=frame,
            inputs=FakeInputs(min_lift=0.1, max_length=2),
        )
        with reactive.isolate():
            rules = functions["Rules"]()
        lengths = rules["antecedents"].map(len) + rules["consequents"].map(len)
        assert not rules.empty
        assert lengths.max() <= 2

    @pytest.mark.unit
    def test_redundant_specialised_rule_is_pruned(self, card_module):
        _, functions = recorded_helpers(card_module)
        rules = pd.DataFrame({
            "antecedents": [
                frozenset({"A"}),
                frozenset({"A", "B"}),
                frozenset({"B"}),
            ],
            "consequents": [
                frozenset({"C"}),
                frozenset({"C"}),
                frozenset({"A"}),
            ],
            "support": [0.5, 0.4, 0.4],
            "confidence": [0.9, 0.9, 0.8],
        })
        result = functions["_remove_redundant"](rules)
        assert len(result) == 2
        assert frozenset({"A", "B"}) not in set(result["antecedents"])
        assert frozenset({"B"}) in set(result["antecedents"])

    @pytest.mark.unit
    def test_itemset_labels_are_sorted_case_insensitively(self, card_module):
        _, functions = recorded_helpers(card_module)
        assert functions["_itemset_label"]({"beta", "Alpha"}) == "Alpha, beta"

    @pytest.mark.unit
    def test_rules_table_has_readable_labels_metrics_and_counts(self, card_module):
        _, functions = recorded_helpers(card_module)
        with reactive.isolate():
            table = functions["RulesTable"]()
        assert table.columns.tolist() == [
            "LHS", "RHS", "Support", "Confidence", "Lift",
            "Leverage", "Conviction", "Count",
        ]
        assert len(table) == 2
        assert set(table["LHS"]) == {"A", "B"}
        assert set(table["RHS"]) == {"A", "B"}
        assert table["Count"].eq(4).all()
        assert table["Lift"].eq(2.0).all()

    @pytest.mark.unit
    def test_empty_rules_table_preserves_schema(self, card_module):
        frame = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        _, functions = recorded_helpers(card_module, frame=frame)
        with reactive.isolate():
            table = functions["RulesTable"]()
        assert table.empty
        assert table.columns.tolist() == [
            "LHS", "RHS", "Support", "Confidence", "Lift",
            "Leverage", "Conviction", "Count",
        ]


class TestNetworkFigure:
    @pytest.mark.unit
    def test_empty_rules_figure_has_explanation(self, card_module):
        _, functions = recorded_helpers(card_module)
        figure = functions["_network_figure"](pd.DataFrame())
        assert len(figure.data) == 0
        assert figure.layout.annotations[0].text == "No significant rules to display"
        assert not figure.layout.xaxis.visible
        assert not figure.layout.yaxis.visible

    @pytest.mark.unit
    def test_network_contains_edges_variables_and_rules(self, card_module):
        _, functions = recorded_helpers(card_module)
        with reactive.isolate():
            rules = functions["Rules"]()
        figure = functions["_network_figure"](rules)
        assert [trace.name for trace in figure.data] == [None, "Variables", "Rules"]
        assert figure.data[0].mode == "lines"
        assert figure.data[1].marker.symbol == "square"
        assert figure.data[2].marker.symbol == "circle"
        assert set(figure.data[1].text) == {"A", "B"}
        assert len(figure.data[2].x) == 2
        assert "Lift: 2.000" in figure.data[2].text[0]

    @pytest.mark.unit
    def test_network_limit_selects_requested_number_of_rules(self, card_module):
        _, functions = recorded_helpers(card_module)
        rules = pd.DataFrame({
            "antecedents": [frozenset({"A"})] * 5,
            "consequents": [frozenset({f"B{i}"}) for i in range(5)],
            "support": np.linspace(0.5, 0.1, 5),
            "confidence": np.linspace(1.0, 0.6, 5),
            "lift": np.linspace(3.0, 1.0, 5),
        })
        figure = functions["_network_figure"](rules, limit=3)
        assert len(figure.data[2].x) == 3


class TestWebKitUI:
    @pytest.mark.ui
    def test_card_outputs_and_current_settings_render(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        expect(get_card(page)).to_be_visible()
        expect(page.get_by_text("Missingness rules", exact=True)).to_be_visible()
        expect(by_id(page, "Network")).to_be_visible()
        expect(by_id(page, "Check")).to_be_visible()
        for control in (
            "MinSupport", "MinLift", "MaxLength", "RemoveRedundant", "MaxObs"
        ):
            expect(by_id(page, control)).to_be_attached()

    @pytest.mark.ui
    def test_default_lift_reports_no_strong_rules(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        expect(by_id(page, "Check")).to_contain_text(
            "No significant rules explain", timeout=10_000
        )

    @pytest.mark.ui
    def test_lowering_lift_generates_rule_network(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        set_shiny_input(page, "MinLift", 0.1)
        expect(by_id(page, "Check")).to_contain_text("generate", timeout=10_000)
        expect(by_id(page, "Network").locator(".plotly")).to_be_attached(timeout=10_000)

    @pytest.mark.ui
    def test_flip_displays_rules_table_after_lowering_lift(
        self, page: Page, app: ShinyAppProc
    ):
        page.goto(app.url)
        set_shiny_input(page, "MinLift", 0.1)
        expect(by_id(page, "Check")).to_contain_text("generate", timeout=10_000)
        by_id(page, "FlipButton").click(force=True)
        expect(by_id(page, "Table")).to_be_visible()
        expect(by_id(page, "Table2")).to_be_visible(timeout=10_000)
        table = by_id(page, "Table2")
        for heading in ("LHS", "RHS", "Support", "Confidence", "Lift"):
            expect(table).to_contain_text(heading)

    @pytest.mark.ui
    def test_redundancy_toggle_keeps_status_available(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        set_shiny_input(page, "MinLift", 0.1)
        set_shiny_input(page, "RemoveRedundant", True)
        expect(by_id(page, "Check")).to_be_visible()

    @pytest.mark.ui
    def test_fullscreen_toggle_keeps_network_available(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        by_id(page, "ExpandButton").click(force=True)
        expect(get_card(page)).to_be_visible()
        expect(by_id(page, "Network")).to_be_visible()
