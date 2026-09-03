from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from playwright.sync_api import Page, expect
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

path = Path(__file__).resolve().parents[2] / "app"
os.chdir(path)
if str(path) not in sys.path:
    sys.path.insert(0, str(path))

app = create_app_fixture(app="../scenarios/data_homogeneous.py", scope="function")


@pytest.fixture(scope="module")
def card_module():
    return importlib.import_module("cards.data_homogeneity")


def drift_data():
    from proxy_data import proxy_data
    from roles import Role, RoleMap

    rng = np.random.default_rng(47)
    count = 60
    frame = pd.DataFrame({
        "when": pd.date_range("2024-01-01", periods=count, freq="D"),
        "numeric": np.r_[rng.normal(0, 0.1, 30), rng.normal(5, 0.1, 30)],
        "category": ["before"] * 30 + ["after"] * 30,
        "stable": np.tile([0.0, 1.0, 2.0, 3.0, 4.0], 12),
        "missing": np.r_[np.arange(30, dtype=float), np.full(30, np.nan)],
    })
    roles = RoleMap()
    roles.set_roles("when", [Role.SEQUENCE])
    for column in frame.columns.drop("when"):
        roles.set_roles(column, [Role.PREDICTOR])
    return proxy_data(_df=frame, _roles=roles), frame


@pytest.mark.unit
def test_card_is_immutable_with_heatmap_summary_and_settings(card_module):
    card = card_module.instance()

    assert card.mutable is False
    assert card.long_name == "Data homogeneity"
    assert 'id="DriftChart"' in str(card.front)
    assert 'id="Summary"' in str(card.back)
    assert 'id="Check"' in str(card.footer)
    settings = str(card.settings)
    assert 'id="Sequence"' in settings
    assert 'id="Variables"' in settings
    assert 'id="Groups"' in settings
    assert 'id="Reference"' in settings
    assert 'id="Threshold"' in settings


@pytest.mark.unit
def test_sequence_candidates_prefer_sequence_role(card_module):
    data, _ = drift_data()

    choices = card_module._sequence_candidates(data)

    assert list(choices)[:2] == [card_module.ROW_ORDER, "when"]
    assert choices["when"] == "when"


@pytest.mark.unit
def test_unique_string_is_a_sequence_candidate_and_cardinality_is_respected(card_module):
    from proxy_data import proxy_data

    frame = pd.DataFrame({
        "CODE": ["A01", "A02", "A03", "A04"],
        "repeated_text": ["north", "north", "south", "south"],
        "unique_with_gap": [10.0, 20.0, np.nan, 40.0],
    })

    choices = card_module._sequence_candidates(proxy_data(_df=frame))

    assert "CODE" in choices
    assert "repeated_text" not in choices
    assert "unique_with_gap" in choices
    assert list(choices).index("CODE") < list(choices).index("unique_with_gap")


@pytest.mark.unit
def test_groups_respect_minimum_size(card_module):
    groups = card_module._assign_groups(23, requested=20, minimum_size=5)

    counts = pd.Series(groups).value_counts()
    assert len(counts) == 4
    assert counts.min() >= 5
    assert list(groups) == sorted(groups)


@pytest.mark.unit
def test_sequence_sort_is_stable_and_places_missing_last(card_module):
    frame = pd.DataFrame({"sequence": [2, 1, 1, np.nan], "value": ["d", "a", "b", "z"]})

    ordered = card_module._order_frame(frame, "sequence")

    assert list(ordered["value"]) == ["a", "b", "d", "z"]


@pytest.mark.unit
def test_known_numeric_categorical_and_missingness_drift_are_detected(card_module):
    data, frame = drift_data()

    result = card_module._analyse_homogeneity(
        data,
        sequence="when",
        variables=["numeric", "category", "stable", "missing"],
        group_count=6,
        reference="overall",
        threshold=0.25,
        permutations=40,
    )

    table = result.summary.set_index("Variable")
    assert table.loc["numeric", "Status"] == "Strong"
    assert table.loc["category", "Status"] == "Strong"
    assert table.loc["missing", "Maximum missingness difference"] >= 0.5
    assert table.loc["stable", "Maximum drift"] < table.loc["numeric", "Maximum drift"]
    assert result.groups == 6
    assert result.observations == len(frame)


@pytest.mark.unit
def test_chance_correction_is_zero_at_boundary_and_uses_random_scale(card_module):
    corrected = card_module._chance_correct(
        np.array([0.20, 0.30, 0.35, 0.40]),
        boundary=0.30,
        random_scale=0.10,
    )

    assert corrected.tolist() == pytest.approx([0.0, 0.0, 0.5, 1.0])


@pytest.mark.unit
def test_ass2_age25_random_order_noise_is_calibrated_away(card_module):
    from proxy_data import proxy_data

    frame = pd.read_csv("data/Ass2.csv")
    result = card_module._analyse_homogeneity(
        proxy_data(_df=frame), sequence="CODE",
        variables=["AGE25_PROPTN"], group_count=20,
        reference="overall", threshold=0.25,
        permutations=60, random_state=2025,
    )
    row = result.summary.iloc[0]

    assert row["Raw maximum"] > 0.8
    assert row["Random boundary"] > 0.8
    assert row["Maximum drift"] < 0.25
    assert row["Status"] != "Strong"


@pytest.mark.unit
def test_reference_modes_change_the_question(card_module):
    data, _ = drift_data()
    arguments = dict(
        data=data, sequence="when", variables=["numeric"], group_count=6,
        threshold=0.25, permutations=40,
    )

    first = card_module._analyse_homogeneity(reference="first", **arguments)
    previous = card_module._analyse_homogeneity(reference="previous", **arguments)

    assert first.scores.iloc[0]["Drift"] == pytest.approx(0)
    assert previous.scores.iloc[0]["Drift"] == pytest.approx(0)
    assert first.summary.iloc[0]["Maximum drift"] > 0.25
    assert previous.summary.iloc[0]["Maximum drift"] > 0.25


@pytest.mark.unit
def test_structured_and_high_cardinality_variables_are_excluded(card_module):
    from list_pandas import as_list
    from proxy_data import proxy_data

    frame = pd.DataFrame({
        "numeric": range(40),
        "identifier_text": [f"id-{value}" for value in range(40)],
    })
    frame["basket"] = as_list(pd.Series([[value] for value in range(40)]))

    eligible, excluded = card_module._eligible_columns(proxy_data(_df=frame))

    assert "numeric" in eligible
    assert "basket" in excluded
    assert "List" in excluded["basket"]
    assert "identifier_text" in excluded
    assert "30" in excluded["identifier_text"]

    analysis = card_module._analyse_homogeneity(
        proxy_data(_df=frame), sequence=card_module.ROW_ORDER,
        variables=["numeric"], group_count=4, reference="overall",
        threshold=0.25, permutations=20,
    )
    reported = analysis.summary.set_index("Variable")
    assert reported.loc["basket", "Status"] == "Excluded"
    assert reported.loc["identifier_text", "Status"] == "Excluded"


@pytest.mark.unit
def test_heatmap_aligns_groups_and_variables(card_module):
    data, _ = drift_data()
    analysis = card_module._analyse_homogeneity(
        data, sequence="when", variables=["numeric", "category"],
        group_count=6, reference="overall", threshold=0.25, permutations=30,
    )

    figure = card_module._homogeneity_figure(analysis, threshold=0.25)

    assert figure.data[0].type == "heatmap"
    assert np.asarray(figure.data[0].z).shape == (2, 6)
    assert list(figure.data[0].x) == [1, 2, 3, 4, 5, 6]
    assert list(figure.data[0].y) == ["numeric", "category"]
    assert figure.layout.xaxis.title.text.startswith("Consecutive groups")
    assert figure.layout.yaxis.title.text == "Variables"
    assert figure.layout.yaxis.autorange == "reversed"
    assert "Excess" in figure.data[0].colorbar.title.text


def get_card(page: Page):
    return page.locator(".card").first


def by_id(page: Page, local_id: str):
    card_id = get_card(page).get_attribute("id")
    assert card_id is not None
    return page.locator(f"#{card_id.partition('-')[0]}-{local_id}")


class TestWebKitUI:
    @pytest.mark.ui
    def test_chart_controls_and_status_render(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)

        expect(by_id(page, "DriftChart").locator(".plotly")).to_be_attached(timeout=20_000)
        expect(by_id(page, "Check")).to_contain_text("Compared 3 variables", timeout=20_000)
        expect(by_id(page, "Check")).to_contain_text("12 groups", timeout=20_000)
        expect(by_id(page, "Sequence")).to_be_attached()
        expect(by_id(page, "Variables")).to_be_attached()

    @pytest.mark.ui
    def test_flip_displays_ranked_drift_table(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        expect(by_id(page, "Check")).to_contain_text("Compared", timeout=20_000)

        by_id(page, "FlipButton").click(force=True)

        table = by_id(page, "SummaryTable")
        expect(table).to_be_visible(timeout=10_000)
        for heading in ("Variable", "Maximum drift", "Peak location", "Trend", "Status"):
            expect(table).to_contain_text(heading)
        expect(table).to_contain_text("shifted")
        expect(table).to_contain_text("Strong")
