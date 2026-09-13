from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2] / "app"
os.chdir(APP_ROOT)
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from playwright.sync_api import Page, expect
from proxy_data import proxy_data
from roles import Role, RoleMap
from scipy.spatial.distance import pdist, squareform
from shiny.playwright import controller
from shiny.pytest import create_app_fixture
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from threadpoolctl import threadpool_limits

app = create_app_fixture(app="../scenarios/obs_k_clusters.py", scope="function")


@pytest.fixture(scope="session")
def browser_context_args():
    return {"viewport": {"width": 1600, "height": 1000}}


@pytest.fixture(scope="module")
def module():
    return importlib.import_module("cards.obs_k_clusters")


@pytest.fixture(autouse=True)
def single_thread():
    with threadpool_limits(limits=1):
        yield


def data(weights=True):
    rng = np.random.default_rng(1729)
    x = np.vstack([rng.normal(-4, .3, (12, 2)), rng.normal(4, .3, (12, 2))])
    frame = pd.DataFrame(x, columns=["x", "y"])
    roles = RoleMap()
    for col in frame:
        roles.set_roles(col, [Role.PREDICTOR])
    if weights:
        frame["importance"] = np.resize([.5, 1., 2., 4.], len(frame))
        roles.set_roles("importance", [Role.WEIGHTING])
    return proxy_data(_df=frame, _roles=roles, _name="Two groups")


def analyze(module, source=None, **kwargs):
    options = dict(maximum=3, min_points=3, graph_neighbours=3,
                   stability_repeats=2, gap_references=2)
    options.update(kwargs)
    return module._analyse(data() if source is None else source, **options)


@pytest.mark.unit
class TestEvidence:
    def test_unweighted_families_and_two_group_partition(self, module):
        result = analyze(module, data(False))
        assert set(result["scores"].Family) == set(module.FAMILIES)
        assert set(result["votes"].query("Family == 'Partition'").K) == {2}
        assert result["votes"].K.between(1, 3).all()
        assert np.isfinite(result["scores"].Score).all()

    @pytest.mark.parametrize("centre", ["centroids", "medoids"])
    def test_real_weights_affect_scores_and_supported_families(self, module, centre):
        weighted = analyze(module, centre=centre)
        unweighted = analyze(module, centre=centre, use_weights=False)
        assert set(weighted["scores"].Family) == {"Partition", "Density", "Stability", "Gap statistic"}
        left = weighted["scores"].query("Family == 'Partition'").Score.to_numpy()
        right = unweighted["scores"].query("Family == 'Partition'").Score.to_numpy()
        assert not np.allclose(left, right)
        assert "unavailable" in " ".join(weighted["notes"])

    def test_equal_weights_reproduce_unweighted_results(self, module):
        source = data()
        source.frame["importance"] = 7.
        assert_frame_equal(analyze(module, source)["scores"], analyze(module, source, use_weights=False)["scores"])

    def test_weight_scale_does_not_change_scores(self, module):
        source = data()
        first = analyze(module, source)
        source.frame["importance"] *= 100
        second = analyze(module, source)
        assert_frame_equal(first["scores"], second["scores"], atol=1e-9, rtol=1e-9)
        assert_frame_equal(first["votes"], second["votes"])

    def test_only_predictors_enter_analysis_even_when_weighting_disabled(self, module):
        source = data()
        expected = analyze(module, source, use_weights=False)
        roles = RoleMap.from_primitive(source.role_map.to_primitive())
        frame = source.frame.copy()
        for column, role in [("target", Role.TARGET), ("identifier", Role.IDENTIFIER)]:
            frame[column] = np.nan
            roles.set_roles(column, [role])
        for column, values in [("constant", 1.), ("flag", True), ("text", "a"), ("shadow__x", np.nan)]:
            frame[column] = values
            roles.set_roles(column, [Role.PREDICTOR])
        actual = analyze(module, proxy_data(_df=frame, _roles=roles), use_weights=False)
        assert_frame_equal(expected["scores"], actual["scores"])

    def test_zero_weight_and_incomplete_rows_are_excluded_with_duplicate_indices(self, module):
        source = data()
        source.frame.index = [0] * len(source)
        source.frame.iloc[0, source.frame.columns.get_loc("importance")] = 0
        source.frame.iloc[1, 0] = np.nan
        expected = proxy_data(_df=source.frame.iloc[2:].copy(), _roles=source.role_map)
        assert_frame_equal(analyze(module, source)["scores"], analyze(module, expected)["scores"])

    @pytest.mark.parametrize("bad", [-1., np.nan, np.inf])
    def test_invalid_weight_is_rejected_and_can_be_disabled(self, module, bad):
        source = data()
        source.frame.loc[0, "importance"] = bad
        with pytest.raises(ValueError, match="Importance weights"):
            analyze(module, source)
        assert not analyze(module, source, use_weights=False)["votes"].empty

    def test_all_zero_weights_rejected(self, module):
        source = data()
        source.frame["importance"] = 0.
        with pytest.raises(ValueError, match="positive"):
            analyze(module, source)

    def test_no_numeric_predictor_gives_explanation(self, module):
        source = proxy_data(_df=pd.DataFrame({"label": ["a", "b", "c"]}))
        result = analyze(module, source)
        assert result["scores"].empty
        assert "varying numeric predictor" in " ".join(result["notes"])

    def test_resampling_can_be_disabled(self, module):
        result = analyze(module, stability=False, gap=False)
        assert not set(result["scores"].Family) & {"Stability", "Gap statistic"}

    def test_weighted_metrics_match_standard_metrics_for_unit_weights(self, module):
        x = data(False).frame.to_numpy()
        labels = np.repeat([0, 1], 12)
        scores = module._weighted_metrics(x, squareform(pdist(x)), labels, np.ones(len(x)))
        expected = [silhouette_score(x, labels), calinski_harabasz_score(x, labels), davies_bouldin_score(x, labels)]
        assert [s[1] for s in scores] == pytest.approx(expected)

    def test_prediction_strength_counts_weighted_distinct_pairs(self, module):
        labels = np.array([0, 0, 0])
        predicted = np.array([0, 0, 1])
        assert module._pair_prediction_strength(labels, predicted, 1, np.array([1., 2., 3.])) == pytest.approx(2 / 11)
        assert module._pair_prediction_strength(np.array([0, 1, 1]), np.array([0, 1, 1]), 2) == 0

    def test_spectral_detects_two_disconnected_groups(self, module):
        x = np.array([[0.], [.1], [.2], [10.], [10.1], [10.2]])
        scores, _ = module._spectral_scores(squareform(pdist(x)), 3, 2)
        assert max(scores, key=lambda row: row[3])[2] == 2
        scores, notes = module._spectral_scores(squareform(pdist(x)), 1, 2)
        assert scores == []
        assert "more components" in " ".join(notes)

    def test_resampling_selection_rules_and_display_boundary(self, module, monkeypatch):
        monkeypatch.setattr(module, "_stability_cached", lambda *args: ([(1, 1., 0.), (2, .9, .03), (3, .7, .05)], []))
        monkeypatch.setattr(module, "_gap_cached", lambda *args: ([(1, .1, .01), (2, 1., .05), (3, 1.02, .1), (4, 1.03, .1)], []))
        scores, votes, _ = module._resampling_evidence(data(False).frame.to_numpy(), 3)
        assert ("Stability", "Prediction strength", 2) in votes
        assert ("Gap statistic", "Gap", 2) in votes
        assert max(row[2] for row in scores) == 3

    def test_gap_no_qualifying_count_casts_no_vote(self, module, monkeypatch):
        monkeypatch.setattr(module, "_gap_cached", lambda *args: ([(1, 0., .01), (2, 1., .01), (3, 2., .01)], []))
        _, votes, notes = module._resampling_evidence(data(False).frame.to_numpy(), 2, stability=False)
        assert votes == []
        assert "no vote" in " ".join(notes)

    def test_weighted_gap_changes_and_cache_distinguishes_weights(self, module):
        module._gap_cached.cache_clear()
        weighted = analyze(module)
        first = module._gap_cached.cache_info()
        again = analyze(module)
        assert module._gap_cached.cache_info().hits == first.hits + 1
        assert_frame_equal(weighted["scores"], again["scores"])
        unweighted = analyze(module, use_weights=False)
        assert module._gap_cached.cache_info().misses == first.misses + 1
        a = weighted["scores"].query("Family == 'Gap statistic' and Criterion == 'Gap'").Score
        b = unweighted["scores"].query("Family == 'Gap statistic' and Criterion == 'Gap'").Score
        assert not np.allclose(a, b)

    def test_sampling_keeps_weights_aligned_and_uses_weighted_standardization(self, module, monkeypatch):
        source = data()
        source.frame.index = [0] * len(source)
        captured = []

        def capture(x, maximum, **kwargs):
            captured.append((x.copy(), kwargs["weights"].copy()))
            return [], [], []

        monkeypatch.setattr(module, "_resampling_evidence", capture)
        analyze(module, source, limit=12)
        x, weights = captured[0]
        positions = np.sort(np.random.default_rng(2025).choice(24, size=12, replace=False))
        raw = source.frame.iloc[positions]
        expected_w = raw.importance.to_numpy()
        expected_w = expected_w / expected_w.mean()
        np.testing.assert_allclose(weights, expected_w)
        expected_x = raw[["x", "y"]].to_numpy()
        mean = np.average(expected_x, axis=0, weights=expected_w)
        spread = np.sqrt(np.average((expected_x - mean) ** 2, axis=0, weights=expected_w))
        np.testing.assert_allclose(x, (expected_x - mean) / spread)
        assert len(source.frame) == 24

    def test_cluster_decision_clones_metadata_without_changing_source(self):
        source = data()
        assert source.cluster_count == 1
        result = source.with_cluster_count(3)
        assert result.cluster_count == result.clone().cluster_count == 3
        assert source.cluster_count == 1
        assert_frame_equal(result.frame, source.frame)
        assert result.role_map == source.role_map
        assert result.processing_records == source.processing_records
        assert not result.equals(source)
        assert result.equals(source, check_cluster_count=False)


def by_id(page, name):
    card_id = page.locator(".card").first.get_attribute("id")
    return page.locator(f"#{card_id.partition('-')[0]}-{name}")


def open_settings(page):
    page.locator(".card").first.hover()
    page.locator(".card").first.locator("button.collapse-toggle").click()


@pytest.mark.ui
class TestWeb:
    def test_incoming_choice_exports_and_includes_maximum(self, page: Page, app):
        page.goto(app.url)
        expect(by_id(page, "ExportProbe")).to_contain_text("Export K=2; rows=32; weighting=importance; unchanged=True", timeout=30000)
        expect(by_id(page, "K").locator('input[value="2"]')).to_be_checked()
        by_id(page, "K").locator('input[value="10"]').check()
        expect(by_id(page, "ExportProbe")).to_contain_text("Export K=10")
        by_id(page, "K").locator('input[value="1"]').check()
        expect(by_id(page, "ExportProbe")).to_contain_text("Export K=1")

    def test_weighting_toggle_recalculates_evidence(self, page: Page, app):
        page.goto(app.url)
        expect(by_id(page, "Summary")).to_contain_text("Importance column: importance", timeout=60000)
        expect(by_id(page, "Summary")).to_contain_text("Analysis uses 31 rows")
        open_settings(page)
        by_id(page, "UseWeights").uncheck()
        expect(by_id(page, "Summary")).to_contain_text("Observation weighting is disabled", timeout=60000)
        expect(by_id(page, "Summary")).to_contain_text("Analysis uses 32 rows")
        by_id(page, "UseWeights").check()
        expect(by_id(page, "Summary")).to_contain_text("Importance column: importance", timeout=60000)
        expect(by_id(page, "ExportProbe")).to_contain_text("unchanged=True")

    def test_chart_discrete_shared_range_and_evidence_table(self, page: Page, app):
        page.goto(app.url)
        expect(by_id(page, "Summary")).to_contain_text("Importance column", timeout=60000)
        page.wait_for_function("() => document.querySelector('.js-plotly-plot')?.data?.length > 0")
        chart = page.locator('.js-plotly-plot').first.evaluate("el => ({axis:el.layout.xaxis, traces:el.data.map(t=>({x:t.x,width:t.width,name:t.name}))})")
        assert chart['axis']['type'] == 'category'
        assert chart['axis']['range'] == [-.5, 9.5]
        for trace in chart['traces']:
            assert trace['x'] == [str(k) for k in range(1, 11)]
            assert trace['width'] == .8
        assert {t['name'] for t in chart['traces']} <= {'Partition', 'Density', 'Stability', 'Gap statistic'}
        page.locator('.card').first.hover()
        by_id(page, 'FlipButton').click(force=True)
        expect(by_id(page, 'EvidenceTable')).to_contain_text('Recommended')
        expect(by_id(page, 'Notes')).to_contain_text('Unequal importance weights')

    def test_lowering_maximum_caps_decision_and_chart(self, page: Page, app):
        page.goto(app.url)
        expect(by_id(page, "K").locator('input[value="2"]')).to_be_checked(timeout=30000)
        by_id(page, "K").locator('input[value="10"]').check()
        expect(by_id(page, "ExportProbe")).to_contain_text("Export K=10")
        open_settings(page)
        slider = controller.InputSlider(page, by_id(page, "Maximum").get_attribute("id"))
        slider.set("3")
        expect(by_id(page, "K").locator('input[type="radio"]')).to_have_count(3)
        expect(by_id(page, "K").locator('input[value="3"]')).to_be_checked()
        expect(by_id(page, "ExportProbe")).to_contain_text("Export K=3")
        page.wait_for_function("() => document.querySelector('.js-plotly-plot')?.layout?.xaxis?.categoryarray?.length === 3")
        chart = page.locator('.js-plotly-plot').first.evaluate("el => ({range:el.layout.xaxis.range, xs:el.data.map(t=>t.x)})")
        assert chart['range'] == [-.5, 2.5]
        assert all(x == ['1', '2', '3'] for x in chart['xs'])
