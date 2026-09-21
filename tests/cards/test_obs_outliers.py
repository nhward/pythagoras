"""Complementary outlier rankings and diagnostic-only browser behavior."""
import os
import sys
from pathlib import Path
from threading import Event

ROOT = Path(__file__).resolve().parents[2] / "app"
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest
from cards import obs_outliers as m
from playwright.sync_api import expect
from proxy_data import proxy_data
from roles import Role, RoleMap
from shiny.pytest import create_app_fixture

app = create_app_fixture(app="../scenarios/obs_outliers.py", scope="function")

@pytest.fixture(scope="session")
def browser_context_args():
    return {"viewport": {"width": 1800, "height": 1100}}


def source():
    rng = np.random.default_rng(2025)
    x = rng.normal(size=100)
    z = rng.normal(size=100)
    x[-1], z[-1] = 12, 15
    frame = pd.DataFrame({"x": x, "z": z, "target": 2*x + rng.normal(size=100),
                          "id": [f"person-{i}" for i in range(100)], "weight": 1.,
                          "category": pd.Categorical(["a", "b"]*50), "shadow__x": x})
    roles = RoleMap()
    for c in frame:
        roles.set_roles(c, [Role.PREDICTOR])
    for c, role in [("id", Role.IDENTIFIER), ("weight", Role.WEIGHTING), ("target", Role.TARGET)]:
        roles.set_roles(c, [role])
    return proxy_data(_df=frame, _roles=roles)


@pytest.mark.unit
class TestOutliers:
    def test_predictors_identity_and_scores(self):
        data = source()
        before = data.clone()
        result = m._analyze(data, trees=25)
        assert data.equals(before)
        assert result.predictors == ["x", "z"]
        assert list(result.raw) == list(m.METHODS)
        assert np.isfinite(result.raw.to_numpy()).all()
        table = m._table(result)
        assert table.sort_values("Aggregate").iloc[-1].Row == 100
        assert table.iloc[-1].Identifier == "person-99"
        assert result.ranks.min().min() >= 0 and result.ranks.max().max() <= 100
        assert any("weights are not used" in note for note in result.notes)

    def test_percentiles_ties_constant_and_scale(self):
        assert m._percentiles([1, 1, 1]).tolist() == [0, 0, 0]
        assert m._percentiles([0, 1, 1, 2]).tolist() == [0, 50, 50, 100]
        np.testing.assert_allclose(m._percentiles([1, 7, 2]), m._percentiles([10, 70, 20]))

    def test_missing_sampling_duplicate_index(self):
        data = source()
        data.frame.index = [0]*100
        data.frame.iloc[0, 0] = np.nan
        data.frame.iloc[1, 1] = np.inf
        a = m._analyze(data, limit=30, trees=25)
        b = m._analyze(data, limit=30, trees=25)
        assert a.eligible == 98 and len(a.raw) == 30 and a.total == 100
        assert not {1, 2} & set(a.observations.Row)
        pd.testing.assert_frame_equal(a.raw, b.raw)
        pd.testing.assert_frame_equal(a.observations, b.observations)

    def test_cooks_unavailable_does_not_break_others(self):
        data = source()
        data.role_map.set_roles("target", [Role.NONE])
        r = m._analyze(data, trees=25)
        assert len(r.raw.columns) == 4 and not r.message
        assert any("Cook's distance unavailable" in note for note in r.notes)
        data.role_map.set_roles("target", [Role.TARGET])
        data.frame.loc[0, "target"] = np.nan
        r = m._analyze(data, trees=25)
        assert len(r.raw) == 100 and len(r.raw.columns) == 4

    def test_collinear_predictors_and_perfect_fit(self):
        data = source()
        data.frame["z"] = data.frame.x
        r = m._analyze(data, trees=25)
        assert "Mahalanobis" in r.raw and "Cook's distance" not in r.raw
        data = source()
        data.frame["target"] = 2*data.frame.x
        assert "Cook's distance" not in m._analyze(data, trees=25).raw

    def test_no_data_constant_no_methods_and_cancellation(self):
        assert m._analyze(None).message
        data = source()
        data.frame["x"] = data.frame["z"] = 1
        assert "varying" in m._analyze(data).message
        assert "No usable" in m._analyze(source(), methods=[]).message
        cancel = Event(); cancel.set()
        assert "superseded" in m._analyze(source(), cancel=cancel).message
        data = source()
        data.role_map.set_roles("x", [Role.NONE]); data.role_map.set_roles("z", [Role.NONE])
        assert "No numeric" in m._analyze(data).message

    def test_figures_sort_by_own_scores_and_stack_equal_contributions(self):
        r = m._analyze(source(), trees=25)
        a = m._figure(r, top=10)
        for method in m.METHODS:
            chart = m._figure(r, top=10, view=method)
            expected = r.raw[method].sort_values(ascending=False, kind="stable").head(10).index
            assert list(chart.data[0].x) == r.observations.loc[expected, "Row"].astype(str).tolist()
            assert np.all(np.diff(chart.data[0].y) <= 0)
        assert a.layout.xaxis.type == "category"
        assert len(a.data) == 5 and not a.layout.showlegend
        assert m._figure(r, full_screen=True).layout.showlegend
        table = m._table(r).sort_values("Aggregate", ascending=False, kind="stable").head(10)
        np.testing.assert_allclose(sum(np.array(trace.y) for trace in a.data), table.Aggregate)
        assert m._figure(m.OutlierAnalysis(message="Nothing to draw")).layout.images


def by_id(page, name):
    ns = page.locator(".card").first.get_attribute("id").partition("-")[0]
    return page.locator(f"#{ns}-{name}")

@pytest.mark.ui
def test_chart_tables_controls_and_unchanged_output(page, app):
    page.goto(app.url)
    expect(by_id(page, "Status")).to_contain_text("100 of 100 observations analyzed", timeout=60000)
    expect(by_id(page, "PassThrough")).to_have_text("unchanged=True")
    page.locator(".card").first.hover()
    by_id(page, "CodeButton").click(force=True)
    modal = page.locator(".modal.show")
    expect(modal).to_contain_text("def _analyze(")
    expect(modal).to_contain_text("def _percentiles(")
    expect(modal).to_contain_text("def _figure(")
    expect(modal).not_to_contain_text("@record_code")
    modal.get_by_role("button", name="Dismiss", exact=True).click()
    plot = by_id(page, "Chart0").locator(".js-plotly-plot")
    expect(plot).to_be_visible()
    assert plot.evaluate("el => el.data.length") == 5
    page.get_by_role("tab", name="Local Outlier Factor", exact=True).click()
    expect(by_id(page, "Chart2").locator(".js-plotly-plot")).to_be_visible()
    page.locator(".card").first.hover()
    by_id(page, "FlipButton").click(force=True)
    expect(by_id(page, "Scores")).to_contain_text("Disagreement")
    expect(by_id(page, "Scores")).to_contain_text("person-99")
    by_id(page, "Raw").check()
    expect(by_id(page, "Scores")).to_contain_text("Mahalanobis")
    by_id(page, "EmptySource").click()
    expect(by_id(page, "Status")).to_contain_text("No varying numeric predictors", timeout=60000)
    by_id(page, "FlipButton").click(force=True)
    page.get_by_role("tab", name="Aggregate", exact=True).click()
    expect(plot).to_be_visible()
    expect(plot.locator(".annotation-text")).to_contain_text("No varying numeric predictors")
    assert plot.evaluate("el => el.layout.images.length") == 1


@pytest.mark.unit
def test_one_method_failure_preserves_other_scores(monkeypatch):
    def fail(self, x):
        raise ValueError("fixture fitting failure")
    monkeypatch.setattr(m.LocalOutlierFactor, "fit", fail)
    result = m._analyze(source(), trees=25)
    assert not result.message and len(result.raw.columns) == 4
    assert any("fixture fitting failure" in note for note in result.notes)
    assert len(m._figure(result).data) == 4


@pytest.mark.ui
def test_method_controls_without_order_setting(page, app):
    page.goto(app.url)
    expect(by_id(page, "Status")).to_contain_text("5 methods", timeout=60000)
    page.locator(".card").first.hover()
    page.locator(".card").first.locator("button.collapse-toggle").click()
    expect(by_id(page, "Order")).to_have_count(0)
    for checkbox in by_id(page, "Methods").locator('input[type="checkbox"]').all():
        checkbox.uncheck()
    expect(by_id(page, "Status")).to_contain_text("No usable evaluation methods", timeout=60000)
    by_id(page, "Methods").locator('input[value="Isolation Forest"]').check()
    expect(by_id(page, "Status")).to_contain_text("1 methods", timeout=60000)
    expect(by_id(page, "PassThrough")).to_have_text("unchanged=True")


@pytest.mark.unit
def test_recorded_helpers_are_run_scoped_and_replayable():
    from module import Module

    # Use the production recorder without constructing a Shiny session.
    class Recorder:
        record_code = Module.record_code

        def __init__(self):
            self.code_registry = {}

    recorder, other = Recorder(), Recorder()
    analyze, table, figure, _ = m._code_functions(recorder.record_code)
    m._code_functions(other.record_code)
    assert not recorder.code_registry
    data = source()
    result = analyze(data, methods=["Isolation Forest"], trees=25)
    assert set(recorder.code_registry) == {"_analyze", "_predictors", "_numeric", "_percentiles"}
    expected = figure(result, top=8)
    table(result)
    assert {"_figure", "_table"} <= recorder.code_registry.keys()
    assert "_cooks" not in recorder.code_registry
    assert not other.code_registry

    analyze(data, methods=["Cook's distance"])
    assert "_cooks" in recorder.code_registry
    # The displayed definitions can execute with the module's imports, constants
    # and result container, without the factory, recorder, or a reactive context.
    namespace = {name: value for name, value in vars(m).items()
                 if not name.startswith("_")}
    for code in recorder.code_registry.values():
        cleaned = Module.clean_code(code)
        assert "@record_code" not in cleaned
        exec(cleaned, namespace)  # noqa: S102
    replay = namespace["_analyze"](data, methods=["Isolation Forest"], trees=25)
    pd.testing.assert_frame_equal(replay.raw, result.raw)
    assert namespace["_figure"](replay, top=8).to_json() == expected.to_json()
