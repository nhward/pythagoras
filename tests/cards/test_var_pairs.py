import numpy as np
import pandas as pd
import pytest
from cards import var_pairs as m
from cyclic_pandas import as_cyclic
from playwright.sync_api import expect
from proxy_data import proxy_data
from roles import RoleMap
from shiny.pytest import create_app_fixture

app = create_app_fixture(app="../scenarios/var_pairs.py", scope="function")


def source(n=120):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({"x": rng.normal(size=n), "y": rng.normal(size=n),
        "cat": pd.Categorical(np.resize(["b", "a", "c"], n), categories=["c", "b", "a"], ordered=True),
        "cat2": pd.Categorical(np.resize(["yes", "no"], n)),
        "angle": as_cyclic(np.arange(n) % 24, period=24),
        "date": pd.date_range("2020-01-01", periods=n),
        "code": pd.Series([f"id{i}" for i in range(n)], dtype="string"),
        "shadow__x": 1., "weight": 1., "id": np.arange(n),
        "facet": pd.Categorical(np.resize(["A", "B"], n))})
    roles = RoleMap.from_primitive({"predictor": ["x", "cat", "cat2", "angle", "date", "code", "shadow__x"],
        "target": ["y"], "weighting": ["weight"], "identifier": ["id"], "stratifier": ["facet"]})
    return proxy_data(_df=frame, _roles=roles)


@pytest.mark.unit
class TestPairs:
    def test_roles_semantics_cap_and_order(self):
        data = source()
        assert set(m._eligible(data)) == {"x", "y", "cat", "cat2", "angle", "date"}
        assert "y" not in m._eligible(data, False)
        data.role_map.set_roles("cat2", ["treatment"])
        assert "cat2" in m._eligible(data)
        assert m._facets(data) == ["facet"]
        assert m._column("cat", data.frame.cat).labels == ["c", "b", "a"]
        column = m._column("angle", data.frame.angle)
        assert column.values[1] == 15
        assert m._column("date", data.frame.date).kind == "numeric"
        for i in range(20):
            data.frame[f"v{i}"] = np.arange(len(data.frame))
            data.role_map.set_roles(f"v{i}", ["predictor"])
        r = m._build(data, [f"v{i}" for i in range(20)], limit=10)
        assert len(r.table) == 66

    def test_all_six_types_axes_hover_and_sampling(self):
        data = source()
        before = data.clone()
        names = ["x", "cat", "y", "cat2", "angle"]
        result = m._build(data, names, "facet", 40)
        assert set(result.table.Chart) == {"Scatter", "Vertical bars", "Horizontal bars", "Mosaic", "Polar scatter", "Polar bars"}
        assert len(result.table) == 10 and result.sampled == 40
        assert data.equals(before)
        assert m._build(data, names, "facet", 40).figure.to_json() == result.figure.to_json()
        small = m._display(result, False)
        full = m._display(result, True)
        assert not small.layout.showlegend and small.layout.hovermode is False
        assert all(t.hoverinfo == "skip" and t.hovertemplate is None for t in small.data)
        assert full.layout.showlegend and any(t.hovertemplate for t in full.data)
        assert any(getattr(full.layout[k], "matches", None) for k in full.layout if k.startswith("xaxis"))
        for row, col in m._cells(5, "lower"):
            sub = full.get_subplot(row+1, col+1)
            if hasattr(sub, "xaxis"):
                assert sub.xaxis.showticklabels == (row == 4)
                assert sub.yaxis.showticklabels == (col == 0)
        full.to_json()

    def test_mosaic_and_band_count_conservation(self):
        data = source()
        r = m._build(data, ["cat", "cat2"], "facet")
        bars = list(r.figure.data)
        area = sum(np.dot(t.width, t.y) for t in bars)
        assert area == pytest.approx(1)
        count = sum(sum(row[2] for row in t.customdata) for t in bars)
        assert count == 120
        r = m._build(data, ["x", "cat"], "facet")
        assert sum(sum(row[1] for row in t.customdata) for t in r.figure.data) == 120

    def test_pairwise_missing_and_unchanged_pipeline(self):
        from sklearn.preprocessing import FunctionTransformer
        data = source()
        data.frame.loc[0, "x"] = np.nan
        data.frame.loc[1, "y"] = np.inf
        data.frame.index = [0]*120
        data = data.with_pipeline_step(FunctionTransformer(), name="identity", preview_frame=data.frame.copy())
        before = data.clone()
        r = m._build(data, ["x", "y", "cat"])
        assert r.table["Plotted rows"].tolist() == [118, 119, 119]
        assert data.equals(before) and data.has_pipeline
        assert m._build(source(0), ["x", "y"]).table.empty
        assert m._build(data, []).table.empty

    @pytest.mark.parametrize("values", [["high", None, "low", "high"], [None, None], []])
    def test_categorical_unused_levels_preserve_order_missingness_and_source(self, values):
        import warnings

        series = pd.Series(pd.Categorical(
            values, categories=["unused", "low", "middle", "high"], ordered=True,
        ))
        before = series.copy()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            codes, labels, note = m._categorical(series)
        expected = [1., np.nan, 0., 1.] if values and values[0] == "high" else [np.nan] * len(values)
        np.testing.assert_allclose(codes, expected, equal_nan=True)
        assert labels == (["low", "high"] if values and values[0] == "high" else [])
        assert not note
        pd.testing.assert_series_equal(series, before)

    def test_checkerboard_unique_pairs_and_grouping(self):
        cells = list(m._cells(12, "checker"))
        assert len(cells) == 66 and len({frozenset(c) for c in cells}) == 66
        assert any(r < c for r, c in cells) and any(r > c for r, c in cells)
        series = pd.Series(pd.Categorical(range(30), categories=range(30), ordered=True))
        codes, labels, note = m._categorical(series)
        assert np.all(np.diff(codes) >= 0) and len(labels) == 12 and note
        data = source()
        data.frame["facet"] = pd.Categorical(np.arange(120))
        r = m._build(data, ["x", "cat"], "facet", layout="checker")
        assert len(r.figure.data) <= 8
        assert "Facet" in " ".join(r.notes)


@pytest.mark.ui
def test_mixed_grid_restoration_and_empty(page, app):
    page.set_viewport_size({"width": 1800, "height": 1100})
    page.goto(app.url)
    by_id = lambda name: page.locator(f'[id$="-{name}"]')
    expect(by_id("Status")).to_contain_text("10 pairs", timeout=60000)
    expect(by_id("PassThrough")).to_have_text("unchanged=True")
    plot = by_id("Chart").locator(".js-plotly-plot")
    expect(plot).to_be_visible()
    assert plot.evaluate("el => el.layout.showlegend") is False
    assert by_id("Facet").input_value() == "site"
    assert by_id("Layout").input_value() == "checker"
    by_id("ExpandButton").click(force=True)
    page.wait_for_function("() => document.querySelector('.js-plotly-plot')?.layout?.showlegend === true")
    assert plot.evaluate("el => el.layout.hovermode") == "closest"
    expect(plot).to_be_visible()
    page.screenshot(path="/tmp/var-pairs-fullscreen.png")
    expect(by_id("FlipButton")).to_have_count(0)
    by_id("Empty").click(force=True)
    expect(by_id("Status")).to_contain_text("0 of 0", timeout=60000)
    expect(plot.locator(".annotation-text")).to_contain_text("Select at least two")
