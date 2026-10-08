import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "app"
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import pytest
from card import Card
from cards import obs_weighting as m
from playwright.sync_api import expect
from proxy_data import proxy_data
from roles import Role, RoleMap
from shiny.pytest import create_app_fixture
from sklearn.base import clone
from weighting import ExcessiveZeroWeightsError, WeightingError

app = create_app_fixture(app="../scenarios/obs_weighting.py", scope="function")


def source(nweights=2, zeros=1):
    frame = pd.DataFrame({"importance": np.arange(1., 21.), "balance__weights": np.tile([1., 2.], 10),
                          "extra": np.linspace(2., 3., 20)})
    if zeros:
        frame.loc[:zeros-1, "importance"] = 0
    roles = RoleMap.from_primitive({"weighting": list(frame)[:nweights]})
    return proxy_data(_df=frame, _roles=roles)


@pytest.mark.unit
@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("count,panels", [(1, 2), (2, 4), (3, 5)])
def test_adaptive_charts_and_full_data_counts(count, panels, full):
    data = source(count)
    result = m._analyze(data)
    figure = m._figure(data, result, limit=10, full=full)
    assert len(figure.data) == (panels if full else 2)
    if not full:
        np.testing.assert_allclose(figure.data[0].y, result["effective"].iloc[np.unique(np.linspace(0, 19, 10, dtype=int))])
    assert len(figure.data[0].x) == 10
    assert figure.data[-1].orientation == "h"
    assert sum(figure.data[-1].x) == 20
    assert result["zero_fraction"] == .05
    effective_bar, distribution = figure.data[-2:]
    bar_y = figure.layout[effective_bar.yaxis.replace("y", "yaxis", 1)]
    dist_y = figure.layout[distribution.yaxis.replace("y", "yaxis", 1)]
    assert dist_y.matches == effective_bar.yaxis
    assert dist_y.domain == bar_y.domain
    assert not dist_y.showticklabels
    bar_x = figure.layout[effective_bar.xaxis.replace("x", "xaxis", 1)].domain
    dist_x = figure.layout[distribution.xaxis.replace("x", "xaxis", 1)].domain
    assert (bar_x[1] - bar_x[0]) / (dist_x[1] - dist_x[0]) == pytest.approx(7 / 3)
    assert dist_x[0] - bar_x[1] == pytest.approx(.025)
    if full and count > 1:
        component_x = [figure.layout[t.xaxis.replace("x", "xaxis", 1)].domain for t in figure.data[:-2]]
        component_y = [figure.layout[t.yaxis.replace("y", "yaxis", 1)].domain for t in figure.data[:-2]]
        widths = [right - left for left, right in component_x]
        np.testing.assert_allclose(widths, widths[0])
        assert all(domain == component_y[0] for domain in component_y)
        assert component_y[0][0] > bar_y.domain[1]
        assert component_x[1][0] - component_x[0][1] > .025


@pytest.mark.unit
def test_excessive_and_all_zero_weights_remain_visible():
    for zeros in (3, 20):
        data = source(zeros=zeros)
        result = m._analyze(data)
        assert not result["message"]
        assert result["zeros"] == zeros
        assert len(m._figure(data, result).data) == 2
        with pytest.raises(ExcessiveZeroWeightsError):
            Card.effective_weights(data)


@pytest.mark.unit
def test_power_is_applied_to_components_and_refitted_without_mutation():
    data = source()
    original = data.frame.copy()
    result = m._apply(data, power=.5)
    pd.testing.assert_frame_equal(data.frame, original)
    for column in m._columns(data):
        assert result.frame[column].mean() == pytest.approx(data.frame[column].mean())
    before = Card.effective_weights(data).to_numpy()
    after = Card.effective_weights(result).to_numpy()
    ratios = after[before > 0] / np.sqrt(before[before > 0])
    np.testing.assert_allclose(ratios, ratios[0])
    step = result.pipeline.steps[-1][1]
    assert not hasattr(step, "scales_")
    fitted = clone(step).fit(original.iloc[:10])
    transformed = fitted.transform(original.iloc[10:].drop(columns="balance__weights"))
    assert "balance__weights" not in transformed
    assert result.role_map == data.role_map
    pd.testing.assert_frame_equal(result.clean_frame, data.clean_frame)


@pytest.mark.unit
def test_single_component_and_original_reset():
    data = source()
    result = m._apply(data, target="importance", power=1 / 3)
    expected = np.cbrt(data.frame.importance)
    expected *= data.frame.importance.mean() / expected.mean()
    np.testing.assert_allclose(result.frame.importance, expected)
    pd.testing.assert_series_equal(result.frame.balance__weights, data.frame.balance__weights)
    reset = m._apply(data, power=1.)
    pd.testing.assert_frame_equal(reset.frame, data.frame)


@pytest.mark.unit
def test_test_policy_is_saved_and_preview_is_explicit():
    data = source()
    uniform = m._apply(data, test_policy="uniform")
    assert Card.effective_weights(uniform, purpose="test") is None
    np.testing.assert_allclose(Card.effective_weights(uniform, purpose="test", test_policy="importance"), data.frame.importance)
    assert Card.effective_weights(uniform.clone(), purpose="test") is None
    importance = m._apply(data)
    np.testing.assert_allclose(Card.effective_weights(importance, purpose="test"), data.frame.importance)
    preview = m._analyze(uniform, test=True, test_policy="uniform")
    assert preview["zeros"] == 0 and preview["ess"] == 20
    assert preview["test"]
    fitted = clone(uniform.pipeline.steps[-1][1]).fit(data.frame)
    assert fitted.evaluation_weights(data.frame.drop(columns="balance__weights")) is None


@pytest.mark.unit
def test_training_normalization_is_not_refitted_on_test_rows():
    train = pd.DataFrame({"importance": [1., 4., 9.]})
    step = m.WeightPowerTransformer(("importance",), ("importance",), .5).fit(train)
    test = pd.DataFrame({"importance": [16., 25.]})
    expected = np.sqrt(test.importance) * train.importance.mean() / np.sqrt(train.importance).mean()
    np.testing.assert_allclose(step.transform(test).importance, expected)


@pytest.mark.unit
def test_empty_invalid_constant_and_duplicate_indices():
    assert m._analyze(source(0))["message"]
    data = source(1)
    data.frame.index = [7] * 20
    data.frame["importance"] = 2.
    result = m._analyze(data)
    assert result["ess"] == 20
    assert sum(m._figure(data, result).data[-1].x) == 20
    transformed = m._apply(data, power=.5)
    assert transformed.frame.index.equals(data.frame.index)
    data.frame.iloc[0, 0] = -1
    assert "nonnegative" in m._analyze(data)["message"]
    with pytest.raises(WeightingError):
        m._apply(data, power=.5)


@pytest.mark.unit
def test_card_has_no_flip_and_controls_restore():
    card = m.instance()
    assert card.mutable and not card.hasFlipSide()
    footer = str(card.footer)
    assert "Powers" in footer
    card.restore_configuration_state({"inputs": {"Power": ".5", "Component": "importance", "TestPolicy": "uniform"}})
    assert 'value="uniform" selected' in str(card.settings)


@pytest.mark.ui
def test_dynamic_charts_and_downstream_power(page, app, tmp_path):
    page.set_viewport_size({"width": 1600, "height": 1000})
    page.goto(app.url)
    chart = page.locator('[id$="-Chart"] .js-plotly-plot')
    expect(chart).to_be_attached(timeout=30000)
    expect(page.locator('[id$="-Status"]')).to_contain_text("5.0%")
    page.wait_for_function("document.querySelector('[id$=\"-Chart\"] .js-plotly-plot')?.data?.length === 2")
    page.locator('[id$="-ExpandButton"]').click(force=True)
    page.wait_for_function("document.querySelector('[id$=\"-Chart\"] .js-plotly-plot')?.data?.length === 4")
    page.locator('[id$="-ContractButton"]').click(force=True)
    page.wait_for_function("document.querySelector('[id$=\"-Chart\"] .js-plotly-plot')?.data?.length === 2")
    page.get_by_label("Square root", exact=True).check()
    expect(page.locator('[id$="-Probe"]')).not_to_contain_text("last=19.000")
    page.get_by_label("Original", exact=True).check()
    expect(page.locator('[id$="-Probe"]')).to_contain_text("last=19.000")
    page.get_by_role("button", name="One weight", exact=True).click()
    page.wait_for_function("document.querySelector('[id$=\"-Chart\"] .js-plotly-plot')?.data?.length === 2")
    page.locator('[id$="-ExpandButton"]').click(force=True)
    page.wait_for_function("document.querySelector('[id$=\"-Chart\"] .js-plotly-plot')?.data?.length === 2")
    assert page.locator('[id$="-FlipButton"]').count() == 0

    page.get_by_role("button", name="No weights", exact=True).click()
    expect(page.locator('[id$="-Status"]')).to_contain_text("No variables have the Weighting role.")
    for label in ("Square", "Original", "Square root", "Cube root"):
        expect(page.get_by_label(label, exact=True)).to_be_disabled()
    page.get_by_role("button", name="Two weights", exact=True).click()
    expect(page.get_by_label("Original", exact=True)).to_be_enabled()
    expect(page.get_by_label("Original", exact=True)).to_be_checked()
