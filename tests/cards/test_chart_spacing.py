"""Shared caption-to-widget spacing across card and navbar layouts."""

import pytest
from playwright.sync_api import Page, expect
from shiny.pytest import create_app_fixture

miss_rules_app = create_app_fixture(app="../scenarios/miss_rules.py", scope="function")
miss_informative_app = create_app_fixture(app="../scenarios/miss_informative.py", scope="function")
var_transform_app = create_app_fixture(app="../scenarios/var_transform.py", scope="function")
obs_outliers_app = create_app_fixture(app="../scenarios/obs_outliers.py", scope="function")


@pytest.mark.ui
@pytest.mark.parametrize("scenario", [
    "miss_rules_app", "miss_informative_app", "var_transform_app", "obs_outliers_app",
])
def test_caption_touches_chart_widget(page: Page, request, scenario):
    app = request.getfixturevalue(scenario)
    page.goto(app.url)
    chart = page.locator(".front .shiny-ipywidget-output").first
    expect(chart).to_be_visible(timeout=30_000)
    expect(chart.locator(".plotly")).to_be_attached(timeout=30_000)
    expect(chart).to_be_visible(timeout=30_000)
    gap = chart.evaluate("""el => {
        const wrapper = el.parentElement.id.endsWith('_wrapper') ? el.parentElement : el;
        const title = wrapper.previousElementSibling;
        const caption = getComputedStyle(title).display === 'contents'
            ? title.querySelector('span') : title;
        return el.getBoundingClientRect().top
            - caption.getBoundingClientRect().bottom;
    }""")
    assert gap == pytest.approx(0, abs=1)
