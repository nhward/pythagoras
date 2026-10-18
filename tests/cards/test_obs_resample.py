import pytest
from playwright.sync_api import expect
from shiny.pytest import create_app_fixture

app = create_app_fixture(app="../scenarios/obs_resample.py", scope="function")


@pytest.mark.ui
def test_restoration_never_exports_an_intermediate_no_plan(page, app):
    page.goto(app.url)
    expect(page.locator('[id$="-ExportState"]')).to_have_text(
        "plan=True;ready=True;rows=60", timeout=30000,
    )
    expect(page.locator('[id$="-ExportHistory"]')).not_to_contain_text("False")
    expect(page.locator('[id$="-ExportHistory"]')).to_contain_text("True")


@pytest.mark.ui
def test_restore_compact_fullscreen_flip_and_stale_source(page, app):
    page.set_viewport_size({"width": 850, "height": 950})
    page.goto(app.url)
    by_id = lambda name: page.locator(f'[id$="-{name}"]')
    expect(by_id("ExportState")).to_have_text("plan=True;ready=True;rows=60", timeout=60000)
    expect(by_id("Summary")).to_contain_text("7 splits")
    plot = by_id("Chart").locator(".js-plotly-plot")
    expect(plot).to_be_visible()
    assert plot.evaluate("el => el.layout.showlegend") is True
    assert plot.evaluate("el => el.layout.hovermode") is False
    expect(by_id("GoalControl").locator('input[value="strata"]')).to_be_enabled()
    page.screenshot(path="/tmp/obs-resample-compact.png")
    # Wait for card handlers, then use a normal click.
    page.wait_for_function("() => !!document.querySelector('[id$=\"-ExpandButton\"]')")
    card = page.locator(".card").first
    card.hover()
    by_id("ExpandButton").click()
    expect(card).to_have_attribute("aria-expanded", "true")
    page.wait_for_function("() => document.querySelector('.js-plotly-plot')?.layout?.hovermode === 'closest'")
    expect(plot.locator(".heatmaplayer image").first).to_be_visible()
    page.wait_for_function("() => document.querySelector('.js-plotly-plot')?._fullLayout?.height > 300")
    page.screenshot(path="/tmp/obs-resample-fullscreen.png", animations="disabled")
    by_id("FlipButton").click()
    expect(by_id("Table")).to_contain_text("CV1", timeout=30000)
    with page.expect_download() as download:
        by_id("Membership").click()
    path = download.value.path()
    from pathlib import Path
    lines = Path(path).read_text().splitlines()
    assert len(lines) == 421
    assert "source_position" in lines[0]
    by_id("FlipButton").click()
    by_id("ChangeSource").click()
    expect(by_id("Status")).to_contain_text("Plan regenerated for current source", timeout=30000)
    expect(by_id("ExportState")).to_have_text("plan=True;ready=True;rows=60", timeout=30000)
    expect(by_id("UseSource")).to_have_count(0)
    by_id("GoalControl").locator('input[value="none"]').check()
    expect(by_id("ExportState")).to_have_text("plan=False;ready=False;rows=60")

partition_app = create_app_fixture(app="../scenarios/obs_resample_partition.py", scope="function")


@pytest.mark.ui
def test_multilevel_partition_settings_and_invalid_assignments(page, partition_app):
    page.set_viewport_size({"width": 1400, "height": 950})
    page.goto(partition_app.url)
    by_id = lambda name: page.locator(f'[id$="-{name}"]')
    expect(by_id("ExportState")).to_have_text("No plan", timeout=30000)
    card = page.locator(".card").first
    card.hover()
    card.locator("button.collapse-toggle").click()
    expect(by_id("Stratifier")).to_have_value("site")
    assert by_id("Stratifier").locator("option").evaluate_all("els => els.map(el => el.value)") == ["site", "cluster__membership", ""]
    table = by_id("PartitionMapping")
    expect(table).to_contain_text("Partition: partition")
    expect(table.locator("tr").filter(has=page.get_by_role("cell", name="train a", exact=True)).locator("select")).to_have_value("development")
    def assign(level, purpose):
        table.locator("tr").filter(has=page.get_by_role("cell", name=level, exact=True)).locator("select").select_option(purpose)
    by_id("GoalControl").locator('input[value="observations"]').check()
    expect(by_id("Status")).to_contain_text("Assign every Partition level", timeout=30000)
    for level, purpose in (("train a", "development"), ("train b", "development"),
                           ("validation", "validation"), ("holdout", "holdout"), ("prediction", "prediction")):
        assign(level, purpose)
    expect(by_id("ExportState")).to_have_text("ready=True;development=24;reserved=36", timeout=30000)
    expect(by_id("Summary")).to_contain_text("1 split")
    expect(by_id("Status")).to_contain_text("12 holdout")
    expect(by_id("Status")).to_contain_text("12 prediction")
    expect(by_id("Folds")).to_be_visible()
    expect(by_id("Seed")).to_be_visible()
    expect(by_id("Window")).to_be_visible()
    expect(by_id("MappingState")).to_contain_text('"str:holdout": "holdout"')
    # Removing explicit validation returns to the role-derived stratified K-fold.
    assign("validation", "development")
    expect(by_id("ExportState")).to_have_text("ready=True;development=36;reserved=24", timeout=30000)
    expect(by_id("Summary")).to_contain_text("Stratified K-fold")
    expect(by_id("Folds")).to_be_visible()
    plot = by_id("Chart").locator(".js-plotly-plot")
    page.wait_for_function("() => document.querySelector('.js-plotly-plot')?.layout?.xaxis?.title?.text.includes('Stratifier order')")
    assert plot.evaluate("el => el.data.filter(t => t.type === 'heatmap').length") == 2
    page.screenshot(path="/tmp/obs-resample-strata.png")
    assign("holdout", "development")
    assign("prediction", "development")
    by_id("GoalControl").locator('input[value="strata"]').check()
    by_id("Folds").fill("10")
    by_id("Folds").press("Tab")
    expect(by_id("Summary")).to_contain_text("Grouped K-fold", timeout=30000)
    expect(by_id("Status")).to_contain_text("Using K=3", timeout=30000)
    expect(card.locator('.card-footer [id$="-Status"] span')).to_have_class("text-primary")
    expect(card.locator('.front [id$="-Status"]')).to_have_count(0)
    by_id("Window").fill("99")
    by_id("Window").press("Tab")
    expect(by_id("Status")).to_contain_text("Using K=3")
    # Exercise a real chart-size change, with the production settle configuration.
    by_id("GoalControl").locator('input[value="observations"]').check()
    by_id("Folds").fill("4")
    by_id("Folds").press("Tab")
    expect(by_id("Summary")).to_contain_text("4 splits", timeout=30000)
    page.wait_for_function("() => document.querySelector('[id$=\"-Chart\"] .js-plotly-plot')?.data?.[0]?.z?.length === 4")
    for folds in (5, 6, 7):
        by_id("Folds").fill(str(folds))
        by_id("Folds").press("Tab")
        page.wait_for_timeout(500)
        assert plot.evaluate("el => el.data[0].z.length") == 4
        expect(by_id("Summary")).to_contain_text("4 splits")
        expect(by_id("RecipeState")).to_contain_text('"folds": 4')
    page.wait_for_timeout(700)
    assert plot.evaluate("el => el.data[0].z.length") == 4
    expect(by_id("Summary")).to_contain_text("7 splits", timeout=30000)
    page.wait_for_function("() => document.querySelector('[id$=\"-Chart\"] .js-plotly-plot')?.data?.[0]?.z?.length === 7")
    by_id("GoalControl").locator('input[value="strata"]').check()
    by_id("Folds").fill("10")
    by_id("Folds").press("Tab")
    expect(by_id("Status")).to_contain_text("Using K=3", timeout=30000)
    # All consumers retain their settled state throughout a burst of fold edits.
    previous_status = by_id("Status").inner_text()
    previous_summary = by_id("Summary").inner_text()
    previous_recipe = by_id("RecipeState").inner_text()
    plot.evaluate("el => { window.resampleSettledPlot = el; }")
    for folds in (8, 9):
        by_id("Folds").fill(str(folds))
        by_id("Folds").press("Tab")
        page.wait_for_timeout(300)
        assert plot.evaluate("el => el === window.resampleSettledPlot")
        expect(by_id("Status")).to_have_text(previous_status)
        expect(by_id("Summary")).to_have_text(previous_summary)
        expect(by_id("RecipeState")).to_have_text(previous_recipe)
    expect(by_id("Status")).to_contain_text("Using K=3: requested K=9", timeout=30000)
    expect(by_id("RecipeState")).to_contain_text('"requested_folds": 9')
    page.wait_for_function("() => document.querySelector('[id$=\"-Chart\"] .js-plotly-plot') !== window.resampleSettledPlot")
    by_id("Seed").fill("7")
    by_id("Seed").press("Tab")
    expect(by_id("RecipeState")).to_contain_text('"seed": 7', timeout=30000)
    expect(by_id("Status")).to_contain_text("Using K=3", timeout=30000)
    expect(by_id("ExportState")).to_have_text("ready=True;development=60;reserved=0", timeout=30000)
    expect(page.locator(".shiny-output-error")).to_have_count(0)

    # Unsupported goals are selectable and report validation errors in the footer.
    for goal in ("entities", "future"):
        button = by_id("GoalControl").locator(f'input[value="{goal}"]')
        expect(button).to_be_enabled()
        button.check()
        expect(by_id("Status").locator("span.text-danger")).to_be_visible(timeout=30000)
    by_id("GoalControl").locator('input[value="strata"]').check()
    expect(by_id("ExportState")).to_have_text("ready=True;development=60;reserved=0", timeout=30000)

    by_id("Stratifier").select_option("cluster__membership")
    expect(by_id("Status")).to_contain_text("Using K=2", timeout=30000)
    expect(by_id("ExportState")).to_have_text("ready=True;development=60;reserved=0", timeout=30000)
    by_id("Stratifier").select_option("")
    expect(by_id("Status")).to_contain_text("Select a Stratifier", timeout=30000)
    by_id("GoalControl").locator('input[value="observations"]').check()
    expect(by_id("Summary")).to_contain_text("K-fold", timeout=30000)
    expect(by_id("ExportState")).to_have_text("ready=True;development=60;reserved=0", timeout=30000)
    expect(by_id("Stratifier")).to_have_value("")
