import pytest
from shiny import reactive
from playwright.sync_api import Page, expect
from shiny.run import ShinyAppProc
from shiny.pytest import create_app_fixture
import importlib

app = create_app_fixture(app="../cards/DataPlaceholders.py")

# ---------- fixtures ----------------------------------------------------

@pytest.fixture(scope="session")
def browser_context_args():
    return {
        "viewport": {"width": 1600, "height": 1000},
    }

@pytest.fixture(scope="session")
def browser_type_launch_args():
    return {
        "headless": False,
        "slow_mo": 200,
    }

@pytest.fixture
def card_module():
    """
    Import the card module itself so we can inspect test-mode seeded state.
    """
    return importlib.import_module("cards.DataPlaceholders")


# ---------- helpers -----------------------------------------------------

def get_card(page: Page):
    return page.locator(".card").first


def get_card_id(page: Page) -> str:
    return get_card(page).get_attribute("id")


def get_ns(page: Page) -> str:
    card_id = get_card_id(page)
    return card_id.partition("-")[0]


def ns_wrap(page: Page, local_id: str) -> str:
    return f"{get_ns(page)}-{local_id}"


def by_id(page: Page, local_id: str):
    return page.locator(f"#{ns_wrap(page, local_id)}")


def get_flip_button(page: Page):
    return by_id(page, "FlipButton")


def get_fullscreen_button(page: Page):
    return by_id(page, "ExpandButton")


def set_shiny_input(page: Page, local_id: str, value):
    """
    Set a Shiny input directly from the browser.
    Useful for selectize / checkbox / slider inputs in UI tests.
    """
    input_id = ns_wrap(page, local_id)
    page.evaluate(
        """
        ([inputId, value]) => {
            if (!window.Shiny || !window.Shiny.setInputValue) {
                throw new Error("Shiny.setInputValue not available");
            }
            window.Shiny.setInputValue(inputId, value, { priority: "event" });
        }
        """,
        [input_id, value],
    )


# ---------- unit tests --------------------------------------------------

@pytest.mark.unit
def test_instance_metadata(card_module):
    this = card_module.this
    assert this.name == "dataPlaceholders"
    assert this.long_name == "Missing value placeholders"
    assert "placeholder" in this.description.lower()
    assert callable(this.settings)
    assert callable(this.front)
    assert callable(this.back)
    assert callable(this.footer)
    assert callable(this.server)


@pytest.mark.unit
def test_test_mode_seeds_import_data(card_module):
    """
    In test mode the module-level `this` should already have seeded data.
    """
    this = card_module.this
    with reactive.isolate():
        assert this._imports.is_set()
        pxd = this._imports.get()

    df = pxd.to_native()
    assert list(df.columns) == ["y", "x1", "x2", "id", "part"]
    assert df.shape == (4, 5)


@pytest.mark.unit
def test_front_returns_ui_object(card_module):
    tag = card_module.this.front()
    assert tag is not None


@pytest.mark.unit
def test_back_ui_contains_summary_output(card_module):
    tag = card_module.this.back()
    html = str(tag)
    assert "Placeholder Summary" in html
    assert "Summary" in html


# ---------- ui smoke tests ---------------------------------------------

@pytest.mark.ui
def test_card_renders(page: Page, app: ShinyAppProc):
    page.goto(app.url)
    expect(get_card(page)).to_be_visible()
    expect(page.get_by_text("Missing value placeholders")).to_be_visible()


@pytest.mark.ui
def test_front_has_expected_tabs(page: Page, app: ShinyAppProc):
    page.goto(app.url)
    for label in [
        "All variables",
        "Integer",
        "Decimal",
        "Character",
        "Dates & Times",
    ]:
        expect(page.get_by_role("tab", name=label)).to_be_visible()


@pytest.mark.ui
def test_default_all_chart_output_exists(page: Page, app: ShinyAppProc):
    page.goto(app.url)
    # Default tab should be "All variables"
    expect(by_id(page, "AllChart")).to_be_visible()


@pytest.mark.ui
def test_can_switch_tabs(page: Page, app: ShinyAppProc):
    page.goto(app.url)
    page.get_by_role("tab", name="Integer").click()
    expect(by_id(page, "IntegerChart")).to_be_visible()
    page.get_by_role("tab", name="Decimal").click()
    expect(by_id(page, "FloatChart")).to_be_visible()
    page.get_by_role("tab", name="Character").click()
    expect(by_id(page, "CharacterChart")).to_be_visible()
    page.get_by_role("tab", name="Dates & Times").click()
    expect(by_id(page, "DateChart")).to_be_visible()


@pytest.mark.ui
def test_replace_checkbox_group_starts_empty_for_seeded_data(page: Page, app: ShinyAppProc):
    """
    The seeded test data has no placeholder-like values under the default settings,
    so no replacement choices should be offered initially.
    """
    page.goto(app.url)
    replace_group = by_id(page, "Replace")
    expect(replace_group).to_be_visible()
    expect(replace_group.locator(".checkbox")).to_have_count(0)


# ---------- ui behavior tests ------------------------------------------

@pytest.mark.ui
def test_adding_string_placeholder_creates_replace_choice(page: Page, app: ShinyAppProc):
    """
    The seeded test data contains x2 values A/B.
    If we tell the card that "A" is a string placeholder, a replace option should appear.
    """
    page.goto(app.url)
    set_shiny_input(
        page,
        "NA_Strings",
        ["NA", "-", "--", "N/A", "Missing", "Not Applicable", "Not Available", "A"],
    )
    # Give the reactive/UI chain a moment; this card has shown timing sensitivity.
    page.wait_for_timeout(150)
    replace_group = by_id(page, "Replace")
    expect(replace_group).to_contain_text("Replace str: A", timeout=5000)


@pytest.mark.ui
def test_changing_maxobs_does_not_break_chart(page: Page, app: ShinyAppProc):
    page.goto(app.url)
    set_shiny_input(page, "MaxObs", 3)
    page.wait_for_timeout(150)
    expect(by_id(page, "AllChart")).to_be_visible()


@pytest.mark.ui
def test_flip_reveals_back_summary(page: Page, app: ShinyAppProc):
    """
    This test is intentionally simple because flip-card visibility can be
    transform/CSS-sensitive in Playwright.
    """
    page.goto(app.url)
    summary = by_id(page, "Summary")
    get_flip_button(page).click(force=True)
    page.wait_for_timeout(200)
    expect(summary).to_be_visible(timeout=5000)


@pytest.mark.ui
def test_back_summary_contains_expected_columns(page: Page, app: ShinyAppProc):
    page.goto(app.url)
    get_flip_button(page).click(force=True)
    page.wait_for_timeout(200)
    summary = by_id(page, "Summary")
    expect(summary).to_be_visible(timeout=5000)
    expect(summary).to_contain_text("Variable")
    # With default seeded data there should be at least one variable name shown
    expect(summary).to_contain_text("y")
    expect(summary).to_contain_text("x1")
    expect(summary).to_contain_text("x2")


@pytest.mark.ui
def test_fullscreen_toggle_does_not_break_card(page: Page, app: ShinyAppProc):
    page.goto(app.url)
    btn = get_fullscreen_button(page)
    btn.click(force=True)
    page.wait_for_timeout(150)
    expect(get_card(page)).to_be_visible()


# ---------- ui integration-ish test ------------------------------------

@pytest.mark.ui
def test_selecting_replace_option_updates_summary(page: Page, app: ShinyAppProc):
    """
    End-to-end-ish behavior:
      1. declare "A" as a missing string placeholder
      2. wait for a replace option to appear
      3. select that replace option
      4. confirm the back summary is still rendered
    This is intentionally conservative because the card's internal helpers
    are nested in server() and are best tested through visible behavior.
    """
    page.goto(app.url)
    set_shiny_input(
        page,
        "NA_Strings",
        ["NA", "-", "--", "N/A", "Missing", "Not Applicable", "Not Available", "A"],
    )
    page.wait_for_timeout(150)
    replace_group = by_id(page, "Replace")
    expect(replace_group).to_contain_text("Replace str: A", timeout=5000)
    set_shiny_input(page, "Replace", ["Replace str: A"])
    page.wait_for_timeout(150)
    get_flip_button(page).click(force=True)
    page.wait_for_timeout(200)
    summary = by_id(page, "Summary")
    expect(summary).to_be_visible(timeout=5000)
    expect(summary).to_contain_text("x2")