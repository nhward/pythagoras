import pytest
from pathlib import Path
import sys
from shiny.run import ShinyAppProc
from shiny.pytest import create_app_fixture
from playwright.sync_api import Page, expect

# Ensure app root is importable when pytest is run outside the IDE
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roles import Role  # noqa: E402


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

# Adjust this path to wherever the card file lives
app = create_app_fixture(app = "../cards/RoleAssignment.py")

def get_card(page: Page):
    return page.locator(".card").first

def get_card_id(page: Page):
    return get_card(page).get_attribute("id")

def get_role_bucket(page: Page, role: str):
    return page.locator(selector = f'[data-role="{role}"]')

def nsWrap(page, id):
    prefix = get_ns(page)
    return f"{prefix}-{id}"

def get_ns(page: Page):
    card_id = get_card_id(page)
    return card_id.partition("-")[0]

def get_commit_button(page: Page):
    id = nsWrap(page, "Commit")
    page.locator(selector = f"#{id}")


def set_role_map_input(page: Page, payload: dict):
    page.wait_for_function("() => !!window.populateRolesHandler")
    page.evaluate(
        """
        ([cardId, payload]) => {
            window.populateRolesHandler({
                card: cardId,
                role_map: payload
            });
        }
        """,
        [get_card_id(page), payload],
    )
    page.wait_for_timeout(200)


def current_bucket_labels(page: Page, role: str):
    bucket = get_role_bucket(page, role)
    chips = bucket.locator(".var-chip")
    return chips.all_inner_texts()


# ---------- smoke tests -------------------------------------------------
@pytest.mark.ui
def test_card_renders(page: Page, app: ShinyAppProc):
    page.goto(app.url)
    # expect(page.get_by_text("Role Assignment")).to_be_visible()
    expect(page.locator('[data-role="target"]')).to_be_visible()
    expect(page.locator('[data-role="predictor"]')).to_be_visible()
    id = nsWrap(page, "Commit")
    expect(page.locator(selector = f"#{id}")).to_be_visible()


@pytest.mark.ui
def test_commit_button_starts_enabled(page: Page, app: ShinyAppProc):
    page.goto(app.url)
    id = nsWrap(page, "Commit")
    expect(page.locator(selector = f"#{id}")).to_be_enabled()


# ---------- data population tests --------------------------------------

@pytest.mark.ui
def test_roles_grid_has_all_role_buckets(page: Page, app: ShinyAppProc):
    page.goto(app.url)
    expected_roles = [r.value for r in Role]
    for role in expected_roles:
        expect(get_role_bucket(page, role)).to_be_visible()


@pytest.mark.ui
def test_default_buckets(page: Page, app: ShinyAppProc):
    page.goto(app.url)
    expect(get_role_bucket(page, "identifier").locator(".var-chip")).to_have_count(0)
    expect(get_role_bucket(page, "target").locator(".var-chip")).to_have_count(0)
    expect(get_role_bucket(page, "predictor").locator(".var-chip")).to_have_count(5)
    expect(get_role_bucket(page, "partition").locator(".var-chip")).to_have_count(0)
    expect(get_role_bucket(page, "none").locator(".var-chip")).to_have_count(0)
    id = nsWrap(page, "Commit")
    expect(page.locator(selector = f"#{id}")).to_be_enabled()



@pytest.mark.ui
def test_populate_roles_message_populates_buckets(page: Page, app: ShinyAppProc):
    page.goto(app.url)
    payload = {
        "target": ["y"],
        "predictor": ["x1", "x2"],
        "identifier": ["id"],
        "weighting": [],
        "stratifier": [],
        "geometry": [],
        "sequence": [],
        "sensitive": [],
        "partition": ["part"],
        "none": []
    }
    set_role_map_input(page, payload)
    expect(get_role_bucket(page, "target").locator(".var-chip")).to_have_count(1) 
    expect(get_role_bucket(page, "predictor").locator(".var-chip")).to_have_count(2)
    expect(get_role_bucket(page, "identifier").locator(".var-chip")).to_have_count(1)
    expect(get_role_bucket(page, "partition").locator(".var-chip")).to_have_count(1)
    expect(get_role_bucket(page, "none").locator(".var-chip")).to_have_count(0)


# ---------- role_map / validation tests --------------------------------

@pytest.mark.ui
def test_valid_role_map_enables_commit(page: Page, app: ShinyAppProc):
    page.goto(app.url)
    payload = {
        "target": ["y"],
        "predictor": ["x1", "x2"],
        "identifier": ["id"],
        "weighting": [],
        "stratifier": [],
        "geometry": [],
        "sequence": [],
        "sensitive": [],
        "partition": ["part"],
        "none": []
    }
    set_role_map_input(page, payload)
    id = nsWrap(page, "Commit")
    expect(page.locator(selector = f"#{id}")).to_be_visible()
    expect(page.locator(selector = f"#{id}")).to_be_enabled()


@pytest.mark.ui
def test_invalid_role_map_disables_commit(page: Page, app: ShinyAppProc):
    page.goto(app.url)
    # Two targets should violate the singleton rule
    payload = {
        "target": ["y", "x1"],
        "predictor": ["x2"],
        "identifier": ["id"],
        "weighting": [],
        "stratifier": [],
        "geometry": [],
        "sequence": [],
        "sensitive": [],
        "partition": ["part"],
        "none": []
    }
    set_role_map_input(page, payload)
    id = nsWrap(page, "Commit")
    expect(page.locator(selector = f"#{id}")).to_be_disabled()

@pytest.mark.ui
def test_check_output_shows_ready_or_error(page: Page, app: ShinyAppProc):
    page.goto(app.url)
    payload = {
        "target": ["y"],
        "predictor": ["x1", "x2"],
        "identifier": ["id"],
        "weighting": [],
        "stratifier": [],
        "geometry": [],
        "sequence": [],
        "sensitive": [],
        "partition": ["part"],
        "none": []
    }
    set_role_map_input(page, payload)
    expect(page.get_by_text("Assignments ready to commit")).to_be_visible()


# ---------- commit tests ------------------------------------------------

@pytest.mark.ui
def test_clicking_commit_applies_assignments(page: Page, app: ShinyAppProc):
    page.goto(app.url)
    payload = {
        "target": ["y"],
        "predictor": ["x1", "x2"],
        "identifier": ["id"],
        "weighting": [],
        "stratifier": [],
        "geometry": [],
        "sequence": [],
        "sensitive": [],
        "partition": ["part"],
        "none": []
    }
    set_role_map_input(page, payload)
    id = nsWrap(page, "Commit")
    btn = page.locator(selector = f"#{id}")
    expect(btn).to_be_enabled()
    btn.click()
    expect(page.get_by_text("Assignments applied")).to_be_visible()


# ---------- back-side table tests --------------------------------------

@pytest.mark.ui
def test_back_side_assignments_table_becomes_visible(page: Page, app: ShinyAppProc):
    page.goto(app.url)
    tid = nsWrap(page, "Assignments")
    table = page.locator(selector = f"#{tid}")
    expect(table).to_be_attached()
    fid = nsWrap(page, "FlipButton")
    page.locator(selector = f"#{fid}").click(force = True)  # Because hidden by lack of hover
    expect(table).to_be_visible()
