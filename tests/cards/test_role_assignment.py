from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

path = Path(__file__).resolve().parent.parent.parent / 'app'
os.chdir(path)
if str(path) not in sys.path:
    sys.path.insert(0, str(path))


import pandas as pd
import pytest
from playwright.sync_api import Page, expect
from proxy_data import proxy_data
from roles import Role, RoleMap
from shiny import reactive
from shiny.playwright import controller
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

app = create_app_fixture(app="../../app/cards/role_assignment.py", scope="function")
_HELPER_CARDS = {}

VALID_ROLE_MAP = {
    "target": ["y"],
    "predictor": ["x1", "x2"],
    "identifier": ["id"],
    "weighting": [],
    "stratifier": [],
    "treatment": [],
    "geometry": [],
    "sequence": [],
    "sensitive": [],
    "partition": ["part"],
    "none": [],
}


@pytest.fixture(scope="session")
def browser_context_args():
    return {"viewport": {"width": 1600, "height": 1000}}


@pytest.fixture
def card_module():
    return importlib.import_module("cards.role_assignment")


def seeded_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "y": [1, 0, 1, 0],
        "x1": [10.0, 11.0, 12.0, 13.0],
        "x2": ["A", "B", "A", "B"],
        "id": [100, 101, 102, 103],
        "part": ["Train", "Train", "Test", "Test"],
    })


class FakeInputs:
    def __init__(self, role_map=None, *, separator="|", cardinality=4, max_obs=3):
        self.current_role_map = role_map or VALID_ROLE_MAP
        self.separator = separator
        self.cardinality = cardinality
        self.max_obs = max_obs

    def role_map(self):
        return self.current_role_map

    def Separator(self):
        return self.separator

    def CardinalityThreshold(self):
        return self.cardinality

    def MaxObs(self):
        return self.max_obs

    def Commit(self):
        return 0


def recorded_helpers(card_module, *, frame=None, role_map=None, max_obs=3):
    """Expose nested server helpers through inert test decorators."""
    card = _HELPER_CARDS.get(card_module)
    if card is None:
        card = card_module.instance()
        _HELPER_CARDS[card_module] = card
    functions = {}

    def capture(function):
        functions[function.__name__] = function
        return function

    card.record_code = capture
    card.suspendable = lambda **kwargs: capture
    card.throttle = lambda *args, **kwargs: capture
    card.isFullScreen = lambda: False
    source = seeded_frame() if frame is None else frame
    proxy = source if isinstance(source, proxy_data) else proxy_data(_df=source, _name="Test")
    with reactive.isolate():
        card._imports.set(proxy)
    inputs = FakeInputs(role_map, max_obs=max_obs)
    card.server(inputs, lambda function: function, None)
    return card, inputs, functions


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


def role_bucket(page: Page, role: Role | str):
    value = role.value if isinstance(role, Role) else role
    return get_card(page).locator(f'[data-role="{value}"]')


def populate_roles(page: Page, payload: dict[str, list[str]]):
    page.wait_for_function("() => !!window.populateRolesHandler")
    expect(get_card(page).locator(".var-chip")).to_have_count(5)
    expect(by_id(page, "Commit")).to_be_enabled()
    expect(by_id(page, "Check")).to_contain_text("Assignments applied")
    page.evaluate(
        """
        ([cardId, roleMap]) => window.populateRolesHandler({
            card: cardId,
            role_map: roleMap
        })
        """,
        [get_card(page).get_attribute("id"), payload],
    )


class TestInstance:
    @pytest.mark.unit
    def test_metadata(self, card_module):
        card = card_module.this
        assert card.name == "role_assignment"
        assert card.long_name == "Role Assignment"
        assert "assigned to roles" in card.description
        assert card.mutable

    @pytest.mark.unit
    def test_expected_ui_regions(self, card_module):
        card = card_module.this
        assert card.front is not None
        assert card.back is not None
        assert card.settings is not None
        assert card.footer is not None
        assert card.hasFlipSide()
        assert card.hasSidebar()
        assert card.hasFooter()

    @pytest.mark.unit
    def test_front_contains_every_role_bucket(self, card_module):
        html = str(card_module.this.front)
        for role in Role:
            assert f'data-role="{role.value}"' in html
            assert f'id="role-{role.value}"' in html

    @pytest.mark.unit
    def test_settings_contain_expected_controls(self, card_module):
        html = str(card_module.this.settings)
        for control in ("Separator", "CardinalityThreshold", "MaxObs"):
            assert f'id="{control}"' in html

    @pytest.mark.unit
    def test_footer_contains_commit_and_status(self, card_module):
        html = str(card_module.this.footer)
        assert 'id="Commit"' in html
        assert 'id="Check"' in html
        assert "Commit Assignments" in html

    @pytest.mark.unit
    def test_test_mode_seeds_predictor_roles(self, card_module):
        with reactive.isolate():
            proxy = card_module.this._imports.get()
        assert proxy.to_native().equals(seeded_frame())
        assert proxy.name == "Test"
        assert proxy.role_map.columns_with_role(Role.PREDICTOR) == {
            "y", "x1", "x2", "id", "part"
        }


class TestServerHelpers:
    @pytest.mark.unit
    def test_max_observations_uses_logarithmic_input(self, card_module):
        _, _, functions = recorded_helpers(card_module, max_obs=5)
        assert functions["MaxObs"]() == 100_000

    @pytest.mark.unit
    def test_prepared_data_samples_at_configured_limit(self, card_module):
        frame = pd.DataFrame({"value": range(1500)})
        _, _, functions = recorded_helpers(card_module, frame=frame, max_obs=3)
        with reactive.isolate():
            result = functions["PreparedData"]()
        assert isinstance(result, proxy_data)
        assert result.shape == (1000, 1)
        assert result.to_native().index.is_monotonic_increasing

    @pytest.mark.unit
    def test_validate_map_accepts_valid_assignment(self, card_module):
        _, _, functions = recorded_helpers(card_module, role_map=VALID_ROLE_MAP)
        with reactive.isolate():
            assert functions["ValidateMap"]() == []

    @pytest.mark.unit
    def test_validate_map_rejects_two_targets(self, card_module):
        invalid = {key: value.copy() for key, value in VALID_ROLE_MAP.items()}
        invalid["target"] = ["y", "x1"]
        invalid["predictor"] = ["x2"]
        _, _, functions = recorded_helpers(card_module, role_map=invalid)
        with reactive.isolate():
            messages = functions["ValidateMap"]()
        assert "Target role must be singular" in messages

    @pytest.mark.unit
    def test_validate_map_rejects_missing_predictor(self, card_module):
        invalid = {key: value.copy() for key, value in VALID_ROLE_MAP.items()}
        invalid["predictor"] = []
        invalid["none"] = ["x1", "x2"]
        _, _, functions = recorded_helpers(card_module, role_map=invalid)
        with reactive.isolate():
            messages = functions["ValidateMap"]()
        assert "Predictor role must be assigned to a variable" in messages

    @pytest.mark.unit
    def test_committed_proxy_preserves_data_name_and_roles(self, card_module):
        _card, _, functions = recorded_helpers(card_module, role_map=VALID_ROLE_MAP)
        with reactive.isolate():
            result = functions["Committed"]()
        assert isinstance(result, proxy_data)
        assert result.to_native().equals(seeded_frame())
        assert result.name == "Test"
        assert result.role_map == RoleMap.from_primitive(VALID_ROLE_MAP)

    @pytest.mark.unit
    def test_commit_event_sets_export(self, card_module):
        _card, _, functions = recorded_helpers(card_module, role_map=VALID_ROLE_MAP)
        with reactive.isolate():
            functions["CommitEvent"]()
            exported = _card._exports.get()
        assert exported.role_map == RoleMap.from_primitive(VALID_ROLE_MAP)

    @pytest.mark.unit
    def test_assignments_output_has_one_row_per_role(self, card_module):
        card, _, functions = recorded_helpers(card_module, role_map=VALID_ROLE_MAP)
        with reactive.isolate():
            card._exports.set(functions["Committed"]())
            result = functions["Assignments"]()
        assert result.columns.tolist() == ["Role", "Variable"]
        assert len(result) == len(Role)
        assert result["Role"].tolist() == [role.value.title() for role in Role]
        target = result.loc[result["Role"] == "Target", "Variable"].item()
        assert target == "y"


class TestWebKitRoles:
    @pytest.mark.ui
    def test_card_and_all_role_buckets_render(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        expect(get_card(page)).to_be_visible()
        expect(page.get_by_text("Role Assignment", exact=True)).to_be_visible()
        for role in Role:
            expect(role_bucket(page, role)).to_be_visible()

    @pytest.mark.ui
    def test_default_data_are_all_predictors(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        chips = role_bucket(page, Role.PREDICTOR).locator(".var-chip")
        expect(chips).to_have_count(5)
        assert set(chips.all_inner_texts()) == {"y", "x1", "x2", "id", "part"}
        for role in Role:
            if role is not Role.PREDICTOR:
                expect(role_bucket(page, role).locator(".var-chip")).to_have_count(0)

    @pytest.mark.ui
    def test_valid_role_map_populates_expected_buckets(
        self, page: Page, app: ShinyAppProc
    ):
        page.goto(app.url)
        populate_roles(page, VALID_ROLE_MAP)
        expect(role_bucket(page, "target").locator(".var-chip")).to_have_text(["y"])
        expect(role_bucket(page, "predictor").locator(".var-chip")).to_have_count(2)
        expect(role_bucket(page, "identifier").locator(".var-chip")).to_have_text(["id"])
        expect(role_bucket(page, "partition").locator(".var-chip")).to_have_text(["part"])

    @pytest.mark.ui
    def test_valid_role_map_enables_commit_and_ready_status(
        self, page: Page, app: ShinyAppProc
    ):
        page.goto(app.url)
        populate_roles(page, VALID_ROLE_MAP)
        expect(by_id(page, "Commit")).to_be_enabled()
        expect(by_id(page, "Check")).to_contain_text("Assignments ready to commit")

    @pytest.mark.ui
    def test_invalid_role_map_disables_commit_and_shows_error(
        self, page: Page, app: ShinyAppProc
    ):
        page.goto(app.url)
        invalid = {key: value.copy() for key, value in VALID_ROLE_MAP.items()}
        invalid["target"] = ["y", "x1"]
        invalid["predictor"] = ["x2"]
        populate_roles(page, invalid)
        expect(by_id(page, "Commit")).to_be_disabled()
        expect(by_id(page, "Check")).to_contain_text("Target role must be singular")

    @pytest.mark.ui
    def test_commit_applies_valid_assignments(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        populate_roles(page, VALID_ROLE_MAP)
        commit = by_id(page, "Commit")
        expect(commit).to_be_enabled()
        commit.click()
        expect(by_id(page, "Check")).to_contain_text("Assignments applied")

    @pytest.mark.ui
    def test_back_table_shows_committed_assignments(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        populate_roles(page, VALID_ROLE_MAP)
        by_id(page, "Commit").click()
        get_card(page).hover()
        by_id(page, "FlipButton").click(force=True)
        table = controller.OutputTable(page, namespaced_id(page, "Assignments"))
        table.expect_ncol(2)
        table.expect_column_labels(["Role", "Variable"])
        expect(by_id(page, "Assignments")).to_contain_text("Target")
        expect(by_id(page, "Assignments")).to_contain_text("y")

    @pytest.mark.ui
    def test_settings_controls_are_attached(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        expect(by_id(page, "Separator")).to_be_attached()
        expect(by_id(page, "CardinalityThreshold")).to_be_attached()
        expect(by_id(page, "MaxObs")).to_be_attached()

    @pytest.mark.ui
    def test_expand_and_restore_keep_roles_available(
        self, page: Page, app: ShinyAppProc
    ):
        page.goto(app.url)
        get_card(page).hover()
        by_id(page, "ExpandButton").click(force=True)
        restore = by_id(page, "ContractButton")
        expect(restore).to_be_visible()
        restore.click(force=True)
        expect(role_bucket(page, Role.PREDICTOR)).to_be_visible()
