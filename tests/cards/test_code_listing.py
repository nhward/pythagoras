"""The Code modal exposes executed calculations across card families."""
import re
import time
from pathlib import Path

import pytest
from playwright.sync_api import expect
from shiny.pytest import create_app_fixture

# Each fixture runs the normal card server with its existing deterministic data.
CASES = [
    ("data_import", "read_file"),
    ("data_coverage", "_figure"),
    ("data_homogeneous", "_homogeneity_figure"),
    ("data_parallel", "_parallel_figure"),
    ("data_provenance", "_journey_figure"),
    ("data_strata", "_figure"),
    ("data_tabulation", "_clean_frame"),
    ("miss_impute", "ImputationStep"),
    ("miss_informative", "_importance_figure"),
    ("miss_map", "_missingness_figure"),
    ("miss_placeholders", "PlaceholderCodes"),
    ("miss_rules", "_association_rules"),
    ("miss_sets", "_upset_figure"),
    ("miss_type", "_tree_figure"),
    ("obs_cluster_profile", "_analyze"),
    ("obs_clusters", "_figure"),
    ("obs_depence", "_figure"),
    ("obs_duplicates", "_duplicates_figure"),
    ("obs_k_clusters", "_figure"),
    ("sys_configuration", "get_loaded_packages"),
    ("system_log", "_log_frame"),
    ("var_cardinality", "_cardinality_figure"),
    ("var_correlation", "_chord_figure"),
    ("var_dissimilar", "_dissimilarity_matrix"),
    ("var_encode", "NominalEncodingTransformer"),
    ("var_modify", "is_numeric_like"),
    ("var_pairs", "_build"),
    ("var_roles", "_validate_roles"),
    ("var_text_encode", "TextEncodingTransformer"),
    ("var_time_encode", "TimeEncodingTransformer"),
    ("var_transform", "VariableTransformStep"),
]
for scenario, _ in CASES:
    app_file = "data_import_server" if scenario == "data_import" else scenario
    globals()["code_app_" + scenario] = create_app_fixture(
        app=str(Path(__file__).resolve().parents[1] / "scenarios" / f"{app_file}.py"),
        scope="function",
    )


@pytest.mark.ui
@pytest.mark.parametrize("scenario, expected", CASES)
def test_executed_code_modal(page, request, tmp_path, scenario, expected):
    app = request.getfixturevalue("code_app_" + scenario)
    page.goto(app.url)
    card = page.locator(".card").first
    expect(card).to_be_visible()
    if scenario == "data_import":
        csv = tmp_path / "source.csv"
        csv.write_text("x,y\n1,2\n3,4\n")
        card.locator('[id$="-ServerFile"]').set_input_files(str(csv))
    elif scenario == "miss_placeholders":
        expect(card.locator(".js-plotly-plot").first).to_be_visible(timeout=60000)
    elif scenario == "var_transform":
        card.locator('[id$="-Transform"] input[value="Scale"]').check(force=True)
    button = card.locator('[id$="-CodeButton"]')
    deadline = time.monotonic() + 60

    def remaining_ms():
        return max(1, int((deadline - time.monotonic()) * 1000))

    # A visible card can precede Shiny input binding during startup. A forced
    # click at that point is lost rather than queued for the server.
    expect(button).to_have_class(re.compile(r"\bshiny-bound-input\b"), timeout=remaining_ms())
    while True:
        card.hover(timeout=remaining_ms())
        button.click(timeout=remaining_ms())
        dialog = page.get_by_role("dialog")
        expect(dialog).to_be_visible(timeout=remaining_ms())
        headings = dialog.locator("h3").all_text_contents()
        if "# " + expected in headings or time.monotonic() >= deadline:
            break
        # The listing is a snapshot. Reopen it while background work completes.
        dialog.get_by_role("button", name="Dismiss", exact=True).click(timeout=remaining_ms())
        expect(dialog).to_have_count(0, timeout=remaining_ms())
        page.wait_for_timeout(350)
    expect(dialog.locator("h3", has_text="# " + expected)).to_be_visible()
    listing = dialog.locator("pre").all_text_contents()
    assert listing
    for source in listing:
        assert "@recordable" not in source
        assert "@this.record_code" not in source
        assert "@this.record_context" not in source
        assert "@this.extended_task" not in source
        assert "input." not in source
        compile(source, "<code modal>", "exec")
