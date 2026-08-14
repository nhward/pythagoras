import importlib
from pathlib import Path
from types import SimpleNamespace

import geopandas as gpd
import pandas as pd
import pytest
from playwright.sync_api import Page, expect
from shapely.geometry import Point
from shiny.pytest import create_app_fixture
from shiny.run import ShinyAppProc

from proxyData import ProxyData

app = create_app_fixture(app="../../cards/DataImport.py", scope="function")
_HELPER_CARDS = {}


@pytest.fixture(scope="session")
def browser_context_args():
    return {"viewport": {"width": 1600, "height": 1000}}


@pytest.fixture(scope="session", autouse=True)
def deterministic_dataset_catalogues():
    """Keep unit collection independent of remote catalogue services."""
    import seaborn
    import ucimlrepo

    original_seaborn = seaborn.get_dataset_names
    original_uci = ucimlrepo.list_available_datasets
    seaborn.get_dataset_names = lambda: ["tips", "iris"]

    def list_uci():
        print("Iris 53")
        print("Wine Quality 186")

    ucimlrepo.list_available_datasets = list_uci
    yield
    seaborn.get_dataset_names = original_seaborn
    ucimlrepo.list_available_datasets = original_uci


@pytest.fixture
def card_module(deterministic_dataset_catalogues):
    return importlib.import_module("cards.DataImport")


@pytest.fixture
def csv_file(tmp_path):
    path = tmp_path / "observations.csv"
    path.write_text("id,value,group\n1,10.25,A\n2,20.75,B\n", encoding="utf-8")
    return path


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


def set_shiny_input(page: Page, local_id: str, value):
    page.evaluate(
        """
        ([inputId, value]) => {
            if (!window.Shiny || !window.Shiny.setInputValue) {
                throw new Error("Shiny.setInputValue not available");
            }
            window.Shiny.setInputValue(inputId, value, {priority: "event"});
        }
        """,
        [namespaced_id(page, local_id), value],
    )


def recorded_helpers(
    card_module,
    *,
    navset="File based",
    file_path=None,
    separator=",",
    sheet=1,
    dataset=None,
    url="",
    uci_dataset=None,
):
    """Expose the server's pure helpers through inert reactive decorators."""
    card = _HELPER_CARDS.get(card_module)
    if card is None:
        card = card_module.instance()
        _HELPER_CARDS[card_module] = card
    functions = {}

    def record(function):
        functions[function.__name__] = function
        return function

    card.record_code = record
    card.suspendable = lambda **kwargs: record
    card.throttle = lambda *args, **kwargs: lambda function: function
    card.isFullScreen = lambda: False
    uploaded = None if file_path is None else [{
        "name": Path(file_path).name,
        "datapath": str(file_path),
        "size": Path(file_path).stat().st_size,
        "type": "text/csv",
    }]
    inputs = SimpleNamespace(
        ServerFile=lambda: uploaded,
        Navset=lambda: navset,
        Separator=lambda: separator,
        Sheet=lambda: sheet,
        Dataset=lambda: dataset,
        UciDataset=lambda: uci_dataset,
        Url=lambda: url,
        FName=lambda: "file-data",
        DName=lambda: "package-data",
        UName=lambda: "web-data",
        IName=lambda: "uci-data",
        Commit=lambda: 0,
    )
    session = SimpleNamespace(
        ns=lambda value: value,
        send_custom_message=lambda *args, **kwargs: None,
    )
    card.server(inputs, lambda function: function, session)
    return card, functions


class TestCaptureOutput:
    @pytest.mark.unit
    def test_captures_stdout_and_return_is_ignored(self, card_module):
        def function(value):
            print(f"value={value}")
            return "ignored"

        assert card_module.capture_output(function, 3) == "value=3\n"

    @pytest.mark.unit
    def test_forwards_keyword_arguments(self, card_module):
        def function(*, label):
            print(label)

        assert card_module.capture_output(function, label="ready") == "ready\n"


class TestInstance:
    @pytest.mark.unit
    def test_metadata(self, card_module):
        card = card_module.this
        assert card.name == "dataImport"
        assert card.long_name == "Data import"
        assert "ingestion of data" in card.description
        assert card.mutable
        assert not card.requires_import

    @pytest.mark.unit
    def test_expected_ui_regions(self, card_module):
        card = card_module.this
        assert card.front is not None
        assert card.back is not None
        assert card.footer is not None
        assert card.settings is not None
        assert card.hasFlipSide()
        assert card.hasFooter()
        assert card.hasSidebar()

    @pytest.mark.unit
    def test_front_contains_four_import_modes(self, card_module):
        html = str(card_module.this.front.tagify())
        for label in ("File based", "Dataset based", "Web based", "UC Irvine"):
            assert label in html
        for input_id in ("ServerFile", "Dataset", "Url", "UciDataset"):
            assert f'id="{input_id}"' in html

    @pytest.mark.unit
    def test_footer_contains_disabled_commit_and_status(self, card_module):
        html = str(card_module.this.footer)
        assert 'id="Commit"' in html
        assert "Commit Import" in html
        assert "disabled" in html
        assert 'id="Check"' in html

    @pytest.mark.unit
    def test_settings_defaults(self, card_module):
        html = str(card_module.this.settings)
        assert 'id="Separator"' in html
        assert 'value=","' in html or 'value=","' in html.replace("'", '"')
        assert 'id="Sheet"' in html
        assert 'value="1"' in html


class TestFileHelpers:
    @pytest.mark.unit
    def test_temp_file_path_none_without_upload(self, card_module):
        _, functions = recorded_helpers(card_module)
        assert functions["TempFilePath"]() is None

    @pytest.mark.unit
    def test_temp_file_path_rejects_missing_path(self, card_module, tmp_path):
        path = tmp_path / "missing.csv"
        # Re-register with a file that disappears after metadata is constructed.
        path.write_text("a\n1\n")
        _, functions = recorded_helpers(card_module, file_path=path)
        path.unlink()
        assert functions["TempFilePath"]() is None

    @pytest.mark.unit
    def test_csv_import(self, card_module, csv_file):
        _, functions = recorded_helpers(card_module, file_path=csv_file)
        result = functions["GetData"]()
        assert isinstance(result, pd.DataFrame)
        assert result.columns.tolist() == ["id", "value", "group"]
        assert result.shape == (2, 3)
        assert result["value"].tolist() == [10.25, 20.75]

    @pytest.mark.unit
    def test_tsv_import_with_explicit_separator(self, card_module, tmp_path):
        path = tmp_path / "values.tsv"
        path.write_text("id\tvalue\n1\talpha\n2\tbeta\n", encoding="utf-8")
        _, functions = recorded_helpers(
            card_module, file_path=path, separator="\t"
        )
        result = functions["GetData"]()
        assert result.to_dict("list") == {"id": [1, 2], "value": ["alpha", "beta"]}

    @pytest.mark.unit
    def test_json_import(self, card_module, tmp_path):
        path = tmp_path / "values.json"
        path.write_text('[{"id":1,"value":"a"},{"id":2,"value":"b"}]')
        _, functions = recorded_helpers(card_module, file_path=path)
        result = functions["GetData"]()
        assert result.to_dict("records") == [
            {"id": 1, "value": "a"}, {"id": 2, "value": "b"}
        ]

    @pytest.mark.unit
    def test_parquet_import(self, card_module, tmp_path):
        path = tmp_path / "values.parquet"
        expected = pd.DataFrame({"id": [1, 2], "value": [1.5, 2.5]})
        expected.to_parquet(path, index=False)
        _, functions = recorded_helpers(card_module, file_path=path)
        pd.testing.assert_frame_equal(functions["GetData"](), expected)

    @pytest.mark.unit
    def test_csv_wkt_geometry_is_promoted(self, card_module, tmp_path):
        path = tmp_path / "points.csv"
        path.write_text('id,geometry\n1,"POINT (1 2)"\n2,"POINT (3 4)"\n')
        _, functions = recorded_helpers(card_module, file_path=path)
        result = functions["GetData"]()
        assert isinstance(result, gpd.GeoDataFrame)
        assert result.geometry.iloc[0].equals(Point(1, 2))
        assert result.geometry.iloc[1].equals(Point(3, 4))

    @pytest.mark.unit
    def test_unsupported_extension_raises(self, card_module, tmp_path):
        path = tmp_path / "values.txt"
        path.write_text("value\n1\n")
        _, functions = recorded_helpers(card_module, file_path=path)
        with pytest.raises(ValueError, match="Unsupported file extension: .txt"):
            functions["GetData"]()

    @pytest.mark.unit
    def test_get_data_returns_none_without_file(self, card_module):
        _, functions = recorded_helpers(card_module)
        assert functions["GetData"]() is None

    @pytest.mark.unit
    def test_get_proxy_data_wraps_native_frame(self, card_module, csv_file):
        _, functions = recorded_helpers(card_module, file_path=csv_file)
        result = functions["GetPxyData"]()
        assert isinstance(result, ProxyData)
        assert result.shape == (2, 3)
        assert result.to_native().columns.tolist() == ["id", "value", "group"]

    @pytest.mark.unit
    def test_dataset_import_from_sklearn(self, card_module):
        _, functions = recorded_helpers(
            card_module, navset="Dataset based", dataset="sklearn::iris"
        )
        result = functions["GetData"]()
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (150, 5)
        assert "target" in result.columns


class TestSummary:
    @pytest.mark.unit
    def test_dataframe_summary_contains_shape_types_and_variables(
        self, card_module, csv_file
    ):
        _, functions = recorded_helpers(card_module, file_path=csv_file)
        result = functions["Summary"]()
        html = str(result)
        assert "Dataset info" in html
        assert "DataFrame" in html
        assert "Columns" in html
        assert "Distinct Data Types" in html
        assert "Variables" in html
        assert "group" in html
        assert "Memory usage" in html

    @pytest.mark.unit
    def test_geodataframe_summary_contains_geometry_details(
        self, card_module, tmp_path
    ):
        path = tmp_path / "points.csv"
        path.write_text('id,geometry\n1,"POINT (1 2)"\n')
        _, functions = recorded_helpers(card_module, file_path=path)
        html = str(functions["Summary"]())
        assert "Geometry" in html
        assert "Point" in html
        assert "Bounds" in html


class TestWebKitInitialState:
    @pytest.mark.ui
    def test_card_and_file_tab_render(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        expect(get_card(page)).to_be_visible()
        expect(page.get_by_text("Data import", exact=True)).to_be_visible()
        expect(page.get_by_role("tab", name="File based", exact=True)).to_be_visible()
        expect(by_id(page, "ServerFile")).to_be_attached()
        expect(by_id(page, "FName")).to_be_visible()

    @pytest.mark.ui
    def test_four_import_tabs_are_available(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        for label in ("File based", "Dataset based", "Web based", "UC Irvine"):
            expect(page.get_by_role("tab", name=label, exact=True)).to_be_visible()

    @pytest.mark.ui
    def test_initial_status_and_commit_state(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        expect(by_id(page, "Check")).to_contain_text("No file supplied yet")
        expect(by_id(page, "Commit")).to_be_disabled()

    @pytest.mark.ui
    def test_settings_inputs_are_attached(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        expect(by_id(page, "Separator")).to_be_attached()
        expect(by_id(page, "Sheet")).to_be_attached()


class TestWebKitFileWorkflow:
    @pytest.mark.ui
    def test_upload_sets_short_name_and_ready_status(
        self, page: Page, app: ShinyAppProc, csv_file
    ):
        page.goto(app.url)
        by_id(page, "ServerFile").set_input_files(str(csv_file))
        expect(by_id(page, "FName")).to_have_value("observations")
        expect(by_id(page, "Check")).to_contain_text("File import ready")
        expect(by_id(page, "Check")).to_contain_text("Obs = 2, Vars = 3")
        expect(by_id(page, "Commit")).to_be_enabled()

    @pytest.mark.ui
    def test_upload_summary_on_reverse_side(
        self, page: Page, app: ShinyAppProc, csv_file
    ):
        page.goto(app.url)
        by_id(page, "ServerFile").set_input_files(str(csv_file))
        expect(by_id(page, "Check")).to_contain_text("File import ready")
        get_card(page).hover()
        by_id(page, "FlipButton").click(force=True)
        summary = by_id(page, "Summary")
        expect(summary).to_be_visible()
        expect(summary).to_contain_text("Dataset info")
        expect(summary).to_contain_text("group")

    @pytest.mark.ui
    def test_commit_reports_success(
        self, page: Page, app: ShinyAppProc, csv_file
    ):
        page.goto(app.url)
        by_id(page, "ServerFile").set_input_files(str(csv_file))
        commit = by_id(page, "Commit")
        expect(commit).to_be_enabled()
        commit.click()
        expect(by_id(page, "Check")).to_contain_text("File import successful")

    @pytest.mark.ui
    def test_dataset_tab_has_package_selector(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        page.get_by_role("tab", name="Dataset based", exact=True).click()
        expect(by_id(page, "Dataset")).to_be_attached()
        expect(by_id(page, "DName")).to_be_visible()
        expect(by_id(page, "DName")).not_to_have_value("")
        expect(by_id(page, "Check")).to_contain_text("Dataset import ready")
        expect(by_id(page, "Commit")).to_be_enabled()

    @pytest.mark.ui
    def test_close_confirmation_removes_card(self, page: Page, app: ShinyAppProc):
        page.goto(app.url)
        get_card(page).hover()
        by_id(page, "CloseButton").click(force=True)
        dialog = page.get_by_role("dialog")
        expect(dialog).to_be_visible()
        dialog.get_by_role("button", name="Yes, remove").click()
        expect(page.locator('#cards-container > [id$="Card"]')).to_have_count(0)
