import importlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

APP_ROOT = Path(__file__).resolve().parents[2] / "app"
os.chdir(APP_ROOT)
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


@pytest.fixture
def bookmark_module():
    return importlib.import_module("cards.sys_bookmark")


@pytest.fixture
def configuration():
    return {
        "version": 1,
        "settings": {
            "section_style": "tab",
            "show_start": False,
            "reuse_cards": True,
            "max_card_height": "450px",
            "max_dupl_cards": 10,
        },
        "layout": [
            {
                "section": "Data prep",
                "cards": [
                    {
                        "module": "data_import",
                        "state": {
                            "inputs": {
                                "FName": "Customer accounts",
                                "DName": "iris",
                            },
                            "last_committed_tab": "File based",
                        },
                    }
                ],
            }
        ],
    }


@pytest.mark.unit
def test_filename_uses_data_name_and_timestamp(
    bookmark_module, configuration
):
    created = datetime(2026, 9, 11, 14, 42, 8, tzinfo=timezone.utc)

    assert bookmark_module.bookmark_filename(configuration, created) == (
        "Customer-accounts--20260911T144208+0000.pythagoras.json"
    )


@pytest.mark.unit
def test_filename_parts_round_trip(bookmark_module):
    filename = "Assmnt--20260911T154501+1200.pythagoras.json"

    data_name, timestamp = bookmark_module.split_bookmark_filename(filename)

    assert data_name == "Assmnt"
    assert timestamp == "20260911T154501+1200"
    assert bookmark_module.bookmark_filename_from_parts(
        data_name, timestamp
    ) == filename


@pytest.mark.unit
def test_local_catalogue_exposes_data_name_and_local_time(
    bookmark_module, tmp_path
):
    filename = "Assmnt--20260911T154501+1200.pythagoras.json"
    (tmp_path / filename).write_text("{}", encoding="utf-8")

    records = bookmark_module.list_local_bookmarks(tmp_path)

    assert records[0]["data_name"] == "Assmnt"
    assert records[0]["timestamp"] == "20260911T154501+1200"
    assert records[0]["display_time"] == datetime.strptime(
        "20260911T154501+1200", "%Y%m%dT%H%M%S%z"
    ).astimezone().strftime("%c")


@pytest.mark.unit
def test_local_save_refuses_to_replace_existing_file(
    bookmark_module, configuration, tmp_path
):
    created = datetime(2026, 9, 11, 14, 42, 8, tzinfo=timezone.utc)
    filename = bookmark_module.bookmark_filename(configuration, created)
    candidate = bookmark_module.with_bookmark_metadata(
        configuration,
        filename=filename,
        created_at=created,
    )
    existing = tmp_path / filename
    existing.write_text("original", encoding="utf-8")

    with pytest.raises(FileExistsError):
        bookmark_module.save_local_bookmark(candidate, directory=tmp_path)

    assert existing.read_text(encoding="utf-8") == "original"


@pytest.mark.unit
def test_latest_local_bookmark_uses_creation_order(
    bookmark_module, configuration, tmp_path, monkeypatch
):
    older = tmp_path / f"older{bookmark_module.BOOKMARK_SUFFIX}"
    newer = tmp_path / f"newer{bookmark_module.BOOKMARK_SUFFIX}"
    older.write_text(json.dumps({**configuration, "selected": "older"}))
    newer.write_text(json.dumps({**configuration, "selected": "newer"}))
    monkeypatch.setattr(
        bookmark_module,
        "filesystem_creation_time",
        lambda path: 2 if path.name == newer.name else 1,
    )

    loaded = bookmark_module.latest_local_bookmark(directory=tmp_path)

    assert loaded["selected"] == "newer"


@pytest.mark.unit
def test_invalid_newest_bookmark_is_skipped(
    bookmark_module, configuration, tmp_path, monkeypatch
):
    invalid = tmp_path / f"invalid{bookmark_module.BOOKMARK_SUFFIX}"
    valid = tmp_path / f"valid{bookmark_module.BOOKMARK_SUFFIX}"
    invalid.write_text("not JSON", encoding="utf-8")
    valid.write_text(json.dumps(configuration), encoding="utf-8")
    monkeypatch.setattr(
        bookmark_module,
        "filesystem_creation_time",
        lambda path: 2 if path.name == invalid.name else 1,
    )

    assert bookmark_module.latest_local_bookmark(directory=tmp_path) == configuration


@pytest.mark.unit
def test_manager_card_cannot_be_removed_or_dragged(bookmark_module):
    card = bookmark_module.instance()

    assert card.long_name == "Bookmarks"
    assert card.allow_remove is False
    assert card.allow_drag is False
    markup = str(card.call_ui())
    assert "CloseButton" not in markup
    assert "drag-handle" not in markup
    assert "Save bookmark" in markup
