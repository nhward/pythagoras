"""Validation tests for the Pythagoras JSON configuration schema."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import ValidationError, validate

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "app"


@pytest.fixture(scope="module")
def schema() -> dict[str, object]:
    return json.loads((CONFIG_ROOT / "pythagoras.schema.json").read_text())


@pytest.fixture(scope="module")
def configuration() -> dict[str, object]:
    return json.loads((CONFIG_ROOT / "default.pythagoras.json").read_text())


@pytest.mark.unit
def test_current_configuration_satisfies_schema(schema, configuration):
    assert configuration["version"] == 2
    validate(instance=configuration, schema=schema)


@pytest.mark.unit
def test_unknown_future_configuration_version_is_rejected(schema, configuration):
    candidate = deepcopy(configuration)
    candidate["version"] = 3

    with pytest.raises(ValidationError):
        validate(instance=candidate, schema=schema)


@pytest.mark.unit
@pytest.mark.parametrize("name", ["Pre training", "Stage 2", "A"])
def test_section_names_accept_letters_numbers_and_single_spaces(
    schema, configuration, name,
):
    candidate = deepcopy(configuration)
    candidate["layout"][0]["section"] = name

    validate(instance=candidate, schema=schema)


@pytest.mark.unit
@pytest.mark.parametrize(
    "name",
    ["Pre-training", " leading", "trailing ", "two  spaces", "Pre_training"],
)
def test_invalid_section_names_are_rejected(schema, configuration, name):
    candidate = deepcopy(configuration)
    candidate["layout"][0]["section"] = name

    with pytest.raises(ValidationError, match="does not match"):
        validate(instance=candidate, schema=schema)


@pytest.mark.unit
def test_card_module_must_be_a_python_identifier(schema, configuration):
    candidate = deepcopy(configuration)
    candidate["layout"][0]["cards"][0]["module"] = "data-import"

    with pytest.raises(ValidationError, match="does not match"):
        validate(instance=candidate, schema=schema)


@pytest.mark.unit
def test_layout_is_required(schema, configuration):
    candidate = deepcopy(configuration)
    del candidate["layout"]

    with pytest.raises(ValidationError, match="'layout' is a required property"):
        validate(instance=candidate, schema=schema)


@pytest.mark.unit
def test_data_import_state_is_accepted(schema, configuration):
    candidate = deepcopy(configuration)
    candidate["layout"][0]["cards"][0]["state"] = {
        "inputs": {
            "Navset": "Web based",
            "ServerFile": None,
            "LocalFilePath": "",
            "FName": "file draft",
            "Dataset": "sklearn::iris",
            "DName": "package draft",
            "Url": "https://example.test/data.csv",
            "UName": "committed web data",
            "UciDataset": "Iris",
            "IName": "uci draft",
            "Separator": ",",
            "Sheet": 1,
        },
        "last_committed_tab": "Web based",
    }

    validate(instance=candidate, schema=schema)


@pytest.mark.unit
def test_bookmark_metadata_is_accepted(schema, configuration):
    candidate = deepcopy(configuration)
    candidate["bookmark"] = {
        "filename": "iris--20260911T120000+1200.pythagoras.json",
        "created_at": "2026-09-11T12:00:00+12:00",
    }

    validate(instance=candidate, schema=schema)


@pytest.mark.unit
def test_card_state_is_an_opaque_json_object(schema, configuration):
    candidate = deepcopy(configuration)
    candidate["layout"][0]["cards"][0]["state"] = {
        "inputs": {
            "UnknownFutureInput": "value",
            "Selections": ["alpha", "beta"],
        },
        "card_owned": {
            "nested": {"enabled": True, "threshold": 0.25},
            "nullable": None,
        },
    }

    validate(instance=candidate, schema=schema)


@pytest.mark.unit
def test_card_state_must_be_an_object(schema, configuration):
    candidate = deepcopy(configuration)
    candidate["layout"][0]["cards"][0]["state"] = ["not", "an", "object"]

    with pytest.raises(ValidationError, match="is not of type 'object'"):
        validate(instance=candidate, schema=schema)


@pytest.mark.unit
@pytest.mark.parametrize("value", [True, False, "false", 0, None])
def test_restore_last_active_section_requires_boolean(schema, configuration, value):
    candidate = deepcopy(configuration)
    candidate["settings"]["restore_last_active_section"] = value
    if isinstance(value, bool):
        validate(instance=candidate, schema=schema)
    else:
        with pytest.raises(ValidationError):
            validate(instance=candidate, schema=schema)
