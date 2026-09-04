"""Validation tests for the Pythagoras JSON configuration schema."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from jsonschema import ValidationError, validate


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "app" / "config"


@pytest.fixture(scope="module")
def schema() -> dict[str, object]:
    return json.loads((CONFIG_ROOT / "pythagoras.schema.json").read_text())


@pytest.fixture(scope="module")
def configuration() -> dict[str, object]:
    return json.loads((CONFIG_ROOT / "pythagoras.json").read_text())


@pytest.mark.unit
def test_current_configuration_satisfies_schema(schema, configuration):
    validate(instance=configuration, schema=schema)


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
