"""Dynamic bookmark choices survive delayed upstream role/type restoration."""
import pytest
from selection_restore import SelectionRestore

pytestmark = pytest.mark.unit


def test_partial_restoration_waits_for_remaining_columns():
    state = SelectionRestore(["x", "y"])
    assert state.resolve([], ["x"], ["x"]) == ["x"]
    assert state.resolve(["x"], ["x"], ["x"]) == ["x"]
    assert state.resolve(["x"], ["x", "y"], ["x"]) == ["x", "y"]
    assert state.resolve(["y"], ["x", "y"], ["x"]) == ["y"]


def test_colour_arrives_late():
    state = SelectionRestore(["group"])
    assert state.resolve([], ["score"], []) == []
    assert state.resolve([], ["score", "group"], []) == ["group"]


def test_manual_change_overrides_pending_bookmark():
    state = SelectionRestore(["late"])
    assert state.resolve([], ["x"], ["x"]) == []
    assert state.resolve(["x"], ["x", "late"], ["x"]) == ["x"]


def test_empty_saved_selection_and_manual_clear_stay_empty():
    state = SelectionRestore([])
    assert state.resolve([], ["x"], ["x"]) == []
    assert state.resolve([], ["x", "y"], ["x"]) == []
    state = SelectionRestore()
    assert state.resolve([], [], []) == []
    assert state.resolve([], ["x"], ["x"]) == ["x"]
    state.observe(["x"])
    assert state.resolve([], ["x", "y"], ["x"]) == []


def test_saved_selection_survives_updates_before_browser_acknowledgement():
    state = SelectionRestore(["x", "y"])
    assert state.resolve([], ["x", "y"], ["x"]) == ["x", "y"]
    assert state.resolve([], ["x", "y", "target"], ["x"]) == ["x", "y"]
    state.observe(["x", "y"])
    state.observe([])
    assert state.resolve([], ["x", "y"], ["x"]) == []


def test_saved_selection_survives_transient_column_loss():
    state = SelectionRestore(["x", "y"])
    assert state.resolve([], ["x", "y"], ["x"]) == ["x", "y"]
    assert state.resolve(["x", "y"], ["x"], ["x"]) == ["x"]
    assert state.resolve(["x"], ["x", "y"], ["x"]) == ["x", "y"]
