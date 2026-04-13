# test_role_map.py

import pytest   # noqa: F401
from roles import RoleMap, Role

def test_set_and_get_roles():
    rm = RoleMap()
    rm.set_roles("x1", {Role.PREDICTOR})
    assert rm.get_roles("x1") == {Role.PREDICTOR}
    assert rm.get_roles("missing") == set()   # default


def test_add_role():
    rm = RoleMap()
    rm.add_role("y", Role.TARGET)
    assert rm.get_roles("y") == {Role.TARGET}
    # adding another role to same column
    rm.add_role("y", Role.IDENTIFIER)
    assert rm.get_roles("y") == {Role.TARGET, Role.IDENTIFIER}


def test_remove_role():
    rm = RoleMap()
    rm.set_roles("id", {Role.IDENTIFIER, Role.PREDICTOR})
    rm.remove_role("id", Role.PREDICTOR)
    assert rm.get_roles("id") == {Role.IDENTIFIER}
    # removing role that isn’t present should not throw
    rm.remove_role("id", Role.TARGET)
    assert rm.get_roles("id") == {Role.IDENTIFIER}


def test_columns_with_role():
    rm = RoleMap()
    rm.set_roles("y", {Role.TARGET})
    rm.set_roles("x1", {Role.PREDICTOR})
    rm.add_role("x2", Role.PREDICTOR)
    assert set(rm.columns_with_role(Role.PREDICTOR)) == {"x1", "x2"}
    assert set(rm.columns_with_role(Role.TARGET)) == {"y"}
    # assert rm.columns_with_role(Role.IDENTIFIER) == {}


def test_to_primitive():
    rm = RoleMap()
    rm.set_roles("y", {Role.TARGET})
    rm.set_roles("x", {Role.PREDICTOR, Role.IDENTIFIER})
    prim = rm.to_primitive()
    # role values are sorted strings
    assert prim["y"] == [Role.TARGET.value]
    assert prim["x"] == sorted([Role.PREDICTOR.value, Role.IDENTIFIER.value])


def test_from_primitive_roundtrip():
    original = {
        "a": ["predictor"],
        "b": ["target", "identifier"],
    }
    rm = RoleMap.from_primitive(original)
    # structure restored
    assert rm.get_roles("a") == {Role.from_value("predictor")}
    assert rm.get_roles("b") == {
        Role.from_value("target"),
        Role.from_value("identifier"),
    }
    # roundtrip consistency
    assert rm.to_primitive() == {
        "a": ["predictor"],
        "b": sorted(["target", "identifier"]),
    }
