import pytest
from pathlib import Path
import sys

# Ensure app root is importable when pytest is run outside the IDE
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roles import RoleMap, Role  # noqa: E402


@pytest.mark.unit
def test_set_and_get_roles():
    rm = RoleMap()
    rm.set_roles("x1", {Role.PREDICTOR})
    assert rm.get_roles("x1") == {Role.PREDICTOR}
    assert rm.get_roles("missing") == set()   # default


@pytest.mark.unit
def test_add_role():
    rm = RoleMap()
    rm.add_role("y", Role.TARGET)
    assert rm.get_roles("y") == {Role.TARGET}
    # adding another role to same column
    rm.add_role("y", Role.IDENTIFIER)
    assert rm.get_roles("y") == {Role.TARGET, Role.IDENTIFIER}


@pytest.mark.unit
def test_remove_role():
    rm = RoleMap()
    rm.set_roles("id", {Role.IDENTIFIER, Role.PREDICTOR})
    rm.remove_role("id", Role.PREDICTOR)
    assert rm.get_roles("id") == {Role.IDENTIFIER}
    # removing role that isn’t present should not throw
    rm.remove_role("id", Role.TARGET)
    assert rm.get_roles("id") == {Role.IDENTIFIER}


@pytest.mark.unit
def test_columns_with_role():
    rm = RoleMap()
    rm.set_roles("y", {Role.TARGET})
    rm.set_roles("x1", {Role.PREDICTOR})
    rm.add_role("x2", Role.PREDICTOR)
    assert set(rm.columns_with_role(Role.PREDICTOR)) == {"x1", "x2"}
    assert set(rm.columns_with_role(Role.TARGET)) == {"y"}
    # assert rm.columns_with_role(Role.IDENTIFIER) == {}


@pytest.mark.unit
def test_to_primitive():
    rm = RoleMap()
    rm.set_roles("y", {Role.TARGET})
    rm.set_roles("x1", {Role.PREDICTOR})
    rm.set_roles("x2", {Role.PREDICTOR})
    rm.set_roles("id", {Role.IDENTIFIER})
    prim = rm.to_primitive()
    # role values are sorted strings
    assert prim[Role.TARGET.value] == ["y"]
    assert prim[Role.PREDICTOR.value] == sorted(["x1","x2"])
    assert prim[Role.IDENTIFIER.value] == ["id"]


    # assert prim["y"] == [Role.TARGET.value]
    # assert prim["x"] == sorted([Role.PREDICTOR.value, Role.IDENTIFIER.value])


@pytest.mark.unit
def test_from_primitive_roundtrip():
    original = {
        "target" : ["y"],
        "predictor" : ["x1", "x2"],
        "identifier" : ["id1", "id2"]
    }
    rm = RoleMap.from_primitive(original)
    # structure restored
    assert rm.get_roles("x1") == {Role.from_value("predictor")}
    assert rm.get_roles("id1") == {Role.from_value("identifier")}
    # roundtrip consistency (and test __eq__)
    assert  rm == RoleMap.from_primitive(rm.to_primitive())
    
