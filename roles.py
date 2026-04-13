from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Set
from enum import Enum


class Role(str, Enum):
    TARGET     = "target"
    PREDICTOR  = "predictor"
    IDENTIFIER = "identifier"
    WEIGHTING  = "weighting"
    STRATIFIER = "stratifier"
    GEOMETRY   = "geometry"
    SEQUENCE   = "sequence"
    SENSITIVE  = "sensitive"
    PARTITION  = "partition"
    NONE       = "none"

    @classmethod
    def from_value(cls, value: str) -> "Role":
        """
        Robust conversion from an arbitrary string to a Role.

        - Accepts exact matches of the enum values.
        - Case-insensitive for convenience.
        - Raises ValueError for invalid roles.
        """
        if isinstance(value, Role):
            return value
        if not isinstance(value, str):
            raise ValueError(f"Cannot convert {value!r} to Role")

        normalized = value.strip().lower()
        for r in cls:
            if r.value == normalized:
                return r
        raise ValueError(f"Unknown role {value!r}. Valid roles: {[r.value for r in cls]}")

@dataclass
class RoleMap:
    """
    Column → set of roles.
    Example internal structure:
        {
            "y": {Role.TARGET},
            "x1": {Role.PREDICTOR},
            "id": {Role.IDENTIFIER},
        }
    """
    column_roles: Dict[str, Set[Role]] = field(default_factory=dict)

    # --- basic mutation -------------------------------------------------
    def __str___(self):
        return self.column_roles

    def set_roles(self, column: str, roles: Iterable[Role]) -> None:
        """Replace all roles for a column."""
        self.column_roles[column] = {Role.from_value(r) for r in roles}

    def get_roles(self, column: str) -> set:
        """
        Return the set of roles assigned to the column.
        Returns an empty set if the column has no roles.
        """
        return self.column_roles.get(column, set())

    def add_role(self, column: str, role: Role) -> None:
        """Add a single role to a column."""
        role = Role.from_value(role)
        self.column_roles.setdefault(column, set()).add(role)

    def remove_role(self, column: str, role: Role) -> None:
        """Remove a role from a column; silently ignore if absent."""
        role = Role.from_value(role)
        roles = self.column_roles.get(column)
        if not roles:
            return
        roles.discard(role)
        if not roles:
            # Optional: drop column entry if it has no roles left
            self.column_roles.pop(column, None)

    def clear_roles(self, column: str) -> None:
        """Remove all roles from a column."""
        self.column_roles.pop(column, None)

    # --- queries --------------------------------------------------------

    def roles_for(self, column: str) -> Set[Role]:
        """Return the set of roles (possibly empty) for a column."""
        return set(self.column_roles.get(column, set()))

    def columns_with_role(self, role: Role) -> Set[str]:
        """Return all columns that have a given role."""
        role = Role.from_value(role)
        return {col for col, roles in self.column_roles.items() if role in roles}

    def has_role(self, column: str, role: Role) -> bool:
        """Convenience: does this column have the given role?"""
        role = Role.from_value(role)
        return role in self.column_roles.get(column, set())

    # --- convenience helpers -------------------------------------------

    def ensure_singleton(self, role: Role, column: str) -> None:
        """
        Enforce that only one column has a singleton role (e.g. TARGET, PARTITION).
        This will:
          - remove that role from any other column
          - assign it to `column`
        """
        role = Role.from_value(role)
        for col in list(self.column_roles.keys()):
            if role in self.column_roles[col] and col != column:
                self.column_roles[col].remove(role)
                if not self.column_roles[col]:
                    self.column_roles.pop(col, None)
        self.add_role(column, role)

    # --- (de)serialisation ---------------------------------------------

    def to_primitive(self) -> Dict[str, list[str]]:
        """
        Convert to a dict of {column: [role_value, ...]} for JSON/YAML/etc.
        """
        return {
            col: sorted(r.value for r in roles)
            for col, roles in self.column_roles.items()
        }

    @classmethod
    def from_primitive(cls, data: Mapping[str, Iterable[str]]) -> "RoleMap":
        """
        Inverse of to_primitive(): build a RoleMap from string roles.
        """
        rm = cls()
        for col, roles in data.items():
            rm.set_roles(col, [Role.from_value(r) for r in roles])
        return rm