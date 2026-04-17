from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Set
from enum import Enum
import pandas as pd
import json

class Role(str, Enum):
    TARGET     = "target"
    PREDICTOR  = "predictor"
    IDENTIFIER = "identifier"
    WEIGHTING  = "weighting"
    STRATIFIER = "stratifier"
    TREATMENT  = "treatment"
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

# ----------------------------------------------------------------------------------------

@dataclass
class RoleMap:
    """
        A class to represent the assignment of roles to variables.
        In the MOST general case a variable might be allowed more than one role - this is what the class allows.
        We will use it assuming the a variable has only a single role ("none" being an allowed role)
    """
    column_roles: Dict[str, Set[Role]] = field(default_factory=dict)

    # --- basic mutation -------------------------------------------------
    def __str___(self):
        return self.to_primitive()

    def __eq__(self, other) -> bool:
        if not isinstance(other, RoleMap):
            return NotImplemented

        return {
            k: set(v) for k, v in self.column_roles.items() if v
        } == {
            k: set(v) for k, v in other.column_roles.items() if v
        }

    def variable_to_frame(self) -> pd.DataFrame:
        """
        Return a DataFrame with columns:
            - Variable
            - Role

        Every variable appears once.
        Roles are comma-separated.
        """
        rows = []
        for col in sorted(self.column_roles):
            roles = self.column_roles[col]
            role_str = ", ".join(sorted(r.value.title() for r in roles)) if roles else ""
            rows.append({
                "Variable": col,
                "Role": role_str
            })
        return pd.DataFrame(rows)

    def roles_to_frame(self) -> pd.DataFrame:
        """
        Return a DataFrame with columns:
            - Role
            - Variable

        Every role appears once.
        Variables are comma-separated.
        """
        rows = []
        for role in Role:
            cols = [
                col for col, roles in self.column_roles.items()
                if role in roles
            ]
            var_str = ", ".join(sorted(cols)) if cols else ""
            rows.append({
                "Role": role.value.title(),
                "Variable": var_str
            })
        return pd.DataFrame(rows)

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
    
    # --- (de)serialisation ---------------------------------------------

    @classmethod
    def from_primitive(cls, data: Mapping[str, Iterable[str]]) -> "RoleMap":
        """
        Build a RoleMap from a mapping of:
            {role_name: [column_name, ...]}
        """
        rm = cls()
        for role_name, columns in data.items():
            role = Role.from_value(role_name)
            for col in columns:
                rm.add_role(col, role)
        return rm


    def to_primitive(self) -> dict[str, list[str]]:
        """
        Serialise RoleMap to JSON string.
        Example internal structure:
            {
            "Target": {y},
            "Predictor": {x1, x2, x3},
            "identifier": {id},
            }
        """
        result = {r.value: [] for r in Role}
        for col, roles in self.column_roles.items():
            for role in roles:
                result[role.value].append(col)
        return result


    @classmethod
    def from_json(cls, data: str | bytes) -> "RoleMap":
        """
        Deserialize JSON string into a RoleMap.
        """
        parsed = json.loads(data)
        return cls.from_primitive(parsed)


    