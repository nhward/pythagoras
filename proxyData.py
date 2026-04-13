# proxy_data.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional, Union
from collections.abc import Iterable as _Iterable
from roles import RoleMap, Role
import numpy as np
import pandas as pd
import geopandas as gpd

RoleLike = Union["Role", str]

@dataclass
class ProxyData:
    """
    Thin, opinionated wrapper around a Pandas or GeoPandas object.

    - Internally stores a `pd.DataFrame` or `gpd.GeoDataFrame`.
    - Tracks per-column roles via `RoleMap`.
    - Provides a consistent `.sample()` that preserves row order.
    """
    _df: pd.DataFrame
    _roles: RoleMap = field(default_factory=RoleMap)

    # ----------------- post-init: default roles -------------------------

    def __post_init__(self):
        """
        If no RoleMap is effectively provided (i.e. empty column_roles),
        assign *all* columns the PREDICTOR role.

        Rationale:
        - Default is: "everything is a predictor until we know better",
          which is more useful than putting everything in Role.NONE.
        """
        # Normalise _roles to a RoleMap instance
        if not isinstance(self._roles, RoleMap):
            # Allow passing primitive dicts like {col: ["predictor", ...]}
            self._roles = RoleMap.from_primitive(self._roles)

        # If no roles defined at all, default every column to PREDICTOR
        if not self._roles.column_roles:
            rm = RoleMap()
            for col in self._df.columns:
                rm.set_roles(col, [Role.PREDICTOR])
            self._roles = rm

    # ----------------- construction & inspection ------------------------

    @classmethod
    def from_native(cls, df) -> "ProxyData":
        """
        Build a ProxyData from a 'native' object.

        Accepted:
          - Pandas DataFrame
          - GeoPandas GeoDataFrame
          - Any object with `.to_pandas()`
        """
        if gpd is not None and isinstance(df, gpd.GeoDataFrame):
            native = df.copy()
        elif isinstance(df, pd.DataFrame):
            native = df.copy()
        elif hasattr(df, "to_pandas"):
            native = df.to_pandas().copy()
        else:
            raise TypeError(f"Unsupported native type for ProxyData: {type(df)!r}")
        return cls(native)

    def to_native(self):
        """
        Return the underlying native object (Pandas or GeoPandas).
        """
        return self._df

    def to_csv(self, *args, **kwargs):
        return self.to_native().to_csv(*args, **kwargs)

    @property
    def is_geodata(self) -> bool:
        return gpd is not None and isinstance(self._df, gpd.GeoDataFrame)

    @property
    def columns(self):
        return self._df.columns

    @property
    def shape(self):
        return self._df.shape

    # ----------------- dtype helpers -----------------------------------

    def select_dtypes(self, include=None, exclude=None) -> "ProxyData":
        """
        Pandas-style dtype selection, returning a new ProxyData
        (roles are preserved, since this is column-wise).
        """
        selected = self._df.select_dtypes(include=include, exclude=exclude)

        # Filter roles to only the remaining columns
        new_rm = RoleMap()
        for col in selected.columns:
            if col in self._roles.column_roles:
                new_rm.column_roles[col] = set(self._roles.column_roles[col])

        return ProxyData(selected, new_rm)

    # ----------------- drole helpers -----------------------------------

    @staticmethod
    def _normalize_roles(spec: Union[RoleLike, _Iterable[RoleLike], None]) -> set["Role"]:
        """
        Normalise include/exclude arguments into a set[Role].

        Accepts:
          - None
          - single Role or str
          - iterable of Role/str
        """
        if spec is None:
            return set()
        # Single item?
        if not isinstance(spec, _Iterable) or isinstance(spec, (str, bytes)):
            spec = [spec]

        roles: set[Role] = set()
        for item in spec:
            if isinstance(item, Role):
                roles.add(item)
            else:
                if isinstance(item, str):
                    try:
                        roles.add(Role(item))
                    except ValueError:
                        # Try case-insensitive match to Role.value
                        matched = [r for r in Role if r.value.lower() == item.lower()]
                        if matched:
                            roles.add(matched[0])
                        else:
                            raise ValueError(f"Unknown role: {item!r}")
                else:
                    raise TypeError(f"Role spec must be Role or str, got {type(item)}")
        return roles

    def select_drole(
        self,
        include: Union[RoleLike, _Iterable[RoleLike], None] = None,
        exclude: Union[RoleLike, _Iterable[RoleLike], None] = None,
    ) -> "ProxyData":
        """
        Return a new ProxyData with columns selected by *roles* rather than dtypes.

        Parameters
        ----------
        include : Role | str | Iterable[Role|str] | None
            Roles a column must have (at least one) to be *kept*.
            If None, all roles are allowed (subject to `exclude`).
        exclude : Role | str | Iterable[Role|str] | None
            Roles that, if present on a column, cause it to be *dropped*.

        Behaviour
        ---------
        - A column is selected if:
            * (include is empty OR column roles ∩ include ≠ ∅)
          AND
            * (exclude is empty OR column roles ∩ exclude = ∅)
        - Columns with no role metadata are treated as having an empty set().
          If you use Role.NONE explicitly, this still works: just assign it
          in `self._roles.column_roles[col]` and include/exclude it like any other role.
        """
        inc = self._normalize_roles(include)
        exc = self._normalize_roles(exclude)
        if inc and exc and (inc & exc):
            raise ValueError(
                f"select_drole(): overlapping roles in include and exclude: {inc & exc}"
            )

        df = self._df
        role_map = self._roles.column_roles
        selected_cols: list[str] = []

        for col in df.columns:
            col_roles = role_map.get(col, set())
            # include filter
            if inc and not (col_roles & inc):
                continue
            # exclude filter
            if exc and (col_roles & exc):
                continue
            selected_cols.append(col)

        selected = df[selected_cols].copy()
        new_rm = RoleMap()
        for c in selected_cols:
            new_rm.column_roles[c] = set(role_map.get(c, set()))

        return ProxyData(selected, new_rm)

    # ----------------- sampling ----------------------------------------

    def sample(
        self,
        *,
        n: int = 4,
        mode: Literal["headtail", "random", "all"] = "headtail",
        keep_geometry: bool = True,
        seed: Optional[int] = 2025,
    ) -> "ProxyData":
        """
        Return a small sample, preserving row order.
        - mode="headtail": concat(head, tail) with in-order rows.
        - mode="random"  : pick k distinct rows by index, then sort by index.
        For GeoDataFrames, preserves geometry column and CRS if `keep_geometry` is True.
        """
        df = self._df
        # Geo-aware checks
        is_geo = gpd is not None and isinstance(df, gpd.GeoDataFrame)
        geom_col = df.geometry.name if is_geo else None
        crs = df.crs if is_geo else None

        n = max(int(n), 1)
        total = len(df)
        if (mode == "all") | (total <= n):
            sampled = df.copy()
        else:               
            if mode == "headtail":
                n_head = n // 2
                n_tail = n - n_head
                head = df.head(n_head)
                tail = df.tail(n_tail)
                sampled = pd.concat([head, tail], axis=0)
            elif mode == "random":
                rng = np.random.default_rng(seed)
                k = min(total, n)
                idx = df.index.to_numpy()
                chosen_pos = rng.choice(len(idx), size=k, replace=False)
                chosen_pos.sort()
                sampled = df.iloc[chosen_pos]
            else:
                raise ValueError(
                    f"Unknown mode {mode!r}; expected 'headtail' or 'random'"
                )

        if is_geo and keep_geometry and geom_col is not None:
            sampled = gpd.GeoDataFrame(sampled, geometry=geom_col, crs=crs)

        # Roles are column-level, so we can keep them (same RoleMap instance)
        return ProxyData(sampled, self._roles)

    # ----------------- role-related convenience ------------------------

    def with_roles(self, role_map: RoleMap) -> "ProxyData":
        """Return a new ProxyData with the same data but a different RoleMap."""
        if not isinstance(role_map, RoleMap):
            role_map = RoleMap.from_primitive(role_map)
        return ProxyData(self._df, role_map)

    def clone(self) -> "ProxyData":
        """
        Deep-ish copy: copy the DataFrame, shallow copy of the roles structure.
        (Fine because RoleMap contains sets of enums / immutable values.)
        """
        df_copy = self._df.copy()
        roles_copy = RoleMap()
        for col, roles in self._roles.column_roles.items():
            roles_copy.column_roles[col] = set(roles)
        return ProxyData(df_copy, roles_copy)