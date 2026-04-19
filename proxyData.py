# proxy_data.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional, Union
from collections.abc import Iterable as _Iterable
from roles import RoleMap, Role
import numpy as np
import pandas as pd
import geopandas as gpd
from typing import ClassVar

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
    _name: str = None
    LOW: ClassVar[int] = 8

    # ----------------- post-init: default roles -------------------------

    def __post_init__(self):
        """
        If no RoleMap is effectively provided (i.e. empty column_roles),
        assign *all* columns to the PREDICTOR role.

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
            print("Creating a default role-assignment (all predictors)")
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
        kwargs.setdefault("index", False)
        return self._df.to_csv(*args, **kwargs)

    @property
    def is_geodata(self) -> bool:
        return gpd is not None and isinstance(self._df, gpd.GeoDataFrame)

    @property
    def columns(self):
        return self._df.columns

    @property
    def shape(self):
        return self._df.shape
    
    @property
    def role_map(self) -> "RoleMap":
        return self._roles
    @role_map.setter
    def role_map(self, value):
        self._role_map = value


    @property
    def data(self) -> pd.DataFrame | gpd.GeoDataFrame:
        return self._df
    @data.setter
    def data(self, value):
        self._data = value


    @property
    def name(self) -> str:
        return self._name   
    @name.setter
    def name(self, value):
        self._name = value

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
    def _normalize_roles(spec: Union[RoleLike, _Iterable[RoleLike], None]) -> set[Role]:
        if spec is None:
            return set()
        if not isinstance(spec, _Iterable) or isinstance(spec, (str, bytes)):
            spec = [spec]
        return {Role.from_value(item) for item in spec}

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
        if (mode == "all") or (total <= n):
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
        return ProxyData(sampled, self._copy_roles())

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

    def _copy_roles(self) -> RoleMap:
        roles_copy = RoleMap()
        for col, roles in self._roles.column_roles.items():
            roles_copy.column_roles[col] = set(roles)
        return roles_copy

    # ----------------- role-related validation ------------------------

    def validate(self, role_map: Optional[RoleMap] = None, separator = "|") -> list[str]:
        # Private functions
        def _is_unique(values) -> bool:
            return pd.Series(values).is_unique
        def _merge_cols(*vectors, sep=separator, na="") -> list[str]:
            return [sep.join(str(x) if x is not None else na for x in items) for items in zip(*vectors)]

        role_map = self._roles if role_map is None else role_map
        data = self._df
        errors: list[str] = []
        
        # Check that data columns and role_map columns match exactly (order irrelevant)
        data_cols = set(map(str, data.columns))
        role_cols = set(map(str, role_map.column_roles.keys()))
        missing_in_roles = sorted(data_cols - role_cols)
        missing_in_data = sorted(role_cols - data_cols)
        if missing_in_roles or missing_in_data:
            if missing_in_roles:
                errors.append("Columns present in data but missing from role map: " + ", ".join(missing_in_roles))
            if missing_in_data:
                errors.append("Columns present in role map but missing from data: " + ", ".join(missing_in_data))
        dims = data.shape
        if dims[0] == 0:
            errors.append("Data has no observations")
        if dims[1] == 0:
            errors.append("Data has no variables")
        try:
            # Target
            # Singular, numeric/categorical (None missing)
            # If non-numeric, low cardinality proportion 
            tar_cols = sorted(role_map.columns_with_role(Role.TARGET))
            if len(tar_cols) > 1:
                errors.append("Target role must be singular")
            elif len(tar_cols) == 1:
                value = data[tar_cols[0]]
                ok = (
                    pd.api.types.is_numeric_dtype(value)
                    or pd.api.types.is_categorical_dtype(value)
                    or pd.api.types.is_object_dtype(value)
                    or pd.api.types.is_string_dtype(value)
                    or pd.api.types.is_bool_dtype(value)
                )
                if not ok:
                    errors.append("Target role must be numeric or categorical")
                if value.isna().values.any():
                    errors.append("Target role has missing values")

            # Predictor
            # At least one
            prd_cols = sorted(role_map.columns_with_role(Role.PREDICTOR))
            if len(prd_cols) == 0:
                errors.append("Predictor role must be assigned to a variable")
            
            # Identifier + Sequence 
            # jointly unique (none missing)
            id_cols = sorted(role_map.columns_with_role(Role.IDENTIFIER))
            seq_cols = sorted(role_map.columns_with_role(Role.SEQUENCE))
            key_cols = id_cols + seq_cols
            if key_cols:
                merged = _merge_cols(*(data[col] for col in key_cols), sep="|", na="")
                if not _is_unique(merged):
                    errors.append("Identifier/Sequence combination must be unique")

            # Partition
            # Singular, cardinality=2 (None missing)
            par_cols = sorted(role_map.columns_with_role(Role.PARTITION))
            if len(par_cols) > 1:
                errors.append("Partition role must be singular")
            elif len(par_cols) == 1:
                value = data[par_cols[0]]
                if value.nunique(dropna=False) != 2:
                    errors.append("Partition role must have cardinality 2")
                if value.isna().values.any():
                    errors.append("Partition role has missing values")

            # Sequence
            # Singular (None missing)
            seq_cols = sorted(role_map.columns_with_role(Role.SEQUENCE))
            if len(seq_cols) > 1:
                errors.append("Sequence role must be singular")
            elif len(seq_cols) == 1:
                value = data[seq_cols[0]]
                if value.isna().values.any():
                    errors.append("Sequence role has missing values")

            # Weighting
            # Singular, numeric, strickly non-negative (none missing)
            wgh_cols = sorted(role_map.columns_with_role(Role.WEIGHTING))
            if len(wgh_cols) > 1:
                errors.append("Weighting role must be singular")
            elif len(wgh_cols) == 1:
                value = data[wgh_cols[0]]
                if not pd.api.types.is_numeric_dtype(value):
                    errors.append("Weighting role must be numeric")
                elif (value < 0).any():
                    errors.append("Weighting role must be non-negative")
                if value.isna().values.any():
                    errors.append("Weighting role has missing values")

            # Treatment
            # Singular, categorical, low cardinality (none missing)
            trt_cols = sorted(role_map.columns_with_role(Role.TREATMENT))
            if len(trt_cols) > 1:
                errors.append("Treatment role must be singular")
            elif len(trt_cols) == 1:
                value = data[trt_cols[0]]
                if value.nunique(dropna=False) > self.LOW:
                    errors.append("Treatment role must have low cardinality")
                if value.isna().values.any():
                    errors.append("Treatment role has missing values")

            # Sensitive
            # Singular, categorical, low cardinality (none missing)
            sen_cols = sorted(role_map.columns_with_role(Role.SENSITIVE))
            if len(sen_cols) > 1:
                errors.append("Sensitive role must be singular")
            elif len(sen_cols) == 1:
                value = data[sen_cols[0]]
                if value.nunique(dropna=False) > self.LOW:
                    errors.append("Sensitive role must have low cardinality")
                if value.isna().values.any():
                    errors.append("Sensitive role has missing values")

            # Stratifier
            # Singular, categorical, low cardinality (none missing)
            str_cols = sorted(role_map.columns_with_role(Role.STRATIFIER))
            if len(str_cols) > 1:
                errors.append("Stratifier role must be singular")
            elif len(str_cols) == 1:
                value = data[str_cols[0]]
                if value.nunique(dropna=False) > self.LOW:
                    errors.append("Stratifier role must have low cardinality")
                if value.isna().values.any():
                    errors.append("Stratifier role has missing values")

        except Exception as e:
            errors.append(f"The role assignments and data are incompatible: {e}")
        return errors    

    def __repr__(self) -> str:
        kind = "GeoDataFrame" if self.is_geodata else "DataFrame"
        return f"ProxyData({kind}, shape={self._df.shape}, columns={list(self._df.columns)!r})"

    def __len__(self) -> int:
        return len(self._df)
