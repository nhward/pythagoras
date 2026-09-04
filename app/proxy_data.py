# proxy_data.py
from __future__ import annotations

import sys
from pathlib import Path

path = Path(__file__).resolve().parents
if str(path) not in sys.path:
    sys.path.insert(0, str(path))

from collections.abc import Iterable as _Iterable
from dataclasses import dataclass, field
from typing import Literal

import geopandas as gpd
import numpy as np
import pandas as pd
from cleaning_record import CleaningRecord
from processing_record import ProcessingRecord
from roles import Role, RoleMap
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline


@dataclass
class proxy_data:
    """
    Thin, opinionated wrapper around a Pandas or GeoPandas object.

    - Exposes the current materialised preview through `.frame`.
    - Retains the stable post-cleaning source through `.clean_frame`.
    - Tracks per-column roles via `RoleMap`.
    - Carries descriptive records of non-pipeline cleaning operations.
    - Carries an unfitted sklearn pipeline for learned transformations.
    - Provides a consistent `.sample()` that preserves row order.

    Card code should use `.frame` to read the DataFrame,
    `.with_cleaned_data()` for cleaning, and `.with_pipeline_step()` for
    learned transformations. `.data` and `.to_native()` remain compatibility
    aliases while older code migrates.
    """
    _df: pd.DataFrame
    _roles: RoleMap = field(default_factory=RoleMap)
    _name: str = None
    _cleaning_records: tuple[CleaningRecord, ...] = field(default_factory=tuple)
    _pipeline: Pipeline | None = None
    _clean_df: pd.DataFrame | None = None
    _processing_records: tuple[ProcessingRecord, ...] = field(default_factory=tuple)

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

        self._cleaning_records = tuple(self._cleaning_records)
        if not all(
            isinstance(record, CleaningRecord)
            for record in self._cleaning_records
        ):
            raise TypeError("_cleaning_records must contain CleaningRecord instances")
        self._processing_records = tuple(self._processing_records)
        if not all(
            isinstance(record, ProcessingRecord)
            for record in self._processing_records
        ):
            raise TypeError("_processing_records must contain ProcessingRecord instances")

        if self._pipeline is not None and not isinstance(self._pipeline, Pipeline):
            raise TypeError("_pipeline must be a scikit-learn Pipeline or None")
        if self._clean_df is None:
            self._clean_df = self._df.copy()
        elif not isinstance(self._clean_df, pd.DataFrame):
            raise TypeError("_clean_df must be a pandas or GeoPandas DataFrame")
        if list(self._clean_df.columns) != list(self._df.columns):
            raise ValueError("clean and current frames must have identical columns")

        # If no roles defined at all, default every column to PREDICTOR
        if not self._roles.column_roles:
            #print("Creating a default role-assignment (all predictors)")
            rm = RoleMap()
            for col in self._df.columns:
                rm.set_roles(col, [Role.PREDICTOR])
            self._roles = rm

    # ----------------- construction & inspection ------------------------

    @classmethod
    def from_native(cls, df) -> proxy_data:
        """
        Build a proxy_data from a 'native' object.

        Accepted:
          - Pandas DataFrame
          - GeoPandas GeoDataFrame
          - Any object with `.to_pandas()`
        """
        if gpd is not None and isinstance(df, gpd.GeoDataFrame) or isinstance(df, pd.DataFrame):
            native = df.copy()
        elif hasattr(df, "to_pandas"):
            native = df.to_pandas().copy()
        else:
            raise TypeError(f"Unsupported native type for proxy_data: {type(df)!r}")
        return cls(native)

    def to_native(self):
        """
        Return the underlying native object (Pandas or GeoPandas).

        Compatibility alias for `.frame`. New card code should use `.frame`.
        """
        return self.frame

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
    def frame(self) -> pd.DataFrame | gpd.GeoDataFrame:
        """Return the materialised DataFrame currently visible to cards.

        Callers must treat the returned frame as read-only. It is a fitted
        full-data preview when a learning pipeline exists, not the data that
        should be used to fit a resampled model.
        """
        return self._df
    
    @property
    def role_map(self) -> RoleMap:
        return self._roles
    @role_map.setter
    def role_map(self, value):
        if not isinstance(value, RoleMap):
            value = RoleMap.from_primitive(value)
        self._roles = value

    @property
    def data(self) -> pd.DataFrame | gpd.GeoDataFrame:
        """Compatibility alias for :attr:`frame`."""
        return self.frame
    @data.setter
    def data(self, value):
        if self.has_pipeline:
            raise RuntimeError(
                "Cannot replace data directly after the learning pipeline has begun"
            )
        self._df = value
        self._clean_df = value.copy()

    @property
    def name(self) -> str:
        return self._name   
    @name.setter
    def name(self, value):
        self._name = value

    @property
    def cleaning_records(self) -> tuple[CleaningRecord, ...]:
        """Return the ordered, immutable cleaning provenance."""
        return self._cleaning_records

    @property
    def processing_records(self) -> tuple[ProcessingRecord, ...]:
        """Return the ordered descriptive history of active and inactive steps."""
        return self._processing_records

    @property
    def clean_frame(self) -> pd.DataFrame | gpd.GeoDataFrame:
        """Return the stable, post-cleaning input to the learning pipeline."""
        return self._clean_df

    @property
    def has_pipeline(self) -> bool:
        """Whether at least one learned transformation has been registered."""
        return self._pipeline is not None and bool(self._pipeline.steps)

    @property
    def pipeline(self) -> Pipeline | None:
        """Return an unfitted clone of the accumulated sklearn pipeline."""
        return clone(self._pipeline) if self._pipeline is not None else None

    @property
    def pipeline_steps(self) -> tuple[str, ...]:
        """Return registered pipeline step names in execution order."""
        if self._pipeline is None:
            return ()
        return tuple(name for name, _ in self._pipeline.steps)

    def pipeline_for_training(self) -> Pipeline:
        """Return a fresh pipeline for fitting inside a training partition."""
        if not self.has_pipeline:
            raise ValueError("No learning pipeline has been registered")
        return clone(self._pipeline)

    def with_pipeline_step(
        self,
        transformer: BaseEstimator,
        *,
        name: str,
        operation: str | None = None,
        preview_frame: pd.DataFrame | gpd.GeoDataFrame,
    ) -> proxy_data:
        """Return a successor with an unfitted learned transformer appended.

        ``preview_frame`` is the card's full-data, fitted result for interactive
        display only. The stored estimator is cloned, so fitted attributes from
        that preview cannot leak into later train/test or cross-validation fits.
        """
        if not isinstance(transformer, BaseEstimator):
            raise TypeError("transformer must be a scikit-learn estimator")
        if not isinstance(preview_frame, pd.DataFrame):
            raise TypeError("preview_frame must be a pandas or GeoPandas DataFrame")
        if list(preview_frame.columns) != list(self.frame.columns):
            raise ValueError("pipeline steps must preserve DataFrame columns")
        if not preview_frame.index.equals(self.frame.index):
            raise ValueError("pipeline steps must preserve the DataFrame index")

        base_name = str(name).strip().replace(" ", "_")
        if not base_name:
            raise ValueError("name must be a non-empty string")
        existing = set(self.pipeline_steps)
        step_name = base_name
        suffix = 2
        while step_name in existing:
            step_name = f"{base_name}_{suffix}"
            suffix += 1

        steps = [] if self._pipeline is None else list(self._pipeline.steps)
        steps.append((step_name, clone(transformer)))
        variables = getattr(transformer, "columns", None)
        if not variables:
            variables = getattr(transformer, "eligible", ())
        omitted = {"columns", "eligible", "predictors"}
        parameters = {
            key: value
            for key, value in transformer.get_params(deep=False).items()
            if key not in omitted
        }
        processing = ProcessingRecord(
            stage="Learning",
            card=step_name,
            operation=(
                str(operation).strip()
                if operation is not None and str(operation).strip()
                else base_name.replace("_", " ").title()
            ),
            attempted=True,
            parameters=parameters,
            input_shape=self.shape,
            output_shape=preview_frame.shape,
            method=type(transformer).__name__,
            variables=tuple(map(str, variables or ())),
        )
        return proxy_data(
            _df=preview_frame.copy(),
            _roles=self._copy_roles(),
            _name=self.name,
            _cleaning_records=self.cleaning_records,
            _pipeline=Pipeline(steps),
            _clean_df=self.clean_frame.copy(),
            _processing_records=(*self.processing_records, processing),
        )

    def with_inactive_step(
        self,
        *,
        stage: Literal["Cleaning", "Learning"],
        card: str,
        operation: str,
        parameters: dict[str, object] | None = None,
    ) -> proxy_data:
        """Return a successor recording a data-changing option as disabled."""
        record = ProcessingRecord(
            stage=stage,
            card=card,
            operation=operation,
            attempted=False,
            parameters=parameters or {},
            input_shape=self.shape,
            output_shape=self.shape,
            method="Not enabled",
        )
        return proxy_data(
            _df=self.frame.copy(),
            _roles=self._copy_roles(),
            _name=self.name,
            _cleaning_records=self.cleaning_records,
            _pipeline=self.pipeline,
            _clean_df=self.clean_frame.copy(),
            _processing_records=(*self.processing_records, record),
        )

    def with_cleaned_data(
        self,
        frame: pd.DataFrame | gpd.GeoDataFrame,
        *,
        card: str,
        operation: str,
        parameters: dict[str, object] | None = None,
        role_map: RoleMap | dict | None = None,
    ) -> proxy_data:
        """Return a successor containing materialised cleaned data.

        This is the preferred mutation boundary for non-learning cards. It
        preserves the existing metadata, appends a :class:`CleaningRecord`,
        and leaves the incoming proxy unchanged.
        """
        if self.has_pipeline:
            raise RuntimeError(
                "Cleaning operations cannot run after the learning pipeline has begun"
            )
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas or GeoPandas DataFrame")

        if role_map is None:
            roles = self._copy_roles(columns=frame.columns)
        elif isinstance(role_map, RoleMap):
            roles = RoleMap()
            for column, assigned in role_map.column_roles.items():
                if column in frame.columns:
                    roles.column_roles[column] = set(assigned)
        else:
            roles = RoleMap.from_primitive(role_map)

        record = CleaningRecord(
            card=card,
            operation=operation,
            parameters=parameters or {},
            input_shape=self.shape,
            output_shape=frame.shape,
        )
        processing = ProcessingRecord(
            stage="Cleaning",
            card=card,
            operation=operation,
            attempted=True,
            parameters=parameters or {},
            input_shape=self.shape,
            output_shape=frame.shape,
            method="Materialised operation",
        )
        return proxy_data(
            _df=frame.copy(),
            _roles=roles,
            _name=self.name,
            _cleaning_records=(*self.cleaning_records, record),
            _clean_df=frame.copy(),
            _processing_records=(*self.processing_records, processing),
        )

    # ----------------- dtype helpers -----------------------------------

    def select_dtypes(self, include=None, exclude=None) -> proxy_data:
        """
        Pandas-style dtype selection, returning a new proxy_data
        (roles are preserved, since this is column-wise).
        """
        selected = self._df.select_dtypes(include=include, exclude=exclude)

        # Filter roles to only the remaining columns
        new_rm = RoleMap()
        for col in selected.columns:
            if col in self._roles.column_roles:
                new_rm.column_roles[col] = set(self._roles.column_roles[col])

        return proxy_data(
            _df=selected,
            _roles=new_rm,
            _name=self.name,
            _cleaning_records=self.cleaning_records,
            _processing_records=self.processing_records,
        )

    # ----------------- drole helpers -----------------------------------

    @staticmethod
    def _normalize_roles(spec: Role | str | None) -> set[Role]:
        if spec is None:
            return set()
        if not isinstance(spec, _Iterable) or isinstance(spec, (str, bytes)):
            spec = [spec]
        return {Role.from_value(item) for item in spec}

    def select_drole(
        self,
        include: Role | str | None = None,
        exclude: Role | str | None = None,
    ) -> proxy_data:
        """
        Return a new proxy_data with columns selected by *roles* rather than dtypes.

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

        return proxy_data(
            _df=selected,
            _roles=new_rm,
            _name=self.name,
            _cleaning_records=self.cleaning_records,
            _processing_records=self.processing_records,
        )

    # ----------------- sampling ----------------------------------------

    def sample(
        self,
        *,
        n: int = 4,
        mode: Literal["headtail", "random", "all"] = "headtail",
        keep_geometry: bool = True,
        seed: int | None = 2025,
    ) -> proxy_data:
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
        return proxy_data(
            _df=sampled,
            _roles=self._copy_roles(),
            _name=self.name,
            _cleaning_records=self.cleaning_records,
            _processing_records=self.processing_records,
        )

    # ----------------- role-related convenience ------------------------

    def with_roles(self, role_map: RoleMap) -> proxy_data:
        """Return a new proxy_data with the same data but a different RoleMap."""
        if not isinstance(role_map, RoleMap):
            role_map = RoleMap.from_primitive(role_map)
        return proxy_data(
            _df=self._df,
            _roles=role_map,
            _name=self.name,
            _cleaning_records=self.cleaning_records,
            _pipeline=self.pipeline,
            _clean_df=self.clean_frame.copy(),
            _processing_records=self.processing_records,
        )

    def clone(self) -> proxy_data:
        """
        Deep-ish copy: copy the DataFrame, shallow copy of the roles structure.
        (Fine because RoleMap contains sets of enums / immutable values.)
        """
        df_copy = self._df.copy()
        roles_copy = RoleMap()
        for col, roles in self._roles.column_roles.items():
            roles_copy.column_roles[col] = set(roles)
        return proxy_data(
            _df=df_copy,
            _roles=roles_copy,
            _name=self.name,
            _cleaning_records=self.cleaning_records,
            _pipeline=self.pipeline,
            _clean_df=self.clean_frame.copy(),
            _processing_records=self.processing_records,
        )

    def _copy_roles(self, columns=None) -> RoleMap:
        selected = None if columns is None else set(columns)
        roles_copy = RoleMap()
        for col, roles in self._roles.column_roles.items():
            if selected is not None and col not in selected:
                continue
            roles_copy.column_roles[col] = set(roles)
        return roles_copy

    # ----------------- role-related validation ------------------------

    def validate(self, role_map: RoleMap | None = None, separator = "|", low_cardinality:int = 8) -> list[str]:
        # Private functions
        def _is_unique(values) -> bool:
            return pd.Series(values).is_unique
        def _merge_cols(*vectors, sep=separator, na="") -> list[str]:
            return [sep.join(str(x) if x is not None else na for x in items) for items in zip(*vectors)]

        if role_map is None:
            role_map = self._roles  
        data = self._df
        errors: list[str] = []

        # Check that data columns and role_map columns match exactly (order irrelevant)
        data_cols = set(map(str, data.columns))
        role_cols = set(map(str, role_map.column_roles.keys()))
        if len(role_cols) == 0:
            return [""]  # return with a blind message that does not display but triggers a failed validation
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
                if value.nunique(dropna=False) > low_cardinality:
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
                if value.nunique(dropna=False) > low_cardinality:
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
                if value.nunique(dropna=False) > low_cardinality:
                    errors.append("Stratifier role must have low cardinality")
                if value.isna().values.any():
                    errors.append("Stratifier role has missing values")

        except Exception:  # noqa: BLE001
            errors.append("The role assignments and data are incompatible")
        return errors    

    def __repr__(self) -> str:
        kind = "GeoDataFrame" if self.is_geodata else "DataFrame"
        return (
            f"proxy_data({kind}, shape={self._df.shape}, "
            f"columns={list(self._df.columns)!r}, "
            f"cleaning_records={len(self.cleaning_records)}, "
            f"pipeline_steps={len(self.pipeline_steps)}, "
            f"processing_records={len(self.processing_records)})"
        )

    def __len__(self) -> int:
        return len(self._df)
    
    def equals(self, other: object, *, check_name: bool = True, check_roles: bool = True, check_crs: bool = True, check_geometry_column: bool = True, check_cleaning_records: bool = True, check_pipeline: bool = True, check_processing_records: bool | None = None) -> bool:
        """
        Compare two proxy_data instances for equality.
        By default this checks:
        - same proxy_data type
        - same data values, index, columns, and dtypes via DataFrame.equals()
        - same role map
        - same name
        - same GeoDataFrame CRS and active geometry column, where relevant
        """
        if not isinstance(other, proxy_data):
            return False
        if check_name and self.name != other.name:
            return False
        if self.is_geodata != other.is_geodata:
            return False
        if check_crs and self.is_geodata and self._df.crs != other._df.crs:
            return False
        if check_geometry_column and self.is_geodata and self._df.geometry.name != other._df.geometry.name:
            return False
        if not self._df.equals(other._df):
            return False
        if check_roles and self._roles.column_roles != other._roles.column_roles:
            return False
        if check_cleaning_records and self.cleaning_records != other.cleaning_records:
            return False
        if check_processing_records is None:
            check_processing_records = check_cleaning_records
        if check_processing_records and self.processing_records != other.processing_records:
            return False
        if check_pipeline:
            if self.pipeline_steps != other.pipeline_steps:
                return False
            own_steps = [] if self._pipeline is None else self._pipeline.steps
            other_steps = [] if other._pipeline is None else other._pipeline.steps
            for (_, own), (_, candidate) in zip(own_steps, other_steps):
                if type(own) is not type(candidate):
                    return False
                if repr(own.get_params(deep=True)) != repr(candidate.get_params(deep=True)):
                    return False
            if not self.clean_frame.equals(other.clean_frame):
                return False
        return True

    def __eq__(self, other: object) -> bool:
        return self.equals(other)
