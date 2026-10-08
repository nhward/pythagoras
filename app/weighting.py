"""Shared observation-weight composition for cards and saved estimators."""
import numpy as np
import pandas as pd
import warnings

from code_recording import recordable
from roles import Role, weighting_purpose


class WeightingError(ValueError):
    """Invalid weight components or effective weights."""


class ExcessiveZeroWeightsError(WeightingError):
    """More than ten percent of effective observation weights are zero."""


class WeightingFallbackWarning(UserWarning):
    """A requested weighted calculation is proceeding without weights."""


@recordable
def observation_weights(data, columns=None, *, enabled=True, **kwargs):
    """Opt-in consumer policy; retain effective_weights as the strict validator."""
    if not enabled:
        return None
    try:
        return effective_weights(data, columns, **kwargs)
    except WeightingError as error:
        warnings.warn(f"Using unweighted calculations: {error}", WeightingFallbackWarning, stacklevel=2)
        return None


@recordable
def effective_weights(data, columns=None, *, purpose="training", test_policy=None):
    """Return an aligned product, or None for uniform/unassigned weighting.

    DataFrames require explicit column names; proxy_data supplies role metadata.
    Test policies are importance (exclude balance__), uniform, or all (only
    when the caller has explicitly supplied training-fitted balance factors).
    An omitted policy uses the latest obs_weighting pipeline step, otherwise
    importance. Explicit policies override saved defaults.
    No imputation, clipping, normalization or silent unweighted fallback occurs.
    """
    frame = data if isinstance(data, pd.DataFrame) else data.frame
    if test_policy is None:
        test_policy = "importance"
        if not isinstance(data, pd.DataFrame):
            for _, step in reversed(getattr(data.pipeline, "steps", ())):
                if hasattr(step, "weighting_test_policy"):
                    test_policy = step.weighting_test_policy
                    break
    if purpose not in ("training", "test"):
        raise ValueError("Weight purpose must be training or test")
    if test_policy not in ("importance", "uniform", "all"):
        raise ValueError("Test weighting policy must be importance, uniform or all")
    if purpose == "test" and test_policy == "uniform":
        return None
    if columns is None:
        if isinstance(data, pd.DataFrame):
            raise ValueError("DataFrame weighting requires explicit column names")
        columns = sorted(data.role_map.columns_with_role(Role.WEIGHTING))
    elif isinstance(columns, str):
        columns = [columns]
    columns = list(dict.fromkeys(columns))
    if purpose == "test" and test_policy == "importance":
        columns = [c for c in columns if weighting_purpose(c) == "importance"]
    product = weight_product(frame, columns)
    if not columns:
        return None
    values = product.to_numpy()
    zeros = int(np.count_nonzero(values == 0))
    if zeros * 10 > len(values):
        raise ExcessiveZeroWeightsError(
            f"Effective weights are zero for {zeros}/{len(values)} observations (more than 10%); disable weighting or revise the components"
        )
    with np.errstate(over="ignore"):
        total = values.sum()
    if not np.isfinite(total) or total <= 0:
        raise WeightingError("Effective weights must have a finite, positive total")
    return pd.Series(values, index=frame.index, name="effective_weight")


@recordable
def weight_product(frame, columns):
    """Validated raw product for diagnostics, including excessive/all zero weights.

    Consumers must use effective_weights to enforce the zero-mass policy.
    """
    values = np.ones(len(frame), dtype=float)
    for column in columns:
        if column not in frame:
            raise WeightingError(f"Weighting variable {column!r} is absent")
        series = frame[column]
        if (not pd.api.types.is_numeric_dtype(series.dtype)
                or pd.api.types.is_bool_dtype(series.dtype)
                or pd.api.types.is_complex_dtype(series.dtype)):
            raise WeightingError(f"Weighting variable {column!r} must be real numeric")
        component = series.to_numpy(dtype=float, na_value=np.nan)
        if not np.isfinite(component).all() or (component < 0).any():
            raise WeightingError(f"Weighting variable {column!r} must be finite, nonnegative and nonmissing")
        try:
            with np.errstate(over="raise", under="raise", invalid="raise"):
                values = values * component
        except FloatingPointError as error:
            raise WeightingError("Effective weights overflow or underflow; rescale the components") from error
    return pd.Series(values, index=frame.index, name="effective_weight")
