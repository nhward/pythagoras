from __future__ import annotations

import numpy as np
import pandas as pd
from cyclic_pandas import is_cyclic
from geometry_pandas import is_geometry
from list_pandas import is_list
from text_pandas import is_text

TYPES = {
    "cyc": "cyclic",    #for cyclic embedding
    "txt": "text",      #for text embedding
    "geo": "geometry",
    "bkt": "basket",    #for pivot embedding
    "cde": "code",      #for target encoding
    "ord": "ordered",   #for contrast encoding
    "nom": "nominal",   #for one-hot encoding
    "int": "integer",
    "dec": "decimal",
    "cpx": "complex",
    "log": "logical",
    "dte": "date-time", #for date-time encoding
    "dur": "duration",
    "obj": "object",
    "unk": "unknown"
}

def key_from_dtype(dtype: np.dtype) -> str:
    if is_cyclic(dtype):
        return "cyc"
    if is_text(dtype):
        return "txt"
    if is_geometry(dtype):
        return "geo"
    if is_list(dtype):
        return "bkt"
    if isinstance(dtype, pd.StringDtype):
        return "cde"
    if isinstance(dtype, pd.CategoricalDtype):
        return "ord" if dtype.ordered else "nom"
    if pd.api.types.is_integer_dtype(dtype):
        return "int"
    if pd.api.types.is_float_dtype(dtype):
        return "dec"
    if pd.api.types.is_complex_dtype(dtype):
        return "cpx"
    if pd.api.types.is_bool_dtype(dtype):
        return "log"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "dte"
    if pd.api.types.is_timedelta64_dtype(dtype):
        return "dur"
    if pd.api.types.is_object_dtype(dtype):
        return "obj"
    return str(dtype)


def var_kind(obj: pd.Series | np.dtype) -> str:
    """
    Return the unabreviated kind of thing that the variable is. This goes beyond numpy and pandas data types
    """
    dtype = obj.dtype if isinstance(obj, pd.Series) else obj
    key = key_from_dtype(dtype)
    return TYPES.get(key, "unknown")

__all__ = [
    "TYPES",
    "key_from_dtype",
    "var_kind",
]
