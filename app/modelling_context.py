"""Versioned outcome semantics, independent of evaluation memberships."""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from roles import Role


@dataclass(frozen=True)
class ModellingContext:
    task: str = 'auto'
    duration: str = ''
    event: str = ''
    event_value: str = '1'
    censored_value: str = '0'
    time_unit: str = 'days'
    time_origin: str = 'Study entry'
    censoring: str = 'right'
    version: int = 1

    def __post_init__(self):
        if self.version != 1 or self.task not in {'auto', 'regression', 'classification', 'ordinal', 'survival'}:
            raise ValueError('Unsupported modelling context version or task.')

    def to_primitive(self):
        return asdict(self)

    @property
    def outcome_columns(self):
        return (self.duration, self.event) if self.task == 'survival' else ()

    def resolved_task(self, source):
        if self.task != 'auto':
            return self.task
        targets = list(source.role_map.columns_with_role(Role.TARGET))
        if len(targets) != 1 or targets[0] not in source.clean_frame:
            return 'unresolved'
        dtype = source.clean_frame[targets[0]].dtype
        if isinstance(dtype, pd.CategoricalDtype) and dtype.ordered:
            return 'ordinal'
        return ('regression' if pd.api.types.is_numeric_dtype(dtype)
                and not pd.api.types.is_bool_dtype(dtype) else 'classification')

    def survival_outcome(self, source, positions):
        """Return event/time records; censoring is observed, not missingness."""
        if self.censoring != 'right':
            raise ValueError('Only right censoring is supported in this stage.')
        if not self.duration or not self.event or self.duration == self.event:
            raise ValueError('Choose distinct survival duration and event columns.')
        if not self.time_unit.strip() or not self.time_origin.strip():
            raise ValueError('Specify survival time units and origin.')
        frame = source.clean_frame
        if any(c not in frame for c in self.outcome_columns):
            raise ValueError('Survival outcome columns must exist in the stable source.')
        duration, event = frame[self.duration].iloc[positions], frame[self.event].iloc[positions]
        if (not pd.api.types.is_numeric_dtype(duration.dtype) or pd.api.types.is_bool_dtype(duration.dtype)
                or duration.isna().any() or not np.isfinite(duration.to_numpy(dtype=float)).all()
                or (duration < 0).any()):
            raise ValueError('Survival duration must be finite, nonnegative numeric data outside prediction sets.')
        labels = event.map(str)
        observed, censored = self.event_value, self.censored_value
        if pd.api.types.is_numeric_dtype(event.dtype) and not pd.api.types.is_bool_dtype(event.dtype):
            try:
                observed, censored = float(observed), float(censored)
                labels = event
            except ValueError:
                pass
        if (observed == censored or event.isna().any()
                or not labels.isin([observed, censored]).all()):
            raise ValueError('Event values must match the explicit event/censored codes outside prediction sets.')
        result = np.empty(len(duration), dtype=[('event', bool), ('time', float)])
        result['event'] = labels.eq(observed)
        result['time'] = duration.to_numpy(dtype=float)
        return result
