"""Calendar and clock feature generation with fixed cyclic schemas."""
from __future__ import annotations

from datetime import date, datetime, time

import numpy as np
import pandas as pd
from code_recording import recordable
from cyclic_pandas import as_cyclic
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

FEATURES = {
    'numeric_time':'Numeric time', 
    'year':'Year', 
    'quarter':'Quarter', 
    'month':'Month',
    'day_of_week':'Day of week', 
    'day_of_month':'Day of month', 
    'day_of_year':'Day of year',
    'lunar_cycle':'Lunar cycle (approximate)', 
    'hour_24':'Hour of day (24)',
    'hour_12':'Hour of day (12)', 
    'minute':'Minutes', 
    'second':'Seconds'}
CLOCK = {'hour_24','hour_12','minute','second'}
CALENDAR = {'year','quarter','month','day_of_week','day_of_month','day_of_year','lunar_cycle'}
# Mean synodic month and reference new moon from NASA's 2001–2025 UT table.
# https://eclipse.gsfc.nasa.gov/phase/phase2001gmt.html
LUNAR_PERIOD = 29.530588
LUNAR_ORIGIN = pd.Timestamp('2001-01-24 13:07:00')


@recordable
def inspect_time(series):
    """Return (kind, timezone, warning); no strings or durations are parsed."""
    values = list(series.dropna())
    typed = pd.api.types.is_datetime64_any_dtype(series.dtype)
    if not typed and not values:
        return None, None, ''
    if not typed and not all(isinstance(v,(date,time,np.datetime64)) for v in values):
        return None, None, ''
    if not values:
        return 'unavailable', None, 'All values are missing; source retained.'
    times = all(isinstance(v,time) for v in values)
    dates = all(isinstance(v,date) and not isinstance(v,datetime) for v in values)
    if any(isinstance(v,time) for v in values) and not times:
        return 'unavailable', None, 'Mixed date and time-only values; normalize upstream.'
    if times:
        if any(v.tzinfo is not None for v in values):
            return 'unavailable', None, 'Timezone-aware time-only values are unsupported; normalize upstream.'
        return 'time', None, ''
    stamps = [pd.Timestamp(v) for v in values]
    zones = {str(v.tzinfo) if v.tzinfo is not None else None for v in stamps}
    if len(zones) != 1:
        return 'unavailable', None, 'Mixed timezones or naive/aware values; normalize upstream. No features generated.'
    zone = next(iter(zones))
    midnight = all(v.hour == v.minute == v.second == v.microsecond == v.nanosecond == 0 for v in stamps)
    kind = 'date' if dates or midnight else 'datetime'
    warning = ('Local calendar/clock components; numeric time and lunar phase use UTC. DST can repeat or skip clock times.'
        if zone else 'No timezone supplied; numeric time and lunar phase use the naive clock as UTC by convention.')
    if midnight and not dates:
        warning = 'All observed times are midnight: treated as date-only. ' + warning
    return kind, zone, warning


@recordable
class TimeEncodingTransformer(TransformerMixin, BaseEstimator):
    """Append numeric and cyclic features; preserve the schema chosen at fitting.

    ``schemas`` optionally fixes the card's detected date/time resolution across
    training folds. It contains no learned target information or numeric statistics.
    """
    def __init__(self, columns, *, features=tuple(FEATURES), remove_original=True, schemas=None):
        self.columns=columns
        self.features=features
        self.remove_original=remove_original
        self.schemas=schemas

    def _validate(self,X):
        if not isinstance(X,pd.DataFrame) or not X.columns.is_unique:
            raise ValueError('Time encoding requires a DataFrame with unique column names.')
        if any(c not in X for c in self.columns):
            raise ValueError('A source time column is missing.')

    def fit(self,X,y=None):
        self._validate(X)
        if any(f not in FEATURES for f in self.features):
            raise ValueError('Unknown time feature requested.')
        self.encodings_={}; self.output_columns_=[]; self.removed_columns_=[]; self.summary_=[]
        self.feature_names_in_=np.asarray(X.columns,dtype=object); self.n_features_in_=len(X.columns)
        used=set(X.columns)
        for column in dict.fromkeys(self.columns):
            kind,zone,warning = (self.schemas[column] if self.schemas is not None else inspect_time(X[column]))
            if kind not in ('date','datetime','time','unavailable'):
                raise ValueError(f'{column!r} is not a supported date/time variable.')
            eligible = set(FEATURES) if kind=='datetime' else CALENDAR|{'numeric_time'} if kind=='date' else CLOCK|{'numeric_time'} if kind=='time' else set()
            mapping={}
            for feature in dict.fromkeys(self.features):
                if feature not in eligible: continue
                base=f'{column}__{feature}'; name=base; suffix=2
                while name in used:
                    name=f'{base}_{suffix}'; suffix+=1
                used.add(name); mapping[feature]=name
            self.encodings_[column]=(kind,zone,mapping)
            self.output_columns_.extend(mapping.values())
            if mapping and self.remove_original:self.removed_columns_.append(column)
            self.summary_.append({'Variable':str(column),'Interpretation':kind,'Timezone':zone or 'None',
                'Missing':int(X[column].isna().sum()),'Outputs':len(mapping),
                'New variables':', '.join(mapping.values()),'Original retained?':'No' if column in self.removed_columns_ else 'Yes',
                'Notes':warning or ('No applicable features selected; source retained.' if not mapping else '')})
        return self

    def transform(self,X):
        check_is_fitted(self,'encodings_'); self._validate(X)
        if set(self.output_columns_) & set(X.columns):
            raise ValueError('A generated time feature name already exists.')
        out=X.drop(columns=self.removed_columns_).copy()
        for column,(kind,zone,mapping) in self.encodings_.items():
            if not mapping: continue
            data={f:[] for f in mapping}
            for value in X[column]:
                if pd.isna(value):
                    for f in data: # noqa: PLC0206
                        data[f].append(np.nan)
                    continue
                if kind=='time':
                    if not isinstance(value,time) or value.tzinfo is not None:
                        raise ValueError(f'{column!r}: expected timezone-naive time-only values.')
                    v=value
                    numeric=v.hour*3600+v.minute*60+v.second+v.microsecond/1e6
                else:
                    if not isinstance(value,(date,np.datetime64)) or isinstance(value,time):
                        raise ValueError(f'{column!r}: expected date/datetime values, not strings or durations.')
                    v=pd.Timestamp(value)
                    current_zone=str(v.tzinfo) if v.tzinfo is not None else None
                    if current_zone!=zone:
                        raise ValueError(f'{column!r}: timezone changed or mixed; normalize upstream and refit.')
                    absolute=v.tz_convert('UTC').tz_localize(None) if zone else v
                    # Seconds rather than raw integer timestamp units (which vary by dtype).
                    numeric=(absolute-pd.Timestamp('1970-01-01')).total_seconds()
                for f in data:  # noqa: PLC0206
                    if f=='numeric_time': result=numeric
                    elif f=='year':result=v.year
                    elif f=='quarter':result=(v.month-1)//3+1
                    elif f=='month':result=v.month
                    elif f=='day_of_week':result=v.weekday()
                    elif f=='day_of_month':result=v.day
                    elif f=='day_of_year':result=v.dayofyear
                    elif f=='lunar_cycle':result=((absolute-LUNAR_ORIGIN).total_seconds()/86400)%LUNAR_PERIOD
                    elif f=='hour_24':result=v.hour
                    elif f=='hour_12':result=v.hour%12
                    elif f=='minute':result=v.minute
                    else:result=v.second+v.microsecond/1e6+getattr(v,'nanosecond',0)/1e9
                    data[f].append(result)
            for f,name in mapping.items():
                series=pd.Series(data[f],index=X.index,dtype=float)
                if f in ('quarter','month','day_of_month','day_of_year','day_of_week'):
                    categories={'quarter':range(1,5),'month':range(1,13),'day_of_month':range(1,32),
                        'day_of_year':range(1,367),'day_of_week':range(7)}[f]
                    encoded=as_cyclic(pd.Series(pd.Categorical(series,categories=categories,ordered=True),index=X.index))
                elif f in CLOCK or f=='lunar_cycle':
                    period={'hour_24':24,'hour_12':12,'minute':60,'second':60,'lunar_cycle':LUNAR_PERIOD}[f]
                    encoded=as_cyclic(series,period=period)
                elif f=='year':encoded=series.astype('Int64')
                else:encoded=series
                out[name]=encoded.array
        return out

    def get_feature_names_out(self,input_features=None):
        check_is_fitted(self,'encodings_')
        return np.asarray([c for c in self.feature_names_in_ if c not in self.removed_columns_]+self.output_columns_,dtype=object)
