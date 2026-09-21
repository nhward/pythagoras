"""Sklearn-compatible sine/cosine encoding using explicit cyclic schema."""
from __future__ import annotations

import numpy as np
import pandas as pd
from code_recording import recordable
from cyclic_pandas import is_cyclic
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


@recordable
class CyclicEncodingTransformer(TransformerMixin,BaseEstimator):
    """Map each cyclic predictor to sin(theta), cos(theta), theta=2*pi*x/P.

    Numeric cycles use zero as origin. Categorical cycles use the first declared
    level, with equally spaced positions. Periods and category order are schema,
    never estimated from a training subset. Transform uses only the fitted schema.
    """

    def __init__(self,columns,*,remove_original=True,handle_unknown='missing'):
        self.columns=columns
        self.remove_original=remove_original
        self.handle_unknown=handle_unknown

    def _validate(self,X):
        if not isinstance(X,pd.DataFrame) or not X.columns.is_unique:
            raise ValueError('Cyclic encoding requires a DataFrame with unique column names.')
        absent=[c for c in self.columns if c not in X]
        if absent:
            raise ValueError(f'Missing Cyclic predictor columns: {absent}')

    def fit(self,X,y=None):
        self._validate(X)
        if self.handle_unknown not in ('missing','error'):
            raise ValueError('Unsupported unseen cyclic level behavior.')
        self.encodings_={}
        self.output_columns_=[]
        self.removed_columns_=[]
        self.summary_=[]
        self.n_features_in_=len(X.columns)
        self.feature_names_in_=np.asarray(X.columns,dtype=object)
        used=set(X.columns)
        for column in dict.fromkeys(self.columns):
            series=X[column]
            dtype=series.dtype
            if not is_cyclic(dtype):
                raise ValueError(f'{column!r} needs an explicit Cyclic dtype during fitting.')
            categories=tuple(dtype.categories) if dtype.is_categorical else None
            period=len(categories) if categories is not None else float(dtype.period)
            if not np.isfinite(period) or period<=0:
                raise ValueError(f'{column!r} needs a finite positive period.')
            names=[]
            for suffix in ('sin','cos'):
                base=f'{column}__{suffix}'
                name,number=base,2
                while name in used:
                    name=f'{base}_{number}';number+=1
                used.add(name);names.append(name)
            self.encodings_[column]=(period,categories,names)
            self.output_columns_.extend(names)
            if self.remove_original:self.removed_columns_.append(column)
            self.summary_.append({
                'Variable':str(column),
                'Cycle type':'Categorical' if categories is not None else 'Numeric',
                'Period':period,
                'Period units':'category positions' if categories is not None else 'input units',
                'Origin':str(categories[0]) if categories is not None else '0',
                'Cycle order':' → '.join(map(str,categories)) if categories is not None else 'Numeric values modulo period',
                'Observed values':int(series.nunique(dropna=True)), 
                'Missing':int(series.isna().sum()),
                'Outputs':2,
                'New variables':', '.join(names)
            })
        self.output_features_=[c for c in X if c not in self.removed_columns_]+self.output_columns_
        return self

    def transform(self,X):
        check_is_fitted(self,'encodings_')
        self._validate(X)
        if set(self.output_columns_) & set(X.columns):
            raise ValueError('A cyclic output name already exists in the incoming data.')
        parts=[X.drop(columns=self.removed_columns_).copy()]
        for column,(period,categories,names) in self.encodings_.items():
            series=X[column]
            if is_cyclic(series.dtype):
                incoming_categories=tuple(series.dtype.categories) if series.dtype.is_categorical else None
                incoming_period=len(incoming_categories) if incoming_categories is not None else float(series.dtype.period)
                if incoming_categories!=categories or incoming_period!=period:
                    raise ValueError(f'Cyclic schema changed for {column!r}; refit after correcting the upstream period or order.')
            missing=series.isna().to_numpy()
            if categories is not None:
                positions=pd.Index(categories).get_indexer(series.to_numpy()).astype(float)
                unknown=(positions<0)&~missing
                if unknown.any() and self.handle_unknown=='error':
                    raise ValueError(f'Unknown cyclic level in {column!r}.')
                positions[missing|unknown]=np.nan
            else:
                raw=series.to_numpy(dtype=object,na_value=np.nan)
                positions=pd.to_numeric(pd.Series(raw),errors='raise').to_numpy(dtype=float,na_value=np.nan,copy=True)
                positions[~np.isfinite(positions)]=np.nan
            # Reduce before multiplying to avoid overflow for very large values.
            angle=(np.mod(positions,period)/period)*(2*np.pi)
            values=np.column_stack((np.sin(angle),np.cos(angle)))
            parts.append(pd.DataFrame(values,index=X.index,columns=names))
        return pd.concat(parts,axis=1)

    def get_feature_names_out(self,input_features=None):
        check_is_fitted(self,'output_features_')
        return np.asarray(self.output_features_,dtype=object)
