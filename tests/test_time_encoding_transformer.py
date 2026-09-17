from datetime import date, time

import pandas as pd
import pytest
from cards import var_encode
from cards import var_time_encode as card
from cyclic_pandas import is_cyclic
from proxy_data import proxy_data
from roles import Role, RoleMap
from sklearn.base import clone
from TimeEncodingTransformer import LUNAR_ORIGIN, TimeEncodingTransformer, inspect_time

pytestmark=pytest.mark.unit


def test_calendar_clock_missing_and_cycles():
    X=pd.DataFrame({'t':pd.to_datetime(['2024-02-29 13:45:30.125',None,'2023-12-31 00:00:00'],format='mixed')})
    X.index=[0,0,2]
    m=TimeEncodingTransformer(['t']);out=m.fit_transform(X)
    assert out.index.equals(X.index) and out.shape==(3,12)
    assert out.iloc[1].isna().all()
    for f,expected in [('year',2024),('quarter',1),('month',2),('day_of_week',3),('day_of_month',29),('day_of_year',60),('hour_24',13),('hour_12',1),('minute',45),('second',30.125)]:
        assert out['t__'+f].iloc[0]==expected
    assert out.t__numeric_time.iloc[0]==(X.t.iloc[0]-pd.Timestamp('1970-01-01')).total_seconds()
    assert out.t__day_of_year.dtype.categories[-1]==366
    assert out.t__second.dtype.period==60
    assert all(is_cyclic(out['t__'+f]) for f in ['quarter','month','day_of_week','day_of_month','day_of_year','hour_24','hour_12','minute','second','lunar_cycle'])
    assert not hasattr(clone(m),'encodings_')
    assert m.get_feature_names_out().tolist()==out.columns.tolist()
    assert m.transform(X.iloc[:0]).shape==(0,12)


def test_date_only_time_only_and_no_features():
    X=pd.DataFrame({'date':pd.to_datetime(['2024-01-01',None]),'clock':[time(0),time(12,30)],
        'object_date':[date(2024,1,1),None],'duration':pd.to_timedelta([1,2],unit='D')})
    result=card._analyze(proxy_data(X));assert not result.error
    assert 'date__hour_24' not in result.frame
    assert 'clock__year' not in result.frame
    assert result.frame.clock__numeric_time.tolist()==[0,45000]
    assert result.frame.clock__hour_12.tolist()==[0,0]
    assert result.frame.duration.equals(X.duration)
    assert inspect_time(X.date)[0]=='date'
    source=proxy_data(X)
    assert card._apply(source,card._analyze(source,features=())) is source


def test_timezone_local_components_utc_numeric_and_mixed_warning():
    aware=pd.Series(pd.to_datetime(['2024-01-01 00:30','2024-06-01 12:00']).tz_localize('Pacific/Auckland'))
    mixed=pd.Series([pd.Timestamp('2024-01-01',tz='UTC'),pd.Timestamp('2024-01-01',tz='Pacific/Auckland')],dtype=object)
    X=pd.DataFrame({'aware':aware,'mixed':mixed})
    result=card._analyze(proxy_data(X));assert not result.error
    assert result.frame.aware__hour_24.iloc[0]==0
    assert result.frame.aware__numeric_time.iloc[0]==(aware.iloc[0].tz_convert('UTC')-pd.Timestamp('1970-01-01',tz='UTC')).total_seconds()
    assert 'Mixed timezones' in result.table.iloc[1]['Notes']
    assert 'mixed' in result.frame and not any(c.startswith('mixed__') for c in result.frame)
    changed=X.copy();changed['aware']=aware.dt.tz_convert('UTC')
    with pytest.raises(ValueError,match='timezone changed'):result.transformer.transform(changed)
    assert inspect_time(pd.Series([time(1,tzinfo=__import__('datetime').timezone.utc)]))[0]=='unavailable'


def test_lunar_reference_and_fixed_schema_across_folds():
    X=pd.DataFrame({'t':[LUNAR_ORIGIN,pd.Timestamp('2001-01-25')]})
    source=proxy_data(X);result=card._analyze(source)
    assert result.frame.t__lunar_cycle.iloc[0]==0
    recipe=clone(result.transformer).fit(X.iloc[1:])
    assert 't__hour_24' in recipe.output_columns_
    pd.testing.assert_frame_equal(recipe.transform(X),result.frame)


def test_roles_collision_retention_and_downstream_cyclic_pipeline():
    X=pd.DataFrame({'t':pd.to_datetime(['2024-01-01 01:00','2024-02-01 02:00']),
        'target':pd.to_datetime(['2024-01-01','2024-01-02']),'t__month':[1,2]})
    roles=RoleMap()
    for c in X:roles.set_roles(c,[Role.TARGET if c=='target' else Role.PREDICTOR])
    source=proxy_data(_df=X,_roles=roles)
    result=card._analyze(source,features=('month','year'))
    out=card._apply(source,result)
    assert out.pipeline_steps==('var_time_encode',)
    assert 't__month_2' in out.columns and out.role_map.roles_for('t__month_2')=={Role.PREDICTOR}
    assert out.role_map.roles_for('target')=={Role.TARGET}
    assert out.role_map.roles_for('t')==set()
    encoded=var_encode._apply(out,var_encode._analyze_cyclic(out))
    pd.testing.assert_frame_equal(encoded.pipeline_for_training().fit_transform(encoded.clean_frame),encoded.frame)
    keep=card._apply(source,card._analyze(source,remove_original=False))
    assert 't' in keep.columns


def test_missing_and_unsupported_columns_are_benign():
    X=pd.DataFrame({'missing':pd.to_datetime([None,None]),'text':['2024-01-01','2024-01-02'],
        'duration':pd.to_timedelta([1,2],unit='D')})
    source=proxy_data(X);result=card._analyze(source)
    assert 'All values are missing' in result.table.iloc[0]['Notes']
    assert card._apply(source,result) is source
