import pandas as pd
import pytest
from cards import var_encode as card
from proxy_data import proxy_data
pytestmark = pytest.mark.unit


def test_audit_matches_export_and_retention():
    source = proxy_data(pd.DataFrame({'flag':[True,False], 'kind':pd.Categorical(['a','b']),
        'constant':pd.Categorical(['x','x']), 'missing':pd.Categorical([None,None]), 'n':[1,2]}))
    selected = ('nominal','logical')
    results = card._analyze_panels(source, {'selected':selected})
    summary, table = card._encoding_audit(source,results,selected)
    assert summary == {'before':5,'after':4,'removed':3,'steps':2}
    rows = table.set_index('Source variable')
    assert rows.loc['flag','Generated variables'] == 'flag__integer'
    assert rows.loc['constant','Outputs'] == 0
    assert rows.loc['constant','Original retained?'] == 'No'
    assert rows.loc['missing','Original retained?'] == 'Yes'
    assert 'No observed levels' in rows.loc['missing','Notes']
    assert 'Reference:' in rows.loc['kind','Notes']
    results = card._analyze_panels(source, {'selected':selected,'logical':{'remove_original':False}})
    summary, table = card._encoding_audit(source,results,selected)
    assert summary['after'] == 5 and summary['removed'] == 2
    assert table.set_index('Source variable').loc['flag','Original retained?'] == 'Yes'


def test_unavailable_empty_unselected_and_existing_pipeline():
    source = proxy_data(pd.DataFrame({'code':pd.Series(['a','b'],dtype='string'),'flag':[True,False]}))
    selected = ('code','cyclic')
    results = card._analyze_panels(source, {'selected':selected})
    summary, table = card._encoding_audit(source,results,selected)
    assert summary == {'before':2,'after':2,'removed':0,'steps':0}
    assert 'Target' in table.iloc[0]['Notes']
    assert 'No suitable predictors' in table.iloc[1]['Notes']
    upstream = card._apply(source,card._analyze_logical(source))
    summary, table = card._encoding_audit(upstream,{},[])
    assert table.empty and summary['steps'] == 0 and summary['before'] == summary['after']
