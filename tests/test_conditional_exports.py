"""Missing inactive controls must not block; missing active controls must block."""
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'app'
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
import pytest
from playwright.sync_api import expect
from shiny.pytest import create_app_fixture

CASES=[
    {'card': 'miss_impute','inputs': {'Apply':[]},'clear': ['Method','Neighbours','Iterations','Repeats','Holdout','Jobs'],'pipeline': False},
    {'card': 'miss_impute','inputs': {'Apply':['Apply'],'Method':'simple','Repeats':1},'clear': ['Neighbours','Iterations','Repeats','Holdout','Jobs'],'pipeline': True,'required': 'Method'},
    {'card': 'miss_map','inputs': {'Remove':[]},'clear': ['VariableThreshold','ObservationThreshold'],'pipeline': False},
    {'card': 'miss_map','inputs': {'Remove':['Variables']},'clear': ['ObservationThreshold'],'pipeline': True,'required': 'VariableThreshold'},
    {'card': 'miss_informative','inputs': {'Shadow':[]},'clear': ['InformativeOnly','MaxObs','MinMissProp','CVFolds','MinBalancedAccuracy','UseWeights'],'pipeline': False},
    {'card': 'miss_manual','clear': ['MaxTreeDepth','MinLeafAccuracy','ExtraLevel'],'pipeline': False},
    {'card': 'miss_placeholders','inputs': {'Replace':[]},'clear': ['NA_CaseSensitive','NA_Extrema'],'pipeline': False},
    {'card': 'obs_duplicates','inputs': {'RemoveExact':[]},'clear': ['SignificantFigures','UseTarget'],'pipeline': False},
    {'card': 'var_transform','inputs': {'Transform':[]},'clear': ['IncludeTarget','UseWeights','Robust'],'pipeline': False},
    {'card': 'var_transform','inputs': {'Transform':['Deskew']},'clear': ['IncludeTarget','UseWeights','Robust'],'pipeline': True},
    {'card': 'var_time_encode','inputs': {'Encode':[]},'clear': ['Features','RemoveOriginal'],'pipeline': False},
    {'card': 'var_encode','inputs': {'Encode':['logical']},'clear': ['CodeFolds','CodeSmoothing','OrderedDegree','MinFrequency','MaxCategories','Unknown'],'pipeline': True,'required': 'RemoveOriginal'},
    {'card': 'var_text_encode','inputs': {'Encode':['characteristics']},'clear': ['MaxDF','Components','Weighting','EmbeddingLowercase','MinDF','StopWords'],'pipeline': True,'required': 'RemoveOriginal'},
    {'card': 'obs_resample','inputs': {'Goal':'none','Task':'inherit'},'clear': ['Stratifier','Folds','Seed','Gap','Expanding','Duration','Event','EventValue','CensoredValue','TimeUnit','TimeOrigin'],'pipeline': False},
    {'card': 'obs_resample','inputs': {'Goal':'observations','Task':'inherit','Folds':3},'clear': ['Duration','Event','TimeUnit','TimeOrigin','Gap','Expanding','Window','Horizon','Step'],'pipeline': False,'required': 'Folds'},
    {'card': 'obs_balance','inputs': {'Mode':'none'},'clear': ['UseWeights','Up','Down','CountFraction','Metric','Normalize','Evaluation','Voting','Neighbors','MedoidLimit','Iterations'],'pipeline': False},
    {'card': 'obs_balance','inputs': {'Mode':'reweight'},'clear': ['Up','Down','CountFraction','Metric','Normalize','Voting','Neighbors','MedoidLimit','Iterations'],'pipeline': True,'required': 'UseWeights'},
    {'card': 'obs_weighting','clear': ['Component','Power','TestPolicy'],'pipeline': False},
    {'card': 'obs_weighting','inputs': {'Power':'1'},'weight': True,'clear': ['Component'],'pipeline': True,'required': 'TestPolicy'},
    {'card': 'obs_clusters','inputs': {'IncludeMembership':[],'Projection':'pca'},'clear': ['Limit','Standardize','UseWeights','Metric','Centre','Linkage','MinPoints','Border','Neighbours','Projection','Perplexity'],'pipeline': False},
    {'card': 'obs_clusters','inputs': {'IncludeMembership':['Mixture'],'Projection':'pca'},'clear': ['Centre','Metric','MinPoints','Border','Projection','Linkage','Neighbours','Perplexity'],'pipeline': True},
    {'card': 'obs_k_clusters','inputs': {'K':'2'},'clear': ['MaxObs','Metric','Method','Centre','MinPoints','GraphNeighbours','EvaluationLimit','StabilityRepeats','GapReferences','StabilityThreshold'],'pipeline': False,'required': 'Maximum'},
    {'card': 'var_roles','clear': ['Separator','CardinalityThreshold','MaxObs'],'pipeline': False},
    {'card': 'var_select','inputs': {'Recommendation':'Boruta','Trees':30,'Iterations':15,'Workers':1},'clear': ['Folds','Repeats','Cutoff'],'pipeline': True,'required': 'Iterations'},
    {'card': 'var_select','inputs': {'Recommendation':'GrootCV','Trees':30,'Folds':2,'Repeats':1,'Workers':1},'clear': ['Policy','Iterations','Alpha','Percentile','TwoStep'],'pipeline': True,'required': 'Folds'},
]

@pytest.fixture(params=CASES,ids=lambda c:c['card']+'-'+str(c.get('inputs',{})))
def case(request,monkeypatch):
    monkeypatch.setenv('PYTHAGORAS_CONDITIONAL_CARD',json.dumps(request.param))
    return request.param

conditional_app=create_app_fixture(app='scenarios/conditional_exports.py',scope='function')

@pytest.fixture
def app(case,conditional_app):
    return conditional_app

@pytest.mark.ui
def test_inactive_inputs_do_not_gate_export(page,case,app):
    errors=[]
    page.on('pageerror',lambda error:errors.append(str(error)))
    page.goto(app.url)
    probe=page.locator('[id$="-ConditionalProbe"]')
    expect(probe).to_contain_text('ready;',timeout=60000)
    def set_missing(names):
        for name in names:
            control=page.locator(f'[id$="-{name}"]')
            if control.count()==0:
                continue
            expect(control).to_have_count(1)
            page.evaluate('(id)=>Shiny.setInputValue(id,null,{priority:"event"})',control.get_attribute('id'))
    set_missing(case['clear'])
    # A heartbeat proves the session processed the nulls and remains alive.
    tick=page.locator('[id$="-ProbeTick"]').get_attribute('id')
    page.evaluate('(id)=>Shiny.setInputValue(id,1,{priority:"event"})',tick)
    expect(probe).to_contain_text('tick=1',timeout=30000)
    expect(probe).to_contain_text('ready;',timeout=30000)
    expect(probe).to_contain_text('pipeline='+str(case['pipeline']))
    assert not errors
    if case.get('required'):
        set_missing([case['required']])
        expect(probe).to_have_text('waiting',timeout=10000)
