"""One card at a time: exercise irrelevant missing inputs through its public export."""
import importlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]/'app'
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from proxy_data import proxy_data
from roles import RoleMap
from shiny import render,ui
from shiny.types import SilentException
from text_pandas import as_text

config=json.loads(os.environ['PYTHAGORAS_CONDITIONAL_CARD'])
module=importlib.import_module('cards.'+config['card'])
this=module.instance()
n=240 if config['card']=='var_select' else 12
frame=pd.DataFrame({'x':[np.nan,*range(1,n)],'y':np.arange(n,dtype=float)**2,
    'nominal':pd.Categorical(['a','b']*(n//2)),'flag':[True,False]*(n//2),
    'text':as_text(pd.Series(['one two three']*n)),
    'time':pd.date_range('2020-01-01',periods=n),
    'target':pd.Categorical(['A']*(n*2//3)+['B']*(n-n*2//3))})
predictors=['x','y','nominal','flag'] if config['card']=='var_select' else ['x','y','nominal','flag','text','time']
roles=RoleMap.from_primitive({'predictor':predictors,'target':['target']})
if config.get('weight'):
    frame['weight']=np.linspace(1.,2.,n);roles.set_roles('weight',['weighting'])
this._imports.set(proxy_data(frame,_roles=roles,_cluster_count=2))
this.restore_configuration_state({'inputs':config.get('inputs',{})})
original_server=this.server
original_footer=this._footer
this.footer=lambda:ui.TagList(original_footer() if original_footer else None,ui.input_action_button('ProbeTick',label='Probe export'),ui.output_text('ConditionalProbe'))

def bound_server(input,output,session):
    selected=original_server(input,output,session)
    @output(suspend_when_hidden=False)
    @render.text
    def ConditionalProbe():
        tick=input.ProbeTick()
        try:
            this.export_readiness.check(input)
            data=selected()
        except SilentException:
            return 'waiting'
        return f'ready; tick={tick}; pipeline={data.has_pipeline}; rows={len(data)}; K={data.cluster_count}; missing={data.frame.isna().sum().sum()}'
    return selected
this.server=bound_server
app=this.application()
