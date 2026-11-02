"""Variable selection with observable commitments and upstream invalidation."""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
APP_ROOT = Path(__file__).resolve().parents[2] / 'app'
os.chdir(APP_ROOT)
sys.path.insert(0, str(APP_ROOT))
from cards.var_select import instance
from proxy_data import proxy_data
from roles import RoleMap
from shiny import reactive, render, ui
from shiny.types import SilentException

rng=np.random.default_rng(83)
n=240
frame=pd.DataFrame({'signal':rng.normal(size=n),'noise':rng.normal(size=n),
    'category':pd.Categorical(rng.choice(['a','b',None],n)), 'empty':np.nan})
for j in range(28):
    frame[f'noise_{j:02}']=rng.normal(size=n)
frame['target']=frame.signal*5+rng.normal(size=n)*.1
source=proxy_data(frame,_roles=RoleMap.from_primitive(
    {'predictor':[c for c in frame if c != 'target'],'target':['target']}))
this=instance()
this.restore_configuration_state({'inputs':{'Trees':40,'Iterations':18,'Folds':3,'Repeats':1,'Workers':1}})
this._imports.set(source)
original_server=this.server
original_footer=this._footer
this.footer=lambda:ui.TagList(original_footer(),ui.output_text('ExportProbe'),
    ui.input_action_button('ChangeSource', label='Change upstream data'),
    ui.input_action_button('FewIterations',label='Use three Boruta iterations'),
    ui.input_action_button('InvalidWeights',label='Invalidate weights'))

def server(input,output,session):
    result=original_server(input,output,session)
    @reactive.effect
    @reactive.event(input.ChangeSource)
    def change_source():
        frame=source.frame.copy();frame['signal'] *= 2
        this._imports.set(proxy_data(frame,_roles=source.role_map,_name='Updated source'))
    @reactive.effect
    @reactive.event(input.FewIterations)
    def few_iterations():
        ui.update_numeric('Iterations',value=3)
    @reactive.effect
    @reactive.event(input.InvalidWeights)
    def invalidate_weights():
        data=source.frame.copy();data['weight']=np.nan
        roles=RoleMap.from_primitive(source.role_map.to_primitive())
        roles.set_roles('weight',['weighting'])
        this._imports.set(proxy_data(data,_roles=roles))
    @output(suspend_when_hidden=False)
    @render.text
    def ExportProbe():
        try:
            data=result()
        except SilentException:
            return 'waiting'
        recipe=data.pipeline.named_steps['var_select'] if 'var_select' in data.pipeline_steps else None
        return f"name={data.name}; selected={recipe.method if recipe is not None else 'None'}; iterations={recipe.iterations if recipe is not None else 0}; columns={','.join(data.columns)}; pipeline={','.join(data.pipeline_steps) or 'none'}; clean={','.join(data.clean_frame.columns)}"
    return result
this.server=server
app=this.application()
