"""Deterministic numeric strata, target, identifiers and explicit unused importance."""
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]/'app'
os.chdir(ROOT);sys.path.insert(0,str(ROOT))

import numpy as np
import pandas as pd
from cards.data_strata import instance
from proxy_data import proxy_data
from roles import Role, RoleMap
from shiny import reactive, render, ui

rng=np.random.default_rng(2025)
frame=pd.DataFrame({'x':np.r_[rng.normal(0,1,30),rng.normal(5,2,30)],'y':rng.normal(size=60),'target':np.arange(60,dtype=float),
                    'group':['a']*30+['b']*30,'other':np.resize(['early','late'],60),'id':np.arange(60),'weight':np.arange(60)+1.})
roles=RoleMap()
for c in ['x','y']:roles.set_roles(c,[Role.PREDICTOR])
for c,r in [('target',Role.TARGET),('group',Role.STRATIFIER),('other',Role.STRATIFIER),('id',Role.IDENTIFIER),('weight',Role.WEIGHTING)]:roles.set_roles(c,[r])
source=proxy_data(_df=frame,_roles=roles)
this=instance()
this.restore_configuration_state({'inputs':{'Variables':['x','y'],'Stratifier':'other','Method':'welch','Kind':'box'}})
early=source.clone()
early.frame['y']=early.frame['y'].astype(str)
this._imports.set(early)
original_server,original_footer=this.server,this._footer
this.footer=lambda:ui.TagList(original_footer(),ui.output_text('PassThrough'),ui.input_action_button('FinishRestore',label='Finish upstream restore'))
def server(input,output,session):
    selected=original_server(input,output,session)
    @reactive.effect
    @reactive.event(input.FinishRestore)
    def finish():
        this._imports.set(source.clone())
    @output
    @render.text
    def PassThrough():return f'unchanged={selected().equals(source)}'
    return selected
this.server=server
app=this.application()
