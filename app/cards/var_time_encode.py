"""Generate date/time Predictor features before Variable encoding."""
from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

if __name__ == '__main__':
    ROOT=Path(__file__).resolve().parent.parent
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))


import pandas as pd
from card import Card
from module import Module
from proxy_data import proxy_data
from roles import Role, RoleMap
from shiny import render, req, ui
from shiny.types import SilentOperationInProgressException
from TimeEncodingTransformer import FEATURES, TimeEncodingTransformer, inspect_time


@dataclass
class AnalysisResult:
    transformer: TimeEncodingTransformer | None = None
    frame: pd.DataFrame | None = None
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    error: str = ''


def _analyze(source, *, features=tuple(FEATURES), remove_original=True):
    schemas={}
    for c in source.columns:
        roles=source.role_map.roles_for(c)
        if Role.PREDICTOR not in roles or Role.TARGET in roles or str(c).startswith(Card.SHADOW_PREFIX):continue
        schema=inspect_time(source.frame[c])
        if schema[0] is not None:schemas[c]=schema
    if not schemas:return AnalysisResult()
    try:
        model=TimeEncodingTransformer(list(schemas),features=tuple(features),remove_original=remove_original,schemas=schemas)
        frame=model.fit_transform(source.frame)
        return AnalysisResult(model,frame,pd.DataFrame(model.summary_))
    except (ValueError,TypeError,OverflowError) as error:
        return AnalysisResult(error=str(error))


def _apply(source,result):
    if result.error:raise ValueError(result.error)
    model=result.transformer
    if model is None or not model.output_columns_:return source
    roles=RoleMap()
    for c in model.output_columns_:roles.set_roles(c,[Role.PREDICTOR])
    return source.with_pipeline_step(model,name='var_time_encode',operation='Generate time features',
        preview_frame=result.frame,added_roles=roles,removed_columns=model.removed_columns_)


def instance():
    this=Card(file=__file__,mutable=True)
    this.long_name='Time feature encoding'
    this.description='Generate numeric and cyclic time predictors before Variable encoding.'
    
    this.front=lambda:ui.TagList(
        ui.output_ui('Message'),
        ui.output_data_frame(
            id='TimeTable',
            guide=this, title='Time feature preview', position='left',
            text='Lists any date/time Predictor variables with their detected resolution, timezone, missing counts and proposed output names. All-midnight datetime columns are treated as date-only and receive no clock components. Time-only values receive no calendar features. Mixed timezones are reported and left unchanged.')
        )

    this.back=lambda:ui.TagList(
        ui.card_header(id='Generated time features', class_='text-primary text-center'),
        ui.output_ui(id='AuditSummary'),
        ui.output_data_frame(
            id='FeatureTable',
            guide=this, title='Generated feature audit', position='left',
            text='Lists the source, generated column, data-type and cyclic period for enabled outputs. Cyclic features can feed the "Cyclic" panel of the "Variable encoding" card. Numeric time and year are ordinary numeric Predictors.'
        ),
        ui.output_ui(id='AuditMessage'),
    )
    
    this.footer=lambda:ui.TagList(
        ui.input_checkbox_group(
            id='Encode', label="Encode", choices={'time':'Time features'}, selected=[], inline=True,
            guide=this, title='Apply time features', position='top',
            text='Choose to generate new predictor columns. Feature selection and original-column retention are controlled in the Settings side-bar.'),
        ui.output_ui('Busy'),
        ui.output_text('Status'))

    this.settings=lambda:ui.TagList(
        ui.input_checkbox_group(
            id='Features', label='Time features', choices=FEATURES, selected=list(FEATURES),
            guide=this, title='Choose time features', position='left',
            text='Choose components for all eligible Predictors. Numeric time is seconds since 1970-01-01 (UTC for aware timestamps, assumed UTC for naive values), or seconds since midnight for time-only values. Year is numeric. Quarter, month, weekday (Monday=0), day of month and day of year are categorical cycles with 4, 12, 7, 31 and 366 positions. Short months and non-leap years leave unused positions. Hour (24/12), minute and second have periods 24/12, 60 and 60; seconds include fractional precision. Lunar cycle is approximate age in days modulo 29.530588 from the 2001-01-24 13:07 UT new moon. It is not an astronomical ephemeris. Date-only inputs omit clock components; time-only inputs omit calendar/lunar components.'
        ),
        ui.input_checkbox(
            id='RemoveOriginal', label='Remove original variables', value=True,
            guide=this, position='left',
            text='Remove a source only when it generates the required features. Uncheck to retain sources with their existing roles. Missing inputs remain missing in every generated feature. Mixed timezone columns, all-missing sources, durations and unsupported types are retained. No timezone conversion choice, latitude-derived season or daylight calculation is provided. Normalize mixed timezones upstream.'
        )
    )

    def server(input,output,session):
        busy=this.busy()

        @this.reactable(calc=True)
        def incomingproxy_data():
            try:
                value = this.input_data()
            except SilentOperationInProgressException:
                # This card does not own the upstream task's progress lifecycle.
                # Clear it normally while waiting, avoiding Shiny's persistent
                # output state when hidden/unhidden or refreshed during that task.
                req(False)
            req(value is not None)
            return value


        @this.settle(1)
        @this.reactable(calc=True)
        def Options():
            return {'features':tuple(input.Features() or []),'remove_original':bool(input.RemoveOriginal())}

        @busy.track('Preparing time features…')
        @this.extended_task
        async def Calculate(source,options):
            result=_analyze(source,**options) if Module.IS_SHINYLIVE else await asyncio.to_thread(_analyze,source,**options)
            return source,options,result

        @this.reactable()
        def Start():
            Calculate.cancel();Calculate.invoke(incomingproxy_data().clone(),Options())

        @this.reactable(calc=True)
        def Analysis():
            source,options,result=Calculate.result()
            req(source.equals(incomingproxy_data()) and options==Options())
            return result

        @this.reactable(calc=True)
        def Export():
            source=incomingproxy_data()
            if 'time' not in (input.Encode() or []):return source
            result=Analysis()
            return source if result.error else _apply(source,result)

        @output
        @render.data_frame
        def TimeTable():return render.DataTable(Analysis().table,width='100%',height='auto')

        @output
        @render.ui
        def Message():
            result=Analysis()
            if result.error:return ui.p(result.error,class_='text-danger')
            if result.transformer is None:
                return ui.p('No date/time predictors are available.')
            return ui.p('Review detected resolution and timezone notes. Calendar/clock features use local time; numeric-timestamps use UTC when available.')

        @output
        @render.text
        def Status():
            if 'time' not in (input.Encode() or []):
                return None
            result=Analysis()
            if result.error:
                return 'Time encoding unavailable: '+result.error
            if result.transformer is None or not result.transformer.output_columns_:
                return None
            return f'Time encoding enabled: {len(result.transformer.output_columns_)} Predictor features.'

        @output
        @render.ui
        def AuditSummary():
            source=incomingproxy_data()
            out=Export()
            count=lambda data:sum(Role.PREDICTOR in data.role_map.roles_for(c) for c in data.columns)
            return ui.p(f'Predictors: {count(source)} → {count(out)}, Originals removed: {len(set(source.columns)-set(out.columns))}')


        @output
        @render.ui
        def AuditMessage():
            result=Analysis()
            enabled='time' in (input.Encode() or [])
            if not enabled:
                return ui.p('No time features enabled; incoming data passes through unchanged.')
            if result.error or result.transformer is None:return ui.p(result.error or 'No applicable time features.')
            return ui.p(f'{len(result.transformer.output_columns_)} generated Predictors; {len(result.transformer.removed_columns_)} original variables removed.')

        @output
        @render.data_frame
        def FeatureTable():
            rows=[]
            result=Analysis()
            if 'time' in (input.Encode() or []) and result.transformer is not None:
                from var_types import var_kind
                for source,(_,_,mapping) in result.transformer.encodings_.items():
                    for feature,name in mapping.items():
                        dtype=result.frame[name].dtype
                        period=(len(dtype.categories) if dtype.is_categorical else dtype.period) if var_kind(dtype)=='cyclic' else '—'
                        rows.append({
                            'Source': str(source),
                            'Feature': FEATURES[feature],
                            'Column': name,
                            'Type': var_kind(dtype),
                            'Period': str(period)
                        })
            return render.DataTable(pd.DataFrame(rows,columns=['Source','Feature','Column','Type','Period']),width='100%',height='auto')

        @output
        @render.ui
        def Busy():return busy.ui()
        session.on_ended(Calculate.cancel)
        return Export

    
    this.server=server
    return this


if Module.running_directly(name=__name__):
    from datetime import time
    this=instance()
    frame=pd.DataFrame({
        'visit':pd.to_datetime(['2024-02-29 13:45:30','2024-12-31 23:59:00',None]),
        'date':pd.to_datetime(['2024-02-29','2024-12-31',None]),
        'clock':[time(8,30),time(17,45),None]
    })
    this._imports.set(proxy_data(_df=frame,_name='Time feature example'))
    this.run()
