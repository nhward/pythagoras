"""Learned encoding panels for Nominal, Code, Ordered, Cyclic, Logical and Basket predictors."""
from __future__ import annotations

import os
import sys
from pathlib import Path

if __name__ == '__main__':
    ROOT = Path(__file__).resolve().parent.parent
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0,str(ROOT))

import asyncio
from dataclasses import dataclass, field

import pandas as pd
from BasketEncodingTransformer import BasketEncodingTransformer
from card import Card
from cyclic_pandas import as_cyclic
from CyclicEncodingTransformer import CyclicEncodingTransformer
from list_pandas import as_list
from LogicalEncodingTransformer import LogicalEncodingTransformer
from module import Module
from NominalEncodingTransformer import NominalEncodingTransformer
from OrderedEncodingTransformer import METHODS as ORDERED_METHODS
from OrderedEncodingTransformer import OrderedEncodingTransformer
from proxy_data import proxy_data
from roles import Role, RoleMap
from shiny import render, req, ui
from shiny.types import SilentOperationInProgressException
from TargetEncodingTransformer import METHODS, TargetEncodingTransformer
from var_types import var_kind


def _nominal_predictors(data):
    return [c for c in data.columns if Role.PREDICTOR in data.role_map.roles_for(c)
        and not str(c).startswith(Card.SHADOW_PREFIX) and var_kind(data.frame[c]) == 'nominal']


@dataclass
class Encoding:
    transformer: NominalEncodingTransformer | TargetEncodingTransformer | OrderedEncodingTransformer | CyclicEncodingTransformer | LogicalEncodingTransformer | BasketEncodingTransformer | None = None
    frame: pd.DataFrame | None = None
    table: pd.DataFrame = field(default_factory=pd.DataFrame)
    error: str = ''


def _analyze(source, *, remove_original=True, min_frequency=1, max_categories=0, handle_unknown='ignore'):
    columns = _nominal_predictors(source)
    if not columns:
        return Encoding()
    try:
        transformer = NominalEncodingTransformer(columns, remove_original=remove_original,
            min_frequency=int(min_frequency) if min_frequency > 1 else None,
            max_categories=int(max_categories) if max_categories >= 2 else None,
            handle_unknown=handle_unknown)
        preview = transformer.fit_transform(source.frame)
        return Encoding(transformer,preview,pd.DataFrame(transformer.summary_))
    except (ValueError, TypeError) as error:
        return Encoding(error=str(error))


def _code_predictors(data):
    return [c for c in data.columns if Role.PREDICTOR in data.role_map.roles_for(c)
        and Role.TARGET not in data.role_map.roles_for(c)
        and not str(c).startswith(Card.SHADOW_PREFIX) and var_kind(data.frame[c]) == 'code']


def _analyze_code(source, *, remove_original=True, method='auto', smoothing=10., cv=5, target_type='auto'):
    columns = _code_predictors(source)
    table = pd.DataFrame([{'Variable':str(c), 'Cardinality':int(source.frame[c].nunique(dropna=True)),
        'Levels':', '.join(map(str,pd.unique(source.frame[c].dropna()))),
        'Missing':int(source.frame[c].isna().sum())} for c in columns])
    targets = [c for c in source.columns if Role.TARGET in source.role_map.roles_for(c)]
    if len(targets) != 1:
        return Encoding(table=table, error='Code encoding requires exactly one Target-role variable. '
            + ('Assign a Target upstream.' if not targets else 'More than one Target is assigned.'))
    if not columns:
        return Encoding()
    try:
        target = targets[0]
        if target_type == 'auto':
            kind = var_kind(source.frame[target])
            if kind in ('integer','decimal'):
                target_type = 'continuous'
            elif kind in ('nominal','ordered','code','logical'):
                target_type = 'classification'
            else:
                raise ValueError('The Target must be numeric or categorical. Set its semantic type upstream.')
        transformer = TargetEncodingTransformer(columns,target=target,target_type=target_type,
            method=method,smoothing=float(smoothing),cv=int(cv),remove_original=remove_original)
        preview = transformer.fit_transform(source.frame)
        return Encoding(transformer,preview,pd.DataFrame(transformer.summary_))
    except (ValueError,TypeError) as error:
        return Encoding(table=table,error=str(error))


def _ordered_predictors(data):
    return [c for c in data.columns if Role.PREDICTOR in data.role_map.roles_for(c)
        and Role.TARGET not in data.role_map.roles_for(c)
        and not str(c).startswith(Card.SHADOW_PREFIX) and var_kind(data.frame[c]) == 'ordered']


def _analyze_ordered(source, *, remove_original=True, method='ordinal', degree=0, handle_unknown='missing'):
    columns = _ordered_predictors(source)
    if not columns:
        return Encoding()
    try:
        transformer = OrderedEncodingTransformer(columns,remove_original=remove_original,
            method=method,degree=int(degree),handle_unknown=handle_unknown)
        preview = transformer.fit_transform(source.frame)
        return Encoding(transformer,preview,pd.DataFrame(transformer.summary_))
    except (ValueError,TypeError) as error:
        return Encoding(error=str(error))


def _cyclic_predictors(data):
    return [c for c in data.columns if Role.PREDICTOR in data.role_map.roles_for(c)
        and Role.TARGET not in data.role_map.roles_for(c)
        and not str(c).startswith(Card.SHADOW_PREFIX) and var_kind(data.frame[c]) == 'cyclic']


def _analyze_cyclic(source, *, remove_original=True, handle_unknown='missing'):
    columns=_cyclic_predictors(source)
    if not columns:
        return Encoding()
    try:
        transformer=CyclicEncodingTransformer(columns,remove_original=remove_original,handle_unknown=handle_unknown)
        preview=transformer.fit_transform(source.frame)
        return Encoding(transformer,preview,pd.DataFrame(transformer.summary_))
    except (ValueError,TypeError) as error:
        return Encoding(error=str(error))


def _logical_predictors(data):
    return [c for c in data.columns if Role.PREDICTOR in data.role_map.roles_for(c)
        and Role.TARGET not in data.role_map.roles_for(c)
        and not str(c).startswith(Card.SHADOW_PREFIX) and var_kind(data.frame[c]) == 'logical']


def _analyze_logical(source, *, remove_original=True):
    columns = _logical_predictors(source)
    if not columns:
        return Encoding()
    try:
        transformer = LogicalEncodingTransformer(columns, remove_original=remove_original)
        preview = transformer.fit_transform(source.frame)
        return Encoding(transformer, preview, pd.DataFrame(transformer.summary_))
    except (ValueError, TypeError) as error:
        return Encoding(error=str(error))


def _basket_predictors(data):
    return [c for c in data.columns if Role.PREDICTOR in data.role_map.roles_for(c)
        and Role.TARGET not in data.role_map.roles_for(c)
        and not str(c).startswith(Card.SHADOW_PREFIX) and var_kind(data.frame[c]) == 'basket']


def _analyze_basket(source, *, remove_original=True):
    columns = _basket_predictors(source)
    if not columns:
        return Encoding()
    try:
        transformer = BasketEncodingTransformer(columns, remove_original=remove_original)
        preview = transformer.fit_transform(source.frame)
        return Encoding(transformer, preview, pd.DataFrame(transformer.summary_))
    except (ValueError, TypeError) as error:
        return Encoding(error=str(error))


def _analyze_panels(source, options):
    results = {}
    intermediate = source
    for kind,analyze in [('nominal',_analyze),('code',_analyze_code),('ordered',_analyze_ordered),('cyclic',_analyze_cyclic),('logical',_analyze_logical),('basket',_analyze_basket)]:
        result = analyze(intermediate,**options.get(kind,{}))
        results[kind] = result
        if kind in options['selected'] and not result.error:
            intermediate = _apply(intermediate,result)
    return results


def _apply(source, result):
    if result.error:
        raise ValueError(result.error)
    transformer = result.transformer
    if transformer is None or (not transformer.output_columns_ and not transformer.removed_columns_):
        return source
    roles = RoleMap()
    for column in transformer.output_columns_:
        roles.set_roles(column,[Role.PREDICTOR])
    if isinstance(transformer,TargetEncodingTransformer):
        name,operation = 'var_encode_code','Target encode Code predictors'
    elif isinstance(transformer,BasketEncodingTransformer):
        name,operation = 'var_encode_basket','Encode Basket predictors'
    elif isinstance(transformer,LogicalEncodingTransformer):
        name,operation = 'var_encode_logical','Encode Logical predictors'
    elif isinstance(transformer,CyclicEncodingTransformer):
        name,operation = 'var_encode_cyclic','Encode Cyclic predictors'
    elif isinstance(transformer,OrderedEncodingTransformer):
        name,operation = 'var_encode_ordered','Encode Ordered predictors'
    else:
        name,operation = 'var_encode','Encode nominal predictors'
    return source.with_pipeline_step(transformer, name=name, operation=operation,
        preview_frame=result.frame, added_roles=roles, removed_columns=transformer.removed_columns_)


def _encoding_audit(source, results, selected):
    """Describe only enabled encodings and the data actually passed downstream."""
    rows = []
    outgoing = source
    families = [('nominal', _nominal_predictors), ('code', _code_predictors),
        ('ordered', _ordered_predictors), ('cyclic', _cyclic_predictors),
        ('logical', _logical_predictors), ('basket', _basket_predictors)]
    for kind, predictors in families:
        if kind not in selected:
            continue
        result = results[kind]
        model = result.transformer
        if result.error or model is None:
            columns = predictors(outgoing)
            for column in columns or [None]:
                rows.append({'Source variable': str(column) if column is not None else '—',
                    'Encoding': kind.title(), 'Generated variables': '—', 'Outputs': 0,
                    'Original retained?': 'Yes' if column is not None else '—',
                    'Notes': result.error or 'No suitable predictors; no step added.'})
            continue
        for column, info in zip(dict.fromkeys(model.columns), model.summary_):
            method = {'nominal': 'One-hot', 'code': 'Target encoding',
                'ordered': 'Ordered', 'cyclic': 'Sine / cosine', 'logical': 'Boolean → integer', 'basket': 'Binary item presence'}[kind]
            if kind in ('code', 'ordered'):
                method += ': ' + info['Method']
            elif kind == 'cyclic':
                method += f"; period {info['Period']:g} {info['Period units']}"
            notes = info.get('Status', info.get('Meaning', info.get('Mapping', '')))
            if kind == 'nominal':
                if info['Reference']:
                    notes += ' Reference: ' + info['Reference'] + '.'
                if info['Pooled levels']:
                    notes += ' Pooled: ' + info['Pooled levels'] + '.'
            rows.append({'Source variable': str(column), 'Encoding': method,
                'Generated variables': info.get('New variables', info.get('New variable', '')) or '—',
                'Outputs': info.get('Outputs', info.get('Indicators', 1 if kind == 'logical' else 0)),
                'Original retained?': 'No' if column in model.removed_columns_ else 'Yes',
                'Notes': notes})
        outgoing = _apply(outgoing, result)
    count = lambda data: sum(Role.PREDICTOR in data.role_map.roles_for(c) for c in data.columns)
    summary = {'before': count(source), 'after': count(outgoing),
        'removed': len(set(source.columns) - set(outgoing.columns)),
        'steps': len(outgoing.pipeline_steps) - len(source.pipeline_steps)}
    return summary, pd.DataFrame(rows, columns=['Source variable', 'Encoding',
        'Generated variables', 'Outputs', 'Original retained?', 'Notes'])


def instance():
    this = Card(file=__file__, mutable=True)
    this.long_name = 'Variable encoding'
    this.description = 'Encode nominal, Code, Ordered, Cyclic, Logical and Basket predictors through sklearn pipeline steps.'
    this.front = lambda: ui.navset_bar(
        ui.nav_panel(
            'Nominal',
            ui.output_ui('NominalMessage'), 
            ui.output_data_frame(
                id='NominalTable', 
                guide=this, title='Nominal encoding preview', position='left',
                text='Lists observed non-missing levels and cardinality for nominal Predictor variables. The proposed encoding shows the dropped reference, pooled rare levels and collision-safe output names. Unused declared categories are ignored. This is a full-data preview: each training fold relearns its own categories and reference. The footer checkbox applies or removes this encoding step.'
            )
        ),
        ui.nav_panel(
            'Code', 
            ui.output_ui('CodeMessage'),
            ui.output_data_frame(
                id='CodeTable', 
                guide=this, title='Code target encoding', position='left',
                text='Lists only Code-type Predictors, with observed cardinality, levels and proposed outputs. Exactly one usable Target is required. The preview uses cross-fitting: each row is encoded from other folds. Numeric targets yield means; categorical targets yield probabilities, with one output for binary and one per class for multiclass outcomes. No outcomes are read when transforming future rows.'
            )
        ),
        ui.nav_panel(
            'Ordered', 
            ui.output_ui('OrderedMessage'),
            ui.output_data_frame(
                id='OrderedTable', 
                guide=this, title='Ordered encoding preview', position='left',
                text='Lists Ordered Predictors, declared cardinality, observed levels and level order. The order assigned in Variable modification is preserved, including declared levels absent in the current rows. Ordinal ranks produce one feature; full polynomial contrasts produce d−1 features. New variables have Predictor roles. No Target is required.'
            )
        ),
        ui.nav_panel(
            'Cyclic', 
            ui.output_ui('CyclicMessage'),
            ui.output_data_frame(
                id='CyclicTable', 
                guide=this, title='Cyclic encoding and period', position='left',
                text='Lists only Cyclic Predictors. Check the period, its units, origin and complete cycle order before encoding. Numeric cycles use the explicit upstream period; categorical periods equal the number of declared levels, including unobserved ones. Each variable produces sine and cosine Predictor columns. Correct an incorrect period or cycle order upstream; observed range is never used to infer the period.'
            )
        ),
        ui.nav_panel(
            'Logical', 
            ui.output_ui('LogicalMessage'),
            ui.output_data_frame(
                id='LogicalTable', 
                guide=this, title='Logical encoding preview', position='left',
                text='Lists only Logical Predictor variables, their True, False and missing counts, and proposed output names. True becomes integer 1 and False becomes integer 0. Missing values remain missing. Each output has the Predictor role. No Target, observation weighting or estimated mapping is needed. The shared Remove original variables setting also applies here.'
            )
        ),
        ui.nav_panel(
            'Basket', 
            ui.output_ui('BasketMessage'),
            ui.output_data_frame(
                id='BasketTable', 
                guide=this, title='Basket encoding preview', position='left',
                text='Lists Basket Predictors, their distinct observed items, missing and empty basket counts, and generated names. Each item becomes a binary Predictor: present 1, absent 0. Duplicates count once and no reference item is dropped. Training fits learn their own vocabulary; unseen items are ignored without adding columns. Empty and unknown-only baskets give zeros; missing baskets remain missing. Missing items inside a basket are ignored. With no observed items, the original is retained and no features are added. The shared Remove original variables setting applies.'
            )
        ),
        id='EncodingType', selected='Nominal', title=None, padding=0, fillable=True)
    
    this.back = lambda: ui.TagList(
        ui.card_header('Encoding audit', class_='text-primary text-center'),
        ui.output_ui(id='AuditSummary'),
        ui.output_data_frame(
            id='AuditTable', 
            guide=this, title='Outgoing encoding audit', position='left',
            text="""Shows only encodings selected in the footer, in pipeline order. 
            Each row links a source variable to its generated Predictor columns and reports whether the original remains. 
            Notes explain reference levels, pooling and variables that produce no output. 
            Unavailable selections are listed with their reason and add no step. """
        )
    )

    choices={
        'nominal':'Nominal',
        'code':'Code',
        'ordered':'Ordered',
        'cyclic':'Cyclic',
        'logical':'Logical',
        'basket':'Basket'
    }
    this.footer = lambda: ui.TagList(
        ui.input_checkbox_group(
            id='Encode', label="Encode", choices=choices, selected=[], inline=True,
            guide=this, title='Apply variable encoding', position='top',
            text='Select any combination of encodings independently of the active tab. Selected steps run in Nominal, Code, Ordered, Cyclic, Logical, Basket order. All new variables have Predictor roles.'
        ),
        ui.output_ui('Busy'), 
        ui.output_text('Status')
    )

    this.settings = lambda: ui.TagList(
        ui.input_checkbox(
            id='RemoveOriginal', label='Remove original variables', value=True, 
            guide=this, position='left',
            text="Remove encoded source columns from the card's output. Alternatively, retain them with their existing roles alongside the encoded numeric indicators."
        ),
        ui.h5('Nominal encoding'),
        ui.input_numeric(
            id='MinFrequency', label='Minimum level count', value=1, min=1, step=1, 
            guide=this, position='left',
            text='Rare nominal-levels that are observed fewer than this minimum are pooled into an infrequent group. "1" disables this threshold.'
        ),
        ui.input_numeric( #TODO slider?
            id='MaxCategories', label='Maximum categories (0 = unlimited)', value=0, min=0, step=1, 
            guide=this, position='left',
            text='Keep at most this many nominal-levels, including the pooled infrequent category. "0" or "1" disables the limit. 4096 nominal-levels is the algorithmic upper limit.'
        ),
        ui.input_select(
            id='Unknown', label='Unseen levels', choices={'ignore':'All-zero indicators','infrequent_if_exist':'Use infrequent group if available','error':'Report an error'}, selected='ignore', 
            guide=this, position='left',
            text='Controls transform-time levels absent during fitting. All-zero indicators can coincide with a dropped binary reference. The infrequent option uses the pooled group when it exists, otherwise all zeros. Error rejects unseen values.'
        ),
        ui.h5('Code target-encoding'),
        ui.input_select(
            id='CodeMethod', label='Target encoding method', choices=METHODS, selected='auto', 
            guide=this, position='left',
            text='The "Empirical Bayes" option chooses smoothing automatically and is the general default. The "Fixed smoothing" option blends each level mean with the global mean using the specified prior strength. The "No smoothing" option uses raw level means and can be unstable for rare codes.'
        ),
        ui.input_numeric(  #TODO slider?
            id='CodeSmoothing', label='Target smoothing strength', value=10, min=0, step=1, 
            guide=this, position='left',
            text='Used only by the "fixed-strength smoothing" option: (level target sum + strength × overall mean) / (level count + strength). Larger values shrink rare levels more. For classification the calculation applies to class indicators.'
        ),
        ui.input_slider(
            id='CodeFolds', label='Target encoding folds', min=2, max=10, value=5, step=1, 
            guide=this, position='left',
            text='Training encodings use shuffled, reproducible held-out folds, stratified for classification. Fold count is capped by available rows or the smallest class count. At least two rows, and two per class for classification, are needed. Random folds are not suitable protection for grouped or temporal dependence.'
        ),
        ui.input_select(
            id='CodeTargetType', label='Interpret Target as', choices={'auto':'Use semantic type','continuous':'Numeric outcome','classification':'Categorical classes'}, selected='auto', 
            guide=this, position='left',
            text='By default integer and decimal Targets are numeric outcomes; nominal, ordered, Code and logical Targets are classes. Override for numeric class labels. Binary encoding models the second observed class, named in the output; multiclass adds one probability per class.'
        ),
        ui.h5('Ordered encoding'),
        ui.input_select(
            id='OrderedMethod', label='Ordered encoding method', choices=ORDERED_METHODS, selected='ordinal', 
            guide=this, position='left',
            text='Ordinal ranks uses sklearn OrdinalEncoder to produce 0, 1, …, d−1 in the declared level order. Polynomial contrasts provide an R contr.poly-style alternative: linear (L), quadratic (Q), cubic (C), and higher terms. With all degrees, d levels produce d−1 columns, excluding the constant. Both methods treat rank spacing as equal; actual numeric level labels do not supply distances.'
        ),
        ui.input_numeric( #TODO slider?
            id='OrderedDegree', label='Maximum polynomial degree (0 = all)', value=0, min=0, max=64, step=1, 
            guide=this, position='left',
            text='Used only for polynomial contrasts. Zero includes all d−1 degrees; a positive value caps the degree at that value or d−1, whichever is smaller. At most 64 degrees are supported. Full contrasts allow arbitrary level effects in a linear model; restricting degree imposes a simpler trend. Columns are orthogonal over equally weighted levels, not necessarily across an unbalanced sample.'
        ),
        ui.input_select(
            id='OrderedUnknown', label='Unknown Ordered levels', choices={'missing':'Encode as missing','error':'Report an error'}, selected='missing', 
            guide=this, position='left',
            text='A level absent from the fitted declared order has no known rank. By default every corresponding output is missing; Error rejects such a value. Ordinary missing values always remain missing. Declared levels absent from a training fold remain valid because their order is part of the upstream schema.'
        ),
        ui.h5('Cyclic encoding'),
        ui.input_select(
            id='CyclicUnknown', label='Unknown Cyclic levels', choices={'missing':'Encode as missing','error':'Report an error'}, selected='missing', 
            guide=this, position='left',
            text='For categorical cycles, an unknown label has no known angle: encode both outputs as missing or reject it. Missing inputs always give two missing outputs; nonfinite numeric inputs also become missing. Numeric values wrap modulo the fitted period, so negative values and complete revolutions are supported. Categorical levels are equally spaced in their declared order. Numeric zero, or the first categorical level, maps to sine 0 and cosine 1.'
        )
    )


    def server(input, output, session):
        busy = this.busy()

        @this.reactable(calc=True)
        def IncomingData():
            try:
                source = this.input_data()
            except SilentOperationInProgressException:
                # Outputs use the card's busy indicator, not Shiny's persistent
                # progress state inherited from an upstream extended task.
                req(False)
            req(source is not None)
            return source

        @this.reactable(calc=True)
        @this.settle(2)
        def Encode():
            return tuple(input.Encode() or [])

        @this.reactable(calc=True)
        @this.settle(2)
        def Settings():
            return {
                'nominal': {
                    'remove_original': bool(input.RemoveOriginal()),
                    'min_frequency': max(1,int(input.MinFrequency() or 1)),
                    'max_categories': max(0,int(input.MaxCategories() or 0)), 
                    'handle_unknown': input.Unknown()
                },
                'code': {
                    'remove_original': bool(input.RemoveOriginal()), 
                    'method': input.CodeMethod(),
                    'smoothing': float(input.CodeSmoothing() or 0), 
                    'cv': int(input.CodeFolds()),
                    'target_type': input.CodeTargetType()
                },
                'ordered': {
                    'remove_original': bool(input.RemoveOriginal()), 
                    'method': input.OrderedMethod(),
                    'degree': int(input.OrderedDegree() or 0), 
                    'handle_unknown': input.OrderedUnknown()
                },
                'cyclic': {
                    'remove_original': bool(input.RemoveOriginal()), 
                    'handle_unknown': input.CyclicUnknown()
                },
                'logical':{'remove_original': bool(input.RemoveOriginal())},
                'basket':{'remove_original': bool(input.RemoveOriginal())},
            }

        @this.reactable(calc=True)
        def Options():
            return {**Settings(), 'selected':Encode()}

        @busy.track('Preparing variable encodings…')
        @this.extended_task
        async def Calculate(source,options):
            result = _analyze_panels(source, options) if Module.IS_SHINYLIVE else await asyncio.to_thread(_analyze_panels,source,options)
            return source,options,result


        @this.reactable()
        def Start():
            Calculate.cancel()
            Calculate.invoke(IncomingData().clone(), Options())

        @this.reactable(calc=True)
        def Analysis():
            try:
                source,options,result = Calculate.result()
            except SilentOperationInProgressException:
                # Clear normally while fitting; a later result invalidates this
                # calculation and redraws every dependent output.
                req(False)
            req(source.equals(IncomingData()) and options==Options())
            return result

        @this.reactable(calc=True)
        def Export():
            source = IncomingData()
            selected = Encode()
            if not selected:
                return source
            results = Analysis()
            for kind in ('nominal','code','ordered','cyclic','logical','basket'):
                if kind in selected and not results[kind].error:
                    source = _apply(source, results[kind])
            return source

        @this.reactable(calc=True)
        def Audit():
            selected = Encode()
            return _encoding_audit(IncomingData(), Analysis() if selected else {}, selected)

        @output
        @render.ui
        def AuditSummary():
            summary, _table = Audit()
            return ui.TagList(
                ui.p(f"Predictors: {summary['before']} → {summary['after']}, Originals removed: {summary['removed']}")
            )

        @output
        @render.data_frame
        def AuditTable():
            return render.DataTable(Audit()[1], width='100%', height='auto')

        @output
        @render.data_frame
        def NominalTable():
            return render.DataTable(Analysis()['nominal'].table, width='100%', height='auto')

        @output
        @render.ui
        def NominalMessage():
            result = Analysis()['nominal']
            if result.error:
                return ui.p(result.error,class_='text-danger')
            if result.transformer is None:
                return ui.p('No nominal predictors are available.')
            return ui.p('Nominal: Binary nominals retain one encoding; constants add none. Any missing values remain missing.')

        @output
        @render.data_frame
        def CodeTable():
            return render.DataTable(Analysis()['code'].table,width='100%',height='auto')

        @output
        @render.ui
        def CodeMessage():
            result = Analysis()['code']
            if result.error:
                return ui.p(result.error,class_='text-danger')
            if result.transformer is None:
                return ui.p('No Code-type predictors are available.')
            model = result.transformer
            return ui.p(f'Code: Unseen codes use the {model.target!r} mean or class proportions. Missing codes are a category.')

        @output
        @render.data_frame
        def OrderedTable():
            return render.DataTable(Analysis()['ordered'].table,width='100%',height='auto')

        @output
        @render.ui
        def OrderedMessage():
            result = Analysis()['ordered']
            if result.error:
                return ui.p(result.error,class_='text-danger')
            if result.transformer is None:
                return ui.p('No Ordered predictors are available.')
            return ui.p('Ordered: Uses the declared level order, including levels absent from these rows. Ranks assume equal steps. '
                'Polynomial contrasts are orthogonal over equally weighted levels and omit the constant; sample columns need not be orthogonal when counts differ.')

        @output
        @render.data_frame
        def CyclicTable():
            return render.DataTable(Analysis()['cyclic'].table,width='100%',height='auto')

        @output
        @render.ui
        def CyclicMessage():
            result=Analysis()['cyclic']
            if result.error:
                return ui.p(result.error,class_='text-danger')
            if result.transformer is None:
                return ui.p('No Cyclic predictors are available.')
            return ui.p('Cyclic: Uses the declared level order and period. Any missing values remain missing.')

        @output
        @render.data_frame
        def LogicalTable():
            return render.DataTable(Analysis()['logical'].table,width='100%',height='auto')

        @output
        @render.ui
        def LogicalMessage():
            result = Analysis()['logical']
            if result.error:
                return ui.p(result.error,class_='text-danger')
            if result.transformer is None:
                return ui.p('No Logical predictor are available.')
            return ui.p('Logical: True becomes integer 1; False becomes integer 0. Missing values remain missing.')

        @output
        @render.data_frame
        def BasketTable():
            return render.DataTable(Analysis()['basket'].table,width='100%',height='auto')

        @output
        @render.ui
        def BasketMessage():
            result = Analysis()['basket']
            if result.error:
                return ui.p(result.error,class_='text-danger')
            if result.transformer is None:
                return ui.p('No Basket predictors are available.')
            return ui.p('Basket: One binary column per unique item; duplicates count once. Unseen items are ignored. Empty baskets give zeros; missing baskets remain missing. At most 4096 indicators are supported.')

        def NominalStatus():
            result = Analysis()['nominal']
            if result.error:
                return 'Nominal encoding unavailable: '+result.error
            if result.transformer is None:
                return 'No suitable nominal predictors.'
            model = result.transformer
            active = 'nominal' in Encode()
            if not active:
                return None
            if not model.output_columns_ and not model.removed_columns_:
                return None # 'No encodable nominal levels.'
            return f'Nominal: {len(model.output_columns_)} new predictors; {len(model.removed_columns_)} original predictors removed.'

        def CodeStatus():
            result = Analysis()['code']
            if result.error:
                return 'Code encoding unavailable: '+result.error
            if result.transformer is None:
                return 'No suitable code predictors.'
            model = result.transformer
            active = 'code' in Encode()
            if not active:
                return None
            if not model.output_columns_ and not model.removed_columns_:
                return None # 'No encodable code levels.'
            return f'Code: {len(model.output_columns_)} new predictors; {len(model.removed_columns_)} original predictors removed.'


        def OrderedStatus():
            result = Analysis()['ordered']
            if result.error:
                return 'Ordered encoding unavailable: '+result.error
            if result.transformer is None:
                return 'No suitable ordered predictors.'
            model = result.transformer
            active = 'ordered' in Encode()
            if not active:
                return None
            if not model.output_columns_ and not model.removed_columns_:
                return None # 'No encodable ordered levels.'
            return f'Ordered: {len(model.output_columns_)} new predictors; {len(model.removed_columns_)} original predictors removed.'


        def CyclicStatus():
            result = Analysis()['cyclic']
            if result.error:
                return 'Cyclic encoding unavailable: '+result.error
            if result.transformer is None:
                return 'No suitable cyclic predictors.'
            model = result.transformer
            active = 'cyclic' in Encode()
            if not active:
                return None
            if not model.output_columns_ and not model.removed_columns_:
                return None # 'No encodable cyclic levels.'
            return f'Cyclic: {len(model.output_columns_)} new predictors; {len(model.removed_columns_)} original predictors removed.'


        def LogicalStatus():
            result = Analysis()['logical']
            if result.error:
                return 'Logical encoding unavailable: '+result.error
            if result.transformer is None:
                return 'No suitable logical predictors.'
            model = result.transformer
            active = 'logical' in Encode()
            if not active:
                return None
            if not model.output_columns_ and not model.removed_columns_:
                return None # 'No encodable logical levels.'
            return f'Logical: {len(model.output_columns_)} new predictors; {len(model.removed_columns_)} original predictors removed.'

        def BasketStatus():
            result = Analysis()['basket']
            if result.error:
                return 'Basket encoding unavailable: '+result.error
            if result.transformer is None:
                return 'No suitable basket predictors.'
            model = result.transformer
            active = 'basket' in Encode()
            if not active:
                return None
            if not model.output_columns_ and not model.removed_columns_:
                return None # 'No encodable logical levels.'
            return f'Basket: {len(model.output_columns_)} new predictors; {len(model.removed_columns_)} original predictors removed.'


        @output
        @render.text
        def Status():
            print("\n".join(filter(None, [NominalStatus(), CodeStatus(), OrderedStatus(), CyclicStatus(), LogicalStatus(), BasketStatus()])))
            return "\n".join(filter(None, [NominalStatus(), CodeStatus(), OrderedStatus(), CyclicStatus(), LogicalStatus(), BasketStatus()]))

        @output
        @render.ui
        def Busy():
            return busy.ui()

        session.on_ended(Calculate.cancel)

        return Export
    this.server = server
    return this


if Module.running_directly(name=__name__):
    this = instance()
    frame = pd.DataFrame({'color':pd.Categorical(['red','green','blue','red',None]),
        'binary':pd.Categorical(['yes','no','yes','no','yes']), 'account':pd.Series(['a','a','b','c','c'],dtype='string'),
        'size':pd.Categorical(['small','large','medium','small',None],categories=['small','medium','large'],ordered=True), 'value':[1.,2.,3.,4.,5.]})
    frame['hour']=as_cyclic(pd.Series([0.,6.,12.,18.,None]),period=24)
    frame['flag']=pd.Series([True,False,True,False,None],dtype='boolean')
    frame['basket']=as_list(pd.Series([['apple','apple','pear'],['pear'],[],None,['apple']]))
    roles=RoleMap()
    for c in frame: roles.set_roles(c,[Role.TARGET if c == 'value' else Role.PREDICTOR])
    this._imports.set(proxy_data(_df=frame,_roles=roles,_name='Variable encoding example'))
    this.run()
