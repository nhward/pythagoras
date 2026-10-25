"""Inspect model recipes against current data; never fit or alter data."""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

if __name__ == '__main__':
    ROOT = Path(__file__).resolve().parent.parent
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from card import Card
from code_recording import recordable
from estimator_catalogue import (
    assess_profiles,
    installed_versions,
    load_catalogue,
    requirements_from_proxy,
)
from faicons import icon_svg as icon
from module import Module
from proxy_data import proxy_data
from roles import Role
from shiny import render, req, ui
from shiny.types import SilentOperationInProgressException
from weighting import WeightingError, effective_weights

CHOICES = {
    '': 'No additional requirement',
    'class_probability': 'Class probabilities', 
    'survival_probability': 'Survival probabilities',
    'risk_score': 'Survival risk score', 
    'point_forecast': 'Point forecast', 
    'prediction_interval': 'Prediction interval'
}

@recordable
def _analyze(source, catalogue, versions, *, use_weights=False, forecasting=False, output=''):
    notes = []
    weighted = False
    if use_weights:
        try:
            weighted = effective_weights(source) is not None
        except WeightingError as error:
            notes.append(f'Using unweighted requirements: {error}')
        # if not weighted and not notes:
        #     notes.append('No nonuniform observation weights; using unweighted requirements.')
    summary = requirements_from_proxy(source, use_weights=weighted)
    task = 'forecasting' if forecasting else summary['task']
    roles = source.role_map
    sequences = list(roles.columns_with_role(Role.SEQUENCE))
    identifiers = list(roles.columns_with_role(Role.IDENTIFIER))
    repeated = any(c in source.clean_frame and not source.clean_frame[c].is_unique for c in identifiers)
    spatial = bool(list(roles.columns_with_role(Role.GEOMETRY)))
    structure = []
    available = []
    if sequences:
        available.append('time_index')
    if identifiers:
        available.append('subject_groups')
    if spatial:
        structure.append('spatial')
    if repeated:
        structure.append('longitudinal')
    if forecasting:
        structure.append('panel' if repeated else 'single_series')
        notes.append('Forecast horizon and lag-window recipes require a future forecasting adapter; split windows are not forecast horizons.')
    elif sequences:
        structure.append('temporal')
    if task == 'survival':
        structure.append('baseline_subjects')
        context = source.modelling_context
        plan = source.evaluation_plan
        positions = np.setdiff1d(np.arange(len(source.clean_frame)), plan.prediction if plan else [])
        try:
            context.survival_outcome(source, positions)
            available.extend(['event_time_target', 'duration_event_columns'])
        except ValueError as error:
            notes.append(str(error))
    elif task == 'ordinal':
        available.append('ordered_target')
    if not structure:
        structure = ['cross_sectional']
    if not summary['evaluation_ready']:
        notes.append('Evaluation not ready: configure a valid upstream resampling plan. Catalogue remains browsable.')
    if task == 'unresolved':
        notes.append('Assign a Target or configure the outcome task in Observation resampling.')
    blockers = []
    if not len(source.frame):
        blockers.append('No observations')
    if summary['unsupported_columns']:
        blockers.append('Unsupported predictor types: ' + ', '.join(summary['unsupported_columns']))
    for column in source.model_predictors:
        values = source.frame[column]
        if pd.api.types.is_numeric_dtype(values.dtype) and not pd.api.types.is_bool_dtype(values.dtype) and np.isinf(values.to_numpy(dtype=float, na_value=np.nan)).any():
            blockers.append(f'Infinite predictor values: {column}')
    rows = assess_profiles(catalogue, task=task, required=summary['required'], structure=structure,
                           outputs=[output] if output else [], available_requirements=available,
                           package_versions=versions)
    for row in rows:
        assessment = row['input_assessment']
        if row['profile'].get('construction', {}).get('preparation') == 'pandas_categories':
            categorical = [source.frame[c] for c in source.model_predictors
                           if not pd.api.types.is_numeric_dtype(source.frame[c].dtype)
                           or pd.api.types.is_bool_dtype(source.frame[c].dtype)]
            if all(isinstance(v.dtype, pd.CategoricalDtype) for v in categorical):
                assessment['actions'] = [a for a in assessment['actions']
                                         if a != 'Check categorical representation and estimator configuration']
                if not assessment['actions'] and assessment['readiness'] == 'Requires preparation':
                    assessment['readiness'] = 'Meets reviewed input requirements'
        if blockers:
            row['compatibility'] = 'incompatible'
            row['reasons'].extend(blockers)
            row['selectable'] = False
    notes.extend(blockers)
    notes.append('Readiness combines current input and broader requirements. Evidence describes the complete tested recipe. Preparation is not executed by this card.')
    notes.append('Coverage is incomplete. Documentation-only recipes require adapters and runtime validation.')
    return {'summary': summary, 'task': task, 'structure': structure, 'rows': rows, 'notes': notes}


def _display_label(value):
    """Human-readable catalogue labels; retain raw codes for logic and configuration."""
    return str(value).replace("_", " ")


def _readiness(row):
    messages = []
    if row.get('superseded_by'):
        messages.append('Superseded in catalogue by ' + row['superseded_by']['method'])
    preprocessing = list(row['input_assessment']['actions'])
    for reason in row['reasons']:
        key, _, status = reason.partition(': ')
        if status == 'outside this recipe':
            if key.startswith('missing_') and key.endswith('_predictors'):
                preprocessing.append('Resolve missing predictor values')
            elif key == 'categorical_predictors':
                preprocessing.append('Represent categorical predictors numerically')
            else:
                label = {'sample_weight_fit': 'weighting', 'numeric_predictors': 'numeric predictors'}.get(key, _display_label(key))
                messages.append(f'Does not support {label}')
        elif status == 'unreviewed':
            messages.append(f'Untested: {_display_label(key)}')
        else:
            messages.append(_display_label(reason))
    if row['input_assessment']['readiness'] == 'Unreviewed':
        messages.append('Untested: input requirements')
    if preprocessing:
        messages.append('Preprocessing: ' + '; '.join(dict.fromkeys(preprocessing)))
    return '; '.join(dict.fromkeys(messages)) or 'Meets requirements'


def _model_labels(identifier):
    parts = identifier.split('.')
    # Keep acronyms together: GBMRegressor -> GBM Regressor.
    model = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', parts[-1])
    model = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', model).replace('_', ' ')
    kind = parts[-2] if len(parts) > 1 else parts[0]
    return {'Model': model, 'Type': kind.replace('_', ' ').capitalize(), 'Package': parts[0]}


def _recipe_rank(row):
    """Choose one intact recipe for the method summary; never pool capabilities."""
    return (_readiness(row) != 'Meets requirements',
            row['compatibility'] != 'compatible',
            row['evidence'] != 'probe_passed',
            row['availability'] != 'version_matches', row['id'])


def _method_groups(result, *, show_all=False, installed_only=False):
    groups = {}
    for row in result['rows']:
        if row.get('purpose') == 'benchmark_only':
            continue
        if not show_all and _readiness(row) != 'Meets requirements':
            continue
        if installed_only and row['availability'] == 'not_installed':
            continue
        groups.setdefault(row['estimator'], []).append(row)
    return {method: sorted(rows, key=_recipe_rank) for method, rows in groups.items()}


def _table(result, *, show_all=False, installed_only=False):
    records = []
    for method, rows in _method_groups(result, show_all=show_all, installed_only=installed_only).items():
        row = rows[0]
        profile = row['profile']
        records.append({**_model_labels(method), 'Recipes': len(rows),
            'Summary recipe': _display_label(profile.get('label', profile['id'].split(':')[-1])),
            'Input requirements': row['input_assessment']['requirements'],
            'Readiness': _readiness(row), 'Evidence': row['evidence'],
            'Installation': row['availability'],
            'Scale': profile.get('scalability', {}).get('many_rows', 'See assessment'),
            'ID': method})
    table = pd.DataFrame(records, columns=['Model', 'Type', 'Package', 'Recipes', 'Summary recipe',
        'Input requirements', 'Readiness', 'Evidence', 'Installation', 'Scale', 'ID'])
    for column in ('Input requirements', 'Readiness', 'Evidence', 'Installation', 'Scale'):
        table[column] = table[column].map(_display_label)
    return table


def _recipe_details(row, catalogue_version):
    """One flowing document: no competing fill-height prose outputs."""
    profile = row['profile']
    labels = _model_labels(row['estimator'])
    readiness = _readiness(row)
    ready = readiness == 'Meets requirements'

    def section(title, glyph, *content):
        return ui.tags.section(ui.h6(icon(glyph), title), *content, class_='model-recipe-section')

    def links(urls):
        return ui.tags.ul(*(ui.tags.li(ui.a(f'Source {i + 1}', href=url, target='_blank',
                                          rel='noopener', title=url)) for i, url in enumerate(urls)),
                          class_='model-recipe-sources')

    relation = row.get('superseded_by')
    supersession = None
    if relation:
        supersession = section('Related broader method', 'code-branch',
            ui.p('Broader method: ' + relation['method']), ui.p(relation['reason']),
            ui.p(relation['caveats']),
            ui.tags.details(ui.tags.summary('Parameter mapping'),
                            ui.tags.pre(json.dumps(relation['configuration'], indent=2))),
            ui.p('Display grouping only; validation and availability do not transfer between methods.',
                 class_='model-recipe-muted'), links(relation['sources']))
    steps = profile.get('steps', profile.get('recipe', {}).get('steps', []))
    conditions = profile.get('conditions', profile.get('assumptions', []))
    return ui.div(
        ui.div(ui.span(f"{labels['Package']} · {labels['Type']}", class_='model-recipe-eyebrow'),
               ui.h4(labels['Model'], class_='model-recipe-title'),
               ui.code(row['estimator'], class_='model-recipe-path'),
               ui.div(ui.span(icon('circle-check' if ready else 'circle-info'), readiness,
                              class_='model-recipe-badge ' + ('is-ready' if ready else 'is-review')),
                      ui.span(icon('clipboard-check'), 'Evidence: ' + _display_label(row['evidence']),
                              class_='model-recipe-badge'), class_='model-recipe-badges'),
               class_='model-recipe-header'),
        ui.div(
            section('Input requirements', 'filter', ui.p(row['input_assessment']['requirements'])),
            section('Tested preparation', 'sliders',
                    ui.p(_display_label(profile.get('construction', {}).get('preparation', 'See documented recipe'))),
                    ui.p('Other preparation may meet the requirements; the recorded evidence applies to this recipe.',
                         class_='model-recipe-muted')),
            section('Configuration', 'code',
                    ui.tags.pre(json.dumps(profile.get('construction', profile.get('parameters', {})), indent=2))),
            section('Preparation and fitting', 'list-ol', ui.tags.ol(*(ui.tags.li(s) for s in steps))),
            section('Conditions', 'circle-info', ui.tags.ul(*(ui.tags.li(s) for s in conditions))),
            section('Assessment', 'magnifying-glass',
                    ui.p(profile.get('subjective_assessment', {}).get('rationale', 'Not reviewed')),
                    ui.p('Expert guidance, not a performance benchmark.', class_='model-recipe-muted')),
            supersession,
            section('References', 'book-open', links(profile['sources']),
                    ui.p('Integration: ' + _display_label(row['integration']), class_='model-recipe-muted')),
            class_='model-recipe-grid'),
        ui.div(f'Catalogue {catalogue_version} · Inspection only; no model is trained here.',
               class_='model-recipe-footer'),
        class_='model-recipe')


def instance():
    this = Card(file=__file__, mutable=False)
    this.long_name = 'Model method candidates'
    this.description = 'Identify modeling methods compatible with the current outcome, predictors and observation structure.'
    this.front = lambda: ui.TagList(
        ui.output_ui(id='Summary', guide=this, title="Summary", text="A one-line summary of the outcome, observation structure and predictors"), 
        ui.output_data_frame(id='Table', guide=this, title="Candidate methods", text="A table of methods that might be applicable to the data in its current form when executed as a pipeline.").add_style("margin-top: 0.75rem;")
    )
    this.back = lambda: ui.TagList(
        ui.output_ui(id='RecipeDetails', guide=this, title="Recipe", text="Details of the currently selected method recipe", class_="model-recipe-output")
    )
    this.settings = lambda: ui.TagList(
        ui.input_checkbox(id='ShowAll', label='Show excluded and unreviewed recipes', value=False,
            guide=this, position='left',
            text='Include superseded methods, methods needing preprocessing, and unsupported or untested recipes. Readiness names the broader method; details explain the configuration and limitations. Unchecked hides these entries. Supersession groups methods for display and does not substitute fitted models.'),
        ui.input_checkbox(id='InstalledOnly', label='Show installed packages only', value=False,
            guide=this, position='left',
            text='Hide methods whose required package is not installed in this runtime. Installation alone does not establish a matching version or validated support; check Installation and Evidence.'),
        ui.input_checkbox(id='UseWeights', label='Use available observation weightings', value=False,
            guide=this, position='left',
            text='Require support for fitting with the effective observation weights assigned upstream. Absent, uniform or invalid weights use unweighted requirements; The front summary explains any fallback. This card does not fit models or apply weights.'),
        ui.input_checkbox(id='Forecasting', label='Explore forecasting recipes', value=False,
            guide=this, position='left',
            text='Explore forecasting methods instead of the incoming outcome task. Assigned Sequence and Identifier roles determine time and panel requirements. These documented recipes still require forecasting adapters and validation.'),
        ui.input_select(id='Output', label='Required prediction output', choices=CHOICES, selected='',
            guide=this, position='left',
            text='Require a particular prediction output in addition to the data and observation-structure requirements. Leave the default for no extra output requirement. Unreviewed output support appears as Untested when excluded recipes are shown.'),
    )
    this.footer = lambda: ui.output_text('Status')

    def server(input, output, session):
        catalogue = load_catalogue()
        versions = installed_versions()

        @this.reactable(calc=True)
        def Incoming():
            try:
                source = this.input_data()
            except SilentOperationInProgressException:
                req(False)
            req(source is not None)
            return source

        @this.reactable(calc=True)
        @this.record_context
        def Results():
            return _analyze(Incoming(), catalogue, versions, use_weights=bool(input.UseWeights()),
                            forecasting=bool(input.Forecasting()), output=input.Output() or '')

        @this.reactable(calc=True)
        def Display():
            return _table(Results(), show_all=bool(input.ShowAll()), installed_only=bool(input.InstalledOnly()))


        def pluralize(word, count):
            return word if count == 1 else f"{word}s"


        @output
        @render.ui
        def Summary():
            result = Results()
            predcnt=result['summary']['predictors']
            kind = result['summary']['predictor_kind']
            predictors = ' '.join(part for part in (str(predcnt), kind, pluralize('predictor', predcnt)) if part)
            alerts = [note for note in result['notes'][:-2]
                      if not note.startswith('Evaluation not ready:')]
            return ui.TagList(
                ui.span(f"{result['task'].title()} · {', '.join(result['structure'])} · {predictors}",
                        class_="text-primary text-center d-block"),
                *(ui.div(icon('circle-info'), ' ', note, class_='model-data-notice') for note in alerts))

        @output
        @render.data_frame
        def Table():
            return render.DataGrid(Display().set_index('ID'), selection_mode='row', filters=False, width='100%')

        @output
        @render.text
        def Status():
            count = len(Display())
            return f'{count} methods shown · Select a row and flip to compare recipes.' if count else 'No matching methods. Show excluded and unreviewed recipes for explanations.'

        @output
        @render.ui
        def RecipeDetails():
            selected = Table.data_view(selected=True)
            if selected.empty:
                return ui.div(icon('hand-pointer'), ui.h5('Choose a method to inspect'), ui.p('Select a row on the front, then flip back for its recipe.'), class_='model-recipe model-recipe-empty')
            # The stable ID is the hidden index, retained through sorting/filtering.
            selected_id = selected.index[0]
            groups = _method_groups(Results(), show_all=bool(input.ShowAll()),
                                    installed_only=bool(input.InstalledOnly()))
            rows = groups.get(selected_id, [])
            req(rows)
            if len(rows) == 1:
                return _recipe_details(rows[0], catalogue['catalogue_version'])
            return ui.div(
                ui.p(f'{len(rows)} recipes match the current display filters. '
                     'The table summarizes the first recipe; evidence and conditions apply separately to each.',
                     class_='model-recipe-muted'),
                *(ui.tags.details(
                    ui.tags.summary(_display_label(row['profile'].get('label', row['id'].split(':')[-1])),
                                    ' · ', _readiness(row)),
                    _recipe_details(row, catalogue['catalogue_version']),
                    open=(index == 0), class_='model-method-recipe')
                  for index, row in enumerate(rows)), class_='model-method-recipes')

        return Incoming

    this.server = server
    return this


if Module.running_directly(name=__name__):
    from roles import RoleMap
    this = instance()
    this._imports.set(proxy_data(pd.DataFrame({'x': [1., 2., np.nan], 'y': [2., 4., 6.]}),
                               _roles=RoleMap.from_primitive({'predictor': ['x'], 'target': ['y']})))
    this.run()
