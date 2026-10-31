"""Review and commit finite search spaces for the current pipeline."""
import json
from copy import deepcopy

import numpy as np
import pandas as pd
from card import Card
from faicons import icon_svg as icon
from module import Module
from proxy_data import proxy_data
from shiny import reactive, render, req, ui
from shiny.types import SilentOperationInProgressException
from tuning import compile_spec, discover, fingerprint, search_values


def instance():
    this = Card(file=__file__, mutable=True)
    this.long_name = 'Tuning parameters'
    this.description = 'Define and commit search values for published pipeline parameters.'
    
    this.front = lambda: ui.TagList(
        ui.div(
            ui.input_text(id='Search', label='Find parameter or source', value=''),
            class_='tuning-search'
        ),
        ui.output_ui(id='Table', fill=True, fillable=True, class_='tuning-parameters-table'), 
        ui.output_ui(id='Editor', class_='tuning-parameters-editor'), 
        ui.div(ui.output_text(id='Message'), class_='tuning-parameters-message')
    )
    
    this.back = lambda: ui.TagList(
        ui.span('Search-space review', class_="text-primary text-center d-block"), 
        ui.output_ui('Review', class_='tuning-review-output')
    )
    
    this.footer = lambda: ui.TagList(
        ui.div(
            ui.input_action_button(
                id='Commit', label='Commit', icon=icon("gavel", title="Commit", a11y="sem"), width="220px", 
                class_="btn rounded-pill btn-sm btn-primary", style="border: 0px; box-shadow: none;",
                guide=this, position = "top", text="<strong>Commit</strong> applies the configured tuning to the pipeline. It does not initiate the tuning."
            ),
            ui.input_action_button(
                id='Revert', label=None, icon=icon("arrow-rotate-left", title="Revert", a11y="sem"),
                class_="btn rounded-pill btn-sm btn-primary", style="border: 0px; box-shadow: none;",
                guide=this, title = "Revert", position = "top", text="<strong>Revert</strong> restores the last committed draft"
            ),
            id="Commit-Revert",
            class_="btn-group mx-auto text-center gap-2",
        ),

        ui.output_text(
            id='Status',
            guide=this, title ="Summary", position = "top", text="A summary of the auto-tuning scenario. The speed of training slows with more combinations."
        )
    )

    this.settings = lambda: ui.input_checkbox(
        id='ShowEverything', label='Show everything', value=False,
        guide=this, position='left',
        text='Include fixed-configuration parameters in the table. These cannot be enabled for automatic tuning.')

    def server(input, output, session):
        restored = this.restored_configuration_field('tuning', {})
        drafts = reactive.Value(restored.get('drafts', {}) if isinstance(restored, dict) else {})
        committed = reactive.Value(restored.get('committed') if isinstance(restored, dict) else None)
        message = reactive.Value('')
        this.register_configuration_state_provider('tuning', lambda: {
            'drafts': deepcopy(drafts()), 'committed': deepcopy(committed())})

        @this.reactable(calc=True)
        def Incoming():
            try:
                source = this.input_data()
            except SilentOperationInProgressException:
                req(False)
            req(source is not None)
            return source

        @this.reactable(calc=True)
        def Rows():
            return discover(Incoming().pipeline)

        @this.reactable()
        def Reconcile():
            known = {row['path']:row['owner'] for row in Rows()}
            with reactive.isolate():
                previous = drafts()
            retained = {path:draft for path,draft in previous.items()
                        if known.get(path) == draft.get('owner')}
            if retained != previous:
                drafts.set(retained)
                message.set('Pipeline changed: unavailable parameter definitions were removed. Review and commit again.')

        def definition(row):
            draft = drafts().get(row['path'])
            if draft is None or draft.get('owner') != row['owner']:
                return dict(deepcopy(row['definition']), owner=row['owner'], active=False)
            return deepcopy(draft)

        def DisplayFrame():
            query = (input.Search() or '').casefold()
            records = []
            for row in Rows():
                if row['fixed'] and not input.ShowEverything():
                    continue
                if query not in f'{row["name"]} {row["source"]} {row["path"]}'.casefold(): continue
                draft = definition(row)
                try:
                    values = search_values(draft, row['metadata'])
                    summary = f'{len(values)} values: {values[:6]}' + (' …' if len(values) > 6 else '')
                except ValueError:
                    summary = 'Select row to supply search values'
                records.append({'ID':row['path'], 'Tune': '✅' if draft.get('active') else '❌',
                    'Parameter':row['name'], 'Source':row['source'], 'Current value':row['current'],
                    'Default':row['default'], 'Search values':'Fixed configuration' if row['fixed'] else summary})
            frame = pd.DataFrame(records, columns=['ID','Tune','Parameter','Source','Current value','Default','Search values'])
            return frame.set_index('ID')

        @output
        @render.ui
        def Table():
            req(Incoming())
            return ui.output_data_frame(id = "Table2")

        @output
        @render.data_frame
        def Table2():
            req(Incoming())
            Rows()
            input.Search()
            input.ShowEverything()
            with reactive.isolate():
                frame = DisplayFrame()
            return render.DataGrid(frame, selection_mode='row', width="100%")

        @this.reactable(triggers=[drafts])
        async def RefreshDrafts():
            Table2.data()
            await Table2.update_data(DisplayFrame())

        @this.reactable(calc=True)
        def Selected():
            selected = Table2.data_view(selected=True)
            if selected.empty: return None
            return next((row for row in Rows() if row['path'] == selected.index[0]), None)

        @output
        @render.ui
        def Editor():
            row = Selected()
            if row is None: return ui.p('Select a parameter to define its search values.')
            detail = ui.div(ui.strong(row['name']), title=f'{row["path"]} · {row["owner"]}', class_='tuning-editor-detail')
            if row['fixed']: return ui.TagList(detail, ui.p('Fixed configuration; not offered for automatic tuning.'))
            draft = definition(row)
            try:
                search_values(draft, row['metadata'])
                ready = True
            except ValueError:
                ready = False
            choices = row['metadata'].get('choices')
            if row['definition']['kind'] == 'boolean':
                return ui.div(ui.input_checkbox(
                    id='Active', label='Auto-tune this parameter',
                    value=bool(draft.get('active'))), class_='tuning-editor-controls')
            if choices:
                values_editor = ui.input_selectize(id='ChoiceValues', label='Choices', multiple=True,
                    choices={json.dumps(v):str(v) for v in choices}, selected=[json.dumps(v) for v in draft.get('values',[])])
            else:
                values_editor = ui.input_text(id='Values', label='Choices as a JSON list (strings in quotes; null means None)', value=json.dumps(draft.get('values',[])))
            return ui.TagList(detail, ui.div(
                ui.input_select(id='Kind', label='Parameter type', choices={'':'Choose a type', 'integer':'Integer', 'float':'Decimal', 'categorical':'Categorical', 'boolean':'Boolean'}, selected=draft.get('kind','')),
                ui.input_select(id='Mode', label='Search definition', choices={'values':'Explicit choices', 'range':'Numeric range'}, selected=draft.get('mode','values')),
                ui.panel_conditional("input.Mode === 'range'",
                    ui.div(ui.input_numeric(id='Minimum', label='Minimum', value=draft.get('min',0)),
                    ui.input_numeric(id='Maximum', label='Maximum', value=draft.get('max',1)),
                    ui.input_numeric(id='Increment', label='Step', value=draft.get('step',1)), class_='tuning-range-fields'), class_='tuning-range'),
                ui.panel_conditional("input.Mode === 'values'",
                    values_editor, class_='tuning-values'),
                ui.input_action_button(id='SaveDefinition', label='Save search definition'),
                ui.tags.fieldset(ui.input_checkbox(id='Active', label='Auto-tune this parameter', value=bool(draft.get('active'))), disabled=not ready), class_='tuning-editor-controls'),
                ui.p('No suggested search values are published for this parameter. Choose its type, enter explicit choices (for example [1, 2, 5]) or a numeric range, then save and enable Auto-tune.' if not ready else 'Search definition ready.'))

        @this.reactable(triggers=[input.SaveDefinition])
        def SaveDefinition():
            row = Selected()
            req(row is not None, not row['fixed'])
            try:
                draft = {'owner':row['owner'], 'kind':input.Kind(), 'mode':input.Mode(),
                         'active':definition(row).get('active',False)}
                if draft['mode'] == 'range':
                    draft.update(min=input.Minimum(), max=input.Maximum(), step=input.Increment())
                else:
                    previous = definition(row)
                    if previous.get('kind') == 'boolean' or row['metadata'].get('choices'):
                        draft['values'] = [json.loads(v) for v in (input.ChoiceValues() or [])]
                    else:
                        draft['values'] = json.loads(input.Values())
                search_values(draft, row['metadata'])
                updated = deepcopy(drafts()); updated[row['path']] = draft; drafts.set(updated)
                message.set('Search definition saved. Tick Auto-tune to include it.')
            except (ValueError, TypeError) as exc:
                message.set(str(exc))

        @this.reactable(triggers=[input.Active])
        def Active():
            row = Selected()
            req(row is not None, not row['fixed'])
            draft = definition(row)
            if row['definition']['kind'] == 'boolean':
                draft.update(kind='boolean', mode='values', values=[False, True])
            active = bool(input.Active())
            if active == draft.get('active',False): return
            try:
                if active: search_values(draft, row['metadata'])
                draft['active'] = active
                updated = deepcopy(drafts()); updated[row['path']] = draft; drafts.set(updated)
                message.set('')
            except ValueError as exc:
                message.set(str(exc))

        @this.reactable(triggers=[input.Commit])
        def Commit():
            try:
                spec = compile_spec(Rows(), drafts())
                committed.set({'spec':spec, 'drafts':deepcopy(drafts())})
                message.set('Search space committed. No models have been fitted.')
            except ValueError as exc:
                message.set(str(exc))

        @this.reactable(triggers=[input.Revert])
        def Revert():
            saved = committed()
            drafts.set(deepcopy(saved['drafts']) if saved else {})
            message.set('Draft restored to the last commitment.' if saved else 'Draft cleared.')

        @output
        @render.text
        def Message(): return message()

        @output
        @render.text
        def Status():
            rows = Rows()
            if not rows: return '0 parameters · No pipeline is available.'
            try:
                spec = compile_spec(rows, drafts())
                saved = committed()
                state = 'Committed' if saved and saved['spec'] == spec and saved['drafts'] == drafts() else 'Uncommitted changes'
                return f'{len(rows)} parameters · {len(spec["grid"])} selected for auto-tuning · {spec["combinations"]:,} possible combinations · {state}'
            except ValueError as exc:
                return f'{len(rows)} parameters · Search needs review: {exc}'

        @output
        @render.ui
        def Review():
            try:
                spec = compile_spec(Rows(), drafts())
                items = [ui.tags.li(ui.code(path), ': ', str(values), f' ({len(values)} values)') for path, values in spec['grid'].items()]
                return ui.TagList(ui.p(f'{spec["combinations"]:,} candidate combinations. This is not the number of cross-validation fits.'),
                    ui.tags.ul(*items) if items else ui.p('No parameters enabled: one fixed configuration.'),
                    ui.tags.details(ui.tags.summary('Technical search specification'),
                        ui.tags.pre(json.dumps(spec, indent=2), tabindex='0',
                                    class_='tuning-specification-scroll')))
            except ValueError as exc: return ui.p(str(exc))

        @this.reactable(calc=True)
        def Exported():
            source = Incoming()
            saved = committed()
            if saved is None or saved['spec']['fingerprint'] != fingerprint(Rows()):
                return source.with_inactive_step(
                    stage='Learning', card='tuning_parameters',
                    operation='Define parameter tuning')
            return source.with_tuning_spec(saved['spec'])

        return Exported
    this.server = server
    return this

if Module.running_directly(name=__name__):
    from roles import RoleMap
    this = instance()
    this._imports.set(proxy_data(pd.DataFrame({'x': [1., 2., np.nan], 'y': [2., 4., 6.]}),
                               _roles=RoleMap.from_primitive({'predictor': ['x'], 'target': ['y']})))
    this.run()
