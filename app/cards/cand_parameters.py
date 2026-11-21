"""Review and commit finite search spaces for the current pipeline."""
import json
import re
from copy import deepcopy
from uuid import uuid4

import numpy as np
import pandas as pd
from card import Card
from faicons import icon_svg as icon
from module import Module
from proxy_data import proxy_data
from shiny import reactive, render, req, ui
from shiny.types import SilentOperationInProgressException
from tuning import (
    PURPOSES,
    VISIBILITY,
    compile_spec,
    discover_source,
    fingerprint,
    migrate_drafts,
    restore_commitment,
    search_values,
    visible,
)

#TODO: Reviw quarto

def source_label(source, cards):
    """Display card titles without changing sklearn paths or nested steps."""
    root, *children = source.split(' › ')
    name, suffix = root, ''
    if root not in cards:
        match = re.fullmatch(r'(.+)_(\d+)', root)
        if match and match[1] in cards:
            name, suffix = match[1], f' ({match[2]})'
    title = cards.get(name, {}).get('long_name') or name
    return ' › '.join([title + suffix, *children])


def instance():
    this = Card(file=__file__, mutable=True)
    this.register_configuration_state_field('tuning')
    this.long_name = 'Tuning parameters'
    this.description = 'Mark candidate parameters as tuning hyper-parameters.'
    
    this.front = lambda: ui.TagList(
        ui.div(
            ui.input_text(id='Search', label='Find parameter or source', value=''),
            class_='tuning-search'
        ),
        ui.output_ui(id='Table', fill=True, fillable=True, class_='tuning-parameters-table'), 
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

    this.exclude_configuration_input('ShowEverything')
    for transient in ('RowEvent', 'Active', 'SaveDefinition', 'Kind', 'Mode', 'Values',
                      'ChoiceValues', 'Minimum', 'Maximum', 'Increment'):
        this.exclude_configuration_input(transient)

    def settings():
        # Preserve the meaning of older saved checkbox settings.
        legacy = this.restored_configuration_input('ShowEverything', False, warn_if_missing=False)
        return ui.input_radio_buttons(
            id='Visibility', label='Show parameters', choices=VISIBILITY,
            selected='everything' if legacy else 'recommended',
            guide=this, title='Parameter visibility', position='left',
            text='Recommended shows reviewed modeling choices. Add advanced choices or all tunable parameters, including unreviewed ones. '
                 'Everything also shows operational settings and recipe parameters, which cannot be tuned. '
                 'Recipe parameters prepare model inputs; the upstream pipeline is expected to manage this preparation. '
                 'Already-enabled parameters remain visible. '
                 'Missing search values do not establish that a parameter is unimportant.')
    this.settings = settings

    def server(input, output, session):
        restored = this.restored_configuration_field('tuning', {})
        drafts = reactive.Value(restored.get('drafts', {}) if isinstance(restored, dict) else {})
        pending_commitment = restored.get('committed') if isinstance(restored, dict) else None
        committed = reactive.Value(None)
        message = reactive.Value('')
        editing = reactive.Value(None)
        modal_error = reactive.Value('')
        promoted = reactive.Value(restored.get('promoted', []) if isinstance(restored, dict) else [])
        restore_view_pending = bool(restored)
        def tuning_state():
            current = committed()
            return {'drafts': deepcopy(drafts()), 'promoted': list(promoted()),
                    'committed': deepcopy(pending_commitment if pending_commitment is not None else current)}
        this.register_configuration_state_provider('tuning', tuning_state)

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
            return discover_source(Incoming())

        @this.reactable()
        def RestoreCommitment():
            nonlocal pending_commitment
            if pending_commitment is None:
                return
            rows = Rows()
            saved_spec = pending_commitment.get('spec') if isinstance(pending_commitment, dict) else None
            if not rows and isinstance(saved_spec, dict) and saved_spec.get('grid'):
                return
            saved, pending_commitment = pending_commitment, None
            try:
                committed.set(restore_commitment(rows, saved))
                message.set('Saved search space restored and committed.')
            except (ValueError, TypeError, KeyError) as exc:
                message.set(f'Saved search space needs review: {exc}')

        @this.reactable(triggers=[Rows, drafts])
        def Reconcile():
            known = {row['path']:row for row in Rows()}
            with reactive.isolate():
                previous = drafts()
            migrated = migrate_drafts(Rows(), previous)
            retained = {path:dict(draft, active=False) if known[path]['fixed'] else draft
                        for path,draft in migrated.items()
                        if path in known and known[path]['owner'] == draft.get('owner')}
            if retained != previous:
                drafts.set(retained)
                message.set('Parameter definitions changed: unavailable definitions were removed and tuning of excluded parameters was disabled. Review the search space.')

        def definition(row):
            draft = drafts().get(row['path'])
            if draft is None or draft.get('owner') != row['owner']:
                return dict(deepcopy(row['definition']), owner=row['owner'], active=False)
            return deepcopy(draft)

        def DisplayFrame():
            query = (input.Search() or '').casefold()
            cards = Module.card_catalogue.snapshot()['cards']
            records = []
            version = fingerprint(Rows())
            for row in Rows():
                draft = definition(row)
                if not visible(row, input.Visibility(), active=bool(draft.get('active'))):
                    continue
                source = row.get('display_source') or source_label(row['source'], cards)
                if query not in f'{row["name"]} {source} {row["source"]} {row["path"]}'.casefold(): continue
                try:
                    values = search_values(draft, row['metadata'])
                    summary = f'{len(values)} values: {values[:6]}' + (' …' if len(values) > 6 else '')
                except ValueError:
                    summary = 'Use Edit to supply search values'
                    values = None
                attributes = {'data-tuning-input': this.ns('RowEvent'),
                              'data-tuning-path': row['path'], 'data-tuning-owner': row['owner'],
                              'data-tuning-version': version}
                disabled = row['fixed'] or values is None
                toggle = ui.tags.input(type='checkbox', checked=bool(draft.get('active')),
                    disabled=disabled, class_='form-check-input tuning-row-toggle',
                    title='Fixed configuration' if row['fixed'] else 'Use Edit to define valid search values' if disabled else f'Tune {row["name"]}',
                    **attributes, **{'data-tuning-row-action': 'toggle', 'aria-label': f'Tune {row["name"]}'})
                edit = ui.tags.button('Edit', type='button', class_='btn btn-outline-primary btn-sm tuning-row-edit',
                    title=f'{row["path"]} · {row["owner"]}', **attributes,
                    **{'data-tuning-row-action': 'edit', 'aria-label': f'Edit {row["name"]}'})
                records.append({'ID':row['path'], 'Tune': toggle, 'Edit': edit,
                    'Parameter':row['name'], 'Source':source, 'Current value':row['display_current'],
                    'Purpose':PURPOSES[row['purpose']], 'Reason':row['reason'],
                    'Search values':'Fixed configuration' if row['fixed'] else summary})
            frame = pd.DataFrame(records, columns=['ID','Tune','Edit','Parameter','Source','Search values','Current value','Reason','Purpose'])
            frame = frame.set_index('ID')
            first = [path for path in promoted() if path in frame.index]
            return frame.loc[first + [path for path in frame.index if path not in first]]

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
            input.Visibility()
            with reactive.isolate():
                if restore_view_pending:
                    checked = [row['path'] for row in Rows() if definition(row).get('active')]
                    promoted.set([path for path in promoted() if path in checked] +
                                 [path for path in checked if path not in promoted()])
                frame = DisplayFrame()
            return render.DataGrid(frame, selection_mode='row', width="100%", height='100%')

        async def scroll_to_top():
            await session.send_custom_message('table_scroll_top', {'id': this.ns('Table2')})

        @this.reactable()
        async def RestoreTableView():
            nonlocal restore_view_pending
            Table2.data_view_rows()  # Wait until the browser has bound its grid.
            if restore_view_pending:
                restore_view_pending = False
                await Table2.update_sort([])
                await scroll_to_top()

        @this.reactable(triggers=[drafts])
        async def RefreshDrafts():
            Table2.data()
            await Table2.update_data(DisplayFrame())

        def editor_controls(row, draft):
            detail = ui.TagList(
                ui.div(ui.strong(row['name']), ui.code(row['path']), ui.p(row['owner']), class_='tuning-editor-detail'),
                ui.p(ui.strong('Method default: '), row['default'], ' ',
                    ui.a('Default documentation', href=row['metadata'].get('default_reference', row['reference']),
                         target='_blank', rel='noopener') if row['reference'] else None),
                ui.div(ui.span(f"{PURPOSES[row['purpose']]}: {row['reason']} "),
                    ui.a('Documentation', href=row['reference'], target='_blank', rel='noopener') if row['reference'] else None,
                    class_='tuning-purpose'))
            if row['fixed']:
                return ui.TagList(detail, ui.p('Fixed configuration; not offered for automatic tuning.'))
            choices = row['metadata'].get('choices')
            if row['definition']['kind'] == 'boolean':
                choices = [False, True]
            values = (ui.input_selectize('ChoiceValues', label='Choices', multiple=True,
                choices={json.dumps(v): str(v) for v in choices},
                selected=[json.dumps(v) for v in draft.get('values', [])]) if choices else
                ui.input_text('Values', label='Choices as a JSON list (strings in quotes; null means None)',
                              value=json.dumps(draft.get('values', []))))
            return ui.TagList(detail, ui.div(
                ui.input_select('Kind', label='Parameter type', choices={'': 'Choose a type', 'integer': 'Integer',
                    'float': 'Decimal', 'categorical': 'Categorical', 'boolean': 'Boolean'}, selected=draft.get('kind', '')),
                ui.input_select('Mode', label='Search definition', choices={'values': 'Explicit choices', 'range': 'Numeric range'},
                                selected=draft.get('mode', 'values')),
                ui.panel_conditional("input.Mode === 'range'", ui.div(
                    ui.input_numeric('Minimum', label='Minimum', value=draft.get('min', 0)),
                    ui.input_numeric('Maximum', label='Maximum', value=draft.get('max', 1)),
                    ui.input_numeric('Increment', label='Step', value=draft.get('step', 1)), class_='tuning-range-fields')),
                ui.panel_conditional("input.Mode === 'values'", values), class_='tuning-editor-controls'))

        @output
        @render.text
        def ModalError():
            return modal_error()

        def save_editor(event):
            current = editing()
            if current is None or event.get('token') != current['token']:
                return
            try:
                row = next((r for r in Rows() if r['path'] == current['path']), None)
                if (Incoming() is not current['source'] or row is None or row['owner'] != current['owner']
                        or fingerprint(Rows()) != current['version']):
                    raise ValueError('Upstream parameters changed. Cancel and reopen the current parameter.')
                if not row['fixed']:
                    draft = {'owner': row['owner'], 'kind': input.Kind(), 'mode': input.Mode(),
                             'active': definition(row).get('active', False)}
                    if draft['mode'] == 'range':
                        draft.update(min=input.Minimum(), max=input.Maximum(), step=input.Increment())
                    elif row['definition']['kind'] == 'boolean' or row['metadata'].get('choices'):
                        draft['values'] = [json.loads(v) for v in (input.ChoiceValues() or [])]
                    else:
                        draft['values'] = json.loads(input.Values())
                    search_values(draft, row['metadata'])
                    updated = deepcopy(drafts()); updated[row['path']] = draft; drafts.set(updated)
                    message.set('Search definition saved. Use the row checkbox to enable tuning, then Commit.')
                editing.set(None)
                ui.modal_remove()
            except (ValueError, TypeError, KeyError) as exc:
                modal_error.set(str(exc))

        @this.reactable(triggers=[input.RowEvent])
        def RowEvent():
            event = input.RowEvent()
            if not isinstance(event, dict):
                return
            action = event.get('action')
            if action == 'dismiss':
                current = editing()
                if current and current['token'] == event.get('token'):
                    editing.set(None)
                    ui.modal_remove()
                return
            if action == 'save':
                save_editor(event)
                return
            row = next((r for r in Rows() if r['path'] == event.get('path')), None)
            if row is None or row['owner'] != event.get('owner') or fingerprint(Rows()) != event.get('version'):
                message.set('Parameter definitions changed. Use the current table controls.')
                return
            if action == 'toggle' and not row['fixed'] and isinstance(event.get('checked'), bool):
                draft = definition(row)
                try:
                    if event['checked']:
                        search_values(draft, row['metadata'])
                    if event['checked'] != bool(draft.get('active')):
                        draft['active'] = event['checked']
                        updated = deepcopy(drafts()); updated[row['path']] = draft; drafts.set(updated)
                    message.set('')
                except (ValueError, TypeError) as exc:
                    message.set(str(exc))
            elif action == 'edit':
                token = uuid4().hex
                editing.set({'path': row['path'], 'owner': row['owner'], 'source': Incoming(),
                             'version': fingerprint(Rows()), 'token': token})
                modal_error.set('')
                ui.modal_show(ui.modal(
                    editor_controls(row, definition(row)), ui.output_text('ModalError'),
                    title=('Parameter details: ' if row['fixed'] else 'Edit parameter: ') + row['name'],
                    footer=ui.TagList(ui.modal_button('Cancel'), ui.tags.button('OK', type='button',
                        class_='btn btn-primary', **{'data-tuning-input': this.ns('RowEvent'),
                        'data-tuning-row-action': 'save', 'data-tuning-token': token})),
                    size='l', easy_close=True, **{'data-tuning-modal-input': this.ns('RowEvent'),
                                                'data-tuning-token': token}))

        @this.reactable(triggers=[input.Commit])
        async def Commit():
            nonlocal pending_commitment
            try:
                spec = compile_spec(Rows(), drafts())
                pending_commitment = None
                committed.set({'spec':spec, 'drafts':deepcopy(drafts())})
                message.set('Search space committed. No models have been fitted.')
                order = list(Table2.data_view().index)
                checked = [row['path'] for row in Rows() if definition(row).get('active')]
                promoted.set([path for path in order if path in checked] +
                             [path for path in checked if path not in order])
                frame = DisplayFrame()
                await Table2.update_data(frame)
                await Table2.update_sort([])
                await Table2.update_cell_selection({'type': 'row', 'rows': [0] if len(frame) else []})
                await scroll_to_top()
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
            try:
                valid = saved is not None and saved['spec'] == compile_spec(Rows(), saved['drafts'])
            except (ValueError, TypeError, KeyError):
                valid = False
            if not valid:
                return source.with_inactive_step(
                    stage='Learning', card='cand_parameters',
                    operation='Choose hyper-parameters')
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
