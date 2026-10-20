"""Human-reviewed nominal-label reconciliation; no learned transformations."""
from __future__ import annotations

import hashlib
import html
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))

import pandas as pd
from card import Card
from code_recording import recordable
from module import Module
from roles import Role
from shiny import reactive, render, req, ui
from shiny.types import SilentException, SilentOperationInProgressException

MAX_LEVELS = 200
MAX_PROPOSALS = 100


@recordable
def eligible_columns(source, include_special=False):
    allowed = {Role.PREDICTOR}
    if include_special:
        allowed |= {Role.TARGET, Role.STRATIFIER}
    return [c for c in source.frame if isinstance(c, str)
            and isinstance(source.frame[c].dtype, pd.CategoricalDtype)
            and not source.frame[c].dtype.ordered
            and not c.startswith(Card.SHADOW_PREFIX)
            and source.role_map.roles_for(c) & allowed
            and not source.role_map.roles_for(c) - allowed
            and all(isinstance(v, str) for v in source.frame[c].cat.categories)]


@recordable
def edit_distance(left, right, insertion=1.0):
    """Levenshtein distance; discounted insertions are used only for prefixes."""
    previous = [j * insertion for j in range(len(right) + 1)]
    for i, a in enumerate(left, 1):
        current = [float(i)]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + insertion, previous[j] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


@recordable
def similarity(source, destination):
    left, right = source.strip().casefold(), destination.strip().casefold()
    score = 1 - edit_distance(left, right) / max(len(left), len(right), 1)
    reason = "Typo: normalized Levenshtein similarity"
    prefix = left.rstrip('.')
    if len(prefix) >= 3 and len(prefix) < len(right) and right.startswith(prefix):
        abbreviated = 1 - edit_distance(prefix, right, insertion=0.25) / max(len(right), 1)
        if abbreviated > score:
            score, reason = abbreviated, "Abbreviation: matching prefix; discounted omitted suffix"
    if left == right:
        reason = "Same label after trimming whitespace and ignoring case"
    return score, reason


def proposal(column, sources, destination, reason, score=1.0, kind="merge"):
    sources = sorted(set(sources))
    key = hashlib.sha256(json.dumps([column, sources, destination], ensure_ascii=False).encode()).hexdigest()[:20]
    return {"key": key, "column": column, "sources": sources, "destination": destination,
                "reason": reason, "score": score, "kind": kind}


@recordable
def proposals(frame, column, expected=(), minimum=0.65, rare=0.02):
    counts = frame[column].value_counts().loc[lambda s: s > 0].to_dict()
    if len(counts) > MAX_LEVELS:
        return []
    candidates = []
    for source in sorted(counts):
        for destination in sorted(set(counts) | set(expected)):
            if source == destination:
                continue
            if destination not in expected and counts.get(destination, 0) <= counts[source]:
                continue
            if source in expected:
                continue
            score, reason = similarity(source, destination)
            if score >= minimum:
                candidates.append(proposal(column, [source], destination, reason, score))
    candidates.sort(key=lambda p: (-p['score'], p['sources'], p['destination']))
    candidates = candidates[:MAX_PROPOSALS]
    total = sum(counts.values())
    small = sorted(c for c, n in counts.items() if total and n / total < rare and c != "Other")
    if small:
        reason = "Pool low-frequency levels; these labels are not necessarily equivalent"
        if expected and "Other" not in expected:
            reason += "; Other is outside the expected vocabulary"
        candidates.append(proposal(column, small, "Other", reason, 0, "pool"))
    return candidates


@recordable
def visible_proposals(items, decisions):
    eliminated = {}
    for p in items:
        if decisions.get(p['key']) == 'accept':
            eliminated.setdefault(p['column'], set()).update(p['sources'])
    return [p for p in items if decisions.get(p['key']) == 'accept'
            or not (set(p['sources']) | {p['destination']}) & eliminated.get(p['column'], set())]


@recordable
def apply_merges(frame, items):
    result = frame.copy()
    for column in dict.fromkeys(p['column'] for p in items):
        mapping = {}
        for p in items:
            if p['column'] != column:
                continue
            for label in p['sources']:
                if label in mapping and mapping[label] != p['destination']:
                    raise ValueError(f"Conflicting destinations for {label}")
                mapping[label] = p['destination']
        def resolve(label, mapping):
            seen = set()
            while label in mapping:
                if label in seen:
                    raise ValueError("Merge decisions contain a cycle")
                seen.add(label)
                label = mapping[label]
            return label
        mapping = {key: resolve(key, mapping) for key in mapping}
        series = result[column]
        categories = list(dict.fromkeys(mapping.get(c, c) for c in series.cat.categories))
        values = [v if pd.isna(v) else mapping.get(v, v) for v in series.array]
        result[column] = pd.Categorical(values, categories=categories, ordered=False)
    return result


def fresh_state(fingerprint=None):
    return {"fingerprint": fingerprint, "batches": [], "ledger": [], "decisions": {}, "manual": [], "expected": {}}


def source_fingerprint(source):
    columns = eligible_columns(source, True)
    payload = [source.name, source.has_pipeline, source.frame[columns].to_json(orient='split'),
               [(c, list(source.frame[c].cat.categories), sorted(r.value for r in source.role_map.roles_for(c))) for c in columns]]
    return hashlib.sha256(json.dumps(payload, default=str).encode()).hexdigest()


@recordable
def committed_frame(source, state):
    frame = source.frame
    for batch in state['batches']:
        frame = apply_merges(frame, batch)
    return frame


def bar_chart(series, expected=()):
    counts = series.value_counts().loc[lambda s: s > 0]
    labels = sorted(set(counts.index) | set(expected), key=lambda c: (-counts.get(c, 0), c))
    if not labels:
        return ui.p("No observed or expected levels.")
    width, height = max(360, len(labels) * 85), 220
    peak = max(counts.max() if len(counts) else 0, 1)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Level frequencies" width="{width}" height="{height}">']
    for i, label in enumerate(labels):
        x, count = 20 + i * (width - 40) / len(labels), int(counts.get(label, 0))
        bar = 130 * count / peak
        parts.append(f'<rect x="{x}" y="{150-bar}" width="45" height="{bar}" fill="#154c79"><title>{html.escape(label)}: {count}</title></rect>')
        parts.append(f'<text x="{x+22}" y="{143-bar}" text-anchor="middle" fill="currentColor" font-size="11">{count}</text>')
        parts.append(f'<text transform="translate({x+22},165) rotate(25)" fill="currentColor" font-size="11"><title>{html.escape(label)}</title>{html.escape(label[:24])}</text>')
    return ui.div(ui.HTML(''.join(parts) + '</svg>'), style="overflow-x:auto; flex-shrink:0;")


def instance():
    this = Card(file=__file__, mutable=True)
    this.long_name = "Nominal variables"
    this.description = "Review and commit nominal label corrections and rare-level pooling."
    this.front = lambda: ui.div(
        ui.output_ui('Panels'), style='flex:1;min-height:0;overflow:auto;',
    )
    this.back = lambda: ui.output_ui('History')
    this.footer = lambda: ui.TagList(
        ui.div(
            ui.input_action_button(
                id='Commit', label='Commit merges', class_='btn rounded-pill btn-sm btn-primary', 
                guide = this, position="top", text="Commits the accepted nominal-level merging proposals by performing the merges. Rejected proposals are discraded. Only a reset can reverse this."
            ),
            ui.input_action_button(
                id='Reset', label='Reset', class_='btn rounded-pill btn-sm btn-primary',
                guide = this, position="top", text="Reset restores variable levels and clears decisions; expectations are retained. Automatic proposals are regenerated."
            ),
            class_='d-flex flex-wrap justify-content-center gap-2',
        ),
        ui.output_ui(
            id='Status', title = "Status",
            guide = this, position="top", text="The status of the nominal-level merging process."
        )
    )
    this.settings = lambda: ui.TagList(
        ui.input_slider(id='Similarity', label='Minimum similarity', min=0.5, max=1, value=0.65, step=0.05),
        ui.input_slider(id='Rare', label='Low-frequency threshold (%)', min=0, max=10, value=2, step=0.5),
        ui.input_checkbox(id='Special', label='Include nominal Targets and Stratifiers', value=False),
    )

    def server(input, output, session):
        restored = this.restored_configuration_field('reconciliation', fresh_state())
        state = reactive.Value(restored if isinstance(restored, dict) and set(fresh_state()) <= restored.keys() else fresh_state())
        this.register_configuration_state_provider('reconciliation', lambda: deepcopy(state()))

        @this.reactable(calc=True)
        def Source():
            try:
                value = this.input_data()
            except SilentOperationInProgressException:
                req(False)
            req(value is not None)
            return value

        @reactive.calc
        def State():
            current = state()
            fingerprint = source_fingerprint(Source())
            if current['fingerprint'] != fingerprint:
                return fresh_state(fingerprint)
            return current

        @reactive.effect
        def reconcile_source():
            current = State()
            if state() != current:
                if state()['fingerprint'] is not None:
                    ui.notification_show('Incoming nominal data changed; reconciliation decisions were reset.', type='warning')
                state.set(current)

        @reactive.calc
        def Frame():
            return committed_frame(Source(), State())

        @reactive.calc
        def Items():
            current = State()
            dismissed = {p['key'] for p in current['ledger']}
            items = []
            for column in eligible_columns(Source(), bool(input.Special())):
                expected = current['expected'].get(column, {}).get('levels', [])
                items.extend(proposals(Frame(), column, expected, float(input.Similarity() or .65), float(input.Rare() or 0) / 100))
            # Preserve reviewed proposals if thresholds change; never silently alter an acceptance.
            existing = {p['key']: p for p in items}
            for p in current['manual']:
                if p['column'] in Frame() and set(p['sources']) <= set(Frame()[p['column']].dropna()):
                    existing[p['key']] = p
            return [p for p in existing.values() if p['key'] not in dismissed]

        editor_columns = set()
        drafts = {}

        def editor_id(column, name):
            # Column labels may contain spaces, punctuation, or HTML characters.
            token = hashlib.sha256(column.encode()).hexdigest()[:16]
            return f"Editors_{token}_{name}"

        def bind_editor(column):
            def field(name):
                return input[editor_id(column, name)]()

            @reactive.effect
            def remember_draft():
                values = {name: field(name) for name in
                          ('ExpectedLevels', 'ExpectedCount', 'ManualSources', 'Destination')}
                if all(value is not None for value in values.values()):
                    drafts[column] = values

            @reactive.effect
            @reactive.event(
                input[editor_id(column, 'ExpectedLevels')],
                input[editor_id(column, 'ExpectedCount')],
                ignore_none=False,
            )
            def save_expected():
                req(field('ExpectedLevels') is not None, field('ExpectedCount') is not None)
                req(column in Frame())
                current = deepcopy(State())
                levels = list(dict.fromkeys(line.strip() for line in
                              (field('ExpectedLevels') or '').splitlines() if line.strip()))
                expected = {
                    'levels': levels,
                    'count': len(levels) if levels else max(0, int(field('ExpectedCount') or 0)),
                }
                if current['expected'].get(column, {'levels': [], 'count': 0}) == expected:
                    return
                current['expected'][column] = expected
                state.set(current)

            @reactive.effect
            @reactive.event(input[editor_id(column, 'AddProposal')], ignore_init=True)
            def add_proposal():
                sources = field('ManualSources') or []
                destination = (field('Destination') or '').strip()
                if (column not in Frame() or not sources or not destination
                        or destination in sources
                        or not set(sources) <= set(Frame()[column].dropna())):
                    ui.notification_show('Choose source labels and a different destination.', type='warning')
                    return
                current = deepcopy(State())
                p = proposal(column, sources, destination, 'Manual proposal')
                current['manual'] = [v for v in current['manual'] if v['key'] != p['key']] + [p]
                state.set(current)

        @reactive.effect
        def bind_editors():
            for column in eligible_columns(Source(), True):
                if column not in editor_columns:
                    bind_editor(column)
                    editor_columns.add(column)

        def variable_editor(column):
            expected = State()['expected'].get(column, {})
            defaults = {
                'ExpectedLevels': '\n'.join(expected.get('levels', [])),
                'ExpectedCount': expected.get('count', 0),
                'ManualSources': [], 'Destination': '',
            }
            values = defaults | drafts.get(column, {})
            with reactive.isolate():
                for name in defaults:
                    try:
                        value = input[editor_id(column, name)]()
                        if value is not None:
                            values[name] = value
                    except SilentException:
                        pass
            choices = sorted(Frame()[column].dropna().unique())
            return ui.tags.details(
                ui.tags.summary('Expected levels and manual merges'),
                ui.input_text_area(
                    id=editor_id(column, 'ExpectedLevels'), label='Expected levels (optional)',
                    value=values['ExpectedLevels'],
                    guide=this, position='left', text='One label per line. A supplied list determines the expected count.',
                ),
                ui.input_numeric(
                    id=editor_id(column, 'ExpectedCount'), label='Expected cardinality (0 = unspecified)',
                    value=values['ExpectedCount'], min=0,
                ),
                ui.input_selectize(
                    id=editor_id(column, 'ManualSources'), label='Labels to merge', choices=choices,
                    selected=[v for v in values['ManualSources'] if v in choices], multiple=True,
                ),
                ui.input_text(id=editor_id(column, 'Destination'), label='Destination label', value=values['Destination']),
                ui.input_action_button(id=editor_id(column, 'AddProposal'), label='Add manual proposal'),
                id=session.ns(editor_id(column, 'Dialog')),
                style='flex-shrink:0; margin-bottom:8px;',
            )

        @reactive.effect
        @reactive.event(input.Decision)
        def decide():
            event = input.Decision()
            if not isinstance(event, dict) or event.get('value') not in ('undecided', 'accept', 'reject'):
                return
            current = deepcopy(State())
            available = {p['key']: p for p in visible_proposals(Items(), current['decisions'])}
            if event.get('key') not in available:
                return
            p = available[event['key']]
            current['decisions'][p['key']] = event['value']
            current['manual'] = [v for v in current['manual'] if v['key'] != p['key']] + [p]
            state.set(current)

        @output
        @render.ui
        def Panels():
            columns = eligible_columns(Source(), bool(input.Special()))
            if not columns:
                return ui.p('No unordered, string-labelled nominal variables with eligible roles.')
            current, frame = State(), Frame()
            items = visible_proposals(Items(), current['decisions'])
            panels = []
            for column in columns:
                expected = current['expected'].get(column, {})
                rows = []
                for p in items:
                    if p['column'] != column:
                        continue
                    controls = []
                    for value, label in [('undecided', 'Undecided'), ('accept', 'Accept'), ('reject', 'Reject')]:
                        script = f"Shiny.setInputValue({json.dumps(session.ns('Decision'))},{{key:{json.dumps(p['key'])},value:this.value}},{{priority:'event'}})"
                        attrs = {"type": 'radio', "name": session.ns('decision_' + p['key']), "value": value, "onchange": script}
                        if current['decisions'].get(p['key'], 'undecided') == value:
                            attrs['checked'] = True
                        controls.append(ui.tags.label(ui.tags.input(**attrs), ' ' + label, style='margin-right:8px;white-space:nowrap;'))
                    rows.append(ui.tags.tr(
                        ui.tags.td(', '.join(p['sources']) + ' → ' + p['destination']),
                        ui.tags.td(str(int(frame[column].isin(p['sources']).sum()))),
                        ui.tags.td(p['reason'] + (f" ({p['score']:.0%})" if p['kind'] != 'pool' else '')),
                        ui.tags.td(*controls, style='white-space:normal;min-width:140px;'), **{'data-proposal': p['key']},
                    ))
                count = frame[column].nunique()
                summary = None
                if this.isFullScreen():
                    levels = set(expected.get('levels', []))
                    observed = set(frame[column].dropna())
                    summary = ui.p(f"Observed: {count} · Declared: {len(frame[column].cat.categories)} · Expected: {expected.get('count') or 'unspecified'}"
                                   + (f" · Unexpected: {len(observed-levels)} · Absent: {len(levels-observed)}" if levels else ''))
                if count > MAX_LEVELS:
                    limit = ui.p(f'Automatic suggestions require fewer than {MAX_LEVELS} levels; expand the card to add manual proposals.')
                else:
                    limit = ui.p(
                        'Top similarity proposals, ranked by plausibility.'
                        if rows else 'No unresolved proposals.',
                        class_='small text-muted',
                    )
                table = ui.tags.table(ui.tags.thead(ui.tags.tr(*[ui.tags.th(v) for v in ['Proposed merge', 'Rows', 'Reason', 'Decision']])), ui.tags.tbody(*rows), class_='table table-sm') if rows else None
                panels.append(ui.nav_panel(column, summary, bar_chart(frame[column], expected.get('levels', [])),
                                           variable_editor(column) if this.isFullScreen() else None, limit,
                                           ui.div(table, style='max-height:280px;overflow:auto;'), value=column))
            with reactive.isolate():
                try:
                    selected = input.VariablePanel()
                except SilentException:
                    selected = None
            return ui.navset_tab(
                *panels, id='VariablePanel', selected=selected if selected in columns else columns[0]
            )

        @reactive.effect
        @reactive.event(input.Commit)
        def commit():
            current = deepcopy(State())
            visible = visible_proposals(Items(), current['decisions'])
            accepted = [p for p in visible if current['decisions'].get(p['key']) == 'accept']
            if accepted and Source().has_pipeline:
                ui.notification_show('Commit is unavailable after learning steps. Place Nominal variables before the learning pipeline.', type='error')
                return
            try:
                apply_merges(Frame(), accepted)
            except ValueError as error:
                ui.notification_show(str(error), type='error')
                return
            if accepted:
                current['batches'].append(accepted)
            for p in Items():
                decision = current['decisions'].get(p['key'])
                if decision in ('accept', 'reject'):
                    current['ledger'].append(dict(p, decision=decision))
                    current['decisions'].pop(p['key'], None)
            state.set(current)

        @reactive.effect
        @reactive.event(input.Reset)
        def reset():
            current = fresh_state(source_fingerprint(Source()))
            current['expected'] = deepcopy(State()['expected'])
            current['manual'] = deepcopy(State()['manual'])
            state.set(current)

        @output
        @render.text
        def Status():
            current = State()
            pending = sum(current['decisions'].get(p['key']) == 'accept' for p in Items())
            mess = f"{pending} accepted proposals pending · {len(current['ledger'])} recorded decisions" + (' · Cleaning must precede learning steps.' if Source().has_pipeline else '')
            return ui.span(mess, class_ = "text-primary")

        @output
        @render.ui
        def History():
            current = State()
            rows = [ui.tags.tr(ui.tags.td(p['column']), ui.tags.td(', '.join(p['sources'])), ui.tags.td(p['destination']), ui.tags.td(p['decision']), ui.tags.td(p['reason'])) for p in current['ledger']]
            rows += [ui.tags.tr(ui.tags.td(p['column']), ui.tags.td(', '.join(p['sources'])), ui.tags.td(p['destination']), ui.tags.td('pending ' + current['decisions'][p['key']]), ui.tags.td(p['reason'])) for p in Items() if current['decisions'].get(p['key'], 'undecided') != 'undecided']
            vocabulary = []
            for column, expected in current['expected'].items():
                if column in Frame() and expected.get('levels'):
                    observed, levels = set(Frame()[column].dropna()), set(expected['levels'])
                    vocabulary.append(ui.p(column + ': absent — ' + ', '.join(sorted(levels-observed)) + '; unexpected — ' + ', '.join(sorted(observed-levels))))
            return ui.div(ui.p('Decision history. Reset restores incoming data and clears decisions; expectations are retained.'), *vocabulary,
                          ui.tags.table(ui.tags.thead(ui.tags.tr(*[ui.tags.th(v) for v in ['Variable', 'Sources', 'Destination', 'Decision', 'Reason']])), ui.tags.tbody(*rows), class_='table table-sm'), style='overflow:auto;')

        @this.reactable(calc=True)
        def Export():
            source, current = Source(), State()
            if not current['batches']:
                return source
            return source.with_cleaned_data(Frame(), card='var_nominal', operation='Reconcile nominal labels', parameters={'merges': current['batches']})

        return Export

    this.server = server
    return this


if Module.running_directly(name=__name__):
    from proxy_data import proxy_data
    this = instance()
    this._imports.set(proxy_data(pd.DataFrame({'Day': pd.Categorical(['Monday'] * 30 + ['Mon.'] * 3 + ['Mondya'] + ['Tuesday'] * 20)})))
    this.run()
