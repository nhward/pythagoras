"""Tree-guided manual imputation with fixed, reviewable pipeline rules."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

if __name__ == '__main__':
    ROOT = Path(__file__).resolve().parent.parent
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from card import Card
from code_recording import recordable
from faicons import icon_svg as icon
from ManualImputationTransformer import (
    ManualImputationTransformer,
    manual_feature,
    manual_kind,
    manual_levels,
    manual_value,
)
from module import Module
from proxy_data import proxy_data
from roles import Role
from shiny import reactive, render, req, ui
from shiny.types import SilentException, SilentOperationInProgressException
from sklearn.tree import DecisionTreeClassifier


@recordable
def eligible_columns(source):
    targets = set(source.role_map.columns_with_role(Role.TARGET))
    allowed, skipped = [], {}
    for column in source.frame:
        if column in targets or str(column).startswith(Card.SHADOW_PREFIX):
            continue
        if source.frame[column].isna().any():
            if manual_kind(source.frame[column]) == 'unsupported':
                skipped[column] = 'Ordered, Cyclic, or unsupported complex data type'
            else:
                allowed.append(column)
    return allowed, skipped


@recordable
def propose_tree(source, column, max_depth=3):
    """Descriptive proposal only. No target-role values or self-missingness."""
    frame = source.frame
    if column not in eligible_columns(source)[0]:
        raise ValueError('Select an eligible incomplete non-Target variable.')
    excluded = set(source.role_map.columns_with_role(Role.TARGET)) | {column}
    features, arrays = [], []
    omitted = []
    for name in frame:
        if name in excluded or str(name).startswith(Card.SHADOW_PREFIX):
            continue
        kind = manual_kind(frame[name])
        if kind == 'unsupported':
            omitted.append(name)
            continue
        if kind == 'categorical':
            levels = manual_levels(frame[name])
            # Every level is eligible, including high-cardinality Identity fields.
            minimum = max(1, int(np.ceil(.02 * len(frame))))
            missing_count = int(frame[name].isna().sum())
            counts = frame[name].value_counts(dropna=True)
            for level in levels:
                count = int(counts.get(level, 0))
                # A unique string ID without missing values cannot form a
                # legal 2%-sized leaf. Avoid allocating its useless dummy columns.
                if not ((count >= minimum and len(frame) - count >= minimum) or
                        (count + missing_count >= minimum and len(frame) - count - missing_count >= minimum)):
                    continue
                feature = {'column': name, 'kind': kind, 'level': level, 'levels': levels}
                features.append(feature)
                arrays.append(manual_feature(frame, feature))
        else:
            feature = {'column': name, 'kind': kind}
            features.append(feature)
            arrays.append(manual_feature(frame, feature))
    truth = frame[column].isna().to_numpy(dtype=bool)
    nodes = []
    model = None
    if arrays and len(frame) and np.unique(truth).size > 1:
        matrix = np.column_stack(arrays)
        model = DecisionTreeClassifier(max_depth=int(max_depth), min_samples_leaf=0.02, random_state=2025)
        model.fit(matrix, truth)
        assignment = model.apply(matrix)
    else:
        assignment = np.zeros(len(frame), dtype=int)

    def visit(node, depth, conditions):
        tree = None if model is None else model.tree_
        leaf = tree is None or tree.children_left[node] == tree.children_right[node]
        item = {'id': node, 'depth': depth, 'conditions': conditions, 'leaf': leaf}
        nodes.append(item)
        if leaf:
            rows = assignment == node
            item.update(rows=int(rows.sum()), missing=int(truth[rows].sum()))
        else:
            threshold = float(tree.threshold[node])
            condition = {'feature': features[tree.feature[node]],
                         'threshold': threshold if np.isfinite(threshold) else None,
                         'missing_split': not np.isfinite(threshold),
                         'missing_left': bool(tree.missing_go_to_left[node])}
            item['split'] = condition
            item['left'], item['right'] = int(tree.children_left[node]), int(tree.children_right[node])
            visit(item['left'], depth + 1, conditions + [dict(condition, left=True)])
            visit(item['right'], depth + 1, conditions + [dict(condition, left=False)])
    visit(0, 0, [])
    # JSON primitives are also the persisted transformer configuration.
    return json.loads(json.dumps({'column': column, 'kind': manual_kind(frame[column]), 'nodes': nodes,
                                  'omitted': omitted, 'max_depth': int(max_depth)}, default=lambda v: v.item()))


def source_fingerprint(source):
    targets = set(source.role_map.columns_with_role(Role.TARGET))
    frame = source.frame.loc[:, [c for c in source.frame if c not in targets]]
    payload = pd.util.hash_pandas_object(frame.astype(str), index=True).values.tobytes()
    payload += pd.util.hash_pandas_object(frame.isna(), index=True).values.tobytes()
    schema = repr([(str(c), repr(source.frame[c].dtype)) for c in source.frame]) + repr(source.role_map.to_primitive())
    return hashlib.sha256(payload + schema.encode()).hexdigest()


def proposal_key(proposal):
    return hashlib.sha256(json.dumps(proposal, sort_keys=True).encode()).hexdigest()[:20]


def condition_label(condition):
    feature = condition['feature']
    name = feature['column']
    if condition.get('missing_split'):
        return f'{name} is observed?'
    if feature['kind'] == 'categorical':
        return f"{name} = {feature['level']}?"
    threshold = condition['threshold']
    if feature['kind'] == 'datetime':
        value = str(pd.Timestamp(threshold * 86400000000000))
    else:
        value = f'{threshold:.6g}'
    return f'{name} ≤ {value}?'


def branch_label(condition, left):
    yes = not left if condition['feature']['kind'] == 'categorical' and not condition.get('missing_split') else left
    return 'Yes' if yes else 'No'


def condition_sentence(condition):
    """Readable observed-value condition; routing stays in optional details."""
    feature = condition['feature']
    name = feature['column']
    if condition.get('missing_split'):
        return f"{name} is {'observed' if condition['left'] else 'missing'}"
    if feature['kind'] == 'categorical':
        relation = 'does not equal' if condition['left'] else 'equals'
        return f"{name} {relation} {json.dumps(feature['level'], ensure_ascii=False)}"
    threshold = condition['threshold']
    value = str(pd.Timestamp(threshold * 86400000000000)) if feature['kind'] == 'datetime' else f'{threshold:.6g}'
    relation = 'is at most' if condition['left'] else 'is greater than'
    return f'{name} {relation} {value}'


def status_text(rules, filled, remaining):
    return (f"{rules} committed {'rule' if rules == 1 else 'rules'} · "
            f"{filled} missing {'value' if filled == 1 else 'values'} filled · "
            f"{remaining} {'remains' if remaining == 1 else 'remain'}. Unchecked leaves stay missing.")


def leaf_qualifies(node, minimum_accuracy):
    """Positive-class leaf purity; a tied leaf predicts non-missingness."""
    return (node['leaf'] and node['rows'] > 0 and node['missing'] * 2 > node['rows']
            and node['missing'] / node['rows'] + 1e-12 >= minimum_accuracy)


def rules_from_drafts(source, proposal, drafts, minimum_accuracy=0.5):
    """Validate a complete variable draft before any commitment is changed."""
    import datetime
    column = proposal['column']
    series = source.frame[column]
    key = proposal_key(proposal)
    extra = proposal.get('extra_level', '')
    rules = []
    for node in proposal['nodes']:
        if not leaf_qualifies(node, minimum_accuracy):
            continue
        draft = drafts.get(f"{key}_{node['id']}", {})
        if not draft.get('enabled'):
            continue
        try:
            value = draft.get('value')
            levels = None
            if proposal['kind'] == 'categorical':
                levels = manual_levels(series)
                if extra and extra not in levels:
                    levels.append(extra)
                if value is None or value == '':
                    raise ValueError('Choose a replacement category.')
                value = levels[int(value)]
            if proposal['kind'] == 'integer' and value is not None and abs(float(value)) > 2**53 - 1:
                raise ValueError('Whole-number entry exceeds the exact browser integer range (±9,007,199,254,740,991).')
            if proposal['kind'] == 'datetime':
                time = datetime.time.fromisoformat(draft.get('time') or '00:00:00').isoformat()
                if not value:
                    raise ValueError('Choose a replacement date.')
                value = f'{value}T{time}'
            value = manual_value(series, value, extra)
            if isinstance(value, np.generic):
                value = value.item()
            rules.append({'column': column, 'kind': proposal['kind'], 'conditions': node['conditions'],
                          'value': value, 'extra_level': extra, 'levels': levels})
        except (ValueError, TypeError, OverflowError, IndexError) as error:
            raise ValueError(f'{column}: {error}') from error
    return rules


@recordable
def apply_rules(source, rules, name='miss_manual'):
    excluded = set(source.role_map.columns_with_role(Role.TARGET))
    for rule in rules:
        used = {rule['column']} | {c['feature']['column'] for c in rule['conditions']}
        if used & excluded:
            raise ValueError('Manual imputation cannot use Target-role variables.')
    if not rules:
        return source.with_inactive_step(stage='Learning', card=name, operation='Manual imputation')
    transformer = ManualImputationTransformer(rules=rules)
    preview = transformer.fit_transform(source.frame)
    return source.with_pipeline_step(transformer, name=name, operation='Manual imputation', preview_frame=preview)


STYLE = """
.flip-container:has(.miss-manual-variables):not(.flipped) > .flippable > .back {visibility:hidden;pointer-events:none;}
.flip-container:has(.miss-manual-variables).flipped > .flippable > .front {visibility:hidden;pointer-events:none;}
.front:has(.miss-manual-variables) {display:flex;flex-direction:column;overflow:hidden;}
.front:has(.miss-manual-variables) > .form-group {flex:0 0 auto;margin-bottom:8px;}
.miss-manual-variables > div, .miss-manual-variables .tab-content {display:flex;flex-direction:column;flex:1;min-height:0;}
.miss-manual-variables .tab-pane.active {display:flex;flex-direction:column;flex:1;min-height:0;}
.miss-manual-variables .navbar {flex:0 0 auto;}
.miss-manual-scroll {overflow:auto; flex:1 1 auto; min-height:0; height:100%; touch-action:pan-x pan-y; position:relative;}
.miss-manual-tree {position:relative; isolation:isolate; margin-inline:auto;}
.miss-manual-tree svg {position:absolute; inset:0; pointer-events:none; z-index:-1;}
.miss-manual-node {position:absolute; box-sizing:border-box; background:var(--bs-body-bg,white);
 border:1px solid var(--bs-border-color,#bbc); border-radius:8px; padding:10px; font-size:12px;}
.miss-manual-split {text-align:center; overflow:auto; max-height:96px;}
.miss-manual-leaf {border-top:4px solid var(--bs-primary,#2670ae); min-height:0;}
.miss-manual-leaf .form-group {margin-bottom:6px;}
.miss-manual-leaf label {font-size:12px;}
.miss-manual-branch {color:var(--bs-secondary-color,#666); font-size:11px;}
"""


def instance():
    this = Card(file=__file__, mutable=True)
    this.long_name = 'Manual imputation'
    this.description = 'Review missingness branches and supply optional, typed replacements.'
    this.export_inputs('MaxTreeDepth', 'MinLeafAccuracy', 'ExtraLevel')
    this.register_configuration_state_field('manual')
    this.defer_configuration_input('Variable')

    this.front = lambda: ui.TagList(
        ui.tags.style(STYLE),
        ui.output_ui('Variables', class_='miss-manual-variables', style='flex:1;min-height:0;display:flex;flex-direction:column;'),
    )

    this.back = lambda: ui.TagList(
        ui.h6('Committed rules'),
        ui.output_ui('Rules')
    )

    this.footer = lambda: ui.div(
        ui.div(
            ui.input_action_button(
                id='Commit', label='Commit rules', icon=icon("gavel", title="Commit", a11y="sem"), width="220px",
                class_="btn rounded-pill btn-sm btn-primary", style="border: 0px; box-shadow: none;",
                guide=this, position = "top", text="<strong>Commit</strong> applies completed drafts across all variable tabs together. An invalid enabled field prevents the whole commit."
            ),
            ui.input_action_button(
                id='Reset', label="Reset", icon=icon("arrow-rotate-left", title="Reset", a11y="sem"),
                class_="btn rounded-pill btn-sm btn-primary", style="border: 0px; box-shadow: none;",
                guide=this, title = "Reset rules", position = "top", text="<strong>Reset</strong> removes all committed rules, unchecks every leaf, and clears replacement fields across all variables. Incoming data is restored; settings stay unchanged."
            ),
            id="Commit-Reset",
            class_="btn-group mx-auto text-center gap-2",
        ),
        ui.output_text('Status'), class_='vertically-scrollable-footer',
    )

    this.settings = lambda: ui.TagList(
        ui.input_slider('MaxTreeDepth', label='Maximum Tree depth parameter', min=1, max=5, value=3, step=1,
                        guide=this, text='Controls proposal complexity, as in Missingness types. Existing committed rules stay fixed until replaced by another commit.', position='left'),
        ui.input_slider('MinLeafAccuracy', label='Minimum leaf accuracy', min=0.5, max=1.0, value=0.8, step=0.05,
                        guide=this, text='Show fields only in leaves predicting missingness where at least this fraction of rows are missing. Hide variable tabs with no qualifying leaves. This is descriptive accuracy on the incoming data; committed rules stay fixed until another commit or Reset.', position='left'),
        ui.input_text('ExtraLevel', label='Additional categorical level (optional)', value='',
                      guide=this, text='Makes this label available for unordered categorical variables. It is added only when selected in a committed rule.', position='left'),
    )

    def server(input, output, session):
        stored = this.restored_configuration_field('manual', {})
        state = reactive.Value(stored or {'fingerprint': None, 'rules': [], 'drafts': {}})
        this.register_configuration_state_provider('manual', lambda: deepcopy(state()))

        @this.reactable(calc=True)
        def Source():
            try:
                source = this.input_data()
            except SilentOperationInProgressException:
                req(False)
            req(source is not None)
            return source

        @reactive.calc
        def Fingerprint():
            return source_fingerprint(Source())

        @reactive.calc
        def State():
            current = state()
            if current.get('fingerprint') != Fingerprint():
                return {'fingerprint': Fingerprint(), 'rules': [], 'drafts': {}}
            return current

        @reactive.effect
        def reconcile():
            current = State()
            if current != state():
                if state().get('fingerprint') is not None:
                    ui.notification_show('Incoming data or roles changed; manual imputation decisions were cleared.', type='warning')
                state.set(current)

        generation = reactive.Value(0)
        restored_variable = this.restored_configuration_input('Variable', None)
        registered_trees = set()

        def tree_id(column):
            return 'Tree_' + hashlib.sha256(str(column).encode()).hexdigest()[:16]

        @output
        @render.ui
        def Variables():
            _, skipped = eligible_columns(Source())
            columns = VisibleColumns()
            if not columns:
                return ui.p(f'No variables have a leaf predicting missingness with accuracy ≥ {MinimumAccuracy():.0%}.' + (' Skipped: ' + ', '.join(skipped) if skipped else ''))
            with reactive.isolate():
                try:
                    selected = input.Variable() or restored_variable
                except SilentException:
                    selected = restored_variable
            panels = []
            for column in columns:
                panels.append(ui.nav_panel(
                    column, 
                    ui.output_ui(
                        tree_id(column), 
                        class_='miss-manual-output',
                        style='flex:1;min-height:0;display:flex;flex-direction:column;'
                    ), 
                    value=column
                ))
            return ui.navset_bar(
                *panels, 
                id='Variable', 
                selected=selected if selected in columns else columns[0],
                title=None, 
                padding=0, 
                fillable=True,
                navbar_options=ui.navbar_options(
                    collapsible=not this.isFullScreen() and len(panels) > this.config.get("settings", {}).get("max_collapse", 6),
                    underline = True,
                    bg="lightgrey",
                    class_=(
                        "nominal-navbar force-collapse"
                        if not this.isFullScreen() and len(panels) > this.config.get("settings", {}).get("max_collapse", 6)
                        else "nominal-navbar"
                    )
                )
            )

        @this.reactable(calc=True)
        @this.settle(seconds=0.3)
        def Depth():
            return int(input.MaxTreeDepth() or 3)

        @reactive.calc
        def ProposalCache():
            return Source(), Depth(), (input.ExtraLevel() or '').strip(), Fingerprint(), {}

        @this.record_context
        def proposal_for(column):
            source, depth, extra, fingerprint, cache = ProposalCache()
            if column not in cache:
                proposal = propose_tree(source, column, depth)
                proposal['source'] = fingerprint
                proposal['extra_level'] = extra
                cache[column] = proposal
            return cache[column]

        @this.reactable(calc=True)
        @this.settle(seconds=0.3)
        def MinimumAccuracy():
            value = input.MinLeafAccuracy()
            return float(value if value is not None else 0.8)

        @reactive.calc
        def VisibleColumns():
            minimum = MinimumAccuracy()
            return [column for column in eligible_columns(Source())[0]
                    if any(leaf_qualifies(n, minimum) for n in proposal_for(column)['nodes'])]

        bound = set()

        def editor_id(key, leaf, field, revision=None):
            revision = generation() if revision is None else revision
            return f'Leaf_{key}_{leaf}_{field}_r{revision}'

        def bind_editor(key, leaf, kind):
            revision = generation()
            if (key, leaf, revision) in bound:
                return
            bound.add((key, leaf, revision))

            @reactive.effect
            def remember():
                if generation() != revision:
                    return
                enabled = input[editor_id(key, leaf, 'Enabled', revision)]()
                value = input[editor_id(key, leaf, 'Value', revision)]()
                time = input[editor_id(key, leaf, 'Time', revision)]() if kind == 'datetime' else None
                if hasattr(value, 'isoformat'):
                    value = value.isoformat()
                if enabled is None:
                    return
                with reactive.isolate():
                    current = deepcopy(State())
                    draft = {'enabled': bool(enabled), 'value': value, 'time': time}
                    token = f'{key}_{leaf}'
                    if current['drafts'].get(token) != draft:
                        current['drafts'][token] = draft
                        state.set(current)

        def leaf_input(proposal, key, node):
            leaf = node['id']
            bind_editor(key, leaf, proposal['kind'])
            with reactive.isolate():
                draft = State()['drafts'].get(f'{key}_{leaf}', {})
            enabled = bool(draft.get('enabled', False))
            value = draft.get('value')
            ident = editor_id(key, leaf, 'Value')
            kind = proposal['kind']
            series = Source().frame[proposal['column']]
            if kind == 'categorical':
                levels = manual_levels(series)
                extra = (input.ExtraLevel() or '').strip()
                if extra and extra not in levels:
                    levels.append(extra)
                # Index tokens preserve the actual type of category labels.
                choices = {'': 'Choose a category'} | {str(i): str(v) for i, v in enumerate(levels)}
                control = ui.input_select(ident, label='Replacement', choices=choices, selected=value if value in choices else '')
            elif kind == 'boolean':
                control = ui.input_select(ident, label='Replacement', choices={'': 'Choose a value', 'true': 'True', 'false': 'False'}, selected=value or '')
            elif kind == 'datetime':
                control = ui.TagList(ui.input_date(ident, label='Replacement date', value=value or ''),
                                     ui.input_text(editor_id(key, leaf, 'Time'), label='Replacement time', value=draft.get('time') or '', placeholder='HH:MM:SS (blank = midnight)'))
            else:
                control = ui.input_numeric(ident, label='Whole-number replacement' if kind == 'integer' else 'Numeric replacement', value=value, step=1 if kind == 'integer' else None)
            return ui.TagList(ui.input_checkbox(editor_id(key, leaf, 'Enabled'), label='Fill missing values', value=enabled), control)

        def build_tree(column):
            columns, skipped = eligible_columns(Source())
            notice = ui.p('Skipped: ' + ', '.join(skipped)) if skipped else None
            if not columns:
                return ui.TagList(ui.p('No supported incomplete non-Target variables.'), notice)
            proposal = deepcopy(proposal_for(column))
            key = proposal_key(proposal)
            nodes = {n['id']: n for n in proposal['nodes']}
            leaves = [n for n in nodes.values() if n['leaf']]
            width = max(1, len(leaves)) * 250
            leaf_y = max(n['depth'] for n in nodes.values()) * 150 + 20
            leaf_height = 265 if proposal['kind'] == 'datetime' else 165
            for i, node in enumerate(leaves):
                node['x'], node['y'] = i * 250 + 125, leaf_y

            def position(node):
                if not node['leaf']:
                    left, right = nodes[node['left']], nodes[node['right']]
                    position(left)
                    position(right)
                    node['x'] = (left['x'] + right['x']) / 2
                    node['y'] = node['depth'] * 150 + 20
            position(nodes[0])
            edges, panels = [], []
            for node in nodes.values():
                x, y = node['x'], node['y']
                if node['leaf']:
                    percentage = node['missing'] / node['rows'] if node['rows'] else 0
                    body = ui.TagList(ui.strong(f"{node['missing']} missing / {node['rows']} {'row' if node['rows'] == 1 else 'rows'} ({percentage * 100:.3g}%)"),
                                      leaf_input(proposal, key, node) if leaf_qualifies(node, MinimumAccuracy()) else ui.p('No missing values to fill.' if not node['missing'] else 'Does not meet the missingness accuracy requirement.'))
                    panels.append(ui.div(body, class_='miss-manual-node miss-manual-leaf', style=f'left:{x-115}px;top:{y}px;width:230px;', **{'data-leaf': str(node['id'])}))
                else:
                    missing_side = branch_label(node['split'], node['split']['missing_left'])
                    panels.append(ui.div(ui.strong(condition_label(node['split'])),
                                         ui.div(f'Missing → {missing_side}', class_='miss-manual-branch'),
                                         class_='miss-manual-node miss-manual-split', style=f'left:{x-110}px;top:{y}px;width:220px;'))
                    for side, left in [('left', True), ('right', False)]:
                        label = branch_label(node['split'], left)
                        child = nodes[node[side]]
                        mid = y + 105
                        edges.append(f'<path d="M {x} {y+70} V {mid} H {child["x"]} V {child["y"]}" fill="none" stroke="currentColor" opacity=".5"/>')
                        edges.append(f'<text x="{(x+child["x"])/2}" y="{mid-5}" text-anchor="middle" fill="currentColor" font-size="12">{label}</text>')
            svg = ui.HTML(f'<svg width="{width}" height="{leaf_y+leaf_height}" aria-hidden="true">' + ''.join(edges) + '</svg>')
            unavailable = [c for c in proposal['omitted'] if c not in skipped]
            omitted = ui.p('Unavailable split types: ' + ', '.join(unavailable)) if unavailable else None
            return ui.TagList(notice, omitted, ui.div(ui.div(svg, *panels, class_='miss-manual-tree', style=f'width:{width}px;height:{leaf_y+leaf_height}px;'), class_='miss-manual-scroll'))

        def register_tree(column):
            @output(id=tree_id(column))
            @render.ui
            def variable_tree():
                req(input.Variable() == column)
                return build_tree(column)

        @reactive.effect(priority=100)
        def register_tree_outputs():
            # Register outside the Variables renderer: output registration reads
            # hidden-state inputs, which must not become navbar dependencies.
            for column in eligible_columns(Source())[0]:
                if column not in registered_trees:
                    register_tree(column)
                    registered_trees.add(column)

        @reactive.effect
        @reactive.event(input.Commit)
        def commit():
            this.require_settled(Depth, MinimumAccuracy)
            source = Source()
            columns = VisibleColumns()
            minimum = MinimumAccuracy()
            current = deepcopy(State())
            drafts = current['drafts']
            try:
                # Capture the active inputs directly, including the most recent
                # edit, before validating drafts from inactive variable tabs.
                try:
                    active = input.Variable()
                except SilentException:
                    active = None
                if active in columns:
                    proposal = proposal_for(active)
                    key = proposal_key(proposal)
                    for node in proposal['nodes']:
                        if not leaf_qualifies(node, minimum):
                            continue
                        enabled = input[editor_id(key, node['id'], 'Enabled')]()
                        if enabled is None:
                            raise ValueError('Wait for the current tree controls to finish loading.')
                        value = input[editor_id(key, node['id'], 'Value')]()
                        if hasattr(value, 'isoformat'):
                            value = value.isoformat()
                        drafts[f"{key}_{node['id']}"] = {
                            'enabled': bool(enabled), 'value': value,
                            'time': input[editor_id(key, node['id'], 'Time')]() if proposal['kind'] == 'datetime' else None,
                        }
                rules = list(current['rules'])
                for column in columns:
                    proposal = proposal_for(column)
                    key = proposal_key(proposal)
                    if not any(f"{key}_{n['id']}" in drafts for n in proposal['nodes'] if leaf_qualifies(n, minimum)):
                        continue  # Preserve commitments with no current proposal draft.
                    replacements = rules_from_drafts(source, proposal, drafts, minimum)
                    rules = [r for r in rules if r['column'] != column] + replacements
                apply_rules(source, rules, this.namespace)
            except (ValueError, TypeError, OverflowError, IndexError) as error:
                ui.notification_show(str(error), type='error', duration=8)
                return
            current['rules'] = rules
            state.set(current)

        @reactive.effect
        @reactive.event(input.Reset)
        def reset():
            state.set({'fingerprint': Fingerprint(), 'rules': [], 'drafts': {}})
            # New DOM IDs discard bound browser values. Older observers cannot
            # repopulate cleared drafts with values from an inactive tab.
            generation.set(generation() + 1)

        @this.reactable(calc=True)
        @this.record_context
        def Export():
            return apply_rules(Source(), State()['rules'], this.namespace)

        @output
        @render.text
        def Status():
            source, result = Source(), Export()
            count = int(source.frame.isna().sum().sum() - result.frame.isna().sum().sum())
            return status_text(len(State()['rules']), count, int(result.frame.isna().sum().sum()))

        @output
        @render.ui
        def Rules():
            rules = State()['rules']
            if not rules:
                return ui.p('No committed replacements. Review a tree and commit optional leaf values.')
            panels = []
            for rule in rules:
                conditions = rule['conditions']
                lines = [ui.div(condition_sentence(c) + (' and' if i < len(conditions) - 1 else ''))
                         for i, c in enumerate(conditions)]
                routing = [ui.div(f"{c['feature']['column']}: missing or unseen values follow this branch.")
                           for c in conditions if not c.get('missing_split') and c['left'] == c['missing_left']]
                panels.append(ui.div(
                    ui.strong(f"missing {rule['column']} ← {rule['value']} when"),
                    *lines if lines else [ui.div('Any row')],
                    ui.tags.details(ui.tags.summary('Missing-value handling'), *routing) if routing else None,
                    style='margin-bottom:12px;',
                ))
            return ui.div(*panels, style='overflow:auto;')

        return Export

    this.server = server
    return this


if Module.running_directly(name=__name__):
    this = instance()
    frame = pd.DataFrame({'HEALTHCARE_COST': pd.Series([None, None, 20, 40], dtype='Int64'), 'HEALTHCARE_BASIS': pd.Categorical(['FREE', 'FREE', 'PAID', 'PAID'])})
    this._imports.set(proxy_data(_df=frame, _name='Manual imputation example'))
    this.run()
