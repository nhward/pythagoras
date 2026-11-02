"""Rebuild an upstream evaluation design inside the current training partition."""
from code_recording import recordable


@recordable
def selection_design(source):
    """Export design parameters only: never absolute memberships or reserved rows."""
    plan = source.evaluation_plan
    if plan is None:
        return None
    keys = ('strategy', 'objective', 'group', 'sequence', 'entity', 'stratify',
            'shuffle', 'seed', 'folds', 'requested_folds', 'window', 'horizon',
            'gap', 'step', 'expanding')
    return {k:v for k,v in plan.recipe.items() if k in keys}


@recordable
def selection_inner_splits(frame, y, *, task, design, folds, repeats, seed):
    """Generate local splits through the same engine as obs_resample.

    Temporal horizons/gaps/steps retain their units (distinct sequence values).
    Only the initial training window shrinks when the current partition cannot
    fit the original one. Predefined external validation never becomes an inner
    holdout: derive independent/grouped boundaries or local forward-time blocks.
    """
    import numpy as np
    import pandas as pd
    from evaluation_plan import build_plan
    from modelling_context import ModellingContext
    from proxy_data import proxy_data
    from roles import RoleMap

    recipe = dict(design)
    strategy = recipe.get('strategy')
    notes = []
    derived_temporal = False
    if strategy == 'predefined':
        objective = recipe.get('objective')
        if objective in ('unseen', 'unseen_strata'):
            strategy = 'grouped'
        elif objective in ('future', 'known'):
            derived_temporal = not all(k in recipe for k in ('window', 'horizon'))
            strategy = 'temporal'
        else:
            strategy = 'stratified' if task == 'classification' else 'kfold'
        notes.append(f'Predefined external validation stays outside selection; rebuilt {strategy} inner splits.')
    if strategy not in ('kfold', 'stratified', 'grouped', 'logo', 'temporal'):
        raise ValueError('The upstream plan has no supported inner resampling design')
    metadata = list(dict.fromkeys(recipe[k] for k in ('group','sequence','entity','stratify') if recipe.get(k)))
    missing = [c for c in metadata if c not in frame]
    if missing:
        raise ValueError(f'Inner resampling metadata missing after upstream processing: {missing}')
    local = frame[metadata].copy().reset_index(drop=True)
    target = '__selection_outcome__'
    while target in local:
        target += '_'
    local[target] = y
    roles = RoleMap()
    roles.set_roles(target, ['target'])
    for key in ('group','stratify'):
        if recipe.get(key):
            roles.set_roles(recipe[key], ['stratifier'])
    if recipe.get('sequence'):
        roles.set_roles(recipe['sequence'], ['sequence'])
    if recipe.get('entity'):
        roles.set_roles(recipe['entity'], ['identifier'])
    if strategy == 'stratified' and not recipe.get('stratify'):
        # Predefined independent classification: derive target stratification.
        local[target] = pd.Categorical(local[target])
        recipe['stratify'] = target
    options = {k:v for k,v in recipe.items() if k not in ('strategy','requested_folds')}
    options['strategy'] = strategy
    options['folds'] = recipe.get('requested_folds', recipe.get('folds', folds))
    if strategy == 'temporal':
        column = recipe.get('sequence')
        if not column or column not in local:
            raise ValueError('Temporal inner resampling needs its Sequence column')
        available = local[column].nunique()
        if derived_temporal:
            # Predefined validation supplies a future boundary, not a window
            # size. Divide only current training times into forward blocks.
            gap = recipe.get('gap', 0)
            count = min(options['folds'], available - gap - 1)
            if count < 1:
                raise ValueError('Too few distinct training times for forward inner validation')
            horizon = max(1, (available - gap) // (count + 1))
            recipe.update(window=available-gap-count*horizon, horizon=horizon, gap=gap,
                          step=horizon, expanding=recipe.get('expanding', True))
            options.update({k:recipe[k] for k in ('window','horizon','gap','step','expanding')})
            notes.append(f'Predefined future boundary: derived {count} local forward splits, initial window {recipe["window"]}, horizon {horizon}, gap {gap} distinct times.')
        original = recipe['window']
        window = min(original, available - recipe.get('gap', 0) - recipe['horizon'])
        if window < 1:
            raise ValueError('Too few distinct training times for the inherited inner horizon and gap')
        options['window'] = window
        if window != original:
            notes.append(f'Inner initial window reduced from {original} to {window} distinct times in this training partition; horizon and gap unchanged.')
    source = proxy_data(local, _roles=roles, _modelling_context=ModellingContext(task=task))
    splits, summaries = [], []
    for repeat in range(repeats):
        options['seed'] = int(recipe.get('seed', seed)) + repeat
        plan = build_plan(source, **options)
        if not plan.ready:
            raise ValueError('Inner resampling: ' + '; '.join(plan.errors))
        notes.extend(plan.warnings)
        for index, (train, valid) in enumerate(plan.splits(source), 1):
            train, valid = np.asarray(train), np.asarray(valid)
            splits.append((train, valid))
            summary = {'Repeat':repeat + 1, 'Fold':index, 'Training':len(train), 'Validation':len(valid)}
            if recipe.get('group'):
                values = local[recipe['group']]
                summary['Group overlap'] = len(set(values.iloc[train]) & set(values.iloc[valid]))
            if recipe.get('sequence'):
                values = local[recipe['sequence']]
                summary['Last training time'] = str(values.iloc[train].max())
                summary['First validation time'] = str(values.iloc[valid].min())
            summaries.append(summary)
    return splits, {'strategy':strategy, 'origin':'upstream obs_resample design',
                    'notes':list(dict.fromkeys(notes)), 'summary':pd.DataFrame(summaries)}
