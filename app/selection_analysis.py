"""Card-independent preview fitting and recipe export for variable selection."""
from code_recording import recordable


@recordable
def selection_identity(source):
    import pandas as pd
    from joblib import hash as job_hash
    def frame_hash(frame):
        # DataFrame pickle hashes include mutable pandas caches. Hash values,
        # index, ordered schema and categorical definitions explicitly instead.
        try:
            values = pd.util.hash_pandas_object(frame, index=True).to_numpy()
        except TypeError:
            values = tuple(tuple(map(repr, row)) for row in frame.itertuples(index=True, name=None))
        schema = [(c, str(frame[c].dtype),
                   tuple(map(repr, frame[c].cat.categories)) if isinstance(frame[c].dtype, pd.CategoricalDtype) else None,
                   frame[c].cat.ordered if isinstance(frame[c].dtype, pd.CategoricalDtype) else None) for c in frame]
        return job_hash((values, schema, tuple(frame.index.names)))
    return job_hash((frame_hash(source.frame), frame_hash(source.clean_frame),
        source.role_map.to_primitive(), job_hash(source.pipeline),
        source.modelling_context.to_primitive(), job_hash(source.evaluation_plan),
        source.name, source.cluster_count, job_hash(source.processing_records),
        job_hash(source.cleaning_records), job_hash(source.tuning_spec)))


@recordable
def selection_recipe(source, method, options):
    from FeatureSelectionTransformer import FeatureSelectionTransformer
    from roles import Role, weighting_purpose
    from selection_resampling import selection_design
    # Any additional role protects the column, including Predictor + Treatment.
    columns = [c for c in source.frame
               if source.role_map.roles_for(c) == {Role.PREDICTOR}]
    task = source.modelling_context.resolved_task(source)
    targets = list(source.role_map.columns_with_role(Role.TARGET))
    if len(targets) != 1 or task not in ('classification', 'regression'):
        raise ValueError('Assign one classification or regression target; ordinal/survival selection is not yet supported')
    weighting = tuple(c for c in source.role_map.columns_with_role(Role.WEIGHTING)
                      if weighting_purpose(c) == 'importance') if options['use_weights'] else ()
    strategy = (source.evaluation_plan.recipe.get('strategy') if source.evaluation_plan else '')
    structure = tuple(source.role_map.columns_with_role(Role.SEQUENCE))
    if strategy in ('grouped', 'logo', 'temporal'):
        structure = (*structure, strategy)
    kwargs = {k:v for k,v in options.items() if k != 'use_weights'}
    return FeatureSelectionTransformer(tuple(columns), method=method, task=task,
        weighting=weighting, structure=structure, predictors=tuple(source.model_predictors),
        resampling=selection_design(source), **kwargs), targets[0]


@recordable
def analyze_selection(source, method, options, progress=None):
    import time
    import importlib.metadata
    import numpy as np
    from pipeline import Pipeline
    from sklearn.base import clone
    from selection_analysis import selection_recipe
    started = time.monotonic()
    result = dict(source=source, method=method, options=options, error='', model=None)
    try:
        recipe, target = selection_recipe(source, method, options)
        positions = np.arange(len(source.clean_frame))
        plan = source.evaluation_plan
        if plan is not None:
            plan.require(source, supervised=True)
            positions = np.asarray(plan.development, dtype=int)
        raw = source.clean_frame.iloc[positions]
        # The pipeline owns positional y alignment through upstream samplers.
        steps = [] if source.pipeline is None else [(n,s) for n,s in source.pipeline.steps if n != 'data_models']
        pipeline = Pipeline([*steps, ('selection_preview', clone(recipe))])
        pipeline.fit(raw, raw[target].to_numpy(), selection_preview__progress=progress)
        model = pipeline.named_steps['selection_preview']
        result.update(model=model, recipe=recipe, target=target,
            observations=len(model.fit_positions_), exploratory=plan is None,
            versions={p:importlib.metadata.version(p) for p in ('lightgbm','Boruta','scikit-learn')})
    except (ValueError, TypeError, ImportError, RuntimeError) as error:
        result['error'] = str(error)
    result['seconds'] = time.monotonic() - started
    return result


@recordable
def export_selection(result):
    source, model = result['source'], result['model']
    if result['error'] or model is None:
        raise ValueError(result['error'] or 'Selection is not ready')
    if not model.selected_columns_:
        raise ValueError('No variables were selected. Review the recipe or choose None.')
    # Preview schema may differ from the development-refitted upstream schema.
    removed = [c for c in model.removed_columns_ if c in source.frame]
    preview = source.frame.drop(columns=removed)
    if not any(c in preview for c in model.selected_columns_):
        raise ValueError('The development selection and upstream preview schemas differ; review upstream preprocessing')
    exported = source.with_pipeline_step(result['recipe'], name='var_select',
        operation=f"Select variables: {result['method']}", preview_frame=preview,
        removed_columns=removed)
    # Memberships remain valid: raw observations/outcomes are unchanged and only
    # eligible preview roles were removed. Rebase the validated plan fingerprint
    # rather than forcing a new random split after a learned feature selection.
    if source.evaluation_plan is not None:
        from dataclasses import replace
        from evaluation_plan import source_fingerprint
        source.evaluation_plan.require(source, supervised=True)
        plan = replace(source.evaluation_plan, fingerprint=source_fingerprint(exported))
        exported = replace(exported, _evaluation_plan=plan)
    return exported


@recordable
def selection_evidence(result):
    """Show every incoming variable without inventing importance for protected roles."""
    import pandas as pd
    from roles import Role
    source = result['source']
    table = (result['model'].diagnostics_.drop(columns='Rank').set_index('Variable')
             .reindex(source.frame.columns).rename_axis('Variable').reset_index())
    roles = [source.role_map.roles_for(c) for c in table.Variable]
    table.insert(1, 'Role', [', '.join(sorted(r.value for r in values)) for values in roles])
    for i, values in enumerate(roles):
        if values != {Role.PREDICTOR}:
            table.loc[i, 'Decision'] = 'Pass through'
            table.loc[i, 'Included'] = True
        elif pd.isna(table.loc[i, 'Decision']):
            table.loc[i, 'Decision'] = 'Not evaluated'
            table.loc[i, 'Included'] = True
    table['Included'] = table['Included'].astype(bool)
    return table


@recordable
def selection_stability(source, method, options, progress=None):
    """Diagnostic only: refit the full recipe on each upstream training split."""
    import pandas as pd
    from sklearn.base import clone
    from pipeline import Pipeline
    from selection_analysis import selection_recipe

    result = {'method':method, 'options':options, 'error':'', 'summary':pd.DataFrame(),
              'details':pd.DataFrame(), 'successful':0, 'planned':0, 'failures':[], 'notes':[]}
    plan = source.evaluation_plan
    try:
        if plan is None:
            raise ValueError('Selection stability needs an upstream obs_resample plan')
        plan.require(source, supervised=True)
        recipe, target = selection_recipe(source, method, options)
        result['planned'] = plan.count
        if plan.count < 2:
            result['notes'].append('A single upstream training split cannot show between-fold variation.')
        rows = []
        upstream = [] if source.pipeline is None else [(n,s) for n,s in source.pipeline.steps if n != 'data_models']
        for fold, (train, validation) in enumerate(plan.splits(source), 1):
            raw = source.clean_frame.iloc[train]
            try:
                pipeline = Pipeline([*[(n,clone(s)) for n,s in upstream], ('selection_preview',clone(recipe))])
                def report(message):
                    if progress:
                        progress(f'{method} stability CV{fold}/{plan.count}: {message}')
                pipeline.fit(raw, raw[target].to_numpy(), selection_preview__progress=report)
                model = pipeline.named_steps['selection_preview']
                decisions = model.diagnostics_.set_index('Variable')['Decision'].to_dict()
                selected = set(model.selected_columns_)
                for column in recipe.columns:
                    rows.append({'Fold':f'CV{fold}', 'Variable':column,
                        'Decision':decisions.get(column, 'Unavailable after upstream processing'),
                        'Eligible':column in decisions, 'Selected':column in selected,
                        'Training rows':len(train), 'Validation rows':len(validation),
                        'Fitted rows':len(model.fit_positions_), 'Error':''})
                result['successful'] += 1
            except (ValueError, TypeError, ImportError, RuntimeError) as error:
                result['failures'].append(f'CV{fold}: {error}')
                for column in recipe.columns:
                    rows.append({'Fold':f'CV{fold}', 'Variable':column, 'Decision':'Fit failed',
                        'Eligible':False, 'Selected':False, 'Training rows':len(train),
                        'Validation rows':len(validation), 'Fitted rows':None, 'Error':str(error)})
            if progress:
                progress(f'{method} stability: {fold}/{plan.count} training folds completed')
        details = pd.DataFrame(rows)
        summary = []
        for column in recipe.columns:
            values = details[details.Variable == column]
            eligible = int(values.Eligible.sum())
            selected = int(values.Selected.sum())
            summary.append({'Variable':column, 'Selected folds':selected, 'Eligible folds':eligible,
                'Successful folds':result['successful'], 'Planned folds':plan.count,
                'Selection frequency':selected / eligible if eligible else float('nan'),
                'Tentative folds':int((values.Decision == 'Tentative').sum()),
                'Unavailable folds':int((values.Decision == 'Unavailable after upstream processing').sum()),
                'Failed folds':len(result['failures'])})
        result['summary'], result['details'] = pd.DataFrame(summary), details
        if not result['successful']:
            result['error'] = 'No stability fits succeeded. ' + '; '.join(result['failures'])
    except (ValueError, TypeError, ImportError, RuntimeError) as error:
        result['error'] = str(error)
    return result
