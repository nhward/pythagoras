import pandas as pd
import pytest
from MissingnessTransformer import MissingnessTransformer
from proxy_data import proxy_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from tuning import compile_spec, discover, search_values

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('source,expected', [
    ('miss_impute', 'Learned Imputation'),
    ('miss_impute_1', 'Learned Imputation (1)'),
    ('miss_impute_2', 'Learned Imputation (2)'),
    ('data_models_2 › estimator', 'Model method candidates (2) › estimator'),
    ('other_step_1 › nested', 'other_step_1 › nested'),
])
def test_source_titles_preserve_instance_suffix_and_nested_step(source, expected):
    from cards.tuning_parameters import source_label
    cards = {'miss_impute': {'long_name': 'Learned Imputation'},
             'data_models': {'long_name': 'Model method candidates'}}
    assert source_label(source, cards) == expected


def test_discovers_estimator_defaults_and_custom_metadata():
    rows = discover(Pipeline([('miss',MissingnessTransformer()),('model',RandomForestRegressor())]))
    by_path = {r['path']:r for r in rows}
    assert by_path['miss__observation_threshold']['name'] == 'Observation missingness proportion'
    assert by_path['model__n_estimators']['default'] == '100'
    assert by_path['model__max_depth']['current'] == 'None'
    assert by_path['model__max_samples']['definition']['kind'] == 'float'
    assert search_values(by_path['model__max_samples']['definition']) == [0.5, 0.8, 1.0]
    assert by_path['model__random_state']['fixed']
    assert search_values(by_path['miss__observation_threshold']['definition']) == [.3,.4,.5,.6]


def test_reviewed_parameter_purposes_are_class_specific_and_not_based_on_values():
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import FunctionTransformer
    rows = {r['path']: r for r in discover(Pipeline([
        ('select', FunctionTransformer()), ('ridge', Ridge()),
        ('logistic', LogisticRegression()), ('boost', HistGradientBoostingRegressor())]))}
    for name in ('accept_sparse', 'validate', 'check_inverse', 'feature_names_out'):
        row = rows[f'select__{name}']
        assert row['purpose'] == 'operational' and row['fixed']
        assert row['reason'] and row['reference']
        with pytest.raises(ValueError, match='fixed configuration'):
            compile_spec(list(rows.values()), {row['path']:dict(row['definition'], active=True, owner=row['owner'])})
    assert rows['ridge__max_iter']['purpose'] == 'advanced'
    assert rows['boost__max_iter']['purpose'] == 'recommended'
    assert rows['logistic__C']['purpose'] == 'recommended'
    assert rows['logistic__C']['definition']['values'] == [0.01, 0.1, 1.0, 10.0, 100.0]


def test_visibility_levels_keep_enabled_parameters_and_separate_unknown_booleans():
    from tuning import visible
    rows = {r['path']:r for r in discover(Pipeline([('forest', RandomForestRegressor())]))}
    from sklearn.base import BaseEstimator
    class Unreviewed(BaseEstimator):
        def __init__(self, flag=False):
            self.flag = flag
    recommended, advanced, fixed = [rows[f'forest__{name}'] for name in (
        'max_depth', 'max_samples', 'random_state')]
    assert rows['forest__bootstrap']['purpose'] == 'advanced'
    unknown = discover(Pipeline([('unknown', Unreviewed())]))[0]
    assert unknown['purpose'] == 'unreviewed'
    assert unknown['definition']['values'] == [False, True]
    assert visible(recommended, 'recommended')
    assert not visible(advanced, 'recommended') and visible(advanced, 'advanced')
    assert not visible(unknown, 'advanced') and visible(unknown, 'unreviewed')
    assert visible(unknown, 'recommended', active=True)
    assert not visible(fixed, 'unreviewed') and visible(fixed, 'everything')


def test_old_visibility_checkbox_migrates_to_everything_radio():
    from cards.tuning_parameters import instance
    card = instance()
    card.restore_configuration_state({'inputs': {'ShowEverything': True}})
    with card.configuration_ui_context():
        settings = str(card.settings)
    assert 'value="everything" checked="checked"' in settings
    assert 'ShowEverything' in card._configuration_excluded_input_ids


@pytest.mark.parametrize('definition', [
    {'kind':'','values':[1]}, {'kind':'integer','values':[1.5]},
    {'kind':'float','mode':'range','min':0,'max':1,'step':0},
    {'kind':'integer','mode':'range','min':0,'max':1,'step':.1},
    {'kind':'categorical','values':[]}, {'kind':'boolean','values':['false']},
])
def test_incomplete_or_invalid_definitions_rejected(definition):
    with pytest.raises(ValueError): search_values(definition)


def test_committed_grid_propagates_and_invalidates_when_pipeline_changes():
    source = proxy_data(pd.DataFrame({'x':[1,2]}))
    step = MissingnessTransformer()
    source = source.with_pipeline_step(step, name='miss', preview_frame=source.frame)
    rows = discover(source.pipeline)
    row = next(r for r in rows if r['path']=='miss__observation_threshold')
    drafts = {row['path']:dict(row['definition'],active=True,owner=row['owner'])}
    spec = compile_spec(rows,drafts)
    assert spec['combinations']==4
    result = source.with_tuning_spec(spec)
    assert source.tuning_spec is None
    assert result.clone().tuning_spec == spec
    copy = result.tuning_spec; copy['grid'].clear()
    assert result.tuning_spec['grid']
    assert not source.equals(result)
    changed = result.with_estimator(RandomForestRegressor(),label='RF',recipe_id='test')
    assert changed.tuning_spec is None
    assert result.frame.equals(source.frame)
    assert result.pipeline.get_params()['miss__observation_threshold'] == .5
    with pytest.raises(ValueError): compile_spec([],drafts)


def test_counts_unique_choices_without_materializing_combinations():
    assert search_values({'kind':'categorical','values':['a','a','b',None]})==['a','b',None]
    assert compile_spec([], {})['combinations']==1


def test_required_constructor_value_is_not_misreported_as_default():
    from sklearn.base import BaseEstimator
    class Required(BaseEstimator):
        def __init__(self, count): self.count=count
    rows = discover(Pipeline([('required',Required(7))]))
    assert rows[0]['current']=='7'
    assert rows[0]['default']=='No constructor default'
    assert discover(None)==[]
    assert 'tuning_metadata' not in MissingnessTransformer().get_params()


def test_known_bounds_and_stale_implementation_rejected():
    with pytest.raises(ValueError, match='between'):
        search_values({'kind':'float','values':[1.5]}, {'bounds':[0,1]})
    rows = discover(Pipeline([('miss',MissingnessTransformer())]))
    row = next(r for r in rows if r['path']=='miss__observation_threshold')
    with pytest.raises(ValueError, match='implementation'):
        compile_spec(rows,{row['path']:dict(row['definition'],active=True,owner='Other')})


def test_tuning_commit_records_grid_in_data_journey_without_duplicate_entries():
    from cards.data_provenance import _journey_table
    source = proxy_data(pd.DataFrame({'x': [1, 2]}))
    spec = compile_spec([], {})
    result = source.with_tuning_spec(spec).with_tuning_spec(spec)
    assert source.processing_records == ()
    assert len(result.processing_records) == 1
    record = result.processing_records[0]
    assert record.card == 'tuning_parameters'
    assert record.stage == 'Learning' and record.attempted
    assert record.parameters == {'grid': {}, 'combinations': 1}
    assert record.input_shape == record.output_shape == source.shape
    journey = _journey_table(result)
    tuning_rows = journey.loc[journey['Card'] == 'tuning_parameters']
    assert len(tuning_rows) == 1
    assert 'combinations' in tuning_rows.iloc[0]['Parameters']


def test_custom_imputation_styles_and_independent_transform_switches():
    from cards.miss_impute import ImputationStep
    from cards.var_transform import VariableTransformStep
    from sklearn.base import clone
    pipeline = Pipeline([
        ('impute', ImputationStep(('x',), ('x',), 'simple')),
        ('transform', VariableTransformStep(('x',), center=True)),
    ])
    rows = {r['path']: r for r in discover(pipeline)}
    assert search_values(rows['impute__method']['definition']) == ['simple', 'knn', 'iterative']
    assert search_values(rows['impute__knn__n_neighbors']['definition']) == [3, 4, 5, 6]
    assert rows['impute__knn__n_neighbors']['metadata']['default'] == 5
    assert rows['impute__knn__n_neighbors']['default'] == '5'
    assert 'transform__transforms' not in rows
    for name in ('deskew', 'scale', 'center'):
        row = rows[f'transform__{name}']
        assert not row['fixed']
        assert search_values(row['definition']) == [False, True]
    step = clone(pipeline).set_params(transform__scale=True).named_steps['transform']
    frame = pd.DataFrame({'x': [1., 2., 6.]})
    transformed = step.fit_transform(frame)
    assert transformed.x.mean() == pytest.approx(0)
    assert transformed.x.std(ddof=0) == pytest.approx(1)


def test_seed_and_weight_configuration_are_fixed():
    from cards.miss_impute import ImputationStep
    from cards.var_transform import VariableTransformStep
    rows = {r['path']: r for r in discover(Pipeline([
        ('impute', ImputationStep(('x',), ('x',), 'iterative')),
        ('transform', VariableTransformStep(('x',), weight_column='w')),
    ]))}
    for path in ('impute__seed', 'transform__weight_column'):
        row = rows[path]
        assert row['fixed']
        with pytest.raises(ValueError, match='fixed configuration'):
            compile_spec(list(rows.values()), {
                path: dict(row['definition'], owner=row['owner'], active=True)})


@pytest.mark.parametrize('encoder', ['cyclic', 'ordered', 'nominal', 'sklearn'])
def test_unknown_handling_is_fixed(encoder):
    from CyclicEncodingTransformer import CyclicEncodingTransformer
    from NominalEncodingTransformer import NominalEncodingTransformer
    from OrderedEncodingTransformer import OrderedEncodingTransformer
    from sklearn.preprocessing import OneHotEncoder
    estimator = {
        'cyclic': lambda: CyclicEncodingTransformer(('x',)),
        'ordered': lambda: OrderedEncodingTransformer(('x',)),
        'nominal': lambda: NominalEncodingTransformer(('x',)),
        'sklearn': OneHotEncoder,
    }[encoder]()
    rows = discover(Pipeline([('encode', estimator)]))
    row = next(r for r in rows if r['path'] == 'encode__handle_unknown')
    assert row['fixed']
    with pytest.raises(ValueError, match='fixed configuration'):
        compile_spec(rows, {row['path']: dict(
            row['definition'], owner=row['owner'], active=True)})


@pytest.mark.parametrize('classifier', [False, True])
def test_catboost_reviewed_parameters_include_unset_defaults_and_filter(classifier):
    from catboost_model import PreparedCatBoostClassifier, PreparedCatBoostRegressor
    from tuning import visible
    cls = PreparedCatBoostClassifier if classifier else PreparedCatBoostRegressor
    model = cls(iterations=3, verbose=False, thread_count=1)
    pipeline = Pipeline([('model', model)])
    before = model.get_params().copy()
    rows = {row['path'].split('__')[-1]: row for row in discover(pipeline)}
    recommended = {'iterations', 'depth', 'learning_rate', 'l2_leaf_reg'}
    advanced = {'random_strength', 'bagging_temperature', 'bootstrap_type', 'subsample',
                'rsm', 'border_count', 'grow_policy', 'min_data_in_leaf', 'max_leaves'}
    operational = {'thread_count', 'task_type', 'random_seed', 'early_stopping_rounds'}
    for names, purpose in [(recommended, 'recommended'), (advanced, 'advanced'), (operational, 'operational')]:
        for name in names:
            row = rows[name]
            assert row['purpose'] == purpose
            assert row['reason'] and row['reference']
            assert search_values(row['definition'], row['metadata'])
            assert visible(row, 'recommended') == (purpose == 'recommended')
            assert visible(row, 'advanced') == (purpose != 'operational')
            assert visible(row, 'everything')
            assert row['fixed'] == (purpose == 'operational')
    row = rows['depth']
    spec = compile_spec(list(rows.values()), {row['path']: dict(row['definition'], active=True, owner=row['owner'])})
    assert spec['grid'] == {'model__depth': [4, 6, 8]}
    assert model.get_params() == before  # Discovery must not change native defaults.
    pipeline.set_params(model__depth=4)
    assert model.get_params()['depth'] == 4


def test_gaussian_process_reviewed_defaults_and_explicit_kernel_parameters():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        RBF,
        ConstantKernel,
        Matern,
        WhiteKernel,
    )
    from tuning import visible
    model = GaussianProcessRegressor()
    rows = {r['path']: r for r in discover(Pipeline([('gp', model)]))}
    assert {r['path'] for r in rows.values() if visible(r, 'recommended')} == {'gp__alpha', 'gp__normalize_y'}
    for row in rows.values():
        assert row['purpose'] != 'unreviewed'
        assert row['reason'] and row['reference']
        if row['definition']['values']:
            search_values(row['definition'], row['metadata'])
    assert rows['gp__kernel']['fixed']
    assert rows['gp__random_state']['fixed']
    assert rows['gp__n_restarts_optimizer']['purpose'] == 'advanced'
    assert model.kernel is None  # Discovery preserves the default fixed kernel.
    model.kernel = ConstantKernel() * RBF() + WhiteKernel()
    rows = {r['path']: r for r in discover(Pipeline([('gp', model)]))}
    paths = ['gp__kernel__k1__k1__constant_value', 'gp__kernel__k1__k2__length_scale',
             'gp__kernel__k2__noise_level']
    for path in paths:
        row = rows[path]
        assert row['purpose'] == 'advanced'
        assert visible(row, 'advanced')
        assert not visible(row, 'recommended')
        spec = compile_spec(list(rows.values()), {path: dict(row['definition'], active=True, owner=row['owner'])})
        assert spec['grid'][path] == row['definition']['values']
        assert rows[path+'_bounds']['fixed']
    model.kernel = Matern()
    rows = {r['path']: r for r in discover(Pipeline([('gp', model)]))}
    assert search_values(rows['gp__kernel__nu']['definition']) == [0.5, 1.5, 2.5]


def test_model_recipe_preparation_is_visible_only_under_everything_and_not_tunable():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from tuning import visible

    pipeline = Pipeline([
        ('upstream', SimpleImputer()),
        ('data_models', Pipeline([
            ('prepare', Pipeline([('impute', SimpleImputer()), ('scale', StandardScaler())])),
            ('estimator', Ridge()),
        ])),
    ])
    rows = discover(pipeline)
    paths = {row['path'] for row in rows}
    assert 'upstream__strategy' in paths
    assert 'data_models__estimator__alpha' in paths
    recipe = next(row for row in rows if row['path'] == 'data_models__prepare__impute__strategy')
    assert recipe['purpose'] == 'recipe'
    assert recipe['fixed']
    assert visible(recipe, 'everything')
    for level in ('recommended', 'advanced', 'unreviewed'):
        assert not visible(recipe, level)
    with pytest.raises(ValueError, match='fixed configuration'):
        compile_spec(rows, {recipe['path']: dict(recipe['definition'], active=True, owner=recipe['owner'])})
