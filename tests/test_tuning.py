import pandas as pd
import pytest
from MissingnessTransformer import MissingnessTransformer
from proxy_data import proxy_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from tuning import compile_spec, discover, search_values

pytestmark = pytest.mark.unit


def test_discovers_estimator_defaults_and_custom_metadata():
    rows = discover(Pipeline([('miss',MissingnessTransformer()),('model',RandomForestRegressor())]))
    by_path = {r['path']:r for r in rows}
    assert by_path['miss__observation_threshold']['name'] == 'Observation missingness proportion'
    assert by_path['model__n_estimators']['default'] == '100'
    assert by_path['model__max_depth']['current'] == 'None'
    assert by_path['model__max_samples']['definition']['kind'] == ''
    assert by_path['model__random_state']['fixed']
    assert search_values(by_path['miss__observation_threshold']['definition']) == [.3,.4,.5,.6]


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
