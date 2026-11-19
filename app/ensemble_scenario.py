"""Small deterministic reviewed candidates for the ensemble example and tests."""
import numpy as np
import pandas as pd
from candidate_selection import candidate_frame, decisions
from model_commit import prepare_predictors
from model_review import completed_evidence
from modelling_context import ModellingContext
from pipeline_search import fresh, prepare, run
from proxy_data import proxy_data
from roles import RoleMap
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def example_source(task='regression', count=2):
    x = np.arange(40, dtype=float)
    y = 2 * x + 5 * np.sin(x) if task == 'regression' else (x % 4 >= 2).astype(int)
    source = proxy_data(pd.DataFrame({'x': x, 'y': y}),
        _roles=RoleMap.from_primitive({'predictor': ['x'], 'target': ['y']}))
    if task == 'classification':
        source = source.with_modelling_context(ModellingContext(task='classification'))
    members = []
    for i in range(count):
        estimator = (Ridge(alpha=i + 1) if i % 2 == 0 else DecisionTreeRegressor(max_depth=i + 2, random_state=42)) if task == 'regression' else (
            LogisticRegression(C=i + 1) if i % 2 == 0 else DecisionTreeClassifier(max_depth=i + 2, random_state=42))
        model = Pipeline([('predictors', FunctionTransformer(prepare_predictors,
            kw_args={'columns': ['x'], 'preparation': 'numeric'})), ('scale', StandardScaler()), ('estimator', estimator)])
        members.append({'id': f'method_{i}', 'label': f'{type(estimator).__name__} {i + 1}', 'estimator': model})
    source = source.with_model_candidates(members)
    problem = prepare(source, method='grid')
    state = run(problem, fresh(problem), trials=0, publish=lambda s: None, pause=lambda: False)
    results = completed_evidence(problem, state)
    frame, _ = candidate_frame(results)
    chosen = decisions(frame, problem.specification['metric'], [1, 0, 0], 1e-12)
    results['review'] = {'version': 2, 'policy': 'weighted_product',
        'representatives': {key: trial - 1 for key, trial in chosen['methods'].items()},
        'winner_trial': chosen['overall'] - 1}
    source = source.with_tuning_results(results)
    return source.with_selected_model(chosen['overall'] - 1, card='cand_models')
