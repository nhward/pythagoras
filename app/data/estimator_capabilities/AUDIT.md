# Estimator package audit

Generated: 2026-10-24T19:55:41.215224+00:00

All top-level project entries in Related Projects before translations, all contrib README project headings, and all public contrib repositories (including archived/infrastructure repositories). Broader directories and method-family searches remain future audit phases.

Registry: 94 projects; decisions: {'deferred': 49, 'excluded': 19, 'included': 26}.
Unreviewed source entries: 0.
Catalogue: 538 estimator aliases, 113 reviewed profiles, 21 documentation-only recipes.
Profile validation: {'passed': 113}.

Inspection means public-class/signature/tag discovery, not verified pipeline compatibility. Aliases can identify the same implementation. Missing optional packages remain visible in package metadata; their profiles are unavailable.

## Source coverage

| Source | Entries | Retrieved |
|---|---:|---|
| [related_projects](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/related_projects.rst) | 59 | 2026-10-24T08:50:45.908405+00:00 |
| [contrib_readme](https://raw.githubusercontent.com/scikit-learn-contrib/scikit-learn-contrib/master/README.md) | 15 | 2026-10-24T08:50:45.908723+00:00 |
| [contrib_repositories](https://api.github.com/orgs/scikit-learn-contrib/repos?per_page=100&type=public) | 32 | 2026-10-24T08:50:45.909497+00:00 |

## Package decisions and inspection

| Project | Decision | Distribution | Inspection | Estimators | Reason |
|---|---|---|---|---:|---|
| [aeon](https://github.com/aeon-toolkit/aeon) | deferred | — | not_requested | 0 | Time-series containers, horizons and split semantics require specialist recipes; broad module discovery deferred. |
| [astroml](https://www.astroml.org/) | deferred | — | not_requested | 0 | Domain-specific input representations and optional dependencies require per-estimator review; deferred from tabular-first inspection. |
| [auto-sklearn](https://github.com/automl/auto-sklearn/) | deferred | — | not_requested | 0 | AutoML/search orchestration needs bounded search budgets and explicit wrapper construction; deferred from first inspection batch. |
| [autoviml](https://github.com/AutoViML/Auto_ViML/) | deferred | — | not_requested | 0 | AutoML/search orchestration needs bounded search budgets and explicit wrapper construction; deferred from first inspection batch. |
| [bde](https://github.com/scikit-learn-contrib/bde) | deferred | — | not_requested | 0 | Neural backend and model-construction requirements need a separate dependency environment and probe plan. |
| [biocsklearn](https://bioconductor.org/packages/BiocSklearn) | excluded | — | not_requested | 0 | R/Bioconductor interoperability; not a Python estimator distribution for this scope. |
| [boruta](https://github.com/scikit-learn-contrib/boruta_py) | included | Boruta | inspected | 1 | Wrapper feature selection; constructor requires a base estimator. |
| [catboost](https://pypi.org/project/catboost/) | included | catboost | inspected | 4 | Previously curated package; retained for the fresh inventory. |
| [category-encoders](https://github.com/scikit-learn-contrib/categorical-encoding) | included | category-encoders | inspected | 22 | Categorical encoders; inspect public transformer classes. |
| [concrete-ml](https://github.com/zama-ai/concrete-ml/) | deferred | — | not_requested | 0 | Hardware/runtime-specific wrapper compatibility requires a dedicated environment; deferred. |
| [cubist](https://pypi.org/project/cubist/) | included | cubist | inspected | 1 | Previously curated package; retained for the fresh inventory. |
| [denmune](https://github.com/scikit-learn-contrib/denmune-clustering-algorithm) | deferred | — | not_requested | 0 | Relevant specialized algorithms; package/import mapping and dependency-compatible module scope remain to be reviewed. |
| [denmune-skl](https://github.com/scikit-learn-contrib/denmune-skl) | deferred | — | not_requested | 0 | Relevant specialized algorithms; package/import mapping and dependency-compatible module scope remain to be reviewed. |
| [deslib](https://github.com/scikit-learn-contrib/DESlib) | included | deslib | inspected | 27 | Dynamic classifier/ensemble selection; prediction and dependency compatibility require probes. |
| [dtreeviz](https://github.com/parrt/dtreeviz/) | excluded | — | not_requested | 0 | Model visualization; outside fitting-estimator inventory. |
| [emlearn](https://emlearn.org) | excluded | — | not_requested | 0 | Embedded model conversion/runtime; outside fitting-estimator inventory. |
| [evalml](https://github.com/alteryx/evalml) | deferred | — | not_requested | 0 | AutoML/search orchestration needs bounded search budgets and explicit wrapper construction; deferred from first inspection batch. |
| [fastcan](https://github.com/scikit-learn-contrib/fastcan) | included | fastcan | inspected | 3 | Feature selection and nonlinear autoregression; estimator interface varies by class. |
| [feature-engine](https://github.com/solegalli/feature_engine) | included | feature-engine | inspected | 60 | Existing reviewed package; retained and reinspected. |
| [featuretools](https://github.com/alteryx/featuretools) | deferred | — | not_requested | 0 | Relational feature synthesis needs an entity-set/input adapter; deferred. |
| [flwr](https://flower.dev/) | excluded | — | not_requested | 0 | Federated orchestration framework; no general tabular estimator discovery in this phase. |
| [forest-confidence-interval](https://github.com/scikit-learn-contrib/forest-confidence-interval) | excluded | — | not_requested | 0 | Post-fit uncertainty functions; no estimator class inventory selected for this phase. |
| [gensim](https://radimrehurek.com/gensim/) | deferred | — | not_requested | 0 | Domain-specific input representations and optional dependencies require per-estimator review; deferred from tabular-first inspection. |
| [giotto-tda](https://github.com/giotto-ai/giotto-tda) | deferred | — | not_requested | 0 | Domain-specific input representations and optional dependencies require per-estimator review; deferred from tabular-first inspection. |
| [gplearn](https://github.com/trevorstephens/gplearn) | included | gplearn | inspected | 3 | Symbolic regression/classification and supervised feature construction. |
| [hdbscan](https://github.com/scikit-learn-contrib/hdbscan) | included | hdbscan | inspected | 3 | Density clustering and prediction utilities; include despite overlapping sklearn HDBSCAN. |
| [hiclass](https://github.com/scikit-learn-contrib/hiclass) | included | hiclass | inspected | 5 | Hierarchical classifiers; multi-level targets need specialist recipes. |
| [hmmlearn](https://github.com/hmmlearn/hmmlearn) | included | hmmlearn | inspected | 11 | Hidden Markov estimators; sequence lengths and structured evaluation need separate recipes. |
| [imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn) | included | imbalanced-learn | inspected | 27 | Existing reviewed package; retained and reinspected. |
| [implicit](https://github.com/benfred/implicit) | deferred | — | not_requested | 0 | User-item representations and recommendation evaluation need adapters; similar fit/predict names do not establish sklearn compatibility. |
| [kmodes](https://github.com/nicodv/kmodes) | included | kmodes | inspected | 2 | Categorical and mixed-data clustering. |
| [lda](https://github.com/lda-project/lda/) | deferred | — | not_requested | 0 | Domain-specific input representations and optional dependencies require per-estimator review; deferred from tabular-first inspection. |
| [lifelines](https://lifelines.readthedocs.io/en/latest/fitters/regression/CoxPHFitter.html) | deferred | lifelines | not_requested | 0 | Previously documented specialist recipes retained; runtime adapter and comprehensive discovery deferred. |
| [lightfm](https://github.com/lyst/lightfm) | deferred | — | not_requested | 0 | User-item representations and recommendation evaluation need adapters; similar fit/predict names do not establish sklearn compatibility. |
| [lightgbm](https://lightgbm.readthedocs.io) | included | lightgbm | inspected | 7 | Existing reviewed package; retained and reinspected. |
| [lightning](http://contrib.scikit-learn.org/lightning/) | deferred | — | not_requested | 0 | Repository is archived; retain discovery evidence, defer legacy-build/API compatibility review rather than treating it as unsupported. |
| [linear-tree](https://pypi.org/project/linear-tree/) | included | linear-tree | inspected | 6 | Previously curated package; retained for the fresh inventory. |
| [mapie](https://github.com/scikit-learn-contrib/MAPIE) | included | mapie | inspected | 3 | Conformal wrappers; newer fit/conformalize APIs may not expose the ordinary sklearn parameter interface. |
| [metric-learn](https://github.com/scikit-learn-contrib/metric-learn) | included | metric-learn | inspected | 17 | Metric-learning transformers; pair/constraint inputs need recipe review. |
| [mimic](https://github.com/scikit-learn-contrib/mimic) | deferred | — | not_requested | 0 | Repository is archived; retain discovery evidence, defer legacy-build/API compatibility review rather than treating it as unsupported. |
| [mlflow](https://mlflow.org/) | excluded | — | not_requested | 0 | Experiment tracking and registry tooling; not an estimator provider in this audit. |
| [mljar-supervised](https://github.com/mljar/mljar-supervised) | deferred | — | not_requested | 0 | AutoML/search orchestration needs bounded search budgets and explicit wrapper construction; deferred from first inspection batch. |
| [mloptimizer](https://github.com/scikit-learn-contrib/mloptimizer) | deferred | — | not_requested | 0 | AutoML/search orchestration needs bounded search budgets and explicit wrapper construction; deferred from first inspection batch. |
| [mlxtend](https://github.com/rasbt/mlxtend) | included | mlxtend | inspected | 18 | Existing reviewed package; retained and reinspected. |
| [model-diagnostics](https://lorentzenchr.github.io/model-diagnostics/) | excluded | — | not_requested | 0 | Model assessment utilities; outside fitting-estimator inventory. |
| [neptune](https://neptune.ai/) | excluded | — | not_requested | 0 | Experiment tracking service; outside estimator discovery. |
| [nilearn](https://nilearn.github.io/) | deferred | — | not_requested | 0 | Domain-specific input representations and optional dependencies require per-estimator review; deferred from tabular-first inspection. |
| [nltk](https://www.nltk.org/) | deferred | — | not_requested | 0 | Domain-specific input representations and optional dependencies require per-estimator review; deferred from tabular-first inspection. |
| [pandas](https://pandas.pydata.org/) | excluded | — | not_requested | 0 | Data container/manipulation library; not an estimator provider. |
| [polylearn](https://github.com/scikit-learn-contrib/polylearn) | deferred | — | not_requested | 0 | Repository is archived; retain discovery evidence, defer legacy-build/API compatibility review rather than treating it as unsupported. |
| [pomegranate](https://github.com/jmschrei/pomegranate) | deferred | — | not_requested | 0 | Probabilistic modelling uses distinct construction/inference APIs; sklearn compatibility must be established per wrapper. |
| [project-template](https://github.com/scikit-learn-contrib/project-template) | excluded | — | not_requested | 0 | Development template; not an estimator library. |
| [py-earth](https://github.com/scikit-learn-contrib/py-earth) | deferred | — | not_requested | 0 | Repository is archived; retain discovery evidence, defer legacy-build/API compatibility review rather than treating it as unsupported. |
| [pymc](https://www.pymc.io/) | deferred | — | not_requested | 0 | Probabilistic modelling uses distinct construction/inference APIs; sklearn compatibility must be established per wrapper. |
| [qolmat](https://github.com/scikit-learn-contrib/qolmat) | deferred | — | not_requested | 0 | Relevant specialized algorithms; package/import mapping and dependency-compatible module scope remain to be reviewed. |
| [radius_clustering](https://github.com/scikit-learn-contrib/radius_clustering) | deferred | — | not_requested | 0 | Relevant specialized algorithms; package/import mapping and dependency-compatible module scope remain to be reviewed. |
| [sacred](https://github.com/IDSIA/Sacred) | excluded | — | not_requested | 0 | Experiment configuration and tracking; outside estimator discovery. |
| [scikeras](https://github.com/adriangb/scikeras) | deferred | — | not_requested | 0 | Neural backend and model-construction requirements need a separate dependency environment and probe plan. |
| [scikit-dimension](https://github.com/scikit-learn-contrib/scikit-dimension) | included | scikit-dimension | inspected | 14 | Intrinsic-dimension estimators; outputs differ from ordinary feature transformations. |
| [scikit-image](https://scikit-image.org/) | deferred | — | not_requested | 0 | Domain-specific input representations and optional dependencies require per-estimator review; deferred from tabular-first inspection. |
| [scikit-learn](https://pypi.org/project/scikit-learn/) | included | scikit-learn | inspected | 210 | Previously curated package; retained for the fresh inventory. |
| [scikit-learn-contrib](https://github.com/scikit-learn-contrib/scikit-learn-contrib) | excluded | — | not_requested | 0 | Project directory infrastructure; not an estimator library. |
| [scikit-learn-contrib.github.io](https://github.com/scikit-learn-contrib/scikit-learn-contrib.github.io) | excluded | — | not_requested | 0 | Project website infrastructure; not an estimator library. |
| [scikit-learn-extra](https://github.com/scikit-learn-contrib/scikit-learn-extra) | included | scikit-learn-extra | not_installed | 0 | Additional clustering/robust estimators; wheel unavailable on this inspection platform, retained as not installed. |
| [scikit-learn-intelex](https://github.com/intel/scikit-learn-intelex) | deferred | — | not_requested | 0 | Hardware/runtime-specific wrapper compatibility requires a dedicated environment; deferred. |
| [scikit-lego](https://github.com/koaning/scikit-lego) | included | scikit-lego | inspected | 48 | Additional regressors, classifiers, meta-estimators and preprocessing. |
| [scikit-matter](https://github.com/scikit-learn-contrib/scikit-matter) | deferred | — | not_requested | 0 | Relevant specialized algorithms; package/import mapping and dependency-compatible module scope remain to be reviewed. |
| [scikit-multilearn](https://github.com/scikit-multilearn/scikit-multilearn) | deferred | — | not_requested | 0 | Multilabel task/target semantics and legacy dependencies require a dedicated review. |
| [scikit-network](https://scikit-network.readthedocs.io/) | deferred | — | not_requested | 0 | Domain-specific input representations and optional dependencies require per-estimator review; deferred from tabular-first inspection. |
| [scikit-surprise](https://surpriselib.com/) | deferred | — | not_requested | 0 | User-item representations and recommendation evaluation need adapters; similar fit/predict names do not establish sklearn compatibility. |
| [scikit-survival](https://scikit-survival.readthedocs.io/) | deferred | — | not_requested | 0 | Existing documentation-only specialist recipes retained; comprehensive package inspection and adapters deferred. |
| [seaborn](https://stanford.edu/~mwaskom/software/seaborn/) | excluded | — | not_requested | 0 | Visualization library; not an estimator provider. |
| [skdag](https://github.com/scikit-learn-contrib/skdag) | deferred | — | not_requested | 0 | Pipeline/dataframe integration utilities; defer detailed composition semantics to a subsequent inspection batch. |
| [skforecast](https://github.com/JoaquinAmatRodrigo/skforecast) | deferred | — | not_requested | 0 | Time-series containers, horizons and split semantics require specialist recipes; broad module discovery deferred. |
| [skglm](https://contrib.scikit-learn.org/skglm) | included | skglm | inspected | 13 | Penalized generalized linear models; constructor and task-specific recipes require review. |
| [sklearn-ann](https://github.com/scikit-learn-contrib/sklearn-ann) | deferred | — | not_requested | 0 | Relevant specialized algorithms; package/import mapping and dependency-compatible module scope remain to be reviewed. |
| [sklearn-onnx](https://github.com/onnx/sklearn-onnx) | excluded | — | not_requested | 0 | Model serialization/export; outside fitting-estimator inventory. |
| [sklearn-pandas](https://github.com/scikit-learn-contrib/sklearn-pandas) | deferred | — | not_requested | 0 | Pipeline/dataframe integration utilities; defer detailed composition semantics to a subsequent inspection batch. |
| [sklearn2pmml](https://github.com/jpmml/sklearn2pmml) | excluded | — | not_requested | 0 | Model serialization/export; outside fitting-estimator inventory. |
| [skll](https://skll.readthedocs.io/en/latest/index.html) | deferred | — | not_requested | 0 | Experiment runner wrapping other libraries; underlying estimators are audited separately; wrapper review deferred. |
| [skope-rules](https://github.com/scikit-learn-contrib/skope-rules) | included | skope-rules | import_failed | 0 | Existing reviewed package; retained and reinspected. |
| [skops](https://skops.readthedocs.io/en/stable/persistence.html) | excluded | — | not_requested | 0 | Persistence tooling; outside fitting-estimator inventory. |
| [skorch](https://github.com/dnouri/skorch) | deferred | — | not_requested | 0 | Neural backend and model-construction requirements need a separate dependency environment and probe plan. |
| [skrub](https://skrub-data.org) | included | skrub | inspected | 27 | Existing reviewed package; retained and reinspected. |
| [sktime](https://github.com/sktime/sktime) | deferred | — | not_requested | 0 | Time-series containers, horizons and split semantics require specialist recipes; broad module discovery deferred. |
| [spreg](https://pysal.org/spreg/generated/spreg.ML_Lag.html) | deferred | spreg | not_requested | 0 | Previously documented specialist recipes retained; runtime adapter and comprehensive discovery deferred. |
| [stability-selection](https://github.com/scikit-learn-contrib/stability-selection) | deferred | — | not_requested | 0 | Repository is archived; retain discovery evidence, defer legacy-build/API compatibility review rather than treating it as unsupported. |
| [statsmodels](https://www.statsmodels.org) | deferred | — | not_requested | 0 | Existing documentation-only specialist recipes retained; comprehensive package inspection and adapters deferred. |
| [tpot](https://github.com/rhiever/tpot) | deferred | — | not_requested | 0 | AutoML/search orchestration needs bounded search budgets and explicit wrapper construction; deferred from first inspection batch. |
| [treelite](https://treelite.readthedocs.io) | excluded | — | not_requested | 0 | Tree compilation/runtime; outside fitting-estimator inventory. |
| [tslearn](https://github.com/tslearn-team/tslearn) | deferred | — | not_requested | 0 | Time-series containers, horizons and split semantics require specialist recipes; broad module discovery deferred. |
| [u8darts](https://unit8co.github.io/darts/) | deferred | — | not_requested | 0 | Time-series containers, horizons and split semantics require specialist recipes; broad module discovery deferred. |
| [xgboost](https://github.com/dmlc/xgboost) | included | xgboost | inspected | 6 | Existing reviewed package; retained and reinspected. |
| [yellowbrick](https://github.com/DistrictDataLabs/yellowbrick) | excluded | — | not_requested | 0 | Estimator visualizers; excluded from predictive/preprocessing methods in this phase. |

## Method grouping

21 methods have reviewed supersession metadata. Linked recipes are hidden by default and restored by Show excluded and unreviewed recipes. This is display grouping; objective-family nesting does not guarantee interchangeable solvers, capabilities or validated recipes.

| Method | Broader method | Relationship |
|---|---|---|
| category_encoders.BinaryEncoder | category_encoders.BaseNEncoder | configured_special_case |
| feature_engine.transformation.ReciprocalTransformer | feature_engine.transformation.PowerTransformer | configured_special_case |
| mlxtend.regressor.LinearRegression | sklearn.linear_model.ElasticNet | objective_family |
| skglm.Lasso | skglm.ElasticNet | configured_special_case |
| sklearn.feature_selection.SelectFdr | sklearn.feature_selection.GenericUnivariateSelect | configured_special_case |
| sklearn.feature_selection.SelectFpr | sklearn.feature_selection.GenericUnivariateSelect | configured_special_case |
| sklearn.feature_selection.SelectFwe | sklearn.feature_selection.GenericUnivariateSelect | configured_special_case |
| sklearn.feature_selection.SelectKBest | sklearn.feature_selection.GenericUnivariateSelect | configured_special_case |
| sklearn.feature_selection.SelectPercentile | sklearn.feature_selection.GenericUnivariateSelect | configured_special_case |
| sklearn.linear_model.GammaRegressor | sklearn.linear_model.TweedieRegressor | configured_special_case |
| sklearn.linear_model.Lasso | sklearn.linear_model.ElasticNet | configured_special_case |
| sklearn.linear_model.LassoCV | sklearn.linear_model.ElasticNetCV | configured_special_case |
| sklearn.linear_model.LinearRegression | sklearn.linear_model.ElasticNet | objective_family |
| sklearn.linear_model.MultiTaskLasso | sklearn.linear_model.MultiTaskElasticNet | configured_special_case |
| sklearn.linear_model.MultiTaskLassoCV | sklearn.linear_model.MultiTaskElasticNetCV | configured_special_case |
| sklearn.linear_model.PassiveAggressiveClassifier | sklearn.linear_model.SGDClassifier | upstream_replacement |
| sklearn.linear_model.PassiveAggressiveRegressor | sklearn.linear_model.SGDRegressor | upstream_replacement |
| sklearn.linear_model.Perceptron | sklearn.linear_model.SGDClassifier | configured_special_case |
| sklearn.linear_model.PoissonRegressor | sklearn.linear_model.TweedieRegressor | configured_special_case |
| sklearn.linear_model.Ridge | sklearn.linear_model.ElasticNet | objective_family |
| sklego.linear_model.LADRegression | sklego.linear_model.QuantileRegression | configured_special_case |

## Discovery failures

- `skrules`: ImportError: cannot import name 'Iterable' from 'collections' (/Users/nickward/.pyenv/versions/3.11.8/lib/python3.11/collections/__init__.py)

## New source entries awaiting a decision

None in this source snapshot.
