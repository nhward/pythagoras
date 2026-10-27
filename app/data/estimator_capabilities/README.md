# Estimator capabilities catalogue, version 1.5.0


Model descriptions use schema **1.4.0**. `taxonomy.json` is the curated source:
explicit method groups provide core mechanism, composition, fitting criterion,
learning strategy and transparency; exact recipe overrides resolve configuration-dependent
facts. The existing `outputs` list is retained and normalized to `class_label`, with
`output_review` explaining its scope. Probability estimates are not certified calibrated.
Capacity growth is reserved with status `deferred` for stage 2.

The registry records documentation sources and review dates. Transparency is expert
guidance, not a measurement. An unlisted method remains `unreviewed`; standalone
preparation steps receive `not_applicable` for this predictive-model taxonomy.
Method-level output support is configuration-dependent; recipe outputs are not
inferred from the presence of a class method. Recipe refinements include an exact
configuration and reviewed-version guard. If either changes, refinements and output
claims require review. Do not copy classifications across wrappers or related classes
without reviewing their component structure.

Refresh descriptions while preserving the inspection snapshot and probe results:

```sh
.venv/bin/python scripts/build_estimator_catalogue.py --refresh-descriptions
```

Normal builds use the same expansion. Edit source profiles/specialists for output
claims and update any corresponding taxonomy recipe override together. Review statuses
are separate from runtime evidence, readiness, installation and adapter support.
The default SVC and hinge-loss SGDClassifier recipes provide labels and decision
scores, not class probabilities. This correction does not alter their configurations
or invalidate their previously recorded fit/predict checks.

`benchmark_methods.json` identifies null/reference models used only to benchmark
evaluation metrics. Their estimator records carry `purpose: benchmark_only` and
a reason. They remain in the catalogue with their validation evidence, but cannot
be selected and never appear in the model-candidate table, including when
**Show excluded and unreviewed recipes** is enabled. Rebuilds preserve this policy.

The sixth priority batch adds twenty passing preprocessing configurations:
PCA, IncrementalPCA, TruncatedSVD, FactorAnalysis, FastICA, KernelPCA,
SparsePCA, GaussianRandomProjection, SparseRandomProjection, Nystroem,
RBFSampler, PolynomialCountSketch, SplineTransformer, PolynomialFeatures,
PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler,
MinMaxScaler and MaxAbsScaler. These are transformers, not standalone prediction
models; their output and structure lists remain empty, consistent with existing
transformer recipes. All use complete dense numeric inputs and fit independently
inside each training fold before an unweighted Ridge compatibility predictor.

All twenty pass cloning, parameter round trips, three-fold held-out prediction,
finite transformed-feature checks, explicit training/held-out feature-count
consistency, row-count checks and convergence checks. StandardScaler and
SplineTransformer additionally accept nonuniform fitting weights, routed to the
transformer only. No warnings occurred in the final run. IncrementalPCA is tested
through fit with internal batches, not direct partial_fit or streaming. Sparse
input, missing values, inverse transforms, reconstruction accuracy, component
stability and predictive benefit are not certified. Specific kernels, dimensions,
seeds, transformation methods and other limits are recorded per recipe.

RobustScaler and MinMaxScaler use their default range tuples: JSON arrays for
these parameters were rejected by the tested sklearn version. The probe's new
transform_contract checks are opt-in so older mixed-type or missingness-preserving
transformers are not retrospectively treated as finite numeric feature producers.

```sh
.venv/bin/python scripts/probe_estimator_catalogue.py --batch priority_6_2026_10_25
```

The fifth priority batch adds ten passing configurations: PoissonRegressor,
GammaRegressor, MultinomialNB, ComplementNB, LarsCV, LassoLarsCV,
OrthogonalMatchingPursuitCV, CalibratedClassifierCV, DummyClassifier and
DummyRegressor. All use complete numeric inputs and pass cloning, parameter
round trips, three-fold held-out prediction and convergence checks. Seven accept
nonuniform fitting weights and all four classifiers pass probability checks.
The probe now supports explicit fixtures: nonnegative count predictors for the
Naive Bayes pair, a Poisson count target, and a strictly positive continuous target
for Gamma. The latter two also assert positive held-out predictions. These domain
restrictions are recipe conditions, not guarantees for arbitrary numeric data.
Internal selection and calibration use no learned preprocessing before internal
folds. Calibration tests the default LinearSVC with sigmoid calibration and a
three-fold ensemble; no calibration-quality claim is made. Dummy models are
reference baselines that ignore predictors. Poisson recorded a sandbox joblib
physical-core detection warning; no convergence warning occurred.

```sh
.venv/bin/python scripts/probe_estimator_catalogue.py --batch priority_5_2026_10_24
```

The fourth priority batch adds ten passing configurations: RidgeCV,
LogisticRegressionCV, RidgeClassifier, RidgeClassifierCV, NearestCentroid,
BernoulliNB, Lars, LassoLars, LassoLarsIC and OrthogonalMatchingPursuit.
All pass cloning, parameter round trips, three-fold held-out prediction and
convergence checks. Five accept nonuniform fitting weights; three classifiers
also pass held-out probability checks. Ridge classifiers claim class labels only.
The three internally cross-validated estimators use complete numeric inputs with
no learned preprocessing before their internal folds. Lars, LassoLars,
LassoLarsIC and OrthogonalMatchingPursuit also use complete numeric inputs;
the remaining three use fold-fitted imputation, scaling and encoding.
BernoulliNB thresholds prepared features at zero. LassoLarsIC uses BIC and
estimated noise variance on a low-dimensional fixture. No predictive-quality,
calibration, feature-selection stability or scalability guarantee is implied.
NearestCentroid recorded a joblib CPU-detection warning in the sandbox; its
predictions and probabilities passed, with no convergence warnings.

```sh
.venv/bin/python scripts/probe_estimator_catalogue.py --batch priority_4_2026_10_24
```

The third priority batch adds ten passing configurations: BaggingRegressor,
BaggingClassifier, ExtraTreeRegressor, ExtraTreeClassifier, NuSVR, NuSVC,
KernelRidge, GaussianProcessClassifier, RadiusNeighborsRegressor and
RadiusNeighborsClassifier. All use fold-fitted imputation, scaling and one-hot
encoding, and pass cloning, parameter round trips, three-fold held-out predictions,
unseen-category prediction and convergence checks. Seven accept nonuniform
sample weights; all five classifiers pass held-out probability checks.
Gaussian-process classification uses a fixed kernel without optimization.
NuSVC records a deprecation warning: its probability option is scheduled for
removal in sklearn 1.11; a replacement calibration wrapper has not been validated.
Radius-neighbor probes use radius 10; the regression probe does not certify
behavior when a query has no neighbors, and the classifier's configured
outlier fallback is not specifically exercised. Binary classification only;
no accuracy, calibration or scalability guarantee is implied.

```sh
.venv/bin/python scripts/probe_estimator_catalogue.py --batch priority_3_2026_10_24
```

The second priority batch adds ten passing configurations: GradientBoostingRegressor,
GradientBoostingClassifier, AdaBoostRegressor, AdaBoostClassifier,
LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis, GaussianNB,
ARDRegression, TheilSenRegressor and RANSACRegressor. All pass the same cloning,
parameter, held-out Pipeline prediction and convergence checks. The five classifiers
also pass held-out probability checks. QDA is restricted to complete numeric
inputs with covariance regularization; the other nine use fold-fitted imputation,
scaling and one-hot encoding. No warnings were recorded for this batch.

```sh
.venv/bin/python scripts/probe_estimator_catalogue.py --batch priority_2_2026_10_24
```

The priority validation batch adds ten passing configurations: MLPRegressor,
MLPClassifier, ElasticNet, ElasticNetCV, TweedieRegressor, QuantileRegressor,
BayesianRidge, HuberRegressor, KNeighborsRegressor and KNeighborsClassifier.
All exercise cloning, parameter round trips and three-fold held-out Pipeline
prediction, with convergence warnings treated as failures. The two classifiers
also check held-out probability shape, finiteness, bounds and normalization.
Supported nonuniform fitting weights and unseen categories are exercised.
ElasticNetCV uses complete numeric inputs with no learned preprocessing before
its internal folds. Tweedie is tested only at power=0 with an identity link;
QuantileRegressor only at the median. KNN distance weights do not establish
observation-weight support. These checks do not measure accuracy or calibration.

There are now 113 passing recipes covering 111 distinct estimators; 79 recipes
include the additional Pipeline cross-validation checks. Rerun just this batch
without invalidating unrelated recorded results using:

```sh
.venv/bin/python scripts/probe_estimator_catalogue.py --batch priority_2026_10_24
```

`supersession.json` records 21 reviewed method-grouping relationships, copied into
the corresponding estimator's `superseded_by` field on rebuild. The model card
hides these recipes by default and restores them with **Show excluded and
unreviewed recipes**. Details name the broader method, parameter mapping, official
sources and limitations. Mathematical objective-family nesting (notably ordinary
least squares and Ridge within Elastic Net) is distinguished from configured
special cases and upstream replacements. Dedicated solvers can remain preferable.
This changes display grouping only: compatibility, validation, installation and
execution are not transferred or changed. The broader method need not have a
validated/installed recipe. The source records screening scope and important
retained distinctions; it does not assert exhaustive mathematical subsumption.

This is a searchable JSON snapshot, not a runtime import registry. Package
discovery is now driven by the source-backed `packages.json` registry. The first
systematic audit accounts for all 59 project entries in sklearn's Related Projects
source (before documentation translations), all 15 contrib README projects, and
all 32 public contrib repositories, including archived and infrastructure projects.
After merging aliases and retaining previously curated projects, 94 projects have
explicit decisions: 26 included, 49 deferred and 19 excluded. This is coverage of
those source snapshots, not an exhaustive inventory of the sklearn ecosystem.

The fresh snapshot contains 538 estimator aliases, 113 passing executable profiles
and 21 documentation-only recipes. Nine new executable profiles cover Cubist,
gplearn, skglm, scikit-lego and category-encoders; they additionally exercise
cloning, parameter round trips and three-fold held-out predictions through a
Pipeline. Other discovered estimators are not thereby certified compatible.
Optional wheels were inspected in a temporary directory and have **not** been
added to Pythagoras dependencies. See `AUDIT.md` for decisions and actual failures.

The catalogue also includes documentation-reviewed recipes for `cubist.Cubist`,
all six `lineartree` Tree/Forest/Boost regressors and classifiers, and
`skrules.SkopeRules`. Their editable source is `specialists.json`; the builder
copies it into `catalogue.json` and now includes these three packages in discovery.
These broader documentation recipes remain `not_run` / `adapter_required`.
Cubist now also has a separate version-pinned, probed preprocessing recipe.
Linear-tree classes were inspected but not fitted. SkopeRules 1.0.1 failed to
import on Python 3.11 (`collections.Iterable`); scikit-learn-extra 0.3.0 had no
matching binary wheel. No compatibility patches were applied to third-party code.

## Files

- `catalogue.json`: generated inventory, reviewed configurations and validation outcomes.
- `profiles.json`: editable, reviewed configuration recipes and subjective assessments.
- `packages.json`: package decisions, import/module scope, source references and review dates.
- `audit_sources.json`: normalized primary-source entries, retrieval dates and payload hashes.
- `AUDIT.md`: generated coverage, package decisions, inventory counts and discovery failures.
- `inspection_environment.json`: interpreter/platform and installed distribution versions.
- `inspection-requirements.txt`: pinned inspection packages and extra dependencies.
- `installation_attempts.json`: initial optional-wheel installation outcomes.
- `../../../scripts/audit_estimator_packages.py`: optional source refresh and offline report generation.
- `schema.json`: JSON Schema (draft 2020-12).
- `../../estimator_catalogue.py`: lightweight JSON loader and capability filtering.
- `../../../scripts/build_estimator_catalogue.py`: rebuilds the inventory.
- `../../../scripts/probe_estimator_catalogue.py`: exercises reviewed recipes.

The inventory contains default-instance tags for discovery, **not** claims about
maximum capabilities. Some wrappers require constructor arguments; some legacy
estimators lack modern tags. Those facts are recorded without inventing tags.
Discovery issues are recorded, not silently swallowed. The module scope for each
package is explicit: this is not a catalogue of every compatible Python library.

The reviewed profiles describe achievable **joint configurations**. For example,
XGBoost's mixed-data profile enables categorical handling; Ridge's mixed-data
profile includes a fitted imputation/encoding/scaling pipeline. A feature available
only in one configuration cannot be combined with a different configuration's
feature by OR-ing flags. `false` in a profile means outside that recipe's contract,
not that the estimator can never support it. `null` and absent claims mean unknown.
An estimator without a reviewed profile remains discoverable but is not returned
as a verified compatible candidate.

## Objective and subjective evidence

The JSON deliberately separates:

1. `default_tags` and signatures: declarations inspected in installed packages.
2. `construction`, `capabilities`, `conditions`, `recipe`, `sources`: reviewed ways
   to achieve capabilities, including non-default parameters and preprocessing.
3. `validation_details`: small synthetic compatibility checks with package versions,
   operations, warnings and resource-only parameter overrides. These check fitting,
   prediction, missing values, unseen categories, sparse data, nonuniform weights,
   and incremental fitting when the profile claims those capabilities. They do not
   establish generalization quality, weight semantics, GPU operation, or big-data speed.
4. `subjective_assessment`: explicit expert judgments, confidence and rationale.
   Use these to rank promising choices, not as measured superiority claims.

`catalogue_version` versions the content format/review release; `schema_version`
versions its structure. `generated_at`, package versions and core dependency versions
identify the inspected snapshot. Review/bump content versions when changing claims.
Rebuilding resets validation. Updating a package **or a core dependency** invalidates
its reviewed configuration until the versioned review and probes are renewed.
The builder never silently replaces the version in a reviewed recipe.

## Big data

Planning regimes, not universal cutoffs:

- Many rows: at least 100,000 observations.
- Many features: at least 1,000 dense or 100,000 sparse features.
- Out-of-core: data exceed available RAM.

Each profile discusses rows, features, memory and prediction. Exact kernel SVMs
and exact Gaussian processes are poor many-row defaults. Histogram boosting is a
strong candidate for large tabular data; SGD and linear SVM solvers are attractive
for wide sparse data. Forest size and one-hot expansion can be memory bottlenecks.
These are subjective/algorithmic guidance, not timing measurements. `partial_fit`
only makes out-of-core learning possible when the reader and preprocessing also
stream; a standard Pipeline does not automatically expose incremental training.
GPU/distributed/external-memory features mentioned in conditions are **not** tested
capabilities of the local sklearn-wrapper recipes.

## Imbalance and weights

Imbalance tolerance cannot be reduced to a universally true Boolean. The catalogue
records class weighting, per-row weighting and internal resampling separately,
plus a subjective preference and its rationale. It deliberately leaves
`better_than_most` null: superiority requires a specified dataset/metric/benchmark.
Use minority-class recall/precision, PR-AUC or decision costs, and inspect calibration.

Importance weights express observation relevance; rebalance weights change class
contribution. Many estimators accept both through the same `sample_weight` argument.
That does not make them identical statistical concepts or imply survey-design
inference. Class weights may multiply per-row weights. Avoid applying the same
rebalance twice. Sampling changes row membership and is not equivalent to general
importance weighting. K-neighbour distance weights and ensemble voting weights
are not evidence of observation-weight support.

A `fit(sample_weight=...)` signature is recorded as an advertised API, not proof
that the estimator uses it as expected. Reviewed profiles cite documentation and
probe acceptance of nonuniform weights. Preprocessing in the generic pipeline
recipes is fitted **unweighted**; weights reach the final estimator through
`model__sample_weight`. Weighted preprocessing requires a separate reviewed recipe.
Evaluation weights must be selected separately; passing weights to `score` does
not establish weighted fitting. Missing targets are not handled by the supervised
recipes. Resampling must be confined to the training fold.

## Preparation recipes

For all recipes, retain a training-fitted preparation and use it on held-out data.
The accompanying probe script contains executable implementations, rather than
executing arbitrary Python strings from the JSON.

- `numeric`: complete numeric features unless that profile explicitly permits
  numeric NaNs. No categorical encoding is supplied.
- `pandas_categories`: declare categorical columns as pandas `category`. HGB has
  a per-feature cardinality limit. Explicit ordered-to-nominal conversion must
  be a deliberate application choice. LightGBM auto-detection is for unordered
  categories; specify categorical feature names when necessary.
- `catboost_strings`: convert designated categorical features to strings, fill
  categorical missing values with a collision-free reserved token, and pass their
  names through `cat_features`. Persist the token chosen during training for later
  transforms. Numeric NaNs can remain. This is not a claim that CatBoost accepts
  arbitrary raw categorical NaNs.
- `impute_onehot`: `ColumnTransformer` with numeric median imputation and scaling;
  categorical most-frequent imputation followed by `OneHotEncoder(handle_unknown=
  'ignore', sparse_output=False)`; final estimator named `model` in a Pipeline.
  Normalize categorical missing markers to `np.nan` before sklearn imputation.
  This dense recipe can be inappropriate for high cardinalities.
- `categorical_complete`: mixed numeric and categorical inputs passed directly
  to the estimator, with no missing values unless the profile explicitly permits
  them; intended for categorical encoder profiles with reviewed missing handling.

The small-data probes reduce ensemble iterations/trees and report these overrides.
They are smoke checks, not exhaustive parameter validation. Do not treat a passed
probe as a guarantee for every dtype, target, task or dataset size.

## Search without importing estimators

```python
from estimator_catalogue import load_catalogue, find_profiles

catalogue = load_catalogue()
choices = find_profiles(
    catalogue,
    task="regressor",
    required=("categorical_predictors", "missing_numeric_predictors",
              "missing_categorical_predictors", "sample_weight_fit"),
)
for choice in choices:
    print(choice["id"], choice["subjective_assessment"]["overall"])
```

This returns validated configurations for the **catalogue environment**. To check
the current deployment too, pass `package_versions`, a mapping of installed
package names to versions (including the profile's core dependencies). Missing
packages and version differences then exclude candidates. Reading package metadata
does not import the estimator libraries:

```python
from importlib.metadata import distributions
versions = {d.metadata["Name"].lower(): d.version for d in distributions()}
choices = find_profiles(catalogue, task="regressor",
                        required=("categorical_predictors",),
                        package_versions=versions)
```

A schema-valid profile with `validation="failed"`, missing libraries or a version
mismatch is excluded by default. Subjective labels do not override capability or
version checks. Conditions still require application checks: cardinality, loss,
input shape, sparse/dense format, missingness and data size.

## Rebuild and validate

To refresh the discovery lists, run
`scripts/audit_estimator_packages.py --refresh`. This updates source snapshots
and reports newly uncovered entries; it never automatically approves or installs
packages. Review those entries in `packages.json`, including deferral/exclusion
reasons and explicit public-module scope for included packages. A refresh failure
does not overwrite the prior complete snapshot. Broader directories and targeted
method-family searches are subsequent audit phases.

The recorded inspection used the existing application environment plus optional
wheels under `/tmp/estimator-audit/packages`. To reproduce it, use a matching base
environment from `inspection_environment.json`, install `inspection-requirements.txt`
into a separate directory with `pip install --no-deps --only-binary=:all: --target`,
and pass that directory to **both** the builder and probe script. Optional packages
must not silently change the base dependency versions. For example, after installing:

```sh
.venv/bin/python scripts/build_estimator_catalogue.py \
  --extra-path /tmp/estimator-audit/packages \
  --environment-output app/data/estimator_capabilities/inspection_environment.json
.venv/bin/python scripts/probe_estimator_catalogue.py \
  --extra-path /tmp/estimator-audit/packages
.venv/bin/python scripts/audit_estimator_packages.py
.venv/bin/python -m pytest -m unit tests/test_estimator_catalogue.py tests/test_estimator_package_audit.py
```

XGBoost remains pinned to the previously reviewed 3.2.0 configuration even though
the package metadata reported a newer release. Refreshing inventory is not an
implicit review of changed versions. Full `check_estimator` conformance testing,
structured-task adapters and exhaustive module enumeration remain outstanding;
the recorded probes establish only their explicit operation/configuration scope.

From the repository root:

```sh
.venv/bin/python scripts/build_estimator_catalogue.py
.venv/bin/python scripts/probe_estimator_catalogue.py
.venv/bin/python -m pytest tests/test_estimator_catalogue.py
```

To include optional packages without changing project dependencies, install the
reviewed versions into a separate directory and pass it to both commands:

```sh
.venv/bin/python -m pip install --target /private/tmp/capability-boosters --no-deps \
  xgboost==3.2.0 lightgbm==4.7.0 catboost==1.2.10
.venv/bin/python scripts/build_estimator_catalogue.py \
  --extra-path /private/tmp/capability-boosters
.venv/bin/python scripts/probe_estimator_catalogue.py \
  --extra-path /private/tmp/capability-boosters
```

Optional packages still require their transitive dependencies/native libraries;
`--no-deps` assumes the host already supplies them. Without optional packages the
builder records their absence and those profiles cannot pass validation. It does
not pretend to verify their capabilities from documentation alone.

## Schema 1.1 and specialist recipes

Version 1.1 adds `specialists.json`, embedded unchanged by the builder under
`catalogue.specialists`. These twelve documentation-reviewed recipes cover
survival (including a Kaplan–Meier reference), ordinal regression, longitudinal
mixed models, forecasting, spatial regression and spatial panel regression.
They are not imported into the default-instance inventory: several are functions
or statistical interfaces that do not implement sklearn's estimator contract.

Each recipe records official sources, documentation version where known, review
date, parameters, preparation steps, assumptions, required auxiliary inputs,
outputs, structural scope and subjective advice. `validation: not_run` and
`integration: adapter_required` are deliberate. Neither installation nor matching
a documentation version promotes these entries to tested executable candidates.
Optional packages have not been added to application dependencies. Advanced
survival designs and specialist training adapters remain separate work.

Existing tabular profiles now identify cross-sectional scope and prediction outputs.
Absence of temporal/spatial scope is unknown, not a claim that the estimator can
never be used through an appropriate adapter. Specialist claims apply only to the
stated recipe; e.g. the interval-censored Weibull recipe is distinct from right
censoring, and the recursive Ridge reduction does not claim panel/exogenous support.

### Access API

`app/estimator_catalogue.py` is the application boundary:

- `load_catalogue(path=DEFAULT_PATH)` caches parsing by file revision, rejects
  unknown schemas/duplicate identifiers/broken references, and returns isolated
  snapshots. It reads both legacy 1.0 and current 1.1 files. Full JSON Schema
  validation is performed in the catalogue tests rather than requiring json schema
  in application imports.
- `find_profiles(...)` retains the conservative existing API for probed tabular recipes.
- `assess_profiles(..., task, required, structure, outputs, available_requirements,
  package_versions)` returns explanations for compatible, incompatible and unknown
  recipes. Evidence, adapter requirements and deployment availability are independent
  fields. Only probed, version-matched recipes with checked deployment dependencies
  are selectable. Pass `installed_versions()` for metadata-only deployment inspection.
- `requirements_from_proxy(source, use_weights=False)` summarizes predictor types,
  missingness, survival censoring and evaluation status using `model_predictors`.
  It reports unsupported predictor types. The caller must resolve these and choose
  structural requirements/objective explicitly before recommending a model. It does
  not equate a numeric target plus temporal rows with a forecasting task.
- `coverage(catalogue)` reports task counts and explicitly marks coverage non-exhaustive.

Example (no estimator library imports):

```python
from estimator_catalogue import load_catalogue, assess_profiles, installed_versions
rows = assess_profiles(
    load_catalogue(), task='survival', required=['right_censoring'],
    available_requirements=['event_time_target'],
    package_versions=installed_versions(),
)
```

JSON contains descriptions, never executable factories. The access layer never
imports a class path or evaluates a recipe. Rebuilds preserve specialist source
content and reset the existing executable recipe probes as before. To promote a
specialist recipe, implement its adapter and versioned dependency checks, extend
probes with its task-specific output/metric contracts, and update its schema/evidence
record. Do not replace `not_run` with `passed` by hand.

Prediction outputs replaces the former Formulation attribute in version 1.4.0.
Regression summaries are `Point prediction` or `Point prediction & spread`, backed
by explicit outputs in the individual recipe. GPR records predictive standard deviation
and covariance; BayesianRidge and ARDRegression record predictive standard deviation.
These outputs are optional prediction calls, not a claim of calibration. GPR uncertainty
must be interpreted in light of its noise model (training alpha versus WhiteKernel).
The precise output list and interpretation remain in the recipe details. Existing
OLS recipes remain point-only; this does not deny other implementations' support
for input-dependent prediction intervals. A lone quantile, global residual variance,
coefficient uncertainty or component disagreement does not establish spread support.

Learning strategy (1.5.0) describes eager, lazy or hybrid predictive model construction.
Nearest-neighbor predictors are lazy; nearest-centroid, GP and SVM predictors are
not lazy merely because they perform distance/kernel calculations at prediction time.
Ordinary preprocessing and building a neighbor index do not alone imply hybrid learning.
Generic wrappers are configuration dependent; exact reviewed recipes resolve their
components. Cubist without neighbor correction is eager; its method-level description
explains the optional hybrid configuration. Strategy remains unreviewed for unmapped
methods and not applicable to standalone preparation steps. Sources and qualifications
stay separate from runtime validation and scalability guidance.
