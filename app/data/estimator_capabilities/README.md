# Estimator capabilities catalogue, version 1.0.0

This is a searchable JSON snapshot, not a runtime import registry. It covers
scikit-learn plus the explicitly listed public modules of imbalanced-learn,
Feature-engine, mlxtend, skrub, XGBoost, LightGBM and CatBoost. The optional
boosting packages were inspected in a temporary environment; they have **not**
been added to Pythagoras dependencies.

## Files

- `catalogue.json`: generated inventory, reviewed configurations and validation outcomes.
- `profiles.json`: editable, reviewed configuration recipes and subjective assessments.
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
- `categorical_complete`: mixed numeric and categorical inputs with no missing
  values; intended for a categorical encoder profile.

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
