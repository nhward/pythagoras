# Intrinsic transparency review — 16 November 2026

The review is recorded in `taxonomy.json` (version 1.3.0), with explicit method
allowlists, official-provider source links, review dates and explanatory notes.
`catalogue.json` is generated from it. The classifications and readability
judgments are Pythagoras expert guidance, not ratings supplied by those providers.

Intrinsic transparency means understanding the whole fitted prediction rule in
meaningful inputs. It does not mean SHAP availability, causal identification,
fairness, accuracy, or successful runtime validation. A model's prediction rule
and its training/selection procedure may have different levels of complexity.

## Coverage

These counts refer to the 538 estimator entries, not unique implementations or
recipe rows. Aliases are counted separately, as they are in the catalogue.

| Transparency | Before | After |
|---|---:|---:|
| Usually transparent | 44 | 61 |
| Conditional | 32 | 120 |
| Usually opaque | 37 | 46 |
| Unreviewed | 157 | 8 |
| Not applicable | 268 | 303 |

Of the 157 formerly unreviewed entries, 114 received a transparency classification,
35 received an explicit scope decision, and 8 remain unreviewed with a reason.
The scope decisions concern standalone clustering, covariance/precision,
representation and neighbor-retrieval tools, a feature selector, and a
label-informed evaluation oracle. They are outside this outcome-prediction
classification, not proven uninterpretable. Clustering geometry and covariance
graph interpretation deserve separate assessments rather than a borrowed
supervised-prediction label.

The 113 reviewed profiles still have 34 usually transparent, 20 conditional,
33 usually opaque and 26 not-applicable entries. The 21 specialist recipes still
have 6 usually transparent, 11 conditional and 4 usually opaque entries. Their
conditional notes have been refined along with the estimator-level notes.

## What makes a conditional model easier or harder to understand?

The conditions are directions for inspection, not numeric thresholds or rules
that automatically relabel a fitted model. Inspect fitted complexity, rather
than relying only on requested parameter limits. Few components are insufficient
if their inputs or mechanisms are themselves opaque.

| Family | More transparent circumstances | Less transparent circumstances / inspection factors |
|---|---|---|
| Single trees and model trees | Short paths, few leaves, meaningful split variables; short leaf equations | Deep paths, many leaves, many coefficients per leaf; `max_depth`, `max_leaf_nodes`, pruning and fitted size |
| Voting and stacking | Few readable base models and a simple combining rule | Opaque components, many models or stacking layers; weights, final estimator, passthrough and meta-feature count |
| Tree ensembles | A small total collection of shallow readable trees can sometimes be traced | Many trees/leaves, deep trees and complex transformed features; fitted tree count and total size; low depth alone is insufficient |
| Linear predictors | Few meaningful active features and stable coefficients | Dense/high-dimensional transformed features and unstable coefficients; nonzero coefficients, scaling and collinearity |
| Kernel models | A simple linear kernel can expose a short coefficient rule | Nonlinear kernels, many support/reference points, complex feature maps; kernel parameters and fitted reference set |
| Neighbors and local smoothing | Meaningful distances and a small inspectable reference neighborhood | High dimension, complex metrics/weights or many overlapping neighborhoods; neighborhood size/radius and bandwidth |
| Mixtures, naive Bayes and discriminants | Few classes/features/components, simple probability tables or covariance | Many categories/components, overlapping latent components, dense covariance or many quadratic terms |
| Symbolic, rule and NARX models | Short rules/expressions, low degree, short lags and familiar operators | Long/deep expressions, overlapping rules, many polynomial terms/lags or recurrent feedback |
| Calibration and threshold wrappers | Readable base scores and explicit simple maps/thresholds | Opaque base models, many classes/fold models; calibration does not explain the origin of raw scores |
| Grouped, hierarchical and multioutput wrappers | Few meaningful groups/outputs, readable local models and explicit routing | Many levels/models, long chains, shrinkage or fallback interactions |
| Dynamic selection | Small readable pool and a meaningful competence neighborhood | Query-dependent selection/weighting over many opaque models; pool size, neighborhood `k`, metric and selected subset |
| HMM/state-space, mixed and spatial models | Few meaningful states/effects, simple transitions/covariance/weights | Overlapping states, many effects, high orders or spatial feedback; state dimension, sequence length, lag orders and covariance |

The notes are already shown in the existing recipe details. No new settings,
automatic transparency thresholds or filters were introduced. "Usually opaque"
remains the normal whole-model assessment for large ensembles; the small-model
exception is explanatory guidance, not a blanket downgrade to "conditional".

## Other characteristics

The review also supplied documentation-backed core mechanism, composition,
fitting criterion and learning-strategy descriptions where the named implementation
supports them. Configurable wrappers retain `depends_on_configuration` with a
reason instead of inheriting their default component's properties.

Examples include linear versus nonlinear XGBoost boosters, parallel random-forest
wrappers versus sequential boosting, cross-validation selection versus a prediction
ensemble, decay weighting during training versus prediction-time composition,
and local target averaging versus a global linear equation. Dynamic selection is
hybrid where query-local competence construction is involved; cluster-precomputed
selection is eager. Concrete HMM emission families and their likelihood/variational
objectives are recorded separately.

No prediction-output promises, executable recipes, package inspection results,
installation decisions, capability evidence, or probe results were added or upgraded.
The existing out-of-scope `capacity_growth` field remains deferred.

## Remaining unreviewed entries

| Entries | Reason |
|---|---|
| `deslib.dcs.BaseDCS`, `deslib.des.BaseDES`, `deslib.des.BaseProbabilistic` | Extension interfaces need concrete subclass competence/selection behavior and pool configuration. |
| `hmmlearn.hmm.BaseHMM`, `hmmlearn.vhmm.BaseCategoricalHMM`, `hmmlearn.vhmm.BaseGaussianHMM`, `hmmlearn.vhmm.VariationalBaseHMM` | Base interfaces require concrete emission/state/subclass behavior; no whole-model default classification. |
| `sklego.meta.HierarchicalPredictor` | Base grouping interface leaves `predict` to concrete classifier/regressor subclasses, which have been reviewed. |

## Evidence and maintenance

Every reviewed allowlist group links its sources in `taxonomy.json`. Sources are
provider API/user documentation and official repositories. Installed scikit-learn
1.9.1, scikit-lego 0.9.10 and skglm 0.5 source/docstrings were also checked for
important prediction-rule distinctions. Some provider documentation is newer or
development documentation; this review records method-family guidance, not a
fresh certification of every historical package configuration. Existing version
and recipe guards continue to apply.

Representative sources:

- [scikit-learn trees](https://scikit-learn.org/stable/modules/tree.html), [ensembles](https://scikit-learn.org/stable/modules/ensemble.html), [multiclass/multioutput](https://scikit-learn.org/stable/modules/multiclass.html), and [calibration](https://scikit-learn.org/stable/modules/calibration.html).
- [DESlib selection taxonomy](https://deslib.readthedocs.io/en/latest/api.html) and [cluster-based selection](https://deslib.readthedocs.io/en/latest/modules/des/des_clustering.html).
- [scikit-lego meta-estimators](https://koaning.github.io/scikit-lego/api/meta/) and [linear/local models](https://koaning.github.io/scikit-lego/api/linear-model/).
- [skglm API](https://contrib.scikit-learn.org/skglm/api.html), [HiClass algorithms](https://hiclass.readthedocs.io/en/latest/algorithms/index.html), and [hmmlearn tutorial](https://hmmlearn.readthedocs.io/en/stable/tutorial.html).
- [XGBoost random forests](https://xgboost.readthedocs.io/en/stable/tutorials/rf.html), [LightGBM parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html), and [CatBoost](https://catboost.ai/docs/en/concepts/python-reference_catboost).

Regenerate descriptions without importing/fitting estimators or resetting probe
results, from the repository root:

```sh
.venv/bin/python scripts/build_estimator_catalogue.py --refresh-descriptions
```
