import json

import numpy as np
import pandas as pd
import pytest
from cards.obs_resample import _figure, _overview
from evaluation_plan import build_plan, level_key, restore_plan
from proxy_data import proxy_data
from roles import RoleMap
from sklearn.base import BaseEstimator, TransformerMixin

pytestmark = pytest.mark.unit


def source(n=24, partition=False):
    frame = pd.DataFrame({"x": np.arange(n, dtype=float), "time": np.repeat(np.arange((n+1)//2), 2)[:n],
        "id": np.tile(["a", "b"], (n+1)//2)[:n], "group": pd.Categorical(np.arange(n)//4),
        "y": pd.Categorical(np.tile(["yes", "no"], (n+1)//2)[:n])}, index=[7]*n)
    roles = {"predictor": ["x"], "sequence": ["time"], "identifier": ["id"], "stratifier": ["group"], "target": ["y"]}
    if partition:
        frame["partition"] = np.resize(["dev-a", "held", "dev-b", "held"], n)
        roles["partition"] = ["partition"]
    return proxy_data(_df=frame, _roles=RoleMap.from_primitive(roles))


def plan(data, **kw):
    return build_plan(data, **({"strategy": "kfold", "objective": "independent", "folds": 3} | kw))


def test_fold_reproducibility_coverage_and_duplicate_indices():
    data = source()
    before = data.clone()
    p = plan(data)
    assert p.ready and p.supervised
    assert p == plan(data)
    assert data.equals(before)
    splits = list(p.splits(data))
    assert sorted(np.concatenate([v for _, v in splits])) == list(range(24))
    for train, val in splits:
        assert not set(train) & set(val)
        assert len(train) == 16 and len(val) == 8
    assert any("dependence" in s for s in p.warnings)
    assert "seed" not in plan(data, shuffle=False).recipe
    assert p.recipe == plan(data, group="ignored", window=100).recipe


@pytest.mark.parametrize("strategy", ["grouped", "logo"])
def test_groups_and_absent_training_classes(strategy):
    data = source()
    p = plan(data, strategy=strategy, objective="unseen", group="group")
    assert p.ready
    for train, val in p.splits(data):
        assert not set(data.clean_frame.group.iloc[train]) & set(data.clean_frame.group.iloc[val])
    assert p.count == (6 if strategy == "logo" else 3)
    p = plan(data, strategy="grouped", group="id", folds=2)
    assert not p.ready and "target class" in p.errors[0]


def test_stratification_and_invalid_counts():
    data = source()
    assert plan(data, strategy="stratified", stratify="y").ready
    assert not plan(data, strategy="stratified", stratify="x").ready
    assert not plan(data, strategy="stratified", stratify="group", folds=5).ready
    assert not plan(data, folds=25).ready
    assert not plan(data, folds=2.5).ready
    assert plan(data, strategy="grouped", group="group", folds=7).count == 6


@pytest.mark.parametrize("expanding", [True, False])
def test_temporal_exact_ranges_ties_and_source_order(expanding):
    data = source()
    # Permuted source positions make accidental use of sorted-row positions visible.
    data = proxy_data(_df=data.frame.iloc[::-1], _roles=data.role_map)
    p = plan(data, strategy="temporal", objective="known", sequence="time", entity="id",
             window=5, horizon=2, gap=1, step=2, expanding=expanding)
    assert p.ready and p.count == 3
    for j, (train, val) in enumerate(p.splits(data)):
        b = 5 + 2*j
        assert set(data.clean_frame.time.iloc[train]) == set(range(0 if expanding else b-5, b))
        assert set(data.clean_frame.time.iloc[val]) == set(range(b+1, b+3))
        assert set(data.clean_frame.id.iloc[val]).issubset(set(data.clean_frame.id.iloc[train]))
    assert not plan(data, strategy="temporal", objective="future", sequence="time", window=30, horizon=2).ready
    assert not plan(data, strategy="temporal", objective="future", sequence="time", window=5, horizon=2, step=1).ready


def test_temporal_new_entities_nonfinite_and_objectives():
    data = source()
    data.clean_frame.loc[:, "id"] = ["a", "b"]*6 + ["new", "b"]*6
    p = plan(data, strategy="temporal", objective="known", sequence="time", entity="id", window=5, horizon=2)
    assert not p.ready and "absent" in p.errors[0]
    assert not plan(data, objective="future_unseen").ready
    assert not plan(data, objective="unseen", group="group").ready
    data.clean_frame["time"] = np.inf
    assert not plan(data, strategy="temporal", objective="future", sequence="time", window=5, horizon=2).ready


def partition_options(purpose="holdout"):
    return {"purpose": purpose, "development_levels": [level_key("dev-a"), level_key("dev-b")], "reserved_levels": [level_key("held")]}


def test_multilevel_partition_mapping_remap_and_predictions():
    data = source(partition=True)
    # Give development and reserved data both classes.
    data.clean_frame["y"] = pd.Categorical(np.resize(["yes", "yes", "no", "no"], 24))
    p = plan(data, **partition_options())
    assert p.ready and len(p.reserved) == 12
    for (a, b), (local_a, local_b) in zip(p.splits(data), p.splits(data, development_only=True)):
        np.testing.assert_array_equal(np.asarray(p.development)[local_a], a)
        np.testing.assert_array_equal(np.asarray(p.development)[local_b], b)
        assert not set(p.reserved) & set(a) | set(p.reserved) & set(b)
    assert not plan(data).ready
    assert not plan(data, **(partition_options() | {"reserved_levels": [level_key("dev-a")]})).ready
    predefined = plan(data, strategy="predefined", **partition_options("validation"))
    assert predefined.ready and predefined.count == 1
    with pytest.raises(ValueError, match="complete source"):
        list(predefined.splits(data, development_only=True))
    data.clean_frame.loc[:, "y"] = data.clean_frame.y.astype(object)
    data.clean_frame.iloc[list(p.reserved), data.clean_frame.columns.get_loc("y")] = None
    assert plan(data, **partition_options("prediction")).ready
    assert not plan(data, **partition_options()).ready


def test_reserved_temporal_boundary_and_group_boundary():
    data = source(partition=True)
    assert not plan(data, strategy="temporal", objective="future", sequence="time", window=3, horizon=1, **partition_options()).ready
    assert not plan(data, strategy="grouped", objective="unseen", group="group", **partition_options()).ready


def test_bookmark_versions_invalidation_and_proxy_successors():
    data = source()
    p = plan(data)
    assert restore_plan(data, json.loads(json.dumps(p.bookmark()))) == p
    bad = p.bookmark()
    bad["recipe"]["version"] = 999
    assert not restore_plan(data, bad).ready
    exported = data.with_evaluation_plan(p, card="obs_resample")
    assert not exported.equals(data)
    assert exported.clone().equals(exported)
    assert exported.with_cluster_count(2).evaluation_plan == p
    assert exported.with_roles(data.role_map).evaluation_plan == p
    for changed in (exported.with_cleaned_data(data.frame.iloc[::-1], card="reverse", operation="Reverse"),
                    exported.sample(n=4), exported.select_dtypes(include="number")):
        assert changed.evaluation_plan is not None and not changed.evaluation_plan.ready
        with pytest.raises(ValueError):
            list(changed.evaluation_splits())
    changed = data.clone()
    changed.clean_frame.iloc[0, 0] = 999
    assert not restore_plan(changed, p.bookmark()).ready
    exported.clean_frame.iloc[0, 0] = 999
    assert not exported.evaluation_plan.ready


class SpyTransformer(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.seen_ = list(X.x)
        return self
    def transform(self, X):
        return X.copy()


class SpySampler(BaseEstimator):
    def fit(self, X, y=None):
        return self

    def fit_resample(self, X, y):
        self.seen_ = list(X.x)
        return pd.concat([X, X]), np.concatenate([y, y])


def test_fresh_pipeline_training_balancing_and_final_holdout():
    data = source(partition=True)
    data.clean_frame["y"] = pd.Categorical(np.resize(["yes", "yes", "no", "no"], 24))
    data.frame["y"] = data.clean_frame.y.to_numpy()
    p = plan(data, **partition_options())
    assert p.ready
    exported = data.with_evaluation_plan(p, card="obs_resample")
    preview = SpyTransformer().fit(data.frame)
    exported = exported.with_pipeline_step(preview, name="spy", preview_frame=data.frame)
    exported = exported.with_sampling_step(SpySampler(), name="sampler", preview_frame=pd.concat([data.frame, data.frame]))
    assert len(exported.frame) == 48 and len(exported.clean_frame) == 24
    assert exported.evaluation_plan == p
    pipelines = []
    for pipeline, train, val in exported.training_partitions():
        assert not hasattr(pipeline.steps[0][1], "seen_")
        pipeline.fit_resample(train, train.y)
        for _, step in pipeline.steps:
            assert set(step.seen_) == set(train.x)
            assert not set(step.seen_) & set(val.x)
            assert not set(step.seen_) & set(data.clean_frame.x.iloc[list(p.reserved)])
        pipelines.append(pipeline)
    assert len({id(p) for p in pipelines}) == 3
    pipeline, development, reserved = exported.final_training_partition()
    assert len(development) == len(reserved) == 12
    assert not hasattr(pipeline.steps[0][1], "seen_")
    assert exported.with_holdout_used().evaluation_plan.holdout_used
    assert not exported.evaluation_plan.holdout_used
    bad = exported.with_evaluation_plan(plan(data, folds=100), card="later")
    with pytest.raises(ValueError):
        list(bad.training_partitions())


def test_supervision_empty_source_and_bounded_display_download():
    data = source()
    roles = data._copy_roles()
    roles.clear_roles("y")
    data = data.with_roles(roles)
    p = plan(data)
    assert p.ready and not p.supervised
    with pytest.raises(ValueError, match="Target"):
        list(data.with_evaluation_plan(p, card="r").training_partitions())
    assert not plan(source(0)).ready
    p = plan(data)
    overview = _overview(data, p, columns=5, rows=2)
    assert overview["counts"].shape == (2, 5, 5)
    assert overview["counts"].sum() == 24*3
    fig = _figure(overview, full=True)
    assert not any(t.name == "Mixed" for t in fig.data)
    np.testing.assert_array_equal(fig.data[0].z, overview["counts"].argmax(axis=2))
    fig.to_json()
    assert not _figure(overview).layout.hovermode
    assert _figure(overview).layout.showlegend
    frames = list(p.membership_frames(data))
    assert len(frames) == 3 and len(frames[0]) == 24
    assert frames[0].source_position.tolist() == list(range(24))
    assert set(frames[0].membership) == {"Training", "Validation"}


def test_missing_source_roles_and_duplicate_variable_names_block_plan():
    data = source()
    data.role_map.set_roles("preview_partition", ["partition"])
    assert not plan(data).ready
    assert "stable source" in plan(data).errors[0]
    frame = pd.DataFrame([[1, 2], [3, 4]], columns=["duplicate", "duplicate"])
    data = proxy_data(_df=frame)
    assert "unique" in plan(data).errors[0]


def test_footer_goals_derive_methods_only_from_roles():
    data = source()
    p = build_plan(data, goal="observations", folds=3)
    assert p.ready and p.recipe["strategy"] == "stratified" and p.recipe["stratify"] == "group"
    p = build_plan(data, goal="strata", folds=3)
    assert p.ready and p.recipe["group"] == "group"
    roles = data._copy_roles()
    roles.clear_roles("group")
    unstratified = data.with_roles(roles)
    assert build_plan(unstratified, goal="observations", folds=3).recipe["strategy"] == "kfold"
    assert not build_plan(unstratified, goal="strata", folds=3).ready
    roles.clear_roles("y")
    longitudinal = data.with_roles(roles)
    p = build_plan(longitudinal, goal="entities", folds=2)
    assert p.ready and p.recipe["group"] == "id"
    for a, b in p.splits(longitudinal):
        assert not set(data.clean_frame.id.iloc[a]) & set(data.clean_frame.id.iloc[b])
    p = build_plan(data, goal="future", window=5, horizon=2)
    assert p.ready and p.recipe["objective"] == "known" and p.recipe["entity"] == "id"


def test_longitudinal_primary_key_and_role_availability():
    from cards.obs_resample import _unavailable
    data = source()
    roles = data._copy_roles()
    roles.clear_roles("time")
    invalid = data.with_roles(roles)
    assert "Repeated Identifier requires" in build_plan(invalid, goal="observations").errors[0]
    assert "entities" in _unavailable(invalid)
    bad = source()
    bad.clean_frame.iloc[1, bad.clean_frame.columns.get_loc("id")] = "a"
    assert "primary key" in build_plan(bad, goal="future", window=5, horizon=2).errors[0]
    data.clean_frame["id"] = np.arange(24)
    assert "entities" in _unavailable(data)
    assert build_plan(data, goal="future", window=5, horizon=2).recipe["objective"] == "future"


def four_way_source():
    frame = pd.DataFrame({"x": np.arange(40.), "partition": np.repeat(["train", "valid", "test", "predict"], 10),
                          "strata": pd.Categorical(np.tile(["A", "B"], 20)),
                          "target": pd.Categorical(np.tile(["yes", "no"], 20))})
    frame.loc[30:, "target"] = None
    data = proxy_data(_df=frame, _roles=RoleMap.from_primitive({"predictor": ["x"], "target": ["target"],
        "partition": ["partition"], "stratifier": ["strata"]}))
    mapping = {level_key(k): v for k, v in zip(["train", "valid", "test", "predict"],
                                              ["development", "validation", "holdout", "prediction"])}
    return data, mapping


def test_four_way_partition_refit_boundaries_and_bookmark():
    data, mapping = four_way_source()
    p = build_plan(data, goal="observations", partition_map=mapping)
    assert p.ready and p.count == 1 and p.recipe["strategy"] == "predefined"
    train, val = next(p.splits(data))
    assert list(train) == list(range(10)) and list(val) == list(range(10, 20))
    assert p.holdout == tuple(range(20, 30)) and p.prediction == tuple(range(30, 40))
    assert p.refit_positions == tuple(range(20))
    assert set(p.membership(train, val)) == {1, 2, 3, 4}
    assert p.diagnostics.Unused.tolist() == [0]
    assert restore_plan(data, json.loads(json.dumps(p.bookmark()))) == p
    exported = data.with_evaluation_plan(p, card="resampling")
    with pytest.raises(ValueError, match="purpose"):
        exported.final_training_partition()
    for purpose in ("holdout", "prediction"):
        _, fitting, reserved = exported.final_training_partition(purpose=purpose)
        assert len(fitting) == 20 and len(reserved) == 10
        assert set(fitting.x).isdisjoint(reserved.x)
    assert exported.with_holdout_used().evaluation_plan.holdout_used
    assert all("Partition" not in e for e in data.validate())
    mapping[level_key("valid")] = "development"
    generated = build_plan(data, goal="observations", partition_map=mapping, folds=5)
    assert generated.ready and generated.count == 5 and not generated.validation
    assert generated.recipe["strategy"] == "stratified"
    assert len(generated.refit_positions) == 20
    for train, val in generated.splits(data):
        assert set(train) | set(val) == set(range(20))


@pytest.mark.parametrize("change", ["unassigned", "no_development", "missing_level", "unknown_purpose", "extra_level"])
def test_partition_requires_complete_explicit_mapping(change):
    data, mapping = four_way_source()
    if change == "unassigned":
        mapping[level_key("test")] = ""
    elif change == "no_development":
        mapping[level_key("train")] = "validation"
    elif change == "missing_level":
        del mapping[level_key("test")]
    elif change == "unknown_purpose":
        mapping[level_key("test")] = "train"
    else:
        mapping[level_key("stale")] = "holdout"
    assert not build_plan(data, goal="observations", partition_map=mapping).ready


def test_stratifier_strip_order_aggregation_and_temporal_priority():
    data = source()
    data = proxy_data(_df=data.frame.iloc[np.random.default_rng(5).permutation(24)], _roles=data.role_map)
    p = build_plan(data, goal="observations", folds=3)
    before = list(p.splits(data))
    overview = _overview(data, p)
    assert overview["order"] == "Stratifier then validation fold order"
    codes, _ = pd.factorize(data.clean_frame.group, sort=False)
    assert np.all(np.diff(codes[overview["positions"]]) >= 0)
    for stratum in np.unique(codes):
        positions = overview["positions"][codes[overview["positions"]] == stratum]
        assert np.all(np.diff(np.asarray(p.codes)[positions]) >= 0)
    assert overview["strata"]["counts"].sum() == 24
    small, full = _figure(overview), _figure(overview, full=True)
    assert sum(t.type == "heatmap" for t in small.data) == 2
    assert not any(t.legendgroup == "strata" for t in small.data)
    assert any(t.legendgroup == "strata" for t in full.data)
    assert full.layout.yaxis.domain[0] > full.layout.yaxis2.domain[1]
    aggregate = _overview(data, p, columns=2, rows=1)
    assert aggregate["strata"]["counts"].sum() == 24
    _figure(aggregate).to_json()
    for (a, b), (c, d) in zip(before, p.splits(data)):
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(b, d)
    temporal = build_plan(data, goal="future", window=5, horizon=2)
    overview = _overview(data, temporal)
    assert overview["order"] == "Sequence order"
    assert np.all(np.diff(data.clean_frame.time.iloc[overview["positions"]]) >= 0)
    assert overview["strata"] is not None
    plain = build_plan(data, goal="observations", stratifier=None, folds=3)
    plain_overview = _overview(data, plain)
    assert plain_overview["order"] == "Source order"
    np.testing.assert_array_equal(plain_overview["positions"], np.arange(len(data.clean_frame)))


def test_predefined_future_boundary_includes_validation_before_holdout():
    data, mapping = four_way_source()
    data.clean_frame["time"] = np.arange(40)
    data.frame["time"] = np.arange(40)
    data.role_map.set_roles("time", ["sequence"])
    p = build_plan(data, goal="future", partition_map=mapping)
    assert p.ready
    data.clean_frame.iloc[15, data.clean_frame.columns.get_loc("time")] = 25
    assert not build_plan(data, goal="future", partition_map=mapping).ready


def test_group_seed_and_fold_cap_are_reproducible():
    data = source()
    first = build_plan(data, goal="strata", folds=3, seed=42)
    assert first.ready and first == build_plan(data, goal="strata", folds=3, seed=42)
    different = build_plan(data, goal="strata", folds=3, seed=7)
    assert first.codes != different.codes
    for train, validation in different.splits(data):
        assert not set(data.clean_frame.group.iloc[train]) & set(data.clean_frame.group.iloc[validation])
    capped = build_plan(data, goal="strata", folds=100, seed=7)
    assert capped.ready and capped.count == 6
    assert capped.recipe["requested_folds"] == 100
    assert any("Using K=6" in note for note in capped.warnings)
    assert restore_plan(data, capped.bookmark()) == capped
    data.clean_frame["group"] = "only"
    invalid = build_plan(data, goal="strata", folds=100, seed=7)
    assert not invalid.ready and "at least two" in invalid.errors[0]


def test_aggregated_colors_resolve_ties_without_mixed_state():
    counts = np.array([[[0, 3, 2, 0, 0], [0, 2, 2, 0, 0], [0, 1, 4, 0, 0]]])
    overview = {"counts": counts, "yedges": np.array([0, 1]), "n": 15,
                "splits": 1, "details": ["a", "b", "c"], "order": "Stratifier order",
                "strata": {"name": "site", "labels": ["A", "B"],
                           "counts": np.array([[3, 2], [2, 2], [1, 4]])}}
    for full in (False, True):
        fig = _figure(overview, full=full)
        heatmaps = [trace for trace in fig.data if trace.type == "heatmap"]
        np.testing.assert_array_equal(heatmaps[0].z, [[1, 1, 2]])
        np.testing.assert_array_equal(heatmaps[1].z, [[0, 0, 1]])
        assert "Training: 3" in heatmaps[0].text[0][0]
        assert "Validation: 2" in heatmaps[0].text[0][0]
        assert not any("Mixed" in (trace.name or "") for trace in fig.data)


def test_multiple_stratifiers_selection_defaults_and_none():
    from evaluation_plan import stratifier_choices
    data = source()
    frame = data.clean_frame.copy()
    frame.insert(0, "cluster__first", frame["group"])
    frame["other"] = pd.Categorical(np.arange(len(frame)) // 6)
    roles = data._copy_roles()
    roles.set_roles("cluster__first", ["stratifier"])
    roles.set_roles("other", ["stratifier"])
    data = proxy_data(_df=frame, _roles=roles)
    assert stratifier_choices(data) == ["group", "other", "cluster__first"]
    default = build_plan(data, goal="observations", folds=3)
    assert default.ready and default.recipe["stratify"] == "group"
    for name in stratifier_choices(data):
        selected = build_plan(data, goal="observations", stratifier=name, folds=3)
        assert selected.ready and selected.recipe["stratify"] == name
        grouped = build_plan(data, goal="strata", stratifier=name, folds=2)
        assert grouped.ready and grouped.recipe["group"] == name
    plain = build_plan(data, goal="observations", stratifier=None, folds=3)
    assert plain.ready and plain.recipe["strategy"] == "kfold"
    assert plain.recipe["stratifier"] is None
    restored = restore_plan(data, plain.bookmark())
    assert restored.ready and restored.recipe == plain.recipe
    assert _overview(data, plain)["strata"] is None
    assert not build_plan(data, goal="strata", stratifier=None, folds=3).ready
    assert not build_plan(data, goal="observations", stratifier="x", folds=3).ready
    roles.clear_roles("group")
    roles.clear_roles("other")
    cluster_only = data.with_roles(roles)
    assert build_plan(cluster_only, goal="observations", folds=3).recipe["stratifier"] == "cluster__first"
    roles.clear_roles("cluster__first")
    assert build_plan(data.with_roles(roles), goal="observations", folds=3).recipe["strategy"] == "kfold"


def test_generated_cluster_stratifiers_follow_pipeline_export():
    from cards.obs_clusters import _export_memberships
    from ClusterMembershipTransformer import ClusterMembershipTransformer
    from evaluation_plan import stratifier_choices
    data = source()
    transformer = ClusterMembershipTransformer(("x",), n_clusters=3, output_column="cluster__partition").fit(data.frame)
    exported = _export_memberships(data, {"Partition": transformer})
    assert "cluster__partition" in exported.frame
    assert "cluster__partition" not in exported.clean_frame
    assert stratifier_choices(exported) == ["group", "cluster__partition"]
    for goal in ("observations", "strata"):
        p = build_plan(exported, goal=goal, stratifier="cluster__partition", folds=2)
        assert p.ready, p.errors
        assert _overview(exported, p)["strata"]["name"] == "cluster__partition"
        membership = next(p.membership_frames(exported))
        assert list(membership.stratifier) == list(exported.frame.cluster__partition)
        assert restore_plan(exported, p.bookmark()) == p
        attached = exported.with_evaluation_plan(p, card="obs_resample")
        pipeline, train, validation = next(attached.training_partitions())
        assert "cluster__partition" not in train and "cluster__partition" not in validation
        assert "cluster__partition" in pipeline.fit_transform(train)
        changed = exported.clone()
        changed.frame["cluster__partition"] = changed.frame["cluster__partition"].array[::-1]
        assert not p.compatible(changed)
        if goal == "strata":
            for a, b in p.splits(exported):
                assert not set(exported.frame.cluster__partition.iloc[a]) & set(exported.frame.cluster__partition.iloc[b])
    # Even equal-size sampling cannot establish positional identity from duplicate indices.
    sampled = exported.with_sampling_step(SpySampler(), name="sample", preview_frame=exported.frame.iloc[::-1])
    invalid = build_plan(sampled, goal="observations", stratifier="cluster__partition", folds=2)
    assert not invalid.ready and "row-preserving" in invalid.errors[0]


def generated_source():
    from cards.obs_clusters import _export_memberships
    from ClusterMembershipTransformer import ClusterMembershipTransformer
    data = source()
    transformer = ClusterMembershipTransformer(("x",), n_clusters=3,
        output_column="cluster__partition").fit(data.frame)
    return _export_memberships(data, {"Partition": transformer})


@pytest.mark.parametrize("stratifier", [None, "group"])
def test_unused_generated_strata_do_not_block_sampling_or_invalidate_plan(stratifier):
    data = generated_source()
    p = build_plan(data, goal="observations", stratifier=stratifier, folds=2)
    sampled = data.with_sampling_step(SpySampler(), name="sample",
        preview_frame=data.frame.iloc[:12])
    current = build_plan(sampled, goal="observations", stratifier=stratifier, folds=2)
    assert current.ready, current.errors
    assert p.compatible(sampled)
    assert current.codes == p.codes
    assert _overview(sampled, current) is not None
    assert len(next(current.membership_frames(sampled))) == len(data.clean_frame)
    assert restore_plan(sampled, p.bookmark()) == p


def test_reweighting_preserves_generated_strata_with_duplicate_indices():
    from TargetBalanceSampler import TargetBalanceSampler
    data = generated_source()
    sampler = TargetBalanceSampler(target="y", mode="reweight")
    preview, _ = sampler.fit_resample(data.frame)
    assert sampler.preserves_row_order
    assert not TargetBalanceSampler(target="y", mode="resample").preserves_row_order
    pd.testing.assert_frame_equal(preview[data.frame.columns], data.frame)
    weighted = data.with_sampling_step(sampler, name="balance", preview_frame=preview,
        added_roles=RoleMap.from_primitive({"weighting": ["balance__weights"]}))
    p = build_plan(weighted, goal="strata", stratifier="cluster__partition", folds=2)
    assert p.ready, p.errors
    assert list(next(p.membership_frames(weighted)).stratifier) == list(data.frame.cluster__partition)
    assert restore_plan(weighted, p.bookmark()) == p


@pytest.mark.parametrize("change, message", [
    ("count", "preview has 12 rows"),
    ("index", "index labels or row order differ"),
    ("same_index", "sampling step(s) 'sample'"),
])
def test_generated_strata_alignment_errors_identify_actual_cause(change, message):
    data = generated_source()
    p = build_plan(data, goal="strata", stratifier="cluster__partition", folds=2)
    preview = data.frame.copy()
    if change == "count":
        preview = preview.iloc[:12]
    elif change == "index":
        preview.index = pd.RangeIndex(len(preview))
    else:
        preview = preview.iloc[::-1]  # Equal duplicate labels cannot prove identity.
    sampled = data.with_sampling_step(SpySampler(), name="sample", preview_frame=preview)
    invalid = build_plan(sampled, goal="strata", stratifier="cluster__partition", folds=2)
    assert not invalid.ready
    assert "cluster__partition" in invalid.errors[0]
    assert message in invalid.errors[0]
    assert not p.compatible(sampled)


def test_index_mismatch_without_sampling_is_not_reported_as_sampling():
    data = generated_source()
    data.frame.index = pd.RangeIndex(len(data.frame))
    invalid = build_plan(data, goal="strata", stratifier="cluster__partition", folds=2)
    assert not invalid.ready
    assert "index labels or row order differ" in invalid.errors[0]
    assert "sampling step(s)" not in invalid.errors[0]
    assert build_plan(data, goal="observations", stratifier=None, folds=2).ready
