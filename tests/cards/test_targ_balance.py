"""Target balance arithmetic, training-only recipes, and browser behavior."""
import numpy as np
import pandas as pd
import pytest
from cards import targ_balance as m
from code_recording import recording_context
from module import Module
from playwright.sync_api import expect
from shiny.pytest import create_app_fixture
from proxy_data import proxy_data
from roles import Role, RoleMap
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from TargetBalanceSampler import TargetBalanceSampler

app = create_app_fixture(app='../scenarios/targ_balance.py', scope='function')
restored_app = create_app_fixture(app='../scenarios/targ_balance_restoring.py', scope='function')


def data(weighted=False):
    frame = pd.DataFrame({"x": np.r_[np.arange(12.), [1.5, 3.5, 5.5, 7.5]],
                          "cat": pd.Categorical(["u", "v"] * 8),
                          "target": pd.Categorical(["A"] * 12 + ["B"] * 4, categories=["B", "A", "unused"]),
                          "id": np.arange(16), "treatment": np.arange(16) + 100})
    roles = RoleMap.from_primitive({"predictor": ["x", "cat"], "target": ["target"],
                                   "identifier": ["id"], "treatment": ["treatment"]})
    if weighted:
        frame["importance"] = np.arange(16) + 1.
        roles.set_roles("importance", [Role.WEIGHTING])
    return proxy_data(_df=frame, _roles=roles)


def sampler(**kwargs):
    return TargetBalanceSampler(target="target", predictors=("x", "cat"), **kwargs)


@pytest.mark.unit
def test_reweight_preserves_mass_and_combines_existing_weights():
    source = data(True)
    original = source.clone()
    fitted = sampler(weight="importance", evaluation="balanced")
    out, y = fitted.fit_resample(source.frame)
    assert len(out) == 16 and y.equals(out.target)
    totals = out.groupby("target", observed=True).importance.sum()
    np.testing.assert_allclose(totals, [68., 68.])
    assert out.importance.sum() == pytest.approx(source.frame.importance.sum())
    assert source.equals(original)
    np.testing.assert_allclose(fitted.evaluation_weights(source.frame), out.importance)
    assert not hasattr(clone(fitted), "class_factors_")


@pytest.mark.unit
def test_none_is_identity_and_new_weight_name_avoids_collision():
    source = data()
    source.frame["Weights"] = 9
    assert m._analyze(source, {"mode": "none"}).export is source
    result = m._analyze(source, {"mode": "reweight"})
    assert not result.error, result.message
    assert result.export.role_map.roles_for("Weights_2") == {Role.WEIGHTING}
    assert (result.export.frame.Weights == 9).all()
    assert result.export.frame.Weights_2.mean() == pytest.approx(1)
    assert result.table["Class"].tolist() == ["B", "A"]
    assert "Weights_2" not in result.export.clean_frame
    assert not hasattr(result.export.pipeline.steps[-1][1], "class_factors_")


@pytest.mark.parametrize("mode", ["reweight", "resample"])
@pytest.mark.unit
def test_registered_recipe_refits_on_training_fold(mode):
    source = data()
    result = m._analyze(source, {"mode": mode})
    assert not result.error, result.message
    recipe = result.export.pipeline_for_training()
    fold = source.frame.iloc[[0, 1, 2, 3, 4, 5, 12, 13]]
    output, labels = recipe.fit_resample(fold, fold.target)
    fitted = recipe.steps[-1][1]
    assert fitted.class_factors_["A"] == pytest.approx(8 / 12)
    assert fitted.class_factors_["B"] == 2
    assert labels.equals(output.target)
    assert result.export.clean_frame.equals(source.clean_frame)


@pytest.mark.parametrize("down", ["random", "medoids", "centroids", "nearmiss", "stratified"])
@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
@pytest.mark.unit
def test_downsamplers_exact_counts_and_preserved_metadata(down, metric):
    source = data(True)
    original = source.clone()
    fitted = sampler(mode="resample", count_fraction=.5, down=down, metric=metric, weight="importance")
    output, _ = fitted.fit_resample(source.frame)
    assert output.target.value_counts().loc[["A", "B"]].tolist() == [8, 8]
    assert fitted.desired_count_ == 8
    expected = source.frame.iloc[fitted.sample_indices_]
    for col in source.columns:
        pd.testing.assert_series_equal(output[col], expected[col])
    assert source.equals(original)
    repeated, _ = clone(fitted).fit_resample(source.frame)
    pd.testing.assert_frame_equal(output, repeated)


@pytest.mark.parametrize("kind", ["mixed", "numeric", "nominal"])
@pytest.mark.unit
def test_smote_variants_preserve_donor_metadata_and_target(kind):
    source = data(True)
    predictors = {"mixed": ("x", "cat"), "numeric": ("x",), "nominal": ("cat",)}[kind]
    fitted = TargetBalanceSampler(target="target", predictors=predictors, weight="importance",
                                  mode="resample", up="smote", count_fraction=1.)
    output, _ = fitted.fit_resample(source.frame)
    assert len(output) == 24
    assert output.target.value_counts().loc[["A", "B"]].tolist() == [12, 12]
    assert fitted.synthetic_ == 8
    donor = source.frame.iloc[fitted.sample_indices_]
    for c in [c for c in source.columns if c not in predictors]:
        pd.testing.assert_series_equal(output[c], donor[c])
    assert output.cat.dtype == source.frame.cat.dtype


@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
@pytest.mark.unit
def test_soft_centroids_preserve_other_roles(metric):
    source = data(True)
    fitted = sampler(weight="importance", mode="resample", down="centroids", voting="soft",
                     count_fraction=0., metric=metric)
    output, _ = fitted.fit_resample(source.frame)
    assert len(output) == 8 and fitted.synthetic_ == 4
    donors = source.frame.iloc[fitted.sample_indices_]
    for c in ("id", "target", "importance", "treatment"):
        pd.testing.assert_series_equal(output[c], donors[c])
    assert set(output.cat).issubset(set(source.frame.cat))


@pytest.mark.unit
def test_missing_only_removed_when_required_and_unsupported_types_ignored():
    source = data()
    source.frame.loc[0, "x"] = np.nan
    source.frame.loc[1, "cat"] = np.nan
    source.frame["text"] = pd.Series(["hello"] * 16, dtype="string")
    random = sampler(mode="resample")
    random.fit_resample(source.frame)
    assert random.dropped_ == 0
    fitted = TargetBalanceSampler(target="target", predictors=("x", "cat", "text"),
                                  mode="resample", down="medoids")
    output, _ = fitted.fit_resample(source.frame)
    assert fitted.dropped_ == 2 and fitted.ignored_ == ["text"]
    assert not output[["x", "cat"]].isna().any().any()
    assert "Encode" in " ".join(fitted.warnings_)


@pytest.mark.unit
def test_missing_target_reweight_keeps_and_resample_removes():
    source = data()
    source.frame.loc[0, "target"] = np.nan
    reweighted, _ = sampler().fit_resample(source.frame)
    assert reweighted.loc[0, "Weights"] == 1
    fitted = sampler(mode="resample")
    output, _ = fitted.fit_resample(source.frame)
    assert fitted.dropped_ == 1 and not output.target.isna().any()


@pytest.mark.parametrize("bad", [-1., np.nan, np.inf])
@pytest.mark.unit
def test_invalid_weights_rejected(bad):
    source = data(True)
    source.frame.loc[0, "importance"] = bad
    with pytest.raises(ValueError, match="Weights must"):
        sampler(weight="importance").fit_resample(source.frame)


@pytest.mark.unit
def test_zero_mass_class_and_required_removal_fail_clearly():
    source = data(True)
    source.frame.loc[source.frame.target == "B", "importance"] = 0.
    with pytest.raises(ValueError, match="Every observed"):
        sampler(weight="importance").fit_resample(source.frame)
    source = data()
    source.frame.loc[source.frame.target == "B", "x"] = np.nan
    with pytest.raises(ValueError, match="empties a target class"):
        sampler(mode="resample", up="smote").fit_resample(source.frame)


@pytest.mark.unit
def test_singleton_smote_is_actionable_and_limits_precede_work():
    source = data().frame.iloc[:13]
    with pytest.raises(ValueError, match="at least two complete"):
        sampler(mode="resample", up="smote", count_fraction=1).fit_resample(source)
    with pytest.raises(ValueError, match="K-medoids is limited"):
        sampler(mode="resample", down="medoids", medoid_limit=5).fit_resample(data().frame)
    with pytest.raises(ValueError, match="output would exceed"):
        sampler(mode="resample", count_fraction=1, max_rows=20).fit_resample(data().frame)


@pytest.mark.unit
def test_duplicate_indices_and_identical_predictors_do_not_lose_rows():
    frame = data().frame
    frame.index = [0] * len(frame)
    frame["x"] = 1.
    frame["cat"] = pd.Categorical(["u"] * len(frame))
    fitted = sampler(mode="resample", down="medoids", count_fraction=0)
    output, _ = fitted.fit_resample(frame)
    assert len(output) == 8 and len(set(fitted.sample_indices_)) == 8
    assert list(output.index) == [0] * 8


@pytest.mark.unit
def test_evaluation_policy_and_unseen_classes():
    source = data(True)
    for policy in ("none", "incoming", "balanced"):
        fitted = sampler(weight="importance", evaluation=policy)
        fitted.fit_resample(source.frame)
        weights = fitted.evaluation_weights(source.frame)
        if policy == "none":
            assert weights is None
        elif policy == "incoming":
            np.testing.assert_allclose(weights, source.frame.importance)
        else:
            holdout = source.frame.copy()
            holdout["target"] = pd.Categorical(["new"] * len(holdout))
            with pytest.raises(ValueError, match="known, nonmissing"):
                fitted.evaluation_weights(holdout)


@pytest.mark.unit
def test_ineligible_cards_are_pass_through():
    source = data()
    source.role_map.set_roles("target", [Role.PREDICTOR])
    assert m._analyze(source, {"mode": "resample"}).export is source
    source.role_map.set_roles("target", [Role.TARGET])
    source.frame["target"] = np.arange(16)
    result = m._analyze(source, {"mode": "resample"})
    assert result.export is source and "not nominal" in result.message


@pytest.mark.unit
def test_diagnostic_scale_invariance_and_chart_consistency():
    source = data(True)
    first = m._diagnostic(source.frame, "target", "importance")
    source.frame["importance"] *= 100
    second = m._diagnostic(source.frame, "target", "importance")
    assert first["pvalue"] == pytest.approx(second["pvalue"])
    result = m._analyze(source, {"mode": "reweight"})
    assert result.after["balanced"]
    figure = m._figure(result, True)
    assert len(figure.data) == 5
    assert list(figure.data[0].x) == list(figure.data[2].labels) == ["B", "A"]
    assert tuple(figure.data[0].marker.color) == tuple(figure.data[2].marker.colors)
    assert m._analyze(data(), {"mode": "none"}).before["method"] == "chi-square approximation"
    tiny = data().frame.iloc[[0, 1, 12]]
    assert m._diagnostic(tiny, "target")["method"] == "multinomial simulation"


@pytest.mark.unit
def test_pipeline_preserved_when_transformers_follow_sampler_and_predict_skips_it():
    from sklearn.preprocessing import FunctionTransformer
    source = data()
    result = m._analyze(source, {"mode": "resample"})
    exported = result.export.with_pipeline_step(FunctionTransformer(), name="identity",
                                                preview_frame=result.export.frame)
    from imblearn.pipeline import Pipeline
    assert isinstance(exported.pipeline, Pipeline)
    trained = exported.pipeline_for_training()
    trained.steps.extend([("features", ColumnTransformer([("numeric", StandardScaler(), ["x"])])),
                          ("model", LogisticRegression())])
    trained.fit(source.clean_frame, source.clean_frame.target)
    assert trained.named_steps["targ_balance"].desired_count_ == 8
    holdout = pd.DataFrame({"x": [1., 2., 3.]})
    assert len(trained.predict(holdout)) == 3
    # Training changes neither preview nor the saved unfitted recipe.
    assert not hasattr(exported.pipeline.steps[0][1], "class_factors_")
    assert exported.clean_frame.equals(source.clean_frame)


@pytest.mark.unit
def test_sampling_step_validation_keeps_transformer_row_contract():
    from sklearn.preprocessing import FunctionTransformer
    source = data()
    with pytest.raises(ValueError, match="preserve the DataFrame index"):
        source.with_pipeline_step(FunctionTransformer(), name="bad", preview_frame=source.frame.iloc[:2])
    with pytest.raises(TypeError, match="fit_resample"):
        source.with_sampling_step(FunctionTransformer(), name="bad", preview_frame=source.frame)


@pytest.mark.unit
def test_recorded_helpers_and_sampler_execute_as_plain_python():
    import TargetBalanceSampler as sampling
    registry = {}
    source = data()
    recording_context(registry, m._analyze)(source, {"mode": "resample", "down": "medoids"})
    assert "TargetBalanceSampler" in registry and "balance_medoids" in registry
    namespace = {**vars(sampling), **vars(m)}
    for text in registry.values():
        exec(Module.clean_code(text), namespace)  # noqa: S102
    result = namespace["_analyze"](source, {"mode": "reweight"})
    assert not result.error


@pytest.mark.ui
def test_modes_restore_source_and_flip_keeps_result(page, app):
    errors = []
    page.on('pageerror', lambda error: errors.append(str(error)))
    page.goto(app.url)
    def by_id(name):
        return page.locator(f'[id$="-{name}"]')
    expect(by_id('ExportProbe')).to_contain_text('unchanged=True', timeout=30000)
    expect(by_id('Status')).to_contain_text('All-data preview', timeout=30000)
    page.get_by_label('Reweight', exact=True).check()
    expect(by_id('ExportProbe')).to_contain_text('weights=True steps=1', timeout=30000)
    page.locator('.card').first.hover()
    page.locator('button.flip-btn').click()
    expect(by_id('Summary')).to_contain_text('Class multiplier', timeout=30000)
    expect(by_id('Diagnostic')).to_contain_text('Advisory 5%', timeout=30000)
    page.locator('.card').first.hover()
    page.locator('button.flip-btn').click()
    page.get_by_label('Resample', exact=True).check()
    expect(by_id('Status')).to_contain_text('Requested 8 rows per class', timeout=30000)
    expect(by_id('ExportProbe')).to_contain_text('weights=False steps=1', timeout=30000)
    page.get_by_label('None', exact=True).check()
    expect(by_id('ExportProbe')).to_contain_text('steps=0 unchanged=True', timeout=30000)
    assert not errors


@pytest.mark.ui
def test_ineligible_target_disables_actions_and_passes_through(page, app):
    page.goto(app.url)
    status = page.locator('[id$="-Status"]')
    expect(status).to_contain_text('All-data preview', timeout=30000)
    page.get_by_label('Reweight', exact=True).check()
    expect(page.locator('[id$="-ExportProbe"]')).to_contain_text('weights=True', timeout=30000)
    page.get_by_label('Nominal target', exact=True).uncheck()
    expect(status).to_contain_text('not nominal', timeout=30000)
    expect(page.get_by_label('Reweight', exact=True)).to_be_disabled()
    expect(page.locator('[id$="-ExportProbe"]')).to_contain_text('weights=False steps=0', timeout=30000)


@pytest.mark.ui
def test_bookmarked_mode_and_count_are_restored(page, restored_app):
    page.goto(restored_app.url)
    expect(page.get_by_label('Resample', exact=True)).to_be_checked(timeout=30000)
    expect(page.locator('[id$="-ExportProbe"]')).to_contain_text('rows=24 weights=False steps=1', timeout=30000)
    page.get_by_label('None', exact=True).check()
    expect(page.locator('[id$="-ExportProbe"]')).to_contain_text('steps=0 unchanged=True', timeout=30000)


@pytest.mark.ui
def test_reference_band_hover_explains_limits(page, app):
    page.goto(app.url)
    expect(page.locator('[id$="-Status"]')).to_contain_text('All-data preview', timeout=30000)
    band = page.locator('.scatterlayer .js-fill').first
    expect(band).to_be_visible(timeout=30000)
    bounds = band.bounding_box()
    page.mouse.move(bounds['x'] + bounds['width'] / 2, bounds['y'] + bounds['height'] / 2)
    tooltip = page.locator('.hoverlayer')
    expect(tooltip).to_contain_text('Equal-frequency band:')
    expect(tooltip).to_contain_text('95% simultaneous')
    expect(tooltip).to_contain_text('original total weight')
