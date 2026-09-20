import numpy as np
import pandas as pd
import pytest
from cards import obs_dependence as m
from playwright.sync_api import expect
from proxy_data import proxy_data
from roles import RoleMap
from shiny.pytest import create_app_fixture
from statsmodels.stats.diagnostic import acorr_ljungbox

app = create_app_fixture(app="../scenarios/obs_depence.py", scope="function")


def source(n=200):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame({"id": np.repeat(["A", "B"], n), "time": np.tile(np.arange(n), 2),
                          "trend": np.tile(np.arange(n, dtype=float), 2), "noise": rng.normal(size=2*n),
                          "weight": 1., "shadow__x": 2., "category": pd.Categorical(["a"]*(2*n))})
    roles = RoleMap.from_primitive({"identifier": ["id"], "sequence": ["time"],
                                   "predictor": ["trend", "noise", "shadow__x", "category"], "weighting": ["weight"]})
    return proxy_data(_df=frame, _roles=roles)


@pytest.mark.unit
class TestDependence:
    def test_rolling_uses_only_original_past_observations(self):
        x = np.array([np.nan, 2., 4., np.nan, np.nan, np.nan, 100.])
        filled, count, trimmed = m._rolling_impute(x, 2)
        np.testing.assert_allclose(filled, [2., 4., 3., 4., np.nan, 100.], equal_nan=True)
        assert count == 2 and trimmed == 1
        assert np.isnan(x[3])

    def test_imputation_coverage_entity_boundaries_and_unchanged(self):
        data = source(40)
        data.frame.loc[[0, 1, 10, 20, 40, 41, 50], "trend"] = np.nan
        before = data.clone()
        r = m._analyze(data)
        trend = r.table[r.table.Variable == "trend"]
        assert trend["Imputed"].tolist() == [2, 1]
        assert trend["Leading trimmed"].tolist() == [2, 2]
        assert trend["Analyzed rows"].tolist() == [38, 38]
        assert trend["Observed"].tolist() == [36, 37]
        assert trend["Adjusted p"].notna().all()
        assert trend.Status.str.contains("Exploratory").all()
        assert "imputation can introduce" in " ".join(r.notes)
        assert data.equals(before)
        data.frame.loc[5:15, "trend"] = np.nan
        assert pd.isna(m._analyze(data, window=2).table.iloc[0]["Adjusted p"])
        data.frame["trend"] = np.nan
        assert m._analyze(data).table.query("Variable == 'trend'")["Adjusted p"].isna().all()

    def test_reference_ljung_box(self):
        x = np.random.default_rng(32).normal(size=150)
        for h in [1, 5, 20]:
            p, _ = m._ljung_box(x, h)
            assert p == pytest.approx(acorr_ljungbox(x, lags=[h]).lb_pvalue.iloc[0])

    def test_structure_roles_order_and_unchanged(self):
        data = source()
        data.frame.index = [0]*len(data.frame)
        before = data.clone()
        r = m._analyze(data)
        assert data.equals(before)
        assert set(r.table.Variable) == {"trend", "noise"}
        assert len(r.table) == 4
        assert (r.table.loc[r.table.Variable == "trend", "Adjusted p"] < .05).all()
        assert "Longitudinal structure confirmed" in " ".join(r.notes)
        shuffled = proxy_data(_df=data.frame.sample(frac=1, random_state=1), _roles=data.role_map)
        pd.testing.assert_frame_equal(r.table.sort_values(["Entity", "Variable"]).reset_index(drop=True),
                                     m._analyze(shuffled).table.sort_values(["Entity", "Variable"]).reset_index(drop=True))

    def test_no_bridging_missing_irregular_ties(self):
        data = source()
        data.frame.loc[4, "trend"] = np.nan
        r = m._analyze(data, imputation="none")
        assert "Missing/nonfinite" in r.table.iloc[0].Status
        assert pd.isna(r.table.iloc[0]["Adjusted p"])
        data.frame["time"] = data.frame["time"].astype(float)
        data.frame.loc[4, "time"] = 4.5
        assert "Irregular" in m._analyze(data).table.iloc[0].Status
        data.frame.loc[4, "time"] = 3
        assert "not unique" in " ".join(m._analyze(data).notes)

    def test_short_constant_no_sequence_empty_limits(self):
        assert m._analyze(source(10)).table["Adjusted p"].isna().all()
        data = source()
        data.frame["trend"] = 1.
        assert "Constant" in m._analyze(data).table.iloc[0].Status
        data.role_map.set_roles("time", [])
        assert m._analyze(data).table.empty
        assert not m._analyze(data, row_order=True).table.empty
        data = source(0)
        assert "No observations" in " ".join(m._analyze(data).notes)
        assert "limit exceeded" in " ".join(m._analyze(source(50_001)).notes)

    def test_holm_and_actual_lags(self):
        r = m._analyze(source(30), lags=50)
        assert set(r.table.Lags) == {6}
        raw = r.table[["Ljung–Box p", "Squared p"]].to_numpy()
        from statsmodels.stats.multitest import multipletests
        expected = multipletests(raw.ravel(), method="holm")[1].reshape(raw.shape).min(axis=1)
        np.testing.assert_allclose(r.table["Adjusted p"], expected)

    def test_composite_keys_target_and_singletons(self):
        data = source(40)
        data.frame["site"] = "s"
        data.role_map.set_roles("site", ["identifier"])
        data.role_map.set_roles("trend", ["target"])
        assert len(m._analyze(data).table) == 4
        assert set(m._analyze(data, use_target=False).table.Variable) == {"noise"}
        data.frame["site"] = np.arange(len(data.frame))
        r = m._analyze(data)
        assert "singleton" in " ".join(r.notes)
        assert r.table["Adjusted p"].isna().all()

    def test_datetime_and_learned_pipeline(self):
        from sklearn.preprocessing import FunctionTransformer
        data = source(50)
        data.frame["time"] = pd.to_datetime(data.frame["time"], unit="D")
        learned = data.with_pipeline_step(FunctionTransformer(), name="identity",
                                         preview_frame=data.frame.copy())
        before = learned.clone()
        assert m._analyze(learned).table["Adjusted p"].notna().all()
        assert learned.has_pipeline and learned.equals(before)

    def test_squared_diagnostic_and_small_scales(self):
        rng = np.random.default_rng(7)
        magnitudes = np.repeat([1., 8., 1., 8.], 250)
        x = rng.choice([-1., 1.], len(magnitudes)) * magnitudes
        data = proxy_data(_df=pd.DataFrame({"time": np.arange(len(x)), "x": x}),
                          _roles=RoleMap.from_primitive({"sequence": ["time"], "predictor": ["x"]}))
        assert m._analyze(data).table.iloc[0]["Squared p"] < .001
        assert m._ljung_box(x * 1e-200, 10)[0] == pytest.approx(m._ljung_box(x, 10)[0])


@pytest.mark.ui
def test_card(page, app):
    page.set_viewport_size({"width": 1800, "height": 1100})
    page.goto(app.url)
    by_id = lambda name: page.locator(f'[id$="-{name}"]')
    expect(by_id("Status")).to_contain_text("Longitudinal structure confirmed", timeout=60000)
    expect(by_id("PassThrough")).to_have_text("unchanged=True")
    expect(by_id("Chart").locator(".js-plotly-plot")).to_be_visible()
    by_id("FlipButton").click(force=True)
    expect(by_id("Table")).to_contain_text("Ljung–Box p")
    expect(by_id("Table")).to_contain_text("4")
    expect(by_id("Table")).to_contain_text("Exploratory: rolling-imputed")
    by_id("Empty").click(force=True)
    expect(by_id("Status")).to_contain_text("No observations", timeout=60000)
    by_id("FlipButton").click(force=True)
    expect(by_id("Chart").locator(".annotation-text")).to_contain_text("No testable series")
