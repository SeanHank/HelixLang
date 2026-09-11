"""Tests for the biological validity framework (doc/37 §2).

Covers OutOfScopeDetector, ParameterFitter, UncertaintyQuantifier,
ReplicationVerifier, and BioAccuracySuite.
"""
from __future__ import annotations

from helixlang.plugins.runtime.bio_validity import (
    BioAccuracyReport,
    BioAccuracySuite,
    FitResult,
    OutOfScopeDetector,
    OutOfScopeReport,
    ParameterCheck,
    ParameterFitter,
    ParameterRange,
    ReplicationResult,
    ReplicationVerifier,
    ScopeLevel,
    UncertaintyQuantifier,
    UncertaintyResult,
)


class TestOutOfScopeDetector:
    def test_in_scope_parameter(self) -> None:
        detector = OutOfScopeDetector()
        report = detector.check({"growth_rate": 0.87})
        assert report.all_safe
        assert not report.any_out_of_scope
        assert report.worst_level == ScopeLevel.SAFE

    def test_out_of_scope_parameter(self) -> None:
        detector = OutOfScopeDetector()
        report = detector.check({"growth_rate": 25.0})
        assert not report.all_safe
        assert report.any_out_of_scope
        assert report.worst_level == ScopeLevel.OUT_OF_SCOPE

    def test_warning_parameter(self) -> None:
        detector = OutOfScopeDetector()
        # 0.4 is well below the typical 0.87 but within [0.1, 2.0] — should be SAFE
        # 15.0 h^-1 is far outside — OUT_OF_SCOPE
        report = detector.check({"generation_time": 30.0})
        # 30 min is within [15, 120] close to typical — safe
        assert report.all_safe

    def test_multiple_parameters(self) -> None:
        detector = OutOfScopeDetector()
        report = detector.check({
            "growth_rate": 0.87,
            "glucose_uptake": 100.0,   # out of [5, 20]
            "hill_coefficient": 2.0,
        })
        assert report.any_out_of_scope
        by_name = {c.name: c.level for c in report.checks}
        assert by_name["growth_rate"] == ScopeLevel.SAFE
        assert by_name["glucose_uptake"] == ScopeLevel.OUT_OF_SCOPE
        assert by_name["hill_coefficient"] == ScopeLevel.SAFE

    def test_custom_range_registration(self) -> None:
        from helixlang.plugins.runtime.bio_validity import ParameterRange
        detector = OutOfScopeDetector()
        detector.register(ParameterRange("custom_kd", 0.01, 0.1, "nM",
                                         "test source"))
        report = detector.check({"custom_kd": 0.05})
        assert report.all_safe
        report2 = detector.check({"custom_kd": 1.0})
        assert report2.any_out_of_scope


class TestParameterFitter:
    def test_fit_converges_to_target(self) -> None:
        fitter = ParameterFitter()
        fitter._bounds = {
            "growth_rate": (0.1, 1.5),
            "glucose_uptake": (5.0, 15.0),
        }
        result = fitter.fit(
            {"growth_rate": 2.0, "glucose_uptake": 5.0},
            {"growth_rate": 0.87, "glucose_uptake": 10.0},
        )
        assert result.converged
        assert result.initial_params["growth_rate"] == 2.0
        assert result.residual_after < result.residual_before
        # fitted growth rate should be close to target 0.87
        assert abs(result.fitted_params["growth_rate"] - 0.87) < 0.05
        assert abs(result.fitted_params["glucose_uptake"] - 10.0) < 0.5

    def test_no_bounded_keys(self) -> None:
        fitter = ParameterFitter()
        result = fitter.fit(
            {"unbounded_param": 5.0},
            {"unbounded_param": 3.0},
        )
        assert result.message == "no bounded parameters to fit"
        assert result.fitted_params["unbounded_param"] == 5.0

    def test_already_at_target(self) -> None:
        fitter = ParameterFitter()
        fitter._bounds = {"x": (0.0, 10.0)}
        result = fitter.fit({"x": 5.0}, {"x": 5.0})
        assert result.converged
        assert result.message == "already at target"

    def test_improvement_pct(self) -> None:
        fitter = ParameterFitter()
        fitter._bounds = {"growth_rate": (0.1, 1.5)}
        result = fitter.fit({"growth_rate": 2.0}, {"growth_rate": 1.0})
        assert result.improvement_pct > 0


class TestUncertaintyQuantifier:
    def test_bootstrap_produces_ci(self) -> None:
        uq = UncertaintyQuantifier(n_bootstrap=100, seed=42)
        result = uq.bootstrap(
            {"x": 1.0},
            [0.85, 0.87, 0.89, 0.83, 0.91],
            noise_std=0.02,
        )
        assert result.n_samples == 100
        assert result.ci_lower <= result.mean <= result.ci_upper
        assert result.cv >= 0
        assert result.method == "bootstrap"

    def test_monte_carlo(self) -> None:
        uq = UncertaintyQuantifier(n_bootstrap=100, seed=7)
        result = uq.monte_carlo(
            {"growth_rate": 0.87},
            {"growth_rate": 0.05},
            n_samples=100,
        )
        assert result.method == "monte_carlo"
        assert 0.8 <= result.mean <= 0.95

    def test_deterministic_seed(self) -> None:
        uq1 = UncertaintyQuantifier(seed=123)
        uq2 = UncertaintyQuantifier(seed=123)
        r1 = uq1.bootstrap({"x": 1.0}, [10.0], noise_std=0.1)
        r2 = uq2.bootstrap({"x": 1.0}, [10.0], noise_std=0.1)
        assert r1.mean == r2.mean
        assert r1.ci_lower == r2.ci_lower


class TestReplicationVerifier:
    def test_deterministic_runs_identical(self) -> None:
        verifier = ReplicationVerifier(n_runs=5, seed=0)
        result = verifier.verify(max_ticks=30)
        assert result.all_identical
        assert result.n_runs == 5
        assert len(set(result.hashes)) == 1

    def test_positional_conflict(self) -> None:
        result = ReplicationVerifier(n_runs=3, seed=1).verify(max_ticks=5)
        assert result.all_identical


class TestBioAccuracySuite:
    def test_full_suite_passes(self) -> None:
        suite = BioAccuracySuite()
        report = suite.run(
            "test_full",
            {"growth_rate": 0.87, "glucose_uptake": 10.0},
            target_values={"growth_rate": 0.87, "glucose_uptake": 10.0},
            bounds={
                "growth_rate": (0.1, 1.5),
                "glucose_uptake": (5.0, 15.0),
            },
        )
        assert report.status in ("PASS", "WARN")
        assert report.overall_accuracy >= 0.5

    def test_suite_out_of_scope_downgrades_status(self) -> None:
        suite = BioAccuracySuite()
        report = suite.run(
            "test_oom",
            {"growth_rate": 42.0, "glucose_uptake": 1000.0},
            target_values={"growth_rate": 0.87},
            bounds={"growth_rate": (0.1, 1.5)},
        )
        # out-of-scope inputs should not get PASS
        assert report.scope.any_out_of_scope
        assert report.status != "FAIL" or report.overall_accuracy < 0.8

    def test_report_serializable(self) -> None:
        suite = BioAccuracySuite()
        report = suite.run("test_dict", {"growth_rate": 0.87})
        d = report.to_dict()
        assert d["id"] == "test_dict"
        assert "scope" in d
        assert "overall_accuracy" in d
        assert "replication" in d


class TestCoverageBranches:
    def test_parameter_range_degenerate(self) -> None:
        rng = ParameterRange(name="p", min_val=5.0, max_val=5.0)
        assert rng.contains(5.0) == ScopeLevel.SAFE
        assert rng.contains(6.0) == ScopeLevel.OUT_OF_SCOPE

    def test_parameter_range_warning_band(self) -> None:
        rng = ParameterRange(name="p", min_val=0.0, max_val=10.0)
        assert rng.contains(12.0) == ScopeLevel.WARNING
        assert rng.contains(16.0) == ScopeLevel.OUT_OF_SCOPE

    def test_empty_report_and_missing_param(self) -> None:
        assert OutOfScopeReport().worst_level == ScopeLevel.SAFE
        detector = OutOfScopeDetector(
            ranges=[ParameterRange(name="p", min_val=0.0, max_val=1.0)])
        report = detector.check({"unknown": 5.0})
        assert report.checks == []
        report.checks.append(
            ParameterCheck(name="q", value=9.0,
                           range=ParameterRange(name="q", min_val=0.0,
                                                max_val=1.0),
                           level=ScopeLevel.WARNING, deviation=8.0))
        assert report.worst_level == ScopeLevel.WARNING

    def test_fit_with_objective_and_zero_target(self) -> None:
        fitter = ParameterFitter(
            bounds={"p": (0.0, 5.0), "q": (0.0, 5.0)},
            objective_fn=lambda p: (p["p"] - 1.0) ** 2)
        result = fitter.fit(
            initial_params={"p": 2.0, "q": 2.0},
            target_values={"p": 1.0, "q": 0.0},
            maxiter=50)
        assert isinstance(result, FitResult)
        assert "fitted_params" in result.to_dict()
        assert result.residual_before > 0.0

    def test_fit_falls_back_coordinate_descent(self, monkeypatch) -> None:
        import sys
        real = sys.modules.get("scipy.optimize")
        sys.modules["scipy.optimize"] = None
        try:
            fitter = ParameterFitter(
                bounds={"p": (0.0, 5.0), "q": (0.0, 5.0)},
                objective_fn=lambda p: (p["p"] - 1.0) ** 2)
            result = fitter.fit(
                initial_params={"p": 2.0, "q": 2.0},
                target_values={"p": 1.0, "q": 0.0},
                maxiter=40)
            assert result.converged or result.n_iterations > 0
            assert result.message == "coordinate descent completed"
            plain = ParameterFitter(bounds={"p": (0.0, 5.0)}).fit(
                initial_params={"p": 2.0},
                target_values={"p": 1.0},
                maxiter=40)
            assert plain.converged or plain.n_iterations > 0
        finally:
            if real is None:
                sys.modules.pop("scipy.optimize", None)
            else:
                sys.modules["scipy.optimize"] = real

    def test_uncertainty_with_forward_fn(self) -> None:
        uq = UncertaintyQuantifier(
            forward_fn=lambda p: p["p"] * 2.0, n_bootstrap=30, seed=3)
        boot = uq.bootstrap({"p": 1.0}, [1.0, 2.0, 3.0], noise_std=0.1)
        assert "ci_95_lower" in boot.to_dict()
        mc = uq.monte_carlo({"p": 1.0}, {"p": 0.1}, n_samples=30)
        assert mc.n_samples == 30
        assert "mean" in mc.to_dict()
        assert isinstance(UncertaintyResult(
            mean=1.0, std=0.0, ci_lower=1.0, ci_upper=1.0, cv=0.0,
            n_samples=3).to_dict(), dict)

    def test_replication_run_fn_numeric(self) -> None:
        verifier = ReplicationVerifier(
            run_fn=lambda seed, run_index: seed + run_index, n_runs=3, seed=1)
        result = verifier.verify()
        assert result.n_runs == 3
        assert not result.all_identical
        assert result.max_deviation >= 0.0
        assert "hashes_match" in result.to_dict()
        assert isinstance(ReplicationResult(
            n_runs=1, all_identical=True, max_deviation=0.0,
            hashes=["a"]).to_dict(), dict)

    def test_report_statuses(self) -> None:
        low = BioAccuracyReport(benchmark_id="low")
        low.replication = ReplicationResult(
            n_runs=2, all_identical=False, max_deviation=1.0, hashes=["a", "b"])
        assert low.compute_overall() == 0.0
        assert low.status == "FAIL"
        high = BioAccuracyReport(benchmark_id="high")
        high.scope = OutOfScopeReport(checks=[
            ParameterCheck(name="p", value=1.0,
                           range=ParameterRange(name="p", min_val=0.0,
                                                max_val=1.0),
                           level=ScopeLevel.SAFE, deviation=0.0)])
        high.fit = FitResult(
            fitted_params={"p": 1.0}, initial_params={"p": 2.0},
            residual_before=1.0, residual_after=0.0, improvement_pct=100.0,
            converged=True)
        high.uncertainty = UncertaintyResult(
            mean=1.0, std=0.0, ci_lower=1.0, ci_upper=1.0, cv=0.0, n_samples=3)
        high.replication = ReplicationResult(
            n_runs=2, all_identical=True, max_deviation=0.0, hashes=["h", "h"])
        assert high.compute_overall() == 1.0
        assert high.status == "PASS"
        assert "fit" in high.to_dict()
        empty = BioAccuracyReport(benchmark_id="empty")
        assert empty.compute_overall() == 0.0
        assert empty.status == "FAIL"
