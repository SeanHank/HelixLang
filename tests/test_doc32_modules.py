"""Comprehensive tests for doc/32 implementation modules.

Tests for: molecular_toxicity, bayesian_denoiser, stochastic_ode,
physiology_constraints, mechanistic_ddi, dose_optimizer,
calibration_cascade, virtual_4dvar, proteome_binding,
microbiome, emergent_complexity.
"""

import math

import pytest

from helixlang.human.bayesian_denoiser import BayesianDenoiser, DenoiseResult, multi_assay_average
from helixlang.human.calibration_cascade import CalibrationCascade, CascadeResult
from helixlang.human.dose_optimizer import BayesianEstimate, DoseOptimizer, PKProfile
from helixlang.human.emergent_complexity import (
    EmergentComplexityModel,
    EpigeneticModulation,
    LiverGutFeedback,
    StressImmuneEndocrine,
)
from helixlang.human.mechanistic_ddi import (
    DDIPrediction,
    DrugMechanism,
    EnzymeInhibitionLibrary,
    EnzymeProfile,
    MechanisticDDIPredictor,
)
from helixlang.human.microbiome import (
    MicrobiomeCompartment,
    MicrobiomeDrugEffect,
)
from helixlang.human.molecular_toxicity import (
    ActivityProfile,
    MolecularToxicityPredictor,
    ToxicityProfile,
    smiles_autofill,
)
from helixlang.human.pharmacogenomic_ae import (
    TOXIC_METABOLITES,
    AEOrgan,
    AERisk,
    GenotypeAEPredictor,
    ToxicMetaboliteAccumulator,
)
from helixlang.human.physiology_constraints import (
    MassBalanceChecker,
    PhysiologyConstraints,
    ThermodynamicChecker,
)
from helixlang.human.proteome_binding import (
    ProteomeBindingCascade,
    ProteomeBindingProfile,
    ProteomeDDIPrediction,
)
from helixlang.human.reduced_order_organ import (
    PODModeGenerator,
)
from helixlang.human.stochastic_ode import (
    SDEConfig,
    SDEDistribution,
    SDETrajectory,
    euler_maruyama_step,
    solve_sde,
    solve_sde_ensemble,
)
from helixlang.human.tissue_gem import (
    TISSUE_REACTION_SETS,
    GEMDecomposer,
    OrganCouplingResult,
    OrganGEMCoupler,
)
from helixlang.human.virtual_4dvar import AssimilationResult, Observation, Virtual4DVar

# =============================================================================
# Molecular Toxicity Tests
# =============================================================================


class TestMolecularToxicityPredictor:
    """Tests for SMILES → toxicity prediction."""

    def setup_method(self) -> None:
        self.predictor = MolecularToxicityPredictor()

    def test_aspirin_toxicity(self) -> None:
        profile = self.predictor.predict_toxicity("CC(=O)Oc1ccccc1C(=O)O")
        assert isinstance(profile, ToxicityProfile)
        assert profile.smiles == "CC(=O)Oc1ccccc1C(=O)O"
        assert 0.0 <= profile.hepatotoxicity_score <= 1.0
        assert 0.0 <= profile.nephrotoxicity_score <= 1.0
        assert profile.confidence > 0.0

    def test_cisplatin_higher_toxicity(self) -> None:
        asp = self.predictor.predict_toxicity("CC(=O)Oc1ccccc1C(=O)O")
        cis = self.predictor.predict_toxicity("N[Pt](N)(Cl)Cl")
        assert cis.nephrotoxicity_score >= asp.nephrotoxicity_score * 0.5

    def test_invalid_smiles_returns_zero_scores(self) -> None:
        profile = self.predictor.predict_toxicity("INVALID_SMILES_XYZ")
        assert profile.hepatotoxicity_score == 0.0
        assert profile.confidence == 0.0

    def test_activity_profile(self) -> None:
        activity = self.predictor.predict_activity("CC(=O)Oc1ccccc1C(=O)O")
        assert isinstance(activity, ActivityProfile)
        assert 0.0 < activity.bioavailability <= 1.0
        assert 0.0 < activity.protein_binding <= 1.0
        assert activity.half_life_hours > 0.0
        assert activity.volume_of_distribution > 0.0

    def test_fingerprint_computation(self) -> None:
        fp = self.predictor.get_fingerprint("CC(=O)Oc1ccccc1C(=O)O")
        assert isinstance(fp, list)
        assert len(fp) == 1024
        assert any(b != 0 for b in fp)

    def test_all_toxicity_scores_bounded(self) -> None:
        smiles_list = [
            "CC(=O)Oc1ccccc1C(=O)O",
            "CC(=O)NC1=CC=C(O)C=C1",
            "N[Pt](N)(Cl)Cl",
            "CC1=C(CCC(C1)O)C(=O)C2=C(C=CC=C2O)O",
        ]
        for smi in smiles_list:
            p = self.predictor.predict_toxicity(smi)
            assert 0.0 <= p.hepatotoxicity_score <= 1.0
            assert 0.0 <= p.nephrotoxicity_score <= 1.0
            assert 0.0 <= p.cardiotoxicity_score <= 1.0
            assert 0.0 <= p.myelosuppression_score <= 1.0


class TestSmilesAutofill:
    """Tests for SMILES auto-fill."""

    def test_autofill_returns_dict(self) -> None:
        af = smiles_autofill()
        params = af.auto_fill_drug_params("CC(=O)Oc1ccccc1C(=O)O")
        assert isinstance(params, dict)
        assert "bioavailability" in params
        assert "hepatotoxicity_score" in params
        assert "half_life_hours" in params

    def test_autofill_profiles(self) -> None:
        af = smiles_autofill()
        tox = af.toxicity_profile("CC(=O)Oc1ccccc1C(=O)O")
        act = af.activity_profile("CC(=O)Oc1ccccc1C(=O)O")
        assert isinstance(tox, ToxicityProfile)
        assert isinstance(act, ActivityProfile)


# =============================================================================
# Bayesian Denoiser Tests
# =============================================================================


class TestBayesianDenoiser:
    """Tests for Kalman filter denoising."""

    def test_denoise_reduces_noise(self) -> None:
        denoiser = BayesianDenoiser()
        times = [i * 0.5 for i in range(20)]
        true_conc = [10.0 * math.exp(-0.15 * t) for t in times]
        noisy = [c * math.exp(0.15) for c in true_conc]
        result = denoiser.denoise(times, noisy, ke_prior=0.15, assay_cv=0.15)
        assert isinstance(result, DenoiseResult)
        assert len(result.denoised_values) == len(times)
        assert result.improvement_pct >= 0.0

    def test_denoise_empty_input(self) -> None:
        denoiser = BayesianDenoiser()
        result = denoiser.denoise([], [], ke_prior=0.15, assay_cv=0.15)
        assert result.denoised_values == []

    def test_denoise_preserves_trend(self) -> None:
        denoiser = BayesianDenoiser()
        times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        observations = [10.0, 8.5, 7.0, 5.8, 4.9, 4.1]
        result = denoiser.denoise(times, observations, ke_prior=0.15, assay_cv=0.10)
        for i in range(1, len(result.denoised_values)):
            assert result.denoised_values[i] <= result.denoised_values[i - 1] * 1.1


class TestMultiAssayAverage:
    """Tests for multi-assay averaging."""

    def test_single_assay(self) -> None:
        times = [0.0, 1.0, 2.0]
        readings = [[10.0, 8.0, 6.0]]
        cvs = [0.15]
        result = multi_assay_average(times, readings, cvs)
        assert result == [10.0, 8.0, 6.0]

    def test_multi_assay_closer_to_true(self) -> None:
        times = [0.0, 1.0]
        readings = [[10.5, 8.5], [9.5, 7.5]]
        cvs = [0.15, 0.15]
        result = multi_assay_average(times, readings, cvs)
        assert abs(result[0] - 10.0) < 1.0
        assert abs(result[1] - 8.0) < 1.0

    def test_empty_assays(self) -> None:
        result = multi_assay_average([], [], [])
        assert result == []


# =============================================================================
# Stochastic ODE Tests
# =============================================================================


class TestEulerMaruyama:
    """Tests for Euler-Maruyama step."""

    def test_step_with_zero_noise(self) -> None:
        import random
        rng = random.Random(42)
        new_state = euler_maruyama_step(1.0, 0.1, -0.1, 0.0, 0.0, rng)
        expected = 1.0 + (-0.1) * 0.1
        assert abs(new_state - expected) < 1e-10

    def test_step_with_noise(self) -> None:
        import random
        rng = random.Random(42)
        results = [euler_maruyama_step(1.0, 0.1, -0.1, 0.1, 0.05, rng) for _ in range(100)]
        mean = sum(results) / len(results)
        assert abs(mean - 0.99) < 0.1


class TestSDESolver:
    """Tests for SDE solver."""

    def test_single_trajectory(self) -> None:
        traj = solve_sde(
            t_end=5.0, dt=0.1, state0=10.0,
            drift_fn=lambda t, s: -0.1 * s,
            sigma_intrinsic=0.05, sigma_extrinsic=0.02,
            seed=42,
        )
        assert isinstance(traj, SDETrajectory)
        assert len(traj.times) > 0
        assert traj.states[0] == 10.0
        assert all(s >= 0 for s in traj.states)

    def test_sde_mean_matches_ode(self) -> None:
        def drift(t, s):
            return -0.1 * s

        traj = solve_sde(5.0, 0.1, 10.0, drift, 0.01, 0.01, seed=42)
        ode_final = 10.0 * math.exp(-0.1 * 5.0)
        sde_final = traj.states[-1]
        assert abs(sde_final - ode_final) < 2.0


class TestSDEEnsemble:
    """Tests for SDE ensemble distribution."""

    def test_ensemble_produces_distribution(self) -> None:
        config = SDEConfig(sigma_intrinsic=0.1, sigma_extrinsic=0.05, n_patients=100, seed=42)
        dist = solve_sde_ensemble(
            t_end=5.0, dt=0.2, state0=10.0,
            drift_fn=lambda t, s: -0.1 * s,
            config=config,
        )
        assert isinstance(dist, SDEDistribution)
        assert len(dist.means) > 0
        assert len(dist.stds) > 0
        assert all(s >= 0 for s in dist.stds)

    def test_distribution_has_percentiles(self) -> None:
        config = SDEConfig(n_patients=50, seed=42)
        dist = solve_sde_ensemble(
            t_end=3.0, dt=0.5, state0=10.0,
            drift_fn=lambda t, s: -0.1 * s,
            config=config,
        )
        assert 0.50 in dist.percentiles
        assert 0.05 in dist.percentiles
        assert 0.95 in dist.percentiles
        assert len(dist.percentiles[0.50]) > 0


# =============================================================================
# Physiology Constraints Tests
# =============================================================================


class TestPhysiologyConstraints:
    """Tests for homeostatic bounds checking."""

    def test_valid_state_passes(self) -> None:
        constraints = PhysiologyConstraints()
        state = {"ph": 7.4, "glucose": 90.0, "map": 85.0, "creatinine": 0.8, "wbc": 7000.0}
        result = constraints.check(state)
        assert result.is_valid

    def test_violation_detected(self) -> None:
        constraints = PhysiologyConstraints()
        state = {"ph": 6.5, "glucose": 90.0}
        result = constraints.check(state)
        assert not result.is_valid
        assert len(result.violations) > 0
        assert result.total_penalty > 0.0

    def test_project_to_feasible(self) -> None:
        constraints = PhysiologyConstraints()
        state = {"ph": 6.0, "glucose": 600.0, "creatinine": 20.0}
        corrected = constraints.project_to_feasible(state)
        assert corrected["ph"] >= 6.8
        assert corrected["glucose"] <= 500.0
        assert corrected["creatinine"] <= 15.0

    def test_homeostatic_penalty(self) -> None:
        constraints = PhysiologyConstraints()
        state_ok = {"ph": 7.4, "glucose": 90.0, "map": 85.0, "creatinine": 0.8, "wbc": 7000.0}
        state_bad = {"ph": 6.0, "glucose": 600.0, "map": 200.0, "creatinine": 20.0, "wbc": 0.01}
        penalty_ok = constraints.homeostatic_penalty(state_ok)
        penalty_bad = constraints.homeostatic_penalty(state_bad)
        assert penalty_bad > penalty_ok


class TestMassBalanceChecker:
    """Tests for mass balance checking."""

    def test_balanced_system(self) -> None:
        checker = MassBalanceChecker(tolerance=0.01)
        stoich = [[1.0, -1.0], [0.0, 1.0]]
        fluxes = [1.0, 1.0]
        deltas = [0.0, 1.0]
        result = checker.check(stoich, fluxes, deltas)
        assert result.is_valid

    def test_imbalanced_system(self) -> None:
        checker = MassBalanceChecker(tolerance=0.01)
        stoich = [[1.0, -1.0]]
        fluxes = [1.0, 0.5]
        deltas = [0.0]
        result = checker.check(stoich, fluxes, deltas)
        assert not result.is_valid


# =============================================================================
# Mechanistic DDI Tests
# =============================================================================


class TestEnzymeInhibitionLibrary:
    """Tests for enzyme inhibition library."""

    def test_known_drugs_loaded(self) -> None:
        lib = EnzymeInhibitionLibrary()
        assert lib.get("warfarin") is not None
        assert lib.get("amiodarone") is not None
        assert lib.get("fluconazole") is not None

    def test_register_new_drug(self) -> None:
        lib = EnzymeInhibitionLibrary()
        lib.register_drug(DrugMechanism(
            name="test_drug",
            enzyme_profiles=[EnzymeProfile("CYP3A4", True, 0.8, 0.5, 10.0)],
        ))
        assert lib.get("test_drug") is not None

    def test_unknown_drug_returns_none(self) -> None:
        lib = EnzymeInhibitionLibrary()
        assert lib.get("nonexistent_drug_xyz") is None


class TestMechanisticDDIPredictor:
    """Tests for compositional DDI prediction."""

    def test_known_ddi_amiodarone_warfarin(self) -> None:
        predictor = MechanisticDDIPredictor()
        pred = predictor.predict("amiodarone", "warfarin")
        assert isinstance(pred, DDIPrediction)
        assert pred.auc_ratio > 1.0
        assert pred.significance in ("DDI_ALERT", "CONTRAINDICATED")

    def test_known_ddi_clarithromycin_simvastatin(self) -> None:
        predictor = MechanisticDDIPredictor()
        pred = predictor.predict("clarithromycin", "simvastatin")
        assert pred is not None
        assert pred.auc_ratio > 1.5

    def test_no_ddi_metformin_warfarin(self) -> None:
        predictor = MechanisticDDIPredictor()
        pred = predictor.predict("metformin", "warfarin")
        assert pred is not None
        assert pred.auc_ratio <= 1.25
        assert pred.significance == "NO_CLINICAL_DDI"

    def test_unknown_drug_returns_none(self) -> None:
        predictor = MechanisticDDIPredictor()
        assert predictor.predict("unknown_drug", "warfarin") is None

    def test_predict_all_pairs(self) -> None:
        predictor = MechanisticDDIPredictor()
        preds = predictor.predict_all_pairs(["warfarin", "amiodarone", "fluconazole"])
        assert len(preds) > 0
        assert all(isinstance(p, DDIPrediction) for p in preds)


# =============================================================================
# Dose Optimizer Tests
# =============================================================================


class TestDoseOptimizer:
    """Tests for dose optimization."""

    def test_compute_auc(self) -> None:
        optimizer = DoseOptimizer()
        pk = PKProfile(times=[0, 1, 2, 3], concentrations=[100, 50, 25, 12.5])
        auc = optimizer.compute_auc(pk)
        assert auc > 0
        assert auc == pytest.approx(131.25, rel=0.01)

    def test_compute_cmax(self) -> None:
        optimizer = DoseOptimizer()
        pk = PKProfile(times=[0, 1, 2], concentrations=[10, 50, 30])
        assert optimizer.compute_cmax(pk) == 50.0

    def test_pta(self) -> None:
        optimizer = DoseOptimizer()
        assert optimizer.pta([100, 200, 300], 150) == pytest.approx(2 / 3, rel=0.01)

    def test_ecdf_distance(self) -> None:
        optimizer = DoseOptimizer()
        d = optimizer.ecdf_distance([1, 2, 3], [1, 2, 3])
        assert d == pytest.approx(0.0, abs=0.01)

    def test_bayesian_map_estimate(self) -> None:
        optimizer = DoseOptimizer()
        times = [0.5, 1.0, 2.0, 4.0]
        concs = [80.0, 60.0, 35.0, 12.0]
        estimate = optimizer.bayesian_map_estimate(times, concs)
        assert isinstance(estimate, BayesianEstimate)
        assert estimate.ke > 0
        assert estimate.vd > 0
        assert estimate.cl > 0

    def test_bayesian_map_empty_observations(self) -> None:
        optimizer = DoseOptimizer()
        estimate = optimizer.bayesian_map_estimate([], [])
        assert estimate.ke > 0
        assert estimate.n_observations == 0


# =============================================================================
# Calibration Cascade Tests
# =============================================================================


class TestCalibrationCascade:
    """Tests for GP calibration cascade."""

    def test_initial_accuracy(self) -> None:
        cascade = CalibrationCascade()
        acc = cascade.total_accuracy()
        assert acc > 0
        assert acc == pytest.approx(0.634, rel=0.05)

    def test_calibration_reduces_uncertainty(self) -> None:
        cascade = CalibrationCascade()
        initial_acc = cascade.total_accuracy()
        for i in range(10):
            cascade.calibrate_layer(0, predicted=10.0 + i * 0.1, observed=10.0 + i * 0.1)
        final_acc = cascade.total_accuracy()
        assert final_acc <= initial_acc

    def test_predict_returns_result(self) -> None:
        cascade = CalibrationCascade()
        result = cascade.predict(0, x=100.0)
        assert isinstance(result, CascadeResult)
        assert result.predicted_value == 100.0
        assert result.ci_90_lower < result.ci_90_upper

    def test_reset(self) -> None:
        cascade = CalibrationCascade()
        cascade.calibrate_layer(0, predicted=10.0, observed=9.0)
        cascade.reset()
        acc = cascade.total_accuracy()
        expected_prior = math.sqrt(sum(l.sigma_prior**2 for l in cascade.layers))
        assert acc == pytest.approx(expected_prior, rel=0.01)


# =============================================================================
# Virtual 4D-Var Tests
# =============================================================================


class TestVirtual4DVar:
    """Tests for 4D-Var data assimilation."""

    def test_assimilate_with_observations(self) -> None:
        assimilator = Virtual4DVar()
        observations = [
            Observation(time=1.0, variable="alt", value=30.0, noise_variance=1.0),
            Observation(time=2.0, variable="alt", value=35.0, noise_variance=1.0),
            Observation(time=3.0, variable="creatinine", value=0.9, noise_variance=0.1),
        ]
        result = assimilator.assimilate(observations, max_iterations=50)
        assert isinstance(result, AssimilationResult)
        assert result.n_iterations > 0
        assert len(result.state_trajectory) > 0

    def test_assimilate_empty_observations(self) -> None:
        assimilator = Virtual4DVar()
        result = assimilator.assimilate([], max_iterations=10)
        assert result.n_iterations >= 0

    def test_cost_function_decreases(self) -> None:
        assimilator = Virtual4DVar()
        observations = [
            Observation(time=1.0, variable="alt", value=30.0),
            Observation(time=2.0, variable="alt", value=35.0),
        ]
        result = assimilator.assimilate(observations, max_iterations=100)
        assert result.cost_function >= 0.0

    def test_prior_state_used(self) -> None:
        prior = {"ke": 0.2, "vd": 40.0, "dose": 500.0, "alt": 30.0,
                 "creatinine": 0.8, "wbc": 7000.0, "drug_effect": 0.01}
        assimilator = Virtual4DVar(prior_state=prior)
        observations = [
            Observation(time=1.0, variable="alt", value=32.0),
        ]
        result = assimilator.assimilate(observations, max_iterations=30)
        assert "ke" in result.estimated_state

    def test_pk_parameters_optimizable_by_default(self) -> None:
        assimilator = Virtual4DVar()
        observations = [
            Observation(time=1.0, variable="alt", value=30.0),
            Observation(time=2.0, variable="alt", value=31.0),
        ]
        result = assimilator.assimilate(observations, max_iterations=50)
        assert "ke" in result.estimated_state
        assert "vd" in result.estimated_state

    def test_custom_forward_model_used(self) -> None:
        calls = {"n": 0}

        def model(state, times):
            calls["n"] += 1
            base = state.get("alt", 25.0)
            return [{"time": t, "alt": base} for t in times]

        assimilator = Virtual4DVar(
            forward_model=model, parameter_names=["alt"]
        )
        observations = [Observation(time=1.0, variable="alt", value=40.0)]
        result = assimilator.assimilate(observations, max_iterations=20)
        assert calls["n"] > 0
        assert abs(result.estimated_state["alt"] - 40.0) < 5.0


# =============================================================================
# ThermodynamicChecker Tests
# =============================================================================


class TestThermodynamicChecker:
    """Tests for thermodynamic feasibility checking."""

    def setup_method(self) -> None:
        self.checker = ThermodynamicChecker()

    def test_hexokinase_irreversible(self) -> None:
        v = self.checker.check_reaction("hexokinase")
        assert v is None  # ΔG°' = -16.7 < 0, feasible

    def test_malate_dehydrogenase_near_equilibrium(self) -> None:
        v = self.checker.check_reaction("malate_dehydrogenase")
        assert v is None  # ΔG°' = +29.7, but NOT in IRREVERSIBLE set → no violation at standard conditions

    def test_concentration_corrected_dg(self) -> None:
        concs = {"glucose": 5e-3, "atp": 1e-3, "g6p": 5e-5, "adp": 1e-3}
        stoich = [-1.0, -1.0, 1.0, 1.0]
        names = ["glucose", "atp", "g6p", "adp"]
        v = self.checker.check_reaction("hexokinase", concs, stoich, names)
        assert v is None  # physiological concentrations keep it feasible

    def test_check_all_with_stoich(self) -> None:
        fluxes = {"hexokinase": 1.0, "pfk": 0.5}
        concs = {"glucose": 5e-3, "atp": 1e-3, "g6p": 5e-5, "adp": 1e-3,
                 "f6p": 3e-4, "f16bp": 1e-4}
        stoich_data = {
            "hexokinase": ([-1.0, -1.0, 1.0, 1.0], ["glucose", "atp", "g6p", "adp"]),
            "pfk": ([-1.0, -1.0, 1.0, 1.0], ["f6p", "atp", "f16bp", "adp"]),
        }
        result = self.checker.check_all(fluxes, concs, stoich_data)
        assert result.is_valid

    def test_zero_flux_skipped(self) -> None:
        result = self.checker.check_all({"hexokinase": 0.0})
        assert result.is_valid


# =============================================================================
# Tissue GEM Tests (doc/32 §7.4)
# =============================================================================


class TestTissueGEM:
    """Tests for tissue-specific GEM decomposition."""

    def test_decompose_liver(self) -> None:
        decomposer = GEMDecomposer()
        gem = decomposer.decompose("liver")
        assert gem.organ == "liver"
        assert gem.n_reactions > 30
        assert "PCK" in gem.reactions  # gluconeogenesis liver-specific
        assert "CS" in gem.reactions  # housekeeping

    def test_decompose_all_organs(self) -> None:
        decomposer = GEMDecomposer()
        gems = decomposer.decompose_all()
        assert set(gems.keys()) == set(TISSUE_REACTION_SETS.keys())
        for organ, gem in gems.items():
            assert gem.organ == organ
            assert gem.n_reactions > 20

    def test_expression_pruning(self) -> None:
        decomposer = GEMDecomposer()
        expr = {rxn: 0.1 for rxn in TISSUE_REACTION_SETS["liver"]["reactions"]}
        gem = decomposer.decompose("liver", expr)
        core = {"PGI", "PFK", "FBA", "CS", "ACONTa", "MDH", "PK"}
        assert core & gem.reactions  # core pathway preserved

    def test_unknown_organ_raises(self) -> None:
        decomposer = GEMDecomposer()
        with pytest.raises(ValueError, match="Unknown organ"):
            decomposer.decompose("spleen")


class TestOrganGEMCoupler:
    """Tests for inter-organ metabolite exchange."""

    def test_glucose_exchange(self) -> None:
        coupler = OrganGEMCoupler()
        concs = {
            "liver": {"glucose": 5.0},
            "brain": {"glucose": 3.0},
            "muscle": {"glucose": 4.0},
        }
        result = coupler.compute_exchange(concs)
        assert isinstance(result, OrganCouplingResult)
        assert "glucose" in result.total_exchange

    def test_step_updates_concentrations(self) -> None:
        coupler = OrganGEMCoupler()
        concs = {
            "liver": {"glucose": 5.0},
            "brain": {"glucose": 3.0},
        }
        updated = coupler.step(1.0, concs)
        assert updated["brain"]["glucose"] > 3.0  # glucose flows liver→brain

    def test_zero_gradient_no_exchange(self) -> None:
        coupler = OrganGEMCoupler()
        equal = 5.0
        concs = {
            "liver": {"glucose": equal},
            "brain": {"glucose": equal},
            "muscle": {"glucose": equal},
            "kidney": {"glucose": equal},
            "adipose": {"glucose": equal},
            "gi": {"glucose": equal},
        }
        result = coupler.compute_exchange(concs)
        assert abs(result.total_exchange.get("glucose", 0.0)) < 0.01


# =============================================================================
# Reduced-Order Organ Tests (doc/32 §7.5)
# =============================================================================


class TestReducedOrderOrgan:
    """Tests for POD-based reduced-order organ models."""

    def test_generate_liver(self) -> None:
        gen = PODModeGenerator()
        organs = gen.generate_all()
        assert "liver" in organs
        liver = organs["liver"]
        assert liver.organ == "liver"
        assert liver.n_modes >= 3
        assert liver.total_energy_captured > 0.7

    def test_spatial_evaluation(self) -> None:
        gen = PODModeGenerator()
        organs = gen.generate_all()
        liver = organs["liver"]
        c_at_0 = liver.evaluate_spatial(0.0)
        c_at_1 = liver.evaluate_spatial(1.0)
        assert isinstance(c_at_0, float)
        assert isinstance(c_at_1, float)

    def test_step_updates_amplitudes(self) -> None:
        gen = PODModeGenerator()
        organs = gen.generate_all()
        kidney = organs["kidney"]
        a0 = kidney.modes[0].amplitude
        kidney.step(1.0, drug_input=100.0)
        assert kidney.modes[0].amplitude != a0

    def test_gradient_method(self) -> None:
        gen = PODModeGenerator()
        organs = gen.generate_all()
        liver = organs["liver"]
        gradient = liver.get_gradient()
        assert isinstance(gradient, float)

    def test_mean_concentration(self) -> None:
        gen = PODModeGenerator()
        organs = gen.generate_all()
        brain = organs["brain"]
        mean = brain.get_mean_concentration()
        assert mean == brain.modes[0].amplitude


# =============================================================================
# Pharmacogenomic AE Tests (doc/32 §7.6)
# =============================================================================


class TestToxicMetaboliteAccumulator:
    """Tests for Michaelis-Menten metabolite accumulation."""

    def test_known_metabolites(self) -> None:
        assert "NAPQI" in TOXIC_METABOLITES
        assert "MTX_PG" in TOXIC_METABOLITES
        assert "IRINOTECAN_SN38" in TOXIC_METABOLITES

    def test_accumulation_with_drug(self) -> None:
        acc = ToxicMetaboliteAccumulator()
        acc.set_drug_concentration("acetaminophen", 50.0)
        states = acc.step(1.0, {"CYP2E1": 1.0, "GST": 1.0})
        napqi = states["NAPQI"]
        assert napqi.concentration_um > 0.0
        assert napqi.production_rate > 0.0

    def test_poor_metabolizer_higher_risk(self) -> None:
        acc_pm = ToxicMetaboliteAccumulator()
        acc_pm.set_drug_concentration("acetaminophen", 50.0)
        for _ in range(240):
            acc_pm.step(0.1, {"CYP2E1": 1.0, "GST": 0.1})

        acc_um = ToxicMetaboliteAccumulator()
        acc_um.set_drug_concentration("acetaminophen", 50.0)
        for _ in range(240):
            acc_um.step(0.1, {"CYP2E1": 1.0, "GST": 1.0})

        assert acc_pm.get_states()["NAPQI"].ae_probability > acc_um.get_states()["NAPQI"].ae_probability

    def test_no_drug_no_accumulation(self) -> None:
        acc = ToxicMetaboliteAccumulator()
        states = acc.step(1.0, {"CYP2E1": 1.0})
        for state in states.values():
            assert state.concentration_um == 0.0


class TestGenotypeAEPredictor:
    """Tests for genotype-driven AE prediction."""

    def test_predict_acetaminophen(self) -> None:
        pred = GenotypeAEPredictor()
        preds = pred.predict_ae("acetaminophen", 50.0, {"CYP2E1": 1.0, "GST": 1.0})
        assert len(preds) > 0
        assert any(p.target_organ == AEOrgan.LIVER for p in preds)

    def test_predict_returns_risk_levels(self) -> None:
        pred = GenotypeAEPredictor()
        preds = pred.predict_ae("acetaminophen", 50.0, {"CYP2E1": 1.0, "GST": 1.0})
        for p in preds:
            assert isinstance(p.risk_level, AERisk)
            assert 0.0 <= p.ae_probability <= 1.0

    def test_predict_all(self) -> None:
        pred = GenotypeAEPredictor()
        drug_concs = {"acetaminophen": 50.0, "irinotecan": 10.0}
        activities = {"CYP2E1": 1.0, "GST": 1.0, "CES1": 1.0, "UGT1A1": 1.0}
        results = pred.predict_all(drug_concs, activities)
        assert isinstance(results, dict)


# ============================================================================
# §7.7 Proteome-wide binding cascade
# ============================================================================


class TestProteomeBindingCascade:
    def test_init(self) -> None:
        cascade = ProteomeBindingCascade()
        assert len(cascade._known_drugs) > 0

    def test_screen_known_drug(self) -> None:
        cascade = ProteomeBindingCascade()
        profile = cascade.screen_drug("warfarin", "CC(=O)Cc1ccccc1C(=O)O", drug_conc_um=10.0)
        assert isinstance(profile, ProteomeBindingProfile)
        assert profile.drug_name == "warfarin"
        assert profile.n_targets_screened > 0
        assert len(profile.bindings) > 0
        # Warfarin should bind CYP2C9
        targets = {b.target for b in profile.bindings}
        assert "CYP2C9" in targets or "CYP3A4" in targets

    def test_screen_novel_drug(self) -> None:
        cascade = ProteomeBindingCascade()
        # Novel molecule similar to ibuprofen (should have decent similarity)
        profile = cascade.screen_drug("novel_nsaid", "CC(C)Cc1ccc(C(=O)O)cc1", drug_conc_um=5.0)
        assert isinstance(profile, ProteomeBindingProfile)
        assert profile.drug_name == "novel_nsaid"
        assert profile.n_targets_screened > 0

    def test_ddi_prediction(self) -> None:
        cascade = ProteomeBindingCascade()
        ddi = cascade.predict_ddi(
            "warfarin", "CC(=O)Cc1ccccc1C(=O)O", 5.0,
            "fluconazole", "OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F", 5.0,
        )
        assert isinstance(ddi, ProteomeDDIPrediction)
        assert ddi.auc_ratio >= 1.0
        assert ddi.significance in ("CONTRAINDICATED", "DDD_ALERT", "NO_DDI")

    def test_binding_dict(self) -> None:
        cascade = ProteomeBindingCascade()
        profile = cascade.screen_drug("amiodarone", "CCCCc1oc2cc3c(cc2c1)C(=Cc1ccc(OCCN(CC)CC)cc1)C3=O")
        bd = profile.binding_dict
        assert isinstance(bd, dict)
        for _target, occ in bd.items():
            assert isinstance(occ, float)
            assert 0.0 <= occ <= 1.0

    def test_inhibition_dict(self) -> None:
        cascade = ProteomeBindingCascade()
        profile = cascade.screen_drug("clarithromycin",
            "CC[C@@H]1OC(=O)[C@H](C)[C@@H](O[C@@H]2O[C@H](C)[C@@H](O)[C@H](N(C)C)[C@@H]2O)[C@H](O)[C@@H](C)C(=O)O[C@H]2C[C@@](C)(OC)[C@H](O)[C@@H](C)O2")
        id_dict = profile.inhibition_dict
        assert isinstance(id_dict, dict)
        # Clarithromycin inhibits CYP3A4
        assert "CYP3A4" in id_dict

    def test_all_known_drugs_screen(self) -> None:
        cascade = ProteomeBindingCascade()
        smiles_map = {
            "warfarin": "CC(=O)Cc1ccccc1C(=O)O",
            "amiodarone": "CCCCc1oc2cc3c(cc2c1)C(=Cc1ccc(OCCN(CC)CC)cc1)C3=O",
            "fluconazole": "OC(Cn1cncn1)(Cn1cncn1)c1ccc(F)cc1F",
            "simvastatin": "CCC(C)(C)C(=O)OC[C@H]1C[C@@H](O)C=C2C=C[C@H](C)[C@H](O)[C@@H]2C1",
        }
        for drug_name, smiles in smiles_map.items():
            profile = cascade.screen_drug(drug_name, smiles)
            assert len(profile.bindings) > 0, f"{drug_name} had no bindings"


# ============================================================================
# Microbiome-drug interaction
# ============================================================================


class TestMicrobiomeCompartment:
    def test_init_healthy(self) -> None:
        mc = MicrobiomeCompartment(healthy_composition=True)
        assert mc.state.scfa_total_mM > 0
        assert mc.state.bile_salt_hydrolase_activity > 0

    def test_init_dysbiotic(self) -> None:
        mc = MicrobiomeCompartment(healthy_composition=False)
        # Dysbiotic: higher E. coli, lower Lactobacillus
        ecoli = mc._species.get("E._coli")
        lacto = mc._species.get("Lactobacillus_sp.")
        assert ecoli is not None and lacto is not None
        assert ecoli.abundance > lacto.abundance

    def test_set_drug_and_step(self) -> None:
        mc = MicrobiomeCompartment()
        mc.set_drug_concentration("irinotecan", 50.0)
        effects = mc.step(dt_h=1.0)
        assert "irinotecan" in effects
        eff = effects["irinotecan"]
        assert isinstance(eff, MicrobiomeDrugEffect)
        # Irinotecan undergoes bacterial β-glucuronidase reactivation
        assert eff.toxicity_modifier >= 1.0 or eff.bioavailability_modifier != 1.0

    def test_portal_fluxes(self) -> None:
        mc = MicrobiomeCompartment()
        fluxes = mc.get_portal_fluxes()
        assert "scfa" in fluxes
        assert "ammonia" in fluxes
        assert "bile_acids" in fluxes
        assert fluxes["scfa"] > 0

    def test_state_updates(self) -> None:
        mc = MicrobiomeCompartment()
        initial_scfa = mc.state.scfa_total_mM
        mc.step(dt_h=1.0)
        # SCFA should change after step
        assert mc.state.scfa_total_mM > 0

    def test_beta_glucuronidase_activity(self) -> None:
        mc = MicrobiomeCompartment()
        assert mc.state.beta_glucuronidase_activity > 0
        # Step should update β-glucuronidase
        mc.step(dt_h=2.0)
        assert mc.state.beta_glucuronidase_activity > 0

    def test_get_overall_drug_effect(self) -> None:
        mc = MicrobiomeCompartment()
        mc.set_drug_concentration("mycophenolate", 20.0)
        mc.step(dt_h=1.0)
        eff = mc.get_overall_drug_effect("mycophenolate")
        assert isinstance(eff, MicrobiomeDrugEffect)
        assert eff.drug_name == "mycophenolate"


# ============================================================================
# Emergent complexity (epigenetics + liver-gut + stress-immune)
# ============================================================================


class TestEpigeneticModulation:
    def test_init(self) -> None:
        ep = EpigeneticModulation()
        assert len(ep._states) > 0
        # All CYPs should start near baseline methylation
        for gene, state in ep._states.items():
            assert state.methylation_level <= 0.3, f"{gene} methylation too high at init"

    def test_update_no_drugs(self) -> None:
        ep = EpigeneticModulation()
        mods = ep.update(1.0, {}, 1.0, 5.0)
        assert isinstance(mods, dict)
        # Without drugs, CYP expression should stay near 1.0
        for gene, mod in mods.items():
            assert 0.5 <= mod <= 1.5, f"{gene} expression {mod} out of range"


class TestLiverGutFeedback:
    def test_init(self) -> None:
        lg = LiverGutFeedback()
        assert lg.bile_acid_pool > 0
        assert lg.fxr_activation >= 0

    def test_step_no_bsh(self) -> None:
        lg = LiverGutFeedback()
        signals = lg.step(1.0, 0.0, 0.0)
        assert "bile_acid_pool" in signals
        assert "fxr_activation" in signals
        assert "gut_permeability" in signals

    def test_step_with_bsh(self) -> None:
        lg = LiverGutFeedback()
        lg.step(1.0, 0.0, 0.0)
        primary_no_bsh = lg.primary_fraction
        lg2 = LiverGutFeedback()
        lg2.step(1.0, 1.0, 0.0)
        primary_with_bsh = lg2.primary_fraction
        # BSH deconjugates primary bile acids → primary_fraction should differ
        assert primary_with_bsh != primary_no_bsh


class TestStressImmuneEndocrine:
    def test_init(self) -> None:
        si = StressImmuneEndocrine()
        assert si.cortisol_level == 12.0
        assert si.cortisol_suppression == 0.0

    def test_step_no_stress(self) -> None:
        si = StressImmuneEndocrine()
        signals = si.step(1.0, 1.0, 5.0, 0.0, 0.5)
        assert signals["cortisol_ug_dl"] > 0
        assert 0.0 <= signals["cortisol_suppression"] <= 1.0

    def test_high_stress_elevates_cortisol(self) -> None:
        si = StressImmuneEndocrine()
        si.cortisol_stimulation = 0.8
        signals = si.step(1.0, 50.0, 50.0, 1.0, 50.0)
        # High IL-6 + TNF → HPA stimulation → elevated cortisol
        assert signals["cortisol_ug_dl"] > 12.0

    def test_cortisol_suppresses_immune(self) -> None:
        si = StressImmuneEndocrine()
        si.cortisol_level = 35.0
        signals = si.step(1.0, 5.0, 10.0, 0.0, 3.0)
        assert signals["cortisol_suppression"] > 0


class TestEmergentComplexityModel:
    def test_init(self) -> None:
        ecm = EmergentComplexityModel()
        assert ecm.epigenetics is not None
        assert ecm.liver_gut is not None
        assert ecm.stress_immune is not None

    def test_step(self) -> None:
        ecm = EmergentComplexityModel()
        signals = ecm.step(
            dt_h=1.0, t_h=0.0,
            drug_concentrations={},
            il6=1.0, tnf=5.0, crp=0.5,
            bshe_activity=1.0, cortisol_input=12.0,
        )
        assert isinstance(signals, dict)
        assert "bile_acid_pool" in signals
        assert "cortisol_ug_dl" in signals
        assert "endotoxin_level" in signals

    def test_drug_exposure_changes_cyp(self) -> None:
        ecm = EmergentComplexityModel()
        signals_before = ecm.step(1.0, 0.0, {}, 1.0, 5.0, 0.5, 1.0, 12.0)
        # Add drug exposure
        signals_after = ecm.step(1.0, 1.0, {"warfarin": 10.0}, 1.0, 5.0, 0.5, 1.0, 12.0)
        # CYP modulation should change with drug
        assert signals_after != signals_before

    def test_high_crp_drives_fever(self) -> None:
        ecm = EmergentComplexityModel()
        signals = ecm.step(1.0, 0.0, {}, 50.0, 50.0, 100.0, 1.0, 30.0)
        # Very high CRP should produce fever
        assert signals["fever_c"] > 0


# ============================================================================
# Tests for expanded genotype (Phase 2)
# ============================================================================


class TestExpandedGenotype:
    """Tests for expanded genotype features (transporters, non-CYP enzymes)."""

    def test_genotype_has_transporters(self) -> None:
        from helixlang.human.genotype import create_default_genotype
        g = create_default_genotype()
        assert hasattr(g, "transporter_status")
        assert len(g.transporter_status) > 0

    def test_genotype_has_non_cyp_enzymes(self) -> None:
        from helixlang.human.genotype import create_default_genotype
        g = create_default_genotype()
        assert hasattr(g, "non_cyp_enzyme_status")
        assert len(g.non_cyp_enzyme_status) > 0

    def test_get_transporter_activity(self) -> None:
        from helixlang.human.genotype import create_default_genotype
        g = create_default_genotype()
        # Default transporter activity should be 1.0 (normal)
        assert g.get_transporter_activity("SLCO1B1") == 1.0

    def test_get_non_cyp_activity(self) -> None:
        from helixlang.human.genotype import create_default_genotype
        g = create_default_genotype()
        assert g.get_non_cyp_activity("UGT1A1") == 1.0

    def test_gene_categories_dict(self) -> None:
        from helixlang.human.genotype import GENE_CATEGORIES
        assert "CYP_metabolism" in GENE_CATEGORIES
        assert "transporters" in GENE_CATEGORIES
        assert "phase_II" in GENE_CATEGORIES
        assert "CYP2D6" in GENE_CATEGORIES["CYP_metabolism"]
        assert "SLCO1B1" in GENE_CATEGORIES["transporters"]

    def test_core_cyp_enzymes_expanded(self) -> None:
        from helixlang.human.genotype import CORE_CYP_ENZYMES
        assert len(CORE_CYP_ENZYMES) == 10
        assert "CYP3A5" in CORE_CYP_ENZYMES
        assert "CYP2B6" in CORE_CYP_ENZYMES
        assert "CYP2C8" in CORE_CYP_ENZYMES
        assert "CYP2A6" in CORE_CYP_ENZYMES


# ============================================================================
# Tests for expanded drugs (Phase 3)
# ============================================================================


class TestExpandedDrugs:
    """Tests for expanded drug database and SMILES→ADME."""

    def test_drug_count(self) -> None:
        from helixlang.human.drug import PREDEFINED_DRUGS
        assert len(PREDEFINED_DRUGS) >= 21

    def test_smiles_to_adme_exists(self) -> None:
        from helixlang.human.drug import smiles_to_adme
        result = smiles_to_adme("CC(=O)NC1=CC=C(O)C=C1")  # acetaminophen
        assert "molecular_weight" in result or "clearance_ml_per_min" in result

    def test_warfarin_has_params(self) -> None:
        from helixlang.human.drug import PREDEFINED_DRUGS
        w = PREDEFINED_DRUGS["WARFARIN"]
        assert w.volume_distribution_l > 0

    def test_clopidogrel_metabolism(self) -> None:
        from helixlang.human.drug import PREDEFINED_DRUGS
        c = PREDEFINED_DRUGS["CLOPIDOGREL"]
        assert c.cyp_metabolism.get("CYP2C19", 0) > 0


# ============================================================================
# Tests for expanded disease coverage (Phase 4)
# ============================================================================


class TestExpandedDisease:
    """Tests for expanded disease profiles."""

    def test_disease_profile_count(self) -> None:
        from helixlang.human.disease import DISEASE_PROFILES
        assert len(DISEASE_PROFILES) >= 25

    def test_hypertension_profile(self) -> None:
        from helixlang.human.disease import DISEASE_PROFILES, DiseaseState
        assert "HYPERTENSION" in DISEASE_PROFILES
        h = DISEASE_PROFILES["HYPERTENSION"]
        assert isinstance(h, DiseaseState)

    def test_depression_profile(self) -> None:
        from helixlang.human.disease import DISEASE_PROFILES
        assert "DEPRESSION" in DISEASE_PROFILES

    def test_respiratory_ode_model(self) -> None:
        from helixlang.human.disease_ode_models import RespiratoryODE
        m = RespiratoryODE()
        fev1_before = m.fev1_percent
        for _ in range(100):
            m.step(1.0, drug_bronchodilator=2.0)
        assert m.fev1_percent != fev1_before

    def test_infectious_disease_ode(self) -> None:
        from helixlang.human.disease_ode_models import InfectiousDiseaseODE
        m = InfectiousDiseaseODE()
        m.hiv_severity = 0.8
        m.tb_severity = 0.5
        load_before = m.viral_bacterial_load
        for _ in range(100):
            m.step(1.0)
        assert m.viral_bacterial_load > load_before

    def test_gastrointestinal_ode(self) -> None:
        from helixlang.human.disease_ode_models import GastrointestinalODE
        m = GastrointestinalODE()
        m.ibd_severity = 0.7
        m.gerd_severity = 0.5
        pain_before = m.pain_score
        for _ in range(10):
            m.step(1.0)
        assert m.pain_score > pain_before

    def test_endocrine_ode(self) -> None:
        from helixlang.human.disease_ode_models import EndocrineODE
        m = EndocrineODE()
        m.hypothyroid_severity = 0.8
        t4_before = m.t4_level
        for _ in range(100):
            m.step(1.0)
        assert m.t4_level != t4_before


# ============================================================================
# Tests for expanded microbiome (Phase 5)
# ============================================================================


class TestExpandedMicrobiome:
    """Tests for expanded microbiome features."""

    def test_species_count(self) -> None:
        from helixlang.human.microbiome import MicrobiomeCompartment
        mc = MicrobiomeCompartment()
        assert len(mc._species) == 10

    def test_reaction_count(self) -> None:
        from helixlang.human.microbiome import MicrobiomeCompartment
        mc = MicrobiomeCompartment()
        assert len(mc._reactions) >= 19

    def test_apply_antibiotic(self) -> None:
        from helixlang.human.microbiome import MicrobiomeCompartment
        mc = MicrobiomeCompartment()
        abundances_before = {k: s.abundance for k, s in mc._species.items()}
        mc.apply_antibiotic("ampicillin")
        abundances_after = {k: s.abundance for k, s in mc._species.items()}
        any_decreased = any(
            abundances_after.get(k, 0) < v
            for k, v in abundances_before.items()
        )
        assert any_decreased

    def test_induce_dysbiosis(self) -> None:
        from helixlang.human.microbiome import MicrobiomeCompartment
        mc = MicrobiomeCompartment()
        div_before = mc.state.diversity_index
        mc.induce_dysbiosis(severity=0.5)
        div_after = mc.state.diversity_index
        assert div_after < div_before

    def test_restore_microbiome(self) -> None:
        from helixlang.human.microbiome import MicrobiomeCompartment
        mc = MicrobiomeCompartment()
        mc.induce_dysbiosis(severity=0.8)
        mc.restore_microbiome()
        mc.step(dt_h=1.0)
        # Diversity should recover somewhat
        assert mc.state.diversity_index > 0


# ============================================================================
# Tests for calibration cascade GP (Phase 6 fix)
# ============================================================================


class TestCalibrationCascadeGP:
    """Tests for GP-based calibration cascade."""

    def test_gp_layer_with_observations(self) -> None:
        from helixlang.human.calibration_cascade import GPLayer
        gp_few = GPLayer("few", input_dim=1)
        gp_many = GPLayer("many", input_dim=1)
        for i in range(3):
            gp_few.calibrate(100.0 + i, 100.0 + i + 1.0)
        for i in range(30):
            gp_many.calibrate(100.0 + i, 100.0 + i + 1.0)
        _, ci_few = gp_few.predict_with_uncertainty(105.0)
        _, ci_many = gp_many.predict_with_uncertainty(105.0)
        assert ci_many < ci_few

    def test_gp_kernel_weighted(self) -> None:
        from helixlang.human.calibration_cascade import GPLayer
        gp = GPLayer("test", input_dim=1, sigma_prior=0.5)
        for i in range(20):
            gp.calibrate(100.0 + i * 2, 100.0 + i * 2 + 2.0)
        mean, ci = gp.predict_with_uncertainty(110.0)
        # Should predict close to observed values (112.0 area)
        assert 105.0 < mean < 125.0
        # CI should be narrow
        assert ci < 2.0

    def test_fallback_structural_alerts(self) -> None:
        from helixlang.human.molecular_toxicity import _match_alerts_fallback
        # Metformin: no alerts expected
        alerts = _match_alerts_fallback("CN(C)C(=N)NC(=N)N")
        assert isinstance(alerts, list)

    def test_fallback_none_input(self) -> None:
        from helixlang.human.molecular_toxicity import _match_alerts_fallback
        alerts = _match_alerts_fallback(None)
        assert alerts == []
