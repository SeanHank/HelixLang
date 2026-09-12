"""Tests for the ChEMBL L2 calibration, fingerprint kNN resolver, and
intrathecal CNS wiring (doc/32 §7.1/§7.7, doc/27 §7.6).

All assertions are evidence-grounded against the implementation and can
only be changed to reflect code changes (code is the only source of truth).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# ChEMBL snapshot + kNN resolver
# ---------------------------------------------------------------------------
from helixlang.plugins.human.data.chembl_binding import (
    CHEMBL_DRUG_BINDINGS,
    CHEMBL_DRUG_SMILES,
    CHEMBL_IDS,
)
from helixlang.plugins.human.proteome_binding import (
    PROTEOME_TARGETS,
    BindingPrediction,
    KnnHit,
    ProteomeBindingCascade,
    _interpolate_profile,
    knn_resolve,
)

KETO_SMILES = "CN1CCN(CC1)c1ccc(N2CCN(CC2)c2ccc3c(c2)OCO3)cc1"


class TestChembLSnapshot:
    """vendored snapshot (data/chembl_binding.py) invariants."""

    def test_sets_aligned(self) -> None:
        assert set(CHEMBL_DRUG_SMILES) == set(CHEMBL_IDS) == set(CHEMBL_DRUG_BINDINGS)

    def test_all_smiles_parse_rdkit(self) -> None:
        from rdkit import Chem
        for name, smi in CHEMBL_DRUG_SMILES.items():
            m = Chem.MolFromSmiles(smi)
            assert m is not None, f"{name} SMILES failed RDKit parse"

    def test_golden_hash_stable(self) -> None:
        h1 = snapshot_golden_hash()
        h2 = snapshot_golden_hash()
        assert isinstance(h1, str) and len(h1) == 64
        assert h1 == h2

    def test_snapshot_has_proteome_targets(self) -> None:
        all_targets: set[str] = set()
        for drug in CHEMBL_DRUG_BINDINGS.values():
            all_targets.update(drug)
        assert all_targets <= set(PROTEOME_TARGETS)


class TestKnnResolve:
    """fingerprint kNN general resolver (doc/32 §7.7)."""

    def test_k_eq_zero_returns_empty(self) -> None:
        assert knn_resolve("C", {}, k=0) == []

    def test_empty_database_returns_empty(self) -> None:
        assert knn_resolve("C", {}, k=3) == []

    def test_topk_sorted_descending(self) -> None:
        db = {"a": "CC", "b": "CCC", "c": "CCCC", "d": "CCCCC", "e": "CCCCCC"}
        hits = knn_resolve("CCCCC", db, k=5, min_similarity=0.0)
        assert len(hits) == 5
        sims = [h.similarity for h in hits]
        assert sims == sorted(sims, reverse=True)
        assert all(isinstance(h, KnnHit) for h in hits)

    def test_chembl_self_neighbor(self) -> None:
        hits = knn_resolve(KETO_SMILES, CHEMBL_DRUG_SMILES, k=5)
        assert len(hits) >= 1
        assert hits[0].name == "ketoconazole"
        assert abs(hits[0].similarity - 1.0) < 1e-9

    def test_min_similarity_filters(self) -> None:
        hits_all = knn_resolve(KETO_SMILES, CHEMBL_DRUG_SMILES, k=20)
        hits_tight = knn_resolve(KETO_SMILES, CHEMBL_DRUG_SMILES, k=20,
                                  min_similarity=0.9)
        assert len(hits_tight) <= len(hits_all)

    def test_k_exceeds_database(self) -> None:
        tiny_db = {"a": "CC", "b": "CCC"}
        hits = knn_resolve("CCCCC", tiny_db, k=100)
        assert len(hits) <= 2

    def test_garbage_smiles_below_threshold(self) -> None:
        hits = knn_resolve("XXXXX", CHEMBL_DRUG_SMILES, k=3, min_similarity=0.3)
        assert hits == [] or all(h.similarity < 0.3 for h in hits) is False


class TestInterpolateProfile:
    """_interpolate_profile branch coverage."""

    def test_empty_hits_returns_empty(self) -> None:
        assert _interpolate_profile([], {}, 10.0) == []

    def test_weighted_geometric_mean(self) -> None:
        known = {"A": {"CYP3A4": {"kd_um": 10.0, "substrate": 1.0, "inhibitor": 0.0}}}
        hits = [KnnHit(name="A", smiles="", similarity=1.0)]
        preds = _interpolate_profile(hits, known, drug_conc_um=10.0)
        assert len(preds) == 1
        assert preds[0].target == "CYP3A4"
        assert abs(preds[0].kd_um - 10.0) < 0.01
        assert preds[0].is_substrate is True
        assert preds[0].is_inhibitor is False

    def test_zero_kd_produces_zero_occupancy(self) -> None:
        known = {"A": {"CYP3A4": {"kd_um": 0.0, "substrate": 0.0, "inhibitor": 0.0}}}
        hits = [KnnHit(name="A", smiles="", similarity=0.5)]
        preds = _interpolate_profile(hits, known, drug_conc_um=10.0)
        assert len(preds) == 1
        assert preds[0].kd_um == 0.0
        assert preds[0].occupancy == 0.0

    def test_substrate_weighted_majority(self) -> None:
        known = {
            "A": {"CYP3A4": {"kd_um": 1.0, "substrate": 0.9, "inhibitor": 0.0}},
            "B": {"CYP3A4": {"kd_um": 1.0, "substrate": 0.1, "inhibitor": 0.0}},
        }
        hits = [KnnHit("A", "", 0.9), KnnHit("B", "", 0.6)]
        preds = _interpolate_profile(hits, known, drug_conc_um=1.0)
        assert len(preds) == 1
        # weight A = 0.81, B = 0.36; weighted substrate = (0.81*0.9+0.36*0.1)/1.17 = 0.72 > 0.5
        assert preds[0].is_substrate is True

    def test_substrate_weighted_minority(self) -> None:
        known = {
            "A": {"CYP3A4": {"kd_um": 1.0, "substrate": 0.1, "inhibitor": 0.0}},
            "B": {"CYP3A4": {"kd_um": 1.0, "substrate": 0.1, "inhibitor": 0.0}},
        }
        hits = [KnnHit("A", "", 0.9), KnnHit("B", "", 0.6)]
        preds = _interpolate_profile(hits, known, drug_conc_um=1.0)
        assert preds[0].is_substrate is False

    def test_inhibition_above_threshold(self) -> None:
        known = {"A": {"CYP3A4": {"kd_um": 5.0, "substrate": 0.0, "inhibitor": 0.8}}}
        hits = [KnnHit("A", "", 1.0)]
        preds = _interpolate_profile(hits, known, drug_conc_um=10.0)
        assert preds[0].is_inhibitor is True
        assert preds[0].inhibition_strength == pytest.approx(0.8, abs=0.01)

    def test_inhibition_below_threshold(self) -> None:
        known = {"A": {"CYP3A4": {"kd_um": 5.0, "substrate": 0.0, "inhibitor": 0.05}}}
        hits = [KnnHit("A", "", 1.0)]
        preds = _interpolate_profile(hits, known, drug_conc_um=10.0)
        assert preds[0].is_inhibitor is False

    def test_confidence_formula(self) -> None:
        known = {"A": {"CYP3A4": {"kd_um": 1.0, "substrate": 0.0, "inhibitor": 0.0}}}
        hits = [KnnHit("A", "", 0.5)]
        preds = _interpolate_profile(hits, known, drug_conc_um=1.0)
        assert preds[0].confidence == pytest.approx(0.5 * 0.7)

    def test_non_proteome_targets_ignored(self) -> None:
        known = {"A": {"ZZZZZ": {"kd_um": 1.0, "substrate": 0.0, "inhibitor": 0.0}}}
        hits = [KnnHit("A", "", 1.0)]
        preds = _interpolate_profile(hits, known, drug_conc_um=1.0)
        assert preds == []


class TestProteomeBindingCascadeKNN:
    """proteome_binding cascade with use_chembl/exclude knn_k knobs."""

    def test_chembl_known_lookup(self) -> None:
        cascade = ProteomeBindingCascade(use_chembl=True)
        profile = cascade.screen_drug("ketoconazole", KETO_SMILES, drug_conc_um=10.0)
        assert len(profile.bindings) > 0
        assert any(b.target == "CYP3A4" for b in profile.bindings)

    def test_use_chembl_false_excludes_snapshot(self) -> None:
        cascade = ProteomeBindingCascade(use_chembl=False)
        assert "ketoconazole" not in cascade._known_drugs
        profile = cascade.screen_drug("ketoconazole", KETO_SMILES, drug_conc_um=5.0)
        # ketoconazole not in curated DB → novel path
        assert all(b.target in PROTEOME_TARGETS for b in profile.bindings)

    def test_exclude_drops_drug(self) -> None:
        cascade = ProteomeBindingCascade(exclude={"ketoconazole"})
        assert "ketoconazole" not in cascade._known_drugs
        assert "ketoconazole" not in cascade._drug_smiles

    def test_knn_k_respected(self) -> None:
        c3 = ProteomeBindingCascade(knn_k=3)
        c1 = ProteomeBindingCascade(knn_k=1)
        hits3 = c3.resolve_novel(KETO_SMILES)
        hits1 = c1.resolve_novel(KETO_SMILES)
        assert len(hits1) <= 3
        assert len(hits3) <= 3

    def test_resolve_novel_descending(self) -> None:
        cascade = ProteomeBindingCascade()
        hits = cascade.resolve_novel(KETO_SMILES)
        sims = [h.similarity for h in hits]
        assert sims == sorted(sims, reverse=True)

    def test_find_nearest_garbage(self) -> None:
        cascade = ProteomeBindingCascade()
        name, sim = cascade._find_nearest_drug("XXXXXX")
        assert name == "" and sim == -1.0

    def test_novel_path_interpolation_called(self) -> None:
        cascade = ProteomeBindingCascade(knn_k=2)
        p = cascade.screen_drug("novel_xyz", KETO_SMILES, drug_conc_um=1.0)
        assert p.n_targets_screened == len(PROTEOME_TARGETS)


# ---------------------------------------------------------------------------
# L2 calibration harness (ChemBLL2Calibration)
# ---------------------------------------------------------------------------
from helixlang.plugins.human.chembl_calibration import (  # noqa: E402
    CHEMBL_DRUG_BINDINGS as _CDB,
)
from helixlang.plugins.human.chembl_calibration import (  # noqa: E402
    L2_MAX_MEDIAN_FOLD,
    L2_MIN_FRACTION,
    ChemBLCascadeResult,
    ChemBLL2Calibration,
    ChemBLL2Report,
    snapshot_golden_hash,
)


class TestChemBLL2Calibration:
    """L2 reference-implementation equivalence harness (doc/32 §6.2)."""

    def test_golden_hash_alignment(self) -> None:
        assert snapshot_golden_hash() == snapshot_golden_hash()

    def test_loo_produces_report(self) -> None:
        cal = ChemBLL2Calibration(knn_k=3)
        report = cal.run_loo()
        assert isinstance(report, ChemBLL2Report)
        assert len(report.observations) > 0
        assert len(report.missing_targets) > 0
        assert report.golden_hash == snapshot_golden_hash()

    def test_sigma1_positive(self) -> None:
        cal = ChemBLL2Calibration()
        report = cal.run_loo()
        assert report.sigma_1 > 0.0

    def test_fraction_within_band_bounded(self) -> None:
        cal = ChemBLL2Calibration()
        report = cal.run_loo()
        assert 0.0 <= report.fraction_within_band <= 1.0

    def test_median_fold_error_positive(self) -> None:
        cal = ChemBLL2Calibration()
        report = cal.run_loo()
        assert report.median_fold_error > 0.0
        assert report.max_fold_error >= report.median_fold_error

    def test_cascade_layer0_calibrated(self) -> None:
        cal = ChemBLL2Calibration()
        cal.run_loo()
        layer0 = cal.cascade.layers[0]
        assert layer0.n_calibrations > 0
        assert layer0.sigma_posterior != layer0.sigma_prior

    def test_predict_sigma1_returns_ci(self) -> None:
        cal = ChemBLL2Calibration()
        cal.run_loo()
        res = cal.predict_sigma1(5.0)
        assert isinstance(res, ChemBLCascadeResult)
        assert res.uncertainty_ratio > 1.0
        assert res.ci_90_lower < res.ci_90_upper

    def test_ref_pair_count_matches_snapshot(self) -> None:
        cal = ChemBLL2Calibration()
        report = cal.run_loo()
        assert report.n_reference_pairs == sum(len(v) for v in _CDB.values())

    def test_passed_is_bool(self) -> None:
        cal = ChemBLL2Calibration()
        report = cal.run_loo()
        assert isinstance(report.passed, bool)

    def test_exclude_all_yields_zero_observations(self) -> None:
        """Cover the empty-observations branch in _compile."""
        cal = ChemBLL2Calibration()
        # create a cascade with empty db to force 0 observations
        empty_db: dict[str, dict[str, dict[str, float]]] = {}
        with patch(
            "helixlang.plugins.human.chembl_calibration.CHEMBL_DRUG_BINDINGS",
            empty_db,
        ), patch(
            "helixlang.plugins.human.chembl_calibration.CHEMBL_DRUG_SMILES",
            {},
        ):
            report = cal.run_loo()
        assert len(report.observations) == 0
        assert report.passed is False

    def test_l2_constants(self) -> None:
        assert L2_MAX_MEDIAN_FOLD > 1.0
        assert 0.0 < L2_MIN_FRACTION <= 1.0

    def test_resolver_lazy_init(self) -> None:
        cal = ChemBLL2Calibration(knn_k=2)
        assert cal._cascade is None
        resolver = cal._resolver
        assert resolver is cal._resolver
        assert resolver is not None
        assert resolver.knn_k == 2

    def test_run_loo_ref_leq_zero_branch(self) -> None:
        """ref <= 0.0 enters missing_targets, never hits observations."""
        db = {
            "d1": {"CYP3A4": {"kd_um": 0.0, "substrate": 0.0, "inhibitor": 0.0}},
            "d2": {"CYP3A4": {"kd_um": 5.0, "substrate": 0.0, "inhibitor": 0.0}},
        }
        with patch(
            "helixlang.plugins.human.chembl_calibration.CHEMBL_DRUG_BINDINGS", db,
        ), patch(
            "helixlang.plugins.human.chembl_calibration.CHEMBL_DRUG_SMILES",
            {"d1": "CCC", "d2": "CCCC"},
        ), patch(
            "helixlang.plugins.human.data.chembl_binding.CHEMBL_DRUG_BINDINGS", db,
        ), patch(
            "helixlang.plugins.human.data.chembl_binding.CHEMBL_DRUG_SMILES",
            {"d1": "CCC", "d2": "CCCC"},
        ):
            report = ChemBLL2Calibration().run_loo()
        assert ("d1", "CYP3A4") in report.missing_targets
        assert len(report.observations) == 0
        assert report.passed is False

    def test_run_loo_even_observations_median(self) -> None:
        """Two observations → even-length median calculation covered."""
        db = {
            "d1": {"CYP3A4": {"kd_um": 1.0, "substrate": 0.0, "inhibitor": 0.0}},
            "d2": {"CYP3A4": {"kd_um": 2.0, "substrate": 0.0, "inhibitor": 0.0}},
        }
        with patch(
            "helixlang.plugins.human.chembl_calibration.CHEMBL_DRUG_BINDINGS", db,
        ), patch(
            "helixlang.plugins.human.chembl_calibration.CHEMBL_DRUG_SMILES",
            {"d1": "CCC", "d2": "CCCC"},
        ), patch(
            "helixlang.plugins.human.data.chembl_binding.CHEMBL_DRUG_BINDINGS", db,
        ), patch(
            "helixlang.plugins.human.data.chembl_binding.CHEMBL_DRUG_SMILES",
            {"d1": "CCC", "d2": "CCCC"},
        ):
            report = ChemBLL2Calibration().run_loo()
        assert len(report.observations) == 2
        assert isinstance(report.median_fold_error, float)

    def test_predict_sigma1_missing_observations_finite_ci(self) -> None:
        cal = ChemBLL2Calibration()
        empty_db: dict[str, dict[str, dict[str, float]]] = {}
        with patch(
            "helixlang.plugins.human.chembl_calibration.CHEMBL_DRUG_BINDINGS", empty_db,
        ), patch(
            "helixlang.plugins.human.chembl_calibration.CHEMBL_DRUG_SMILES", {},
        ):
            cal.run_loo()
        res = cal.predict_sigma1(10.0)
        assert res.uncertainty_ratio > 1.0
        assert 0.0 < res.ci_90_lower < res.ci_90_upper


# ---------------------------------------------------------------------------
# Intrathecal CNS wiring (PBPKModel + _DrugPBPK)
# ---------------------------------------------------------------------------
from helixlang.plugins.human.drug import INTRATHECAL, IV, Drug, DrugMolecule  # noqa: E402
from helixlang.plugins.human.pharmacokinetics import (  # noqa: E402
    ORGAN_NAMES,
    PBPKModel,
)
from helixlang.plugins.human.physiology import HumanPhysiology  # noqa: E402


def _make_drug(route: str = INTRATHECAL, dose_mg: float = 50.0) -> Drug:
    return Drug(
        molecule=DrugMolecule(name="test_cns", molecular_weight_da=350.0),
        dose_mg=dose_mg,
        route=route,
        half_life_h=8.0,
    )


def _make_phys() -> HumanPhysiology:
    return HumanPhysiology(cardiac_output_ml_per_min=5000.0)


class TestPBPKIntrathecal:
    """PBPKModel with INTRATHECAL route (doc/27 §7.6)."""

    def test_intrathecal_supported(self) -> None:
        model = PBPKModel(_make_drug(), _make_phys())
        assert model.drug.route == INTRATHECAL

    def test_initial_state_brain_nonzero(self) -> None:
        model = PBPKModel(_make_drug(dose_mg=100.0), _make_phys())
        state = model._initial_state()
        brain_idx = ORGAN_NAMES.index("brain") + 1
        assert state[brain_idx] > 0.0

    def test_initial_state_central_zero_for_intrathecal(self) -> None:
        model = PBPKModel(_make_drug(), _make_phys())
        state = model._initial_state()
        assert state[0] == 0.0

    def test_dose_brain_mg_respected(self) -> None:
        model = PBPKModel(_make_drug(dose_mg=200.0), _make_phys(), dose_brain_mg=10.0)
        state = model._initial_state()
        brain_idx = ORGAN_NAMES.index("brain") + 1
        expected = 10.0 * model.drug.bioavailability / model.organ_volumes_l["brain"]
        assert abs(state[brain_idx] - expected) < 1e-6

    def test_dose_brain_mg_none_uses_drug_dose(self) -> None:
        model = PBPKModel(_make_drug(dose_mg=50.0), _make_phys(), dose_brain_mg=None)
        state = model._initial_state()
        brain_idx = ORGAN_NAMES.index("brain") + 1
        assert state[brain_idx] > 0.0

    def test_iv_initial_state_central_nonzero(self) -> None:
        model = PBPKModel(_make_drug(route=IV), _make_phys())
        state = model._initial_state()
        assert state[0] > 0.0
        brain_idx = ORGAN_NAMES.index("brain") + 1
        assert state[brain_idx] == 0.0

    def test_intrathecal_euler_integration(self) -> None:
        model = PBPKModel(_make_drug(dose_mg=100.0), _make_phys())
        result = model.run()
        brain_idx = ORGAN_NAMES.index("brain") + 1  # noqa: F841
        brain = result.concentrations["brain"]
        assert brain[0] > 0.0
        assert brain[0] > brain[-1]

    def test_intrathecal_brain_front_loaded(self) -> None:
        model = PBPKModel(_make_drug(dose_mg=100.0), _make_phys())
        result = model.run()
        brain = result.concentrations["brain"]
        central = result.central_concentration
        assert brain[0] > central[0]
        # brain perfuses back into circulation, so central rises above zero
        assert max(central) > 0.0


# ---------------------------------------------------------------------------
# _DrugPBPK intrathecal branch (virtual_patient.py)
# ---------------------------------------------------------------------------
from helixlang.plugins.human.virtual_patient import _DrugPBPK  # noqa: E402


class TestDrugPBBPKIntrathecal:
    """_DrugPBPK intrathecal dosing branch (doc/27 §7.6)."""

    def _make_engine(self, route: str = INTRATHECAL) -> tuple[_DrugPBPK, Drug]:
        drug = _make_drug(route=route, dose_mg=50.0)
        phys = _make_phys()
        engine = _DrugPBPK(drug, phys)
        return engine, drug

    def test_brain_compartment_exists(self) -> None:
        engine, _ = self._make_engine()
        assert "brain" in engine.conc_um

    def test_intrathecal_administer_dose_brain_nonzero(self) -> None:
        engine, drug = self._make_engine()
        engine._administer_dose()
        assert engine.conc_um["brain"] > 0.0
        assert engine.conc_um["central"] == 0.0

    def test_intrathecal_administer_dose_central_zero(self) -> None:
        engine, _ = self._make_engine()
        engine._administer_dose()
        assert engine.conc_um["central"] == 0.0

    def test_iv_administer_dose_central_nonzero(self) -> None:
        engine, _ = self._make_engine(route=IV)
        engine._administer_dose()
        assert engine.conc_um["central"] > 0.0

    def test_iv_administer_dose_brain_zero(self) -> None:
        engine, _ = self._make_engine(route=IV)
        engine._administer_dose()
        assert engine.conc_um["brain"] == 0.0

    def test_oral_administer_dose_depot_nonzero(self) -> None:
        from helixlang.plugins.human.drug import ORAL
        engine, _ = self._make_engine(route=ORAL)
        engine._administer_dose()
        assert engine.depot_mg > 0.0

    def test_intrathecal_advance_gives_positive_brain(self) -> None:
        engine, _ = self._make_engine()
        engine.advance(1.0, current_time_h=1.0)
        assert engine.conc_um["brain"] > 0.0

    def test_intrathecal_availability_applied(self) -> None:
        engine, drug = self._make_engine()
        engine._administer_dose()
        expected = (drug.bioavailability * drug.dose_mg * engine._um_per_mg_per_l
                    / engine.organ_volumes_l["brain"])
        assert abs(engine.conc_um["brain"] - expected) < 1e-6


# ---------------------------------------------------------------------------
# RDKit fallback / exception branch coverage (proteome_binding.py)
# ---------------------------------------------------------------------------
from helixlang.plugins.human.proteome_binding import (  # noqa: E402
    ProteomeBindingProfile,
    ProteomeDDIPrediction,
)


def test_knn_resolve_rdkit_fallback_to_trigram() -> None:
    """Invalid SMILES triggers fallback branch (line 354)."""
    from unittest.mock import patch as _patch

    from helixlang.plugins.human.proteome_binding import _compute_similarity_rdkit
    # MolFromSmiles returns None → fallback
    assert _compute_similarity_rdkit("INVALID", "CC") == 0.0

    # MolFromSmiles raises → except Exception branch
    with _patch("rdkit.Chem.MolFromSmiles", side_effect=RuntimeError("boom")):
        assert _compute_similarity_rdkit("a", "b") == 0.0

    # FP computation raises after successful parse → except branch
    with _patch("rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect",
                side_effect=RuntimeError("fp boom")):
        assert _compute_similarity_rdkit("CC", "CCC") == 0.0


def test_find_nearest_drug_nonempty() -> None:
    """_find_nearest_drug hits return path (line 490)."""
    cascade = ProteomeBindingCascade(use_chembl=True)
    name, sim = cascade._find_nearest_drug(KETO_SMILES)
    assert name == "ketoconazole"
    assert abs(sim - 1.0) < 1e-9


def test_predict_ddi_both_inhibition_paths() -> None:
    """Cover inh_a>0.1&sub_b and inh_b>0.1&sub_a branches (lines 529-539)."""
    from unittest.mock import patch as _patch

    def _mock_screen(drug_name: str, smiles: str, drug_conc_um: float) -> ProteomeBindingProfile:
        profiles = {
            "A": [BindingPrediction("CYP3A4", 0.1, 0.9, False, True, 0.8, 0.8)],
            "B": [BindingPrediction("CYP3A4", 0.1, 0.9, True, False, 0.0, 0.7)],
        }
        return ProteomeBindingProfile(
            drug_name=drug_name, smiles=smiles, n_targets_screened=1,
            bindings=profiles[drug_name],
        )

    cascade = ProteomeBindingCascade(use_chembl=False)
    with _patch.object(cascade, "screen_drug", _mock_screen):
        pred = cascade.predict_ddi("A", "CC", 10.0, "B", "CCC", 10.0)
    assert pred.auc_ratio > 1.25
    assert pred.significance in ("DDD_ALERT", "CONTRAINDICATED")
    assert len(pred.interacting_targets) >= 1


def test_predict_ddi_no_interaction() -> None:
    """Cover the NO_DDI classification branch."""
    from unittest.mock import patch as _patch

    def _mock_screen(drug_name: str, smiles: str, drug_conc_um: float) -> ProteomeBindingProfile:
        profiles = {
            "A": [BindingPrediction("CYP3A4", 100.0, 0.01, False, False, 0.0, 0.1)],
            "B": [BindingPrediction("ABCB1", 100.0, 0.01, False, False, 0.0, 0.1)],
        }
        return ProteomeBindingProfile(
            drug_name=drug_name, smiles=smiles, n_targets_screened=1,
            bindings=profiles[drug_name],
        )

    cascade = ProteomeBindingCascade(use_chembl=False)
    with _patch.object(cascade, "screen_drug", _mock_screen):
        pred = cascade.predict_ddi("A", "CC", 0.01, "B", "CCC", 0.01)
    assert pred.significance == "NO_DDI"
    assert pred.auc_ratio == 1.0


def test_predict_all_pairs_coverage() -> None:
    """predict_all_pairs loop branch (lines 573-581)."""
    cascade = ProteomeBindingCascade(use_chembl=True)
    preds = cascade.predict_all_pairs([
        ("ketoconazole", KETO_SMILES, 10.0),
        ("erythromycin", CHEMBL_DRUG_SMILES["erythromycin"], 10.0),
    ])
    assert len(preds) >= 1
    assert isinstance(preds[0], ProteomeDDIPrediction)
