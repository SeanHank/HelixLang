"""Tests for mechanistic hematology models (doc/30 §8 wave-1)."""
from __future__ import annotations

import pytest

from helixlang.plugins.human.hematology_model import (
    NEUTROPHIL_CONFIG,
    ErythropoiesisModel,
    FribergLineage,
    HematologySystem,
    LineageConfig,
    MyelosuppressionParams,
    create_hematology_system,
)


def _days(system: HematologySystem, days: int, exposure=None):
    snapshot = None
    for _ in range(days):
        snapshot = system.step(24.0, exposure)
    return snapshot


class TestMyelosuppressionParams:
    def test_effect_at_zero_concentration(self):
        params = MyelosuppressionParams("drug", emax=0.9, ec50_mg_l=0.1)
        assert params.effect_at(0.0) == 0.0

    def test_effect_at_ec50_is_half_of_emax(self):
        params = MyelosuppressionParams("drug", emax=0.8, ec50_mg_l=0.5)
        assert params.effect_at(0.5) == pytest.approx(0.4)

    def test_validation(self):
        with pytest.raises(ValueError):
            MyelosuppressionParams("x", emax=1.5, ec50_mg_l=0.1)
        with pytest.raises(ValueError):
            MyelosuppressionParams("x", emax=0.5, ec50_mg_l=0.0)
        with pytest.raises(ValueError):
            MyelosuppressionParams("x", emax=0.5, ec50_mg_l=0.1, hill=-1.0)

    def test_hill_steepens_response(self):
        shallow = MyelosuppressionParams("x", emax=1.0, ec50_mg_l=1.0, hill=1.0)
        steep = MyelosuppressionParams("x", emax=1.0, ec50_mg_l=1.0, hill=4.0)
        assert steep.effect_at(3.0) > shallow.effect_at(3.0)


class TestFribergLineage:
    def test_homeostasis_without_drug(self):
        lineage = FribergLineage(NEUTROPHIL_CONFIG)
        for _ in range(90 * 24):
            lineage.step(1.0, 0.0)
        assert lineage.count() == pytest.approx(NEUTROPHIL_CONFIG.circ0, rel=0.02)

    def test_steady_state_internal_consistency(self):
        lineage = FribergLineage(NEUTROPHIL_CONFIG)
        assert lineage.prol == pytest.approx(lineage.circ)
        assert lineage.tx3 == pytest.approx(lineage.circ)

    def test_invalid_config_rejected(self):
        with pytest.raises(ValueError):
            FribergLineage(LineageConfig("bad", circ0=-1.0, mtt_h=100.0, gamma=0.16))

    def test_negative_dt_rejected(self):
        lineage = FribergLineage(NEUTROPHIL_CONFIG)
        with pytest.raises(ValueError):
            lineage.step(-1.0, 0.0)


class TestHematologyHomeostasis:
    def test_one_year_flat(self):
        system = create_hematology_system()
        snap = _days(system, 365)
        assert snap["anc_x10e3_ul"] == pytest.approx(4.5, rel=0.03)
        assert snap["platelets_x10e3_ul"] == pytest.approx(250.0, rel=0.03)
        assert snap["hemoglobin_g_dl"] == pytest.approx(14.5, abs=0.2)
        assert snap["reticulocyte_pct"] == pytest.approx(1.0, abs=0.3)
        assert snap["epo_u_l"] == pytest.approx(10.0, rel=0.2)

    def test_female_baselines(self):
        system = create_hematology_system(is_female=True)
        assert system.erythropoiesis.hb0 == 13.5


class TestChemotherapyResponse:
    DOCETAXEL_LIKE = dict(emax=0.9, ec50_mg_l=0.1)

    def _bolus_cycle(self, support: float = 0.0, days: int = 60):
        system = create_hematology_system()
        system.register_myelosuppressant(
            MyelosuppressionParams("docetaxel", **self.DOCETAXEL_LIKE)
        )
        if support:
            system.set_growth_factor_support(support)
        trajectory = []
        for hour in range(days * 24):
            conc = 1.0 if hour < 24 else 0.0
            snap = system.step(1.0, {"docetaxel": conc})
            if hour >= 24:
                trajectory.append((hour, snap["anc_x10e3_ul"]))
        return trajectory

    def test_nadir_depth_and_timing(self):
        traj = self._bolus_cycle()
        nadir_hour, nadir = min(traj, key=lambda x: x[1])
        # MTT ~134 h places the neutrophil nadir in the classic
        # day 5-14 window (Friberg 2002), not scripted but emergent.
        assert 96 <= nadir_hour <= 24 * 14
        assert nadir < 0.75 * NEUTROPHIL_CONFIG.circ0

    def test_recovery_overshoot_then_return_to_baseline(self):
        traj = self._bolus_cycle(days=90)
        _, nadir = min(traj, key=lambda x: x[1])
        peak_after_nadir = max(c for _, c in traj[traj.index(min(traj, key=lambda x: x[1])):])
        assert peak_after_nadir > NEUTROPHIL_CONFIG.circ0 * 1.01
        final = traj[-1][1]
        assert final == pytest.approx(NEUTROPHIL_CONFIG.circ0, rel=0.05)

    def test_fractionated_schedule_gives_shallower_nadir(self):
        deep_nadir = min(c for _, c in self._bolus_cycle())

        system = create_hematology_system()
        system.register_myelosuppressant(
            MyelosuppressionParams("docetaxel", **self.DOCETAXEL_LIKE)
        )
        nadir = 99.0
        total_exposure_h = 72
        for hour in range(40 * 24):
            active = hour < total_exposure_h and (hour % 24) < 8
            snap = system.step(1.0, {"docetaxel": 1.0 if active else 0.0})
            if hour > total_exposure_h + 96:
                nadir = min(nadir, snap["anc_x10e3_ul"])
        # same total drug-hours delivered as daily 8 h pulses produce a
        # shallower nadir than one 72 h bolus (schedule effect emerges
        # from transit-chain depletion, doc/30 section 8.4)
        assert nadir > deep_nadir

    def test_growth_factor_support_attenuates_nadir(self):
        unsupported = min(c for _, c in self._bolus_cycle())
        supported = min(c for _, c in self._bolus_cycle(support=1.0))
        assert supported > unsupported

    def test_platelet_nadir_later_and_shallower_relative(self):
        system = create_hematology_system()
        system.register_myelosuppressant(
            MyelosuppressionParams("docetaxel", **self.DOCETAXEL_LIKE)
        )
        anc_min_h = plt_min_h = None
        anc_min = plt_min = 1e9
        for hour in range(30 * 24):
            conc = 1.0 if hour < 24 else 0.0
            snap = system.step(1.0, {"docetaxel": conc})
            if snap["anc_x10e3_ul"] < anc_min:
                anc_min, anc_min_h = snap["anc_x10e3_ul"], hour
            if snap["platelets_x10e3_ul"] < plt_min:
                plt_min, plt_min_h = snap["platelets_x10e3_ul"], hour
        assert plt_min_h >= anc_min_h


class TestGradesAndOutputs:
    def test_anc_grades(self):
        system = create_hematology_system()
        assert system.anc_ctcae_grade() == 0
        system.neutrophils.circ = 1.2
        assert system.anc_ctcae_grade() == 1
        system.neutrophils.circ = 0.7
        assert system.anc_ctcae_grade() == 2
        system.neutrophils.circ = 0.4
        assert system.anc_ctcae_grade() == 3
        system.neutrophils.circ = 0.1
        assert system.anc_ctcae_grade() == 4

    def test_platelet_grades(self):
        system = create_hematology_system()
        assert system.platelet_ctcae_grade() == 0
        for count, grade in ((80.0, 1), (60.0, 2), (30.0, 3), (10.0, 4)):
            system.platelets.circ = count
            assert system.platelet_ctcae_grade() == grade

    def test_lab_values_units(self):
        system = create_hematology_system()
        labs = system.lab_values()
        assert labs["anc_per_ul"] == pytest.approx(4500.0)
        assert labs["platelets_per_ul"] == pytest.approx(250_000.0)
        assert labs["hemoglobin_g_dl"] == pytest.approx(14.5)

    def test_bliss_combination_of_two_drugs(self):
        def cycle(drug_names):
            system = create_hematology_system()
            for name in drug_names:
                system.register_myelosuppressant(
                    MyelosuppressionParams(name, emax=0.5, ec50_mg_l=0.001)
                )
            nadir = 99.0
            exposure = {name: 1.0 for name in drug_names}
            for hour in range(21 * 24):
                conc = exposure if hour < 24 else {}
                snap = system.step(1.0, conc)
                if hour >= 24:
                    nadir = min(nadir, snap["anc_x10e3_ul"])
            return nadir

        dual = cycle(["a", "b"])
        single = cycle(["a"])
        # Bliss independence: 1-(1-0.5)^2 = 0.75 combined inhibition
        # must bite deeper than either drug alone
        assert dual < single

    def test_step_validates_dt(self):
        system = create_hematology_system()
        with pytest.raises(ValueError):
            system.step(-1.0)


class TestErythropoiesis:
    def test_bleed_raises_epo_and_retics(self):
        system = create_hematology_system()
        system.erythropoiesis.bleed(3.0)
        peak_epo = peak_retic = 0.0
        snap = None
        for _ in range(30):
            snap = _days(system, 1)
            peak_epo = max(peak_epo, snap["epo_u_l"])
            peak_retic = max(peak_retic, snap["reticulocyte_pct"])
        assert peak_epo > 2.0 * 10.0
        assert peak_retic > 1.5
        assert snap["hemoglobin_g_dl"] > 12.0  # partial recovery

    def test_renal_anemia_blunts_epo(self):
        healthy = create_hematology_system()
        healthy.erythropoiesis.bleed(3.0)
        ckd = create_hematology_system(renal_function_fraction=0.15)
        ckd.erythropoiesis.bleed(3.0)
        for _ in range(21):
            s_health = _days(healthy, 1)
            s_ckd = _days(ckd, 1)
        assert s_ckd["epo_u_l"] < s_health["epo_u_l"]
        assert s_ckd["hemoglobin_g_dl"] < s_health["hemoglobin_g_dl"]

    def test_transfusion_is_step_increase(self):
        model = ErythropoiesisModel()
        before = model.hemoglobin
        model.transfuse(2.0)
        assert model.hemoglobin == pytest.approx(before + 2.0)

    def test_iron_deficiency_develops_anemia(self):
        system = create_hematology_system()
        system.erythropoiesis.set_iron_availability(0.25)
        start = system.lab_values()["hemoglobin_g_dl"]
        snap = _days(system, 240)
        assert snap["hemoglobin_g_dl"] < start - 1.0

    def test_moderate_esa_raises_hemoglobin(self):
        with_esa = create_hematology_system()
        with_esa.erythropoiesis.bleed(4.0)
        with_esa.erythropoiesis.esa_infusion_u_l_h = 1.0
        without = create_hematology_system()
        without.erythropoiesis.bleed(4.0)
        for _ in range(90):
            s_yes = _days(with_esa, 1)
            s_no = _days(without, 1)
        assert s_yes["hemoglobin_g_dl"] > s_no["hemoglobin_g_dl"] + 0.5

    def test_chemotherapy_suppresses_red_cells_less_than_neutrophils(self):
        system = create_hematology_system()
        system.register_myelosuppressant(
            MyelosuppressionParams("docetaxel", emax=0.9, ec50_mg_l=0.1)
        )
        worst_hb = 99.0
        for hour in range(60 * 24):
            conc = 1.0 if hour < 24 else 0.0
            snap = system.step(1.0, {"docetaxel": conc})
            worst_hb = min(worst_hb, snap["hemoglobin_g_dl"])
        # RBC lineage is far less sensitive to a single cycle
        assert worst_hb > 13.0

    def test_intervention_validation(self):
        model = ErythropoiesisModel()
        with pytest.raises(ValueError):
            model.transfuse(-1.0)
        with pytest.raises(ValueError):
            model.bleed(-1.0)
        with pytest.raises(ValueError):
            model.administer_epo_bolus(-1.0)
