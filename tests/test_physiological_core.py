"""Tests for the Phase B physiological core (doc/42 RL-1/2/4/5)."""
from __future__ import annotations

from helixlang.plugins.human.physiological_core import (
    GasExchangeModel,
    HemodynamicModel,
    HemodynamicState,
    PhysiologicalCoupler,
    PhysiologicalVitalsDriver,
    ThermoregulationModel,
    alveolar_arterial_oxygen,
    arterial_oxygen_saturation,
    hco3_from_ph_paco2,
    henderson_hasselbalch_pH,
    po2_from_saturation,
    severinghaus_saturation,
)


class TestBloodGasChemistry:
    def test_severinghaus_reference_points(self):
        # Classic Severinghaus anchor points (P50 ~27 mmHg -> ~50% saturation)
        assert 96.0 <= severinghaus_saturation(90.0) <= 98.0
        assert 73.0 <= severinghaus_saturation(40.0) <= 77.0
        assert 45.0 <= severinghaus_saturation(27.0) <= 55.0
        assert severinghaus_saturation(20.0) < severinghaus_saturation(60.0)

    def test_inverse_roundtrip(self):
        for p in (30.0, 60.0, 100.0):
            sat = severinghaus_saturation(p)
            assert abs(po2_from_saturation(sat) - p) < 1.0

    def test_hh_normal_ph(self):
        assert abs(henderson_hasselbalch_pH(24.0, 40.0) - 7.40) < 0.02

    def test_hh_acidosis(self):
        # metabolic acidosis (low HCO3) lowers pH
        assert henderson_hasselbalch_pH(14.0, 40.0) < 7.25

    def test_hco3_roundtrip(self):
        assert abs(hco3_from_ph_paco2(7.4, 40.0) - 24.0) < 1.0

    def test_alveolar_gas_equation_normal(self):
        # FiO2 0.21, PB 760, PaCO2 40 -> ideal PAO2 ~100
        pao2 = alveolar_arterial_oxygen(0.21, 760.0, 40.0, a_do2_mmhg=0.0)
        assert 98.0 <= pao2 <= 102.0

    def test_arterial_saturation_normal(self):
        assert arterial_oxygen_saturation(0.21, 760.0, 40.0) >= 96.0


class TestHemodynamicCore:
    def test_defaults(self):
        hem = HemodynamicModel()
        assert hem.state.map_mmhg > 80.0
        assert 4.0 < hem.state.cardiac_output_l_min < 8.0
        assert hem.state.systolic_bp_mmhg > hem.state.diastolic_bp_mmhg

    def test_steady_state_matches_setpoint(self):
        hem = HemodynamicModel()
        for _ in range(240):
            hem.step(1.0)
        assert abs(hem.state.map_mmhg - hem.set_point_map_mmhg) < 10.0
        assert 60.0 <= hem.state.systolic_bp_mmhg <= 200.0
        assert 30.0 <= hem.state.diastolic_bp_mmhg <= 130.0

    def test_map_is_co_times_svr(self):
        hem = HemodynamicModel()
        hem.step(1.0)
        expected = hem.state.cardiac_output_l_min * hem.state.svr_dyne / 80.0
        assert abs(hem.state.map_mmhg - expected) < 5.0

    def test_reduced_contractility_lowers_stroke_volume(self):
        # baroreflex compensates MAP, so reduced contractility shows up mainly
        # as reduced stroke volume (HF) rather than a pressure drop
        failing = HemodynamicModel()
        for _ in range(240):
            failing.step(1.0, disease_contractility=0.5)
        assert failing.state.contractility < 0.7
        assert failing.state.stroke_volume_ml < 60.0

    def test_positive_inotrope_raises_co(self):
        base = HemodynamicModel()
        ino = HemodynamicModel()
        for _ in range(240):
            base.step(1.0)
            ino.step(1.0, drug_inotropy=1.5)
        assert ino.state.cardiac_output_l_min > base.state.cardiac_output_l_min

    def test_vasodilation_lowers_svr(self):
        hem = HemodynamicModel()
        for _ in range(48):
            hem.step(1.0, drug_svr_mod=0.6)
        assert hem.state.svr_dyne < 900.0

    def test_volume_expansion_raises_co(self):
        # diuretic reduces effective volume -> lower CO on average
        normal = HemodynamicModel()
        diuretic = HemodynamicModel()
        for _ in range(240):
            normal.step(1.0)
            diuretic.step(1.0, volume_mod=0.5)
        assert diuretic.state.stroke_volume_ml < normal.state.stroke_volume_ml

    def test_create_from_physiology(self):
        class _P:
            body_weight_kg = 70.0
            age_years = 30.0
            body_surface_area_m2 = 1.85
        hem = HemodynamicModel.create_from_physiology(_P())
        assert isinstance(hem.state, HemodynamicState)


class TestGasExchange:
    def test_normal_gas_exchange(self):
        gx = GasExchangeModel()
        gx.step(1.0)
        assert gx.state.sao2_pct >= 96.0
        assert 14.0 <= gx.state.respiratory_rate_per_min <= 18.0
        assert 35.0 <= gx.state.paco2_mmhg <= 45.0

    def test_hypoventilation_lowers_spo2(self):
        gx = GasExchangeModel()
        gx.step(1.0, metabolic_ventilation_mod=0.4, respiratory_depression=0.3)
        assert gx.state.sao2_pct < 90.0
        assert gx.state.paco2_mmhg > 50.0

    def test_hyperventilation_raises_spo2_and_lowers_paco2(self):
        gx = GasExchangeModel()
        gx.step(1.0, metabolic_ventilation_mod=2.0)
        assert gx.state.sao2_pct > 96.0
        assert gx.state.paco2_mmhg < 35.0

    def test_acidosis_drives_respiration(self):
        gx = GasExchangeModel()
        gx.step(1.0, hco3_meq_per_l=14.0)  # metabolic acidosis
        assert gx.state.respiratory_rate_per_min > 20.0


class TestThermoregulation:
    def test_basal_holds_near_37(self):
        th = ThermoregulationModel()
        for _ in range(240):
            th.step(1.0, crp_mg_per_l=0.0, circadian_hour=12.0)
        assert 36.5 <= th.state.core_temperature_c <= 37.5

    def test_fever_raises_core_through_setpoint(self):
        basal = ThermoregulationModel()
        febrile = ThermoregulationModel()
        for _ in range(240):
            basal.step(1.0, crp_mg_per_l=0.0, circadian_hour=12.0)
            febrile.step(1.0, crp_mg_per_l=80.0, circadian_hour=12.0)
        assert febrile.state.set_point_c > basal.state.set_point_c
        assert febrile.state.core_temperature_c > basal.state.core_temperature_c
        assert febrile.state.core_temperature_c > 37.3

    def test_temp_stays_in_physiologic_range(self):
        th = ThermoregulationModel()
        for _ in range(1000):
            th.step(1.0, crp_mg_per_l=150.0, ambient_temperature_c=40.0)
            assert 34.0 <= th.state.core_temperature_c <= 42.0

    def test_circadian_forces_diurnal_setpoint(self):
        th = ThermoregulationModel()
        th.step(1.0, circadian_hour=1.0)
        low_hour = th.state.set_point_c
        th2 = ThermoregulationModel()
        th2.step(1.0, circadian_hour=17.0)
        assert th2.state.set_point_c > low_hour


class TestPhysiologicalCoupler:
    def test_cardiac_output_fraction(self):
        assert PhysiologicalCoupler.cardiac_output_fraction(1.0, 5.0, 0.2) == 0.2
        assert PhysiologicalCoupler.cardiac_output_fraction(0.0, 0.0, 0.2) == 0.2

    def test_renal_clearance_scales_with_egfr(self):
        low = PhysiologicalCoupler.renal_clearance_from_egfr(5.0, 40.0)
        normal = PhysiologicalCoupler.renal_clearance_from_egfr(5.0, 100.0)
        assert low < normal
        assert low > 0.0

    def test_hepatic_clearance_scales_with_function(self):
        low = PhysiologicalCoupler.hepatic_clearance_from_function(5.0, 0.2)
        normal = PhysiologicalCoupler.hepatic_clearance_from_function(5.0, 1.0)
        assert low < normal


class TestPhysiologicalVitalsDriver:
    """Composed CV + gas + thermo vitals driver (RL-1/2/4 + RL-5 wiring)."""

    def test_healthy_vitals_within_normal_range(self):
        d = PhysiologicalVitalsDriver()
        for _ in range(48):
            d.step(1.0)
        vs = d.step(1.0)
        assert 85.0 <= vs.map_mmhg <= 115.0
        assert 60.0 <= vs.systolic_bp_mmhg <= 140.0
        assert 50.0 <= vs.heart_rate_bpm <= 110.0
        assert 90.0 <= vs.spo2_pct <= 100.0
        assert 36.0 <= vs.temperature_c <= 38.0
        assert 12.0 <= vs.respiratory_rate_per_min <= 20.0

    def test_deterministic(self):
        a = PhysiologicalVitalsDriver()
        b = PhysiologicalVitalsDriver()
        assert a.step(1.0) == b.step(1.0)

    def test_fever_raises_temperature_with_crp(self):
        d = PhysiologicalVitalsDriver()
        for _ in range(24):
            d.step(1.0, crp_mg_per_l=80.0)
        assert d.thermo.state.core_temperature_c > 37.6

    def test_opioid_depresses_ventilation_and_lowers_spo2(self):
        control = PhysiologicalVitalsDriver()
        for _ in range(4):
            control.step(1.0)
        baseline_o2 = control.gas.state.sao2_pct

        depressed = PhysiologicalVitalsDriver()
        for _ in range(4):
            depressed.step(1.0, drug_concs={"morphine": 60.0})
        assert depressed.gas.state.sao2_pct < baseline_o2

    def test_diuretic_lowers_blood_pressure(self):
        a = PhysiologicalVitalsDriver()
        b = PhysiologicalVitalsDriver()
        for _ in range(48):
            a.step(1.0)
            b.step(1.0, drug_concs={"furosemide": 60.0})
        assert b.step(1.0).systolic_bp_mmhg < a.step(1.0).systolic_bp_mmhg

    def test_reduced_contractility_lowers_cardiac_output(self):
        a = PhysiologicalVitalsDriver()
        b = PhysiologicalVitalsDriver()
        for _ in range(24):
            a.step(1.0)
            b.step(1.0, disease_contractility=0.5)
        assert b.step(1.0).cardiac_output_l_min < a.step(1.0).cardiac_output_l_min
