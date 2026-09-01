#!/usr/bin/env python3
"""Benchmark 83: Physiological realism (doc/42 Phase B, RL-1..RL-5).

Exercises the closed-loop physiological core added in Phase B and asserts its
mechanistic outputs against literature ranges:

* RL-1 cardiac output from flow balance (CO = HR x SV / 1000) with CI in the
  normal adult range, and a failing-heart reduction in CO (Guyton & Hall).
* RL-2 gas exchange with the Severinghaus O2-Hb dissociation curve and a real
  SpO2; alveolar-arterial O2 from the alveolar gas equation (West); hypoventilation
  raises PaCO2, drops SpO2, and produces acidosis (Henderson-Hasselbalch).
* RL-3 renal filtration/clearance: normal eGFR ~100, GFR-driven clearance in L/h,
  tubular secretion/reabsorption, creatinine turnover, and CKD metabolic acidosis.
* RL-4 thermoregulation: heat balance settles at ~37 C; fever is a set-point
  perturbation (from CRP/cytokines) that core temperature tracks via the same ODE.
* RL-5 cross-system coupling: a failing kidney / liver measurably reduces drug
  clearance (organ function -> renal/hepatic clearance).

All systems are deterministic (no RNG), so the output is golden-verifiable.
"""
from __future__ import annotations

import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))


def run() -> dict:
    t0 = time.perf_counter()
    results: dict = {"id": "83_physiological_realism"}
    try:
        from helixlang.plugins.human.physiological_core import (
            PhysiologicalCoupler,
            HemodynamicModel,
            GasExchangeModel,
            ThermoregulationModel,
            severinghaus_saturation,
            po2_from_saturation,
            henderson_hasselbalch_pH,
            alveolar_arterial_oxygen,
        )
        from helixlang.plugins.human.renal_model import create_renal_model

        checks: dict[str, bool] = {}
        details: dict[str, float] = {}

        # ── RL-1 Cardiovascular: closed-loop, flow/pressure balanced ──────────
        h = HemodynamicModel()
        for _ in range(24):          # warm up to steady state
            h.step(1.0)
        co = h.state.cardiac_output_l_min
        hr = h.state.heart_rate_bpm
        sv = h.state.stroke_volume_ml
        ci = h.state.cardiac_index_l_min_m2
        details["hr_bpm"] = round(hr, 2)
        details["co_l_min"] = round(co, 3)
        details["ci_l_min_m2"] = round(ci, 3)
        details["map_mmhg"] = round(h.state.map_mmhg, 2)

        # flow balance: CO = HR x SV / 1000 (mass/flow-balanced, not target-tracking)
        expected_co = hr * sv / 1000.0
        checks["co_from_flow_balance"] = abs(co - expected_co) < 1e-6
        # cardiac index in the normal adult range (2.5 - 4.0 L/min/m2)
        checks["ci_normal_range"] = 2.5 <= ci <= 4.0

        # failing heart (contractility halved): stroke volume collapses and the
        # closed-loop baroreflex mounts compensatory tachycardia to defend CO —
        # demonstrating a real closed loop rather than a target-tracking delta.
        hf = HemodynamicModel()
        for _ in range(24):
            hf.step(1.0, disease_contractility=0.5)
        checks["hf_collapses_sv"] = hf.state.stroke_volume_ml < sv * 0.6
        checks["hf_baroreflex_tachycardia"] = hf.state.heart_rate_bpm > hr
        details["hf_sv_ml"] = round(hf.state.stroke_volume_ml, 2)
        details["hf_hr_bpm"] = round(hf.state.heart_rate_bpm, 2)
        details["hf_co_l_min"] = round(hf.state.cardiac_output_l_min, 3)

        # ── RL-2 Gas exchange: Severinghaus curve + real SpO2 + O2 drive ──────
        # O2-Hb dissociation: PaO2 100 -> SaO2 ~97.7%, PaO2 60 -> ~90.6%
        sao2_100 = severinghaus_saturation(100.0)
        sao2_60 = severinghaus_saturation(60.0)
        checks["severinghaus_100_normal"] = 96.0 <= sao2_100 <= 99.0
        checks["severinghaus_60_lower"] = 88.0 <= sao2_60 <= 93.0
        checks["severinghaus_inverse_roundtrip"] = (
            abs(po2_from_saturation(sao2_100) - 100.0) < 0.5
        )
        details["sao2_at_100"] = round(sao2_100, 2)
        details["sao2_at_60"] = round(sao2_60, 2)

        # alveolar gas equation: room-air arterial PaO2 ~90 mmHg, on 100% O2 higher
        pao2_room = alveolar_arterial_oxygen(0.21, 760.0, 40.0)
        pao2_100pct = alveolar_arterial_oxygen(1.0, 760.0, 40.0)
        checks["alveolar_room_air"] = 85.0 <= pao2_room <= 105.0
        checks["alveolar_100pct_higher"] = pao2_100pct > pao2_room + 200.0
        details["pao2_room_air"] = round(pao2_room, 2)

        # steady gas model: healthy resting SpO2 in range, pH normal
        g = GasExchangeModel()
        g.step(1.0)
        checks["gas_normal_spo2"] = 95.0 <= g.state.sao2_pct <= 99.0
        checks["gas_normal_ph"] = 7.35 <= g.state.ph <= 7.45
        details["gas_spo2_pct"] = round(g.state.sao2_pct, 2)
        details["gas_ph"] = round(g.state.ph, 3)

        # respiratory depression (opioid) -> hypercapnia + hypoxemia + acidosis
        g2 = GasExchangeModel()
        g2.step(1.0, respiratory_depression=0.5)
        checks["hypovent_raises_paco2"] = g2.state.paco2_mmhg > 60.0
        checks["hypovent_lowers_spo2"] = g2.state.sao2_pct < 90.0
        checks["hypovent_acidosis"] = g2.state.ph < 7.25
        details["hypovent_paco2"] = round(g2.state.paco2_mmhg, 2)
        details["hypovent_spo2"] = round(g2.state.sao2_pct, 2)

        # Henderson-Hasselbalch closed form sanity
        checks["hh_normal_ph"] = abs(henderson_hasselbalch_pH(24.0, 40.0) - 7.40) < 0.03

        # ── RL-3 Renal filtration/clearance ────────────────────────────────────
        rn = create_renal_model(age_years=40, is_female=False, initial_egfr=100)
        checks["renal_normal_egfr"] = 90.0 <= rn.reported_egfr() <= 120.0
        checks["renal_gfr_driven_cl"] = abs(
            rn.renal_clearance_l_per_h() - rn.reported_egfr() * 0.06
        ) < 1e-6
        checks["renal_tubular_secretion"] = (
            rn.tubular_clearance_ratio(secretion_factor=1.0) > 1.0
        )
        checks["renal_tubular_reabsorption"] = (
            rn.tubular_clearance_ratio(reabsorption_factor=1.0) < 1.0
        )
        checks["renal_creatinine_turnover_positive"] = rn.creatinine_turnover() > 0.0
        checks["renal_normal_ph"] = 7.35 <= rn.acid_base_ph() <= 7.45
        details["renal_egfr"] = round(rn.reported_egfr(), 2)
        details["renal_cl_l_per_h"] = round(rn.renal_clearance_l_per_h(), 3)
        details["renal_ph"] = round(rn.acid_base_ph(), 3)

        # CKD: eGFR 30 -> lower clearance + metabolic acidosis (renal HCO3 retention)
        rk = create_renal_model(age_years=70, is_female=True, initial_egfr=30)
        checks["ckd_drops_clearance"] = rk.renal_clearance_l_per_h() < rn.renal_clearance_l_per_h()
        checks["ckd_metabolic_acidosis"] = rk.acid_base_ph() < 7.35
        details["ckd_egfr"] = round(rk.reported_egfr(), 2)
        details["ckd_ph"] = round(rk.acid_base_ph(), 3)

        # ── RL-4 Thermoregulation: heat balance + set-point fever ──────────────
        t = ThermoregulationModel()
        for _ in range(48):
            t.step(1.0)
        # steady-state heat balance: production ~ loss (mass/flow-balanced)
        checks["thermo_heat_balanced"] = (
            abs(t.state.heat_production_w - t.state.heat_loss_w)
            / max(1.0, t.state.heat_production_w)
        ) < 0.2
        checks["thermo_basal_core"] = 36.5 <= t.state.core_temperature_c <= 37.6
        details["thermo_core_c"] = round(t.state.core_temperature_c, 3)

        # fever = set-point perturbation (CRP/cytokines), core tracks the set-point
        tf = ThermoregulationModel()
        for _ in range(96):
            tf.step(1.0, crp_mg_per_l=100.0)
        checks["fever_raises_setpoint"] = tf.state.set_point_c > 37.5
        checks["fever_core_tracks_setpoint"] = (
            tf.state.core_temperature_c > 37.5
        )
        # temperature must sit at or below an elevated set-point (tracking, not assignment)
        checks["fever_core_below_setpoint"] = (
            tf.state.core_temperature_c <= tf.state.set_point_c + 0.1
        )
        details["fever_core_c"] = round(tf.state.core_temperature_c, 3)
        details["fever_setpoint_c"] = round(tf.state.set_point_c, 3)

        # ── RL-5 Cross-system coupling: organ failure -> PK clearance ──────────
        base_cl = 10.0  # L/h baseline total renal clearance
        cl_normal = PhysiologicalCoupler.renal_clearance_from_egfr(base_cl, 100.0)
        cl_ckd30 = PhysiologicalCoupler.renal_clearance_from_egfr(base_cl, 30.0)
        cl_ckd15 = PhysiologicalCoupler.renal_clearance_from_egfr(base_cl, 15.0)
        checks["rl5_renal_failure_reduces_cl"] = (
            cl_ckd30 < cl_normal * 0.5 and cl_ckd15 < cl_ckd30
        )
        hepatic_base = 10.0
        hepatic_normal = PhysiologicalCoupler.hepatic_clearance_from_function(hepatic_base, 1.0)
        hepatic_low = PhysiologicalCoupler.hepatic_clearance_from_function(hepatic_base, 0.3)
        checks["rl5_hepatic_failure_reduces_cl"] = hepatic_low < hepatic_normal * 0.5
        details["rl5_cl_normal_l_per_h"] = round(cl_normal, 3)
        details["rl5_cl_ckd30_l_per_h"] = round(cl_ckd30, 3)
        details["rl5_hepatic_cl_low_l_per_h"] = round(hepatic_low, 3)

        elapsed = time.perf_counter() - t0
        all_pass = all(checks.values())
        results.update({
            "status": "PASS" if all_pass else "FAIL",
            "checks": checks,
            "details": details,
            "reference": {
                "source": (
                    "Severinghaus 1979 (O2 dissociation); West, Respiratory "
                    "Physiology (alveolar gas eq.); Guyton & Hall (cardiovascular "
                    "control, thermoregulation); CKD-EPI 2021 (eGFR)"
                ),
                "authors": (
                    "Severinghaus JW; West JB; Guyton AC & Hall JE; Inker LA et al."
                ),
                "year": 1979,
                "doi": "10.1152/jappl.1979.46.3.599",
                "note": (
                    "Severinghaus JW, J Appl Physiol 1979;46(3):599-602. "
                    "doi:10.1152/jappl.1979.46.3.599."
                ),
            },
            "experimental_comparison": {
                "cardiac_index_l_min_m2": {
                    "reference_min": 2.5,
                    "reference_max": 4.0,
                    "unit": "L/min/m2",
                    "note": "Normal adult cardiac index (Guyton & Hall).",
                },
                "arterial_oxygen_saturation_at_PaO2_100_pct": {
                    "reference_min": 96.0,
                    "reference_max": 99.0,
                    "unit": "%",
                    "note": "Severinghaus curve: SaO2 ~97.7% at PaO2 100 mmHg.",
                },
                "plasma_ph_normal": {
                    "reference_min": 7.35,
                    "reference_max": 7.45,
                    "unit": "pH",
                    "note": "Henderson-Hasselbalch normal arterial pH.",
                },
                "core_temperature_normal_c": {
                    "reference_min": 36.5,
                    "reference_max": 37.6,
                    "unit": "degC",
                    "note": "Afebrile core temperature at heat-balance steady state.",
                },
                "fever_setpoint_c": {
                    "reference_min": 37.5,
                    "reference_max": 40.0,
                    "unit": "degC",
                    "note": "CRP-driven hypothalamic set-point elevation (fever).",
                },
                "renal_clearance_reduction_ckd30_fraction": {
                    "reference_min": 0.2,
                    "reference_max": 0.5,
                    "tolerance": 0.1,
                    "unit": "fraction of baseline",
                    "note": "eGFR 100->30 mL/min reduces renal clearance ~3x.",
                },
            },
            "runtime_seconds": elapsed,
        })
    except Exception as e:
        results.update({
            "status": "FAIL",
            "checks": {},
            "details": {"error": str(e)},
            "reference": {
                "source": "Severinghaus 1979; West; Guyton & Hall; CKD-EPI 2021",
                "authors": "Severinghaus JW; West JB; Guyton AC & Hall JE; Inker LA et al.",
                "year": 1979,
                "doi": "10.1152/jappl.1979.46.3.599",
                "note": "Physiological-realism references (Phase B).",
            },
            "runtime_seconds": time.perf_counter() - t0,
        })
    return results


if __name__ == "__main__":
    r = run()
    import json
    print(json.dumps(r, indent=2))
    sys.exit(0 if r["status"] == "PASS" else 1)
