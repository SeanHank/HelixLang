"""Organ-organ crosstalk map (doc/30 §9).

Bidirectional coupling between organ systems that creates emergent
multi-organ dynamics not captured by isolated models:

1. **Glucose → Cardiovascular**: hyperglycemia drives endothelial
   dysfunction, increases CV risk (UKPDS, DCCT/EDIC)

2. **EPO → Hematology**: renal EPO production → erythropoiesis;
   renal failure → anemia (EPO deficit)

3. **Cortisol → Inflammation**: HPA axis cortisol suppresses innate
   immune response (cortisol_suppression in immune model)

4. **Child-Pugh → Clearance**: hepatic synthetic function (albumin,
   INR) modulates drug clearance (Well-Stirred model extension)

5. **Inflammation → Liver**: IL-6/TNF-α drive acute-phase proteins
   (CRP, ↓albumin, ↑ferritin)

6. **Phosphate → Hematology**: CKD hyperphosphatemia → ↓EPO response
   → worsened anemia

Module structure:
    OrganCrosstalk        coupled crosstalk matrix
    apply_crosstalk       one-step crosstalk injection
    create_crosstalk      factory

References:
- UKPDS 35, Lancet 1998 (glucose-CV)
- Kidney Disease Outcomes Quality Initiative (anemia in CKD)
- Child & Turcotte, Surgery 1964 (hepatic clearance classification)
"""
from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "OrganCrosstalk",
    "apply_crosstalk",
    "create_crosstalk",
]


@dataclass
class OrganCrosstalk:
    """Bidirectional organ coupling signals.

    Each field represents a coupling strength (0-1) or a driving signal
    that modulates downstream organ behavior.
    """

    # --- Glucose → CV ---
    glucose_cv_damage: float = 0.0  # 0=none, 1=severe endothelial dysfunction
    hyperglycemia_threshold: float = 140.0  # mg/dL

    # --- Renal → Hematology (EPO) ---
    renal_function_fraction: float = 1.0  # 0=anephric, 1=normal
    epo_production: float = 1.0          # relative EPO output

    # --- HPA → Immune ---
    cortisol_level: float = 12.0  # µg/dL
    cortisol_immune_threshold: float = 20.0

    # --- Hepatic → Clearance ---
    child_pugh_score: float = 5.0  # 5=A (normal), 15=C (severe)
    hepatic_clearance_fraction: float = 1.0

    # --- Inflammation → Liver ---
    il6_level: float = 1.0  # pg/mL
    tnf_level: float = 5.0  # pg/mL
    acute_phase_response: float = 0.0  # 0=none, 1=full

    # --- Phosphate → Hematology ---
    phosphate_mg_dl: float = 3.5
    phosphate_epo_suppression: float = 0.0  # computed

    # --- Output signals (updated by apply_crosstalk) ---
    cv_risk_multiplier: float = 1.0
    anemia_risk_multiplier: float = 1.0
    immune_suppression_from_cortisol: float = 0.0
    clearance_modifier_from_liver: float = 1.0


def apply_crosstalk(
    crosstalk: OrganCrosstalk,
    glucose_mg_dl: float = 100.0,
    egfr: float = 90.0,
    cortisol_ug_dl: float = 12.0,
    albumin_g_dl: float = 4.0,
    inr: float = 1.0,
    il6_pg_ml: float = 1.0,
    tnf_pg_ml: float = 5.0,
    phosphate_mg_dl: float = 3.5,
) -> OrganCrosstalk:
    """Compute crosstalk signals from current organ states.

    Returns the updated crosstalk with output signals computed.
    """
    # --- Glucose → CV damage ---
    if glucose_mg_dl > crosstalk.hyperglycemia_threshold:
        excess = glucose_mg_dl - crosstalk.hyperglycemia_threshold
        crosstalk.glucose_cv_damage = min(1.0, excess / 160.0)
    else:
        crosstalk.glucose_cv_damage = 0.0
    crosstalk.cv_risk_multiplier = 1.0 + 0.5 * crosstalk.glucose_cv_damage

    # --- Renal → EPO → Hematology ---
    crosstalk.renal_function_fraction = max(0.0, egfr / 120.0)
    # EPO production drops nonlinearly below eGFR 60
    if egfr < 60:
        crosstalk.epo_production = max(0.1, (egfr / 60.0) ** 0.7)
    else:
        crosstalk.epo_production = 1.0

    # Phosphate suppresses EPO response in CKD
    crosstalk.phosphate_mg_dl = phosphate_mg_dl
    if phosphate_mg_dl > 4.5:
        crosstalk.phosphate_epo_suppression = min(
            0.5, (phosphate_mg_dl - 4.5) / 5.0)
    else:
        crosstalk.phosphate_epo_suppression = 0.0

    effective_epo = crosstalk.epo_production * (
        1.0 - crosstalk.phosphate_epo_suppression)
    crosstalk.anemia_risk_multiplier = 1.0 / max(0.2, effective_epo)

    # --- HPA → Immune suppression ---
    crosstalk.cortisol_level = cortisol_ug_dl
    if cortisol_ug_dl > crosstalk.cortisol_immune_threshold:
        crosstalk.immune_suppression_from_cortisol = min(
            1.0, (cortisol_ug_dl - crosstalk.cortisol_immune_threshold) / 30.0)
    else:
        crosstalk.immune_suppression_from_cortisol = 0.0

    # --- Child-Pugh → Clearance ---
    crosstalk.child_pugh_score = max(5.0, min(15.0, inr * 3.0 + (4.0 - albumin_g_dl) * 2.0 + 5.0))
    if crosstalk.child_pugh_score <= 7:
        crosstalk.hepatic_clearance_fraction = 1.0
    elif crosstalk.child_pugh_score <= 9:
        crosstalk.hepatic_clearance_fraction = 0.7
    elif crosstalk.child_pugh_score <= 12:
        crosstalk.hepatic_clearance_fraction = 0.4
    else:
        crosstalk.hepatic_clearance_fraction = 0.2
    crosstalk.clearance_modifier_from_liver = crosstalk.hepatic_clearance_fraction

    # --- Inflammation → Liver acute-phase ---
    crosstalk.il6_level = il6_pg_ml
    crosstalk.tnf_level = tnf_pg_ml
    inflammatory_stimulus = (il6_pg_ml - 1.0) / 10.0 + (tnf_pg_ml - 5.0) / 50.0
    crosstalk.acute_phase_response = min(1.0, max(0.0, inflammatory_stimulus))

    return crosstalk


def create_crosstalk() -> OrganCrosstalk:
    """Factory with healthy baseline."""
    return OrganCrosstalk()
