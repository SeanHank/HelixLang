"""Mechanistic hematology models: myelosuppression + erythropoiesis (doc/30 §8).

This module implements the two validated pillars recommended by doc/30
for the virtual-patient hematology layer:

1. **Friberg semimechanistic myelosuppression** (Friberg et al.,
   JCO 2002) for neutrophils and platelets.  Each lineage is five ODEs
   (proliferating pool -> three transit compartments -> circulation)
   driven by cytotoxic exposure through a shared Emax/EC50 convention.
   The design property that made this the field gold standard carries
   over directly: *system* parameters (baseline count Circ0, mean
   transit time MTT, feedback exponent gamma) are consistent across
   drugs and patients, and only Emax/EC50 are drug-specific.  Nadir
   depth/timing (day 7-14 for neutrophils) and the recovery overshoot
   emerge from the transit chain and the ``(Circ0/Circ)**gamma``
   feedback -- nothing is scripted.

2. **Minimal erythropoiesis with EPO feedback** -- a reduced cell-
   kinetic model (erythroid progenitors -> marrow reticulocytes ->
   circulating red cells -> plasma EPO) following the structure of the
   Fuertinger et al. Sci Rep 2020 and Dor & Alon minimal models.  Kidney
   hypoxia sensing couples to renal function (renal anemia emerges),
   with iron limitation, transfusion and ESA channels.

All drug concentrations enter as plasma values in mg/L (doc/27 PBPK
central compartment).  Multiple concurrent myelosuppressants combine
by Bliss independence ``1 - prod(1 - Ei)``.  Growth-factor stimulation
(G-CSF / TPO-RA) enters as *negative* inhibition on the same feedback
loop plus transit acceleration, mirroring published filgrastim-on-
Friberg extensions.

Module structure:
    MyelosuppressionParams  per-drug Emax/EC50 (+ optional Hill slope)
    LineageConfig           baseline count / MTT / gamma per lineage
    FribergLineage          one lineage (neutrophils OR platelets)
    ErythropoiesisModel     progenitor/reticulocyte/RBC/EPO system
    HematologySystem        facade stepping everything together
    create_hematology_system  factory (sex-aware baselines)

References:
- Friberg LE et al. J Clin Oncol 2002;20:4713 (model development on
  docetaxel/paclitaxel/etoposide; Circ0 ~ 4 x10^3/uL, MTT ~ 134 h,
  gamma ~ 0.16-0.23)
- Kloft C et al. Clin Cancer Res 2006 (application review)
- Hansson EK, Friberg LE, Cancer Chemother Pharmacol 2012
- Fuertinger DH et al. Sci Rep 2020 (erythropoiesis + iron;
  PMC7248076)
- Dor H, Alon U, PLoS Comput Biol 2026 (minimal RBC model validated on
  36 studies)
- CTCAE v5.0 grading (neutropenia, thrombocytopenia)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "MyelosuppressionParams",
    "LineageConfig",
    "NEUTROPHIL_CONFIG",
    "PLATELET_CONFIG",
    "FribergLineage",
    "ErythropoiesisModel",
    "HematologySystem",
    "create_hematology_system",
]


# ============================================================================
# Myelosuppression drug parameters
# ============================================================================

@dataclass(slots=True)
class MyelosuppressionParams:
    """Drug-specific myelosuppressive potency.

    Only ``emax`` and ``ec50_mg_l`` (plus optionally the Hill slope)
    differ between drugs; all system parameters live on the lineage.

    Attributes:
        drug_name: key used in the exposure dict passed to
            :meth:`HematologySystem.step`.
        emax: maximum fractional inhibition of progenitor
            proliferation in [0, 1].
        ec50_mg_l: plasma concentration producing half of ``emax``.
        hill: sigmoid steepness (1.0 = plain Emax).
    """

    drug_name: str
    emax: float
    ec50_mg_l: float
    hill: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.emax <= 1.0:
            raise ValueError("emax must be within [0, 1]")
        if self.ec50_mg_l <= 0.0:
            raise ValueError("ec50_mg_l must be > 0")
        if self.hill <= 0.0:
            raise ValueError("hill must be > 0")

    def effect_at(self, concentration_mg_l: float) -> float:
        """Return fractional proliferation inhibition at *concentration*."""
        if concentration_mg_l <= 0.0:
            return 0.0
        c_hill = concentration_mg_l ** self.hill
        ec50_hill = self.ec50_mg_l ** self.hill
        return self.emax * c_hill / (ec50_hill + c_hill)


# ============================================================================
# Friberg lineage
# ============================================================================

@dataclass(slots=True, frozen=True)
class LineageConfig:
    """Shared (drug-independent) parameters for one cell lineage.

    Attributes:
        name: ``"neutrophil"`` or ``"platelet"``.
        circ0: baseline circulating count (x10^3/uL).
        mtt_h: mean transit time through prol + 3 transit pools
            (hours); ``MTT ~= 4 / k_tr``.
        gamma: feedback exponent on ``(Circ0/Circ)**gamma``; published
            range 0.16-0.23.
    """

    name: str
    circ0: float
    mtt_h: float
    gamma: float


#: docetaxel population fit anchors (Friberg 2002)
NEUTROPHIL_CONFIG = LineageConfig(
    name="neutrophil", circ0=4.5, mtt_h=134.0, gamma=0.161,
)
#: platelet-lineage analog with longer transit (van Hasselt-style TPO)
PLATELET_CONFIG = LineageConfig(
    name="platelet", circ0=250.0, mtt_h=168.0, gamma=0.20,
)


class FribergLineage:
    """One Friberg transit-compartment lineage (5 ODEs).

    State layout (all in lineage-native units, x10^3/uL):
        prol -> tx1 -> tx2 -> tx3 -> circ

    ``step`` integrates forward Euler with substepping capped at one
    hour, ample for ``k_tr ~ 0.03 /h`` dynamics.
    """

    _COUNT_FLOOR = 1e-4

    def __init__(self, config: LineageConfig) -> None:
        if config.circ0 <= 0.0 or config.mtt_h <= 0.0 or config.gamma <= 0.0:
            raise ValueError("lineage config values must be > 0")
        self.config = config
        self.prol: float = config.circ0
        self.tx1: float = config.circ0
        self.tx2: float = config.circ0
        self.tx3: float = config.circ0
        self.circ: float = config.circ0
        self.transit_acceleration: float = 0.0

    @property
    def k_tr_base(self) -> float:
        """Base transit rate constant (/h) implied by MTT."""
        return 4.0 / self.config.mtt_h

    def stimulate_transit(self, level: float) -> None:
        """Shorten the transit chain (G-CSF / TPO-RA effect).

        Args:
            level: acceleration fraction in [0, 1]; 0.45 halves the
                effective MTT.
        """
        self.transit_acceleration = min(max(level, 0.0), 1.0)

    def step(self, dt_h: float, inhibition: float) -> float:
        """Advance the lineage *dt_h* hours.

        Args:
            dt_h: interval (> 0).
            inhibition: net pharmacodynamic driver; positive values
                suppress proliferation (cytotoxic ``E_drug``), negative
                values stimulate it (growth-factor support, entering as
                ``1 - inhibition > 1`` on the replication term).

        Returns:
            The circulating count after the step.
        """
        if dt_h < 0.0:
            raise ValueError(f"dt_h must be >= 0, got {dt_h}")
        inhibition = min(max(inhibition, -0.5), 0.995)
        remaining = dt_h
        k_tr = self.k_tr_base * (1.0 + 0.45 * self.transit_acceleration)
        cfg = self.config
        while remaining > 1e-12:
            h = min(remaining, 1.0)
            circ_safe = max(self.circ, self._COUNT_FLOOR)
            replication = (
                (cfg.circ0 / circ_safe) ** cfg.gamma
                * (1.0 - inhibition)
            )
            new_prol = self.prol + h * k_tr * ((replication - 1.0) * self.prol)
            new_tx1 = self.tx1 + h * k_tr * (self.prol - self.tx1)
            new_tx2 = self.tx2 + h * k_tr * (self.tx1 - self.tx2)
            new_tx3 = self.tx3 + h * k_tr * (self.tx2 - self.tx3)
            new_circ = self.circ + h * k_tr * (self.tx3 - self.circ)
            self.prol = max(new_prol, self._COUNT_FLOOR)
            self.tx1 = max(new_tx1, self._COUNT_FLOOR)
            self.tx2 = max(new_tx2, self._COUNT_FLOOR)
            self.tx3 = max(new_tx3, self._COUNT_FLOOR)
            self.circ = max(new_circ, self._COUNT_FLOOR)
            remaining -= h
        return self.circ

    def count(self) -> float:
        """Current circulating count (x10^3/uL)."""
        return self.circ


# ============================================================================
# Minimal erythropoiesis with EPO feedback
# ============================================================================

class ErythropoiesisModel:
    """Reduced four-pool erythropoiesis system with EPO feedback.

    States:
        ``progenitors``   erythroid progenitors, normalized (SS = 1.0)
        ``marrow_retic``  marrow reticulocyte pool (g/dL-equivalent)
        ``hemoglobin``    circulating red-cell mass carried as Hb g/dL
        ``blood_retic``   circulating reticulocytes (g/dL-equivalent)
        ``epo``           plasma erythropoietin (U/L)

    Kidney hypoxia sensing is log-linear in Hb (plasma EPO roughly
    doubles per ~2 g/dL fall once below baseline) and scales with the
    renal-function fraction, reproducing the diagnostic EPO patterns:
    appropriately high in iron-deficiency/blood-loss anemia,
    inappropriately low in renal anemia.

    Steady state is exact by construction: differentiation out of the
    progenitor pool equals the RBC replacement flux
    ``Hb0 / lifespan``, so an untouched model holds Hb at baseline
    indefinitely.
    """

    def __init__(
        self,
        is_female: bool = False,
        baseline_hb_g_dl: float | None = None,
        renal_function_fraction: float = 1.0,
        iron_availability: float = 1.0,
    ) -> None:
        self.hb0: float = (
            baseline_hb_g_dl if baseline_hb_g_dl is not None
            else (13.5 if is_female else 14.5)
        )
        #: RBC lifespan (hours); ~120 d normal, shorten for hemolysis
        self.lifespan_h: float = 2880.0
        #: marrow reticulocyte maturation time (hours, ~3 d)
        self.retic_marrow_h: float = 72.0
        #: circulating reticulocyte lifespan (hours, ~1.2 d)
        self.retic_blood_h: float = 29.0
        #: plasma EPO equilibration time constant (hours; t1/2 ~ 4-9 h)
        self.tau_epo_h: float = 6.0
        #: log-linear kidney-sensing gain per g/dL below baseline
        self.sensing_gain: float = 0.35
        #: baseline plasma EPO (U/L; reference range 4-26)
        self.epo0: float = 10.0
        #: saturating EPO proliferation response (Hill form chosen so a
        #: plasma EPO equal to baseline maps to factor exactly 1.0)
        self.epo_response_emax: float = 4.0
        self.epo_response_ec50: float = 3.0  # in multiples of epo0
        #: logistic cap on progenitor expansion (relative to SS)
        self.h_max: float = 3.0
        #: chemotherapy sensitivity of the RBC lineage relative to ANC
        self.chemo_sensitivity: float = 0.7

        self.progenitors: float = 1.0
        self.marrow_retic: float = self.hb0 * self.retic_marrow_h / self.lifespan_h
        self.hemoglobin: float = self.hb0
        self.blood_retic: float = self.hb0 * self.retic_blood_h / self.lifespan_h
        self.epo: float = self.epo0

        self.set_renal_function_fraction(renal_function_fraction)
        self.set_iron_availability(iron_availability)
        self.esa_infusion_u_l_h: float = 0.0
        self._k_rel = 1.0 / self.retic_marrow_h
        self._k_diff0 = self.hb0 / (self.lifespan_h * 1.0)
        self._a0 = self._k_diff0 / (1.0 - 1.0 / self.h_max)

    # -- intervention channels -------------------------------------------------

    def set_renal_function_fraction(self, fraction: float) -> None:
        """Set kidney oxygen-sensing capacity in [0.08, 1].

        Coupling point for the doc/30 §6 renal module: falling renal
        function suppresses EPO production despite anemia (renal
        anemia of CKD).
        """
        self.renal_function_fraction = min(max(fraction, 0.08), 1.0)

    def set_iron_availability(self, fraction: float) -> None:
        """Limit erythropoietic capacity (iron-deficiency anemia).

        Args:
            fraction: available iron fraction in [0, 1]; shrinks the
                EPO-expandable progenitor headroom, producing a stable
                hypoproliferative anemia rather than aplasia.
        """
        self.iron_availability = min(max(fraction, 0.0), 1.0)

    def transfuse(self, units: float) -> None:
        """Add packed-RBC units straight to the circulating pool.

        One unit raises Hb by ~1 g/dL in a 70 kg adult.
        """
        if units < 0.0:
            raise ValueError("units must be >= 0")
        self.hemoglobin += units

    def bleed(self, hb_loss_g_dl: float) -> None:
        """Remove circulating red-cell mass (surgery, GI bleed...)."""
        if hb_loss_g_dl < 0.0:
            raise ValueError("hb_loss_g_dl must be >= 0")
        self.hemoglobin = max(0.5, self.hemoglobin - hb_loss_g_dl)

    def administer_epo_bolus(self, dose_u_l: float) -> None:
        """Instantaneous ESA/EPO bolus raising the plasma EPO pool."""
        if dose_u_l < 0.0:
            raise ValueError("dose_u_l must be >= 0")
        self.epo += dose_u_l

    # -- integration -----------------------------------------------------------

    def step(self, dt_h: float, chemo_inhibition: float) -> float:
        """Advance *dt_h* hours; returns hemoglobin (g/dL).

        ``chemo_inhibition`` is the Bliss-combined cytotoxic effect on
        progenitor proliferation already scaled for RBC-lineage
        sensitivity by the caller.
        """
        if dt_h < 0.0:
            raise ValueError(f"dt_h must be >= 0, got {dt_h}")
        chemo_inhibition = min(max(chemo_inhibition, 0.0), 1.0)
        remaining = dt_h
        while remaining > 1e-12:
            h = min(remaining, 1.0)
            epo_ratio = self.epo / self.epo0
            h_cap = 1.0 + (self.h_max - 1.0) * self.iron_availability
            prol_factor = self.epo_response_emax * epo_ratio / (
                self.epo_response_ec50 + epo_ratio
            )
            a_e = (
                self._a0
                * prol_factor
                * max(1.0 - chemo_inhibition, 0.0)
            )
            k_diff = self._k_diff0
            d_progen = (
                a_e * self.progenitors * (1.0 - self.progenitors / h_cap)
                - k_diff * self.progenitors
            )
            d_marrow = k_diff * self.progenitors - self._k_rel * self.marrow_retic
            d_hb = self._k_rel * self.marrow_retic - self.hemoglobin / self.lifespan_h
            d_blood_retic = (
                self._k_rel * self.marrow_retic
                - self.blood_retic / self.retic_blood_h
            )
            production = (
                self.epo0
                * math.exp(self.sensing_gain * (self.hb0 - self.hemoglobin))
                * self.renal_function_fraction
                + self.esa_infusion_u_l_h * self.tau_epo_h
            )
            d_epo = (production - self.epo) / self.tau_epo_h

            self.progenitors = min(
                max(self.progenitors + h * d_progen, 1e-6), self.h_max * 2.0
            )
            self.marrow_retic = max(self.marrow_retic + h * d_marrow, 0.0)
            self.hemoglobin = min(max(self.hemoglobin + h * d_hb, 0.5), 25.0)
            self.blood_retic = max(self.blood_retic + h * d_blood_retic, 0.0)
            self.epo = min(max(self.epo + h * d_epo, 0.0), 100_000.0)
            remaining -= h
        return self.hemoglobin

    # -- outputs ---------------------------------------------------------------

    def reticulocyte_percent(self) -> float:
        """Circulating reticulocytes as percent of red cells (SS ~ 1%)."""
        if self.hemoglobin <= 0.0:
            return 0.0
        return 100.0 * self.blood_retic / self.hemoglobin


# ============================================================================
# Facade
# ============================================================================

class HematologySystem:
    """Neutrophils + platelets + erythropoiesis stepped together.

    Usage inside the doc/28 hourly loop::

        heme = create_hematology_system(is_female=False)
        heme.register_myelosuppressant(MyelosuppressionParams("docetaxel", 0.85, 0.082))
        heme.step(dt_h=1.0, exposures={"docetaxel": pbpk.central_mg_l})
        ...
        labs.platelets_per_ul = heme.lab_values()["platelets_per_ul"]

    Growth-factor support: call :meth:`set_growth_factor_support`
    (neutrophil G-CSF) or :meth:`set_tpo_mimetic` (platelets) before
    stepping; effects persist until changed.
    """

    def __init__(
        self,
        is_female: bool = False,
        renal_function_fraction: float = 1.0,
    ) -> None:
        self.neutrophils = FribergLineage(NEUTROPHIL_CONFIG)
        self.platelets = FribergLineage(PLATELET_CONFIG)
        self.erythropoiesis = ErythropoiesisModel(
            is_female=is_female,
            renal_function_fraction=renal_function_fraction,
        )
        self._drugs: dict[str, MyelosuppressionParams] = {}
        self.rbc_chemo_sensitivity_override: float | None = None
        self._gcsf_level: float = 0.0
        self._tpo_level: float = 0.0

    # -- configuration ---------------------------------------------------------

    def register_myelosuppressant(self, params: MyelosuppressionParams) -> None:
        """Register (or replace) a cytotoxic's PD parameters."""
        self._drugs[params.drug_name] = params

    def set_growth_factor_support(self, level: float) -> None:
        """G-CSF support level in [0, 1] (filgrastim / pegfilgrastim)."""
        level = min(max(level, 0.0), 1.0)
        self.neutrophils.stimulate_transit(level)
        self._gcsf_level = level

    def set_tpo_mimetic(self, level: float) -> None:
        """TPO-receptor agonist support in [0, 1]
        (eltrombopag / romiplostim)."""
        level = min(max(level, 0.0), 1.0)
        self.platelets.stimulate_transit(level)
        self._tpo_level = level

    def set_renal_function_fraction(self, fraction: float) -> None:
        """Propagate renal function to kidney EPO sensing (§6 coupling)."""
        self.erythropoiesis.set_renal_function_fraction(fraction)

    # -- effect composition ----------------------------------------------------

    def _combined_inhibition(
        self,
        exposures: dict[str, float],
        override_sensitivity: float | None,
    ) -> float:
        survival = 1.0
        for name, conc in exposures.items():
            params = self._drugs.get(name)
            if params is None:
                continue
            effect = params.effect_at(conc)
            if override_sensitivity is not None:
                effect *= override_sensitivity
            survival *= 1.0 - min(effect, 1.0)
        return 1.0 - survival

    # -- stepping --------------------------------------------------------------

    def step(self, dt_h: float, exposures: dict[str, float] | None = None) -> dict[str, float]:
        """Advance all lineages *dt_h* hours.

        Args:
            dt_h: interval in hours (> 0).
            exposures: map of registered drug name -> plasma
                concentration (mg/L) for this interval.

        Returns:
            Convenience snapshot of the headline counts.
        """
        if dt_h < 0.0:
            raise ValueError(f"dt_h must be >= 0, got {dt_h}")
        exposures = exposures or {}

        anc_inhibition = self._combined_inhibition(exposures, None)
        plt_inhibition = self._combined_inhibition(exposures, 0.85)
        rbc_inhibition = self._combined_inhibition(
            exposures, self.erythropoiesis.chemo_sensitivity
        )
        anc = self.neutrophils.step(
            dt_h, anc_inhibition - 0.4 * self._gcsf_level
        )
        plt = self.platelets.step(
            dt_h, plt_inhibition - 0.35 * self._tpo_level
        )
        hb = self.erythropoiesis.step(dt_h, rbc_inhibition)
        return {
            "anc_x10e3_ul": anc,
            "platelets_x10e3_ul": plt,
            "hemoglobin_g_dl": hb,
            "reticulocyte_pct": self.erythropoiesis.reticulocyte_percent(),
            "epo_u_l": self.erythropoiesis.epo,
        }

    # -- clinical outputs ------------------------------------------------------

    def anc_ctcae_grade(self) -> int:
        """CTCAE v5 neutropenia grade (0-4)."""
        anc = self.neutrophils.count()
        if anc >= 1.5:
            return 0
        if anc >= 1.0:
            return 1
        if anc >= 0.5:
            return 2
        if anc >= 0.2:
            return 3
        return 4

    def platelet_ctcae_grade(self) -> int:
        """CTCAE v5 thrombocytopenia grade (0-4)."""
        plt = self.platelets.count()
        if plt >= 100.0:
            return 0
        if plt >= 75.0:
            return 1
        if plt >= 50.0:
            return 2
        if plt >= 25.0:
            return 3
        return 4

    def lab_values(self) -> dict[str, float]:
        """Snapshot formatted for ClinicalLabs-style integration."""
        return {
            "anc_per_ul": self.neutrophils.count() * 1000.0,
            "wbc_per_ul": self.neutrophils.count() * 1600.0,
            "platelets_per_ul": self.platelets.count() * 1000.0,
            "hemoglobin_g_dl": self.erythropoiesis.hemoglobin,
            "reticulocyte_pct": self.erythropoiesis.reticulocyte_percent(),
            "epo_u_l": self.erythropoiesis.epo,
        }


def create_hematology_system(
    is_female: bool = False,
    renal_function_fraction: float = 1.0,
) -> HematologySystem:
    """Build a :class:`HematologySystem` with sex-appropriate baselines."""
    return HematologySystem(
        is_female=is_female,
        renal_function_fraction=renal_function_fraction,
    )
