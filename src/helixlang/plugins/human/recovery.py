"""Post-treatment recovery dynamics and long-term effects (doc/28).

While a treatment regimen is active, drug effect dominates and organ
recovery is frozen.  Once :meth:`RecoveryModel.set_treatment_inactive`
is called, every biomarker relaxes exponentially toward its baseline
with an organ-specific first-order rate constant

    k[organ] = ln(2) / recovery_half_life_h,

derived from published regeneration/repopulation half-lives (liver ~14 d,
kidney acute injury ~7 d, bone marrow 14-21 d post-chemo, cardiac EF
3-6 months post-anthracycline).  Two additional phenomena are modeled:

- **Rebound** — transient overshoot/undershoot after stopping drugs that
  suppress a physiological axis (steroid withdrawal, opioid tolerance).
  The excursion follows a gamma-like envelope peaking at
  ``onset_delay_h`` after treatment stop and decaying over
  ``duration_h``.
- **Sequelae** — delayed long-term effects (cisplatin ototoxicity,
  chemo brain) that subtract permanently or semi-permanently from
  organ recovery fractions.

Module structure:
    RecoveryEvent          milestone emitted during recovery
    Sequela                permanent / long-term effect specification
    ReboundSpec            withdrawal-excursion specification
    RecoveryModel          stateful recovery stepper
    ORGAN_RECOVERY_PROFILES built-in half-life profiles
    DEFAULT_SEQUELAE       built-in sequela templates
    create_recovery_model  factory keyed on the administered regimen

References:
- Michalopoulos GK. Hepatology 2007 (hepatocyte regeneration)
- Fisa R et al. / Freifeld AG et al. Clin Infect Dis 2011 (neutropenia
  recovery 14-21 d post-chemotherapy)
- Cardinale D et al. J Am Coll Cardiol 2015 (trastuzumab/anthracycline
  LVEF recovery over 3-6 months)
- Brock PR et al. Lancet Oncol 2012 (cisplatin ototoxicity onset)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "RecoveryEvent",
    "Sequela",
    "ReboundSpec",
    "RecoveryModel",
    "ORGAN_RECOVERY_PROFILES",
    "DEFAULT_SEQUELAE",
    "create_recovery_model",
]

#: first-order rate constant applied to organs without a profile entry
#: (half-life 30 days)
_DEFAULT_RECOVERY_K_PER_H = math.log(2.0) / (30.0 * 24.0)
#: biomarker considered recovered once within this fraction of baseline
_RECOVERY_COMPLETE_TOLERANCE = 0.01


def _rate_from_half_life(half_life_h: float) -> float:
    """Convert a recovery half-life in hours to k in per-hour units."""
    if half_life_h <= 0.0:
        raise ValueError(f"half_life_h must be > 0, got {half_life_h}")
    return math.log(2.0) / half_life_h


# ============================================================================
# Data structures
# ============================================================================

@dataclass(slots=True)
class RecoveryEvent:
    """A single recovery event or milestone.

    Attributes:
        time_h: simulation time at which the event occurred.
        event_type: one of ``"recovery_start"``, ``"recovery_complete"``,
            ``"rebound"``, ``"sequelae_onset"``, ``"relapse"``.
        organ: organ the event pertains to (``"systemic"`` if global).
        description: human-readable summary.
        severity_change: signed change in overall severity attributable
            to the event (positive = worsening).
    """

    time_h: float
    event_type: str
    organ: str
    description: str
    severity_change: float = 0.0

    def __post_init__(self) -> None:
        valid = (
            "recovery_start", "recovery_complete", "rebound",
            "sequelae_onset", "relapse",
        )
        if self.event_type not in valid:
            raise ValueError(f"event_type must be one of {valid}, got {self.event_type!r}")


@dataclass(slots=True)
class Sequela:
    """A permanent or long-term drug effect.

    Attributes:
        name: identifier (e.g. ``"cisplatin_ototoxicity"``).
        organ: organ carrying the deficit.
        severity: deficit magnitude in [0, 1].
        onset_delay_h: hours after first dose before the sequela appears.
        reversible: whether the deficit decays once expressed.
        recovery_half_life_h: decay half-life once expressed;
            ``math.inf`` = permanent.
    """

    name: str
    organ: str
    severity: float
    onset_delay_h: float
    reversible: bool = False
    recovery_half_life_h: float = math.inf

    def __post_init__(self) -> None:
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError(f"severity must be in [0, 1], got {self.severity}")
        if self.onset_delay_h < 0.0:
            raise ValueError("onset_delay_h must be >= 0")
        if self.recovery_half_life_h <= 0.0:
            raise ValueError("recovery_half_life_h must be > 0")


@dataclass(slots=True)
class ReboundSpec:
    """Withdrawal-excursion specification for one biomarker.

    After treatment stops, the biomarker executes a gamma-shaped
    excursion away from baseline peaking at ``onset_delay_h`` and
    decaying over ``duration_h``, of magnitude
    ``excursion_fraction * |baseline| * direction``.

    Attributes:
        name: rebound identifier (e.g. ``"opioid_withdrawal"``).
        biomarker: affected biomarker key.
        excursion_fraction: peak magnitude relative to |baseline|.
        direction: ``+1`` excites above baseline (symptom scores),
            ``-1`` suppresses below baseline (hormone axes).
        onset_delay_h: time from treatment stop to peak excursion.
        duration_h: characteristic decay width after the peak.
    """

    name: str
    biomarker: str
    excursion_fraction: float
    direction: int = -1
    onset_delay_h: float = 24.0
    duration_h: float = 168.0

    def __post_init__(self) -> None:
        if not 0.0 < self.excursion_fraction <= 1.0:
            raise ValueError(
                f"excursion_fraction must be in (0, 1], got {self.excursion_fraction}"
            )
        if self.direction not in (-1, 1):
            raise ValueError("direction must be -1 or 1")
        if self.onset_delay_h <= 0.0 or self.duration_h <= 0.0:
            raise ValueError("onset_delay_h and duration_h must be > 0")

    def envelope(self, t_since_stop_h: float) -> float:
        """Excursion envelope value at *t_since_stop_h*; peaks at 1.

        Linear ramp up to ``onset_delay_h``, then exponential decay
        governed by ``duration_h``.
        """
        x = t_since_stop_h
        if x <= 0.0:
            return 0.0
        if x <= self.onset_delay_h:
            return x / self.onset_delay_h
        return math.exp(-(x - self.onset_delay_h) / self.duration_h)


@dataclass
class RecoveryModel:
    """Models post-treatment recovery dynamics.

    Attributes:
        baseline_biomarkers: healthy reference values per biomarker.
        current_biomarkers: live simulated values per biomarker.
        organ_recovery_rates: organ -> first-order rate constant
            (per hour); use :func:`_rate_from_half_life` for conversion.
        sequela_list: delayed long-term effects to express.
        recovery_events: chronological event log.
        is_treatment_active: while True no recovery occurs.
        rebound_specs: withdrawal-excursion definitions.
    """

    baseline_biomarkers: dict[str, float]
    current_biomarkers: dict[str, float]
    organ_recovery_rates: dict[str, float] = field(default_factory=dict)
    sequela_list: list[Sequela] = field(default_factory=list)
    recovery_events: list[RecoveryEvent] = field(default_factory=list)
    is_treatment_active: bool = True
    rebound_specs: list[ReboundSpec] = field(default_factory=list)

    def __post_init__(self) -> None:
        missing = set(self.current_biomarkers) - set(self.baseline_biomarkers)
        if missing:
            raise ValueError(f"current biomarkers lack baselines: {sorted(missing)}")
        self._stop_time_h: float = math.inf
        self._initial_deviations: dict[str, float] = {}
        self._residual_deviations: dict[str, float] = {}
        self._rebound_emitted: set[str] = set()
        self._sequela_expressed: set[str] = set()
        self._complete_emitted: set[str] = set()
        self._start_emitted: bool = False

    # ------------------------------------------------------------------
    # Phase control
    # ------------------------------------------------------------------

    def set_treatment_inactive(self) -> None:
        """Mark treatment as stopped; begin the recovery phase."""
        self.is_treatment_active = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _biomarkers_for_organ(self, organ: str) -> list[str]:
        """Return biomarker keys mapped to *organ*, else systemic ones."""
        mapped = [
            b for b in self.current_biomarkers
            if _BIOMARKER_ORGAN.get(b, "systemic") == organ
        ]
        return mapped or list(self.current_biomarkers)

    def _organ_rate(self, biomarker: str) -> float:
        """First-order recovery rate constant for one biomarker."""
        organ = _BIOMARKER_ORGAN.get(biomarker, "systemic")
        return self.organ_recovery_rates.get(organ, _DEFAULT_RECOVERY_K_PER_H)

    def _rebound_term(self, biomarker: str, t_since_stop_h: float) -> float:
        """Additive excursion for all rebound specs targeting a biomarker."""
        term = 0.0
        for spec in self.rebound_specs:
            if spec.biomarker != biomarker:
                continue
            base = abs(self.baseline_biomarkers.get(biomarker, 0.0))
            term += (
                spec.direction * spec.excursion_fraction * base
                * spec.envelope(t_since_stop_h)
            )
        return term

    def _sequela_penalty(self, organ: str, t_since_stop_h: float) -> float:
        """Residual sequela burden on an organ, accounting for decay."""
        penalty = 0.0
        for seq in self.sequela_list:
            if seq.organ != organ or seq.name not in self._sequela_expressed:
                continue
            remaining = seq.severity
            if seq.reversible and math.isfinite(seq.recovery_half_life_h):
                elapsed = max(0.0, t_since_stop_h - max(seq.onset_delay_h, 0.0))
                remaining *= math.exp(-_rate_from_half_life(seq.recovery_half_life_h) * elapsed)
            penalty += remaining
        return min(penalty, 1.0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, dt_h: float, current_time_h: float) -> dict[str, float]:
        """Advance recovery by *dt_h* hours. Returns updated biomarkers.

        Active treatment: values are returned unchanged (drug effect
        dominates; recovery = 0).  Stopped treatment: residual deviations
        from baseline decay exponentially at each organ's rate constant,
        rebound envelopes are added absolutely, and due sequelae express
        themselves as events.
        """
        if dt_h < 0.0:
            raise ValueError(f"dt_h must be >= 0, got {dt_h}")

        for seq in self.sequela_list:
            if seq.name in self._sequela_expressed:
                continue
            if current_time_h >= seq.onset_delay_h:
                self._sequela_expressed.add(seq.name)
                self.recovery_events.append(RecoveryEvent(
                    time_h=current_time_h,
                    event_type="sequelae_onset",
                    organ=seq.organ,
                    description=f"{seq.name} expressed in {seq.organ}",
                    severity_change=seq.severity,
                ))

        if self.is_treatment_active:
            return dict(self.current_biomarkers)

        if not self._start_emitted:
            self._start_emitted = True
            self._stop_time_h = current_time_h - dt_h
            self._residual_deviations = {
                b: self.current_biomarkers[b] - self.baseline_biomarkers[b]
                for b in self.current_biomarkers
            }
            self._initial_deviations = dict(self._residual_deviations)
            self.recovery_events.append(RecoveryEvent(
                time_h=current_time_h,
                event_type="recovery_start",
                organ="systemic",
                description="Treatment stopped; recovery phase begins",
            ))

        t_since_stop = max(0.0, current_time_h - self._stop_time_h)
        for name, baseline in self.baseline_biomarkers.items():
            k = self._organ_rate(name)
            deviation = self._residual_deviations.get(name, 0.0) * math.exp(-k * dt_h)
            self._residual_deviations[name] = deviation
            self.current_biomarkers[name] = (
                baseline + deviation + self._rebound_term(name, t_since_stop)
            )

            initial = self._initial_deviations.get(name, 0.0)
            organ = _BIOMARKER_ORGAN.get(name, "systemic")
            if organ not in self._complete_emitted and abs(deviation) <= (
                _RECOVERY_COMPLETE_TOLERANCE * max(abs(initial), 1e-12)
            ):
                self._complete_emitted.add(organ)
                self.recovery_events.append(RecoveryEvent(
                    time_h=current_time_h,
                    event_type="recovery_complete",
                    organ=organ,
                    description=f"{organ} markers returned to baseline",
                ))

        return dict(self.current_biomarkers)

    def check_rebound(self, current_time_h: float) -> list[RecoveryEvent]:
        """Check for rebound effects past their onset window.

        A rebound fires once, when at least half of its delay-to-peak
        has elapsed since treatment stop.  Returns the newly emitted
        events (also appended to :attr:`recovery_events`).
        """
        emitted: list[RecoveryEvent] = []
        if self.is_treatment_active or not math.isfinite(self._stop_time_h):
            return emitted
        t_since_stop = current_time_h - self._stop_time_h
        for spec in self.rebound_specs:
            if spec.name in self._rebound_emitted:
                continue
            if t_since_stop >= 0.5 * spec.onset_delay_h:
                self._rebound_emitted.add(spec.name)
                direction_word = (
                    "surge above baseline"
                    if spec.direction > 0 else "dip below baseline"
                )
                event = RecoveryEvent(
                    time_h=current_time_h,
                    event_type="rebound",
                    organ=_BIOMARKER_ORGAN.get(spec.biomarker, "systemic"),
                    description=(
                        f"{spec.name}: {spec.biomarker} expected to {direction_word} "
                        f"(peak ~{spec.excursion_fraction:.0%} of baseline)"
                    ),
                )
                emitted.append(event)
                self.recovery_events.append(event)
        return emitted

    def get_organ_recovery_fraction(self, organ: str, current_time_h: float) -> float:
        """Return [0, 1] fraction of organ function recovered.

        While treatment remains active the fraction is 0.0.  Afterwards
        it is the share of the organ's initial deviation from baseline
        that has been closed, multiplied by the residual sequela-free
        capacity ``(1 - sequela_penalty)``.
        """
        if self.is_treatment_active:
            return 0.0
        if not math.isfinite(self._stop_time_h):
            return 0.0

        members = self._biomarkers_for_organ(organ.strip().lower())
        fractions: list[float] = []
        for name in members:
            initial = self._initial_deviations.get(name, 0.0)
            residual = self._residual_deviations.get(name, initial)
            if abs(initial) <= 1e-12:
                fractions.append(1.0)
                continue
            closed = 1.0 - abs(residual) / abs(initial)
            fractions.append(min(1.0, max(0.0, closed)))
        recovered = sum(fractions) / len(fractions) if fractions else 0.0

        t_since_stop = max(0.0, current_time_h - self._stop_time_h)
        penalty = self._sequela_penalty(organ.strip().lower(), t_since_stop)
        return min(1.0, max(0.0, recovered * (1.0 - penalty)))


# ============================================================================
# Lookup tables and presets
# ============================================================================

#: biomarker -> owning organ (unlisted biomarkers are treated as systemic)
_BIOMARKER_ORGAN: dict[str, str] = {
    "alt_u_l": "liver", "ast_u_l": "liver", "alp_u_l": "liver",
    "total_bilirubin_mg_dl": "liver", "albumin_g_dl": "liver",
    "creatinine_mg_dl": "kidney", "egfr_ml_min_1_73m2": "kidney",
    "bun_mg_dl": "kidney",
    "wbc_count_per_ul": "bone_marrow", "anc_per_ul": "bone_marrow",
    "platelets_per_ul": "bone_marrow", "hemoglobin_g_dl": "bone_marrow",
    "ejection_fraction": "heart", "troponin_ng_ml": "heart",
    "cortisol_ug_dl": "adrenal",
    "pain_score": "cns",
}

#: organ -> recovery half-life in hours, from published regeneration and
#: repopulation kinetics
ORGAN_RECOVERY_PROFILES: dict[str, float] = {
    "liver": 14.0 * 24.0,           # hepatocyte regeneration, ~3 mo to full recovery
    "kidney": 7.0 * 24.0,           # acute tubular injury resolution
    "bone_marrow": 18.0 * 24.0,     # WBC nadir recovery 14-21 d post-chemo
    "heart": 135.0 * 24.0,          # LVEF recovery 3-6 months post-anthracycline
    "cns": 90.0 * 24.0,
    "adrenal": 21.0 * 24.0,
    "systemic": 30.0 * 24.0,
}

#: built-in sequela templates instantiated by the factory
DEFAULT_SEQUELAE: dict[str, Sequela] = {
    "cisplatin_ototoxicity": Sequela(
        name="cisplatin_ototoxicity",
        organ="ear",
        severity=0.40,
        onset_delay_h=14.0 * 24.0,   # 1-2 weeks post-dose (Brock 2012)
        reversible=False,
    ),
    "chemo_brain": Sequela(
        name="chemo_brain",
        organ="brain",
        severity=0.30,
        onset_delay_h=60.0 * 24.0,
        reversible=True,
        recovery_half_life_h=270.0 * 24.0,   # resolves over 6-12 months
    ),
    "anthracycline_cardiomyopathy": Sequela(
        name="anthracycline_cardiomyopathy",
        organ="heart",
        severity=0.25,
        onset_delay_h=90.0 * 24.0,
        reversible=True,
        recovery_half_life_h=365.0 * 24.0,
    ),
}


# ============================================================================
# Factory
# ============================================================================

#: drugs whose regimens carry the matching long-term risk
_CISPLATIN_FAMILY = {"cisplatin", "carboplatin", "oxaliplatin"}
_ANTHRACYCLINES = {"doxorubicin", "daunorubicin", "epirubicin"}
_CHEMO_BRAIN_DRUGS = {
    "cisplatin", "methotrexate", "cyclophosphamide", "doxorubicin",
    "paclitaxel", "carboplatin", "ifosfamide",
}
_OPIOIDS = {"morphine", "fentanyl", "oxycodone", "hydromorphone", "codeine"}
_CORTICOSTEROIDS = {"prednisone", "prednisolone", "dexamethasone", "hydrocortisone"}


def create_recovery_model(
    drug_names: list[str],
    baseline_biomarkers: dict[str, float],
) -> RecoveryModel:
    """Build a :class:`RecoveryModel` tailored to the administered drugs.

    Organ recovery rates default to :data:`ORGAN_RECOVERY_PROFILES`;
    platinum and anthracycline exposure slows kidney and heart recovery
    respectively.  Sequelae and rebound specs are attached by drug class
    (platin -> ototoxicity, chemo agents -> chemo brain, anthracyclines
    -> cardiomyopathy, opioids/corticosteroids -> withdrawal rebounds).

    Args:
        drug_names: administered drugs (any letter case).
        baseline_biomarkers: healthy reference values per biomarker.
    """
    names = {name.strip().lower() for name in drug_names}

    organ_rates = {
        organ: _rate_from_half_life(half_life_h)
        for organ, half_life_h in ORGAN_RECOVERY_PROFILES.items()
    }
    if names & _CISPLATIN_FAMILY:
        organ_rates["kidney"] *= 0.5      # chronic tubular damage halves recovery speed
        organ_rates["bone_marrow"] *= 0.75
    if names & _ANTHRACYCLINES:
        organ_rates["heart"] *= 0.5

    sequelae: list[Sequela] = []
    if names & _CISPLATIN_FAMILY:
        sequelae.append(DEFAULT_SEQUELAE["cisplatin_ototoxicity"])
    if names & _CHEMO_BRAIN_DRUGS:
        sequelae.append(DEFAULT_SEQUELAE["chemo_brain"])
    if names & _ANTHRACYCLINES:
        sequelae.append(DEFAULT_SEQUELAE["anthracycline_cardiomyopathy"])

    rebound_specs: list[ReboundSpec] = []
    if names & _OPIOIDS:
        rebound_specs.append(ReboundSpec(
            name="opioid_withdrawal",
            biomarker="pain_score",
            excursion_fraction=0.60,
            direction=1,
            onset_delay_h=24.0,
            duration_h=7.0 * 24.0,
        ))
    if names & _CORTICOSTEROIDS:
        rebound_specs.append(ReboundSpec(
            name="corticosteroid_withdrawal",
            biomarker="cortisol_ug_dl",
            excursion_fraction=0.50,
            direction=-1,
            onset_delay_h=36.0,
            duration_h=10.0 * 24.0,
        ))

    model = RecoveryModel(
        baseline_biomarkers=dict(baseline_biomarkers),
        current_biomarkers=dict(baseline_biomarkers),
        organ_recovery_rates=organ_rates,
        sequela_list=sequelae,
        rebound_specs=rebound_specs,
    )
    return model
