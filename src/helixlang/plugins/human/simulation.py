"""Long-term human simulation engine (doc/27 Stage F, section 9).

Integrates whole-body pharmacokinetics (PBPK, minute resolution),
pharmacodynamics (Hill equation, Rowland & Tozer 2020), dynamic flux
balance analysis (Mahadevan et al. 2002 Biophys J 83:1905), and
intracellular metabolite pools into one time-course simulation of a
diseased virtual patient under drug treatment.  The metabolic engine
is a constraint-based tissue proxy: a human GEM loaded from
``base_model_path`` (JSON) when available, otherwise the curated E.
coli core model (Orth 2010 Mol Syst Biol 6:390) held at physiological
blood glucose/oxygen levels by perfusion/washout each step (Guyton &
Hall 2016).  Disease states apply through
:func:`~helixlang.plugins.human.disease.apply_disease_state`; the sibling
modules ``pharmacokinetics`` / ``pharmacodynamics`` are used when
installed, with minimal doc/27-conformant fallbacks below keeping
this engine fully functional on its own.
"""

from __future__ import annotations

import copy
import csv
import json
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

from helixlang.core.errors import BioError
from helixlang.plugins.human.disease import apply_disease_state
from helixlang.plugins.human.drug import (
    BIOLOGIC,
    INTRAMUSCULAR,
    INTRATHECAL,
    IV,
    IV_INFUSION,
    OLIGONUCLEOTIDE,
    ORAL,
    SUBCUTANEOUS,
    Drug,
)
from helixlang.plugins.human.physiology import create_default_physiology
from helixlang.plugins.runtime.metabolism import (
    ECOLI_CORE_MODEL,
    DynamicFBAConfig,
    DynamicFluxBalance,
    MetabolitePool,
    MetabolitePoolConfig,
    load_model_from_json,
)

try:
    from scipy.integrate import solve_ivp

    _HAS_SCIPY = True
except ImportError:
    solve_ivp = None
    _HAS_SCIPY = False

try:
    from helixlang.plugins.human.pharmacokinetics import PBPKConfig
except ImportError:

    @dataclass(slots=True)
    class PBPKConfig:  # type: ignore[no-redef]
        """Fallback PBPK configuration (doc/27 section 7.4)."""

        dt_min: float = 1.0
        total_time_h: float = 24.0
        n_compartments: int = 6
        use_tissue_scaling: bool = True
        protein_binding: bool = True
        cardiac_output_l_per_min: float = 5.0
        plasma_volume_l: float = 3.0
        liver_volume_l: float = 1.5
        kidney_volume_l: float = 0.3
        brain_volume_l: float = 1.4
        muscle_volume_l: float = 24.0
        adipose_volume_l: float = 15.0


if TYPE_CHECKING:
    from helixlang.plugins.human.pharmacodynamics import PDEffect, Pharmacodynamics
else:
    try:
        from helixlang.plugins.human.pharmacodynamics import PDEffect, Pharmacodynamics
    except ImportError:

        @dataclass(slots=True)
        class PDEffect:
            """Fallback single drug-target PD effect (doc/27 section 8.3)."""

            target_reaction: str
            target_gene: str = ""
            effect_type: str = "inhibition"
            ec50_um: float = 1.0
            emax: float = 1.0
            hill_coefficient: float = 1.0
            baseline_effect: float = 0.0

        @dataclass(slots=True)
        class Pharmacodynamics:
            """Fallback complete PD model for a drug (doc/27 section 8.3)."""

            drug_name: str
            effects: list[PDEffect] = field(default_factory=list)
            dose_response_model: str = "hill"
            target_biomarkers: dict[str, float] = field(default_factory=dict)
            toxicity_concentration_um: float = 100.0
            therapeutic_window: tuple[float, float] = (1.0, 50.0)


_BLOOD_GLUCOSE_MM = 5.0
_BLOOD_OXYGEN_MM = 20.0
_POOL_WASHOUT_PER_H = 2.0
_BIOMARKER_HALF_LIFE_H = 72.0
_THERAPEUTIC_RESPONSE_THRESHOLD = 0.5
_INFUSION_DURATION_H = 1.0
_TISSUES = ("liver", "kidney", "brain", "muscle", "adipose")
#: biomarker-id -> E. coli core pool alias (proxy-model mapping)
_BIOMARKER_ALIASES: dict[str, str] = {
    "glucose": "GLC",
    "lactate": "Lac",
    "acetate": "Ac",
    "co2": "CO2",
    "pyruvate": "PYR",
}


def _hill_response(conc_um: float, ec50_um: float, hill: float) -> float:
    """Hill saturation fraction ``C^n / (EC50^n + C^n)`` in [0, 1]."""
    if ec50_um <= 0.0:
        return 1.0 if conc_um > 0.0 else 0.0
    if conc_um <= 0.0:
        return 0.0
    ratio = (conc_um / ec50_um) ** hill
    return float(ratio / (1.0 + ratio))


def _trapz(values: list[float], dt: float) -> float:
    """Trapezoidal integral of ``values`` sampled every ``dt``."""
    if len(values) < 2:
        return 0.0
    return sum(0.5 * (values[i] + values[i + 1]) * dt for i in range(len(values) - 1))


def _tissue_access_factor(drug: Drug) -> float:
    """Tissue-penetration multiplier from drug class and size.

    Small molecules permeate the full tissue water (1.0); large
    biologics are confined to plasma plus a small interstitial
    fraction (Rowland & Tozer 2020, Ch. 10: antibody apparent Vd
    approaches plasma volume); oligonucleotides sit in between.
    """
    mw = drug.molecule.molecular_weight_da
    if drug.molecule.drug_type == BIOLOGIC or mw > 30000.0:
        return 0.12
    if drug.molecule.drug_type == OLIGONUCLEOTIDE or mw > 8000.0:
        return 0.25
    return 1.0


@dataclass
class HumanSimulationConfig:
    """Configuration for the long-term human simulation."""

    physiology: Any = None
    disease: Any = None
    drugs: list = field(default_factory=list)
    pbpk_config: Any = None
    pharmacodynamics: dict = field(default_factory=dict)
    total_duration_days: float = 30.0
    dfa_dt_h: float = 1.0
    pbpk_dt_min: float = 1.0
    target_tissue: str = "liver"
    output_time_resolution_h: float = 1.0
    track_fluxes: bool = True
    track_metabolites: bool = True
    track_drug_levels: bool = True
    track_biomarkers: bool = True
    base_model_path: str = ""


@dataclass
class HumanSimulationResult:
    """Complete time-course result of a human simulation."""

    time_h: list[float] = field(default_factory=list)
    drug_concentrations: dict[str, list[float]] = field(default_factory=dict)
    plasma_concentration: list[float] = field(default_factory=list)
    flux_history: list[dict[str, float]] = field(default_factory=list)
    metabolite_pools: dict[str, list[float]] = field(default_factory=dict)
    biomarker_history: dict[str, list[float]] = field(default_factory=dict)
    disease_severity_over_time: list[float] = field(default_factory=list)
    therapeutic_response_time_h: float = -1.0
    toxicity_events: list[dict] = field(default_factory=list)
    auc_plasma: float = 0.0
    time_in_therapeutic_range_fraction: float = 0.0
    overall_efficacy_score: float = 0.0

    def to_dict(self) -> dict:
        """Return a JSON-safe dict of every trajectory and endpoint."""
        data = {f.name: copy.deepcopy(getattr(self, f.name)) for f in fields(self)}
        return dict(json.loads(json.dumps(data, default=float)))

    def save_csv(self, path: str) -> None:
        """Write the scalar time-course table (drugs + biomarkers) as CSV."""
        n = len(self.time_h)
        drug_cols = sorted(self.drug_concentrations)
        bio_cols = sorted(self.biomarker_history)

        def cell(series: list[float], i: int) -> float | str:
            return series[i] if i < len(series) else ""

        with Path(path).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["time_h", "plasma_concentration_um"]
                + [f"drug_{c}_um" for c in drug_cols]
                + [f"biomarker_{c}" for c in bio_cols]
            )
            for i in range(n):
                writer.writerow(
                    [self.time_h[i], cell(self.plasma_concentration, i)]
                    + [cell(self.drug_concentrations[c], i) for c in drug_cols]
                    + [cell(self.biomarker_history[c], i) for c in bio_cols]
                )

    def summary(self) -> str:
        """Human-readable multi-line summary of the simulation outcome."""
        plasma = self.plasma_concentration
        c_max = max(plasma) if plasma else 0.0
        dt = self.time_h[1] - self.time_h[0] if len(self.time_h) >= 2 else 0.0
        t_max = plasma.index(c_max) * dt if plasma else 0.0
        response = (
            f"{self.therapeutic_response_time_h:.1f} h"
            if self.therapeutic_response_time_h >= 0.0
            else "not achieved"
        )
        return "\n".join(
            [
                "Human simulation summary (doc/27)",
                f"  duration:           {self.time_h[-1] if self.time_h else 0.0:.1f} h",
                f"  plasma AUC:         {self.auc_plasma:.2f} uM*h",
                f"  plasma Cmax:        {c_max:.2f} uM (t={t_max:.1f} h)",
                f"  time in therapeutic range: "
                f"{100.0 * self.time_in_therapeutic_range_fraction:.1f}%",
                f"  toxicity events:    {len(self.toxicity_events)}",
                f"  therapeutic response: {response}",
                f"  overall efficacy:   {100.0 * self.overall_efficacy_score:.1f}%",
            ]
        )


class _PBPKEngine:
    """Minimal whole-body PBPK integrator for one drug.

    Six well-stirred compartments (central/plasma + liver, kidney,
    brain, muscle, adipose) coupled by organ blood flows from
    :class:`~helixlang.plugins.human.physiology.HumanPhysiology`::

        V_c dC_c/dt = Sum_i Q_i (C_c - C_i/Kp_i) - k_el*f_ren*C_c + In(t)
        V_i dC_i/dt = Q_i (C_c - C_i/Kp_i) - k_hep*[i==liver]*C_i
        dA_depot/dt = -ka * A_depot ;  In_oral = ka * A_depot / V_c

    Concentrations uM, volumes L, flows L/h.  Tissues exchange total
    concentrations through partition coefficients Kp (adipose
    accumulates lipophilic compounds, Kp growing with LogP, scaled by
    a class-specific tissue-access factor).  Terminal elimination
    follows the labeled half-life (k_el = ln2/t1/2, Rowland & Tozer
    2020 Ch. 3), split between a renal term on plasma and a hepatic
    term on the liver compartment that is mass-conserving at
    partition equilibrium.  Routes (doc/27 section 7.6): oral/sc/im
    first-order depot absorption, iv bolus, zero-order 1 h iv
    infusion, intrathecal bolus into the brain compartment.
    Integration uses scipy RK45 (``max_step = pbpk_dt_min``) when
    available, otherwise fixed-step Euler.
    """

    def __init__(self, drug: Drug, physiology: Any, dt_min: float) -> None:
        self.drug = drug
        self.mw_da = max(drug.molecule.molecular_weight_da, 1.0)
        self.dt_h = max(dt_min, 1e-6) / 60.0
        pcfg = getattr(physiology, "plasma_volume_ml", None)
        self.vc_l = (pcfg / 1000.0) if pcfg else 3.0
        self.flows_l_h: dict[str, float] = {}
        self.volumes_l: dict[str, float] = {}
        organs = getattr(physiology, "organs", {}) or {}
        for tissue in _TISSUES:
            organ = organs.get(tissue)
            if organ is not None:
                self.flows_l_h[tissue] = organ.blood_flow_ml_per_min * 0.06
                self.volumes_l[tissue] = organ.volume_ml / 1000.0
        log_p = drug.molecule.log_p
        access = _tissue_access_factor(drug)
        self.kp: dict[str, float] = {
            t: (max(1.0, 1.0 + 0.25 * log_p) if t == "adipose" else 1.0) * access for t in _TISSUES
        }
        self.k_el_per_h = math.log(2.0) / max(drug.half_life_h, 1e-6)
        self.renal_fraction = min(max(drug.renal_fraction, 0.0), 1.0)
        self.conc_um: dict[str, float] = {"central": 0.0, **{t: 0.0 for t in _TISSUES}}
        self.depot_umol = 0.0
        self.infusion_rate_umol_h = 0.0
        self.infusion_end_h = -1.0
        self.time_h = 0.0
        self._next_dose_h = 0.0

    def apply_due_doses(self, now_h: float) -> None:
        """Administer every dose scheduled in ``(past, now]``."""
        tau = self.drug.dosing_interval_h
        horizon_h = self.drug.duration_days * 24.0
        while self._next_dose_h <= now_h + 1e-9:
            if self._next_dose_h > horizon_h + 1e-9:
                break
            self._administer_once()
            self._next_dose_h += tau if tau > 0.0 else float("inf")

    def _administer_once(self) -> None:
        dose_umol = self.drug.bioavailability * self.drug.dose_mg * 1000.0 / self.mw_da
        route = self.drug.route
        if route in (ORAL, SUBCUTANEOUS, INTRAMUSCULAR):
            self.depot_umol += dose_umol
        elif route == IV:
            self.conc_um["central"] += dose_umol / max(self.vc_l, 1e-9)
        elif route == IV_INFUSION:
            self.infusion_rate_umol_h = dose_umol / _INFUSION_DURATION_H
            self.infusion_end_h = self.time_h + _INFUSION_DURATION_H
        elif route == INTRATHECAL:
            vb = max(self.volumes_l.get("brain", 1.4), 1e-9)
            self.conc_um["brain"] += dose_umol / vb

    def _order(self) -> list[str]:
        return ["central", *_TISSUES]

    def _derivatives(self, t_h: float, y: list[float]) -> list[float]:
        keys = self._order()
        depot = y[-1]
        c = dict(zip(keys, y[:-1], strict=True))
        vc = max(self.vc_l, 1e-9)
        dy: dict[str, float] = {k: 0.0 for k in keys}
        for tissue in _TISSUES:
            q = self.flows_l_h[tissue]
            v = max(self.volumes_l[tissue], 1e-9)
            exchange = q * (c["central"] - c[tissue] / self.kp[tissue])
            dy[tissue] = exchange / v
            dy["central"] -= exchange / vc
        hepatic_k = self.k_el_per_h * (1.0 - self.renal_fraction)
        liver_v = max(self.volumes_l.get("liver", 1.5), 1e-9)
        kp_liver = self.kp.get("liver", 1.0)
        dy["liver"] -= hepatic_k * (vc / liver_v) / kp_liver * c["liver"]
        dy["central"] -= self.k_el_per_h * self.renal_fraction * c["central"]
        infusion = self.infusion_rate_umol_h if t_h < self.infusion_end_h else 0.0
        ka = self.drug.absorption_rate_h
        dy["central"] += infusion / vc + ka * depot / vc
        return [*dy.values(), -ka * depot]

    def advance(self, dt_h: float) -> None:
        """Integrate the compartment ODEs forward by ``dt_h`` hours."""
        t_end = self.time_h + dt_h
        keys = self._order()
        if _HAS_SCIPY:
            y0 = [self.conc_um[k] for k in keys] + [self.depot_umol]
            sol = solve_ivp(
                lambda t, y: self._derivatives(t, [float(v) for v in y]),
                (self.time_h, t_end),
                y0,
                method="RK45",
                max_step=max(self.dt_h, dt_h / 60.0),
                rtol=1e-6,
                atol=1e-9,
            )
            if sol.success and sol.y.shape[1] > 0:
                for i, key in enumerate(keys):
                    self.conc_um[key] = float(sol.y[i, -1])
                self.depot_umol = float(sol.y[-1, -1])
                self.time_h = t_end
                return
        n_sub = max(1, int(math.ceil(dt_h / self.dt_h)))
        h = dt_h / n_sub
        for _ in range(n_sub):
            derivs = self._derivatives(
                self.time_h, [self.conc_um[k] for k in keys] + [self.depot_umol]
            )
            for i, key in enumerate(keys):
                self.conc_um[key] = max(0.0, self.conc_um[key] + derivs[i] * h)
            self.depot_umol = max(0.0, self.depot_umol + derivs[-1] * h)
            self.time_h += h
        self.time_h = t_end

    def target_concentration(self, tissue: str) -> float:
        """Drug concentration (uM) in ``tissue`` ('plasma' = central)."""
        if tissue in ("plasma", "central", ""):
            return self.conc_um["central"]
        return self.conc_um.get(tissue, self.conc_um["central"])

    def concentrations(self) -> dict[str, float]:
        """Snapshot copy of every compartment concentration (uM)."""
        return dict(self.conc_um)


class HumanSimulation:
    """Long-term integration of PBPK + PD + disease-perturbed dFBA.

    Each outer step (default 1 h): scheduled doses are administered
    and every drug's PBPK system advanced; target-tissue concentrations
    become Hill PD strengths; corrected flux bounds reach the LP
    through :attr:`~helixlang.plugins.runtime.metabolism.DynamicFluxBalance.
    bound_override`; the batch integrates one step; pools accumulate
    from the flux solution; blood substrates are restored (perfusion);
    biomarkers relax toward normal proportional to aggregate PD
    strength (full strength -> 72 h half-life); toxicity is checked
    against the PD threshold (or 10x Kd, doc/27 section 9.4); and the
    therapeutic endpoint (>= 50% biomarker normalization) evaluated.
    """

    def __init__(self, config: HumanSimulationConfig) -> None:
        self.config = config
        if config.pharmacodynamics is None:
            config.pharmacodynamics = {}
        self.physiology = config.physiology or create_default_physiology()
        for drug in config.drugs:
            problems = drug.validate()
            if problems:
                name = drug.molecule.name
                raise ValueError(f"invalid drug {name!r}: {'; '.join(problems)}")
        if config.dfa_dt_h <= 0.0:
            raise ValueError("dfa_dt_h must be positive")
        if config.pbpk_dt_min <= 0.0:
            raise ValueError("pbpk_dt_min must be positive")
        self.model = self._load_base_model()
        self._healthy_bounds: dict[str, tuple[float, float]] = {
            rid: (rxn.lower_bound, rxn.upper_bound) for rid, rxn in self.model.reactions.items()
        }
        self._apply_disease(self.model)
        self._pool_seeds: dict[str, float] = (
            getattr(self.model, "metabolite_pool_initials", None) or {}
        )
        self.dfba = self._build_dfba()
        self.pools = MetabolitePool(
            self.model,
            config=MetabolitePoolConfig(dt_h=config.dfa_dt_h, dilution=False),
            initial=dict(self._pool_seeds),
        )
        self.engines = [
            _PBPKEngine(drug, self.physiology, config.pbpk_dt_min) for drug in config.drugs
        ]
        self._pd_bound_overrides: dict[str, float] = {}

    def _load_base_model(self) -> Any:
        """Load the metabolic engine: human GEM JSON or E. coli core.

        When ``base_model_path`` is set, a failed load is an explicit error
        (doc/36 §3ξ.3) — the requested model is never silently replaced with the
        E. coli core unless the program opts into reduced fidelity.  With no
        path, the E. coli core is the documented default engine.
        """
        path = self.config.base_model_path
        if path:
            try:
                model = load_model_from_json(Path(path))
                self._ensure_biomass(model)
                return model
            except (OSError, ValueError, KeyError, BioError) as exc:
                from helixlang.core import fidelity
                if not fidelity.opt_in("--low-fidelity"):
                    from helixlang.core.errors import ModelMissingError
                    raise ModelMissingError(
                        f"base model from {path}", "fba",
                        detail=f"load failed: {exc}",
                    ) from exc
        return copy.deepcopy(ECOLI_CORE_MODEL)

    @staticmethod
    def _ensure_biomass(model: Any) -> None:
        """Auto-detect the biomass reaction when the GEM omits it."""
        if getattr(model, "biomass_reaction", None):
            return
        for rxn_id, rxn in model.reactions.items():
            name = getattr(rxn, "name", "") or ""
            if "biomass" in rxn_id.casefold() or "biomass" in name.casefold():
                model.set_biomass(rxn_id)
                return

    def _apply_disease(self, model: Any) -> None:
        """Apply the configured disease state onto the private model."""
        disease = self.config.disease
        if disease is None:
            return
        diseased = apply_disease_state(model, disease)
        model.reactions = diseased.reactions
        model.metabolites = diseased.metabolites
        model.biomass_reaction = diseased.biomass_reaction
        model.genes = diseased.genes

    def _build_dfba(self) -> DynamicFluxBalance | None:
        """Build the dFBA batch (None when the model cannot drive an LP)."""
        try:
            fba_cfg = DynamicFBAConfig(
                dt_h=self.config.dfa_dt_h,
                initial_biomass_gdw=0.5,
                initial_glucose_mm=_BLOOD_GLUCOSE_MM,
                initial_oxygen_mm=_BLOOD_OXYGEN_MM,
                max_biomass_gdw=0.55,
            )

            def override(time_h: float, engine: DynamicFluxBalance) -> dict[str, float]:
                del time_h, engine
                return self._pd_bound_overrides

            return DynamicFluxBalance(self.model, config=fba_cfg, bound_override=override)
        except Exception:
            return None

    def _compute_pd_effect(
        self,
        drug_name: str,
        concentrations: dict[str, float],
        time_h: float,
    ) -> dict[str, tuple[str, float]]:
        """Map a drug's tissue concentrations to per-reaction effect sizes.

        Returns ``{reaction_id: (effect_type, fractional_strength)}``
        with the Hill fractional effect
        ``E0 + (Emax - E0) * C^n / (EC50^n + C^n)`` at the target-tissue
        concentration (Rowland & Tozer 2020).  Targets absent from the
        proxy model (human enzymes such as PAH) are skipped here;
        biomarker normalization uses the raw specification separately.
        """
        del time_h
        pd = self._pd_for(drug_name)
        if pd is None:
            return {}
        target = concentrations.get(
            self.config.target_tissue,
            concentrations.get("central", 0.0),
        )
        effects: dict[str, tuple[str, float]] = {}
        for effect in pd.effects:
            rxn_id = effect.target_reaction
            if rxn_id not in self.model.reactions:
                continue
            frac = _hill_response(target, effect.ec50_um, effect.hill_coefficient)
            strength = effect.baseline_effect + effect.emax * (1.0 - effect.baseline_effect) * frac
            strength = max(0.0, min(strength, 1.0))
            _, prev_strength = effects.get(rxn_id, (effect.effect_type, 0.0))
            if effect.effect_type == "activation":
                effects[rxn_id] = ("activation", max(prev_strength, strength))
            else:
                combined = 1.0 - (1.0 - strength) * (1.0 - prev_strength)
                effects[rxn_id] = ("inhibition", combined)
        return effects

    def _apply_pd_bounds(
        self, hour: float, concentrations_by_drug: dict[str, dict[str, float]]
    ) -> None:
        """Write PD-corrected upper bounds for the next dFBA solve.

        Inhibition scales the diseased bound down by the fractional
        strength; activation restores capacity linearly toward the
        healthy pre-disease bound (doc/27 section 8.4).
        """
        self._pd_bound_overrides = {}
        for name, concs in concentrations_by_drug.items():
            for rxn_id, (kind, strength) in self._compute_pd_effect(name, concs, hour).items():
                rxn = self.model.reactions[rxn_id]
                lb, ub_now = rxn.lower_bound, rxn.upper_bound
                _, ub_healthy = self._healthy_bounds.get(rxn_id, (lb, ub_now))
                if kind == "inhibition":
                    new_ub = ub_now * (1.0 - strength)
                else:
                    new_ub = ub_now + strength * max(ub_healthy - ub_now, 0.0)
                self._pd_bound_overrides[rxn_id] = max(new_ub, lb, 0.0)

    def _toxicity_check(self, drug: Drug, conc: float, hour: float) -> dict | None:
        """Return a toxicity event dict when ``conc`` breaches thresholds.

        Threshold: the drug's PD ``toxicity_concentration_um`` when
        configured, otherwise 10x binding affinity Kd (doc/27 section
        9.4); severity grades moderate (>1x) / severe (>2x).
        """
        pd = self._pd_for(drug.molecule.name)
        if pd is not None:
            threshold = float(pd.toxicity_concentration_um)
        else:
            threshold = 10.0 * max(drug.molecule.binding_affinity_kd_um, 1.0)
        if conc <= threshold:
            return None
        severity = "severe" if conc > 2.0 * threshold else "moderate"
        return {
            "time_h": float(hour),
            "drug": drug.molecule.name,
            "tissue": self.config.target_tissue,
            "concentration_um": float(conc),
            "threshold_um": float(threshold),
            "severity": severity,
        }

    def _init_biomarkers(self) -> dict[str, dict]:
        """Build per-biomarker state (value anchors + pool key)."""
        state: dict[str, dict] = {}
        disease = self.config.disease
        if disease is None:
            return state

        def pool_key_of(biomarker_id: str) -> str | None:
            """Map a disease biomarker id onto a proxy-model pool name."""
            pools = self.pools.pools
            if biomarker_id in pools:
                return biomarker_id
            alias = _BIOMARKER_ALIASES.get(biomarker_id.lower())
            if alias and alias in pools:
                return alias
            wanted = biomarker_id.casefold()
            return next((k for k in pools if k.casefold() == wanted), None)

        for mp in disease.metabolite_perturbations:
            pathological = mp.initial_concentration_mm
            seed = self._pool_seeds.get(mp.metabolite_id, pathological)
            state[mp.metabolite_id] = {
                "value": seed,
                "normal": mp.normal_concentration_mm,
                "pathological": pathological,
                "pool_key": pool_key_of(mp.metabolite_id),
            }
        return state

    def _update_biomarkers(self, biomarkers: dict[str, dict], strength: float, dt_h: float) -> None:
        """Advance biomarkers one step (pool-following or exponential).

        Biomarkers with a proxy-model pool follow the pool trajectory
        relative to its start value; the rest relax exponentially
        toward normal at a rate proportional to aggregate PD strength.
        """
        k_max = math.log(2.0) / _BIOMARKER_HALF_LIFE_H
        for info in biomarkers.values():
            pool_key = info["pool_key"]
            start = self._pool_start.get(pool_key, 0.0) if pool_key else 0.0
            if pool_key and abs(start) > 1e-12:
                info["value"] *= self.pools.pools[pool_key] / start
                continue
            relaxed = info["normal"] + (info["value"] - info["normal"]) * math.exp(
                -k_max * strength * dt_h
            )
            cap = max(info["pathological"], info["normal"])
            info["value"] = min(max(relaxed, 0.0), cap)

    def run(self) -> HumanSimulationResult:
        """Run the full simulation and return the assembled result."""
        cfg = self.config
        result = HumanSimulationResult()
        total_steps = max(1, int(round(cfg.total_duration_days * 24.0 / cfg.dfa_dt_h)))
        out_every = max(1, int(round(cfg.output_time_resolution_h / cfg.dfa_dt_h)))
        biomarkers = self._init_biomarkers()
        self._pool_start = dict(self.pools.pools)
        self._record(result, 0.0, biomarkers, {}, {}, 0)
        responses_seen: set[str] = set()
        windowed = [(drug, self._pd_for(drug.molecule.name)) for drug in cfg.drugs]
        in_range_count = 0
        samples_with_windows = 0

        for step in range(total_steps):
            hour = step * cfg.dfa_dt_h
            concentrations_by_drug: dict[str, dict[str, float]] = {}
            for engine in self.engines:
                engine.apply_due_doses(hour)
                engine.advance(cfg.dfa_dt_h)
                concentrations_by_drug[engine.drug.molecule.name] = engine.concentrations()
            self._apply_pd_bounds(hour, concentrations_by_drug)
            strength = self._aggregate_activation(concentrations_by_drug)
            fluxes: dict[str, float] = {}
            growth = 0.0
            if self.dfba is not None:
                state = self.dfba.step(cfg.dfa_dt_h)
                fluxes = dict(self.dfba.last_fluxes)
                growth = float(state.get("growth_rate", 0.0))
                self.pools.integrate(fluxes, growth, cfg.dfa_dt_h)
                self._perfuse()
            self._update_biomarkers(biomarkers, strength, cfg.dfa_dt_h)
            self._check_toxicities(result, windowed, hour)

            if (step + 1) % out_every == 0:
                idx = len(result.time_h)
                self._record(
                    result, hour + cfg.dfa_dt_h, biomarkers, concentrations_by_drug, fluxes, idx
                )
                counted = self._count_therapeutic_range(windowed)
                samples_with_windows += counted[0]
                in_range_count += counted[1]
                self._check_response(result, biomarkers, responses_seen, hour + cfg.dfa_dt_h)

        self._finalize(result, biomarkers, in_range_count, samples_with_windows)
        return result

    def _pd_for(self, drug_name: str) -> Any:
        """Return the PD model for ``drug_name`` (case-insensitive)."""
        pd = self.config.pharmacodynamics.get(drug_name)
        if pd is not None:
            return pd
        wanted = drug_name.casefold()
        for key, value in self.config.pharmacodynamics.items():
            if key.casefold() == wanted:
                return value
        return None

    def _engine_for(self, drug: Drug) -> _PBPKEngine:
        """Return the PBPK engine of ``drug`` (zero-level dummy if none)."""
        for engine in self.engines:
            if engine.drug is drug:
                return engine
        return _PBPKEngine(drug, self.physiology, self.config.pbpk_dt_min)

    def _aggregate_activation(self, concentrations_by_drug: dict[str, dict[str, float]]) -> float:
        """Total therapeutic PD strength across drugs (clamped [0, 1]).

        Evaluated on the raw PD specification so targets absent from
        the proxy model still drive biomarker normalization; both
        activation (restoring clearance) and inhibition (blocking a
        pathological flux such as aerobic glycolysis) count toward
        normalization (doc/27 section 8.5).
        """
        total = 0.0
        for name, concs in concentrations_by_drug.items():
            pd = self._pd_for(name)
            if pd is None:
                continue
            target = concs.get(
                self.config.target_tissue,
                concs.get("central", 0.0),
            )
            for effect in pd.effects:
                frac = _hill_response(target, effect.ec50_um, effect.hill_coefficient)
                strength = (
                    effect.baseline_effect + effect.emax * (1.0 - effect.baseline_effect) * frac
                )
                total += max(0.0, min(strength, 1.0))
        return min(total, 1.0)

    def _perfuse(self) -> None:
        """Restore blood substrates and wash out secreted byproducts."""
        dfba = self.dfba
        assert dfba is not None
        dfba.glucose_mm = _BLOOD_GLUCOSE_MM
        if "oxygen" in dfba.byproducts_mm:
            dfba.byproducts_mm["oxygen"] = _BLOOD_OXYGEN_MM
        washout = math.exp(-_POOL_WASHOUT_PER_H * self.config.dfa_dt_h)
        for pool in dfba.byproducts_mm:
            if pool != "oxygen":
                dfba.byproducts_mm[pool] *= washout

    def _normalization(self, info: dict) -> float:
        """Fractional biomarker movement from pathological toward normal."""
        span = info["pathological"] - info["normal"]
        if abs(span) <= 1e-12:
            return 0.0
        moved = (info["pathological"] - info["value"]) / span
        return float(max(0.0, min(1.0, moved)))

    def _check_toxicities(
        self,
        result: HumanSimulationResult,
        windowed: list[tuple[Drug, Any]],
        hour: float,
    ) -> None:
        """Log the first toxicity excursion of each drug into result.

        The worse of plasma and target-tissue exposure is evaluated
        against the threshold; only the rising edge of the first
        excursion per drug is recorded.
        """
        for drug, _ in windowed:
            engine = self._engine_for(drug)
            tissue_c = engine.target_concentration(self.config.target_tissue)
            plasma_c = engine.conc_um["central"]
            if plasma_c > tissue_c:
                conc, organ = plasma_c, "plasma"
            else:
                conc, organ = tissue_c, self.config.target_tissue
            event = self._toxicity_check(drug, conc, hour)
            already = any(e["drug"] == drug.molecule.name for e in result.toxicity_events)
            if event is not None and not already:
                event["tissue"] = organ
                result.toxicity_events.append(event)

    def _count_therapeutic_range(self, windowed: list[tuple[Drug, Any]]) -> tuple[int, int]:
        """Count (samples_with_windows, all-drugs-in-window) at a snapshot."""
        has_window = False
        all_in_range = True
        for drug, pd in windowed:
            if pd is None:
                continue
            lo, hi = pd.therapeutic_window
            has_window = True
            plasma = self._engine_for(drug).conc_um["central"]
            if not lo <= plasma <= hi:
                all_in_range = False
        return (1, 1) if has_window and all_in_range else ((1, 0) if has_window else (0, 0))

    def _check_response(
        self,
        result: HumanSimulationResult,
        biomarkers: dict[str, dict],
        responses_seen: set[str],
        time_h: float,
    ) -> None:
        """Record the first time all tracked biomarkers normalize >= 50%."""
        if not biomarkers:
            return
        marker = "+".join(sorted(biomarkers))
        normalized = all(
            self._normalization(info) >= _THERAPEUTIC_RESPONSE_THRESHOLD
            for info in biomarkers.values()
        )
        if normalized and marker not in responses_seen:
            responses_seen.add(marker)
            if result.therapeutic_response_time_h < 0.0:
                result.therapeutic_response_time_h = time_h

    def _record(
        self,
        result: HumanSimulationResult,
        time_h: float,
        biomarkers: dict[str, dict],
        concentrations_by_drug: dict[str, dict[str, float]],
        fluxes: dict[str, float],
        idx: int,
    ) -> None:
        """Append one output-resolution snapshot to the result."""
        result.time_h.append(float(time_h))
        plasma = sum(concs.get("central", 0.0) for concs in concentrations_by_drug.values())
        result.plasma_concentration.append(plasma)
        for engine in self.engines:
            name = engine.drug.molecule.name
            concs = concentrations_by_drug.get(name, {})
            value = concs.get(self.config.target_tissue, concs.get("central", 0.0))
            series = result.drug_concentrations.setdefault(name, [])
            series.extend([0.0] * (idx - len(series)))
            series.append(float(value))
        severity = getattr(self.config.disease, "severity", 0.0)
        result.disease_severity_over_time.append(float(severity))
        if self.config.track_fluxes and fluxes:
            result.flux_history.append(dict(fluxes))
        if self.config.track_metabolites:
            for met, value in self.pools.pools.items():
                result.metabolite_pools.setdefault(met, []).append(value)
        if self.config.track_biomarkers:
            for name, info in biomarkers.items():
                history = result.biomarker_history.setdefault(name, [])
                pad = history[-1] if history else info["value"]
                history.extend([pad] * (idx - len(history)))
                history.append(info["value"])

    def _finalize(
        self,
        result: HumanSimulationResult,
        biomarkers: dict[str, dict],
        in_range_count: int,
        samples_with_windows: int,
    ) -> None:
        """Compute AUC, therapeutic-range fraction, and efficacy score."""
        dt = self.config.output_time_resolution_h if len(result.time_h) >= 2 else 1.0
        result.auc_plasma = _trapz(result.plasma_concentration, dt)
        result.time_in_therapeutic_range_fraction = (
            in_range_count / samples_with_windows if samples_with_windows > 0 else 0.0
        )
        if biomarkers:
            norm = sum(self._normalization(info) for info in biomarkers.values()) / len(biomarkers)
        else:
            norm = result.time_in_therapeutic_range_fraction
        n_opportunities = max(1, len(result.time_h) * max(1, len(self.engines)))
        toxicity_frac = min(len(result.toxicity_events) / n_opportunities, 1.0)
        result.overall_efficacy_score = max(
            0.0,
            min(
                1.0,
                0.5 * norm
                + 0.3 * result.time_in_therapeutic_range_fraction
                + 0.2 * (1.0 - toxicity_frac),
            ),
        )


__all__ = [
    "HumanSimulation",
    "HumanSimulationConfig",
    "HumanSimulationResult",
    "PBPKConfig",
    "PDEffect",
    "Pharmacodynamics",
]
