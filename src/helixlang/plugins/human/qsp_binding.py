"""QSP-style pharmacodynamic binding models (doc/31 §2.5, §5.1 #4).

Extends the Hill-only PD primitive (human/pharmacodynamics.py) with
mechanistic binding models required for biologics, immunotherapies, and
drugs where simple Emax/Hill fails:

1. **Mass-action receptor-ligand binding** — drug + target ⇌ complex
   with measured Kd; occupancy drives efficacy via an Emax transfer
   function.  Appropriate for small-molecule receptor agonists/
   antagonists with known binding affinity.

2. **Target-mediated drug disposition (TMDD)** — ligand binding saturates
   target, which is simultaneously internalised and turned over.  The
   quasi-steady-state (QSS) approximation reduces to 2 ODEs (free drug
   + total target).  Essential for monoclonal antibodies (rituximab,
   trastuzumab) where target consumption dominates clearance at low
   concentrations.

3. **Competitive antagonist binding** — two ligands compete for the same
   receptor site.  Classical Schild-shift dynamics: antagonist shifts the
   agonist dose-response curve rightward by factor (1+[A]/Ki).

All models produce a `binding_fraction` ∈ [0,1] that can be fed into
the existing PD multiplier pathway in VirtualPatient, replacing the
bare Hill equation when mechanistic detail is available.

Module structure:
    MassActionBinding     simple receptor-ligand occupancy
    TMDDBinding           target-mediated drug disposition
    CompetitiveBinding    Schild-shift competitive antagonism
    QSPBindingSystem      facade managing multiple binding models
    create_qsp_binding    factory

References:
- Mager DE, J Pharmacokinet Pharmacodyn 2006 (TMDD review)
- Gibiansky L, Gibiansky E, J Pharmacokinet Pharmacodyn 2014 (QSS TMDD)
- Schild HO, Br J Pharmacol 1949 (competitive antagonism)
"""
from __future__ import annotations

from dataclasses import dataclass, field

__all__ = [
    "MassActionBinding",
    "TMDDBinding",
    "CompetitiveBinding",
    "QSPBindingSystem",
    "create_qsp_binding",
]


# ============================================================================
# Mass-Action Receptor-Ligand Binding
# ============================================================================


@dataclass
class MassActionBinding:
    """Simple mass-action receptor-ligand binding.

    drug + receptor ⇌ drug:receptor  (Kd = [D][R]/[DR])

    Binding fraction = [DR] / R_total = [D] / (Kd + [D])

    Parameters:
        kd_nM: dissociation constant (nM)
        target_receptor_total: total receptor concentration (nM)
        emax: maximum effect at full occupancy (0-1)
        baseline: baseline effect without drug (0-1)
    """

    kd_nM: float = 10.0
    target_receptor_total: float = 100.0
    emax: float = 1.0
    baseline: float = 0.0
    drug_conc_nM: float = 0.0

    def compute_occupancy(self, drug_conc_nM: float | None = None) -> float:
        """Compute receptor occupancy [DR]/R_total."""
        c = drug_conc_nM if drug_conc_nM is not None else self.drug_conc_nM
        return c / (self.kd_nM + c + 1e-12)

    def compute_effect(self, drug_conc_nM: float | None = None) -> float:
        """Compute normalized effect (0-1)."""
        occupancy = self.compute_occupancy(drug_conc_nM)
        return self.baseline + self.emax * occupancy


# ============================================================================
# Target-Mediated Drug Disposition (TMDD)
# ============================================================================


@dataclass
class TMDDBinding:
    """Target-mediated drug disposition with QSS approximation.

    Full TMDD system (Mager & Jusko 2001):
        dC/dt = -CL*C/V - kon*C*R + koff*RC - internalisation
        dR/dt = ksyn - kdeg*R - kon*C*R + koff*RC
        dRC/dt = kon*C*R - koff*RC - kint*RC

    QSS approximation (Gibiansky 2014): [DR] ≈ R_total*C/(Kss + C)
    where Kss = (koff + kint) / kon

    States tracked:
        C_free: free drug concentration (nM)
        R_total: total target (bound + free, nM)
    """

    # --- Binding parameters ---
    kss_nM: float = 5.0         # quasi-steady-state constant
    kon: float = 0.1            # association rate (1/(nM·h))
    koff: float = 0.05          # dissociation rate (1/h)
    kint: float = 0.02          # internalisation rate of complex (1/h)

    # --- Target turnover ---
    ksyn: float = 0.5           # target synthesis rate (nM/h)
    kdeg: float = 0.1           # target degradation rate (1/h)

    # --- Drug disposition ---
    cl_non_specific: float = 0.5  # non-specific clearance (L/h)
    volume: float = 50.0          # volume of distribution (L)

    # --- State ---
    c_free_nM: float = 0.0
    r_total_nM: float = 100.0
    rc_complex_nM: float = 0.0

    # --- Effect ---
    emax: float = 1.0
    baseline: float = 0.0

    def step(self, dt_h: float, drug_input_rate_nM_h: float = 0.0) -> None:
        """Advance one hour with QSS TMDD.

        Args:
            dt_h: time step (hours)
            drug_input_rate_nM_h: dosing input rate (nM/h)
        """
        C = self.c_free_nM
        R = self.r_total_nM
        RC = self.rc_complex_nM

        # QSS approximation for bound complex
        Kss = self.kss_nM
        RC_qss = R * C / (Kss + C + 1e-12)

        # Free drug dynamics
        dC = (drug_input_rate_nM_h / self.volume
              - self.cl_non_specific * C / self.volume
              - self.kint * RC_qss)

        # Total target dynamics (synthesis - degradation - internalisation)
        dR = self.ksyn - self.kdeg * R - self.kint * RC_qss

        # Complex dynamics
        dRC = self.kint * RC_qss - self.kint * RC

        self.c_free_nM = max(0.0, C + dt_h * dC)
        self.r_total_nM = max(0.0, R + dt_h * dR)
        self.rc_complex_nM = max(0.0, RC + dt_h * dRC)

    def compute_occupancy(self) -> float:
        """Current receptor occupancy."""
        if self.r_total_nM < 1e-12:
            return 0.0
        return self.rc_complex_nM / self.r_total_nM

    def compute_effect(self) -> float:
        """Normalized effect (0-1)."""
        return self.baseline + self.emax * self.compute_occupancy()


# ============================================================================
# Competitive Antagonist Binding (Schild)
# ============================================================================


@dataclass
class CompetitiveBinding:
    """Competitive antagonism with Schild-shift dynamics.

    Two ligands (agonist + antagonist) compete for the same receptor:
        agonist_effect = E0 + Emax * [A] / (alpha + [A])
        where alpha = Kd_agonist * (1 + [antagonist]/Ki_antagonist)

    The antagonist shifts the agonist dose-response curve rightward
    by factor (1 + [antagonist]/Ki), reducing effect at any given
    agonist concentration.

    Parameters:
        kd_agonist_nM: agonist Kd
        ki_antagonist_nM: antagonist Ki
        emax: maximum agonist effect
    """

    kd_agonist_nM: float = 10.0
    ki_antagonist_nM: float = 5.0
    emax: float = 1.0
    baseline: float = 0.0

    def compute_effect(
        self,
        agonist_conc_nM: float,
        antagonist_conc_nM: float,
    ) -> float:
        """Compute effect with competitive antagonism."""
        # Schild shift factor
        schild_factor = 1.0 + antagonist_conc_nM / (self.ki_antagonist_nM + 1e-12)
        # Effective agonist Kd (shifted right)
        effective_kd = self.kd_agonist_nM * schild_factor
        # Effect
        occupancy = agonist_conc_nM / (effective_kd + agonist_conc_nM + 1e-12)
        return self.baseline + self.emax * occupancy

    def compute_schild_shift(self, antagonist_conc_nM: float) -> float:
        """Compute dose-ratio (Schild shift)."""
        return 1.0 + antagonist_conc_nM / (self.ki_antagonist_nM + 1e-12)


# ============================================================================
# QSP Binding System Facade
# ============================================================================


@dataclass
class QSPBindingModel:
    """A named binding model with its parameters and state."""
    name: str
    kind: str  # "mass_action", "tmdd", "competitive"
    mass_action: MassActionBinding | None = None
    tmdd: TMDDBinding | None = None
    competitive: CompetitiveBinding | None = None

    def compute_effect(self) -> float:
        """Compute current effect based on model type."""
        if self.mass_action is not None:
            return self.mass_action.compute_effect()
        elif self.tmdd is not None:
            return self.tmdd.compute_effect()
        elif self.competitive is not None:
            return self.competitive.compute_effect(0.0, 0.0)
        return 0.0


@dataclass
class QSPBindingSystem:
    """Facade managing multiple QSP binding models.

    In the VirtualPatient loop:
    1. Set drug concentrations for all models
    2. Step TMDD models
    3. Query effects for PD multiplier computation
    """

    models: dict[str, QSPBindingModel] = field(default_factory=dict)

    def add_mass_action(
        self,
        name: str,
        kd_nM: float,
        emax: float = 1.0,
        baseline: float = 0.0,
    ) -> None:
        """Add a mass-action binding model."""
        self.models[name] = QSPBindingModel(
            name=name,
            kind="mass_action",
            mass_action=MassActionBinding(
                kd_nM=kd_nM,
                emax=emax,
                baseline=baseline,
            ),
        )

    def add_tmdd(
        self,
        name: str,
        kss_nM: float = 5.0,
        emax: float = 1.0,
        baseline: float = 0.0,
    ) -> None:
        """Add a TMDD binding model."""
        self.models[name] = QSPBindingModel(
            name=name,
            kind="tmdd",
            tmdd=TMDDBinding(
                kss_nM=kss_nM,
                emax=emax,
                baseline=baseline,
            ),
        )

    def add_competitive(
        self,
        name: str,
        kd_agonist_nM: float = 10.0,
        ki_antagonist_nM: float = 5.0,
        emax: float = 1.0,
    ) -> None:
        """Add a competitive binding model."""
        self.models[name] = QSPBindingModel(
            name=name,
            kind="competitive",
            competitive=CompetitiveBinding(
                kd_agonist_nM=kd_agonist_nM,
                ki_antagonist_nM=ki_antagonist_nM,
                emax=emax,
            ),
        )

    def set_drug_concentration(self, name: str, conc_nM: float) -> None:
        """Set free drug concentration for a model."""
        model = self.models.get(name)
        if model is None:
            return
        if model.mass_action is not None:
            model.mass_action.drug_conc_nM = conc_nM
        elif model.tmdd is not None:
            model.tmdd.c_free_nM = conc_nM

    def step(self, dt_h: float) -> None:
        """Advance all TMDD models one time step."""
        for model in self.models.values():
            if model.tmdd is not None:
                model.tmdd.step(dt_h)

    def get_effect(self, name: str) -> float:
        """Get current effect for a named model."""
        model = self.models.get(name)
        if model is None:
            return 0.0
        return model.compute_effect()

    def get_all_effects(self) -> dict[str, float]:
        """Get effects for all models."""
        return {name: model.compute_effect()
                for name, model in self.models.items()}


def create_qsp_binding() -> QSPBindingSystem:
    """Factory with default binding models for common drugs."""
    sys = QSPBindingSystem()
    # Example: trastuzumab (HER2 TMDD)
    sys.add_tmdd("trastuzumab", kss_nM=2.0, emax=0.9)
    # Example: imatinib (BCR-ABL mass action)
    sys.add_mass_action("imatinib", kd_nM=1.0, emax=0.85)
    return sys
