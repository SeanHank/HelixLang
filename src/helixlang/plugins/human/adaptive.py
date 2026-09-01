"""Adaptive immunity + vaccination module (doc/40 Phase B).

Implements gaps G2 / G3 / G7 / G12 of doc/40 on top of the Phase-A innate
core in :mod:`helixlang.plugins.human.immune`:

- **G2** naive → effector → memory **CD4/CD8** and **B** cell pools.
- **G3** antibody dynamics with **IgM→IgG class-switching** and biphasic
  (**short-lived → long-lived plasma cell**) waning.
- **G7** APC/MHC antigen-presentation → T-cell **priming delay**
  compartment (dendritic-cell maturation rather than an instantaneous
  activation threshold).
- **G12** **vaccination stimulus** (antigen dose, prime/boost schedule)
  that drives the same APC → T-cell → B-cell → antibody machinery as a
  live infection.

Formulation follows the within-host influenza model of Pawelek et al.
(*PLoS Comput Biol* 8:e1002588, 2012 — L5: dual effector arms, delayed
CD8/antibody actions, memory) and the consensus two-dose vaccine antibody
chain of *Front. Immunol.* 16:1596518 (2025 — L8: sigmoid rise → peak
~2–4 wk → biphasic waning, memory B ↔ long-lived plasma cells).

Design contract (doc/40 Phase B):
- Additive and **backward-compatible**: the module is inert unless
  antigen is present. At baseline (no infection, no vaccination) all
  adaptive pools rest at their naive baseline and antibody stays at the
  baseline titer. Existing ``InnateImmuneModel`` output channels are
  untouched.
- Deterministic: no RNG; every rate is a published/formula constant, so
  cohort and repeated runs are bit-identical (doc/39 §5).
- Vectorizable: the integrator is a plain Euler loop over a fixed-length
  state vector, so the same numeric core backs `cohort_adaptive_step`.

State (units: cell pools are relative units scaled to the naive pool,
antibodies in IU/mL, times in hours):

    naive_cd4, naive_cd8, naive_b        naive pools (relative, ~1.0)
    effector_cd4, effector_cd8,           effector T cells (relative)
    effector_b,                           effector B / plasmablasts
    plasma_short, plasma_long,            short/long-lived plasma cells
    memory_b, memory_cd4, memory_cd8      memory pools
    igm_titer, igg_titer,                 circulating antibody (IU/mL)
    apc_primed                            matured (antigen-presenting) DC count
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

try:
    import numpy as _np
    _HAS_NUMPY = True
except ImportError:  # pragma: no cover - numpy is a project dependency
    _np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

# ---------------------------------------------------------------------------
# Published rate constants and anchors (L5 Pawelek 2012; L8 2025 consensus)
# ---------------------------------------------------------------------------

# APC maturation / T-cell priming delay (G7): mDC take ~24-48 h to present
# antigen and prime naive T cells; modelled as a first-order maturation with
# a mean delay tau ~ 18 h (so priming rises over ~1-2 days, not instantly).
_APC_MATURATION_TAU_H = 18.0
_APC_BASELINE_DC = 1.0

# T-cell clonal expansion/differentiation (G2). Doubling times ~8-12 h for
# effector expansion; first-order contraction with ~21 d (memory) half-life.
_CD4_PROLIF = 0.06      # /h maximal naive->effector priming coefficient
_CD8_PROLIF = 0.07
_B_PROLIF = 0.07
_K_APC_HALF = 0.5       # APC signal for half-maximal priming (relative)
_HILL_N = 2.0

_T_EFFECTOR_HALF_LIFE_H = 12.0        # effector contraction (short-lived)
_T_MEMORY_HALF_LIFE_H = 24.0 * 24.0   # memory persistence (~24 d)
_TC_DEATH_HALF_LIFE_H = 2.5 * 24.0    # plasma cell half-life

# Antibody class-switching + plasma-cell dynamics (G3/L8).
_IGM_TURNOVER_HALF_LIFE_H = 7.0            # IgM waning (~7 d)
_IGG_TURNOVER_SHORT_HALF_LIFE_H = 21.0 * 24.0   # short-lived PC -> Ab
_IGG_TURNOVER_LONG_HALF_LIFE_H = 150.0 * 24.0   # long-lived PC (months)
_ANTIBODY_PER_PC_H = 0.35     # antibody production per plasma cell per h
_CLASS_SWITCH_RATE = 0.02       # IgM -> IgG switch rate (/h), L8 chain
_BASELINE_IGG = 10.0            # baseline IgG titer (IU/mL)
_BASELINE_IGM = 0.4

# Effector amplification (naive cells become ~1.0 effector per committed cell,
# keeping the pools bounded); plasma conversion and memory are fractions.
_AMPLIFY = 1.0
_MEMORY_BOOST = 1.5     # memory-driven rechallenge amplification (L8)
_EFFECTOR_TO_PLASMA = 0.6
_PLASMA_TO_MEMORY = 0.12

# Carrying capacity for the expanded pools (relative units, ~10^3 above the
# naive baseline), matching doc/31 §4.5's agent-count budget. Soft logistic
# saturation keeps a chronic-antigen response bounded instead of diverging.
_CAPACITY = 1000.0


def _saturate(x: float) -> float:
    """Soft logistic saturation toward ``_CAPACITY`` (identity below it)."""
    return x / (1.0 + x / _CAPACITY)

# APC priming by a mature DC drives all three arms proportional to the
# antigen signal (unified IL-6/TNF/IFN-independent priming drive).
_ANTIGEN_HALF = 0.3             # antigen level for half-maximal priming
_ANTIGEN_DECAY_TAU_H = 48.0     # vaccine antigen decay time constant (h)

_LN2 = math.log(2.0)


def _hl_rate(half_life_h: float) -> float:
    return _LN2 / half_life_h if half_life_h > 0 else 0.0


# First-order rate for a mean delay tau (maturation/transit), L8 chain-trick.
def _delay_rate(tau_h: float) -> float:
    return 1.0 / tau_h if tau_h > 0 else 1.0


def _hill(x: float, half: float, n: float) -> float:
    xp = x ** n
    hp = half ** n
    return float(xp / (hp + xp))
# ---------------------------------------------------------------------------


@dataclass
class PD1Checkpoint:
    """PD-1 / PD-L1 / PD-L2 + CTLA-4 + LAG-3 checkpoint network (Phase G(b)).

    Ligand load (``pdl1``/``pdl2``, 0..1) from tumour/APC drives PD-1 signalling;
    combination drug occupancies (``anti_pd1``/``anti_ctla4``/``anti_lag3``, 0..1)
    relieve the brake.  ``effective_blockade()`` returns 0..1 in the same
    convention as the legacy G14 ``checkpoint_blockade`` so it is a drop-in.
    """

    # --- PD-1 trafficking --------------------------------------------------
    pd1_surface: float = 1.0
    pd1_internal: float = 0.2
    pd1_pdl1: float = 0.0
    pd1_pdl2: float = 0.0

    # --- ligands (tumour / APC dependent) ----------------------------------
    pdl1: float = 0.0
    pdl2: float = 0.0
    ctla4_ligand: float = 0.0        # CD80/CD86 engagement
    lag3_ligand: float = 0.0         # MHC-II engagement

    # --- ligand setpoints (sustained tumour / APC expression) --------------
    pdl1_setpoint: float = 0.0
    pdl2_setpoint: float = 0.0
    ctla4_ligand_setpoint: float = 0.0
    lag3_ligand_setpoint: float = 0.0

    # Set True once any ligand or combination therapy has been configured;
    # makes the network authoritative (vs. the legacy G14 toggle).
    network_in_use: bool = False

    # --- other checkpoints (expression level, ligand-gated signal) -----------
    ctla4: float = 1.0
    lag3: float = 1.0

    # --- combination therapeutic blockade (0..1 occupancy) ------------------
    anti_pd1: float = 0.0
    anti_ctla4: float = 0.0
    anti_lag3: float = 0.0

    # --- kinetics (first-order /h) ------------------------------------------
    pd1_internalize_rate: float = 0.4       # surface -> internal
    pd1_recycle_rate: float = 0.5           # internal -> surface
    pd1_pdl1_on: float = 0.8
    pd1_pdl2_on: float = 0.7
    complex_off: float = 0.3
    pdl1_turnover: float = 0.2
    pdl2_turnover: float = 0.15
    ctla4_ligand_turnover: float = 0.2
    lag3_ligand_turnover: float = 0.2

    def step(self, dt_h: float) -> None:
        """Advance PD-1 trafficking + ligand binding one hour."""
        # ligands relax toward their sustained setpoint (tumour/APC input)
        self.pdl1 += (self.pdl1_setpoint - self.pdl1) * self.pdl1_turnover * dt_h
        self.pdl2 += (self.pdl2_setpoint - self.pdl2) * self.pdl2_turnover * dt_h
        self.ctla4_ligand += (self.ctla4_ligand_setpoint - self.ctla4_ligand) \
            * self.ctla4_ligand_turnover * dt_h
        self.lag3_ligand += (self.lag3_ligand_setpoint - self.lag3_ligand) \
            * self.lag3_ligand_turnover * dt_h

        # PD-1 surface <-> internal trafficking
        to_int = min(self.pd1_surface, self.pd1_internalize_rate * self.pd1_surface * dt_h)
        self.pd1_surface -= to_int
        self.pd1_internal += to_int
        back = min(self.pd1_internal, self.pd1_recycle_rate * self.pd1_internal * dt_h)
        self.pd1_internal -= back
        self.pd1_surface += back

        # PD-1 ligand binding (only surface PD-1 is signalling-competent)
        available = self.pd1_surface
        on1 = self.pd1_pdl1_on * self.pdl1 * available * dt_h
        on2 = self.pd1_pdl2_on * self.pdl2 * available * dt_h
        off = self.complex_off * dt_h
        # net complex formation (bounded by ligand + surface availability)
        self.pd1_pdl1 = min(available, self.pd1_pdl1 + on1 - off * self.pd1_pdl1)
        self.pd1_pdl2 = min(max(available - self.pd1_pdl1, 0.0),
                            self.pd1_pdl2 + on2 - off * self.pd1_pdl2)
        self.pd1_pdl1 = max(0.0, self.pd1_pdl1)
        self.pd1_pdl2 = max(0.0, self.pd1_pdl2)

    def set_pdl1(self, value: float) -> None:
        v = max(0.0, min(1.0, value))
        self.pdl1 = v
        self.pdl1_setpoint = v
        self.network_in_use = True

    def set_pdl2(self, value: float) -> None:
        v = max(0.0, min(1.0, value))
        self.pdl2 = v
        self.pdl2_setpoint = v
        self.network_in_use = True

    def set_ctla4_ligand(self, value: float) -> None:
        v = max(0.0, min(1.0, value))
        self.ctla4_ligand = v
        self.ctla4_ligand_setpoint = v
        self.network_in_use = True

    def set_lag3_ligand(self, value: float) -> None:
        v = max(0.0, min(1.0, value))
        self.lag3_ligand = v
        self.lag3_ligand_setpoint = v
        self.network_in_use = True

    def pd1_signal(self) -> float:
        """PD-1 signalling strength (bound complexes, blockade-relieved)."""
        bound = self.pd1_pdl1 + self.pd1_pdl2
        relief = 1.0 - min(1.0, max(0.0, self.anti_pd1))
        return max(0.0, bound * relief)

    def ctla4_signal(self) -> float:
        """CTLA-4 signalling, gated on CD80/CD86 engagement (0 at baseline)."""
        engaged = self.ctla4 * self.ctla4_ligand
        relief = 1.0 - min(1.0, max(0.0, self.anti_ctla4))
        return max(0.0, engaged * relief)

    def lag3_signal(self) -> float:
        """LAG-3 signalling, gated on MHC-II engagement (0 at baseline)."""
        engaged = self.lag3 * self.lag3_ligand
        relief = 1.0 - min(1.0, max(0.0, self.anti_lag3))
        return max(0.0, engaged * relief)

    def immune_brake(self) -> float:
        """0..1 immune inhibition (higher = more exhausted / suppressed)."""
        b = (self.pd1_signal() * 0.6
             + self.ctla4_signal() * 0.25
             + self.lag3_signal() * 0.15)
        return min(1.0, max(0.0, b))

    def effective_blockade(self) -> float:
        """0..1 relief of the immune brake (drop-in for legacy G14 toggle)."""
        return 1.0 - self.immune_brake()


@dataclass
class AdaptiveImmuneModel:
    """Adaptive immunity: T/B/plasma pools, antibody, APC priming, vaccine.

    All pools are inert at baseline; the response is driven by an
    ``antigen`` signal (0–1+) raised by live infection ammunition or by a
    vaccine dose (G12). ``step(dt_h, antigen, dose=None)`` integrates one
    hour with optional vaccine administration on that tick.
    """

    # --- T-cell / B-cell pools (relative to naive baseline ~1.0) ---
    naive_cd4: float = 1.0
    naive_cd8: float = 1.0
    naive_b: float = 1.0
    effector_cd4: float = 0.0
    effector_cd8: float = 0.0
    effector_b: float = 0.0
    plasma_short: float = 0.0
    plasma_long: float = 0.0
    memory_b: float = 0.0
    memory_cd4: float = 0.0
    memory_cd8: float = 0.0

    # --- Circulating antibody (IU/mL) ---
    igm_titer: float = _BASELINE_IGM
    igg_titer: float = _BASELINE_IGG

    # --- APC priming (G7) ---
    apc_primed: float = _APC_BASELINE_DC

    # --- Vaccination (G12); inert until ``vaccinate`` administers a dose ---
    antigen_available: float = 0.0      # decaying antigen drive (0-1)
    last_boost_h: float = 0.0

    # --- PD-1/PD-L1 checkpoint blockade (doc/40 Phase D, G14) ---
    # 0 = no blockade (normal immune brake); 1 = full blockade (therapeutic
    # anti-PD-1, e.g. nivolumab/pembrolizumab).  Reduces effector T-cell
    # exhaustion clearance so effectors survive longer and mount a stronger
    # response — modelling the clinical "release the immune brake" phenotype.
    checkpoint_blockade: float = 0.0

    # Full PD-1/PD-L1/PD-L2 + CTLA-4 + LAG-3 network (doc/40 Phase G(b)),
    # replacing the single G14 toggle as the effective brake driver.
    pd1: PD1Checkpoint = field(default_factory=PD1Checkpoint)

    def vaccinate(self, dose: float) -> None:
        """Administer a vaccine dose: a decaying antigen stimulus (G12).

        ``dose`` scales the priming drive; a booster re-challenges the
        memory pools to produce a stronger secondary response (L8).
        """
        self.antigen_available = max(self.antigen_available,
                                     min(1.0, dose))

    def step(self, dt_h: float, antigen: float,
             dose: float | None = None) -> None:
        """Advance adaptive immunity one hour.

        ``antigen`` is the live-infection antigen drive (from the innate
        layer / pathogen signal); ``dose`` (optional) administers a
        vaccine on this tick.
        """
        if dose is not None and dose > 0:
            self.vaccinate(dose)

        # Advance the PD-1/CTLA-4/LAG-3 checkpoint network (Phase G(b)).
        self.pd1.step(dt_h)

        # Antigen drive: live infection plus remaining vaccine antigen,
        # with first-order decay of the vaccine antigen (G12 waning).
        self.antigen_available *= math.exp(-dt_h / _ANTIGEN_DECAY_TAU_H)
        drive = antigen + self.antigen_available

        # --- APC priming delay (G7) ---
        # Mature DC signal rises toward the drive with a first-order delay.
        target_apc = _APC_BASELINE_DC + _hill(drive, _ANTIGEN_HALF, _HILL_N)
        self.apc_primed += dt_h * _delay_rate(_APC_MATURATION_TAU_H) * (
            target_apc - self.apc_primed)

        primed = _hill(self.apc_primed - _APC_BASELINE_DC,
                       _K_APC_HALF, _HILL_N)

        # --- Clonal expansion (G2): naive -> effector, driven by primed APC
        # and gated on naive availability; memory provides anamnesis. ---
        naive_loss_cd4 = min(self.naive_cd4,
                             _CD4_PROLIF * primed * self.naive_cd4 * dt_h)
        naive_loss_cd8 = min(self.naive_cd8,
                             _CD8_PROLIF * primed * self.naive_cd8 * dt_h)
        naive_loss_b = min(self.naive_b,
                           _B_PROLIF * primed * self.naive_b * dt_h)

        self.naive_cd4 -= naive_loss_cd4
        self.naive_cd8 -= naive_loss_cd8
        self.naive_b -= naive_loss_b

        # Memory anamnesis (L8): memory B/C re-expand on rechallenge.
        mem_boost_b = _B_PROLIF * _MEMORY_BOOST * primed * self.memory_b * dt_h
        mem_boost_cd4 = _CD4_PROLIF * _MEMORY_BOOST * primed * self.memory_cd4 * dt_h
        mem_boost_cd8 = _CD8_PROLIF * _MEMORY_BOOST * primed * self.memory_cd8 * dt_h

        # Effector influx = naive priming (1:1) + memory boost (bounded)
        self.effector_cd4 += naive_loss_cd4 * _AMPLIFY + mem_boost_cd4
        self.effector_cd8 += naive_loss_cd8 * _AMPLIFY + mem_boost_cd8
        self.effector_b += naive_loss_b * _AMPLIFY + mem_boost_b

        # --- Plasma cells (G3) ---
        # Effector B -> short-lived plasma cells (bounded fraction); some
        # differentiate to memory B.
        to_plasma = min(self.effector_b,
                        _EFFECTOR_TO_PLASMA * self.effector_b * dt_h)
        to_memory_b = min(self.effector_b - to_plasma,
                          _PLASMA_TO_MEMORY * to_plasma)
        self.effector_b -= (to_plasma + to_memory_b)
        self.plasma_short += to_plasma
        self.memory_b += to_memory_b

        # --- Memory T-cell commitment (L5) ---
        mem_comm_cd4 = min(self.effector_cd4,
                           _PLASMA_TO_MEMORY * self.effector_cd4 * dt_h)
        mem_comm_cd8 = min(self.effector_cd8,
                           _PLASMA_TO_MEMORY * self.effector_cd8 * dt_h)
        self.effector_cd4 -= mem_comm_cd4
        self.effector_cd8 -= mem_comm_cd8
        self.memory_cd4 += mem_comm_cd4
        self.memory_cd8 += mem_comm_cd8

        # --- Decay / homeostasis (PD-1 network slows effector exhaustion) ---
        # A relieved immune brake (effective_blockade -> 1) lengthens the
        # effector half-life toward a non-exhausted state.  Uses the full
        # PD-1/PD-L1/PD-L2 + CTLA-4 + LAG-3 network (Phase G(b)), falling back
        # to the legacy G14 scalar only when no network state is present.
        bd = self.effective_checkpoint_blockade()
        eff_hl = _T_EFFECTOR_HALF_LIFE_H * (1.0 + 5.0 * min(1.0, max(0.0, bd)))
        e = math.exp(-dt_h * _hl_rate(eff_hl))
        self.effector_cd4 *= e
        self.effector_cd8 *= e
        self.effector_b *= e
        m = math.exp(-dt_h * _hl_rate(_T_MEMORY_HALF_LIFE_H))
        self.memory_cd4 *= m
        self.memory_cd8 *= m
        self.memory_b *= m
        pc_short = math.exp(-dt_h * _hl_rate(_TC_DEATH_HALF_LIFE_H))
        pc_long = math.exp(-dt_h * _hl_rate(_IGG_TURNOVER_LONG_HALF_LIFE_H))
        self.plasma_short *= pc_short
        self.plasma_long *= pc_long

        # Convert a fraction of short-lived plasma to long-lived (L8 LLPC).
        llpc = min(self.plasma_short, _CLASS_SWITCH_RATE * 0.2
                   * self.plasma_short * dt_h)
        self.plasma_short -= llpc
        self.plasma_long += llpc

        # --- Soft logistic saturation (bounded chronic response, doc/31 §4.5) ---
        self.effector_cd4 = _saturate(self.effector_cd4)
        self.effector_cd8 = _saturate(self.effector_cd8)
        self.effector_b = _saturate(self.effector_b)
        self.memory_cd4 = _saturate(self.memory_cd4)
        self.memory_cd8 = _saturate(self.memory_cd8)
        self.memory_b = _saturate(self.memory_b)
        self.plasma_short = _saturate(self.plasma_short)
        self.plasma_long = _saturate(self.plasma_long)

        # --- Antibody production + waning (G3) ---
        ab_prod = _ANTIBODY_PER_PC_H * (
            self.plasma_short + self.plasma_long) * dt_h
        igm_decay = math.exp(-dt_h * _hl_rate(_IGM_TURNOVER_HALF_LIFE_H))
        igg_decay = math.exp(-dt_h * _hl_rate(_IGG_TURNOVER_SHORT_HALF_LIFE_H))
        # Long-lived plasma sustains Igg on a slower turnover.
        self.igg_titer = self.igg_titer * igg_decay + ab_prod * 0.7
        self.igm_titer = self.igm_titer * igm_decay + ab_prod * 0.3
        # Memory-driven baseline maintenance keeps IgG above the naive floor.
        self.igg_titer = max(_BASELINE_IGG, self.igg_titer)
        self.igm_titer = max(_BASELINE_IGM, self.igm_titer)

    # --- Public accessors for downstream channels ---
    def get_igg(self) -> float:
        return self.igg_titer

    def get_igm(self) -> float:
        return self.igm_titer

    def get_total_antibody(self) -> float:
        return self.igg_titer + self.igm_titer

    def get_effector_t(self) -> float:
        return self.effector_cd4 + self.effector_cd8

    def get_memory_t(self) -> float:
        return self.memory_cd4 + self.memory_cd8

    def effective_checkpoint_blockade(self) -> float:
        """Effective immune-brake relief (0..1) driving effector survival.

        When the PD-1/CTLA-4/LAG-3 network (Phase G(b)) has been configured
        (any ligand or combination therapy) it is authoritative.  Otherwise
        the fallback is the legacy G14 ``checkpoint_blockade`` scalar, keeping
        untouched models backward-compatible.
        """
        if self.pd1.network_in_use:
            return 1.0 - self.pd1.immune_brake()
        return min(1.0, max(0.0, self.checkpoint_blockade))

    # -- combination checkpoint therapy (Phase G(b)) ------------------------
    def set_checkpoint_therapy(self, anti_pd1: float = 0.0,
                               anti_ctla4: float = 0.0,
                               anti_lag3: float = 0.0) -> None:
        """Set combination PD-1 + CTLA-4 + LAG-3 blockade occupancy (0..1)."""
        self.pd1.anti_pd1 = min(1.0, max(0.0, anti_pd1))
        self.pd1.anti_ctla4 = min(1.0, max(0.0, anti_ctla4))
        self.pd1.anti_lag3 = min(1.0, max(0.0, anti_lag3))
        self.pd1.network_in_use = True

    def set_pdl1(self, value: float) -> None:
        self.pd1.set_pdl1(value)

    def set_pdl2(self, value: float) -> None:
        self.pd1.set_pdl2(value)

    def set_ctla4_ligand(self, value: float) -> None:
        self.pd1.set_ctla4_ligand(value)

    def set_lag3_ligand(self, value: float) -> None:
        self.pd1.set_lag3_ligand(value)


# ---------------------------------------------------------------------------
# Vaccination schedule helper (G12): a list of (time_h, antigen_dose) events
# ---------------------------------------------------------------------------


@dataclass
class VaccineSchedule:
    """A multi-dose vaccination timeline, e.g. prime at t=0, boost at t=168 h.

    ``doses`` is ``[(time_h, dose), ...]``; ``step`` returns the next dose
    due at or before ``t_h`` and clears it.
    """

    doses: list[tuple[float, float]] = field(default_factory=list)

    def due(self, t_h: float) -> float | None:
        """Return the first unused dose due at ``<= t_h`` (or ``None``)."""
        kept: list[tuple[float, float]] = []
        fired: float | None = None
        for time_h, dose in self.doses:
            if fired is None and time_h <= t_h:
                fired = dose
            else:
                kept.append((time_h, dose))
        self.doses = kept
        return fired


def cohort_adaptive_step(models: list[AdaptiveImmuneModel], dt_h: float,
                         antigens: list[float],
                         doses: list[float | None] | None = None,
                         use_numpy: bool | None = None) -> list[float]:
    """Vectorized adaptive step over a cohort (doc/39 O2 idiom).

    Mirrors ``AdaptiveImmuneModel.step`` term-for-term across numpy arrays;
    returns the total-antibody titer per model. Falls back to the scalar
    path when numpy is unavailable or ``use_numpy=False``.
    """
    if use_numpy is None:
        use_numpy = _HAS_NUMPY
    if not use_numpy or _np is None:
        out: list[float] = []
        for i, m in enumerate(models):
            m.step(dt_h, antigens[i],
                   doses[i] if doses is not None else None)
            out.append(m.get_total_antibody())
        return out

    np = _np
    n = len(models)
    if doses is None:
        doses = [None] * n
    e = math.exp
    hl = _hl_rate
    delay = _delay_rate
    pc_short_f = e(-dt_h * hl(_TC_DEATH_HALF_LIFE_H))
    pc_long_f = e(-dt_h * hl(_IGG_TURNOVER_LONG_HALF_LIFE_H))
    igm_f = e(-dt_h * hl(_IGM_TURNOVER_HALF_LIFE_H))
    igg_f = e(-dt_h * hl(_IGG_TURNOVER_SHORT_HALF_LIFE_H))
    mem_f = e(-dt_h * hl(_T_MEMORY_HALF_LIFE_H))
    ant_f = e(-dt_h / _ANTIGEN_DECAY_TAU_H)

    def arr(get: Callable[[Any], float]) -> Any:
        return np.array([get(m) for m in models], dtype=float)

    a_inf = np.array(antigens, dtype=float)
    a_vac = arr(lambda m: m.antigen_available)
    naive4 = arr(lambda m: m.naive_cd4)
    naive8 = arr(lambda m: m.naive_cd8)
    naiveb = arr(lambda m: m.naive_b)
    eff4 = arr(lambda m: m.effector_cd4)
    eff8 = arr(lambda m: m.effector_cd8)
    effb = arr(lambda m: m.effector_b)
    pshort = arr(lambda m: m.plasma_short)
    plong = arr(lambda m: m.plasma_long)
    mem_b = arr(lambda m: m.memory_b)
    mem4 = arr(lambda m: m.memory_cd4)
    mem8 = arr(lambda m: m.memory_cd8)
    igm = arr(lambda m: m.igm_titer)
    igg = arr(lambda m: m.igg_titer)
    apc = arr(lambda m: m.apc_primed)
    # Advance the per-patient PD-1/CTLA-4/LAG-3 network (Phase G(b)) so the
    # effective blockade matches the scalar path bit-for-bit.
    for m in models:
        m.pd1.step(dt_h)
    # PD-1/CTLA-4/LAG-3 checkpoint network slows effector exhaustion per patient.
    ell_hl = arr(lambda m: _T_EFFECTOR_HALF_LIFE_H * (
        1.0 + 5.0 * min(1.0, max(0.0, m.effective_checkpoint_blockade()))))
    eff_f = np.exp(-dt_h * (np.log(2.0) / (ell_hl + 1e-12)))

    for i, m in enumerate(models):
        dose = doses[i]
        if dose is not None and dose > 0:
            m.vaccinate(dose)
        a_vac[i] = m.antigen_available * ant_f

    drive = a_inf + a_vac
    target_apc = _APC_BASELINE_DC + drive ** _HILL_N / (
        _ANTIGEN_HALF ** _HILL_N + drive ** _HILL_N + 1e-12)
    apc += dt_h * delay(_APC_MATURATION_TAU_H) * (target_apc - apc)
    primed = (np.maximum(apc - _APC_BASELINE_DC, 0.0) ** _HILL_N /
              (_K_APC_HALF ** _HILL_N
               + np.maximum(apc - _APC_BASELINE_DC, 0.0) ** _HILL_N + 1e-12))

    na4l = np.minimum(naive4, _CD4_PROLIF * primed * naive4 * dt_h)
    na8l = np.minimum(naive8, _CD8_PROLIF * primed * naive8 * dt_h)
    nabl = np.minimum(naiveb, _B_PROLIF * primed * naiveb * dt_h)
    naive4 -= na4l
    naive8 -= na8l
    naiveb -= nabl

    mb4 = _CD4_PROLIF * _MEMORY_BOOST * primed * mem4 * dt_h
    mb8 = _CD8_PROLIF * _MEMORY_BOOST * primed * mem8 * dt_h
    mbb = _B_PROLIF * _MEMORY_BOOST * primed * mem_b * dt_h
    eff4 += na4l * _AMPLIFY + mb4
    eff8 += na8l * _AMPLIFY + mb8
    effb += nabl * _AMPLIFY + mbb

    to_pl = np.minimum(effb, _EFFECTOR_TO_PLASMA * effb * dt_h)
    to_mb = np.minimum(np.maximum(effb - to_pl, 0.0),
                       _PLASMA_TO_MEMORY * to_pl)
    effb -= (to_pl + to_mb)
    pshort += to_pl
    mem_b += to_mb

    mc4 = np.minimum(eff4, _PLASMA_TO_MEMORY * eff4 * dt_h)
    mc8 = np.minimum(eff8, _PLASMA_TO_MEMORY * eff8 * dt_h)
    eff4 -= mc4
    eff8 -= mc8
    mem4 += mc4
    mem8 += mc8

    eff4 *= eff_f
    eff8 *= eff_f
    effb *= eff_f
    mem4 *= mem_f
    mem8 *= mem_f
    mem_b *= mem_f
    pshort *= pc_short_f
    plong *= pc_long_f
    llpc = np.minimum(pshort, _CLASS_SWITCH_RATE * 0.2 * pshort * dt_h)
    pshort -= llpc
    plong += llpc

    def _sat(a: Any) -> Any:
        return a / (1.0 + a / _CAPACITY)

    eff4 = _sat(eff4)
    eff8 = _sat(eff8)
    effb = _sat(effb)
    mem4 = _sat(mem4)
    mem8 = _sat(mem8)
    mem_b = _sat(mem_b)
    pshort = _sat(pshort)
    plong = _sat(plong)

    ab_prod = _ANTIBODY_PER_PC_H * (pshort + plong) * dt_h
    igg = np.maximum(_BASELINE_IGG, igg * igg_f + ab_prod * 0.7)
    igm = np.maximum(_BASELINE_IGM, igm * igm_f + ab_prod * 0.3)

    for i, m in enumerate(models):
        m.naive_cd4 = float(naive4[i])
        m.naive_cd8 = float(naive8[i])
        m.naive_b = float(naiveb[i])
        m.effector_cd4 = float(eff4[i])
        m.effector_cd8 = float(eff8[i])
        m.effector_b = float(effb[i])
        m.plasma_short = float(pshort[i])
        m.plasma_long = float(plong[i])
        m.memory_b = float(mem_b[i])
        m.memory_cd4 = float(mem4[i])
        m.memory_cd8 = float(mem8[i])
        m.igm_titer = float(igm[i])
        m.igg_titer = float(igg[i])
        m.apc_primed = float(apc[i])
        m.antigen_available = float(a_vac[i])

    return list(igg + igm)


# ---------------------------------------------------------------------------
# Full PD-1 / PD-L1 / PD-L2 checkpoint network (doc/40 Phase G(b)).
#
# Replaces the single G14 ``checkpoint_blockade`` toggle with an explicit
# receptor–ligand network:
#   - PD-1 surface expression with internalization/recycling (trafficking),
#   - binding of PD-L1 and PD-L2 ligands -> PD-1/* complexes that signal,
#   - CTLA-4 (CD80/CD86 competition) and LAG-3 (MHC-II) checkpoints,
#   - combination therapeutic blockade (anti-PD-1, anti-CTLA-4, anti-LAG-3),
#   - a composite ``immune brake`` (inhibition of effector T-cell survival),
#     whose 0..1 relief is what the AdaptiveImmuneModel step consumes.
#
# Deterministic, additive, inert at baseline (no ligand -> no signalling).


__all__ = [
    "AdaptiveImmuneModel",
    "VaccineSchedule",
    "cohort_adaptive_step",
    "PD1Checkpoint",
]
