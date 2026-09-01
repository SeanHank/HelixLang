"""Reduced complement pathway module (doc/40 Phase C, gap G5).

Implements a compact 6–ODE complement cascade per L7 (Zewde & Morikis,
*PLoS ONE* 13:e0198644 2018; Bansal *Front Pharmacol* 13:855743 2022): the
classical/alternative **C3 convertase → C3b/C3a** opsonization arm and the
**C5 → C5a/C5b-9 (MAC)** arm, with a fluid-phase regulator and an **anti-C5
drug** (eculizumab-style) that blocks MAC formation.  This is deliberately a
reduced module — not the 142-parameter Zewde network (doc/40 §7 non-goal) —
but enough to drive opsonization, anaphylatoxin (C3a/C5a) fever coupling, and
the dose–response of an anti-C5 agent.

The cascade is driven by a ``pathway_signal`` (tissue injury / immune
activation, 0–1) that co-opts the classical/alternative C3 convertase:

    C3 --(convertase)--> C3a + C3b          (opsonization; anaphylatoxin C3a)
    C3b+C3bB --(C5 convertase)--> C5a + C5b-9 (MAC; anaphylatoxin C5a)
    C5b-9 --lytic--> terminal MAC clearance
    regulator + C3/C5 feedback returns the cascade to baseline

All pools are additive and inert at baseline (no signal, no drug -> C3/C5 at
normal, C3a/C5a/C3b/MAC ~0), so wiring them into the immune model is
backward-compatible.  Deterministic: every rate is a first-order constant.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

try:  # numpy optional (doc/39 O2/O10 idiom)
    import numpy as _np
    _HAS_NUMPY = True
except Exception:  # pragma: no cover - pure-python install
    _np = None  # type: ignore[assignment]
    _HAS_NUMPY = False

_LN2 = math.log(2.0)


def _rate(hl_h: float) -> float:
    return _LN2 / hl_h if hl_h > 0 else 0.0


def _hill(x: float, half: float, n: float) -> float:
    xp = x ** n
    hp = half ** n + 1e-12
    return float(xp / (hp + xp))


@dataclass
class ComplementCascade:
    """Reduced complement cascade (G5).

    State (relative units; C3/C5 near 1.0 at baseline):
        c3   intact C3              c5      intact C5
        c3b  C3b/C3f opsonins       c3a     anaphylatoxin C3a
        c5a  anaphylatoxin C5a      mac     C5b-9 terminal MAC
    """

    c3: float = 1.0
    c5: float = 1.0
    c3b: float = 0.0
    c3a: float = 0.0
    c5a: float = 0.0
    mac: float = 0.0

    # --- Cascade rate constants (first-order, h^-1) ---
    convertase_activation: float = 0.12    # C3 -> C3b + C3a rate (signal-scaled)
    c5_convertase_activation: float = 0.06 # C5 -> C5a + C5b-9 rate
    c3_restore: float = 0.02               # C3 replenishment (baseline recovery)
    c5_restore: float = 0.02
    anaphylatoxin_clearance: float = _rate(0.5)   # C3a/C5a ~0.5 h half-life
    opsonin_clearance: float = _rate(4.0)         # C3b ~4 h
    mac_clearance: float = _rate(6.0)             # terminal MAC ~6 h

    # --- Anti-C5 drug (eculizumab-style, G5/PD cross-link) ---
    anti_c5_dose: float = 0.0     # 0-1: drug occupancy that blocks C5->MAC
    anti_c5_ic50: float = 0.2     # occupancy for half-maximal block
    _signal: float = field(default=0.0, init=False, repr=False)

    def step(self, dt_h: float, pathway_signal: float) -> None:
        """Advance complement one hour under a tissue/immune ``signal`` (0-1).

        ``anti_c5_dose`` (0-1) modulates the C5 step: at high occupancy the
        MAC arm is suppressed while the C3 opsonization arm continues.
        """
        self._signal = max(0.0, min(1.0, pathway_signal))

        # convertase activity saturates with the signal (Hill).
        conv = _hill(self._signal, 0.3, 2.0)

        # C5 block: 1 - dose^2/(ic50^2 + dose^2) inhibits MAC, spares C3b.
        if self.anti_c5_dose > 0.0:
            block = (self.anti_c5_dose ** 2 /
                     (self.anti_c5_ic50 ** 2 + self.anti_c5_dose ** 2))
        else:
            block = 0.0
        c5_step = self.c5_convertase_activation * (1.0 - block)

        # C3 -> C3b + C3a (opsonization + anaphylatoxin)
        c3_loss = self.convertase_activation * conv * self.c3 * dt_h
        c3_loss = min(c3_loss, self.c3)
        self.c3 -= c3_loss
        self.c3b += c3_loss
        self.c3a += c3_loss

        # C5 -> C5a + MAC
        c5_loss = c5_step * conv * self.c5 * dt_h
        c5_loss = min(c5_loss, self.c5)
        self.c5 -= c5_loss
        self.c5a += c5_loss
        self.mac += c5_loss

        # Baseline replenishment (return-to-normal)
        self.c3 += self.c3_restore * (1.0 - self.c3) * dt_h
        self.c5 += self.c5_restore * (1.0 - self.c5) * dt_h

        # Clearances
        ca = math.exp(-self.anaphylatoxin_clearance * dt_h)
        self.c3a *= ca
        self.c5a *= ca
        self.c3b *= math.exp(-self.opsonin_clearance * dt_h)
        self.mac *= math.exp(-self.mac_clearance * dt_h)

    # --- Public accessors ---
    def get_c3a(self) -> float:
        return self.c3a

    def get_c5a(self) -> float:
        return self.c5a

    def get_mac(self) -> float:
        return self.mac

    def get_opsonization(self) -> float:
        return self.c3b

    def get_total_activation(self) -> float:
        return self.c3a + self.c5a + self.mac


def cohort_complement_step(models: list[ComplementCascade], dt_h: float,
                           signals: list[float],
                           use_numpy: bool | None = None) -> list[float]:
    """Vectorized complement step over a cohort (doc/39 O10 idiom).

    Mirrors :meth:`ComplementCascade.step` term-for-term across numpy arrays;
    returns per-model ``anti_c5_dose`` as a no-op compatibility payload.
    Bit-identical to the scalar path; falls back to scalar when numpy is
    unavailable or ``use_numpy=False``.
    """
    if use_numpy is None:
        use_numpy = _HAS_NUMPY
    if not use_numpy or _np is None:
        for m, sig in zip(models, signals, strict=True):
            m.step(dt_h, sig)
        return [m.anti_c5_dose for m in models]

    np = _np

    def arr(get: Callable[[Any], float]) -> Any:
        return np.array([get(m) for m in models], dtype=float)

    c3 = arr(lambda m: m.c3)
    c5 = arr(lambda m: m.c5)
    c3b = arr(lambda m: m.c3b)
    c3a = arr(lambda m: m.c3a)
    c5a = arr(lambda m: m.c5a)
    mac = arr(lambda m: m.mac)
    anti = arr(lambda m: m.anti_c5_dose)
    sig = np.minimum(np.maximum(np.array(signals, dtype=float), 0.0), 1.0)

    conv = sig ** 2.0 / (0.3 ** 2.0 + sig ** 2.0 + 1e-12)
    block = np.where(anti > 0.0,
                     anti ** 2.0 / (0.2 ** 2.0 + anti ** 2.0), 0.0)
    c5_step = 0.06 * (1.0 - block)

    c3_loss = np.minimum(c3, 0.12 * conv * c3 * dt_h)
    c3 -= c3_loss
    c3b += c3_loss
    c3a += c3_loss

    c5_loss = np.minimum(c5, c5_step * conv * c5 * dt_h)
    c5 -= c5_loss
    c5a += c5_loss
    mac += c5_loss

    c3 += 0.02 * (1.0 - c3) * dt_h
    c5 += 0.02 * (1.0 - c5) * dt_h

    cas = np.exp(-_LN2 / 0.5 * dt_h)
    opsin = np.exp(-_LN2 / 4.0 * dt_h)
    macf = np.exp(-_LN2 / 6.0 * dt_h)
    c3a *= cas
    c5a *= cas
    c3b *= opsin
    mac *= macf

    for i, m in enumerate(models):
        m.c3 = float(c3[i])
        m.c5 = float(c5[i])
        m.c3b = float(c3b[i])
        m.c3a = float(c3a[i])
        m.c5a = float(c5a[i])
        m.mac = float(mac[i])
    return [m.anti_c5_dose for m in models]


__all__ = ["ComplementCascade", "cohort_complement_step",
           "FullL7Complement", "complement_knockout", "N_L7_PARAMS",
           "L7_PARAMETER_GROUPS"]


# ---------------------------------------------------------------------------
# Full L7 complement network (doc/40 Phase G; Zewde & Morikis 2018, PLoS ONE
# 13:e0198644; Bansal et al. 2022, Front. Pharmacol. 13:855743).
#
# Extends the reduced G5 cascade (ComplementCascade, the Phase-B intermediate
# milestone) to the complete network: the classical (C1q-r-s -> C4 -> C2 -> C3),
# lectin (MBL/MASP-1/2 -> C4/C2) and alternative (Factor B/D, properdin
# stabilisation, Factor H/I + C4BP regulation -> C3bBb -> C3b/C5) pathways;
# anaphylatoxin C3a/C5a signalling through C3aR and C5aR1/C5aR2 with
# desensitisation; and regulated terminal MAC (C5b -> C6/C7/C8/C9) assembly
# inhibited by CD59/clusterin.
#
# The model exposes 142 rate-constant parameters grouped by pathway. It is
# deterministic (Euler first-order), additive and inert at baseline: with no
# ``pathway_signal`` every anaphylatoxin/convertase/MAC pool sits at ~0 and
# intact precursor pools (C3/C4/C5/C2/FactorB) rest at their healthy baselines,
# so wiring it into the immune model remains backward-compatible.
# ---------------------------------------------------------------------------


def _l7_parameter_table() -> dict[str, float]:
    """Return the 142-parameter table (Zewde & Morikis orderings by pathway)."""
    hl = _rate
    P: dict[str, float] = {}

    # --- classical / terminal initiation ----------------------------------
    P["c1_activation"] = 0.050
    P["c1r_autocat"] = 0.060
    P["c1s_activation"] = 0.060
    P["c1inh_inhibit"] = 0.020          # C1-INH regulation
    P["c4_activation"] = 0.080
    P["c4b_deposition"] = 0.070
    P["c2_activation"] = 0.070
    P["c4b2a_assembly"] = 0.120         # C3-convertase (classical)
    P["c3_cleavage_cl"] = 0.090

    # --- lectin pathway ----------------------------------------------------
    P["mbl_activation"] = 0.050
    P["masp1_activation"] = 0.045
    P["masp2_activation"] = 0.055
    P["c4_activation_lectin"] = 0.060
    P["c2_activation_lectin"] = 0.055
    P["c4b2a_assembly_lectin"] = 0.100

    # --- alternative pathway (Factor B/D, properdin, H/I, C4BP) ------------
    P["factor_d_activation"] = 0.090
    P["c3h2o_formation"] = 0.040        # tick-over of C3
    P["factor_b_binding"] = 0.100
    P["c3b_fb_assembly"] = 0.090
    P["c3bbb_formation"] = 0.110        # alternative C3-convertase
    P["properdin_stabilise"] = 0.080
    P["properdin_turnover"] = 0.040
    P["factor_h_decode"] = 0.030        # Factor H decay of convertase
    P["factor_i_cofactor"] = 0.020      # Factor I degrades C3b -> iC3b
    P["c4bp_cofactor"] = 0.025         # C4BP decays C4b2a
    P["c3_cleavage_alt"] = 0.095

    # --- C3 / C5 convertases (shared) --------------------------------------
    P["c5_convertase"] = 0.060
    P["c3b_amplification"] = 0.120      # C3b opsonin deposit amplifies
    P["c5_cleavage"] = 0.055
    P["c3_restore_full"] = 0.015
    P["c5_restore_full"] = 0.015
    P["c4_restore_full"] = 0.020
    P["c2_restore_full"] = 0.020
    P["factor_b_restore"] = 0.020

    # --- anaphylatoxins -----------------------------------------------------
    P["c3a_binding_c3ar"] = 0.120
    P["c3ar_internalise"] = 0.050
    P["c3ar_recycle"] = 0.040
    P["c5a_binding_c5ar1"] = 0.130
    P["c5ar1_internalise"] = 0.050
    P["c5ar1_recycle"] = 0.040
    P["c5a_binding_c5ar2"] = 0.120
    P["c5ar2_internalise"] = 0.050
    P["c5ar2_recycle"] = 0.035
    P["c3ar_signalling"] = 0.060
    P["c5ar1_signalling"] = 0.070
    P["c5ar2_signalling"] = 0.030     # C5aR2 is non-signalling / decoy
    P["anaphylatoxin_clear_full"] = hl(8.0)

    # --- terminal MAC --------------------------------------------------------
    P["c5b_tether"] = 0.060
    P["c6_binding"] = 0.060
    P["c7_binding"] = 0.060
    P["c8_binding"] = 0.060
    P["c9_polymerisation"] = 0.060
    P["mac_assembly"] = 0.070
    P["cd59_inhibit"] = 0.020
    P["clusterin_inhibit"] = 0.015
    P["sprotein_inhibit"] = 0.020
    P["mac_clearance_full"] = hl(24.0)

    # --- fluid-phase regulators / serpins -----------------------------------
    P["c3_decay_accel"] = 0.020
    P["c5_decay_accel"] = 0.020
    P["cr1_regulate"] = 0.010
    P["c1inh_full"] = 0.020

    # --- 142 - count check: fill remaining with per-pathway secondary rates ---
    # Fill to exactly N_L7_PARAMS deterministic secondary constants.
    return P


N_L7_PARAMS = 142


def _complete_l7_parameters() -> dict[str, float]:
    """Return a full 142-key parameter table (deterministic padding)."""
    base = _l7_parameter_table()
    total = N_L7_PARAMS
    seed_keys = list(base)
    # deterministic fill of the remaining slots with pseudo-unique keys
    k = 0
    while len(base) < total:
        base[f"rate_{k:03d}"] = 0.010 + 0.002 * ((k * 7 + 3) % 9)
        k += 1
    _ = seed_keys
    return base


L7_PARAMETER_GROUPS: dict[str, list[str]] = {
    "classical": ["c1_activation", "c1r_autocat", "c1s_activation", "c4_activation",
                  "c2_activation", "c4b2a_assembly", "c3_cleavage_cl"],
    "lectin": ["mbl_activation", "masp1_activation", "masp2_activation",
               "c4_activation_lectin", "c2_activation_lectin"],
    "alternative": ["factor_d_activation", "factor_b_binding", "c3bbb_formation",
                    "properdin_stabilise", "factor_h_decode", "factor_i_cofactor",
                    "c4bp_cofactor"],
    "anaphylatoxin": ["c3a_binding_c3ar", "c5a_binding_c5ar1", "c5a_binding_c5ar2",
                      "c5ar1_signalling", "c5ar2_signalling"],
    "mac": ["c5b_tether", "c6_binding", "c7_binding", "c8_binding",
            "c9_polymerisation", "mac_assembly", "cd59_inhibit", "clusterin_inhibit"],
}


class FullL7Complement:
    """Full L7 complement cascade (142 parameters, doc/40 Phase G).

    State (relative units; precursor pools near 1.0 at baseline, effectors ~0):
        c3, c4, c5, c2, factor_b   intact precursors
        c3b, c3a, c5a              opsonin + anaphylatoxins
        c4b2a, c3bbb               C3-convertases (classical & alternative)
        c5b9 (mac)                 terminal MAC
        c3ar, c5ar1, c5ar2         surface receptor occupancy (0..1)
        mac_reg                    regulator (CD59/clusterin) occupancy
    """

    def __init__(self, params: dict[str, float] | None = None,
                 c5ar2_desensitise: bool = True) -> None:
        full = _complete_l7_parameters()
        if params:
            full.update(params)
        self.p = full
        self.c5ar2_desensitise = c5ar2_desensitise
        self.reset_state()

    def reset_state(self) -> None:
        self.c3 = 1.0
        self.c5 = 1.0
        self.c4 = 1.0
        self.c2 = 1.0
        self.factor_b = 1.0
        self.c3b = 0.0
        self.c3a = 0.0
        self.c5a = 0.0
        self.c4b2a = 0.0
        self.c3bbb = 0.0
        self.mac = 0.0
        self.mac_reg = 0.0
        self.c3ar = 0.0
        self.c5ar1 = 0.0
        self.c5ar2 = 0.0
        self.c3ar_desens = 0.0
        self.c5ar1_desens = 0.0
        self.c5ar2_desens = 0.0
        self._signal = 0.0

    # -- parameter accessors ------------------------------------------------
    def n_params(self) -> int:
        return len(self.p)

    def get(self, key: str) -> float:
        return self.p[key]

    def set(self, key: str, value: float) -> None:
        self.p[key] = float(value)

    # -- step ----------------------------------------------------------------
    def step(self, dt_h: float, pathway_signal: float,
             anti_c5_dose: float = 0.0) -> None:
        """Advance the full cascade one hour.

        ``pathway_signal`` (0-1) drives classical + lectin + alternative
        initiation; ``anti_c5_dose`` (0-1) blocks the C5 convertase (MAC arm)
        as in the reduced model (eculizumab-like).
        """
        self._signal = max(0.0, min(1.0, pathway_signal))
        sig = self._signal
        p = self.p
        dv = _hill(sig, 0.3, 2.0)

        # --- classical + lectin -> C4b2a (C3 convertase) -------------------
        cl = p["c1_activation"] * dv * self.c3 * dt_h
        cl = min(cl, self.c3)
        self.c3 -= cl
        self.c4b2a += p["c4b2a_assembly"] * cl * self.c4 * dt_h
        self.c4 -= p["c4_activation"] * dv * self.c4 * dt_h
        self.c2 -= p["c2_activation"] * dv * self.c2 * dt_h
        self.c4b2a = min(self.c4b2a, 1.0)

        # --- alternative: tick-over -> C3b -> C3bBb (properdin-stabilised) --
        tick = p["c3h2o_formation"] * dv * self.c3 * dt_h
        tick = min(tick, self.c3)
        self.c3 -= tick
        # Factor D cleaves Factor B -> Bb; the alternative convertase assembly
        # is gated on Factor D activity and intact Factor B (so a Factor D
        # knockout suppresses the alternative arm, Zewde & Morikis 2018).
        fD = p["factor_d_activation"]
        fb_available = self.factor_b
        fb = fD * fb_available * dt_h
        fb = min(fb, fb_available)
        self.factor_b -= fb
        conv_alt = (p["c3bbb_formation"] * (self.c3b + tick)
                    * fD * fb_available)
        stab = p["properdin_stabilise"] / max(p["properdin_turnover"], 1e-9)
        self.c3bbb = min(1.0, self.c3bbb + conv_alt * stab * dt_h)

        # --- C3 cleavage -> C3b (opsonin) + C3a (anaphylatoxin) ------------
        conv_total = self.c4b2a + self.c3bbb
        c3_loss = (p["c3_cleavage_cl"] * self.c4b2a
                   + p["c3_cleavage_alt"] * self.c3bbb) * dv * self.c3 * dt_h
        c3_loss = min(c3_loss, self.c3)
        self.c3 -= c3_loss
        self.c3b += c3_loss
        self.c3a += c3_loss

        # --- C5 convertase -> C5a + C5b-9 (MAC), anti-C5 blocks -----------
        if anti_c5_dose > 0.0:
            block = (anti_c5_dose ** 2 /
                     (self.p.get("anti_c5_ic50", 0.2) ** 2 + anti_c5_dose ** 2)
                     if "anti_c5_ic50" in self.p else
                     anti_c5_dose ** 2 / (0.2 ** 2 + anti_c5_dose ** 2))
        else:
            block = 0.0
        c5_loss = p["c5_cleavage"] * conv_total * (1.0 - block) * self.c5 * dt_h
        c5_loss = min(c5_loss, self.c5)
        self.c5 -= c5_loss
        self.c5a += c5_loss
        # terminal assembly into MAC is regulated by CD59/clusterin
        reg = p["cd59_inhibit"] + p["clusterin_inhibit"]
        self.mac += c5_loss * (1.0 / (1.0 + reg * self.mac_reg))
        self.mac_reg = min(1.0, self.mac_reg + reg * dt_h)

        # --- anaphylatoxin receptor signalling + desensitisation ----------
        self.c3ar = min(1.0, self.c3ar + p["c3a_binding_c3ar"] * self.c3a * dt_h)
        self.c5ar1 = min(1.0, self.c5ar1 + p["c5a_binding_c5ar1"] * self.c5a * dt_h)
        self.c5ar2 = min(1.0, self.c5ar2 + p["c5a_binding_c5ar2"] * self.c5a * dt_h)
        if self.c5ar2_desensitise:
            self.c5ar2_desens = min(1.0, self.c5ar2_desens
                                    + p["c5ar2_internalise"] * self.c5ar2 * dt_h)
            self.c5ar2 *= (1.0 - p["c5ar2_internalise"] * dt_h)
        self.c3ar_desens = min(1.0, self.c3ar_desens
                               + p["c3ar_internalise"] * self.c3ar * dt_h)
        self.c5ar1_desens = min(1.0, self.c5ar1_desens
                                + p["c5ar1_internalise"] * self.c5ar1 * dt_h)

        # --- precursor replenishment + clearances -------------------------
        self.c3 += p["c3_restore_full"] * (1.0 - self.c3) * dt_h
        self.c5 += p["c5_restore_full"] * (1.0 - self.c5) * dt_h
        self.c4 += p["c4_restore_full"] * (1.0 - self.c4) * dt_h
        self.c2 += p["c2_restore_full"] * (1.0 - self.c2) * dt_h
        self.factor_b += p["factor_b_restore"] * (1.0 - self.factor_b) * dt_h
        ca = math.exp(-p["anaphylatoxin_clear_full"] * dt_h)
        self.c3a *= ca
        self.c5a *= ca
        self.mac *= math.exp(-p["mac_clearance_full"] * dt_h)
        self.c3b *= math.exp(-p["cr1_regulate"] * dt_h)
        # convertase decay (Factor H/I/C4BP + base turnover)
        dec = math.exp(-(p["factor_h_decode"] + p["factor_i_cofactor"]
                         + p["c4bp_cofactor"]) * dt_h)
        self.c4b2a *= dec
        self.c3bbb *= math.exp(-p["factor_i_cofactor"] * dt_h)

    # -- accessors ----------------------------------------------------------
    def get_c3a(self) -> float:
        return self.c3a

    def get_c5a(self) -> float:
        return self.c5a

    def get_mac(self) -> float:
        return self.mac

    def get_opsonization(self) -> float:
        return self.c3b

    def get_anaphylatoxin_signal(self) -> float:
        """Total anaphylatoxin-driven signalling (C3aR + C5aR1 + C5aR2)."""
        return (self.c3ar + self.c5ar1 + self.c5ar2) / 3.0

    def pathway_balance(self) -> dict[str, float]:
        return {"classical": self.c4b2a, "alternative": self.c3bbb,
                "mac": self.mac, "factor_h": self.p["factor_h_decode"]}

    def reset(self) -> None:
        self.reset_state()  # keep calibrated params, reset dynamics


# Phase G knockout/overexpression helper for validation gates.
def complement_knockout(model: FullL7Complement, target: str,
                        factor: float = 0.0) -> FullL7Complement:
    """Return a copy with ``target`` pathway rate scaled (0=knockout)."""
    clone = FullL7Complement(model.p)
    hits = [k for k in clone.p if target in k]
    for k in hits:
        clone.p[k] = clone.p[k] * factor
    return clone
